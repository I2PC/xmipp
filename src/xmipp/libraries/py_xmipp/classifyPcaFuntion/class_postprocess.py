#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""

import math
import torch
import torch.nn.functional as F


class ClassPostprocess:
    """
    Procesa las imágenes de UNA clase, sin depender de otra clase (parent).

    Parámetros principales:
        sampling: tamaño de píxel en unidades físicas por píxel.
        device: dispositivo de cálculo; por defecto CUDA si está disponible.

    process(particles):
        particles: Tensor [N, H, W].
        Devuelve (average_filtrado, resolution).
        Si hay menos de 8 imágenes, FRC devuelve el fallback de 40.0 unidades.

    El promedio de clase se calcula directamente con particles.mean(dim=0).
    """

    def __init__(
        self,
        sampling,
        device=None,
        cut=50.0,
        cut_res=50.0,
        frc_threshold=0.143,
        apply_window=False,
        smooth=True,
        floor_res=100.0,
        clamp_exp=80.0,
        hard_cut=False,
        nyquist_margin=0.95,
        normalize=True,
        f_energy=2.0,
        boost_max=None,
        sharpen_power=None,
        factorR=None,
        eps=1e-8,
    ):
        self.sampling = float(sampling)

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.cut = float(cut)
        self.cut_res = float(cut_res)
        self.frc_threshold = float(frc_threshold)
        self.apply_window = bool(apply_window)
        self.smooth = bool(smooth)

        self.floor_res = float(floor_res)
        self.clamp_exp = float(clamp_exp)
        self.hard_cut = bool(hard_cut)
        self.nyquist_margin = float(nyquist_margin)
        self.normalize = bool(normalize)

        self.f_energy = float(f_energy)
        self.boost_max = boost_max
        self.sharpen_power = sharpen_power
        self.factorR = factorR
        self.eps = float(eps)

        if not (0.0 < self.nyquist_margin <= 1.0):
            raise ValueError("nyquist_margin debe estar en el intervalo (0, 1].")
        if self.boost_max is None and self.f_energy < 1.0:
            raise ValueError("Con boost_max=None, f_energy debe ser >= 1.")
        if self.boost_max is not None and float(self.boost_max) < 1.0:
            raise ValueError("boost_max debe ser >= 1.")

        self._frc_cache = {}
        self._gaussian_cache = {}
        self._sharpen_cache = {}

    @torch.no_grad()
    def process(self, particles):
        """
        Procesa una clase.

        Args:
            particles: Tensor [N,H,W] con las imágenes alineadas de la clase.

        Returns:
            average: Tensor [H,W] en self.device.
            resolution: tensor escalar en self.device.
        """
        if not isinstance(particles, torch.Tensor):
            raise TypeError("particles debe ser un torch.Tensor.")
        if particles.ndim != 3:
            raise ValueError(
                f"particles debe tener forma [N,H,W], recibida {tuple(particles.shape)}."
            )
        if particles.shape[0] == 0:
            raise ValueError("No se puede procesar una clase vacía.")
        if not particles.is_floating_point():
            particles = particles.float()

        particles = particles.to(device=self.device, non_blocking=True)

        # Promedio aritmético directo de la clase.
        average = particles.mean(dim=0)

        resolution = self.calculate_frc(particles)
        del particles

        average = self.gaussian_lowpass(average, resolution)
        average = self.cosine_sharpen(average, resolution)
        return average, resolution

    @torch.no_grad()
    def calculate_frc(self, particles):
        """FRC entre dos promedios formados con mitades aleatorias independientes."""
        n, h, w = particles.shape
        device, dtype = particles.device, particles.dtype

        if n < 8:
            return particles.new_tensor(40.0)

        cache_key = (h, w, device, dtype, self.sampling)
        cache = self._frc_cache.get(cache_key)

        if cache is None:
            # Ancho de anillo aproximadamente igual a la frecuencia fundamental
            # de la dimensión mayor. Se excluyen radios superiores a Nyquist.
            df = 1.0 / (max(h, w) * self.sampling)
            rmax = int(math.floor((0.5 / self.sampling) / df)) + 1

            fy = torch.fft.fftfreq(h, d=self.sampling, device=device, dtype=dtype)
            fx = torch.fft.rfftfreq(w, d=self.sampling, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(fy, fx, indexing="ij")
            radius = torch.sqrt(gx.square() + gy.square())

            valid = radius <= (0.5 / self.sampling + 1e-12)
            r_bin = torch.floor(radius / df + 0.5).long()
            valid = valid & (r_bin >= 0) & (r_bin < rmax)

            # rfft2 almacena solo frecuencias no negativas en X.
            weights_x = torch.full((fx.numel(),), 2.0, device=device, dtype=dtype)
            weights_x[0] = 1.0
            if w % 2 == 0:
                weights_x[-1] = 1.0
            weights = weights_x[None, :].expand(h, -1)

            window = None
            if self.apply_window:
                wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
                wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
                window = wy[:, None] * wx[None, :]
                window_norm = window.norm()
                if window_norm > 0:
                    window = window / window_norm * math.sqrt(h * w)
                del wy, wx

            freq_bins = torch.arange(rmax, device=device, dtype=dtype) * df
            cache = (rmax, freq_bins, r_bin, valid, weights, window)
            self._frc_cache[cache_key] = cache
            del fy, fx, gy, gx, radius
        else:
            rmax, freq_bins, r_bin, valid, weights, window = cache

        perm = torch.randperm(n, device=device)
        split = n // 2
        avg1 = particles.index_select(0, perm[:split]).mean(dim=0)
        avg2 = particles.index_select(0, perm[split:]).mean(dim=0)
        del perm

        if window is not None:
            avg1 = avg1 * window
            avg2 = avg2 * window

        fft1 = torch.fft.rfft2(avg1, norm="forward")
        fft2 = torch.fft.rfft2(avg2, norm="forward")
        del avg1, avg2

        p1 = fft1.real.square() + fft1.imag.square()
        p2 = fft2.real.square() + fft2.imag.square()
        cross = (fft1 * fft2.conj()).real
        del fft1, fft2

        bins = r_bin[valid]
        cross_v = (cross * weights)[valid]
        p1_v = (p1 * weights)[valid]
        p2_v = (p2 * weights)[valid]

        numerator = torch.zeros(rmax, device=device, dtype=dtype)
        denom1 = torch.zeros_like(numerator)
        denom2 = torch.zeros_like(numerator)
        numerator.scatter_add_(0, bins, cross_v)
        denom1.scatter_add_(0, bins, p1_v)
        denom2.scatter_add_(0, bins, p2_v)
        del cross, p1, p2, bins, cross_v, p1_v, p2_v

        frc = numerator / (torch.sqrt(denom1 * denom2) + 1e-12)
        del numerator, denom1, denom2

        if self.smooth and frc.numel() >= 3:
            kernel = frc.new_tensor([0.25, 0.5, 0.25]).view(1, 1, -1)
            frc = F.conv1d(frc.view(1, 1, -1), kernel, padding=1).view(-1)
            del kernel

        # No se usa el anillo DC para detectar el cruce.
        below = torch.where(frc[1:] < self.frc_threshold)[0]
        if below.numel():
            idx = below[0] + 1
            resolution = 1.0 / freq_bins[idx].clamp_min(1e-12)
        else:
            resolution = particles.new_tensor((2.0 * self.sampling) / 0.8)

        fallback = (2.0 * self.sampling) / 0.8
        resolution = torch.nan_to_num(
            resolution, nan=fallback, posinf=fallback, neginf=fallback
        )
        resolution = torch.where(
            resolution > self.cut,
            resolution.new_tensor(self.cut_res),
            resolution,
        )
        return resolution

    def _radial_frequency_squared(self, h, w, device, dtype, cache):
        key = (h, w, device, dtype, self.sampling)
        grid = cache.get(key)
        if grid is None:
            fy = torch.fft.fftfreq(h, d=self.sampling, device=device, dtype=dtype)
            fx = torch.fft.fftfreq(w, d=self.sampling, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(fy, fx, indexing="ij")
            grid = gx.square() + gy.square()
            cache[key] = grid
            del fy, fx, gy, gx
        return grid

    @torch.no_grad()
    def gaussian_lowpass(self, image, resolution):
        """Filtro gaussiano pasa-bajos para una imagen [H,W]."""
        if image.ndim != 2:
            raise ValueError("gaussian_lowpass espera una imagen [H,W].")

        h, w = image.shape
        device, dtype = image.device, image.dtype
        safe_res = (2.0 * self.sampling) / self.nyquist_margin

        res_eff = torch.nan_to_num(
            resolution,
            nan=self.floor_res,
            posinf=self.floor_res,
            neginf=self.floor_res,
        )
        res_eff = torch.minimum(res_eff, res_eff.new_tensor(self.floor_res))
        res_eff = torch.clamp(res_eff, min=safe_res)

        freq2 = self._radial_frequency_squared(
            h, w, device, dtype, self._gaussian_cache
        )
        d0 = 1.0 / res_eff
        sigma2 = (d0 / math.sqrt(2.0 * math.log(2.0))).square()
        exponent = -freq2 / (2.0 * sigma2 + self.eps)
        filt = torch.exp(exponent.clamp(min=-abs(self.clamp_exp), max=0.0))

        if self.hard_cut:
            filt = torch.where(freq2 > d0.square(), torch.zeros_like(filt), filt)

        fft = torch.fft.fft2(image, norm="forward")
        fft.mul_(filt)
        img_filt = torch.fft.ifft2(fft, norm="forward").real
        del fft, filt, exponent, d0, sigma2
        img_filt = torch.nan_to_num(img_filt)

        if self.normalize:
            mean0 = image.mean()
            std0 = image.std(unbiased=True)
            mean_f = img_filt.mean()
            std_f = img_filt.std(unbiased=True)
            normalized = (
                (img_filt - mean_f) / std_f.clamp_min(self.eps) * std0 + mean0
            )
            img_filt = torch.where(std_f > self.eps, normalized, image)
            del mean0, std0, mean_f, std_f, normalized

        return img_filt

    @torch.no_grad()
    def cosine_sharpen(self, average, resolution):
        """
        Realce coseno suave dentro de la frecuencia de corte.
        La máscara se anula en el corte para evitar un salto del filtro.
        """
        if average.ndim != 2:
            raise ValueError("cosine_sharpen espera una imagen [H,W].")

        h, w = average.shape
        device, dtype = average.device, average.dtype
        fft = torch.fft.fft2(average, norm="forward")
        fft_mag2 = fft.real.square() + fft.imag.square()
        energy_orig = fft_mag2.sum()

        freq2 = self._radial_frequency_squared(
            h, w, device, dtype, self._sharpen_cache
        )
        freq_r = torch.sqrt(freq2)
        f_cutoff = 1.0 / torch.clamp(resolution, min=1e-3)

        if self.sharpen_power is None:
            if self.factorR is None:
                factor = torch.where(
                    resolution < 10.0,
                    resolution.new_tensor(0.1),
                    torch.where(
                        resolution < 14.0,
                        resolution.new_tensor(0.08),
                        resolution.new_tensor(0.06),
                    ),
                )
            else:
                factor = resolution.new_tensor(float(self.factorR))
            power = (factor * resolution).clamp(min=0.3, max=2.5)
            del factor
        else:
            power = resolution.new_tensor(float(self.sharpen_power))

        ratio = (freq_r / (f_cutoff + self.eps)).clamp(min=0.0, max=1.0)
        # Suave, 0 en DC y en el corte; máximo en la mitad del intervalo.
        shape = (1.0 - torch.cos(2.0 * math.pi * ratio)) / 2.0
        shape = shape.pow(power)
        shape = torch.where(freq_r < f_cutoff, shape, torch.zeros_like(shape))
        del ratio, power, freq_r, freq2, f_cutoff

        if self.boost_max is None:
            target_energy = self.f_energy * energy_orig
            a = (fft_mag2 * shape.square()).sum()
            b = (fft_mag2 * shape).sum()

            # E(g) = E0 + 2(g-1)b + (g-1)^2 a.
            delta = torch.clamp(target_energy - energy_orig, min=0.0)
            discriminant = torch.clamp(b.square() + a * delta, min=0.0)
            x = (torch.sqrt(discriminant) - b) / a.clamp_min(self.eps)
            boost_max = torch.where(
                a > self.eps,
                (1.0 + x).clamp_min(1.0),
                energy_orig.new_tensor(1.0),
            )
            del target_energy, a, b, delta, discriminant, x
        else:
            boost_max = average.new_tensor(float(self.boost_max))

        boost = 1.0 + (boost_max - 1.0) * shape
        del shape, boost_max, fft_mag2, energy_orig

        fft.mul_(boost)
        filtered = torch.fft.ifft2(fft, norm="forward").real
        del fft, boost
        filtered = torch.nan_to_num(filtered)

        if self.normalize:
            mean0 = average.mean()
            std0 = average.std(unbiased=True)
            mean_f = filtered.mean()
            std_f = filtered.std(unbiased=True)
            normalized = (
                (filtered - mean_f) / std_f.clamp_min(self.eps) * std0 + mean0
            )
            filtered = torch.where(std_f > self.eps, normalized, average)
            del mean0, std0, mean_f, std_f, normalized

        return filtered