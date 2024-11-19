#!/usr/bin/env python3
# *****************************************************************************
# *
# * Authors:     J.L. Vilas (jlvilas@cnb.csic.es) [1]
# *
# * [1] Centro Nacional de Biotecnologia, CSIC, Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# *****************************************************************************



import mrcfile
import numpy as np
import matplotlib.pyplot as plt

import fsc_utils as fsc

import sys
import os

from xmipp_base import XmippScript


class WienerFilterFSC(XmippScript):
    _conda_env="xtomo_tigre" 

    def __init__(self):

        XmippScript.__init__(self)
        
        # Command line parameters
        self.fnIn          = None
        self.sampling      = None
        self.fnOut         = None
        
    #  --------------------- DEFINE PARAMS -----------------------------
    def defineParams(self):
        # Description
        self.addUsageLine('This program perform a Wiener filter of a volume.\n')
        
        # Params
        self.addParamsLine(' -i <fnIn>                            : Volume to be filtered')
        self.addParamsLine(' --sampling <fnAngles>                : Sampling rate in Angstrom')
        self.addParamsLine(' -o <fnOut>                           : Output filename of the tomogram.')
        
        # Examples       
        self.addExampleLine('xmipp_wiener_filter_fsc -i ts.mrc --sampling angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method wbp --filter ram_lak')
 

    def readInputParams(self):
        '''
        In this function the parameters are read. For for information about their use see the help
        '''

        self.fnIn          = self.getParam('-i')
        self.sampling      = self.getParam('--sampling')
        self.fnOut         = self.getParam('-o')
 

    

    def run(self):
        print('Starting ...')
        
        cmap_a = [plt.get_cmap('tab20c').colors[idx] for idx in [0, 4, 8, 12]]
        cmap_b = [plt.get_cmap('tab20c').colors[idx] for idx in [2, 6, 10, 12]]

        tomo_file = self.fnIn

        with mrcfile.open(tomo_file) as mrc:
            tomo_full = mrc.data
            tomo_voxel = mrc.voxel_size.x
            mrc.close()
            
        # Pick region of tomogram to denoise
        subtomo = tomo_full[:, 450:914, 325:789]
        subtomo_slice = subtomo[226]  # denoise just this slice for simplicity

        # Select a noise slice for whitening transform
        noise_slice = tomo_full[50, 450:914, 325:789]

        shape = subtomo_slice.shape

        r = shape[0] // 2
        r_corner = int(np.ceil(np.sqrt(np.sum([s**2 for s in shape]))/2))

        signal_slice_raps = self.compute_spherically_averaged_power_spectrum(subtomo_slice, r_corner) 
        noise_slice_raps = self.compute_spherically_averaged_power_spectrum(noise_slice, r_corner)

        freq = fsc.get_radial_spatial_frequencies(subtomo, tomo_voxel)

        # whiten the subtomogram
        y_whitened = fsc.whitening_transform(subtomo_slice, noise_slice, r_corner)

        # upsample
        y_upsample = fsc.fourier_upsample(y_whitened, 2)

        EPSILON = 1e-8

        ### compute the SFSC ###
        sfsc_upsample = np.mean(fsc.single_image_frc(y_upsample, r, whiten_upsample=True), axis=0)
        sfsc_upsample = np.where(sfsc_upsample > EPSILON, sfsc_upsample, EPSILON)

        # # or the following approximately equal alternatives
        # ssnr_estimate = (signal_slice_raps[:r] - noise_slice_raps[:r]) / noise_slice_raps[:r]
        # fsc_from_ssnr = ssnr_estimate / (1 + ssnr_estimate)
        # fsc_from_ssnr = np.where(fsc_from_ssnr > EPSILON, fsc_from_ssnr, EPSILON)

        # ps_w = fsc.compute_spherically_averaged_power_spectrum(y_whitened, r)
        # fsc_from_ps = (ps_w - 1) / ps_w
        # fsc_from_ps = np.where(fsc_from_ps > EPSILON, fsc_from_ps, EPSILON)

        ### apply a Wiener filter based on the SFSC ###

        Yw = fsc.ft2(y_whitened)
        rdists = fsc.radial_distance_grid(Yw.shape)

        for ri in range(r)[2:]:  # pick range of shells to filter
            mask = fsc.shell_mask(rdists, ri)
            Yw[mask] = sfsc_upsample[ri] * Yw[mask] * np.sqrt(noise_slice_raps[ri])

        # # #optional step - additional low-pass filter based on e.g. 1/7 threshold
        res = fsc.linear_interp_resolution(sfsc_upsample[2:], freq[2:], v=1/7)
        m_idx = np.argmin(abs(freq - 1/res))
        sphere_mask = fsc.sphere_mask(rdists, m_idx)
        Yw = sphere_mask * Yw

        y_filtered = fsc.ift2(Yw)

        print(res)

        # Low-pass filter for comparison to Wiener filter, based on estimated resolution
        subtomo_lpf = fsc.low_pass_filter(subtomo_slice, tomo_voxel, res)

        plt.imshow(fsc.log_abs(fsc.ft2(subtomo_slice)))
        plt.show()
        plt.imshow(fsc.log_abs(fsc.ft2(y_filtered)))
        plt.show()
        plt.imshow(fsc.log_abs(fsc.ft2(subtomo_lpf)))
        plt.show()

        plt.figure(figsize=(7,4))

        plt.semilogy(freq[1:], signal_slice_raps[1:r], color=cmap_a[2], linewidth=2, label='subsection')
        plt.semilogy(freq[1:], noise_slice_raps[1:r], color=cmap_a[0], linewidth=2, label='noise region') 

        plt.xlabel('spatial frequency 'r'(${\AA}^{-1}$)')
        plt.title('spherically averaged power spectrum')

        plt.rc('axes', labelsize=12)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('legend', fontsize=10)

        plt.grid(which='major', linestyle='--')
        plt.ylim(5e5, 2e8)

        plt.legend()
        # plt.savefig('')
        plt.show()

        plt.figure(figsize=(7,4))

        plt.plot(freq[2:], sfsc_upsample[2:], color=cmap_a[2], linewidth=2, label='SFSC')

        plt.xlabel('spatial frequency 'r'(${\AA}^{-1}$)')
        plt.title('Fourier shell correlation')

        plt.rc('axes', labelsize=12)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('legend', fontsize=10)

        plt.grid(which='major', linestyle='--')
        plt.ylim(-0.05, 1.05)

        plt.legend()
        plt.axhline(1/7, linestyle='--', color='grey')
        # plt.savefig('')
        plt.show()

        u = np.mean(subtomo_slice)
        s = np.std(subtomo_slice)

        u_wf = np.mean(y_filtered)
        s_wf = np.std(y_filtered)

        u_lpf = np.mean(subtomo_lpf)
        s_lpf = np.std(subtomo_lpf)

        u_eps = np.mean(noise_slice)
        s_eps = np.std(noise_slice)

        print("\t\t\tfull tomogram slice")
        plt.figure(figsize=(10,10))
        plt.imshow(tomo_full[226,...], cmap='gray')
        plt.axis('off')
        # plt.savefig('')
        plt.show()

        print("\tsub tomogram noise slice")
        plt.figure(figsize=(5,5))
        plt.imshow(noise_slice, cmap='gray')
        plt.axis('off')
        # plt.savefig('')
        plt.show()

        print("\tsub tomogram slice")
        plt.figure(figsize=(5,5))
        plt.imshow(subtomo[226,...], cmap='gray')
        plt.axis('off')
        # plt.savefig('')
        plt.show()

        print("\tsub tomogram slice lpf")
        plt.figure(figsize=(5,5))
        # plt.imshow(subtomo_lpf, cmap='gray')
        plt.imshow(subtomo_lpf, cmap='gray', vmin=u_lpf-(2*s_lpf), vmax=u_lpf+(2*s_lpf)) # auto adjust contrast
        plt.axis('off')
        # plt.savefig('')
        plt.show()

        print("sub tomogram slice Wiener filter")
        plt.figure(figsize=(5,5))
        # plt.imshow(y_filtered, cmap='gray')
        plt.imshow(y_filtered, cmap='gray', vmin=u_wf-(2*s_wf), vmax=u_wf+(2*s_wf))  # auto adjust contrast
        plt.axis('off')
        # plt.savefig('')
        plt.show()

if __name__ == '__main__':
    '''
    scipion xmipp_tomogram_tigre_reconstrucion -g 0
    '''
    exitCode=TomogramReconstruction().tryRun()
    sys.exit(exitCode)