/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include "rotational_spectrum.h"
#include <core/args.h>
#include <array>

// Show CWD ----------------------------------------------------------------
std::ostream & operator << (std::ostream &_out,
                       const Cylindrical_Wave_Decomposition &_cwd)
{
    _out << "ir=" << _cwd.ir << std::endl
    << "numin=" << _cwd.numin << std::endl
    << "numax=" << _cwd.numax << std::endl
    << "x0=" << _cwd.x0 << std::endl
    << "y0=" << _cwd.y0 << std::endl
    << "r1=" << _cwd.r1 << std::endl
    << "r2=" << _cwd.r2 << std::endl
    << "r3=" << _cwd.r3 << std::endl;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(_cwd.out_ampcos)
    _out << _cwd.out_ampcos(i) << " "
    << _cwd.out_ampsin(i) << std::endl;
    return _out;
}

// Interpolate image value -------------------------------------------------
double Cylindrical_Wave_Decomposition::interpolate(
    MultidimArray<double> &img, double y, double x)
{
    y = y - 1; // Since Fortran starts indexing at 1
    x = x - 1;
    auto iy = (int)y; /* Trunc the x, y coordinates to int */
    auto ix = (int)x;
    double scale =  y - iy;
    double *ptr_yx = &DIRECT_A2D_ELEM(img, iy, ix);
    double *ptr_y1x = ptr_yx + XSIZE(img);
    double introw1 = *ptr_yx  + scale * (*(ptr_yx + 1)  - *ptr_yx);
    double introw2 = *ptr_y1x + scale * (*(ptr_y1x + 1) - *ptr_y1x);
    return introw1 + (x - ix)*(introw2 - introw1);
}

// Compute CWD -------------------------------------------------------------
void Cylindrical_Wave_Decomposition::compute_cwd(MultidimArray<double> &img)
{
    std::array<double,1281> coseno = {};
    std::array<double,5191> ampcos = {};
    std::array<double,5191> ampsen = {};

    double rh;
    rh = XMIPP_MIN(r2, x0 - 4.);
    rh = XMIPP_MIN(rh, y0 - 4.);
    rh = XMIPP_MIN(rh, YSIZE(img) - x0 - 3.);
    rh = XMIPP_MIN(rh, XSIZE(img) - y0 - 3.);
    int ir_internal;
    ir_internal = (int)((rh - r1) / r3 + 1);
    int ind;
    ind = 0;
    int numin1;
    int numax1;
    numin1 = numin + 1;
    numax1 = numax + 1;
    int kk;
    for (kk = numin1;kk <= numax1;kk++)
    {
    	int k;
        k = kk - 1;
        double coefca;
        double coefcb;
        double coefsa;
        double coefsb;
        double h;
        int ntot;
        int my;
        int my2;
        int my4;
        int my5;
        int j;
        if (k != 0)
        {
            my = (int)(1 + PI * rh / 2. / k);
            my2 = 2 * my;
            my4 = my * k;
            my5 = my4 - 1;
            ntot = 4 * my4;
            h = 2. * PI / ntot;
            double hdpi;
            hdpi = h / PI;
            double th;
            th = k * h;
            double ys;
            ys = sin(th);
            double zs;
            zs = cos(th);
            double ys2;
            ys2 = sin(2. * th);
            double b1;
            b1 = 2. / (th * th) * (1. + zs * zs - ys2 / th);
            double g1;
            g1 = 4. / (th * th) * (ys / th - zs);
            double d1;
            d1 = 2. * th / 45.;
            double e1;
            e1 = d1 * ys * 2.;
            d1 *= ys2;
            coefca = (b1 + e1) * hdpi;
            coefcb = (g1 - d1) * hdpi;
            coefsa = (b1 - e1) * hdpi;
            coefsb = (g1 + d1) * hdpi;
        }
        else
        {
            my = (int)(1 + PI * rh / 2.);
            my2 = 2 * my;
            my4 = my;
            my5 = my4 - 1;
            ntot = 4 * my4;
            h = 2. * PI / ntot;
            coefca = h / PI / 2.;
        }
        int i;
        for (i = 1;i <= my5;i++)
        {
        	double fi;
            fi = i * h;
            coseno[i] = sin(fi);
        }
        coseno[my4] = 1.;
        int my3;
        my3 = 2 * my4;
        for (i = 1;i <= my5;i++)
        {
            coseno[my3-i] = coseno[i];
            coseno[my3+i] = -coseno[i];
            coseno[ntot-i] = -coseno[i];
            coseno[ntot+i] = coseno[i];
        }
        coseno[my3] = 0.;
        coseno[my3+my4] = -1.;
        coseno[ntot] = 0.;
        coseno[ntot+my4] = 1.;
        double r11;
        r11 = r1 - r3;
        double ac;
        double r;
        for (int jr = 1;jr <= ir_internal;jr++)
        {
            ind++;
            r = r11 + r3 * jr;
            ac = 0.;
            int i1c;
            i1c = my4;
            int i1s;
            i1s = 0;
            double x;
            double y;
            double z;
            if (k != 0)
            {
                double as;
                double bc;
                double bs;
                as = bc = bs = 0.;
                for (i = 1;i <= k;i++)
                {
                	int i2c;
                    i2c = my4;
                	int i2s;
                    i2s = 0;
                    for (j = 1;j <= my2;j++)
                    {
                        i1c++;
                        i1s++;
                        i2c += k;
                        i2s += k;
                        x = x0 + r * coseno[i1c];
                        y = y0 + r * coseno[i1s];
                        z = interpolate(img, y, x);
                        bc += z * coseno[i2c];
                        bs += z * coseno[i2s];
                        i1c++;
                        i1s++;
                        i2c += k;
                        i2s += k;
                        x = x0 + r * coseno[i1c];
                        y = y0 + r * coseno[i1s];
                        z = interpolate(img, y, x);
                        ac += z * coseno[i2c];
                        as += z * coseno[i2s];
                    }
                }
                ampcos[ind] = coefca * ac + coefcb * bc;
                ampsen[ind] = coefsa * as + coefsb * bs;
            }
            else
                for (j = 1;j <= ntot;j++)
                {
                    i1c++;
                    i1s++;
                    x = x0 + r * coseno[i1c];
                    y = y0 + r * coseno[i1s];
                    z = interpolate(img, y, x);
                    ac += z;
                    ampcos[ind] = coefca * ac;
                    ampsen[ind] = 0.;
                }
        }
        if (k == 0)
        {
            ac = 0.;
            for (j = 1;j <= ir_internal;j++)
            {
                r = r11 + j * r3;
                ac += ampcos[j] * 2. * PI * r;
            }
            ac /= (PI * (r * r - r1 * r1));
            for (j = 1;j <= ir_internal;j++)
                ampcos[j] -= ac;
        }
    }

    out_ampcos.initZeros((numax - numin + 1)*ir_internal);
    out_ampsin = out_ampcos;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(out_ampcos)
    {
        A1D_ELEM(out_ampcos, i) = ampcos[i+1];
        A1D_ELEM(out_ampsin, i) = ampsen[i+1];
    }
}

// Show Spectrum -----------------------------------------------------------
std::ostream & operator << (std::ostream &_out, const Rotational_Spectrum &_spt)
{
    _out << "numin=" << _spt.numin << std::endl
         << "numax=" << _spt.numax << std::endl
         << "x0=" << _spt.x0 << std::endl
         << "y0=" << _spt.y0 << std::endl
         << "rl=" << _spt.rl << std::endl
         << "rh=" << _spt.rh << std::endl
         << "dr=" << _spt.dr << std::endl;
    return _out;
}

// Compute spectrum --------------------------------------------------------
#define MAX_HARMONIC 51   /* Max. no of harmonics accepted */
void Rotational_Spectrum::compute_rotational_spectrum(
    Cylindrical_Wave_Decomposition &cwd,
    double xr1, double xr2, double xdr, double xr)
{
    double *e[MAX_HARMONIC];
    double *ep[MAX_HARMONIC];
    double *erp [MAX_HARMONIC];

    // Read the information from the Cylindrical Wave Decomposition .........
    auto c = (double *) calloc(5191, sizeof(double));
    auto s = (double *) calloc(5191, sizeof(double));
    if ((NULL == c) || (NULL == s))
        REPORT_ERROR(ERR_MEM_NOTENOUGH, "compute_rotational_spectrum::no memory");

    int numin  = cwd.numin;
    int numax  = cwd.numax;
    double rl  = cwd.r1;
    double rh  = cwd.r2;
    double dr  = cwd.r3;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(cwd.out_ampcos)
    {
        c[i+1] = A1D_ELEM(cwd.out_ampcos, i);
        s[i+1] = A1D_ELEM(cwd.out_ampsin, i);
    }

    int n = numax - numin + 1;
    auto m = (int)((rh - rl) / dr + 1);
    for (int i = 1; i <= n; i++) {
        e[i] = (double *) calloc(m + 1, sizeof(double));
        if (NULL == e[i])
            REPORT_ERROR(ERR_MEM_NOTENOUGH, "compute_rotational_spectrum::no memory");
    }
    auto rv = (double *) calloc(m + 1, sizeof(double));
    auto st = (double *) calloc(m + 1, sizeof(double));
    if ((NULL == rv) || (NULL == st))
        REPORT_ERROR(ERR_MEM_NOTENOUGH, "compute_rotational_spectrum::no memory");

    // Computations .........................................................
    int j1 = 0;
    if (numin == 0)
        j1 = 2;
    else
        j1 = 1;
    int k = 0;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
        {
            k++;
            e[i][j] = c[k] * c[k] + s[k] * s[k];
        }

    for (int i = 1; i <= m; i++)
    {
        rv[i] = rl + dr * (i - 1);
        st[i] = 0;
        for (int j = j1; j <= n; j++)
            st[i] += e[j][i];
    }

    auto ir1 = (int)((xr1 - rl) / dr + 1);
    if (ir1 < 1)
        ir1 = 1;
    auto ir2 = (int)((XMIPP_MIN(xr2, rh) - rl) / dr + 1);
    if (ir2 < ir1)
        ir2 = ir1;
    auto ndr = (int)(xdr / dr);
    if (ndr < 1)
        ndr = 1;
    int nr = 0;
    if (xr < 0)
        nr = m;
    else
        nr = (int)(xr / dr + 1);
    ir2 = XMIPP_MIN(ir2, m);
    int ncol = ir2 - nr + 1 - ir1;
    if (ncol < 0)
        ncol = 0;
    else
        ncol = 1 + ncol / ndr;
    if (ncol == 0)
        REPORT_ERROR(ERR_VALUE_INCORRECT, "compute_rotational_spectrum::Incorrect data");

    int nvez = (ncol - 1) / 13 + 1;
    for (int i = 1; i <= n; i++) {
        ep[i] = (double *) calloc(ncol + 1, sizeof(double));
        erp[i] = (double *) calloc(ncol + 1, sizeof(double));
        if ((NULL == ep[i]) || (NULL == erp[i]))
            REPORT_ERROR(ERR_MEM_NOTENOUGH, "compute_rotational_spectrum::no memory");
    }
    auto rp1 = (double *) calloc(ncol + 1, sizeof(double));
    auto rp2 = (double *) calloc(ncol + 1, sizeof(double));
    auto sp  = (double *) calloc(ncol + 1, sizeof(double));
    if ((NULL == rp1) || (NULL == rp2) || (NULL == sp))
        REPORT_ERROR(ERR_MEM_NOTENOUGH, "compute_rotational_spectrum::no memory");
    for (k = 1; k <= ncol; k++)
    {
        int irk = ir1 + (k - 1) * ndr - 1;
        rp1[k] = 10 * rv[irk+1];
        rp2[k] = 10 * rv[irk+nr];
        sp[k] = 0;
        for (int k1 = 1; k1 <= nr; k1++)
            sp[k] += st[irk+k1];
        for (int i = 1; i <= n; i++)
        {
            ep[i][k] = 0;
            for (int k1 = 1; k1 <= nr; k1 ++)
                ep [i][k] += e[i][irk+k1];
            erp[i][k] = 1000000. * ep[i][k] / sp[k];
        }
    }

    // Keep results .........................................................
    rot_spectrum.initZeros(n - j1 + 1);
    for (k = 1; k <= nvez; k++)
    {
        int k1 = 13 * (k - 1) + 1;
        for (int i = j1; i <= n; i++)
            rot_spectrum(i - j1) = erp[i][k1] / 10000;
    }

    // Free memory
    for (int i = 1; i <= n; i++)
    {
        free(ep[i]);
        free(erp [i]);
    }
    free(c);
    free(s);
    free(rv);
    free(st);
    free(rp1);
    free(rp2);
    free(sp);
}

// Compute spectrum --------------------------------------------------------
void Rotational_Spectrum::compute_rotational_spectrum(MultidimArray<double> &img,
        double xr1, double xr2, double xdr, double xr)
{
    // Compute the cylindrical wave decomposition
    Cylindrical_Wave_Decomposition cwd;
    cwd.numin = numin;
    cwd.numax = numax;
    cwd.x0    = (x0 == -1) ? (double)XSIZE(img) / 2 : x0;
    cwd.y0    = (y0 == -1) ? (double)YSIZE(img) / 2 : y0;
    cwd.r1    = rl;
    cwd.r2    = rh;
    cwd.r3    = dr;
    cwd.compute_cwd(img);
    ir = cwd.ir;

    // Compute the rotational spectrum
    compute_rotational_spectrum(cwd, xr1, xr2, xdr, xr);
}

