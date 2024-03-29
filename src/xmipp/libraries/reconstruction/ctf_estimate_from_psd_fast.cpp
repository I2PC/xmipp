/***************************************************************************
 *
 * Authors:     Javier Mota Garcia (jmota@cnb.csic.es)
 *              Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es)
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

#include <fstream>
#include <numeric>
#include "ctf_estimate_from_psd.h"
#include "ctf_estimate_from_psd_fast.h"
#include "data/numerical_tools.h"
#include "core/matrix2d.h"
#include "core/xmipp_fftw.h"
#include "core/metadata_vec.h"

/* prototypes */
double CTF_fitness_fast(double *, void *);
extern double CTF_fitness(double *p, void *vprm);

/* Number of CTF parameters */
constexpr int ALL_CTF_PARAMETERS2D =         38;
constexpr int ALL_CTF_PARAMETERS =           28;
constexpr int CTF_PARAMETERS =               20;
constexpr int PARAMETRIC_CTF_PARAMETERS =    16;
constexpr int BACKGROUND_CTF_PARAMETERS =     9;
constexpr int SQRT_CTF_PARAMETERS =           6;
constexpr int ENVELOPE_PARAMETERS =          11;
constexpr int DEFOCUS_PARAMETERS =            3;
constexpr int FIRST_SQRT_PARAMETER =         14;
constexpr int FIRST_ENVELOPE_PARAMETER =      2;
constexpr int FIRST_DEFOCUS_PARAMETER =       0;

//#define DEBUG_WITH_TEXTFILES
#ifdef DEBUG_WITH_TEXTFILES
std::ofstream fhDebug;
#define DEBUG_OPEN_TEXTFILE(fnRoot) fhDebug.open((fnRoot+"_debug.txt").c_str());
#define DEBUG_TEXTFILE(str) fhDebug << time (NULL) << " " << str << std::endl;
#define DEBUG_MODEL_TEXTFILE fhDebug << global_ctfmodel << std::endl;
#else
#define DEBUG_OPEN_TEXTFILE(fnRoot);
#define DEBUG_TEXTFILE(str);
#define DEBUG_MODEL_TEXTFILE
#endif


namespace AdjustCTF1D
{
ProgCTFEstimateFromPSDFast *global_prm;
}

using namespace AdjustCTF1D;

#define ASSIGN_CTF_PARAM(index, paramName) if (ia <= index && l > 0) { ctf1Dmodel.paramName = p[index]; --l; }

/* Assign ctf1Dmodel from a vector and viceversa ----------------------------- */
void ProgCTFEstimateFromPSDFast::assignCTFfromParameters_fast(double *p, CTFDescription1D &ctf1Dmodel, int ia,
                             int l)
{
    ctf1Dmodel.Tm = Tm;

    ASSIGN_CTF_PARAM(0, Defocus);
    ASSIGN_CTF_PARAM(1, kV);
    ASSIGN_CTF_PARAM(2, K);
    ASSIGN_CTF_PARAM(3, Cs);
    ASSIGN_CTF_PARAM(4, Ca);
    ASSIGN_CTF_PARAM(5, espr);
    ASSIGN_CTF_PARAM(6, ispr); //deltaI/I
    ASSIGN_CTF_PARAM(7, alpha);
    ASSIGN_CTF_PARAM(8, DeltaF);
    ASSIGN_CTF_PARAM(9, DeltaR);
    ASSIGN_CTF_PARAM(10, envR0);
    ASSIGN_CTF_PARAM(11, envR1);
    ASSIGN_CTF_PARAM(12, envR2);
    ASSIGN_CTF_PARAM(13, Q0);
    ASSIGN_CTF_PARAM(14, base_line);
    ASSIGN_CTF_PARAM(15, sqrt_K);
    ASSIGN_CTF_PARAM(16, sq);
    ASSIGN_CTF_PARAM(17, bgR1);
    ASSIGN_CTF_PARAM(18, bgR2);
    ASSIGN_CTF_PARAM(19, bgR3);
    ASSIGN_CTF_PARAM(20, gaussian_K);
    ASSIGN_CTF_PARAM(21, sigma1);
    ASSIGN_CTF_PARAM(22, Gc1);
    ASSIGN_CTF_PARAM(23, gaussian_K2);
    ASSIGN_CTF_PARAM(24, sigma2);
    ASSIGN_CTF_PARAM(25, Gc2);
    ASSIGN_CTF_PARAM(26, phase_shift);
    ASSIGN_CTF_PARAM(27, VPP_radius);

}//function assignCTFfromParameters

#define ASSIGN_PARAM_CTF(index, paramName) if (ia <= index && l > 0) { p[index] = ctf1Dmodel.paramName; --l; }

void ProgCTFEstimateFromPSDFast::assignParametersFromCTF_fast(const CTFDescription1D &ctf1Dmodel, double *p, int ia,
                             int l)
{

	 	ASSIGN_PARAM_CTF(0, Defocus);
	    ASSIGN_PARAM_CTF(1, kV);
	    ASSIGN_PARAM_CTF(2, K);
	    ASSIGN_PARAM_CTF(3, Cs);
	    ASSIGN_PARAM_CTF(4, Ca);
	    ASSIGN_PARAM_CTF(5, espr);
	    ASSIGN_PARAM_CTF(6, ispr); //deltaI/I
	    ASSIGN_PARAM_CTF(7, alpha);
	    ASSIGN_PARAM_CTF(8, DeltaF);
	    ASSIGN_PARAM_CTF(9, DeltaR);
	    ASSIGN_PARAM_CTF(10, envR0);
	    ASSIGN_PARAM_CTF(11, envR1);
	    ASSIGN_PARAM_CTF(12, envR2);
	    ASSIGN_PARAM_CTF(13, Q0);
	    ASSIGN_PARAM_CTF(14, base_line);
	    ASSIGN_PARAM_CTF(15, sqrt_K);
	    ASSIGN_PARAM_CTF(16, sq);
	    ASSIGN_PARAM_CTF(17, bgR1);
	    ASSIGN_PARAM_CTF(18, bgR2);
	    ASSIGN_PARAM_CTF(19, bgR3);
	    ASSIGN_PARAM_CTF(20, gaussian_K);
	    ASSIGN_PARAM_CTF(21, sigma1);
	    ASSIGN_PARAM_CTF(22, Gc1);
	    ASSIGN_PARAM_CTF(23, gaussian_K2);
	    ASSIGN_PARAM_CTF(24, sigma2);
	    ASSIGN_PARAM_CTF(25, Gc2);
	    ASSIGN_PARAM_CTF(26, phase_shift);
	    ASSIGN_PARAM_CTF(27, VPP_radius);

}

#define COPY_ctfmodel_TO_CURRENT_GUESS \
    global_prm->assignParametersFromCTF_fast(global_prm->current_ctfmodel, \
                               MATRIX1D_ARRAY(*global_prm->adjust_params),0,ALL_CTF_PARAMETERS);


/* Read parameters --------------------------------------------------------- */
void ProgCTFEstimateFromPSDFast::readBasicParams(XmippProgram *program)
{
	ProgCTFBasicParams::readBasicParams(program);

	initial_ctfmodel.enable_CTF = initial_ctfmodel.enable_CTFnoise = true;
	initial_ctfmodel.readParams(program);
	initial_ctfmodel2D.readParams(program);

	if (initial_ctfmodel.Defocus>100e3)
		REPORT_ERROR(ERR_ARG_INCORRECT,"Defocus cannot be larger than 10 microns (100,000 Angstroms)");
	Tm = initial_ctfmodel.Tm;
}

void ProgCTFEstimateFromPSDFast::readParams()
{
	fn_psd = getParam("--psd");
	readBasicParams(this);
}

/* Usage ------------------------------------------------------------------- */
void ProgCTFEstimateFromPSDFast::defineBasicParams(XmippProgram * program)
{
	CTFDescription::defineParams(program);
}

void ProgCTFEstimateFromPSDFast::defineParams()
{
	defineBasicParams(this);
	ProgCTFBasicParams::defineParams();
}

/* Produce side information ------------------------------------------------ */
void ProgCTFEstimateFromPSDFast::produceSideInfo()
{
    adjust.resize(ALL_CTF_PARAMETERS);
    adjust.initZeros();
    current_ctfmodel.clear();
    ctfmodel_defoci.clear();
    assignParametersFromCTF_fast(initial_ctfmodel, MATRIX1D_ARRAY(adjust), 0, ALL_CTF_PARAMETERS);
    // Read the CTF file, supposed to be the uncentered squared amplitudes
    if (fn_psd != "")
        ctftomodel.read(fn_psd);
    f = &(ctftomodel());

    ProgCTFBasicParams::produceSideInfo();
    current_ctfmodel.precomputeValues(x_contfreq);
}

void ProgCTFEstimateFromPSDFast::generateModelSoFar_fast(MultidimArray<double> &I, bool apply_log = false)
{
    Matrix1D<int> idx(1); // Indexes for Fourier plane
    Matrix1D<double> freq(1); // Frequencies for Fourier plane

    assignCTFfromParameters_fast(MATRIX1D_ARRAY(*adjust_params), current_ctfmodel,
                            0, ALL_CTF_PARAMETERS);
    current_ctfmodel.produceSideInfo();

    I.initZeros(psd_exp_radial);
    FOR_ALL_ELEMENTS_IN_ARRAY1D(I)
    {
        XX(idx) = i;
        FFT_idx2digfreq(*f, idx, freq);
        double w=freq.module();
        if (w>max_freq_psd)
        	continue;
        digfreq2contfreq(freq, freq, Tm);
        // Decide what to save
        current_ctfmodel.precomputeValues(XX(freq));
        if (action <= 1)
            I(i) = current_ctfmodel.getValueNoiseAt();
        else if (action == 2)
        {
            double E = current_ctfmodel.getValueDampingAt();
            I(i) = current_ctfmodel.getValueNoiseAt() + E * E;

        }
        else if (action >= 3 && action <= 6)
        {
            double ctf = current_ctfmodel.getValuePureAt();
            I(i) = current_ctfmodel.getValueNoiseAt() + ctf * ctf;
        }
        else
        {
            double ctf = current_ctfmodel.getValuePureAt();
            I(i) = ctf;
        }
        if (apply_log)
            I(i) = 10 * log10(I(i));

    }
}

void ProgCTFEstimateFromPSDFast::saveIntermediateResults_fast(const FileName &fn_root, bool generate_profiles = true)
{
    std::ofstream plot_radial;
    MultidimArray<double> save;
    generateModelSoFar_fast(save, false);
    if (!generate_profiles)
        return;
    plot_radial.open((fn_root + "_radial.txt").c_str());
    if (!plot_radial)
        REPORT_ERROR(
            ERR_IO_NOWRITE,
            "save_intermediate_results::Cannot open plot file for writing\n");

    //plot_radial << "# freq_dig freq_angstrom model psd enhanced logModel logPsd\n";

    // Generate radial average
    MultidimArray<double> radial_CTFmodel_avg;
    radial_CTFmodel_avg.initZeros(psd_exp_enhanced_radial);

    FOR_ALL_ELEMENTS_IN_ARRAY1D(w_digfreq)
    {
        if (mask(i) <= 0)
            continue;
        double model2 = save(i);

        int r = w_digfreq_r(i);
        radial_CTFmodel_avg(r) = model2;

        plot_radial << w_digfreq(r, 0) << " " << w_contfreq(r, 0)
		<< " " << radial_CTFmodel_avg(r) << " "
		<< psd_exp_enhanced_radial(r) << " " << psd_exp_radial(r)
		<< " " << log10(radial_CTFmodel_avg(r)) << " "
		<< std::endl;
    }

    plot_radial.close();
}

/* CTF fitness ------------------------------------------------------------- */
/* This function measures the distance between the estimated CTF and the
 measured CTF */
double ProgCTFEstimateFromPSDFast::CTF_fitness_object_fast(double *p)
{
	double retval;

    // Generate CTF model
    switch (action)
    {
        // Remind that p is a vector whose first element is at index 1
    case 0:
        assignCTFfromParameters_fast(p - FIRST_SQRT_PARAMETER + 1,
        						current_ctfmodel, FIRST_SQRT_PARAMETER, SQRT_CTF_PARAMETERS);

        if (show_inf >= 2)
        {
            std::cout << "Input vector:";
            for (int i = 1; i <= SQRT_CTF_PARAMETERS; i++)
                std::cout << p[i] << " ";
            std::cout << std::endl;
        }
        break;
    case 1:
            assignCTFfromParameters_fast(p - FIRST_SQRT_PARAMETER + 1,
            					current_ctfmodel, FIRST_SQRT_PARAMETER,
                                    BACKGROUND_CTF_PARAMETERS);
        if (show_inf >= 2)
        {
            std::cout << "Input vector:";
            for (int i = 1; i <= BACKGROUND_CTF_PARAMETERS; i++)
                std::cout << p[i] << " ";
            std::cout << std::endl;
        }
        break;
    case 2:
            assignCTFfromParameters_fast(p - FIRST_ENVELOPE_PARAMETER + 1,
            					current_ctfmodel, FIRST_ENVELOPE_PARAMETER, ENVELOPE_PARAMETERS);
        if (show_inf >= 2)
        {
            std::cout << "Input vector:";
            for (int i = 1; i <= ENVELOPE_PARAMETERS; i++)
                std::cout << p[i] << " ";
            std::cout << std::endl;
        }
        break;
    case 3:
            assignCTFfromParameters_fast(p - FIRST_DEFOCUS_PARAMETER + 1,
            						current_ctfmodel, FIRST_DEFOCUS_PARAMETER, DEFOCUS_PARAMETERS);
        psd_theo_radial_derivative.initZeros();
        if (show_inf >= 2)
        {
            std::cout << "Input vector:";
            for (int i = 1; i <= DEFOCUS_PARAMETERS; i++)
                std::cout << p[i] << " ";
            std::cout << std::endl;
        }
        break;
    case 4:
            assignCTFfromParameters_fast(p - 0 + 1, current_ctfmodel, 0,
                                    CTF_PARAMETERS);
        psd_theo_radial.initZeros();
        if (show_inf >= 2)
        {
            std::cout << "Input vector:";
            for (int i = 1; i <= CTF_PARAMETERS; i++)
                std::cout << p[i] << " ";
            std::cout << std::endl;
        }
        break;
    case 5:
    case 6:
    case 7:
		assignCTFfromParameters_fast(p - 0 + 1, current_ctfmodel, 0,
								ALL_CTF_PARAMETERS);
        psd_theo_radial.initZeros();
        if (show_inf >= 2)
        {
            std::cout << "Input vector:";
            for (int i = 1; i <= ALL_CTF_PARAMETERS; i++)
                std::cout << p[i] << " ";
            std::cout << std::endl;
        }
        break;
    }

    current_ctfmodel.produceSideInfo();
    if (show_inf >= 2)
        std::cout << "Model:\n" << current_ctfmodel << std::endl;
    if (!current_ctfmodel.hasPhysicalMeaning())
    {
        if (show_inf >= 2)
            std::cout << "Does not have physical meaning\n";
        return heavy_penalization;
    }
    if (action > 3
        && (fabs((current_ctfmodel.Defocus - ctfmodel_defoci.Defocus)
                / ctfmodel_defoci.Defocus) > 0.2)  && !noDefocusEstimate)
    {
        if (show_inf >= 2)
            std::cout << "Too large defocus\n";
        return heavy_penalization;
    }
    if (initial_ctfmodel.Defocus != 0 && action >= 3 && !selfEstimation && !noDefocusEstimate)
    {
        // If there is an initial model, the true solution
        // cannot be too far
        if (fabs(initial_ctfmodel.Defocus - current_ctfmodel.Defocus) > defocus_range)
        {
            if (show_inf >= 2)
            {
                std::cout << "Too far from hint: Initial (" << initial_ctfmodel.Defocus << ")"
                << " current guess (" << current_ctfmodel.Defocus << ") max allowed difference: "
                << defocus_range << std::endl;
            }
            return heavy_penalization;


        }
    }
    if ((initial_ctfmodel.phase_shift != 0.0 || initial_ctfmodel.phase_shift != 1.57079) && action >= 5)
    {
    	if (fabs(initial_ctfmodel.phase_shift - current_ctfmodel.phase_shift) > 0.26)
    	{
    		return heavy_penalization;
    	}
    }
    // Now the 1D error
    double distsum = 0;
    int N = 0, Ncorr = 0;
    double enhanced_avg = 0;
    double model_avg = 0;
    double enhanced_model = 0;
    double enhanced2 = 0;
    double model2 = 0;
    double lowerLimit = 1.1 * min_freq_psd;
    double upperLimit = 0.9 * max_freq_psd;
    const MultidimArray<double>& local_enhanced_ctf = psd_exp_enhanced_radial;
    int XdimW=XSIZE(w_digfreq);
    corr13=0;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(w_digfreq)
    {
		if (DIRECT_A1D_ELEM(mask, i) <= 0)
			continue;

		// Compute each component
		current_ctfmodel.precomputeValues(i);
		double bg = current_ctfmodel.getValueNoiseAt();

		double envelope=0, ctf_without_damping, ctf_with_damping=0, current_envelope = 0;
		double ctf2_th=0;
		double ctf2 = DIRECT_A1D_ELEM(psd_exp_radial, i);
		double dist = 0;
		double ctf_with_damping2;
		switch (action)
		{
		case 0:
		case 1:
			ctf2_th = bg;
			dist = fabs(ctf2 - bg);
			if (penalize && bg > ctf2 && DIRECT_A1D_ELEM(w_digfreq, i) > max_gauss_freq)
				dist *= current_penalty;
			break;
		case 2:
			envelope = current_ctfmodel.getValueDampingAt();
			ctf2_th = bg + envelope * envelope;
			dist = fabs(ctf2 - ctf2_th);
			if (penalize && ctf2_th < ctf2 && DIRECT_A1D_ELEM(w_digfreq, i)	> max_gauss_freq)
				dist *= current_penalty;
			break;
		case 3:
		case 4:
		case 5:
		case 6:
		case 7:
			envelope = current_ctfmodel.getValueDampingAt();
			ctf_without_damping = current_ctfmodel.getValuePureWithoutDampingAt();

			ctf_with_damping = envelope * ctf_without_damping;
			ctf2_th = bg + ctf_with_damping * ctf_with_damping;

			if (DIRECT_A1D_ELEM(w_digfreq,i) < upperLimit
							&& DIRECT_A1D_ELEM(w_digfreq,i) > lowerLimit)
			{
				if  (action == 3 ||
					 (action == 4 && DIRECT_A1D_ELEM(mask_between_zeroes,i) == 1) ||
					 (action == 7 && DIRECT_A1D_ELEM(mask_between_zeroes,i) == 1))
				{
					double enhanced_ctf = DIRECT_A1D_ELEM(local_enhanced_ctf, i);
					ctf_with_damping2 = ctf_with_damping * ctf_with_damping;
					enhanced_model += enhanced_ctf * ctf_with_damping2;
					enhanced2 += enhanced_ctf * enhanced_ctf;
					model2 += ctf_with_damping2 * ctf_with_damping2;
					enhanced_avg += enhanced_ctf;
					model_avg += ctf_with_damping2;
					Ncorr++;
					if (action==3)
					{
						int r = A1D_ELEM(w_digfreq_r,i);
						A1D_ELEM(psd_theo_radial,r) = ctf2_th;
					}
				}
			}
			if (envelope > 1e-2)
				dist = fabs(ctf2 - ctf2_th) / (envelope * envelope);
			else
				dist = fabs(ctf2 - ctf2_th);
			// This expression comes from mapping any value so that
			// bg becomes 0, and bg+envelope^2 becomes 1
			// This is the transformation
			//        (x-bg)      x-bg
			//    -------------=-------
			//    (bg+env^2-bg)  env^2
			// If we subtract two of this scaled values
			//    x-bg      y-bg       x-y
			//   ------- - ------- = -------
			//    env^2     env^2     env^2
			break;

		}

		distsum += dist * DIRECT_A1D_ELEM(mask,i);
		N++;
	}
    if (N > 0)
    { retval = distsum/N ;
    }
    else
        retval = heavy_penalization;
    if (show_inf >=2)
        std::cout << "Fitness1=" << retval << std::endl;
    if ( (((action >= 3) && (action <= 4)) || (action == 7))
         && (Ncorr > 0) && (enhanced_weight != 0) )
    {

    	model_avg /= Ncorr;
    	enhanced_avg /= Ncorr;
        double correlation_coeff = enhanced_model/Ncorr - model_avg * enhanced_avg;
        double sigma1 = sqrt(fabs(enhanced2/Ncorr - enhanced_avg * enhanced_avg));
        double sigma2 = sqrt(fabs(model2/Ncorr - model_avg * model_avg));
        double maxSigma = std::max(sigma1, sigma2);
        if (sigma1 < XMIPP_EQUAL_ACCURACY || sigma2 < XMIPP_EQUAL_ACCURACY
            || (fabs(sigma1 - sigma2) / maxSigma > 0.9 && action>=5))
        {
            retval = heavy_penalization;
            if (show_inf>=2)
                std::cout << "Fitness2=" << heavy_penalization << " sigma1=" << sigma1 << " sigma2=" << sigma2 << std::endl;
        }
        else
        {
            correlation_coeff /= sigma1 * sigma2;
            if (action == 7)
                corr13 = correlation_coeff;
            else
                retval -= enhanced_weight * correlation_coeff;
            if (show_inf >= 2)
            {
                std::cout << "model_avg=" << model_avg << std::endl;
                std::cout << "enhanced_avg=" << enhanced_avg << std::endl;
                std::cout << "enhanced_model=" << enhanced_model / Ncorr
                << std::endl;
                std::cout << "sigma1=" << sigma1 << std::endl;
                std::cout << "sigma2=" << sigma2 << std::endl;
                std::cout << "Fitness2="
                << -(enhanced_weight * correlation_coeff)
                << " (" << correlation_coeff << ")" << std::endl;
            }
        }

        // Correlation of the derivative of the radial profile
        if (action==3 || evaluation_reduction==1)
        {
            int state=0;
            double maxDiff=0;
            psd_theo_radial_derivative.initZeros();
            double lowerlimt=1.1*min_freq;
            double upperlimit=0.9*max_freq;
            FOR_ALL_ELEMENTS_IN_ARRAY1D(psd_theo_radial)
            if (A1D_ELEM(w_digfreq_r_iN,i)>0)
            {
                double freq=A1D_ELEM(w_digfreq,i);
                switch (state)
                {
                case 0:
                    if (freq>lowerlimt)
                        state=1;
                    break;
                case 1:
                    if (freq>upperlimit)
                        state=2;
                    else
                    {
                        double diff=A1D_ELEM(psd_theo_radial,i)-A1D_ELEM(psd_theo_radial,i-1);

                        A1D_ELEM(psd_theo_radial_derivative,i)=diff;
                        maxDiff=std::max(maxDiff,fabs(diff));
                    }
                    break;
                }
            }
            double corrRadialDerivative=0,mux=0, muy=0, Ncorr=0, sigmax=0, sigmay=0;
            double iMaxDiff=1.0/maxDiff;

            FOR_ALL_ELEMENTS_IN_ARRAY1D(psd_theo_radial)
            {
                A1D_ELEM(psd_theo_radial_derivative,i)*=iMaxDiff;
                double x=A1D_ELEM(psd_exp_enhanced_radial_derivative,i);
                double y=A1D_ELEM(psd_theo_radial_derivative,i);
                corrRadialDerivative+=x*y;
                mux+=x;
                muy+=y;
                sigmax+=x*x;
                sigmay+=y*y;
                Ncorr++;
            }

            if (Ncorr>0)
            {
                double iNcorr=1.0/Ncorr;
                corrRadialDerivative*=iNcorr;
                mux*=iNcorr;
                muy*=iNcorr;
                sigmax=sqrt(fabs(sigmax*iNcorr-mux*mux));
                sigmay=sqrt(fabs(sigmay*iNcorr-muy*muy));
                corrRadialDerivative=(corrRadialDerivative-mux*muy)/(sigmax*sigmay);

            }
            retval-=corrRadialDerivative;
            if (show_inf>=2)
            {
                std::cout << "Fitness3=" << -corrRadialDerivative << std::endl;
                if (show_inf==3)
                {
                    psd_exp_enhanced_radial.write("PPPexpRadial_fast.txt");
                    psd_theo_radial.write("PPPtheoRadial.txt");
                    psd_exp_enhanced_radial_derivative.write("PPPexpRadialDerivative_fast.txt");
                    psd_theo_radial_derivative.write("PPPtheoRadialDerivative.txt");
                }
           }
        }
    }

    return retval;
}

double CTF_fitness_fast(double *p, void *vprm)
{
	auto *prm=(ProgCTFEstimateFromPSDFast *) vprm;
	return prm->CTF_fitness_object_fast(p);
}

/* Center focus ----------------------------------------------------------- */
void ProgCTFEstimateFromPSDFast::center_optimization_focus_fast(bool adjust_th, double margin = 1)
{
    if (show_optimization)
        std::cout << "Freq frame before focusing=" << min_freq_psd << ","
        << max_freq_psd << std::endl << "Value_th before focusing="
        << value_th << std::endl;

    double w1 = min_freq_psd, w2 = max_freq_psd;

    // Compute maximum value within central region
    if (adjust_th)
    {
        MultidimArray<double> save;
        generateModelSoFar_fast(save, false);
        double max_val = 0;
        FOR_ALL_ELEMENTS_IN_ARRAY1D(w_digfreq)
        {
            double w = w_digfreq(i);
            if (w >= w1 && w <= w2)
                max_val = XMIPP_MAX(max_val, save(i));
        }
        if (value_th != -1)
            value_th = XMIPP_MIN(value_th, max_val * margin);
        else
            value_th = max_val * margin;
    }


    if (show_optimization)
        std::cout << "Freq frame after focusing=" << min_freq_psd << ","
        << max_freq_psd << std::endl << "Value_th after focusing="
        << value_th << std::endl;
}
// Estimate sqrt parameters ------------------------------------------------
// Results are written in current_ctfmodel
void ProgCTFEstimateFromPSDFast::estimate_background_sqrt_parameters_fast()
{
    if (show_optimization)
        std::cout << "Computing first sqrt background ...\n";

    // Estimate the base line taking the value of the CTF
    // for the maximum X
    double base_line = 0;
    int N = 0;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(w_digfreq)
    if (w_digfreq(i) > 0.4)
    {
        N++;
        base_line += psd_exp_radial(i);
    }
    if (0 == N) {
        REPORT_ERROR(ERR_NUMERICAL, "N is zero (0), which would lead to division by zero");
    }
    current_ctfmodel.base_line = base_line / N;

    // Find the linear least squares solution for the sqrt part
    Matrix2D<double> A(2, 2);
    A.initZeros();
    Matrix1D<double> b(2);
    b.initZeros();

    FOR_ALL_ELEMENTS_IN_ARRAY1D(w_digfreq)
    {
        if (mask(i) <= 0)
            continue;

        // Compute weight for this point
        double weight = 1 + max_freq_psd - w_digfreq(i);

        // Compute error
        current_ctfmodel.precomputeValues(x_contfreq(i));
        double explained = current_ctfmodel.getValueNoiseAt();
        double unexplained = psd_exp_radial(i) - explained;
        if (unexplained <= 0)
            continue;
        unexplained = log(unexplained);

        double X = -sqrt(w_contfreq(i));
        A(0, 0) += weight * X * X;
        A(0, 1) += weight * X;
        A(1, 1) += weight * 1;
        b(0) += X * weight * unexplained;
        b(1) += weight * unexplained;

    }
    A(1, 0) = A(0, 1);
    b = A.inv() * b;

    current_ctfmodel.sq = b(0);
    current_ctfmodel.sqrt_K = exp(b(1));
    COPY_ctfmodel_TO_CURRENT_GUESS;

    if (show_optimization)
    {
        std::cout << "First SQRT Fit:\n" << current_ctfmodel << std::endl;
        saveIntermediateResults_fast("step01a_first_sqrt_fit_fast", true);
    }

    // Now optimize .........................................................
	double fitness;
	Matrix1D<double> steps;
	steps.resize(SQRT_CTF_PARAMETERS);
	steps.initConstant(1);

	// Optimize without penalization
	if (show_optimization)
		std::cout << "Looking for best fitting sqrt ...\n";
	penalize = false;
	int iter;
	powellOptimizer(*adjust_params, FIRST_SQRT_PARAMETER + 1,
					SQRT_CTF_PARAMETERS, CTF_fitness_fast, global_prm, 0.05, fitness, iter, steps,
					show_optimization);

	// Optimize with penalization
	if (show_optimization)
		std::cout << "Penalizing best fitting sqrt ...\n";
	penalize = true;
	current_penalty = 2;
	int imax = CEIL(log(penalty) / log(2.0));

	for (int i = 1; i <= imax; i++)
	{
		if (show_optimization)
			std::cout << "     Iteration " << i << " penalty="
			<< current_penalty << std::endl;
		powellOptimizer(*adjust_params, FIRST_SQRT_PARAMETER + 1,
						SQRT_CTF_PARAMETERS, CTF_fitness_fast, global_prm, 0.05, fitness, iter,
						steps, show_optimization);
		current_penalty *= 2;
		current_penalty =
			XMIPP_MIN(current_penalty, penalty);
	}
	// Keep the result in adjust
	current_ctfmodel.forcePhysicalMeaning();
	COPY_ctfmodel_TO_CURRENT_GUESS;

	if (show_optimization)
	{
		std::cout << "Best penalized SQRT Fit:\n" << current_ctfmodel
		<< std::endl;
		saveIntermediateResults_fast("step01b_best_penalized_sqrt_fit_fast", true);
    }

   center_optimization_focus_fast(true, 1.5);

}

// Estimate gaussian parameters --------------------------------------------
//#define DEBUG
void ProgCTFEstimateFromPSDFast::estimate_background_gauss_parameters_fast()
{

    if (show_optimization)
        std::cout << "Computing first background Gaussian parameters ...\n";

    // Compute radial averages
    MultidimArray<double> radial_CTFmodel_avg(YSIZE(*f) / 2);
    MultidimArray<double> radial_CTFampl_avg(YSIZE(*f) / 2);
    MultidimArray<int> radial_N(YSIZE(*f) / 2);
    double w_max_gauss = 0.25;

    FOR_ALL_ELEMENTS_IN_ARRAY1D(w_digfreq)
    {
        if (mask(i) <= 0)
            continue;
        double w = w_digfreq(i);

        if (w > w_max_gauss)
            continue;

        int r = FLOOR(w * (double)YSIZE(*f));

        current_ctfmodel.precomputeValues(x_contfreq(i));
        radial_CTFmodel_avg(r) += current_ctfmodel.getValueNoiseAt();
        radial_CTFampl_avg(r) += psd_exp_radial(i);
        radial_N(r)++;

    }
    // Compute the average radial error
    double  error2_avg = 0;
    int N_avg = 0;
    MultidimArray<double> error;
    error.initZeros(radial_CTFmodel_avg);
    FOR_ALL_ELEMENTS_IN_ARRAY1D(radial_CTFmodel_avg)
    {
        if (radial_N(i) == 0)
            continue;
        error(i) = (radial_CTFampl_avg(i) - radial_CTFmodel_avg(i))
                   / radial_N(i);
        error2_avg += error(i) * error(i);
        N_avg++;
    }
    if (N_avg != 0)
        error2_avg /= N_avg;

#ifdef DEBUG

    std::cout << "Error2 avg=" << error2_avg << std::endl;
#endif

    // Compute the minimum radial error
    bool first = true, OK_to_proceed = false;
    double error2_min = 0, wmin=0;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(radial_CTFmodel_avg)
    {
        if (radial_N(i) == 0)
            continue;

        double w = w_digfreq(i);

        if (error(i) < 0 && first){
        	continue;
        }

        else if (error(i) < 0)
        {
        	break;
        }

        double error2 = error(i) * error(i);
        // If the two lines cross, do not consider any error until
        // the cross is "old" enough
        if (first && error2 > 0.15 * error2_avg)
            OK_to_proceed = true;
        if (first && i > 0)
            OK_to_proceed &= (error(i) < error(i - 1));

        // If the error now is bigger than a 30% (1.69=1.3*1.3) of the error min
        // this must be a rebound. Stop here
        if (!first && error2 > 1.69 * error2_min)
            break;
        if (first && OK_to_proceed)
        {
            wmin = w;
            error2_min = error2;
            first = false;
        }
        if (!first && error2 < error2_min)
        {
            wmin = w;
            error2_min = error2;
        }
#ifdef DEBUG
        std::cout << w << " " << error2 << " " << wmin << " " << std::endl;
#endif

    }

    // Compute the frequency of the minimum error
    max_gauss_freq = wmin;
#ifdef DEBUG

    std::cout << "Freq of the minimum error: " << wmin << " " << fmin << std::endl;
#endif

    // Compute the maximum radial error
    first = true;
    double error2_max = 0, wmax=0, fmax;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(radial_CTFmodel_avg)
    {
        if (radial_N(i) == 0)
            continue;
        double w = w_digfreq(i);
        if (w > wmin)
            continue;

        if (error(i) < 0 && first)
            continue;
        else if (error(i) < 0)
            break;
        double error2 = error(i) * error(i);
        if (first)
        {
            wmax = w;
            error2_max = error2;
            first = false;
        }
        if (error2 > error2_max)
        {
            wmax = w;
            error2_max = error2;
        }
#ifdef DEBUG
        std::cout << w << " " << error2 << " " << wmax << std::endl;
#endif

    }
    fmax = current_ctfmodel.Gc1 = wmax / Tm;
#ifdef DEBUG

    std::cout << "Freq of the maximum error: " << wmax << " " << fmax << std::endl;
#endif

    // Find the linear least squares solution for the gauss part
    Matrix2D<double> A(2, 2);
    A.initZeros();
    Matrix1D<double> b(2);
    b.initZeros();


    FOR_ALL_ELEMENTS_IN_ARRAY1D(w_digfreq)
    {

        if (mask(i) <= 0)
            continue;

        if (w_digfreq(i) > wmin)
        	continue;

        double fmod = w_contfreq(i);

        // Compute weight for this point
        double weight = 1 + max_freq_psd - w_digfreq(i);

        // Compute error
        current_ctfmodel.precomputeValues(x_contfreq(i));
        double explained = current_ctfmodel.getValueNoiseAt();
        double unexplained = psd_exp_radial(i) - explained;
        if (unexplained <= 0)
            continue;
        unexplained = log(unexplained);
        double F = -(fmod - fmax) * (fmod - fmax);

        A(0, 0) += weight * 1;
        A(0, 1) += weight * F;
        A(1, 1) += weight * F * F;
        b(0) += weight * unexplained;
        b(1) += F * weight * unexplained;

    }
    A(1, 0) = A(0, 1);

    if ( (A(0, 0)== 0) && (A(1, 0)== 0) && (A(1, 1)== 0))
    {
        std::cout << "Matrix A is zeros" << std::endl;
    }
    else
    {

        b = A.inv() * b;
        current_ctfmodel.sigma1 = XMIPP_MIN(fabs(b(1)), 95e3); // This value should beconformant with the physical
        // meaning routine in CTF.cc
        current_ctfmodel.gaussian_K = exp(b(0));
        // Store the CTF values in adjust
        current_ctfmodel.forcePhysicalMeaning();
        COPY_ctfmodel_TO_CURRENT_GUESS;

        if (show_optimization)
        {
            std::cout << "First Background Fit:\n" << current_ctfmodel << std::endl;
            saveIntermediateResults_fast("step01c_first_background_fit_fast", true);
        }
        center_optimization_focus_fast(true, 1.5);

    }
}

// Estimate envelope parameters --------------------------------------------
//#define DEBUG
void ProgCTFEstimateFromPSDFast::estimate_envelope_parameters_fast()
{
    if (show_optimization)
        std::cout << "Looking for best fitting envelope ...\n";

    // Set the envelope
    current_ctfmodel.Ca = initial_ctfmodel.Ca;
    current_ctfmodel.K = 1.0;
    current_ctfmodel.espr = 0.0;
    current_ctfmodel.ispr = 0.0;
    current_ctfmodel.alpha = 0.0;
    current_ctfmodel.DeltaF = 0.0;
    current_ctfmodel.DeltaR = 0.0;
    current_ctfmodel.Q0 = initial_ctfmodel.Q0;
    current_ctfmodel.envR0 = 0.0;
    current_ctfmodel.envR1 = 0.0;
    COPY_ctfmodel_TO_CURRENT_GUESS;

    // Now optimize the envelope
    penalize = false;
    int iter;
    double fitness;
    Matrix1D<double> steps;
    steps.resize(ENVELOPE_PARAMETERS);
    steps.initConstant(1);

    steps(1) = 0; // Do not optimize Cs
    steps(5) = 0; // Do not optimize for alpha, since Ealpha depends on the defocus

    if (modelSimplification >= 1)
        steps(6) = steps(7) = 0; // Do not optimize DeltaF and DeltaR
    powellOptimizer(*adjust_params, FIRST_ENVELOPE_PARAMETER + 1,
                    ENVELOPE_PARAMETERS, CTF_fitness_fast, global_prm, 0.05, fitness, iter, steps,
                    show_optimization);

    // Keep the result in adjust
    current_ctfmodel.forcePhysicalMeaning();
    COPY_ctfmodel_TO_CURRENT_GUESS;

    if (show_optimization)
    {
        std::cout << "Best envelope Fit:\n" << current_ctfmodel << std::endl;
        saveIntermediateResults_fast("step02a_best_envelope_fit_fast", true);
    }

    // Optimize with penalization
    if (show_optimization)
        std::cout << "Penalizing best fitting envelope ...\n";
    penalize = true;
    current_penalty = 2;
    int imax = CEIL(log(penalty) / log(2.0));
    for (int i = 1; i <= imax; i++)
    {
        if (show_optimization)
            std::cout << "     Iteration " << i << " penalty="
            << current_penalty << std::endl;
        powellOptimizer(*adjust_params, FIRST_ENVELOPE_PARAMETER + 1,
                        ENVELOPE_PARAMETERS, CTF_fitness_fast, global_prm, 0.05, fitness, iter,
                        steps, show_optimization);
        current_penalty *= 2;
        current_penalty =
            XMIPP_MIN(current_penalty, penalty);
    }
    // Keep the result in adjust
    current_ctfmodel.forcePhysicalMeaning();
    COPY_ctfmodel_TO_CURRENT_GUESS;

    if (show_optimization)
    {
        std::cout << "Best envelope Fit:\n" << current_ctfmodel << std::endl;
        saveIntermediateResults_fast("step02b_best_penalized_envelope_fit_fast", true);
    }

}

// Estimate defoci ---------------------------------------------------------
//#define DEBUG
#define OUTLIERS
//#define FILTER
void ProgCTFEstimateFromPSDFast::estimate_defoci_fast()
{
	double fitness;
    int iter;
	Matrix1D<double> steps(DEFOCUS_PARAMETERS);
	steps.initConstant(1);
	steps(1) = 0; // Do not optimize kV
	steps(2) = 0; // Do not optimize K
	if(!selfEstimation)
	{
		(*adjust_params)(0) = initial_ctfmodel.Defocus;
		(*adjust_params)(2) = current_ctfmodel.K;
		powellOptimizer(*adjust_params, FIRST_DEFOCUS_PARAMETER + 1,
								DEFOCUS_PARAMETERS, CTF_fitness_fast, global_prm, 0.05,
								fitness, iter, steps, false);
	}
	else
	{
		/*
		 * Defocus estimation s^2 stigmator
		 *
		 * CTF = sin(PI*lambda*defocus*R^2)
		 * CTF = sin(PI*lambda*defocus*w^2/Ts^2)
		 * w^2 = omega
		 * omega = 2*K0/Xdim   K0: position of the maximum of the psd in fourier
		 *
		 * Defocus = 2*K0*(2*Ts)^2/lambda
		 */
		MultidimArray<double> psd_exp_radial2;
		MultidimArray<double> background;
		MultidimArray<double> psd_background;
		action = 1;
		generateModelSoFar_fast(background, false);
		action = 3;
		psd_background.initZeros(background);

#ifdef DEBUG
		std::cout << "background\n" << background << std::endl;
		std::cout << "--------------------------------------" << std::endl;

		std::cout << "psd_exp_radial\n" << psd_exp_radial << std::endl;
		std::cout << "--------------------------------------" << std::endl;
#endif

		FOR_ALL_ELEMENTS_IN_ARRAY1D(background)
		{
			if(w_digfreq(i)>min_freq && w_digfreq(i)<max_freq)
				psd_background(i)= background(i)-psd_exp_radial(i);
		}

		//Uncomment to apply a high-pass FILTER to psd_background
#ifdef FILTER
		double alpha = 0.1;
		Matrix1D<double> A_coeff(2);
		VEC_ELEM(A_coeff,0) = 1;
		VEC_ELEM(A_coeff,1) = alpha-1;
		Matrix1D<double> B_coeff(2);
		VEC_ELEM(B_coeff,0) = 1-alpha;
		VEC_ELEM(B_coeff,1) = alpha-1;
		MultidimArray<double> psd_background_filter;
		MultidimArray<double> aux_filter;
		psd_background_filter.initZeros(psd_background);
		aux_filter.initZeros(psd_background);
		matlab_filter(B_coeff,A_coeff,psd_background,psd_background_filter,aux_filter);
		psd_background = psd_background_filter;
#endif

#ifdef DEBUG
		std::cout << "psdbackground\n" << psd_background << std::endl;
		std::cout << "--------------------------------------" << std::endl;
#endif
		psd_exp_radial2.initZeros(psd_exp_radial);

		double deltaW=1.0/(2*XSIZE(w_digfreq)*current_ctfmodel.Tm);
		double deltaW2=1.0/(XSIZE(w_digfreq)*pow(2*current_ctfmodel.Tm,2));
		FOR_ALL_ELEMENTS_IN_ARRAY1D(psd_exp_radial2)
		{
			double w2=i*deltaW2;
			double w=sqrt(w2);
			double widx=w/deltaW;
			size_t lowerIdx=floor(widx);
			double weight=widx-floor(widx);
			if(i < XSIZE(psd_exp_radial2)-1)
			{
				double value =(1-weight)*A1D_ELEM(psd_background,lowerIdx)
													 +weight*A1D_ELEM(psd_background,lowerIdx+1);

				A1D_ELEM(psd_exp_radial2,i) = value*value;
			}
		}
#ifdef DEBUG
		std::cout << "psd_exp_radial2\n" << psd_exp_radial2 << std::endl;
		std::cout << "--------------------------------------" << std::endl;
#endif
		FourierTransformer FourierPSD;
		FourierPSD.FourierTransform(psd_exp_radial2, psd_fft, false);

		const double minDefocus=2000;
		int startIndex = ceil(std::max((double)7,minDefocus*current_ctfmodel.lambda/(2*pow(2*current_ctfmodel.Tm,2))));
#ifdef DEBUG
		std::cout << "Start index=" << minDefocus*current_ctfmodel.lambda/(2*pow(2*current_ctfmodel.Tm,2)) << std::endl;
		std::cout << "--------------------------------------" << std::endl;
#endif
		if (current_ctfmodel.VPP_radius != 0.0)
			startIndex = 10; //avoid low frequencies
		FOR_ALL_ELEMENTS_IN_ARRAY1D(psd_fft)
		{
			if(i>=startIndex) {
				amplitud.push_back(abs(psd_fft[i]));
#ifdef DEBUG
		std::cout << abs(psd_fft[i]) << std::endl;
#endif
			}
		}
#ifdef DEBUG
		std::cout << "--------------------------------------" << std::endl;
#endif

		//Uncomment to remove outliers from amplitud
#ifdef OUTLIERS
		double totalSum = std::accumulate(amplitud.begin(), amplitud.end(), 0.0);

		double amplitudMean = totalSum / amplitud.size();
		double sdSum = 0;
		for(double i : amplitud)
		{
			double diff=amplitud[i]-amplitudMean;
			sdSum += diff*diff;
		}
		double amplitudSD = sqrt(sdSum/amplitud.size());
		double differenceSD = 3*amplitudSD;
		for(double i : amplitud)
		{
			if (amplitud[i]>amplitudMean+differenceSD){
				amplitud[i] = amplitudMean+differenceSD;
			}
		}
#endif

		double maxValue=-1e38;
		int finalIndex=-1;
#ifdef DEBUG
		std::cout << "Looking for maximum ...\n";
#endif
		for (size_t i=1; i<amplitud.size()-1; i++)
		{
#ifdef DEBUG
			std::cout << "i=" << i << " amplitude i= " << amplitud[i] << std::endl;
#endif
			if (amplitud[i]>maxValue && amplitud[i]>=amplitud[i+1])
			{
				maxValue=amplitud[i];
				finalIndex=i;
#ifdef DEBUG
				std::cout << "New maximum " << i << " maxValue=" << maxValue << std::endl;
#endif
			}
		}

		//double maxValue = *max_element(amplitud.begin(),amplitud.end());
		//int finalIndex = distance(amplitud.begin(),max_element(amplitud.begin(),amplitud.end()));
		current_ctfmodel.Defocus = (floor(finalIndex+startIndex+1)*pow(2*current_ctfmodel.Tm,2))/
												current_ctfmodel.lambda;

#ifdef DEBUG
		std::cout << "maxValue: " << maxValue << std::endl;
		std::cout << "finalIndex: " << finalIndex << std::endl;
		std::cout << "defocus: " << current_ctfmodel.Defocus << std::endl;
#endif
	}
	// Keep the result in adjust
	current_ctfmodel.forcePhysicalMeaning();
	COPY_ctfmodel_TO_CURRENT_GUESS;
	ctfmodel_defoci = current_ctfmodel;

	if  (show_optimization)
	{
		std::cout << "First defocus Fit:\n" << ctfmodel_defoci << std::endl;
		saveIntermediateResults_fast("step03a_first_defocus_fit_fast", true);
	}
}
#undef DEBUG
#undef FILTER
#undef OUTLIERS

// Estimate second gaussian parameters -------------------------------------
//#define DEBUG
void ProgCTFEstimateFromPSDFast::estimate_background_gauss_parameters2_fast()
{
    if (show_optimization)
        std::cout << "Computing first background Gaussian2 parameters ...\n";

    // Compute radial averages
    MultidimArray<double> radial_CTFmodel_avg(YSIZE(*f) / 2);
    MultidimArray<double> radial_CTFampl_avg(YSIZE(*f) / 2);
    MultidimArray<int> radial_N(YSIZE(*f) / 2);
    double w_max_gauss = 0.25;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(w_digfreq)
    {
        if (mask(i) <= 0)
            continue;
        double w = w_digfreq(i);
        if (w > w_max_gauss)
            continue;

        int r = FLOOR(w * (double)YSIZE(*f));
        double f_x = DIRECT_A1D_ELEM(x_contfreq, i);
        current_ctfmodel.precomputeValues(f_x);
        double bg = current_ctfmodel.getValueNoiseAt();
        double envelope = current_ctfmodel.getValueDampingAt();
        double ctf_without_damping =
            current_ctfmodel.getValuePureWithoutDampingAt();
        double ctf_with_damping = envelope * ctf_without_damping;
        double ctf2_th = bg + ctf_with_damping * ctf_with_damping;
        radial_CTFmodel_avg(r) += ctf2_th;
        radial_CTFampl_avg(r) += psd_exp_radial(i);
        radial_N(r)++;
    }

    // Compute the average radial error
    MultidimArray <double> error;
    error.initZeros(radial_CTFmodel_avg);
    FOR_ALL_ELEMENTS_IN_ARRAY1D(radial_CTFmodel_avg)
    {
        if (radial_N(i) == 0)
            continue;
        error(i) = (radial_CTFampl_avg(i) - radial_CTFmodel_avg(i))/ radial_N(i);
    }
#ifdef DEBUG
    std::cout << "Error:\n" << error << std::endl;
#endif

    // Compute the frequency of the minimum error
    double wmin = 0.15;

    // Compute the maximum (negative) radial error
    double error_max = 0, wmax=0, fmax;
    FOR_ALL_ELEMENTS_IN_ARRAY1D(radial_CTFmodel_avg)
    {
        if (radial_N(i) == 0)
            continue;
        double w = w_digfreq(i);
        if (w > wmin)
            break;
        if (error(i) < error_max)
        {
            wmax = w;
            error_max = error(i);
        }
    }
    fmax = current_ctfmodel.Gc2 = wmax / Tm;
#ifdef DEBUG

    std::cout << "Freq of the maximum error: " << wmax << " " << fmax << std::endl;
#endif

    // Find the linear least squares solution for the gauss part
    Matrix2D<double> A(2, 2);
    A.initZeros();
    Matrix1D<double> b(2);
    b.initZeros();
    int N = 0;

    FOR_ALL_ELEMENTS_IN_ARRAY1D(w_digfreq)
    {
        if (mask(i) <= 0)
            continue;
        if (w_digfreq(i) > wmin)
            continue;
        double fmod = w_contfreq(i);

        // Compute the zero on the direction of this point
        Matrix1D<double> u(1), fzero(1);
        u = x_contfreq(i) / fmod;
        current_ctfmodel.lookFor(1, u, fzero, 0);
        if (fmod > fzero.module())
            continue;

        // Compute weight for this point
        double weight = 1 + max_freq_psd - w_digfreq(i);
        // Compute error
        double f_x = DIRECT_A1D_ELEM(x_contfreq, i);
        current_ctfmodel.precomputeValues(f_x);
        double bg = current_ctfmodel.getValueNoiseAt();
        double envelope = current_ctfmodel.getValueDampingAt();
        double ctf_without_damping =
            current_ctfmodel.getValuePureWithoutDampingAt();
        double ctf_with_damping = envelope * ctf_without_damping;
        double ctf2_th = bg + ctf_with_damping * ctf_with_damping;
        double explained = ctf2_th;
        double unexplained = explained - psd_exp_radial(i);
        if (unexplained <= 0)
            continue;
        unexplained = log(unexplained);
        double F = -(fmod - fmax) * (fmod - fmax);
        A(0, 0) += weight * 1;
        A(0, 1) += weight * F;
        A(1, 1) += weight * F * F;
        b(0) += weight * unexplained;
        b(1) += F * weight * unexplained;
        N++;
    }

    if (N != 0)
    {
        A(1, 0) = A(0, 1);

        double det=A.det();
        if (fabs(det)>1e-9)
        {
            b = A.inv() * b;
            current_ctfmodel.sigma2 = XMIPP_MIN(fabs(b(1)), 95e3); // This value should be conformant with the physical
            // meaning routine in CTF.cc
            current_ctfmodel.gaussian_K2 = exp(b(0));
        }
        else
        {
            current_ctfmodel.sigma2 = 0;
            current_ctfmodel.gaussian_K2 = 0;
        }
    }
    else
    {
        current_ctfmodel.sigma2 = 0;
        current_ctfmodel.gaussian_K2 = 0;
    }

    // Store the CTF values in adjust
    current_ctfmodel.forcePhysicalMeaning();
    COPY_ctfmodel_TO_CURRENT_GUESS;

    if (show_optimization)
    {
        std::cout << "First Background Gaussian 2 Fit:\n" << current_ctfmodel
        << std::endl;
        saveIntermediateResults_fast("step04a_first_background2_fit_fast", true);
    }
}
#undef DEBUG

/* Main routine ------------------------------------------------------------ */
//#define DEBUG
double ROUT_Adjust_CTFFast(ProgCTFEstimateFromPSDFast &prm, CTFDescription1D &output_ctfmodel, bool standalone)
{
	DEBUG_OPEN_TEXTFILE(prm.fn_psd.removeLastExtension());
	global_prm = &prm;
	if (standalone || prm.show_optimization)
	   prm.show();
	prm.produceSideInfo();
	DEBUG_TEXTFILE(formatString("After producing side info: Avg=%f",prm.ctftomodel().computeAvg()));
	DEBUG_MODEL_TEXTFILE;
	// Build initial frequency mask
	prm.value_th = -1;
	prm.min_freq_psd = prm.min_freq;
	prm.max_freq_psd = prm.max_freq;

	// Set some global variables
	prm.adjust_params = &prm.adjust;
	prm.penalize = false;
	prm.max_gauss_freq = 0;
	prm.heavy_penalization = prm.f->computeMax() * YSIZE(*prm.f);
	prm.show_inf = 0;

	// Some variables needed by all steps
	int iter;
	double fitness;
	Matrix1D<double> steps;
	/************************************************************************/
	/* STEPs 1, 2, 3 and 4:  Find background which best fits the CTF        */
	/************************************************************************/
	prm.current_ctfmodel.enable_CTFnoise = true;
	prm.current_ctfmodel.enable_CTF = false;
	prm.evaluation_reduction = 4;

	// If initial parameters were not supplied for the gaussian curve,
	// estimate them from the CTF file
	prm.action = 0;
	if (prm.adjust(FIRST_SQRT_PARAMETER) == 0)
	{
		prm.estimate_background_sqrt_parameters_fast();
		prm.estimate_background_gauss_parameters_fast();
	}

	 // Optimize the current background
	prm.action = 1;
	prm.penalize = true;
	prm.current_penalty = prm.penalty;
	steps.resize(BACKGROUND_CTF_PARAMETERS);
	steps.initConstant(1);
	if (!prm.modelSimplification >= 3)
		steps(1) = steps(2) = steps(4) = 0;
	powellOptimizer(*prm.adjust_params, FIRST_SQRT_PARAMETER + 1,
					BACKGROUND_CTF_PARAMETERS, CTF_fitness_fast, global_prm, 0.01, fitness, iter,
					steps, prm.show_optimization);

	// Make sure that the model has physical meaning
	// (In some machines due to numerical imprecission this check is necessary
	// at the end)
	prm.current_ctfmodel.forcePhysicalMeaning();
	COPY_ctfmodel_TO_CURRENT_GUESS;

	if (prm.show_optimization)
	{
		std::cout << "Best background Fit:\n" << prm.current_ctfmodel << std::endl;
		prm.saveIntermediateResults_fast("step01d_best_background_fit_fast", true);
	}

	DEBUG_TEXTFILE(formatString("Step 4: CTF_fitness=%f",CTF_fitness_fast));
	DEBUG_MODEL_TEXTFILE;

	/************************************************************************/
	/* STEPs 5 and 6:  Find envelope which best fits the CTF                */
	/************************************************************************/

	prm.action = 2;
	prm.current_ctfmodel.enable_CTF = true;

	prm.current_ctfmodel.kV = prm.initial_ctfmodel.kV;
	prm.current_ctfmodel.Cs = prm.initial_ctfmodel.Cs;
	prm.current_ctfmodel.phase_shift = prm.initial_ctfmodel.phase_shift;
	prm.current_ctfmodel.VPP_radius = prm.initial_ctfmodel.VPP_radius;

	if (prm.initial_ctfmodel.Q0 != 0)
		prm.current_ctfmodel.Q0 = prm.initial_ctfmodel.Q0;
	prm.estimate_envelope_parameters_fast();

	if (prm.show_optimization)
	{
		std::cout << "Best envelope Fit:\n" << prm.current_ctfmodel << std::endl;
		prm.saveIntermediateResults_fast("step02b_best_penalized_envelope_fit_fast", true);
	}

	DEBUG_TEXTFILE(formatString("Step 6: espr=%f",prm.current_ctfmodel.espr));
	DEBUG_MODEL_TEXTFILE;

	/************************************************************************/
	/* STEP 7:  defocus and angular parameters	                            */
	/************************************************************************/
	prm.action = 3;
	if (!prm.noDefocusEstimate)
	   prm.estimate_defoci_fast();
	else
		prm.current_ctfmodel.Defocus = prm.initial_ctfmodel.Defocus;
	DEBUG_TEXTFILE(formatString("Step 7: Defocus=%f",prm.current_ctfmodel.Defocus));
	DEBUG_MODEL_TEXTFILE;
	/************************************************************************/
	/* STEPs 9, 10 and 11: all parameters included second Gaussian          */
	/************************************************************************/
	prm.action = 5;
	if (prm.modelSimplification < 2)
		prm.estimate_background_gauss_parameters2_fast();

	steps.resize(ALL_CTF_PARAMETERS);
	steps.initConstant(1);
	steps(1) = 0; // kV
	steps(3) = 0; // The spherical aberration (Cs) is not optimized
	steps(27) = 0; //VPP radius not optimized
	if (prm.noDefocusEstimate)
		steps(0)=0; // Defocus
	if (prm.initial_ctfmodel.Q0 != 0)
		steps(13) = 0; // Q0
	if (prm.modelSimplification >= 1)
		steps(8) = steps(9) = 0;
	if (prm.initial_ctfmodel.VPP_radius == 0)
		steps(26) = 0; //VPP phase shift

	powellOptimizer(*prm.adjust_params, 0 + 1, ALL_CTF_PARAMETERS, CTF_fitness_fast,
					global_prm, 0.01, fitness, iter, steps, prm.show_optimization);

	prm.current_ctfmodel.forcePhysicalMeaning();
	COPY_ctfmodel_TO_CURRENT_GUESS;

	if (prm.show_optimization)
	{
		std::cout << "Best fit with Gaussian2:\n" << prm.current_ctfmodel
		<< std::endl;
		prm.saveIntermediateResults_fast("step04b_best_fit_with_gaussian2_fast", true);
	}

	/************************************************************************/
	/* STEP 12: 2D estimation parameters          							*/
	/************************************************************************/
	prm.adjust_params->resize(ALL_CTF_PARAMETERS2D);
	auto *prm2D = new ProgCTFEstimateFromPSD(&prm);
	if (prm.noDefocusEstimate)
	{
		prm2D->current_ctfmodel.DeltafU=prm.initial_ctfmodel2D.DeltafU;
		prm2D->current_ctfmodel.DeltafV=prm.initial_ctfmodel2D.DeltafV;
		prm2D->current_ctfmodel.azimuthal_angle=prm.initial_ctfmodel2D.azimuthal_angle;
		prm2D->noDefocusEstimate=true;
	}

	steps.resize(ALL_CTF_PARAMETERS2D);
	steps.initConstant(1);
	steps(3) = 0; // kV
	steps(5) = 0; // The spherical aberration (Cs) is not optimized
	steps(37) = 0; //VPP radius not optimized
	if (prm.noDefocusEstimate)
		steps(0)=steps(1)=steps(2)=0; // Defocus and azimuthal angle
	if (prm2D->initial_ctfmodel.Q0 != 0)
	    steps(15) = 0; // Q0
	 if (prm2D->modelSimplification >= 3)
		steps(20) = steps(21) = steps(23) = 0;
	if (prm2D->modelSimplification >= 2)
		steps(24) = steps(25) = steps(26) = steps(27) = steps(28) = steps(29) = 0;
	if (prm2D->modelSimplification >= 1)
		steps(10) = steps(11) = 0;
	if(prm2D->initial_ctfmodel.VPP_radius == 0)
		steps(36) = 0;

	prm2D->adjust_params = prm.adjust_params;
	prm2D->adjust_params->initZeros();
	prm2D->assignParametersFromCTF(prm2D->current_ctfmodel,MATRIX1D_ARRAY(*prm2D->adjust_params),0, ALL_CTF_PARAMETERS2D, true);

    if (prm.refineAmplitudeContrast)
        steps(15) = 1; // Q0
    powellOptimizer(*prm2D->adjust_params, 0 + 1, ALL_CTF_PARAMETERS2D, CTF_fitness,
					 prm2D, 0.01, fitness, iter, steps, prm2D->show_optimization);

	prm2D->current_ctfmodel.forcePhysicalMeaning();

	//We adopt that always  DeltafU > DeltafV so if this is not the case we change the values and the angle
	if (!prm.noDefocusEstimate && prm2D->current_ctfmodel.DeltafV > prm2D->current_ctfmodel.DeltafU)
	{
		double temp;
		temp = prm2D->current_ctfmodel.DeltafU;
		prm2D->current_ctfmodel.DeltafU = prm2D->current_ctfmodel.DeltafV;
		prm2D->current_ctfmodel.DeltafV = temp;
		prm2D->current_ctfmodel.azimuthal_angle -= 90;
	}
	prm2D->assignParametersFromCTF(prm2D->current_ctfmodel,MATRIX1D_ARRAY(*prm2D->adjust_params),0, ALL_CTF_PARAMETERS2D, true);
	if (prm2D->show_optimization)
	{
		std::cout << "Best fit with 2D parameters:\n" << prm2D->current_ctfmodel << std::endl;
		prm2D->saveIntermediateResults("step05b_estimate_2D_parameters", true);
	}

	/************************************************************************/
	/* STEP 13: Produce output                                              */
	/************************************************************************/
	prm2D->action = 7;
	if (prm2D->fn_psd != "")
	{
		// Define mask between first and third zero
		prm2D->mask_between_zeroes.initZeros(prm2D->mask);
		Matrix1D<double> u(2), z1(2), z3(2);
#ifdef NEVERDEFINED
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(prm2D->mask_between_zeroes)
		{
			VECTOR_R2(u, DIRECT_MULTIDIM_ELEM(prm2D->x_digfreq, n), DIRECT_MULTIDIM_ELEM(prm2D->y_digfreq, n));
			u /= u.module();
			prm2D->current_ctfmodel.lookFor(3, u, z3, 0);
			if (DIRECT_MULTIDIM_ELEM(prm2D->w_contfreq, n) >= z3.module())
				continue;
			prm2D->current_ctfmodel.lookFor(1, u, z1, 0);
			if (z1.module() < DIRECT_MULTIDIM_ELEM(prm2D->w_contfreq, n))
				DIRECT_MULTIDIM_ELEM(prm2D->mask_between_zeroes, n) = 1;

		}
#endif
        XX(u)=1; YY(u)=0;
        prm2D->current_ctfmodel.lookFor(3, u, z3, 0);
        prm2D->current_ctfmodel.lookFor(1, u, z1, 0);
        double z1m = z1.module();
        double z3m = z3.module();
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(prm2D->mask_between_zeroes)
		{
			double wn=DIRECT_MULTIDIM_ELEM(prm2D->w_contfreq, n);
			if (wn<=z3m && wn>=z1m)
				DIRECT_MULTIDIM_ELEM(prm2D->mask_between_zeroes, n) = 1;
		}
		// Evaluate the correlation in this region
		CTF_fitness(prm2D->adjust_params->adaptForNumericalRecipes(), prm2D);
		// Save results
		FileName fn_rootCTFPARAM = prm2D->fn_psd.withoutExtension();

		FileName fn_rootMODEL = fn_rootCTFPARAM;
		size_t atPosition=fn_rootCTFPARAM.find('@');

		if (atPosition!=std::string::npos)
		{
			fn_rootMODEL=formatString("%03d@%s",textToInteger(fn_rootCTFPARAM.substr(0, atPosition)),
									  fn_rootCTFPARAM.substr(atPosition+1).c_str());
			fn_rootCTFPARAM=formatString("region%03d@%s",textToInteger(fn_rootCTFPARAM.substr(0, atPosition)),
										 fn_rootCTFPARAM.substr(atPosition+1).c_str());
		}
		else
			fn_rootCTFPARAM=(String)"fullMicrograph@"+fn_rootCTFPARAM;

		prm2D->saveIntermediateResults(fn_rootMODEL, false);
		prm2D->current_ctfmodel.Tm /= prm2D->downsampleFactor;
		prm2D->current_ctfmodel.azimuthal_angle = std::fmod(prm2D->current_ctfmodel.azimuthal_angle,360.);
		prm2D->current_ctfmodel.phase_shift = (prm2D->current_ctfmodel.phase_shift*180)/3.14;
		if(prm2D->current_ctfmodel.phase_shift<0.0)
			prm2D->current_ctfmodel.phase_shift = 0.0;

		prm2D->current_ctfmodel.write(fn_rootCTFPARAM + ".ctfparam_tmp");
		MetaDataVec MD;
		MD.read(fn_rootCTFPARAM + ".ctfparam_tmp");
		size_t id = MD.firstRowId();
		MD.setValue(MDL_CTF_X0, (double)output_ctfmodel.x0*prm2D->Tm, id);
		MD.setValue(MDL_CTF_XF, (double)output_ctfmodel.xF*prm2D->Tm, id);
		MD.setValue(MDL_CTF_Y0, (double)output_ctfmodel.y0*prm2D->Tm, id);
		MD.setValue(MDL_CTF_YF, (double)output_ctfmodel.yF*prm2D->Tm, id);
		MD.setValue(MDL_CTF_CRIT_FITTINGSCORE, fitness, id);
		MD.setValue(MDL_CTF_CRIT_FITTINGCORR13, prm2D->corr13, id);
        MD.setValue(MDL_CTF_CRIT_ICENESS, evaluateIceness(prm.ctftomodel(), prm.Tm), id);
		MD.setValue(MDL_CTF_DOWNSAMPLE_PERFORMED, prm2D->downsampleFactor, id);
		MD.write(fn_rootCTFPARAM + ".ctfparam",MD_APPEND);
		fn_rootCTFPARAM = fn_rootCTFPARAM + ".ctfparam_tmp";

		fn_rootCTFPARAM.deleteFile();
	}
	output_ctfmodel = prm2D->current_ctfmodel;
	return fitness;
}

void ProgCTFEstimateFromPSDFast::run()
{
	 	CTFDescription1D ctf1Dmodel;
	    ROUT_Adjust_CTFFast(*this, ctf1Dmodel);
}
