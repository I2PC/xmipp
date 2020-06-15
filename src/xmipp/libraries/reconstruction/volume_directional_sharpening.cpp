/***************************************************************************
 *
 * Authors:    Jose Luis Vilas,                     jlvilas@cnb.csic.es
 *             Erney Ramirez                        eramirez@cnb.csic.es
 * 			   Carlos Oscar Sorzano					coss@cnb.csic.es
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

#include "volume_directional_sharpening.h"
#include "resolution_directional.h"
//#define DEBUG
//#define DEBUG_MASK

void ProgDirSharpening::readParams()
{
        fnVol = getParam("--vol");
        fnMask = getParam("--mask");
        sampling = getDoubleParam("--sampling");
        res_step = getDoubleParam("--resStep");
        significance = getDoubleParam("--significance");
        lambda = getDoubleParam("-l");
        K= getDoubleParam("-k");
        Niter = getIntParam("-i");
        Nthread = getIntParam("-n");
        fnOut = getParam("-o");
        test = checkParam("--test");
        icosahedron = checkParam("--ico");
}

void ProgDirSharpening::defineParams()
{
        addUsageLine("This function performs local sharpening");
        addParamsLine("  --vol <vol_file=\"\">                  : Input volume");
        addParamsLine("  --mask <vol_file=\"\">                 : Binary mask");
        addParamsLine("  --sampling <s=1>                       : sampling");
        addParamsLine("  [--significance <significance=0.95>]   : The level of confidence for the hypothesis test.");
        addParamsLine("  [--resStep <res_step=0.5>]  		    : Resolution step (precision) in A");
        addParamsLine("  -o <output=\"Sharpening.vol\">         : Sharpened volume");
        addParamsLine("  [--test]: 								: Launch the test of the algorithm");
        addParamsLine("  [--ico]: 								: Use icosahedron as coverage of the projection sphere");
        addParamsLine("  [-l <lambda=1>]                        : Regularization param");
        addParamsLine("  [-k <K=0.025>]                         : K param");
        addParamsLine("  [-i <Niter=50>]                        : Number of iterations");
        addParamsLine("  [-n <Nthread=1>]                       : Number of threads");
}


void ProgDirSharpening::produceSideInfo()
{
        std::cout << "Starting..." << std::endl;
        Monogenic mono;
        MultidimArray<double> inputVol;
        if (test)
		{
        	std::cout << "Preparing test data ..." << std::endl;
        	size_t xdim = 300, ydim = 300, zdim = 300;
        	double wavelength = 10.0, mean = 0.0, std = 0.5;
        	int maskrad = 125;
        	inputVol = mono.createDataTest(xdim, ydim, zdim, wavelength, mean, 0.0);
//        	inputVol.setXmippOrigin();
        	mask() = mono.createMask(inputVol, maskrad);
        	mask().setXmippOrigin();
        	mono.applyMask(inputVol, mask());
        	mono.addNoise(inputVol, 0, std);
        	FileName fn;
        	Image<double> saveImg;
        	fn = formatString("inputVol.vol");
        	saveImg() = inputVol;
        	saveImg.write(fn);
        	sampling = 1;
		}
        else
        {
        	std::cout << "Reading data..." << std::endl;
        	Image<double> V;
			V.read(fnVol);
			V().setXmippOrigin();
			inputVol = V();
			mask.read(fnMask);
			mask().setXmippOrigin();
        }

    	FourierTransformer transformer;

    	VRiesz.resizeNoCopy(inputVol);
    	maxRes = 18;
    	minRes = 2*sampling;

    	//TODO: check if its possible to use only one transformer instead of transformer_inv and transformer
    	transformer_inv.setThreadsNumber(Nthread);

    	transformer.FourierTransform(inputVol, fftV);

    	// Frequency volume
    	iu = mono.fourierFreqs_3D(fftV, inputVol, freq_fourier_x, freq_fourier_y, freq_fourier_z);
    	Image<double> im;
    	im()=iu;
    	im.write("freq.mrc");

    	MultidimArray<int> &pMask=mask();

    	double radius_aux, radiuslimit;
    	MultidimArray<double> radMap;
    	N_smoothing = 7;
    	double nsmooth = (double) N_smoothing;
    	mono.proteinRadiusVolumeAndShellStatistics(pMask, radius_aux, NVoxelsOriginalMask, radMap);
    	mono.findCliffValue(radMap, inputVol, radius_aux, radiuslimit, pMask, nsmooth);

    	std::cout << "                " << std::endl;
    	std::cout << "Noise radius    " << radiuslimit << " px" << std::endl;
    	std::cout << "Particle radius " << radius_aux << " px" << std::endl;

    	Image<int> saveImg;
		FileName fn;
		fn = formatString("maskrefine.vol");
		saveImg() = pMask;
		saveImg.write(fn);

    	Rparticle = (int) radius_aux;

    	std::cout << "Particle radius " << Rparticle << " px" << std::endl;

//    	size_t xrows = angles.mdimx;
//    	resolutionMatrix.initConstant(xrows, NVoxelsOriginalMask, maxRes);

}


void ProgDirSharpening::simpleGeometryFaces(Matrix2D<double> &faces, Matrix2D<double> &limts)
{
	faces.initZeros(5,3);
	double t0, c45, s45, latc, lats;
	t0 = 36.86*PI/180;
	lats = sin(PI/2-t0);
	latc = cos(PI/2-t0);
	c45 = cos(PI/4);
	s45 = sin(PI/4);
	MAT_ELEM(faces, 0,0) = lats*c45;    MAT_ELEM(faces, 0,1) = lats*s45;    MAT_ELEM(faces, 0,2) = latc;
	MAT_ELEM(faces, 1,0) = -lats*c45;   MAT_ELEM(faces, 1,1) = lats*c45;    MAT_ELEM(faces, 1,2) = latc;
	MAT_ELEM(faces, 2,0) = -lats*c45;   MAT_ELEM(faces, 2,1) = -lats*c45;   MAT_ELEM(faces, 2,2) = latc;
	MAT_ELEM(faces, 3,0) = lats*c45;    MAT_ELEM(faces, 3,1) = -lats*c45;   MAT_ELEM(faces, 3,2) = latc;
	MAT_ELEM(faces, 4,0) = 0;    		MAT_ELEM(faces, 4,1) = 0;    		MAT_ELEM(faces, 4,2) = 1;

	limts.initZeros(5,4);
	MAT_ELEM(limts, 0,0) = 0.0;   MAT_ELEM(limts, 0,1) = 90.0;    MAT_ELEM(limts, 0,2) = 90-36.86;	MAT_ELEM(limts, 0,3) = 90.0;
	MAT_ELEM(limts, 1,0) = -90;   MAT_ELEM(limts, 1,1) = -180.0;  MAT_ELEM(limts, 1,2) = 90-36.86;	MAT_ELEM(limts, 1,3) = 90.0;
	MAT_ELEM(limts, 2,0) = 0.0;   MAT_ELEM(limts, 2,1) = 90.0;    MAT_ELEM(limts, 2,2) = -90.0;   	MAT_ELEM(limts, 2,3) = 90+36.86;
	MAT_ELEM(limts, 3,0) = -90;   MAT_ELEM(limts, 3,1) = -180.0;  MAT_ELEM(limts, 3,2) = -90.0;   	MAT_ELEM(limts, 3,3) = 90+36.86;
	MAT_ELEM(limts, 4,0) = -90.0; MAT_ELEM(limts, 4,1) = 90.0;    MAT_ELEM(limts, 4,2) = 0.0;	    MAT_ELEM(limts, 4,3) = 90-36.86;

}


void ProgDirSharpening::icosahedronVertex(Matrix2D<double> &vertex)
{
	std::cout << "Defining Icosahedron vertex..." << std::endl;

	//The icosahedron vertex are located in (0, +-1, +-phi), (+-1, +-phi, 0), (+-phi, 0, +-1) with phi = (1+sqrt(5))/2
	double phi =  (1+sqrt(5))/2;

	vertex.initZeros(12,3);

	MAT_ELEM(vertex, 0,0) = 0;    		MAT_ELEM(vertex, 0,1) = 1;    		MAT_ELEM(vertex, 0,2) = phi;
	MAT_ELEM(vertex, 1,0) = 0;    		MAT_ELEM(vertex, 1,1) = 1;    		MAT_ELEM(vertex, 1,2) = -phi;
	MAT_ELEM(vertex, 2,0) = 0;    		MAT_ELEM(vertex, 2,1) = -1;    		MAT_ELEM(vertex, 2,2) = phi;
	MAT_ELEM(vertex, 3,0) = 0;    		MAT_ELEM(vertex, 3,1) = -1;    		MAT_ELEM(vertex, 3,2) = -phi;
	MAT_ELEM(vertex, 4,0) = 1;    		MAT_ELEM(vertex, 4,1) = phi;    		MAT_ELEM(vertex, 4,2) = 0;
	MAT_ELEM(vertex, 5,0) = 1;    		MAT_ELEM(vertex, 5,1) = -phi;    		MAT_ELEM(vertex, 5,2) = 0;
	MAT_ELEM(vertex, 6,0) = -1;    		MAT_ELEM(vertex, 6,1) = phi;    		MAT_ELEM(vertex, 6,2) = 0;
	MAT_ELEM(vertex, 7,0) = -1;    		MAT_ELEM(vertex, 7,1) = -phi;    		MAT_ELEM(vertex, 7,2) = 0;
	MAT_ELEM(vertex, 8,0) = phi;    		MAT_ELEM(vertex, 8,1) = 0;    		MAT_ELEM(vertex, 8,2) = 1;
	MAT_ELEM(vertex, 9,0) = phi;    		MAT_ELEM(vertex, 9,1) = 0;    		MAT_ELEM(vertex, 9,2) = -1;
	MAT_ELEM(vertex, 10,0) = -phi;    		MAT_ELEM(vertex, 10,1) = 0;    		MAT_ELEM(vertex, 10,2) = 1;
	MAT_ELEM(vertex, 11,0) = -phi;    		MAT_ELEM(vertex, 11,1) = 0;    		MAT_ELEM(vertex, 11,2) = -1;

	vertex = vertex*(1/sqrt(1+phi*phi));
}

void ProgDirSharpening::icosahedronFaces(Matrix2D<int> &faces, Matrix2D<double> &vertex)
{
	std::cout << " Defining the faces of the icosahedron ..." << std::endl;
	//Each face is defined by three vertex

	//An icosahedron has 20 faces.
	faces.initZeros(20,3);

	int v1, v2, v3, v1_bis, v2_bis, v3_bis;
	double x1, x2, x3, y1, y2, y3, z1, z2, z3, x1_bis, x2_bis, x3_bis, y1_bis, y2_bis, y3_bis, z1_bis, z2_bis, z3_bis;

	int xdim = MAT_YSIZE(vertex); //Number of vertex
	int counter = 0;

	for (int i = 0; i<(xdim-2); ++i)
	{
	    for (int j = (i+1); j<(xdim-1); ++j)
	    {
	        for (int k = (j+1); k<(xdim); ++k)
	        {
	            double dotprodutij, dotprodutjk, dotprodutik;
	            dotprodutij = (MAT_ELEM(vertex, i,0)*MAT_ELEM(vertex, j,0) + \
	            		MAT_ELEM(vertex, i,1)*MAT_ELEM(vertex, j,1) +\
						MAT_ELEM(vertex, i,2)*MAT_ELEM(vertex, j,2));///norm_vertex;

	            dotprodutjk = (MAT_ELEM(vertex, k,0)*MAT_ELEM(vertex, j,0) + \
	            	            		MAT_ELEM(vertex, k,1)*MAT_ELEM(vertex, j,1) + \
	            						MAT_ELEM(vertex, k,2)*MAT_ELEM(vertex, j,2));///norm_vertex;

	            dotprodutik = (MAT_ELEM(vertex, i,0)*MAT_ELEM(vertex, k,0) + \
	            	            		MAT_ELEM(vertex, i,1)*MAT_ELEM(vertex, k,1) + \
	            						MAT_ELEM(vertex, i,2)*MAT_ELEM(vertex, k,2));///norm_vertex;

	            // the number 65 comes because is greater than 60 that is the exact angle between two icosahedron vertex
	            if ((acos(dotprodutij)< 65*PI/180) && (acos(dotprodutjk)< 65*PI/180) && (acos(dotprodutik)< 65*PI/180) )
	            {
	            	MAT_ELEM(faces, counter, 0) = i;
	            	MAT_ELEM(faces, counter, 1) = j;
	            	MAT_ELEM(faces, counter, 2) = k;

	            	z1 = MAT_ELEM(vertex,i, 2);
					z2 = MAT_ELEM(vertex,j, 2);
					z3 = MAT_ELEM(vertex,k, 2);

					if ( ((z1+z2+z3) < 0) )
					{
						MAT_ELEM(faces,counter, 0) = -1; MAT_ELEM(faces,counter, 1) = -1; MAT_ELEM(faces,counter, 2) = -1;
					}

	            	++counter;
	            }

	        }
	    }
	}
	//TODO: Check if both loops can be written together

	//However, only the half of the sphere is used, so 10 faces must be considered
	for (int f1 = 0; f1<(MAT_YSIZE(faces)-1); ++f1)
	{
		if (MAT_ELEM(faces,f1, 0) < 0)
			continue;

		v1 = MAT_ELEM(faces,f1, 0); v2 = MAT_ELEM(faces,f1, 1); v3 = MAT_ELEM(faces,f1, 2);

		for (int f2 = f1+1; f2<MAT_YSIZE(faces); ++f2)
		{
			if (MAT_ELEM(faces,f2, 0) < 0)
				continue;

			v1_bis = MAT_ELEM(faces,f2, 0); v2_bis = MAT_ELEM(faces,f2, 1); v3_bis = MAT_ELEM(faces,f2, 2);

			x1 = MAT_ELEM(vertex,v1, 0); y1 = MAT_ELEM(vertex,v1, 1); z1 = MAT_ELEM(vertex,v1, 2);
			x2 = MAT_ELEM(vertex,v2, 0); y2 = MAT_ELEM(vertex,v2, 1); z2 = MAT_ELEM(vertex,v2, 2);
			x3 = MAT_ELEM(vertex,v3, 0); y3 = MAT_ELEM(vertex,v3, 1); z3 = MAT_ELEM(vertex,v3, 2);

			x1_bis = MAT_ELEM(vertex,v1_bis, 0); y1_bis = MAT_ELEM(vertex,v1_bis, 1); z1_bis = MAT_ELEM(vertex,v1_bis, 2);
			x2_bis = MAT_ELEM(vertex,v2_bis, 0); y2_bis = MAT_ELEM(vertex,v2_bis, 1); z2_bis = MAT_ELEM(vertex,v2_bis, 2);
			x3_bis = MAT_ELEM(vertex,v3_bis, 0); y3_bis = MAT_ELEM(vertex,v3_bis, 1); z3_bis = MAT_ELEM(vertex,v3_bis, 2);

			double x_tot = x1 + x2 + x3;
			double y_tot = y1 + y2 + y3;
			double z_tot = z1 + z2 + z3;
			double norm_tot, norm_tot_bis;

			norm_tot = sqrt(x_tot*x_tot + y_tot*y_tot + z_tot*z_tot);

			double x_tot_bis = x1_bis + x2_bis + x3_bis;
			double y_tot_bis = y1_bis + y2_bis + y3_bis;
			double z_tot_bis = z1_bis + z2_bis + z3_bis;

			norm_tot_bis = sqrt(x_tot_bis*x_tot_bis + y_tot_bis*y_tot_bis + z_tot_bis*z_tot_bis);

			double dotproduct;
			dotproduct = (x_tot*x_tot_bis + y_tot*y_tot_bis + z_tot*z_tot_bis)/(norm_tot*norm_tot_bis);

			if ( (fabs(dotproduct)>0.9 ) )
			{
				MAT_ELEM(faces,f2, 0) = -1;
				MAT_ELEM(faces,f2, 1) = -1;
				MAT_ELEM(faces,f2, 2) = -1;
			}
		}
	}
}

void ProgDirSharpening::getFaceVectorIcosahedron(Matrix2D<int> &faces,
		Matrix2D<double> &vertex, Matrix2D<double> &facesVector)
{
	facesVector.initZeros(MAT_YSIZE(faces), 3);

	double x1, x2, x3, y1, y2, y3, z1, z2, z3;
	int v1, v2, v3;
	//Selecting the vertex number for each face
	for (size_t face_number = 0; face_number<MAT_YSIZE(faces); face_number++)
	{
		v1 = MAT_ELEM(faces, face_number, 0); v2 = MAT_ELEM(faces, face_number, 1); v3 = MAT_ELEM(faces,face_number, 2);

		//Coordinates of each vertex
		x1 = MAT_ELEM(vertex,v1, 0); y1 = MAT_ELEM(vertex,v1, 1); z1 = MAT_ELEM(vertex,v1, 2);
		x2 = MAT_ELEM(vertex,v2, 0); y2 = MAT_ELEM(vertex,v2, 1); z2 = MAT_ELEM(vertex,v2, 2);
		x3 = MAT_ELEM(vertex,v3, 0); y3 = MAT_ELEM(vertex,v3, 1); z3 = MAT_ELEM(vertex,v3, 2);

		//x1, y1, z1 are used instead of defining a new variable to calculate the norm
		x1 = x1 + x2 + x3;
		y1 = y1 + y2 + y3;
		z1 = z1 + z2 + z3;

		double norm_ = sqrt(x1*x1 + y1*y1 + z1*z1);
		x1 /= norm_;
		y1 /= norm_;
		z1 /= norm_;

		MAT_ELEM(facesVector, face_number, 0) = x1;
		MAT_ELEM(facesVector, face_number, 1) = y1;
		MAT_ELEM(facesVector, face_number, 2) = z1;
	}

}

void ProgDirSharpening::getFaceVectorSimple(Matrix2D<double> &facesVector, Matrix2D<double> &faces)
{
	facesVector.initZeros(MAT_YSIZE(faces),3);
	for (size_t face_number = 0; face_number<MAT_YSIZE(faces); face_number++)
	{
		if (face_number<4)
		{
			double angleSemiCap;
			//The z cap semiangle is 36.86 degrees. The XY caps have an angle os 90-36.86
			//It implies a semicap of (90-36.86)/2, and measured from z axis (90-36.86)/2 ) + 36.86)
			angleSemiCap = ( ( (90-36.86)/2 ) + 36.86)*PI/180; //Measured from Z axis
			double angleRot;
			angleRot = (face_number *PI/2.0 + PI/4.0);
			MAT_ELEM(facesVector, face_number, 0) = sin(angleSemiCap)*cos(angleRot);
			MAT_ELEM(facesVector, face_number, 1) = sin(angleSemiCap)*sin(angleRot);
			MAT_ELEM(facesVector, face_number, 2) = cos(angleSemiCap);
		}
		else{
			MAT_ELEM(facesVector, face_number, 0) = 0.0;
			MAT_ELEM(facesVector, face_number, 1) = 0.0;
			MAT_ELEM(facesVector, face_number, 2) = 1.0;
		}
	}
}


void ProgDirSharpening::defineIcosahedronCone(int face_number, double &x1, double &y1, double &z1,
		MultidimArray< std::complex<double> > &myfftV, MultidimArray<double> &conefilter, double coneAngle)
{
//	std::cout << x1 << " " << y1 << " " << z1 << std::endl;

//	MultidimArray<double> conetest;
//	conetest.resizeNoCopy(myfftV);

	conefilter.initZeros(myfftV);

	double uz, uy, ux, cosconeAngle;
	cosconeAngle = cos(coneAngle);
	long n = 0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz *= z1;

		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uy *= y1;

			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				double iun=DIRECT_MULTIDIM_ELEM(iu,n);

				ux = VEC_ELEM(freq_fourier_x, j);
				ux *= x1;

				iun *= (ux + uy + uz);
				double cosine = fabs(iun);
//				DIRECT_MULTIDIM_ELEM(conefilter, n) = 1;
				if (cosine>=cosconeAngle)
				{
					if (DIRECT_MULTIDIM_ELEM(iu,n) <1.99)
					{
						n++;
						continue;
					}
					DIRECT_MULTIDIM_ELEM(conefilter, n) = 1;
//					DIRECT_MULTIDIM_ELEM(conetest, n) = 0;
				}
				++n;
			}
		}
	}
//
	Image<double> icosahedronMasked;
	icosahedronMasked = conefilter;
	FileName fnmasked;
	fnmasked = formatString("maskConeFourier_%i.mrc",face_number);
//	int m1sizeX = YSIZE(myfftV);
//	int m1sizeY = YSIZE(myfftV);
//	int m1sizeZ = YSIZE(myfftV);
//	MultidimArray<double> fullMap;
//	createFullFourier(conefilter, fnmasked, m1sizeX, m1sizeY, m1sizeZ, fullMap);
	icosahedronMasked.write(fnmasked);
}

void ProgDirSharpening::defineSimpleCaps(MultidimArray<int> &coneMask, Matrix2D<double> &limits,
		MultidimArray< std::complex<double> > &myfftV)
{
	coneMask.initZeros(myfftV);

	double uz, uy, ux, cosconeAngle;
	for (int face_number=0; face_number<5; face_number++)
	{
		double c0, c1, c2, c3, c4;
		c0 = tan(MAT_ELEM(limits, face_number, 0)*PI/180);
		c1 = tan(MAT_ELEM(limits, face_number, 1)*PI/180);
		c2 = cos(MAT_ELEM(limits, face_number, 2)*PI/180);
		c3 = cos(MAT_ELEM(limits, face_number, 3)*PI/180);
		long n = 0;
		if (face_number<4)
		{
			for(size_t k=0; k<ZSIZE(myfftV); ++k)
			{
				uz = VEC_ELEM(freq_fourier_z, k);

				for(size_t i=0; i<YSIZE(myfftV); ++i)
				{
					uy = VEC_ELEM(freq_fourier_y, i);

					for(size_t j=0; j<XSIZE(myfftV); ++j)
					{
						double iun=DIRECT_MULTIDIM_ELEM(iu,n);
						ux = VEC_ELEM(freq_fourier_x, j);

						double ctilt = (uz*iun);
						double trot;
						trot = (uy/ux);

						if ( (ctilt<=c2) && (ctilt>=c3) && (trot>=c0) && (trot<=c1) )
						{
							DIRECT_MULTIDIM_ELEM(coneMask, n) = face_number;
						}
						++n;
					}
				}
			}
		}
		else
		{

			for(size_t k=0; k<ZSIZE(myfftV); ++k)
			{
				uz = VEC_ELEM(freq_fourier_z, k);

				for(size_t i=0; i<YSIZE(myfftV); ++i)
				{
					uy = VEC_ELEM(freq_fourier_y, i);

					for(size_t j=0; j<XSIZE(myfftV); ++j)
					{
						double iun=DIRECT_MULTIDIM_ELEM(iu,n);
						ux = VEC_ELEM(freq_fourier_x, j);

						double ctilt = fabs(uz*iun);


						if (ctilt>=c3)
						{
							DIRECT_MULTIDIM_ELEM(coneMask, n) = face_number;
						}
						++n;
					}
				}
			}
		}
	}
	FileName fnmasked;
	fnmasked = "maskCone.mrc";
	int m1sizeX = 240;
	int m1sizeY = 240;
	int m1sizeZ = 240;

	MultidimArray<double> fullMap;
//	createFullFourier(coneMask, fnmasked, m1sizeX, m1sizeY, m1sizeZ, fullMap);

	Image<int> icosahedronMasked;
	icosahedronMasked = coneMask;
	icosahedronMasked.write(fnmasked);
}


void ProgDirSharpening::defineComplexCaps(Matrix2D<double> &facesVector,
		MultidimArray< std::complex<double> > &myfftV, MultidimArray<int> &coneMask)
{
	size_t xdim, ydim, zdim, ndim;
	myfftV.getDimensions(xdim, ydim, zdim, ndim);
	coneMask.resizeNoCopy(zdim, ydim, xdim);
	coneMask.initConstant(-1);
	double nyquist = 2.0; //Nyquist=1/0.5in dig units
	double uz, uy, ux, dotproduct, lastdotprod;
	long n = 0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);

		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);

			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				lastdotprod = 0;
				if (DIRECT_MULTIDIM_ELEM(iu,n)<nyquist)
				{
					n++;
					continue;
				}
				else
				{
					for (int face_number=0; face_number<MAT_YSIZE(facesVector); face_number++)
					{
						dotproduct = MAT_ELEM(facesVector, face_number, 0) * ux +
								MAT_ELEM(facesVector, face_number, 1) * uy +
								MAT_ELEM(facesVector, face_number, 2) * uz;
						dotproduct *= DIRECT_MULTIDIM_ELEM(iu,n);
						dotproduct = fabs(dotproduct);

						if (lastdotprod <= dotproduct)
						{
							DIRECT_MULTIDIM_ELEM(coneMask, n) = face_number;
							lastdotprod = dotproduct;
						}
					}
					n++;
				}
			}
		}
	}

	FileName fnmasked;
	fnmasked = "coneMask.mrc";
	int m1sizeX = 240;
	int m1sizeY = 240;
	int m1sizeZ = 240;
////
//	createFullFourier(conefilter, fnmasked, m1sizeX, m1sizeY, m1sizeZ);

	Image<int> icosahedronMasked;
	icosahedronMasked = coneMask;
	icosahedronMasked.write(fnmasked);
}

void ProgDirSharpening::createFullFourier(MultidimArray<double> &fourierHalf, FileName &fnMap,
		int m1sizeX, int m1sizeY, int m1sizeZ, MultidimArray<double> &fullMap)
{
//	MultidimArray<double> fullMap;
	getCompleteFourier(fourierHalf, fullMap, m1sizeX, m1sizeY, m1sizeZ);
	CenterFFT(fullMap, true);
	Image<double> saveImg;
	saveImg() = fullMap;
    saveImg.write(fnMap);
}


void ProgDirSharpening::getCompleteFourier(MultidimArray<double> &V, MultidimArray<double> &newV,
		int m1sizeX, int m1sizeY, int m1sizeZ)
    {
	newV.resizeNoCopy(m1sizeX, m1sizeY, m1sizeZ);
	int ndim=3;
	if (m1sizeX==1)
	{
		ndim=2;
		if (m1sizeY==1)
			ndim=1;
	}
	double *ptrSource=NULL;
	double *ptrDest=NULL;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(newV)
	{
		ptrDest=(double*)&DIRECT_A3D_ELEM(newV,k,i,j);
		if (j<XSIZE(V))
		{
			ptrSource=(double*)&DIRECT_A3D_ELEM(V,k,i,j);
			*ptrDest=*ptrSource;
//				*(ptrDest+1)=*(ptrSource+1);
		}
		else
		{
			ptrSource=(double*)&DIRECT_A3D_ELEM(V,
												(m1sizeZ-k)%m1sizeZ,
												(m1sizeY-i)%m1sizeY,
												m1sizeX-j);
			*ptrDest=*ptrSource;
//				*(ptrDest+1)=-(*(ptrSource+1));
		}
	}
    }

void ProgDirSharpening::directionalNoiseEstimation(double &x_dir, double &y_dir, double &z_dir,
		MultidimArray<double> &amplitudeMS, MultidimArray<int> &mask, double &cone_angle,
		int &particleRadius, double &NS, double &NN, double &sumS, double &sumS2, double &sumN2, double &sumN,
		double &thresholdNoise)
{
	double uz, uy, ux;
	int n=0;

	int z_size = ZSIZE(amplitudeMS);
	int x_size = XSIZE(amplitudeMS);
	int y_size = YSIZE(amplitudeMS);

	double amplitudeValue;
	std::vector<float> noiseValues;
	NS = 0;
	NN = 0;
//	FileName fnmasked;
//	Image<double> icosahedrod;
//	icosahedrod() = amplitudeMS;
//	fnmasked = formatString("ampl.mrc");
//	icosahedrod.write(fnmasked);
//	exit(0);
	double cosineConAng;
	cosineConAng = cos(cone_angle);

	for(int k=0; k<z_size; ++k)
	{
		for(int i=0; i<y_size; ++i)
		{
			for(int j=0; j<x_size; ++j)
			{
				if (DIRECT_MULTIDIM_ELEM(mask, n)>=1)
				{
					amplitudeValue=DIRECT_MULTIDIM_ELEM(amplitudeMS, n);
//					std::cout << amplitudeValue << std::endl;
					sumS  += amplitudeValue;
					sumS2 += amplitudeValue*amplitudeValue;
					++NS;
				}
				else
				{
					if (DIRECT_MULTIDIM_ELEM(mask, n)==0)
					{
						uz = (k - z_size*0.5);
						ux = (j - x_size*0.5);
						uy = (i - y_size*0.5);

						double rad = sqrt(ux*ux + uy*uy + uz*uz);
						double iun = 1/rad;

						//BE CAREFULL with the order
						double dotproduct = (uy*y_dir + ux*x_dir + uz*z_dir)*iun;

						double cosine = fabs(dotproduct);
						if ( (cosine>cosineConAng) && (rad>particleRadius))
						{
//							std::cout << "rad " << rad << std::endl;
	//						DIRECT_MULTIDIM_ELEM(coneVol, n) = 1;
							amplitudeValue = DIRECT_MULTIDIM_ELEM(amplitudeMS, n);
							noiseValues.push_back((float) amplitudeValue);
							sumN  += amplitudeValue;
							sumN2 += amplitudeValue*amplitudeValue;
							++NN;
						}
					}
				}
				++n;
			}
		}
	}
//	std::cout << "after loop directionalNoiseEstimation" << std::endl;
//	std::cout << "NS " << NS << "  NN " << NN << " sumS " << sumS << std::endl;
	std::sort(noiseValues.begin(),noiseValues.end());
	thresholdNoise = (double) noiseValues[size_t(noiseValues.size()*significance)];

	//std::cout << "thr="<< thresholdNoise << " " << meanN+criticalZ*sqrt(sigma2N) << " " << NN << std::endl;
	noiseValues.clear();

}



void sortArr(double arr[], int n, std::vector<std::pair<double, int> > &vp)
{
    // Inserting element in pair vector
    // to keep track of previous indexes
    for (int i = 0; i < n; ++i) {
        vp.push_back(std::make_pair(arr[i], i));
    }

    // Sorting pair vector
    sort(vp.begin(), vp.end());

    // Displaying sorted element
    // with previous indexes
    // corresponding to each element
    std::cout << "Element\t"
         << "index" << std::endl;
    for (int i = 0; i < vp.size(); i++) {
    	std::cout << vp[i].first << "\t"
             << vp[i].second << std::endl;
    }
}


void ProgDirSharpening::directionalResolutionStep(int face_number,
		const MultidimArray< std::complex<double> > &conefilter,
		MultidimArray<int> &mask, MultidimArray<double> &localResolutionMap,
		double &cone_angle, double &x1, double &y1, double &z1)
{
	std::cout << "Computing local-directional resolution" << std::endl;

	//Setting parameters
	double cut_value = 0.025; //percentage of voxels to stop the frequency analysis

	bool continueIter, breakIter;
	bool doNextIteration = true;
	double freq, freqL, freqH, counter, resolution_2, resolution, step = res_step;
	double last_resolution = 0;
	int fourier_idx, last_fourier_idx = -1, iter = 0, fourier_idx_2, v1, v2, v3;
	std::vector<double> list;

	FileName fnDebug = "Signal";

	MultidimArray<double> amplitudeMS;
	localResolutionMap.initZeros(mask);
	MultidimArray<double> &plocalResolutionMap = localResolutionMap;
	MultidimArray<int> mask_aux = mask;
	MultidimArray<int> &pMask = mask_aux;

	ProgResDir resolutionSweep;

	int aux_idx, volsize;

	volsize = XSIZE(mask);

	DIGFREQ2FFT_IDX(sampling/18.0, volsize, aux_idx);

	fourier_idx = aux_idx;

	std::cout << "fourier_idx = " << fourier_idx << std::endl;

	//Calculating the average of amplitudes
	Monogenic mono;

	amplitudeMS.resizeNoCopy(mask);
	mono.monogenicAmplitude_3D_Fourier(fftV, iu, amplitudeMS, Nthread);

	double AvgNoise;
	double max_meanS = -1e38;
	AvgNoise = mono.averageInMultidimArray(amplitudeMS, pMask);
	double criticalZ=icdf_gauss(significance);
	do
	{
		continueIter = false;
		breakIter = false;

		mono.resolution2evalDir(fourier_idx, step, sampling, volsize,
						resolution, last_resolution, last_fourier_idx,
						freq, freqL, freqH,
						continueIter, breakIter, doNextIteration);

		if (breakIter)
			break;

		if (continueIter)
			continue;

		list.push_back(resolution);

		if (iter<2)
			resolution_2 = list[0];
		else
			resolution_2 = list[iter - 2];

		std::cout << "res = " << resolution << " freq " << sampling/freq << "  freqH " << sampling/freqH << "  freqL " << sampling/freqL << std::endl;

		mono.amplitudeMonoSigDir3D_LPF(conefilter,
		transformer_inv, fftVRiesz, fftVRiesz_aux, VRiesz, freq, freqH, freqL, iu,
		freq_fourier_x, freq_fourier_y, freq_fourier_z, amplitudeMS,
		 iter, face_number, "Signal", N_smoothing);

		double thresholdNoise, sumS=0, sumS2=0, sumN=0, sumN2=0, NN = 0, NS = 0;
		directionalNoiseEstimation(x1, y1, z1, amplitudeMS, pMask, cone_angle,
				Rparticle, NS, NN, sumS, sumS2, sumN2, sumN, thresholdNoise);

		if (NS == 0)
		{
			std::cout << "There are no points to compute inside the mask" << std::endl;
			std::cout << "If the number of computed frequencies is low, perhaps the provided"
					"mask is not enough tight to the volume, in that case please try another mask" << std::endl;
			break;
		}

		if ( (NS/(double) NVoxelsOriginalMask) < cut_value ) //when the 2.5% is reached then the iterative process stops
		{
			std::cout << "Search of resolutions stopped due to mask has been completed" << std::endl;
			doNextIteration =false;
		}

		double meanS=sumS/NS;
		double sigma2S=sumS2/NS-meanS*meanS;
		double meanN=sumN/NN;
		double sigma2N=sumN2/NN-meanN*meanN;

		double thresholdNoiseGauss = meanN+criticalZ*sqrt(sigma2N);
		double z=(meanS-meanN)/sqrt(sigma2S/NS+sigma2N/NN);
		if (meanS>max_meanS)
			max_meanS = meanS;

		if (meanS<0.001*AvgNoise)//0001*max_meanS)
		{
			std::cout << "Search of resolutions stopped due to too low signal" << std::endl;
			std::cout << "\n" << std::endl;
			break;
		}

		std::cout << "It = " << iter << ",   Res= " << resolution << ",   Sig = " << meanS << ",  Thr = " << thresholdNoise << std::endl;
		std::cout << "thresholdNoiseGauss= " << thresholdNoiseGauss << ",   z= " << z << ",   criticalZ = " << criticalZ << std::endl;
		std::cout << "        " << std::endl;

		mono.setLocalResolutionMap(amplitudeMS, pMask, plocalResolutionMap,
				thresholdNoise, resolution, resolution_2);

		if (doNextIteration)
			if (resolution <= (minRes-0.001))
				doNextIteration = false;

		++iter;
		last_resolution = resolution;
	}while(doNextIteration);

	FileName fn;
	Image<double> saveImg;
	fn = formatString("dirMap_%i.vol", face_number);
	saveImg() = plocalResolutionMap;
	saveImg.write(fn);

}


void ProgDirSharpening::bandPassDirectionalFilterFunction(int face_number, MultidimArray<int> &maskCone, MultidimArray< std::complex<double> > &myfftV,
		MultidimArray<double> &Vorig, MultidimArray<double> &iu, FourierTransformer &transformer_inv,
        double w, double wL, MultidimArray<double> &filteredVol, int count, double &coneAngle)
{


	MultidimArray< std::complex<double> > fftVfilter;
	fftVfilter.initZeros(myfftV);

	double delta = wL-w;
	double w_inf = w-delta;
	// Filter the input volume and add it to amplitude
	long n=0;
	double ideltal=PI/(delta);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(myfftV)
	{
		if (DIRECT_MULTIDIM_ELEM(maskCone, n) == face_number)
		{
			double un=DIRECT_MULTIDIM_ELEM(iu,n);
			if (un>=w && un<=wL)
			{
					DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVfilter, n) *= 0.5*(1+cos((un-w)*ideltal));//H;
			} else{
				if (un<=w && un>=w_inf)
				{
					DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVfilter, n) *= 0.5*(1+cos((un-w)*ideltal));//H;
				}
			}
		}
	}

	filteredVol.resizeNoCopy(Vorig);

	transformer_inv.inverseFourierTransform(fftVfilter, filteredVol);
}

void ProgDirSharpening::localDirectionalfiltering(size_t &Nfaces,
		MultidimArray< std::complex<double> > &myfftV, MultidimArray<int> &coneMask,
        MultidimArray<double> &bandfilteredVol, MultidimArray<double> &Vorig,
        double &minRes, double &maxRes, double &step, double &coneAngle)
{
        MultidimArray<double> filteredVol, auxlocalfilteredVol, lastweight, weight;
        bandfilteredVol.initZeros(Vorig);
        auxlocalfilteredVol.initZeros(Vorig);
        weight.initZeros(Vorig);
        lastweight.initZeros(Vorig);
        Monogenic mono;

        Image<double> resVol;
        MultidimArray<double> &presVol=resVol();

        for (size_t face_number = 0; face_number<Nfaces; ++face_number)
		{
            double freq, lastResolution=1e38;
            int idx, lastidx = -1;
			FileName fn;
			fn = formatString("dirMap_%i.vol", face_number);
			resVol.read(fn);

			std::cout << "face_number = " << face_number << std::endl;
			for (double res = minRes; res<maxRes; res+=step)
			{

				std::cout << "resolution = " << res << std::endl;
				freq = sampling/res;
				DIGFREQ2FFT_IDX(freq, ZSIZE(myfftV), idx);

				if (idx == lastidx)
					continue;

				double wL = sampling/(res - step);

				//TODO: Check performance in the mask
				bandPassDirectionalFilterFunction(face_number, coneMask, myfftV, Vorig, iu,
										transformer_inv, freq, wL, filteredVol, idx, coneAngle);

				double nyquist = 2*sampling;
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(filteredVol)
				{
				   if (DIRECT_MULTIDIM_ELEM(presVol, n) < nyquist){
					   DIRECT_MULTIDIM_ELEM(filteredVol, n)=0;
					}
				   else{
					   double res_map = DIRECT_MULTIDIM_ELEM(presVol, n);//+1e-38;
					   DIRECT_MULTIDIM_ELEM(weight, n) = (exp(-K*(res-res_map)*(res-res_map)));
					   DIRECT_MULTIDIM_ELEM(filteredVol, n) *= DIRECT_MULTIDIM_ELEM(weight, n);
					}
				}
				//TODO: lastweight and localfilteredVol can be implemented inside the loop
				auxlocalfilteredVol += filteredVol;
				lastweight += weight;
				lastResolution = res;
				lastidx = idx;
			}

			bandfilteredVol += auxlocalfilteredVol;

//			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(localfilteredVol)
//			{
//				if (DIRECT_MULTIDIM_ELEM(lastweight, n)>0)
//					DIRECT_MULTIDIM_ELEM(localfilteredVol, n) /=DIRECT_MULTIDIM_ELEM(lastweight, n);
//			}

//			localfilteredVol += localfilteredVol;

//			FileName fl;
//			Image<double> saveImg;
//			fl = formatString("localFiltVolt_%i.vol", face_number);
//			saveImg() = bandfilteredVol;
//			saveImg.write(fl);

		}

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bandfilteredVol)
		{
			if (DIRECT_MULTIDIM_ELEM(lastweight, n)>0)
				DIRECT_MULTIDIM_ELEM(bandfilteredVol, n) /=DIRECT_MULTIDIM_ELEM(lastweight, n);
		}
}


void ProgDirSharpening::localdeblurStep(MultidimArray<double> &vol,
		MultidimArray<int> &coneMask, MultidimArray<int> &mask,
		size_t &Nfaces, double &coneAngle)
{
	std::cout << "-------------------------------" << std::endl;
	std::cout << "Starting directional sharpening" << std::endl;
	std::cout << "-------------------------------" << std::endl;

	//TODO Set number of processors in
	//transformer_inv and transformer

	MultidimArray<double> Vorig;
	Vorig = vol;
	transformer.setThreadsNumber(Nthread);
	transformer.FourierTransform(vol, fftV);

	//TODO: check if exist inf in next multidimarray
	//TODO: check if it is neccesary
	//Frequencies are redefined
	iu =1/iu;

	vol.clear();
	Monogenic mono;
	double desvOutside_Vorig, mean;
	mono.statisticsInBinaryMask(Vorig, mask, mean, desvOutside_Vorig);
	//std::cout << "desvOutside_Vorig = " << desvOutside_Vorig << std::endl;

	maxRes = 16;
	minRes = 2*sampling;


	MultidimArray<double> auxVol;
	MultidimArray<double> operatedfiltered, Vk, filteredVol;
	double lastnorm = 0, lastporc = 1;
	double freq;
	double step = 0.2;
	int idx, bool1=1, bool2=1;
	int lastidx = -1;

	maxRes = maxRes + 2;

	//std::cout << "Resolutions between " << minRes << " and " << maxRes << std::endl;

	filteredVol = Vorig;

	MultidimArray<double> sharpenedMap;
	sharpenedMap.resizeNoCopy(Vorig);
	double normOrig=0;

	for (size_t i = 1; i<=Niter; ++i)
	{
		std::cout << "----------------Iteration " << i << "----------------" << std::endl;
		auxVol = filteredVol;
		transformer.FourierTransform(auxVol, fftV);

		localDirectionalfiltering(Nfaces,
				fftV, coneMask, operatedfiltered, Vorig, minRes, maxRes, step, coneAngle);

		FileName fn;
		FileName fs1;
		Image<double> saveImg1;
		fs1 = formatString("operFilt_%i.vol", i);
		saveImg1() = operatedfiltered;
		saveImg1.write(fs1);

		filteredVol = Vorig;

		filteredVol -= operatedfiltered;

		//calculate norm for Vorig
		if (i==1)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Vorig)
				normOrig +=(DIRECT_MULTIDIM_ELEM(Vorig,n)*DIRECT_MULTIDIM_ELEM(Vorig,n));

			normOrig = sqrt(normOrig);
			std::cout << "norma del original  " << normOrig << std::endl;
		}


		//calculate norm for operatedfiltered
		double norm=0;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(operatedfiltered)
			norm +=(DIRECT_MULTIDIM_ELEM(operatedfiltered,n)*DIRECT_MULTIDIM_ELEM(operatedfiltered,n));

		norm=sqrt(norm);
		std::cout << "norma del filtrado  " << norm << std::endl;

		double porc=lastnorm*100/norm;
		//std::cout << "norm " << norm << " percetage " << porc << std::endl;

		double subst=porc-lastporc;

		if ((subst<1) && (i>2))
			bool1=2;
			//std::cout << "-----iteration completed-----" << std::endl;


		lastnorm=norm;
		lastporc=porc;

		if (i==1 && lambda==1)
		{
			lambda=(normOrig/norm)/12;
			std::cout << "  lambda  " << lambda << std::endl;
		}

		////Second operator
		transformer.FourierTransform(filteredVol, fftV);
		localDirectionalfiltering(Nfaces,
						fftV, coneMask, filteredVol, Vorig, minRes, maxRes, step, coneAngle);

		if (i == 1)
				Vk = Vorig;
		else
				Vk = sharpenedMap;

		//sharpenedMap=Vk+lambda*(filteredVol);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sharpenedMap)
		{
			DIRECT_MULTIDIM_ELEM(sharpenedMap,n)=DIRECT_MULTIDIM_ELEM(Vk,n)+
								 lambda*DIRECT_MULTIDIM_ELEM(filteredVol,n);
								 //-0.01*DIRECT_MULTIDIM_ELEM(Vk,n)*SGN(DIRECT_MULTIDIM_ELEM(Vk,n));
			if (DIRECT_MULTIDIM_ELEM(sharpenedMap,n)<-4*desvOutside_Vorig)
				DIRECT_MULTIDIM_ELEM(sharpenedMap,n)=-4*desvOutside_Vorig;
		}
		//TODO: Erney commented the if condition (I uncommented). Has more sense the Erney way

//        		double desv_sharp=0;
//                computeAvgStdev_within_binary_mask(resVol, sharpenedMap, desv_sharp);
//                std::cout << "desv_sharp = " << desv_sharp << std::endl;

		filteredVol = sharpenedMap;

		if (bool1 == 2)
		{
			Image<double> filteredvolume;
			filteredvolume() = sharpenedMap;
			filteredvolume.write(fnOut);
			break;
		}

		FileName fs;
		Image<double> saveImg;
		fs = formatString("dirSharp_%i.vol", i);
		saveImg() = sharpenedMap;
		saveImg.write(fs);
	}



}

void cleanFaces(Matrix2D<int> &faces, Matrix2D<double> &vertex)
{
	int NewNumFaces = 0;

	for (size_t face_number = 0; face_number<MAT_YSIZE(faces); ++face_number)
	{
		if (MAT_ELEM(faces, face_number, 0) < 0)
			continue;
		NewNumFaces++;
	}
	Matrix2D<int> facesNew;
	facesNew.initZeros(NewNumFaces,3);

	NewNumFaces = 0;
	for (size_t face_number = 0; face_number<MAT_YSIZE(faces); ++face_number)
	{
		if (MAT_ELEM(faces, face_number, 0) < 0)
			continue;

		MAT_ELEM(facesNew, NewNumFaces, 0) = MAT_ELEM(faces, face_number, 0);
		MAT_ELEM(facesNew, NewNumFaces, 1) = MAT_ELEM(faces, face_number, 1);
		MAT_ELEM(facesNew, NewNumFaces, 2) = MAT_ELEM(faces, face_number, 2);
		++NewNumFaces;
	}
	faces = facesNew;

}


void ProgDirSharpening::run()
{
//	bool stopError = false;
//	if (test)
//	{
//		Monogenic Mono;
//		stopError = Mono.TestmonogenicAmplitude_3D_Fourier();
//		if (stopError == false)
//			exit(0);
//	}

	//Defining general information to be used
	produceSideInfo();

	std::cout << "Reading data..." << std::endl;
	//Defining the number of vertex and faces of the icosahedron
	Matrix2D<double> vertex, facesSimple, limtSimple, faceVector;
	Matrix2D<int> faces;
	double coneAngle = PI/6;
	MultidimArray< std::complex<double> > fftCone;
	MultidimArray<double> conefilter, localResolutionMap;
	MultidimArray<int> coneMask;
	Monogenic mono;
	size_t Nfaces;

	if (icosahedron == true)
	{
		std::cout << "Using Icosahedron geometry" << std::endl;
		icosahedronVertex(vertex);
		icosahedronFaces(faces, vertex);
		cleanFaces(faces, vertex);
		getFaceVectorIcosahedron(faces, vertex, faceVector);
		defineComplexCaps(faceVector, fftV, coneMask);
		Nfaces = MAT_YSIZE(faces);
	}
	else
	{
		std::cout << "Using Simple geometry" << std::endl;
		simpleGeometryFaces(facesSimple, limtSimple);
		defineSimpleCaps(coneMask, limtSimple, fftV);
		Nfaces = MAT_YSIZE(facesSimple);
		getFaceVectorSimple(faceVector, facesSimple);
		coneAngle = PI/4;
	}

//	std::cout << "Vectex " << vertex << std::endl;
	std::cout << "faceVector " << faceVector << std::endl;

	unsigned t0, t1;
	t0=clock();
	for (size_t face_number = 0; face_number<Nfaces; ++face_number)
	{
		double x1, y1, z1;

		x1 = MAT_ELEM(faceVector, face_number, 0);
		y1 = MAT_ELEM(faceVector, face_number, 1);
		z1 = MAT_ELEM(faceVector, face_number, 2);

		std::cout << x1 << " " << y1 << " " << z1 << std::endl;
		defineIcosahedronCone(face_number, x1, y1, z1, fftV, conefilter, coneAngle);
		//defineMask_test();

		fftCone = mono.applyMaskFourier(fftV, conefilter);
		//defineIcosahedronCone_test();

//		std::cout << "Computing local-directional resolution along face " << face_number << std::endl;

		directionalResolutionStep(face_number, fftCone, mask(), localResolutionMap, coneAngle, x1, y1, z1);
		//directionalResolutionStep_test();
	}
	t1 = clock();

	double time = (double(t1-t0)/CLOCKS_PER_SEC);
	std::cout << "%Execution Time: " << time << std::endl;
	Image<double> Vin;
	Vin.read(fnVol);
	Vin().setXmippOrigin();
	MultidimArray<double> Vorig = Vin();
	localdeblurStep(Vorig, coneMask, mask(), Nfaces, coneAngle);
	//TODO: Think a test...
	//	testlocaldeblur()
}



