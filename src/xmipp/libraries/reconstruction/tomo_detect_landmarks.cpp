/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez			  fp.deisidro@cnb.csic.es
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

#include "tomo_detect_landmarks.h"
#include <chrono>



// --------------------------- INFO functions ----------------------------

void ProgTomoDetectLandmarks::readParams()
{
	fnVol = getParam("-i");
	fnOut = getParam("-o");

	samplingRate = getDoubleParam("--samplingRate");
	fiducialSize = getDoubleParam("--fiducialSize");
}



void ProgTomoDetectLandmarks::defineParams()
{
	addUsageLine("This function determines the location of high contrast features in a volume.");
	addParamsLine("  -i <mrcs_file=\"\">                   	    : Input tilt-series.");
	addParamsLine("  [-o <output=\"./alignemntReport.xmd\">]    : Output file containing the alignemnt report.");

	addParamsLine("  [--samplingRate <samplingRate=1>]			: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--fiducialSize <fiducialSize=100>]		: Fiducial size in Angstroms (A).");
}



void ProgTomoDetectLandmarks::generateSideInfo()
{
	fiducialSizePx = fiducialSize / samplingRate; 
    
    ds_factor = targetFS / fiducialSizePx; 
    xSize_d = xSize * ds_factor;
    ySize_d = ySize * ds_factor;

    std::cout << "Generating side info: " << std::endl;
    std::cout << "fiducialSizePx: " << fiducialSizePx << std::endl;
    std::cout << "ds_factor: " << ds_factor << std::endl;
    std::cout << "xSize_d: " << xSize_d << std::endl;
    std::cout << "ySize_d: " << ySize_d << std::endl;
}


// --------------------------- HEAD functions ----------------------------
void ProgTomoDetectLandmarks::downsample(MultidimArray<double> &tiltImage, MultidimArray<double> &tiltImage_ds)
{
    MultidimArray<std::complex<double>> fftImg;
	MultidimArray<std::complex<double>> fftImg_ds;

	FourierTransformer transformer1;
	FourierTransformer transformer2;

    std::cout << "a" << std::endl;

    fftImg_ds.initZeros(ySize_d, xSize_d/2+1);
    transformer1.FourierTransform(tiltImage, fftImg, false);


    std::cout << "b" << std::endl;

    for (size_t i = 0; i < ySize_d/2; ++i)
    {
        for (size_t j = 0; j < xSize_d/2; ++j)
        {
            // DIRECT_A2D_ELEM(fftImg_ds, i, j) = DIRECT_A2D_ELEM(fftImg, 
            //                                                    (ySize/2)-(ySize_d/2)+i, 
            //                                                    (xSize/2)-(xSize_d/2)+j);

            DIRECT_A2D_ELEM(fftImg_ds, i, j) = DIRECT_A2D_ELEM(fftImg, i, j);
            DIRECT_A2D_ELEM(fftImg_ds, (ySize_d/2)+i, j) = DIRECT_A2D_ELEM(fftImg, ySize-ySize_d/2+i, j);
        }
    }

    std::cout << "c" << std::endl;

    transformer2.inverseFourierTransform(fftImg_ds, tiltImage_ds);

    std::cout << "d" << std::endl;

}

void ProgTomoDetectLandmarks::sobelFiler(MultidimArray<double> &tiltImage)
{  
    // Create the gradient images for x and y directions
    MultidimArray<double>  gradX;
    MultidimArray<double>  gradY;

    gradX.initZeros(ySize_d, xSize_d);
    gradY.initZeros(ySize_d, xSize_d);

    // Apply the Sobel filter in the x-direction
    for (int i = 1; i < ySize_d - 1; ++i)
    {
        for (int j = 1; j < xSize_d - 1; ++j)
        {
            double pixelValue = 0;
            
            for (int k = -1; k <= 1; ++k)
            {
                for (int l = -1; l <= 1; ++l)
                {
                    pixelValue += A2D_ELEM(tiltImage, i + k, j + l) * sobelX[k + 1][l + 1];
                }
            }
            
            A2D_ELEM(gradX, i, j) = pixelValue;
        }
    }
    
    // Apply the Sobel filter in the y-direction
    for (int i = 1; i < ySize_d - 1; ++i)
    {
        for (int j = 1; j < xSize_d - 1; ++j)
        {
            double pixelValue = 0;
            
            for (int k = -1; k <= 1; ++k)
            {
                for (int l = -1; l <= 1; ++l)
                {
                    pixelValue += A2D_ELEM(tiltImage, i + k, j + l) * sobelY[k + 1][l + 1];
                }
            }
            
            A2D_ELEM(gradY, i, j) = pixelValue;
        }
    }
    
    // Compute the gradient magnitude   
    tiltImage.initZeros(ySize_d, xSize_d);

    for (int i = 0; i < ySize_d; ++i)
    {
        for (int j = 0; j < xSize_d; ++j)
        {

            A2D_ELEM(tiltImage, i, j) = sqrt(A2D_ELEM(gradX, i, j) * A2D_ELEM(gradX, i, j) + 
                                             A2D_ELEM(gradY, i, j) * A2D_ELEM(gradY, i, j));
        }
    }
}


// --------------------------- I/O functions ----------------------------

void ProgTomoDetectLandmarks::writeOutputCoordinates()
{
	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameLandmarkCoordinates;
    outputFileNameLandmarkCoordinates = rawname + "/ts_landmarkCoordinates.xmd";

	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < coordinates3D.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_XCOOR, (int)coordinates3D[i].x, id);
		md.setValue(MDL_YCOOR, (int)coordinates3D[i].y, id);
		md.setValue(MDL_ZCOOR, (int)coordinates3D[i].z, id);
	}


	md.write(outputFileNameLandmarkCoordinates);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Output coordinates metadata saved at: " << outputFileNameLandmarkCoordinates << std::endl;
	#endif

}


// --------------------------- MAIN ----------------------------------

void ProgTomoDetectLandmarks::run()
{
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;	

	auto t1 = high_resolution_clock::now();

	std::cout << "Starting..." << std::endl;

	size_t Xdim, Ydim;

	MetaDataVec tiltseriesmd;
    ImageGeneric tiltSeriesImages;

    if (fnVol.isMetaData())
    {
        tiltseriesmd.read(fnVol);
    }
    else
    {
        tiltSeriesImages.read(fnVol, HEADER);

        size_t Zdim, Ndim;
        tiltSeriesImages.getDimensions(Xdim, Ydim, Zdim, Ndim);

        if (fnVol.getExtension() == "mrc" and Ndim == 1)
            Ndim = Zdim;

        size_t id;
        FileName fn;
        for (size_t i = 0; i < Ndim; i++) 
        {
            id = tiltseriesmd.addObject();
            fn.compose(i + FIRST_IMAGE, fnVol);
            tiltseriesmd.setValue(MDL_IMAGE, fn, id);
        }
    }

	tiltSeriesImages.getDimensions(xSize, ySize, zSize, nSize);

	#ifdef DEBUG_DIM
	std::cout << "Input tilt-series dimensions:" << std::endl;
	std::cout << "x " << xSize << std::endl;
	std::cout << "y " << ySize << std::endl;
	std::cout << "z " << zSize << std::endl;
	std::cout << "n " << nSize << std::endl;
	#endif

	generateSideInfo();

	FileName fnTSimg;
	size_t objId, objId_ts;
	Image<double> imgTS;

	MultidimArray<double> &ptrImg = imgTS();
    MultidimArray<double> projImgTS;
    MultidimArray<double> filteredImg;
    MultidimArray<double> freqMap;

	projImgTS.initZeros(Ydim, Xdim);

	size_t Ndim, counter = 0;
	Ndim = tiltseriesmd.size();

	MultidimArray<double> filteredTiltSeries;
	filteredTiltSeries.initZeros(Ndim, 1, ySize_d, xSize_d);

    #ifdef DEBUG_DIM
	std::cout << "Filtered tilt-series dimensions:" << std::endl;
	std::cout << "x " << XSIZE(filteredTiltSeries) << std::endl;
	std::cout << "y " << YSIZE(filteredTiltSeries) << std::endl;
	std::cout << "z " << ZSIZE(filteredTiltSeries) << std::endl;
	std::cout << "n " << NSIZE(filteredTiltSeries) << std::endl;
	#endif

	for(size_t objId : tiltseriesmd.ids())
	{
		tiltseriesmd.getValue(MDL_IMAGE, fnTSimg, objId);

        imgTS.read(fnTSimg);

        MultidimArray<double> tiltImage_ds;
        tiltImage_ds.initZeros(ySize_d, xSize_d);

        std::cout << "Downsampling image " << counter << std::endl;
        downsample(ptrImg, tiltImage_ds);

        #ifdef DEBUG_DIM
        std::cout << "Tilt-image dimensions after dowsampling:" << std::endl;
        std::cout << "x " << XSIZE(ptrImg) << std::endl;
        std::cout << "y " << YSIZE(ptrImg) << std::endl;
        #endif

        std::cout << "Aplying sobel filter to image " << counter << std::endl;
        sobelFiler(tiltImage_ds);

        for (size_t i = 0; i < ySize_d; ++i)
        {
            for (size_t j = 0; j < xSize_d; ++j)
            {
				DIRECT_NZYX_ELEM(filteredTiltSeries, counter, 0, i, j) = DIRECT_A2D_ELEM(tiltImage_ds, i, j);
			}
		}

		counter++;
	}
	
	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "/ts_filtered.mrcs";

	Image<double> saveImage;
	saveImage() = filteredTiltSeries;
	saveImage.write(outputFileNameFilteredVolume);
	#endif

	#ifdef DEBUG_DIM
	std::cout << "Filtered tilt-series dimensions:" << std::endl;
	std::cout << "x " << xSize_d << std::endl;
	std::cout << "y " << ySize_d << std::endl;
	std::cout << "z " << zSize << std::endl;
	std::cout << "n " << nSize << std::endl;
	#endif

	auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1); 	// Getting number of milliseconds as an integer
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}


// --------------------------- UTILS functions ----------------------------

