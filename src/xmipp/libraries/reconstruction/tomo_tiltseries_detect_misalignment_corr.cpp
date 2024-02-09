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

#include "tomo_calculate_landmark_residuals.h"
#include <chrono>



// --------------------------- INFO functions ----------------------------

void ProgTomoTSDetectMisalignmentCorr::readParams()
{
	fnVol = getParam("-i");
	fnTiltAngles = getParam("--tlt");
	fnOut = getParam("-o");

	samplingRate = getDoubleParam("--samplingRate");

	thrSDHCC = getIntParam("--thrSDHCC");

	thrFiducialDistance = getDoubleParam("--thrFiducialDistance");

	targetFS = getDoubleParam("--targetLMsize");
}



void ProgTomoTSDetectMisalignmentCorr::defineParams()
{
	addUsageLine("This function determines the location of high contrast features in a volume.");
	addParamsLine("  -i <mrcs_file=\"\">                   					: Input tilt-series.");
	addParamsLine("  --tlt <xmd_file=\"\">      							: Input file containning the tilt angles of the tilt-series in .xmd format.");
	addParamsLine("  --inputCoord <output=\"\">								: Input coordinates of the 3D landmarks. Origin at top left coordinate (X and Y always positive) and centered at the middle of the volume (Z positive and negative).");

	addParamsLine("  [-o <output=\"./alignemntReport.xmd\">]       			: Output file containing the alignemnt report.");

	addParamsLine("  [--samplingRate <samplingRate=1>]						: Sampling rate of the input tomogram (A/px).");


	addParamsLine("  [--thrSDHCC <thrSDHCC=5>]      						: Threshold number of SD a coordinate value must be over the mean to consider that it belongs to a high contrast feature.");
	addParamsLine("  [--thrFiducialDistance <thrFiducialDistance=0.5>]		: Threshold times of fiducial size as maximum distance to consider a match between the 3d coordinate projection and the detected fiducial.");

	addParamsLine("  [--targetLMsize <targetLMsize=8>]		    : Targer size of landmark when downsampling (px).");
}



void ProgTomoTSDetectMisalignmentCorr::generateSideInfo()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Generating side info..." << std::endl;
	#endif

	// Read tilt angles file
	MetaDataVec inputTiltAnglesMd;
	inputTiltAnglesMd.read(fnTiltAngles);

	size_t objIdTlt;

	double tiltAngle;
	tiltAngleStep=0;

	for(size_t objIdTlt : inputTiltAnglesMd.ids())
	{
		inputTiltAnglesMd.getValue(MDL_ANGLE_TILT, tiltAngle, objIdTlt);
		tiltAngles.push_back(tiltAngle);

		tiltAngleStep += tiltAngle;
	}

	tiltAngleStep /= tiltAngles.size();

	#ifdef VERBOSE_OUTPUT
	std::cout << "Input tilt angles read from: " << fnTiltAngles << std::endl;
	#endif


	// Initialize local alignment vector (depends on the number of acquisition angles)
	localAlignment.resize(nSize, true);


	#ifdef VERBOSE_OUTPUT
	std::cout << "Side info generated succesfully!" << std::endl;
	#endif
}


// --------------------------- HEAD functions ----------------------------

void ProgTomoTSDetectMisalignmentCorr::detectSubtleMisalingment()
{
	applyGeometry
}



// --------------------------- I/O functions ----------------------------



// --------------------------- MAIN ----------------------------------
void ProgTomoTSDetectMisalignmentCorr::run()
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

	detectSubtleMisalingment();

	auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1); 	// Getting number of milliseconds as an integer
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}


// --------------------------- UTILS functions ----------------------------


void ProgTomoTSDetectMisalignmentCorr::adjustCoordinatesCosineStreching()
{
	MultidimArray<int> csProyectedCoordinates;
	csProyectedCoordinates.initZeros(ySize, xSize);

	Point3D<double> dc;
	Point3D<double> c3d;
	int xTA = (int)(xSize/2);

	int goldBeadX;
	int goldBeadY;
	int goldBeadZ;

	for(int j = 0; j < numberOfInputCoords; j++)
	{
		goldBeadX = inputCoords[j].x;
		goldBeadY = inputCoords[j].y;
		goldBeadZ = inputCoords[j].z;

		#ifdef DEBUG_COORDS_CS
		std::cout << "Analyzing residuals corresponding to coordinate 3D " << goldBeadX << ", " << goldBeadY << ", " << goldBeadZ << std::endl;
		#endif

	    std::vector<CM> vCMc;
		getCMFromCoordinate(goldBeadX, goldBeadY, goldBeadZ, vCMc);

		std::vector<Point2D<double>> residuals;
		for (size_t i = 0; i < vCMc.size(); i++)
		{
			residuals.push_back(vCMc[i].residuals);
		}

		#ifdef DEBUG_COORDS_CS
		std::cout << " vCMc.size()" << vCMc.size() << std::endl;
		#endif

		// These are the proyected 2D points. The Z component is the id for each 3D coordinate (cluster projections).
		std::vector<Point3D<double>> proyCoords;

		for (size_t i = 0; i < vCMc.size(); i++)
		{
			CM cm = vCMc[i];
			double tiltAngle = tiltAngles[(int)cm.detectedCoordinate.z]* PI/180.0;
			
			Point3D<double> proyCoord(((((cm.detectedCoordinate.x-xTA)-((cm.coordinate3d.z)*sin(tiltAngle)))/cos(tiltAngle))+xTA), 
									  cm.detectedCoordinate.y,
									  i);
			proyCoords.push_back(proyCoord);

			#ifdef DEBUG_OUTPUT_FILES
			fillImageLandmark(csProyectedCoordinates,
							  (int) ((((cm.detectedCoordinate.x-xTA)-((cm.coordinate3d.z)*sin(tiltAngle)))/cos(tiltAngle))+xTA),
							  (int)cm.detectedCoordinate.y,
							  i);
			#endif
		}
	}

	#ifdef DEBUG_OUTPUT_FILES
	size_t li = fnOut.find_last_of("\\/");
	std::string rn = fnOut.substr(0, li);
	std::string ofn;
    ofn = rn + "/ts_proyected_cs.mrc";

	Image<int> si;
	si() = csProyectedCoordinates;
	si.write(ofn);
	#endif
}


Matrix2D<double> ProgTomoTSDetectMisalignmentCorr::getCosineStretchingMatrix(double tiltAngle)
{
	double cosTiltAngle = cos(tiltAngle * PI/180.0);

	Matrix2D<double> m(3,3);

	MAT_ELEM(m, 0, 0) = 1/cosTiltAngle;
	// MAT_ELEM(m, 0, 1) = 0;
	// MAT_ELEM(m, 0, 2) = 0;
	// MAT_ELEM(m, 1, 0) = 0;
	MAT_ELEM(m, 1, 1) = 1;
	// MAT_ELEM(m, 1, 2) = 0;
	// MAT_ELEM(m, 2, 0) = 0;
	// MAT_ELEM(m, 2, 1) = 0;
	MAT_ELEM(m, 2, 2) = 1;

	return m;
}
