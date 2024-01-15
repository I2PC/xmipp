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

#include "tomo_detect_misalignment_trajectory.h"
#include <chrono>



// --------------------------- INFO functions ----------------------------

void ProgTomoDetectMisalignmentTrajectory::readParams()
{
	fnVol = getParam("-i");
	fnTiltAngles = getParam("--tlt");
	fnOut = getParam("-o");

	samplingRate = getDoubleParam("--samplingRate");
	fiducialSize = getDoubleParam("--fiducialSize");

	thrSDHCC = getIntParam("--thrSDHCC");

 	fnInputCoord = getParam("--inputCoord");

	thrFiducialDistance = getDoubleParam("--thrFiducialDistance");
	// avgResidPercentile_LocalAlignment = getDoubleParam("--avgResidPercentile_LocalAlignment");

	targetFS = getDoubleParam("--targetLMsize");
}



void ProgTomoDetectMisalignmentTrajectory::defineParams()
{
	addUsageLine("This function determines the location of high contrast features in a volume.");
	addParamsLine("  -i <mrcs_file=\"\">                   					: Input tilt-series.");
	addParamsLine("  --tlt <xmd_file=\"\">      							: Input file containning the tilt angles of the tilt-series in .xmd format.");
	addParamsLine("  --inputCoord <output=\"\">								: Input coordinates of the 3D landmarks. Origin at top left coordinate (X and Y always positive) and centered at the middle of the volume (Z positive and negative).");

	addParamsLine("  [-o <output=\"./alignemntReport.xmd\">]       			: Output file containing the alignemnt report.");

	addParamsLine("  [--samplingRate <samplingRate=1>]						: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--fiducialSize <fiducialSize=100>]					: Fiducial size in Angstroms (A).");


	addParamsLine("  [--thrSDHCC <thrSDHCC=5>]      						: Threshold number of SD a coordinate value must be over the mean to consider that it belongs to a high contrast feature.");
	addParamsLine("  [--thrFiducialDistance <thrFiducialDistance=0.5>]		: Threshold times of fiducial size as maximum distance to consider a match between the 3d coordinate projection and the detected fiducial.");
	// addParamsLine("  [--avgResidPercentile_LocalAlignment <avgResidPercentile_LocalAlignment=0.5>]		: Threshold times of fiducial size as maximum distance to consider a match between the 3d coordinate projection and the detected fiducial.");

	addParamsLine("  [--targetLMsize <targetLMsize=8>]		    : Targer size of landmark when downsampling (px).");
}



void ProgTomoDetectMisalignmentTrajectory::generateSideInfo()
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
	avgMahalanobisDistanceV.resize(nSize, 0.0);
	stdMahalanobisDistanceV.resize(nSize, 0.0);

	// Update thresholds depending on input tilt-series sampling rate
	fiducialSizePx = fiducialSize / samplingRate; 

	#ifdef VERBOSE_OUTPUT
	std::cout << "Thresholds:" << std::endl;
	std::cout << "fiducialSizePx: "<< fiducialSizePx << std::endl;
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "Side info generated succesfully!" << std::endl;
	#endif

	// Read input coordinates file
	MetaDataVec inputCoordMd;
	inputCoordMd.read(fnInputCoord);

	size_t objIdCoord;

	int goldBeadX;
	int goldBeadY;
	int goldBeadZ;

	for(size_t objIdCoord : inputCoordMd.ids())
	{
		inputCoordMd.getValue(MDL_XCOOR, goldBeadX, objIdCoord);
		inputCoordMd.getValue(MDL_YCOOR, goldBeadY, objIdCoord);
		inputCoordMd.getValue(MDL_ZCOOR, goldBeadZ, objIdCoord);

		Point3D<int> coord3D(goldBeadX, goldBeadY, goldBeadZ);
		inputCoords.push_back(coord3D);
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Input coordinates read from: " << fnInputCoord << std::endl;
	#endif

	numberOfInputCoords = inputCoords.size();

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of input coordinates: " << numberOfInputCoords << std::endl;
	#endif

	// Initialize landmark detector
	size_t lastIndex = fnOut.find_last_of("\\/");
	std::string lmCoordsFn = fnOut.substr(0, lastIndex);
	lmCoordsFn = lmCoordsFn + "/landmarkCoordinates.xmd";

	lmDetector.fnVol = fnVol;
	lmDetector.fnOut = lmCoordsFn;
	lmDetector.samplingRate = samplingRate;
	lmDetector.fiducialSize = fiducialSize;
	lmDetector.targetFS = targetFS;
	lmDetector.thrSD = thrSDHCC;

	#ifdef VERBOSE_OUTPUT
	std::cout << "----- Run landmark detector" << std::endl;
	#endif

	lmDetector.run();

	#ifdef VERBOSE_OUTPUT
	std::cout << "----- Landmark detector execution finished!" << std::endl;
	#endif
}


// --------------------------- HEAD functions ----------------------------
void ProgTomoDetectMisalignmentTrajectory::calculateResidualVectors()
{
	// ***TODO: homogeneizar PointXD y MatrixXD
	#ifdef VERBOSE_OUTPUT
	std::cout << "Calculating residual vectors" << std::endl;
	#endif

	size_t objId;
	size_t minIndex;
	double tiltAngle;
	double distance;
	double minDistance;

	int goldBeadX;
	int goldBeadY;
	int goldBeadZ;

	Matrix2D<double> projectionMatrix;
	Matrix1D<double> goldBead3d;
	Matrix1D<double> projectedGoldBead;

	std::vector<Point2D<double>> coordinatesInSlice;

	goldBead3d.initZeros(3);

	// Iterate through every tilt-image
	for(size_t n = 0; n < tiltAngles.size(); n++)
	{	
		#ifdef DEBUG_RESID
		std::cout << "Analyzing coorinates in image "<< n <<std::endl;
		#endif

		tiltAngle = tiltAngles[n];

		# ifdef DEBUG
		std::cout << "Calculating residual vectors at slice " << n << " with tilt angle " << tiltAngle << "º" << std::endl;
		#endif

		coordinatesInSlice = getCoordinatesInSlice(n);

		projectionMatrix = getProjectionMatrix(tiltAngle);

		#ifdef DEBUG_RESID
		std::cout << "Projection matrix------------------------------------"<<std::endl;
		std::cout << MAT_ELEM(projectionMatrix, 0, 0) << " " << MAT_ELEM(projectionMatrix, 0, 1) << " " << MAT_ELEM(projectionMatrix, 0, 2) << std::endl;
		std::cout << MAT_ELEM(projectionMatrix, 1, 0) << " " << MAT_ELEM(projectionMatrix, 1, 1) << " " << MAT_ELEM(projectionMatrix, 1, 2) << std::endl;
		std::cout << MAT_ELEM(projectionMatrix, 2, 0) << " " << MAT_ELEM(projectionMatrix, 2, 1) << " " << MAT_ELEM(projectionMatrix, 2, 2) << std::endl;
		std::cout << "------------------------------------"<<std::endl;

		std::cout << "---------------- Coordinates in slice " << n << "------------------------------------" << std::endl;
		for(size_t i = 0; i < coordinatesInSlice.size(); i++)
		{
			std::cout << coordinatesInSlice[i].x << ", " << coordinatesInSlice[i].y << std::endl;
		}
		
		std::cout << "=============================================================================================================" << std::endl;
		#endif
		

		if (coordinatesInSlice.size() != 0)
		{
			size_t coordinate3dId = 0;

			// Iterate through every input 3d gold bead coordinate and project it onto the tilt image
			for(int j = 0; j < numberOfInputCoords; j++)
			{
				minDistance = MAXDOUBLE;

				goldBeadX = inputCoords[j].x;
				goldBeadY = inputCoords[j].y;
				goldBeadZ = inputCoords[j].z;

				#ifdef DEBUG_RESID
				std::cout << "=============================================================================================================" << std::endl;
				std::cout << "Analyzing image " << n << std::endl;
				std::cout << "goldBeadX " << goldBeadX << std::endl;
				std::cout << "goldBeadY " << goldBeadY << std::endl;
				std::cout << "goldBeadZ " << goldBeadZ << std::endl;
				#endif

				// Update coordinates wiht origin as the center of the tomogram (needed for rotation matrix multiplicaiton)
				XX(goldBead3d) = (double) (goldBeadX - (double)xSize/2);
				YY(goldBead3d) = (double) goldBeadY; // Since we are rotating respect to Y axis, no conversion is needed
				ZZ(goldBead3d) = (double) (goldBeadZ);

				projectedGoldBead = projectionMatrix * goldBead3d;

				XX(projectedGoldBead) += (double)xSize/2;
				// YY(projectedGoldBead) += 0; // Since we are rotating respect to Y axis, no conersion is needed
				// ZZ(projectedGoldBead) += (double)zSize/2;

				// Check that the coordinate is not proyected out of the interpolation edges for this tilt-image
				bool coordInIC = checkProjectedCoordinateInInterpolationEdges(projectedGoldBead, n);
				
				if (coordInIC)
				{
					#ifdef DEBUG_RESID
					std::cout << "XX(goldBead3d) " << XX(goldBead3d) << std::endl;
					std::cout << "YY(goldBead3d) " << YY(goldBead3d) << std::endl;
					std::cout << "ZZ(goldBead3d) " << ZZ(goldBead3d) << std::endl;

					std::cout << "tiltAngles[n] " << tiltAngles[n] << std::endl;
					std::cout << "XX(projectedGoldBead) " << XX(projectedGoldBead) << std::endl;
					std::cout << "YY(projectedGoldBead) " << YY(projectedGoldBead) << std::endl;
					std::cout << "ZZ(projectedGoldBead) " << ZZ(projectedGoldBead) << std::endl;
					#endif

					// Iterate though every coordinate in the tilt-image and calculate the minimum distance
					for(size_t i = 0; i < coordinatesInSlice.size(); i++)
					{
						distance = (XX(projectedGoldBead) - coordinatesInSlice[i].x)*(XX(projectedGoldBead) - coordinatesInSlice[i].x) + (YY(projectedGoldBead) - coordinatesInSlice[i].y)*(YY(projectedGoldBead) - coordinatesInSlice[i].y);

						#ifdef DEBUG_RESID
						// std::cout << "------------------------------------------------------------------------------------" << std::endl;
						// std::cout << "i/i_total " << i << "/" << coordinatesInSlice.size()-1 << std::endl;
						
						// std::cout << "tiltAngles[n] " << tiltAngles[n] << std::endl;
						// std::cout << "XX(projectedGoldBead) " << XX(projectedGoldBead) << std::endl;
						// std::cout << "YY(projectedGoldBead) " << YY(projectedGoldBead) << std::endl;
						// std::cout << "ZZ(projectedGoldBead) " << ZZ(projectedGoldBead) << std::endl;
						
						// std::cout << "XX(goldBead3d) " << XX(goldBead3d) << std::endl;
						// std::cout << "YY(goldBead3d) " << YY(goldBead3d) << std::endl;
						// std::cout << "ZZ(goldBead3d) " << ZZ(goldBead3d) << std::endl;

						// std::cout << "coordinatesInSlice[i].x " << coordinatesInSlice[i].x << std::endl;
						// std::cout << "coordinatesInSlice[i].y " << coordinatesInSlice[i].y << std::endl;

						// std::cout << "coordinatesInSlice[i].x - XX(projectedGoldBead) " << coordinatesInSlice[i].x - XX(projectedGoldBead) << std::endl;
						// std::cout << "coordinatesInSlice[i].y - YY(projectedGoldBead) " << coordinatesInSlice[i].y - YY(projectedGoldBead) << std::endl;

						// std::cout << "minDistance " << minDistance << std::endl;
						// std::cout << "distance " << distance << std::endl;
						// std::cout << "------------------------------------------------------------------------------------" << std::endl;					
						#endif

						if(distance < minDistance)
						{
							minDistance = distance;
							minIndex = i;
						}
					}

					// Back to Xmipp origin (top-left corner)
					XX(goldBead3d) = (double) goldBeadX;

					Point3D<double> cis(coordinatesInSlice[minIndex].x, coordinatesInSlice[minIndex].y, n);
					Point3D<double> c3d(XX(goldBead3d), YY(goldBead3d), ZZ(goldBead3d));
					Point2D<double> res(XX(projectedGoldBead)-coordinatesInSlice[minIndex].x, YY(projectedGoldBead) - coordinatesInSlice[minIndex].y); 

					#ifdef DEBUG_RESID
					std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
					std::cout << "XX(projectedGoldBead) " << XX(projectedGoldBead) << std::endl;
					std::cout << "YY(projectedGoldBead) " << YY(projectedGoldBead) << std::endl;
					std::cout << "ZZ(projectedGoldBead) " << ZZ(projectedGoldBead) << std::endl;

					std::cout << "XX(goldBead3d) " << XX(goldBead3d) << std::endl;
					std::cout << "YY(goldBead3d) " << YY(goldBead3d) << std::endl;
					std::cout << "ZZ(goldBead3d) " << ZZ(goldBead3d) << std::endl;

					std::cout << "coordinatesInSlice[minIndex].x " << coordinatesInSlice[minIndex].x << std::endl;
					std::cout << "coordinatesInSlice[minIndex].y " << coordinatesInSlice[minIndex].y << std::endl;

					std::cout << "coordinatesInSlice[minIndex].x - XX(projectedGoldBead) " << coordinatesInSlice[minIndex].x - XX(projectedGoldBead) << std::endl;
					std::cout << "coordinatesInSlice[minIndex].y - YY(projectedGoldBead) " << coordinatesInSlice[minIndex].y - YY(projectedGoldBead) << std::endl;

					std::cout << "minDistance " << minDistance << std::endl;
					std::cout << "sqrt(minDistance) " << sqrt(minDistance) << std::endl;
					std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
					#endif


					CM cm {cis, c3d, res, coordinate3dId, 0};  // 0 is for non-initialized mahalanobis distance
					vCM.push_back(cm);

					coordinate3dId += 1;
				}
			}
		}else
		{
			std::cout << "WARNING: No coorinate peaked in slice " << n << ". IMPOSIBLE TO STUDY MISALIGNMENT IN THIS SLICE." << std::endl;
		}
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Residual vectors calculated: " << vCM.size() << std::endl;
	#endif
}


void ProgTomoDetectMisalignmentTrajectory::pruneResidualVectors()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Pruning resudial vectors..." << std::endl;
	#endif

	MultidimArray<double> fiducial_origin;
	MultidimArray<double> fiducial_end;

	std::vector<CM> vCM_pruned;

	std::vector<Point3D<double>> newCoordinates3D;

	int deletedCoordinates = 0;

	int boxSize = 2 * int(fiducialSizePx);
	int halfBoxSize = boxSize / 2;

	int numberOfCM = vCM.size();

	for(size_t n = 0; n < numberOfCM ; n++)
	{
		CM cm = vCM[n];

		Point3D<double> detectedCoordinate = cm.detectedCoordinate;
        Point2D<double> residuals = cm.residuals;

		int fiducialOriginX = detectedCoordinate.x - halfBoxSize;
		int fiducialOriginY = detectedCoordinate.y - halfBoxSize;
		int fiducialEndX = fiducialOriginX + residuals.x;
		int fiducialEndY = fiducialOriginY + residuals.y;
		int tiltImage = detectedCoordinate.z;

		#ifdef DEBUG_PRUNE_RESIDUALS
		std::cout << "--- Analyzing residual number " << n << std::endl;
		std::cout << "tiltImage " << tiltImage << std::endl;
		std::cout << "fiducialOriginX " << fiducialOriginX << std::endl;
		std::cout << "fiducialOriginY " << fiducialOriginY << std::endl;
		std::cout << "fiducialEndX " << fiducialEndX << std::endl;
		std::cout << "fiducialEndY " << fiducialEndY << std::endl;
		std::cout << "residuals.x " << residuals.x << std::endl;
		std::cout << "residuals.y " << residuals.y << std::endl;
		std::cout << "residual module ^2 " << (residuals.x*residuals.x + residuals.y*residuals.y) << std::endl;
		#endif

		// Check that residual do not point to the fiducial itself
		// (only run test for non-matching vectors)
		if ((residuals.x*residuals.x + residuals.y*residuals.y) > (fiducialSizePx * fiducialSizePx / 4))
		{
			// Construct cropped fiducial at origin and end of the resildual vector
			fiducial_origin.initZeros(boxSize, boxSize);
			fiducial_end.initZeros(boxSize, boxSize);

			for(int j = 0; j < boxSize; j++) // xDim
			{
				for(int i = 0; i < boxSize; i++) // yDim
				{
					// Origin
					if ((fiducialOriginY + i) < 0 || (fiducialOriginY + i) >= ySize ||
						(fiducialOriginX + j) < 0 || (fiducialOriginX + j) >= xSize)
					{
						DIRECT_A2D_ELEM(fiducial_origin, i, j) = 0;
					}
					else
					{
						DIRECT_A2D_ELEM(fiducial_origin, i, j) = DIRECT_A3D_ELEM(lmDetector.tiltSeries,
																				 tiltImage,
																				 fiducialOriginY + i, 
																				 fiducialOriginX + j);
					}

					// End
					if ((fiducialEndY + i) < 0 || (fiducialEndY + i) >= ySize ||
						(fiducialEndX + j) < 0 || (fiducialEndX + j) >= xSize)
					{
						DIRECT_A2D_ELEM(fiducial_end, i, j) = 0;
					}
					else
					{
						DIRECT_A2D_ELEM(fiducial_end, i, j) = DIRECT_A3D_ELEM(lmDetector.tiltSeries,
																			  tiltImage,
																			  fiducialEndY + i, 
																			  fiducialEndX + j);
					}
				}
			}

			fiducial_origin.statisticsAdjust(0.0, 1.0);
			fiducial_end.statisticsAdjust(0.0, 1.0);

			#ifdef DEBUG_PRUNE_RESIDUALS
			Image<double> fiducial;

			std::cout << "Origin fiducial dimensions (" << XSIZE(fiducial_origin) << ", " << YSIZE(fiducial_origin) << ")" << std::endl;
			fiducial() = fiducial_origin;
			size_t lastindex = fnOut.find_last_of(".");
			std::string rawname = fnOut.substr(0, lastindex);
			std::string outputFileNameSubtomo;
			outputFileNameSubtomo = rawname + "_" + std::to_string(n) + "_fid_origin.mrc";
			fiducial.write(outputFileNameSubtomo);

			std::cout << "End fiducial dimensions (" << XSIZE(fiducial_end) << ", " << YSIZE(fiducial_end) << ")" << std::endl;
			fiducial() = fiducial_end;
			outputFileNameSubtomo = rawname + "_" + std::to_string(n) + "_fid_end.mrc";
			fiducial.write(outputFileNameSubtomo);
			#endif

			double dotProduct = 0;

			// Calculate scalar product
			for(int j = 0; j < boxSize; j++) // xDim
			{
				for(int i = 0; i < boxSize; i++) // yDim
				{
					dotProduct += DIRECT_A2D_ELEM(fiducial_origin, i, j) * DIRECT_A2D_ELEM(fiducial_end, i, j);
				}
			}

			dotProduct = dotProduct /(boxSize * boxSize);

			#ifdef DEBUG_PRUNE_RESIDUALS
			std::cout << "Dot product: " << dotProduct << std::endl;
			#endif

			if (dotProduct > 0.3)
			{
				vCM.erase(vCM.begin()+n);
				numberOfCM--;
				n--;
				deletedCoordinates++;

				#ifdef DEBUG_PRUNE_RESIDUALS
				std::cout << "Residual " << n << " removed. Origin-end correlation: " << dotProduct << std::endl;
				#endif
			}
		}
	}

	#ifdef DEBUG_PRUNE_RESIDUALS
	std::cout << "Number of residuals removed by origin-end correlation: " << deletedCoordinates << std::endl;
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "Pruning resudial vectors finished succesfully!" << std::endl;
	#endif
}

void ProgTomoDetectMisalignmentTrajectory::detectMisalignmentFromResidualsMahalanobis()
{
	double sigma = (fiducialSize / samplingRate) / 3;	// Sigma for 99% of the points inside the fiducial radius
	std::cout << "(fiducialSize / samplingRate): " << (fiducialSize / samplingRate) << std::endl;
	std::cout << "sigma: " << sigma << std::endl;

	// double sigma2 = sigma * sigma;

	// Matrix2D<double> covariance_inv;
	// MAT_ELEM(covariance_inv, 0, 0) = 1/sigma2;
	// MAT_ELEM(covariance_inv, 0, 1) = 0;
	// MAT_ELEM(covariance_inv, 1, 0) = 0;
	// MAT_ELEM(covariance_inv, 1, 1) = 1/sigma2;

	// iterate residuals
	for(size_t i = 0; i < vCM.size(); i++)
	{
		vCM[i].mahalanobisDistance = sqrt(vCM[i].residuals.x/sigma + vCM[i].residuals.y/sigma);
	}

	// Global alignment analysis
	std::cout << "---------------- Global misalignemnt analysis" << std::endl;

	std::vector<bool> globalAlingmentVotting(numberOfInputCoords, true);  // Vector saving status of (mis)aligned chains
	double vottingRatio = 0.0;

	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		std::vector<CM> CM_fid;
		getCMbyFiducial(n, CM_fid);

		size_t numberCM = CM_fid.size();

		double sumMahaDist = 0;
		double sumMahaDist2 = 0;

		for (size_t i = 0; i < numberCM; i++)
		{
			sumMahaDist += CM_fid[i].mahalanobisDistance;
			sumMahaDist2 += CM_fid[i].mahalanobisDistance * CM_fid[i].mahalanobisDistance; 
		}

		double avgMahaDist = sumMahaDist / numberCM;
		double stdMahaDist = sqrt(sumMahaDist2 / numberCM - avgMahaDist * avgMahaDist);

		if (avgMahaDist > 1)
		{
			globalAlingmentVotting[n] = false;
			vottingRatio += 1;
		}

		std::cout << "Statistics of mahalanobis distances for 3D coordinate " << n << std::endl;
		std::cout << "Average mahalanobis distance: " << avgMahaDist << std::endl;
		std::cout << "STD mahalanobis distance: " << stdMahaDist << std::endl;
	}

	if ((vottingRatio/numberOfInputCoords) > 0.5)
	{
		globalAlignment = false;
	}
	

	// Local alignment analysis
	std::cout << "---------------- Local misalignemnt analysis" << std::endl;
	
	for (size_t n = 0; n < nSize; n++)
	{
		std::vector<CM> CM_image;
		getCMbyImage(n, CM_image);

		size_t numberCM = CM_image.size();

		double sumMahaDist = 0;
		double sumMahaDist2 = 0;

		for (size_t i = 0; i < numberCM; i++)
		{
			sumMahaDist += CM_image[i].mahalanobisDistance;
			sumMahaDist2 += CM_image[i].mahalanobisDistance * CM_image[i].mahalanobisDistance; 
		}

		double avgMahaDist = sumMahaDist / numberCM;
		double stdMahaDist = sqrt(sumMahaDist2 / numberCM - avgMahaDist * avgMahaDist);

		avgMahalanobisDistanceV[n] = avgMahaDist;
		stdMahalanobisDistanceV[n] = stdMahaDist;

		if (avgMahaDist > 1)
		{
			localAlignment[n] = false;	
		}

		std::cout << "Statistics of mahalanobis distances for 3D coordinate " << n << std::endl;
		std::cout << "Average mahalanobis distance: " << avgMahaDist << std::endl;
		std::cout << "STD mahalanobis distance: " << stdMahaDist << std::endl;
	}
}

void ProgTomoDetectMisalignmentTrajectory::detectMisalignmentFromResiduals()
{
	double mod2Thr = (fiducialSizePx * thrFiducialDistance) * (fiducialSizePx * thrFiducialDistance);

	// Global alignment analysis
	std::vector<bool> globalMialingmentVotting(numberOfInputCoords, true);  // Vector saving status of (mis)aligned chains
	float vottingRatio;

	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		std::vector<CM> CM_fid;
		getCMbyFiducial(n, CM_fid);

		size_t numberCM = CM_fid.size();

		double avg;
		double std;
		size_t imagesOutOfRange = 0;

		double sumResid = 0;
		double sumResid2 = 0;

		for (size_t i = 0; i < numberCM; i++)
		{
			double sum2 = CM_fid[i].residuals.x*CM_fid[i].residuals.x + CM_fid[i].residuals.y*CM_fid[i].residuals.y;
			sumResid2 += sum2;
			sumResid += sqrt(sum2);

			if (sum2 > mod2Thr)
			{
				imagesOutOfRange += 1;
			}
		}

		avg = sumResid / numberCM;
		std = sqrt(sumResid2 / numberCM - avg * avg);

		#ifdef DEBUG_RESIDUAL_ANALYSIS
		std::cout << "n " << n << std::endl;
		std::cout << "numberCM " << numberCM << std::endl;
		std::cout << "sumResid " << sumResid << std::endl;
		std::cout << "sumResid2 " << sumResid2 << std::endl;
		std::cout << "imagesOutOfRange " << imagesOutOfRange << std::endl;
		std::cout << "avg " << avg << std::endl;
		std::cout << "std " << std << std::endl;
		#endif

		if (imagesOutOfRange > 4.5 && std > 43.5)
		{
			globalMialingmentVotting[n] = false;
			
			#ifdef DEBUG_RESIDUAL_ANALYSIS
			std::cout << "Chain number " << n << " present global misalignment with std=" << std << " and imagesOutOfRange=" << imagesOutOfRange << std::endl;
			#endif			
		}
		else if (imagesOutOfRange > 6.5)
		{
			globalMialingmentVotting[n] = false;

			#ifdef DEBUG_RESIDUAL_ANALYSIS
			std::cout << "Chain number " << n << " present global misalignment with imagesOutOfRange=" << imagesOutOfRange << std::endl;
			#endif
		}	
	}

	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		#ifdef DEBUG_RESIDUAL_ANALYSIS
		std::cout << globalMialingmentVotting[n] << " ";
		#endif

		if (globalMialingmentVotting[n])
		{
			vottingRatio += 1;
		}
		
	}

	vottingRatio /= numberOfInputCoords;

	#ifdef DEBUG_RESIDUAL_ANALYSIS
	std::cout  << "\n votting ratio " << vottingRatio << std::endl;
	#endif

	if (vottingRatio < 0.5)
	{
		globalAlignment = false;

		#ifdef VERBOSE_OUTPUT
		std::cout << "GLOBAL MISLAIGNMENT DETECTED" << std::endl;
		#endif

		return;
	}

	#ifdef DEBUG_RESIDUAL_ANALYSIS
	std::cout << "Output global (chain) alingmnet vector" << std::endl;
	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		std::cout << globalMialingmentVotting[n] << ", ";
	}
	std::cout << std::endl;
	#endif

	// Local alignment analysis
	for (size_t n = 0; n < nSize; n++)
	{
		std::vector<CM> CM_image;
		getCMbyImage(n, CM_image);

		size_t numberCM = CM_image.size();
		double vottingRatio = 0;

		#ifdef DEBUG_RESIDUAL_ANALYSIS
		std::cout << "------------ Analyzing image " << n << ". Presenting " << numberCM << " coordinates." <<  std::endl;
		#endif

		if (numberCM > 0)
		{
			for (size_t i = 0; i < numberCM; i++)
			{
				double resid2 = CM_image[i].residuals.x*CM_image[i].residuals.x + CM_image[i].residuals.y*CM_image[i].residuals.y;

				if(resid2 > mod2Thr)
				{
					vottingRatio += 1;
				}
				
			}

			vottingRatio /= float(numberCM);
			
			#ifdef DEBUG_RESIDUAL_ANALYSIS
			std::cout << "-------- For image " << n << " votting ratio=" << vottingRatio << " out of " << numberCM << std::endl;
			#endif

			if (vottingRatio > 0.5)
			{
				localAlignment[n] = false;

				#ifdef VERBOSE_OUTPUT
				std::cout << "LOCAL MISLAIGNMENT DETECTED AT TILT-IMAGE " << n << ". Failed residuals ratio: " << vottingRatio << " out of " << numberCM << std::endl;
				#endif
			}
		}

		else
		{
			#ifdef VERBOSE_OUTPUT
			std::cout << "UNDETECTED COORDINATES IN TILT-IMAGE " << n << ". IMPOSSIBLE TO STUDY MIALIGNMENT" << std::endl;
			#endif
		}
	}
}


void ProgTomoDetectMisalignmentTrajectory::generateResidualStatiscticsFile()
{
	// CODE FOR GENERATING RESIDUAL STATISTICS FILE FOR DECISION TREE TRAINING
	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Run XmippScript for statistical residual analysis

	std::cout << "\nRunning residual statistical analysis..." << std::endl;

	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string fnVCM;
	std::string fnStats;
    fnVCM = rawname + "/vCM.xmd";
	fnStats = rawname + "/residualStatistics.xmd";

	std::string cmd;

	#ifdef DEBUG_RESIDUAL_STATISTICS_FILE
	// Debug command
	cmd = "python3 /home/fdeisidro/xmipp_devel/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnVCM + " -o " + fnStats + " --debug ";
	// cmd = "python3 /home/fdeisidro/data/xmipp/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnVCM + " -o " + fnStats + " --debug";
	#else
	// No debug command
	cmd = "python3 /home/fdeisidro/xmipp_devel/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnVCM + " -o " + fnStats;
	// cmd = "python3 /home/fdeisidro/data/xmipp/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnVCM + " -o " + fnStats;
	#endif
	
	std::cout << cmd << std::endl;
	int systemOut = system(cmd.c_str());
	
	std::string fnStats_lm;
	std::string fnStats_image;

	fnStats_lm = rawname + "/residualStatistics_resid.xmd";
	fnStats_image = rawname + "/residualStatistics_image.xmd";

	std::string statistic;
	std::string statisticName;
	std::string residualNumber;
	int residualNumber_int;
	double value;

	// -- Save residual information --
	MetaDataVec residualStatsMd;
	residualStatsMd.read(fnStats_lm);

	// Vector containing stats: avg, std, chArea, chPerim, pvBinX, pvBinY, pvF, pvADF, ImageOutOfRange, LongestMisaliChain
	std::vector<double> residualStatsLine (10);
	std::vector<std::vector<double>> residualStatsTable;

	for (size_t i = 0; i < numberOfInputCoords; i++)
	{
		residualStatsTable.push_back(residualStatsLine);
	}

	// Save statistics in table
	for(size_t objId : residualStatsMd.ids())
	{
		residualStatsMd.getValue(MDL_IMAGE, statistic, objId);
		residualStatsMd.getValue(MDL_MIN, value, objId);

		statisticName = statistic.substr(statistic.find("_")+1);
		residualNumber = statistic.substr(0, statistic.find("_"));
		residualNumber_int = std::stoi(residualNumber);

		#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
		std::cout << "Reading object " << objId << " from metadata" << std::endl;
		std::cout << "statistic " << statistic << std::endl;
		std::cout << "value " << value << std::endl;
		std::cout << "statisticName " << statisticName << std::endl;
		std::cout << "residualNumber " << residualNumber << std::endl;
		std::cout << "residualNumber_int " << residualNumber_int << std::endl;
		#endif

		if (strcmp(statisticName.c_str(), "chArea") == 0)
		{
			residualStatsTable[residualNumber_int][2] = value;
		}

		else if (strcmp(statisticName.c_str(), "chPerim") == 0)
		{
			residualStatsTable[residualNumber_int][3] = value;
		}
		
		else if (strcmp(statisticName.c_str(), "pvBinX") == 0)
		{
			residualStatsTable[residualNumber_int][4] = value;
		}

		else if (strcmp(statisticName.c_str(), "pvBinY") == 0)
		{
			residualStatsTable[residualNumber_int][5] = value;
		}

		else if (strcmp(statisticName.c_str(), "pvF") == 0)
		{
			residualStatsTable[residualNumber_int][6] = value;
		}

		else if (strcmp(statisticName.c_str(), "pvADF") == 0)
		{
			residualStatsTable[residualNumber_int][7] = value;
		}
	}

	// Complete residual info
	double mod2Thr = (fiducialSizePx * thrFiducialDistance) * (fiducialSizePx * thrFiducialDistance);

	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		std::vector<CM> CM_fid;
		getCMbyFiducial(n, CM_fid);

		size_t numberCM = CM_fid.size();

		double avg;
		double std;
		size_t imagesOutOfRange = 0;
		size_t longestMisaliChain = 0;
		size_t misaliChain = 0;

		double sumResid = 0;
		double sumResid2 = 0;

		for (size_t i = 0; i < numberCM; i++)
		{
			double sum2 = CM_fid[i].residuals.x*CM_fid[i].residuals.x + CM_fid[i].residuals.y*CM_fid[i].residuals.y;
			sumResid2 += sum2;
			sumResid += sqrt(sum2);

			if (sum2 > mod2Thr)
			{
				imagesOutOfRange += 1;
				misaliChain += 1;
			}

			else if (misaliChain > longestMisaliChain)
			{
				longestMisaliChain = misaliChain;
				misaliChain = 0;
			}
		}

		#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
		std::cout << "n " << n << std::endl;
		std::cout << "numberCM " << numberCM << std::endl;
		std::cout << "sumResid " << sumResid << std::endl;
		std::cout << "sumResid2 " << sumResid2 << std::endl;
		std::cout << "longestMisaliChain " << longestMisaliChain << std::endl;
		std::cout << "imagesOutOfRange " << imagesOutOfRange << std::endl;
		#endif

		avg = sumResid / numberCM;
		std = sqrt(sumResid2 / numberCM - avg * avg);

		#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
		std::cout << "avg " << avg << std::endl;
		std::cout << "std " << std << std::endl;
		#endif

		residualStatsTable[n][0] = avg;
		residualStatsTable[n][1] = std;
		residualStatsTable[n][8] = imagesOutOfRange;
		residualStatsTable[n][9] = longestMisaliChain;
	}

	std::cout << " ----------------------------------------------- residualStatsTable" << std::endl;
	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		for (size_t i = 0; i < 10; i++)
		{
			std::cout << residualStatsTable[n][i] << " , ";
		}
		std::cout << "\n" ;
	}
	std::cout << " ----------------------------------------------- " << std::endl;

	// -- Save image information --
	residualStatsMd.read(fnStats_image);

	// Vector containing stats: avg, std, chArea, chPerim, pvBinX, pvBinY, pvF, ResidOutOfRange
	std::vector<double> imageStatsLine (8);
	std::vector<std::vector<double>> imageStatsTable;

	for (size_t i = 0; i < nSize; i++)
	{
		imageStatsTable.push_back(imageStatsLine);
	}
	
	// Save statistics in table
	for(size_t objId : residualStatsMd.ids())
	{
		residualStatsMd.getValue(MDL_IMAGE, statistic, objId);
		residualStatsMd.getValue(MDL_MIN, value, objId);

		statisticName = statistic.substr(statistic.find("_")+1);
		residualNumber = statistic.substr(0, statistic.find("_"));
		residualNumber_int = std::stoi(residualNumber);

		#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
		std::cout << "Reading object " << objId << " from metadata" << std::endl;
		std::cout << "statistic " << statistic << std::endl;
		std::cout << "value " << value << std::endl;
		std::cout << "statisticName " << statisticName << std::endl;
		std::cout << "residualNumber " << residualNumber << std::endl;
		std::cout << "residualNumber_int " << residualNumber_int << std::endl;
		#endif

		if (strcmp(statisticName.c_str(), "chArea") == 0)
		{
			imageStatsTable[residualNumber_int][2] = value;
		}

		else if (strcmp(statisticName.c_str(), "chPerim") == 0)
		{
			imageStatsTable[residualNumber_int][3] = value;
		}
		
		else if (strcmp(statisticName.c_str(), "pvBinX") == 0)
		{
			imageStatsTable[residualNumber_int][4] = value;
		}

		else if (strcmp(statisticName.c_str(), "pvBinY") == 0)
		{
			imageStatsTable[residualNumber_int][5] = value;
		}

		else if (strcmp(statisticName.c_str(), "pvF") == 0)
		{
			imageStatsTable[residualNumber_int][6] = value;
		}
	}

	// Complete image info
	for (size_t n = 0; n < nSize; n++)
	{
		std::vector<CM> CM_image;
		getCMbyImage(n, CM_image);

		size_t numberCM = CM_image.size();

		if (numberCM > 0)
		{
			double avg;
			double std;
			size_t residOutOfRange = 0;

			double sumResid = 0;
			double sumResid2 = 0;

			for (size_t i = 0; i < numberCM; i++)
			{
				double sum = CM_image[i].residuals.x*CM_image[i].residuals.x + CM_image[i].residuals.y*CM_image[i].residuals.y;
				sumResid2 += sum;
				sumResid += sqrt(sum);

				if (sumResid > mod2Thr)
				{
					residOutOfRange += 1;
				}
			}
			
			#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
			std::cout << "n " << n << std::endl;
			std::cout << "numberCM " << numberCM << std::endl;
			std::cout << "sumResid " << sumResid << std::endl;
			std::cout << "sumResid2 " << sumResid2 << std::endl;
			#endif

			avg = sumResid / numberCM;
			std = sqrt(sumResid2 / numberCM - avg * avg);

			#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
			std::cout << "avg " << avg << std::endl;
			std::cout << "std " << std << std::endl;
			#endif

			imageStatsTable[n][0] = avg;
			imageStatsTable[n][1] = std;
			imageStatsTable[n][7] = residOutOfRange;
		}
	}

	std::cout << " ----------------------------------------------- imageStatsTable" << std::endl;
	for (size_t n = 0; n < nSize; n++)
	{
		for (size_t i = 0; i < 8; i++)
		{
			std::cout << imageStatsTable[n][i] << " , ";
		}
		std::cout << "\n" ;
	}
	std::cout << " -----------------------------------------------" << std::endl;

	// -- Write output file for decision tree training --
	std::string decisionTreeStatsFileName_chain;
	std::string decisionTreeStatsFileName_image;

	size_t li = fnOut.find_last_of("\\/");
	std::string fileBaseName = fnOut.substr(0, li);
	li = fileBaseName.find_last_of("\\/");
	fileBaseName = fileBaseName.substr(0, li);

	decisionTreeStatsFileName_chain = fileBaseName + "/decisionTreeStats_chain.txt";
	decisionTreeStatsFileName_image = fileBaseName + "/decisionTreeStats_image.txt";

	#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
	std::cout << "fileBaseName " << fileBaseName << std::endl;
	std::cout << "decisionTreeStatsFileName_chain " << decisionTreeStatsFileName_chain << std::endl;
	std::cout << "decisionTreeStatsFileName_image " << decisionTreeStatsFileName_image << std::endl;
	#endif
	
	std::ofstream myfile;

	myfile.open (decisionTreeStatsFileName_chain, std::ios_base::app);
	for (size_t n = 0; n < residualStatsTable.size(); n++)  // Landmark decision tree stats
	{
		myfile << residualStatsTable[n][0];
		myfile << ", ";
		myfile << residualStatsTable[n][1];
		myfile << ", ";
		myfile << residualStatsTable[n][2];
		myfile << ", ";
		myfile << residualStatsTable[n][3];
		myfile << ", ";
		myfile << residualStatsTable[n][4];
		myfile << ", ";
		myfile << residualStatsTable[n][5];
		myfile << ", ";
		myfile << residualStatsTable[n][6];
		myfile << ", ";
		myfile << residualStatsTable[n][7];
		myfile << ", ";
		myfile << residualStatsTable[n][8];
		myfile << ", ";
		myfile << residualStatsTable[n][9];
		myfile << "\n";
	}
	myfile.close();

	myfile.open (decisionTreeStatsFileName_image, std::ios_base::app);
	for (size_t n = 0; n < imageStatsTable.size(); n++)  // Image decision tree stats
	{
		myfile << imageStatsTable[n][0];
		myfile << ", ";
		myfile << imageStatsTable[n][1];
		myfile << ", ";
		myfile << imageStatsTable[n][2];
		myfile << ", ";
		myfile << imageStatsTable[n][3];
		myfile << ", ";
		myfile << imageStatsTable[n][4];
		myfile << ", ";
		myfile << imageStatsTable[n][5];
		myfile << ", ";
		myfile << imageStatsTable[n][6];
		myfile << ", ";
		myfile << imageStatsTable[n][7];	
		myfile << "\n";
	}
	myfile.close();
	
	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
}


// --------------------------- I/O functions ----------------------------

void ProgTomoDetectMisalignmentTrajectory::writeOutputAlignmentReport()
{
	size_t lastindexInputTS = fnVol.find_last_of(":");
	std::string rawnameTS = fnVol.substr(0, lastindexInputTS);
	
	MetaDataVec md;
	FileName fn;
	size_t id;


	if(!globalAlignment)
	{
		for(size_t i = 0; i < nSize; i++)
		{
			fn.compose(i + FIRST_IMAGE, rawnameTS);
			id = md.addObject();

			// Tilt-image			
			md.setValue(MDL_IMAGE, fn, id);

			// Alignment
			md.setValue(MDL_ENABLED, -1, id);

			// Avg and STD of Mahalanobis distance
			md.setValue(MDL_COST, avgMahalanobisDistanceV[i], id);
			md.setValue(MDL_COST_PERCENTILE, stdMahalanobisDistanceV[i], id);
		}
	}
	else
	{
		for(size_t i = 0; i < localAlignment.size(); i++)
		{
			fn.compose(i + FIRST_IMAGE, rawnameTS);
			id = md.addObject();

			// Tilt-image			
			md.setValue(MDL_IMAGE, fn, id);

			// Alignment
			if(localAlignment[i])
			{
				md.setValue(MDL_ENABLED, 1, id);
			}
			else
			{
				md.setValue(MDL_ENABLED, -1, id);
			}

			// Avg and STD of Mahalanobis distance
			md.setValue(MDL_COST, avgMahalanobisDistanceV[i], id);
			md.setValue(MDL_COST_PERCENTILE, stdMahalanobisDistanceV[i], id);		}
	}

	md.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Alignment report saved at: " << fnOut << std::endl;
	#endif
}



void ProgTomoDetectMisalignmentTrajectory::writeOutputVCM()
{
	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < vCM.size(); i++)
	{
		id = md.addObject();
		// Tilt-series coordinates
		md.setValue(MDL_X, vCM[i].detectedCoordinate.x, id);
		md.setValue(MDL_Y, vCM[i].detectedCoordinate.y, id);
		md.setValue(MDL_Z, vCM[i].detectedCoordinate.z, id);

		// 3D coordinates
		md.setValue(MDL_XCOOR, (int)vCM[i].coordinate3d.x, id);
		md.setValue(MDL_YCOOR, (int)vCM[i].coordinate3d.y, id);
		md.setValue(MDL_ZCOOR, (int)vCM[i].coordinate3d.z, id);

		// Residual vector
		md.setValue(MDL_SHIFT_X, vCM[i].residuals.x, id);
		md.setValue(MDL_SHIFT_Y, vCM[i].residuals.y, id);

		// Mahalanobis distance
		md.setValue(MDL_COST, vCM[i].mahalanobisDistance, id);

		// ID
		md.setValue(MDL_FRAME_ID, vCM[i].id, id);
	}

	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string fnVCM;
    fnVCM = rawname + "/vCM.xmd";

	md.write(fnVCM);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Vector coordinates model metadata saved at: " << fnVCM << std::endl;
	#endif
}



// --------------------------- MAIN ----------------------------------
void ProgTomoDetectMisalignmentTrajectory::run()
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
	coordinates3D = lmDetector.coordinates3D;

	MultidimArray<int> proyectedCoordinates;
	proyectedCoordinates.initZeros(ySize, xSize);

	for(size_t n; n < coordinates3D.size(); n++)
	{
		fillImageLandmark(proyectedCoordinates, (int)coordinates3D[n].x, (int)coordinates3D[n].y, 1);
	}

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindexBis = fnOut.find_last_of("\\/");
	std::string rawnameBis = fnOut.substr(0, lastindexBis);
	std::string outputFileNameFilteredVolumeBis;
    outputFileNameFilteredVolumeBis = rawnameBis + "/ts_proyected.mrc";

	Image<int> saveImageBis;
	saveImageBis() = proyectedCoordinates;
	saveImageBis.write(outputFileNameFilteredVolumeBis);
	#endif

	calculateResidualVectors();
	pruneResidualVectors();

	writeOutputVCM();

	adjustCoordinatesCosineStreching();

	detectMisalignmentFromResidualsMahalanobis();

	#ifdef GENERATE_RESIDUAL_STATISTICS
	generateResidualStatiscticsFile();
	#endif

	writeOutputAlignmentReport();

	auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1); 	// Getting number of milliseconds as an integer
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}


// --------------------------- UTILS functions ----------------------------

void ProgTomoDetectMisalignmentTrajectory::fillImageLandmark(MultidimArray<int> &proyectedImage, int x, int y, int value)
{
	int distance = fiducialSizePx/6;

	for (int i = -distance; i <= distance; i++)
	{
		for (int j = -distance; j <= distance; j++)
		{
			if (j + y > 0 && i + x  > 0 && j + y < ySize && i + x < xSize && i*i+j*j <= distance*distance)
			{
				DIRECT_A2D_ELEM(proyectedImage, (j + y), (i + x)) = value;
			}
		}
	}
}


void ProgTomoDetectMisalignmentTrajectory::adjustCoordinatesCosineStreching()
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

			#ifdef DEBUG_COORDS_CS
			std::cout << "------------------------------------------------------------------------------------------------" << std::endl;

			std::cout << "xTA=" << xTA << std::endl;
			
			std::cout << "cm.detectedCoordinate.x=" << cm.detectedCoordinate.x << std::endl;
			std::cout << "cm.detectedCoordinate.y=" << cm.detectedCoordinate.y << std::endl;
			std::cout << "cm.detectedCoordinate.z=" << cm.detectedCoordinate.z << std::endl; 

			std::cout << "cm.coordinate3d.x=" << cm.coordinate3d.x << std::endl;
			std::cout << "cm.coordinate3d.y=" << cm.coordinate3d.y << std::endl;
			std::cout << "cm.coordinate3d.z=" << cm.coordinate3d.z << std::endl; 
			
			std::cout << "(int)cm.detectedCoordinate.x=" << (int)cm.detectedCoordinate.x << std::endl;
			std::cout << "(int)cm.detectedCoordinate.y=" << (int)cm.detectedCoordinate.y << std::endl;
			std::cout << "(int)cm.detectedCoordinate.z=" << (int)cm.detectedCoordinate.z << std::endl;

			std::cout << "(int)cm.coordinate3d.x=" << (int)cm.coordinate3d.x << std::endl;
			std::cout << "(int)cm.coordinate3d.y=" << (int)cm.coordinate3d.y << std::endl;
			std::cout << "(int)cm.coordinate3d.z=" << (int)cm.coordinate3d.z << std::endl; 

			std::cout << "tiltAngle=" << tiltAngle << std::endl;
			std::cout << "cos(tiltAngle)=" << cos(tiltAngle) << std::endl;
			std::cout << "sin(tiltAngle)=" << sin(tiltAngle) << std::endl;

			std::cout << "Xo=" << (int) (((cm.detectedCoordinate.x-xTA)-((cm.coordinate3d.z)*sin(tiltAngle))/cos(tiltAngle))+xTA) << std::endl;
			#endif

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


Matrix2D<double> ProgTomoDetectMisalignmentTrajectory::getProjectionMatrix(double tiltAngle)
{
	double cosTiltAngle = cos(tiltAngle * PI/180.0);
	double sinTiltAngle = sin(tiltAngle * PI/180.0);

	Matrix2D<double> projectionMatrix(3,3);

	MAT_ELEM(projectionMatrix, 0, 0) = cosTiltAngle;
	// MAT_ELEM(projectionMatrix, 0, 1) = 0;
	MAT_ELEM(projectionMatrix, 0, 2) = sinTiltAngle;
	// MAT_ELEM(projectionMatrix, 1, 0) = 0;
	MAT_ELEM(projectionMatrix, 1, 1) = 1;
	// MAT_ELEM(projectionMatrix, 1, 2) = 0;
	MAT_ELEM(projectionMatrix, 2, 0) = -sinTiltAngle;
	// MAT_ELEM(projectionMatrix, 2, 1) = 0;
	MAT_ELEM(projectionMatrix, 2, 2) = cosTiltAngle;

	return projectionMatrix;
}


std::vector<Point2D<double>> ProgTomoDetectMisalignmentTrajectory::getCoordinatesInSlice(size_t slice)
{
	std::vector<Point2D<double>> coordinatesInSlice;
	Point2D<double> coordinate(0,0);

	for(size_t n = 0; n < coordinates3D.size(); n++)
	{
		if(slice == coordinates3D[n].z)
		{
			coordinate.x = coordinates3D[n].x;
			coordinate.y = coordinates3D[n].y;
			coordinatesInSlice.push_back(coordinate);
		}
	}

	return coordinatesInSlice;
}


void ProgTomoDetectMisalignmentTrajectory::getCMFromCoordinate(int x, int y, int z, std::vector<CM> &vCMc)
{
	for (size_t i = 0; i < vCM.size(); i++)
	{
		auto cm = vCM[i];
		
		if (cm.coordinate3d.x==x && cm.coordinate3d.y==y && cm.coordinate3d.z==z)
		{
			#ifdef DEBUG_COORDS_CS
			std::cout << "ADDED!!!!!! " <<i<<std::endl;
			#endif

			vCMc.push_back(cm);
		}
	}
	
	#ifdef DEBUG_COORDS_CS
	std::cout << "vCMc.size()" << vCMc.size() << std::endl;
	#endif
}


bool ProgTomoDetectMisalignmentTrajectory::checkProjectedCoordinateInInterpolationEdges(Matrix1D<double> projectedCoordinate, size_t slice)
{
	std::vector<Point2D<int>> interpolationLimits = lmDetector.interpolationLimitsVector[slice];

	int x = (int)(XX(projectedCoordinate));
	int y = (int)(YY(projectedCoordinate));

	if (x >= interpolationLimits[y].x || x <= interpolationLimits[y].y)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void ProgTomoDetectMisalignmentTrajectory::getCMbyFiducial(size_t fiducialNumber, std::vector<CM> &vCM_fiducial)
{
	for (size_t i = 0; i < vCM.size(); i++)
	{
		if (vCM[i].id == fiducialNumber)
		{
			vCM_fiducial.push_back(vCM[i]);
		}
	}
}



void ProgTomoDetectMisalignmentTrajectory::getCMbyImage(size_t tiltImageNumber, std::vector<CM> &vCM_image)
{
	for (size_t i = 0; i < vCM.size(); i++)
	{
		if ((size_t)(vCM[i].detectedCoordinate.z) == tiltImageNumber)
		{
			vCM_image.push_back(vCM[i]);
		}
	}
}



// --------------------------- UNUSED functions ----------------------------

// std::vector<size_t> ProgTomoDetectMisalignmentTrajectory::getRandomIndexes(size_t size)
// {
// 	std::vector<size_t> indexes;
// 	size_t randomIndex;

// 	randomIndex = rand() % size;

// 	indexes.push_back(randomIndex);

// 	while (indexes.size() != 3)
// 	{
// 		randomIndex = rand() % size;

// 		for(size_t n = 0; n < indexes.size(); n++)
// 		{
// 			if(indexes[n] != randomIndex)
// 			{
// 				indexes.push_back(randomIndex);
// 				break;
// 			}
// 		}
// 	}
	
// 	return indexes;
// }



// bool ProgTomoDetectMisalignmentTrajectory::detectGlobalAlignmentPoisson(std::vector<int> counterLinesOfLandmarkAppearance, std::vector<size_t> chainIndexesY)
// {
// 	// float totalLM = coordinates3D.size();
// 	float totalLM = 0;
// 	float totalChainLM = 0;
// 	float totalIndexes = chainIndexesY.size();
// 	float top10LM = 0;

// 	// We need to recalculate the number of total landmark considering those HCC counted more than once due to the sliding window effect
// 	for (size_t i = 0; i < counterLinesOfLandmarkAppearance.size(); i++)
// 	{
// 		totalLM += counterLinesOfLandmarkAppearance[i];
// 	}

// 	for (size_t i = 0; i < totalIndexes; i++)
// 	{
// 		totalChainLM += counterLinesOfLandmarkAppearance[(int)chainIndexesY[i]];
// 	}

// 	sort(counterLinesOfLandmarkAppearance.begin(), counterLinesOfLandmarkAppearance.end(), std::greater<int>());

// 	for (size_t i = 0; i < 10; i++)
// 	{
// 		top10LM += counterLinesOfLandmarkAppearance[i];
// 	}

// 	// Thresholds calculation
// 	float top10Chain = 100 * (top10LM / totalLM); 		// Compare to thrTop10Chain
// 	float lmChain = 100 * (totalChainLM / (totalLM)); 	// Compare to thrLMChain

// 	// Thresholds comparison
// 	bool top10ChainBool = top10Chain < thrTop10Chain;
// 	bool lmChainBool = lmChain < thrLMChain;

// 	std::cout << "Global misalignment detection parameters:" << std::endl;
// 	std::cout << "Total number of landmarks: " << totalLM << std::endl;
// 	std::cout << "Total number of landmarks belonging to the selected chains: " << totalChainLM << std::endl;
// 	std::cout << "Total number of landmarks belonging to the top 10 most populated indexes: " << top10LM << std::endl;

// 	std::cout << "Precentage of LM belonging to the top 10 populated chains respecto to the total number of LM: " << top10Chain << std::endl;
// 	std::cout << "Percentage of number LM belonging to the selected chains respect to the number of populated lines and the total number of LM: " << lmChain << std::endl;

// 	std::cout << "Compare top10Chain < thrTop10Chain (" << top10Chain << "<" << thrTop10Chain << "): " << top10ChainBool << std::endl;
// 	std::cout << "Compare lmChain < thrLMChain (" << lmChain << "<" << thrLMChain << "): " << lmChainBool << std::endl;

// 	if(top10ChainBool || lmChainBool)
// 	{
// 		#ifdef VERBOSE_OUTPUT
// 		std::cout << "GLOBAL MISALIGNMENT DETECTED IN TILT-SERIES" << std::endl;
// 		std::cout << "Compare top10Chain < thrTop10Chain (" << top10Chain << "<" << thrTop10Chain << "): " << top10ChainBool << std::endl;
// 		std::cout << "Compare lmChain < thrLMChain (" << lmChain << "<" << thrLMChain << "): " << lmChainBool << std::endl;
// 		#endif

// 		return false;
// 	}
// 	else
// 	{
// 		return true;
// 	}
// }



// int ProgTomoDetectMisalignmentTrajectory::calculateTiltAxisIntersection(Point2D<double> p1, Point2D<double> p2)
// {
// 	// Eccuation of a line given 2 points:
// 	// y = [(x-x1)(y2-y1) / (x2-x1)] - y1
// 	// x = [(y-y1)(x2-x1) / (y2-y1)] - x1

// 	// We calculate the x coordinate at wich the line intersect the tilt axis, given by:
// 	// y = ySize / 2

// 	// We obtain the x coordinate of intersection by substitution:
// 	// x = [(ySize/2-y1)(x2-x1) / (y2-y1)] - x1

// 	std::cout << "p1=(" << p1.x << ", " << p1.y << ")" << std::endl;
// 	std::cout << "p2=(" << p2.x << ", " << p2.y << ")" << std::endl;
// 	std::cout << "xSize/2=" << xSize/2 << std::endl;
// 	std::cout << "ySize/2=" << ySize/2 << std::endl;

// 	// std::cout << (int)((ySize/2-p1.y)*(p2.x-p1.x) / (p2.y-p1.y)) + p1.x << std::endl;
// 	std::cout << (int)(((xSize/2)-p1.x)*(p2.y-p1.y) / (p2.x-p1.x)) + p1.y << std::endl;

// 	// return (int)((ySize/2-p1.y)*(p2.x-p1.x) / (p2.y-p1.y)) + p1.x;
// 	return (int)(((xSize/2)-p1.x)*(p2.y-p1.y) / (p2.x-p1.x)) + p1.y;
// }



// bool ProgTomoDetectMisalignmentTrajectory::detectGlobalMisalignment()
// {
// 	MultidimArray<double> tiltAxisIntersection;

// 	// tiltAxisIntersection.initZeros(ySize, nSize);

// 	tiltAxisIntersection.initZeros(ySize);

// 	// Extract alignment information of e
// 	for (size_t tiIndex = 0; tiIndex < nSize; tiIndex++)
// 	{
// 		std::vector<Point2D<double>> coordinatesInSlice;

// 		coordinatesInSlice = getCoordinatesInSlice(tiIndex);

// 		std::vector<std::vector<Point2D<double>>> splittedCoords = splitCoordinatesInHalfImage(coordinatesInSlice);

// 		std::vector<Point2D<double>> leftSide = splittedCoords[0];
// 		std::vector<Point2D<double>> rightSide = splittedCoords[1];

// 		int smallArray = (leftSide.size()<rightSide.size()) ? leftSide.size() : rightSide.size();

// 		// Calculate the intersection of the pair of coordinates with the tilt axis
// 		for (size_t i = 0; i < smallArray; i++)
// 		{

// 			Point2D<double> p1 = leftSide[i];
// 			Point2D<double> p2 = rightSide[i];

// 			int intersectionIndex =  calculateTiltAxisIntersection(p1, p2);

// 			tiltAxisIntersection[intersectionIndex] += 1;
				
// 		}

// 		// // Calculate the intersection of the pair of coordinates with the tilt axis
// 		// for (size_t i = 0; i < coordinatesInSlice.size(); i++)
// 		// {
// 		// 	for (size_t j = 0; j < coordinatesInSlice.size(); j++)
// 		// 	{
// 		// 		Point2D<double> p1 = coordinatesInSlice[i];
// 		// 		Point2D<double> p2 = coordinatesInSlice[j];

// 		// 		// Do not compare very close coordinates
// 		// 		int distance2 = (p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y);

// 		// 		if(distance2 > (normDim*0.5)*(normDim*0.5))
// 		// 		{
// 		// 			int intersectionIndex =  calculateTiltAxisIntersection(p1, p2);
// 		// 			if (intersectionIndex > 0 && intersectionIndex < ySize)
// 		// 			{
// 		// 				// tiltAxisIntersection[intersectionIndex, tiIndex] += 1;
// 		// 				tiltAxisIntersection[intersectionIndex] += 1;
// 		// 			}
// 		// 		}
// 		// 	}
// 		// }
// 	}

// 	// for (size_t i = 0; i < nSize; i++)
// 	// {
// 	// std::cout << "line" << i <<"-->[" ;

// 	// 	for (size_t j = 0; j < ySize; j++)
// 	// 	{
// 	// 		std::cout << tiltAxisIntersection[j,i] << " ";
// 	// 	}
	
// 	// std::cout << "]" <<  std::endl;
// 	// }

// 	std::cout << "[" ;

// 	for (size_t i = 0; i < ySize; i++)
// 	{
// 		std::cout << tiltAxisIntersection[i] << " ";
// 	}

// 	std::cout << "]" << std::endl;

// 	return true;
	
// }



// std::vector<std::vector<Point2D<double>>> ProgTomoDetectMisalignmentTrajectory::splitCoordinatesInHalfImage(std::vector<Point2D<double>> inCoords)
// {
// 	std::vector<std::vector<Point2D<double>>> splittedCoodinates;

// 	std::vector<Point2D<double>> leftSide;
// 	std::vector<Point2D<double>> rightSide;

// 	for (size_t i = 0; i < inCoords.size(); i++)
// 	{
// 		Point2D<double> coord = inCoords[i];

// 		if(coord.x > xSize/2)
// 		{
// 			rightSide.push_back(coord);
// 		}else
// 		{
// 			leftSide.push_back(coord);
// 		}
// 	}

// 	// Struct to sort 3D coordinates (point3D class) by its x component
// 	struct SortByY
// 	{
// 		bool operator() ( const Point2D<double>& L, const Point2D<double>& R) { return L.y < R.y; };
// 	};

// 	std::sort(leftSide.begin(), leftSide.end(), SortByY());

// 	std::sort(rightSide.begin(), rightSide.end(), SortByY());

// 	splittedCoodinates.push_back(leftSide);
// 	splittedCoodinates.push_back(rightSide);

// 	return splittedCoodinates;	
// }



// void ProgTomoDetectMisalignmentTrajectory::detectGlobalMisalignment()
// {

// 	// MAKE GLOBAL THRESHOLD FOR MINIMUM ANGLE  esto puede que necesite ir en ang y pasar a px
// 	float thrChainDeviation = 5;

// 	// Struct to sort 3D coordinates (point3D class) by its x component
// 	struct SortByX
// 	{
// 		bool operator() const (Point3D const & L, Point3D const & R) { return L.x < R.x; }
// 	};

//     std::vector<Point3D<double>> sortXcoordinates3D = coordinates3D;
// 	std::sort(sortXcoordinates3D.begin(), sortXcoordinates3D.end(),b   SortByX());

// 	// Vector 
// 	std::vector<size_t> coordinatesIndexID (sortXcoordinates3D, 0);
// 	size_t indexID = 1;

// 	for (size_t i = 0; i < sortXcoordinates3D.size(); i++)
// 	{
// 		Point3D referenceCoord3D = sortXcoordinates3D[i];

// 		for (size_t j = 0; j < sortXcoordinates3D.size(); j++)
// 		{
// 			Point3D coord3D = sortXcoordinates3D[j];

// 			if (coordinatesIndexID[i] == 0)
// 			{
// 				float deviation = abs(coord3D.y - referenceCoord3D.y);

// 				if (deviation < thrChainDeviation)
// 				{
// 					chain.push_back(coord3D);		
// 				}
// 			}
// 		}
// 	}
// }



// void ProgTomoDetectMisalignmentTrajectory::writeOutputResidualVectors()
// {
// 	MetaDataVec md;
// 	size_t id;

// 	for(size_t i = 0; i < residualX.size(); i++)
// 	{
// 		id = md.addObject();
// 		md.setValue(MDL_X, residualX[i], id);
// 		md.setValue(MDL_Y, residualY[i], id);
// 		md.setValue(MDL_XCOOR, residualCoordinateX[i], id);
// 		md.setValue(MDL_YCOOR, residualCoordinateY[i], id);
// 		md.setValue(MDL_ZCOOR, residualCoordinateZ[i], id);
// 	}

// 	size_t lastindex = fnOut.find_last_of("\\/");
// 	std::string rawname = fnOut.substr(0, lastindex);
// 	std::string fnOutResiduals;
//     fnOutResiduals = rawname + "/residuals2d.xmd";

// 	md.write(fnOutResiduals);
	
// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Residuals metadata saved at: " << fnOutResiduals << std::endl;
// 	#endif
// }



// void ProgTomoDetectMisalignmentTrajectory::generatePahntomVolume()
// {
// 	std::vector<std::vector<int>> coords = {{768,1080,290}, {256, 576, 150}, {768, 576, 150}, {256, 1080, 150}, {256, 720, 150}, {768, 720, 150}, {512, 1080, 110}, {512, 576, 10}};
// 	MultidimArray<double> tmpMap;
// 	tmpMap.initZeros(300,1440,1024);
// 	tmpMap.initConstant(0);

// 	for (size_t i = 0; i < coords.size(); i++)
// 	{
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2], coords[i][1], coords[i][0]) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2], coords[i][1], coords[i][0]+1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2], coords[i][1], coords[i][0]-1) = 1;

// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]+1, coords[i][1], coords[i][0]) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2], coords[i][1]+1, coords[i][0]) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]+1, coords[i][1]+1, coords[i][0]) = 1;

// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]+1, coords[i][1], coords[i][0]+1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2], coords[i][1]+1, coords[i][0]+1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]+1, coords[i][1]+1, coords[i][0]+1) = 1;

// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]+1, coords[i][1], coords[i][0]-1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2], coords[i][1]+1, coords[i][0]-1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]+1, coords[i][1]+1, coords[i][0]-1) = 1;

// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]-1, coords[i][1], coords[i][0]) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2], coords[i][1]-1, coords[i][0]) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]-1, coords[i][1]-1, coords[i][0]) = 1;

// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]-1, coords[i][1], coords[i][0]+1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2], coords[i][1]-1, coords[i][0]+1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]-1, coords[i][1]-1, coords[i][0]+1) = 1;


// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]-1, coords[i][1], coords[i][0]-1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2], coords[i][1]-1, coords[i][0]-1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]-1, coords[i][1]-1, coords[i][0]-1) = 1;


// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]-1, coords[i][1]+1, coords[i][0]) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]+1, coords[i][1]-1, coords[i][0]) = 1;

// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]-1, coords[i][1]+1, coords[i][0]+1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]+1, coords[i][1]-1, coords[i][0]+1) = 1;


// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]-1, coords[i][1]+1, coords[i][0]-1) = 1;
// 		DIRECT_A3D_ELEM(tmpMap, coords[i][2]+1, coords[i][1]-1, coords[i][0]-1) = 1;	
// 		}

// 	Image<double> saveImage3;
// 	saveImage3() = tmpMap;
// 	saveImage3.write(fnOut.substr(0, fnOut.find_last_of("\\/")) + "/test_map.mrc");
// }



// void ProgTomoDetectMisalignmentTrajectory::binomialTestOnResiduals()
// {
// 	// Sign test (binomial distribution)
// 	size_t x_pos = 0;
// 	size_t y_pos = 0;
// 	size_t nx_sample = residuals.size();
// 	size_t ny_sample = residuals.size();

// 	double p = 0.5;
// 	double binom_x;
// 	double binom_y;

// 	for (size_t i = 0; i < residuals.size(); i++)
// 	{
// 		if (residuals[i].x > 0)
// 		{
// 			x_pos++;
// 		}
// 		else if (residuals[i].x == 0)
// 		{
// 			nx_sample--;
// 		}

// 		if (residuals[i].y > 0)
// 		{
// 			y_pos++;
// 		}
// 		else if (residuals[i].y == 0)
// 		{
// 			ny_sample--;
// 		}	
// 	}

// 	size_t x_pos_fact;
// 	size_t nx_sample_fact;
// 	size_t nkx_sample_fact;
// 	size_t y_pos_fact;
// 	size_t ny_sample_fact;
// 	size_t nky_sample_fact;

// 	factorial(x_pos, x_pos_fact);
// 	factorial(nx_sample, nx_sample_fact);
// 	factorial(nx_sample - x_pos, nkx_sample_fact);
// 	factorial(y_pos, y_pos_fact);
// 	factorial(ny_sample, ny_sample_fact);
// 	factorial(ny_sample - y_pos, nky_sample_fact);

// 	binom_x =  (nx_sample_fact/(x_pos_fact*nkx_sample_fact))*pow(p, x_pos)*pow(1-p, nx_sample-x_pos);
// 	binom_y =  (ny_sample_fact/(y_pos_fact*nky_sample_fact))*pow(p, y_pos)*pow(1-p, ny_sample-y_pos);

// 	std::cout << "binom_x "  << binom_x << std::endl;
// 	std::cout << "binom_y "  << binom_y << std::endl;
// }



// void ProgTomoDetectMisalignmentTrajectory::eigenAutocorrMatrix()
// {
// 	// Eigen values ratio of the residual autocorrelation matrix

// 	double x;
// 	double y;
// 	double x2;
// 	double y2;
// 	double xy;
// 	double radius;
// 	double sumRadius = 0;
// 	double a = 0;
// 	double b = 0;
// 	double c = 0;
// 	double d = 0;
// 	double root;
// 	double lambda1;
// 	double lambda2;
// 	double lambdaRatio;

// 	for (size_t i = 0; i < residuals.size(); i++)
// 	{
// 		x = residuals[i].x;
// 		y = residuals[i].y;

// 		x2 = x*x;
// 		y2 = y*y;
// 		xy = x*y;

// 		radius = sqrt(x2 + y2)	;

// 		a += x2 * radius;
// 		b += xy * radius;
// 		c += xy * radius;
// 		d += y2 * radius;
		
// 		sumRadius += radius;
// 	}

// 	a /= sumRadius;
// 	b /= sumRadius;
// 	c /= sumRadius;
// 	d /= sumRadius;

// 	root = sqrt((a+d)*(a+d) - 4*(-a*d-c*b));

// 	lambda1 = (-(a+d)+root)/2;
// 	lambda2 = (-(a+d)-root)/2;

// 	lambdaRatio = (lambda1>lambda2) ? lambda2/lambda1 : lambda1/lambda2;

// 	std::cout << "lambdaRatio=" << lambdaRatio << std::endl;

// 	return true;
// }



// void ProgTomoDetectMisalignmentTrajectory::calculateAffinity(MetaDataVec &inputCoordMd)
// {
// 	// TODO: homogeneizar PointXD y MatrixXD
// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Calculating residual vectors" << std::endl;
// 	#endif

// 	size_t objId;
// 	size_t minIndex;
// 	double tiltAngle;
// 	double distance;
// 	double minDistance;

// 	int goldBeadX, goldBeadY, goldBeadZ;

// 	Matrix2D<double> projectionMatrix;
// 	Matrix1D<double> goldBead3d;
// 	Matrix1D<double> projectedGoldBead;

// 	std::vector<Point2D<double>> coordinatesInSlice;

// 	// Matrix2D<double> A_alignment;
// 	// Matrix1D<double> T_alignment;
// 	// Matrix2D<double> invW_alignment;
// 	// Matrix2D<double> alignment_matrix;


// 	goldBead3d.initZeros(3);

// 	// Iterate through every tilt-image
// 	for(size_t n = 0; n<tiltAngles.size(); n++)
// 	{	
// 		std::vector<size_t> randomIndexes = getRandomIndexes(projectedGoldBead.size());

// 		for(size_t i = 0; i < coordinatesInSlice.size(); i ++)
// 		{
// 			for(size_t j = 0; j < coordinatesInSlice.size(); j ++)
// 			{
// 				for(size_t k = 0; k < coordinatesInSlice.size(); k ++)
// 				{
// 					def_affinity(XX(projectedGoldBeads[randomIndexes[0]]),
// 									YY(projectedGoldBeads[randomIndexes[0]]),
// 									XX(projectedGoldBeads[randomIndexes[1]]),
// 									YY(projectedGoldBeads[randomIndexes[1]]),
// 									XX(projectedGoldBeads[randomIndexes[2]]),
// 									YY(projectedGoldBeads[randomIndexes[2]]),
// 									XX(coordinatesInSlice[i]),
// 									YY(coordinatesInSlice[i]),
// 									XX(coordinatesInSlice[j]),
// 									YY(coordinatesInSlice[j]),
// 									XX(coordinatesInSlice[k]),
// 									YY(coordinatesInSlice[k]),
// 									A_alignment,
// 									T_alignment,
// 									invW_alignment)

// 					MAT_ELEM(alignment_matrix, 0, 0) = MAT_ELEM(A_alignment, 0, 0);
// 					MAT_ELEM(alignment_matrix, 0, 1) = MAT_ELEM(A_alignment, 0, 1);
// 					MAT_ELEM(alignment_matrix, 1, 0) = MAT_ELEM(A_alignment, 1, 0);
// 					MAT_ELEM(alignment_matrix, 1, 1) = MAT_ELEM(A_alignment, 1, 1);
// 					MAT_ELEM(alignment_matrix, 0, 2) = XX(T_alignment);
// 					MAT_ELEM(alignment_matrix, 1, 2) = YY(T_alignment);
// 					MAT_ELEM(alignment_matrix, 2, 0) = 0;
// 					MAT_ELEM(alignment_matrix, 2, 1) = 0;
// 					MAT_ELEM(alignment_matrix, 2, 2) = 1;
// 				}
// 			}
// 		}

// 			#ifdef DEBUG_RESID
// 			std::cout << XX(goldBead3d) << " " << YY(goldBead3d) << " " << ZZ(goldBead3d) << std::endl;
// 			std::cout << XX(projectedGoldBead) << " " << YY(projectedGoldBead) << " " << ZZ(projectedGoldBead) << std::endl;
// 			std::cout << "------------------------------------"<<std::endl;
// 			#endif	
// }



// void ProgTomoDetectMisalignmentTrajectory::calculateAffinity(MetaDataVec &inputCoordMd)
// 	int goldBeadX;
// 	int goldBeadY;
// 	int goldBeadZ;

// 	for(size_t objId : inputCoordMd.ids())
// 	{
// 		inputCoordMd.getValue(MDL_XCOOR, goldBeadX, objId);
// 		inputCoordMd.getValue(MDL_YCOOR, goldBeadY, objId);
// 		inputCoordMd.getValue(MDL_ZCOOR, goldBeadZ, objId);

// 		#ifdef DEBUG_RESIDUAL_ANALYSIS
// 		std::cout << "Analysis of residuals corresponding to coordinate 3D: x " << goldBeadX << " y " << goldBeadY << " z " << goldBeadZ << std::endl;
// 		#endif

// 	    std::vector<CM> vCMc;
// 		getCMFromCoordinate(goldBeadX, goldBeadY, goldBeadZ, vCMc);

// 		std::vector<Point2D<double>> residuals;
// 		for (size_t i = 0; i < vCMc.size(); i++)
// 		{
// 			residuals.push_back(vCMc[i].residuals);
// 		}
	
// 		double centroidX;
// 		double centroidY;
// 		double distance;
// 		double maxDistance = 0;
// 		double totalDistance = 0;
// 		double chPerimeterer = 0;
// 		double chArea = 0;

// 		std::vector<Point2D<double>> hull;

// 		for (size_t i = 0; i < residuals.size(); i++)
// 		{

// 			#ifdef DEBUG_RESIDUAL_ANALYSIS
// 			std::cout << residuals[i].x << "  " << residuals[i].y << std::endl;
// 			#endif

// 			distance = sqrt((residuals[i].x * residuals[i].x) + (residuals[i].y * residuals[i].y));

// 			// Total distance
// 			totalDistance += distance;

// 			// Maximum distance
// 			if (distance > maxDistance)
// 			{
// 				maxDistance = distance;
// 			}
// 		}

// 		std::cout << "totalDistance "  << totalDistance << std::endl;
// 		std::cout << "maxDistance "  << maxDistance << std::endl;


// 		// Convex Hull
// 		auto p2dVector = residuals;
// 		auto remainingP2d = residuals;
// 		Point2D<double> p2d{0, 0};
// 		Point2D<double> p2d_it{0, 0};
// 		Point2D<double> minX_p2d{MAXDOUBLE, 0};

// 		// Get minimum x component element
// 		for (size_t i = 0; i < p2dVector.size(); i++)
// 		{
// 			if (p2dVector[i].x < minX_p2d.x)
// 			{
// 				minX_p2d = p2dVector[i];
// 			}
// 		}

// 		// struct SortByX
// 		// {
// 		// 	bool operator() (const Point2D<double>& lp2d, const Point2D<double>& rp2d) 	{return lp2d.x < rp2d.x;};
// 		// };

// 		// minX_p2d = std::min_element(p2dVector.begin(), p2dVector.end(), SortByX());

// 		// bool cmp(const Point2D<double>& a, const Point2D<double>& b)
// 		// {
// 		// 	return a.x < b.x;
// 		// }

// 		// minX_p2d = std::min_element(p2dVector.begin(), p2dVector.end(), cmp);

// 		// minX_p2d = std::min_element(p2dVector.begin(), p2dVector.end(), [](Point2D const& l, Point2D const& r) {return l.x < r.x;});

// 		hull.push_back(minX_p2d);

// 		#ifdef DEBUG_RESIDUAL_ANALYSIS
// 		std::cout << " minX_p2d" << minX_p2d.x << " " << minX_p2d.y << std::endl;
// 		std::cout << "p2dVector.size() " << p2dVector.size() << std::endl;
// 		std::cout << "remainingP2d.size() " << remainingP2d.size() << std::endl;
// 		#endif

// 		while (p2dVector.size()>0)
// 		{
// 			p2d = p2dVector[0];

// 			while (remainingP2d.size()>0)
// 			{
// 				p2d_it = remainingP2d[0];

// 				double angle = atan2(p2d_it.y-p2d.y, p2d_it.x-p2d.x) - atan2(hull[hull.size()].y-p2d.y, hull[hull.size()].x-p2d.x);

// 				if (angle<0)
// 				{
// 					angle += 2*PI;
// 				}

// 				if (angle < PI)
// 				{
// 					remainingP2d.erase(remainingP2d.begin());
// 				}
// 				else
// 				{
// 					p2d = p2d_it;
// 					remainingP2d.erase(remainingP2d.begin());
// 				}	
// 			}

// 			if (p2d.x==minX_p2d.x && p2d.y==minX_p2d.y)
// 			{
// 				break;
// 			}

// 			hull.push_back(p2d);

// 			p2dVector.erase(p2dVector.begin());
// 			remainingP2d = p2dVector;
// 		}
		

// 		// CH-Perimeter
// 		int shiftIndex;

// 		for (size_t i = 0; i < hull.size(); i++)
// 		{
// 			shiftIndex = (i+1) % hull.size();
// 			chPerimeterer += sqrt((hull[i].x-hull[shiftIndex].x)*(hull[i].x-hull[shiftIndex].x)+
// 								(hull[i].y-hull[shiftIndex].y)*(hull[i].y-hull[shiftIndex].y));
// 		}

// 		std::cout << "chPerimeterer "  << chPerimeterer << std::endl;
		

// 		// CH-Area
// 		double sum1 = 0;
// 		double sum2 = 0;

// 		for (size_t i = 0; i < hull.size(); i++)
// 		{
// 			shiftIndex = (i+1) % hull.size();
// 			sum1 += hull[i].x * hull[shiftIndex].y;	
// 			sum2 += hull[shiftIndex].x * hull[i].y;	
// 		}

// 		chArea = abs(0.5 * (sum1 - sum2));

// 		std::cout << "chArea "  << chArea << std::endl;
// 	}
// }



// void ProgTomoDetectMisalignmentTrajectory::factorial(size_t base, size_t fact)
// {
// 	if (base == 0 || base == 1)
// 	{
// 		fact = 1;
// 	}
// 	else
// 	{
// 		for (size_t i = base; i > 0; i--)
// 		{
// 			fact *= i;
// 		}
// 	}
// }


// void ProgTomoDetectMisalignmentTrajectory::smoothInterpolationEdges()  // NOT IMPLEMENTED 
// {
	// // Apply smoothing kernel to interpolation edges:
	// //     1/16 1/8 1/16
	// // k = 1/8  1/4 1/8
	// //     1/16 1/8 1/16

	// tmpImage = tiltImage;

	// int jj;
	
	// double m1 = (double)(-y1)/(x1);
	// double m2 = (double)(-y2)/(x2-(double)xSize);
	// double m3 = (double)(y3-(double)ySize)/(-x3);
	// double m4 = (double)(y4-(double)ySize)/((double)xSize-x4);

	// std::vector<double> corners{DIRECT_A2D_ELEM(tiltImage, 0, 0),
	// 							DIRECT_A2D_ELEM(tiltImage, 0, xSize-1),
	// 							DIRECT_A2D_ELEM(tiltImage, ySize-1, 0),
	// 							DIRECT_A2D_ELEM(tiltImage, ySize-1, xSize-1)};

	// sort(corners.begin(), corners.end(), std::greater<double>());

	// double backgroundValue = (corners[1]+corners[2])/2;  // Background value as the median of the corners

	// std::cout<< "m1: " << m1<<std::endl;
	// std::cout<< "m2: " << m2<<std::endl;
	// std::cout<< "m3: " << m3<<std::endl;
	// std::cout<< "m4: " << m4<<std::endl;

	// // tmpImage.initZeros();

	// // Draw horizontal line
	// for (int ii = 1; ii < xSize-1; ii++)
	// {
	// 	DIRECT_A2D_ELEM(tmpImage, 0 ,ii) = backgroundValue;
	// 	DIRECT_A2D_ELEM(tmpImage, ySize-1 ,ii) = backgroundValue;

	// 	// Interpolation line 1
	// 	jj = (int)(m1*(ii)+y1);

	// 	if (jj > 0 and jj < ySize)
	// 	{
	// 		for (int i = 0; i < 11; i++)
	// 		{
	// 			if ((ii + i)>0 && (ii + i)<xSize)
	// 			{
	// 				DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = backgroundValue;
	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = DIRECT_A2D_ELEM(tiltImage, jj,0);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = (DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-1) / 16 +
	// 				// 								   	   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i) / 4);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = -99;
	// 			}		
	// 		}
	// 	}		

	// 	// Interpolation line 2
	// 	jj = (int)(m2*(ii-(double)xSize)+y2);

	// 	if (jj > 0 and jj < ySize)
	// 	{
	// 		for (int i = -11; i < 0; i++)
	// 		{
	// 			if ((ii + i)>0 && (ii + i)<xSize)
	// 			{
	// 				DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = backgroundValue;
	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = DIRECT_A2D_ELEM(tiltImage, jj, xSize-1);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = (DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-1) / 16 +
	// 				// 								   	   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i) / 4);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = -99;
	// 			}		
	// 		}
	// 	}	

	// 	// Interpolation line 3
	// 	jj = (int)(m3*(ii-x3)+(double)ySize);

	// 	if (jj > 0 and jj < ySize)
	// 	{
	// 		for (int i = 0; i < 11; i++)
	// 		{
	// 			if ((ii + i)>0 && (ii + i)<xSize)
	// 			{
	// 				DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = backgroundValue;
	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = DIRECT_A2D_ELEM(tiltImage, jj,0);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = (DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-1) / 16 +
	// 				// 								   	   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i) / 4);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = -99;
	// 			}		
	// 		}
	// 	}	

	// 	// Interpolation line 4
	// 	jj = (int)(m4*(ii-x4)+(double)ySize);

	// 	if (jj > 0 and jj < ySize)
	// 	{
	// 		for (int i = -11; i < 0; i++)
	// 		{
	// 			if ((ii + i)>0 && (ii + i)<xSize)
	// 			{
	// 				DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = backgroundValue;
	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = DIRECT_A2D_ELEM(tiltImage, jj, xSize-1);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = (DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-1) / 16 +
	// 				// 										DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-1) / 16 +
	// 				// 										DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+1) / 16 +
	// 				// 										DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+1) / 16 +
	// 				// 										DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i) / 8 +
	// 				// 										DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i) / 8 +
	// 				// 										DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-1) / 8 +
	// 				// 										DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+1) / 8 +
	// 				// 										DIRECT_A2D_ELEM(tiltImage, jj ,ii+i) / 4);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = -99;
	// 			}		
	// 		}
	// 	}	
	// }

	// int ii;
	// // Draw vertical line
	// m1 = (double)(x1)/(-y1);
	// m2 = (double)(x2-(double)xSize)/(-y2);
	// m3 = (double)(-x3)/(y3-(double)ySize);
	// m4 = (double)((double)xSize-x4)/(y4-(double)ySize);

	// for (int jj = 0; jj < ySize; jj++)
	// {
	// 	DIRECT_A2D_ELEM(tmpImage, jj, 0) = backgroundValue;
	// 	DIRECT_A2D_ELEM(tmpImage, jj, xSize-1) = backgroundValue;

	// 	// Interpolation line 1
	// 	ii = (int)(m1*(jj-y1));

	// 	if (ii > 0 and ii < xSize)
	// 	{
	// 		for (int i = 0; i < 11; i++)
	// 		{
	// 			if ((ii + i)>0 && (ii + i)<xSize)
	// 			{
	// 				DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = backgroundValue;
	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = DIRECT_A2D_ELEM(tiltImage, jj,0);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = (DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-1) / 16 +
	// 				// 								   	   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i) / 4);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = (DIRECT_A2D_ELEM(tiltImage, jj-2 ,ii+i-2) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-2 ,ii+i+2) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+2 ,ii+i-2) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+2 ,ii+i+2) +		
					
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-2 ,ii+i-1) * 4 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-2 ,ii+i+1) * 4 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+2 ,ii+i-1) * 4 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+2 ,ii+i+1) * 4 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-2) * 4 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+2) * 4 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-2) * 4 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+2) * 4 +


	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-2 ,ii+i) * 7 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+2 ,ii+i) * 7 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-2) * 7 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+2) * 7 +
														   
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-1) * 16 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+1) * 16 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-1) * 16 +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+1) * 16 +

	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i) * 26 + 
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i) * 26 + 
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-1) * 26 + 
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+1) * 26 +

	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj ,ii+i) * 41) / 273;

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = (DIRECT_A2D_ELEM(tiltImage, jj-2 ,ii+i-2) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-2 ,ii+i+2) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+2 ,ii+i-2) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+2 ,ii+i+2) +		
					
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-2 ,ii+i-1) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-2 ,ii+i+1) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+2 ,ii+i-1) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+2 ,ii+i+1) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-2) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+2) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-2) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+2) +


	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-2 ,ii+i) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+2 ,ii+i) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-2) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+2) +
														   
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-1) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+1) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-1) +
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+1) +

	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i) + 
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i) + 
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-1) + 
	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+1) +

	// 				// 									   DIRECT_A2D_ELEM(tiltImage, jj ,ii+i)) / 25;

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = -99;
	// 			}		
	// 		}
	// 	}		

	// 	// Interpolation line 2
	// 	ii = (int)(m2*(jj-y2)+((double)xSize));

	// 	if (ii > 0 and ii < xSize)
	// 	{
	// 		for (int i = -11; i < 0; i++)
	// 		{
	// 			if ((ii + i)>0 && (ii + i)<xSize)
	// 			{
	// 				DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = backgroundValue;
	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = DIRECT_A2D_ELEM(tiltImage, jj,xSize-1);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = (DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-1) / 16 +
	// 				// 								   	   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i) / 4);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = -99;
	// 			}		
	// 		}
	// 	}	

	// 	// Interpolation line 3
	// 	ii = (int)(m3*(jj-(double)ySize)+x3);

	// 	if (ii > 0 and ii < xSize)
	// 	{
	// 		for (int i = 0; i < 11; i++)
	// 		{
	// 			if ((ii + i)>0 && (ii + i)<xSize)
	// 			{
	// 				DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = backgroundValue;
	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = DIRECT_A2D_ELEM(tiltImage, jj,0);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = (DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-1) / 16 +
	// 				// 								   	   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i) / 4);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = -99;
	// 			}		
	// 		}
	// 	}	

	// 	// Interpolation line 4
	// 	ii = (int)(m4*(jj-(double)ySize)+x4);

	// 	if (ii > 0 and ii < xSize)
	// 	{
	// 		for (int i = -11; i < 0; i++)
	// 		{
	// 			if ((ii + i)>0 && (ii + i)<xSize)
	// 			{
	// 				DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = backgroundValue;
	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = DIRECT_A2D_ELEM(tiltImage, jj,xSize-1);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = (DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i-1) / 16 +
	// 				// 								   	   DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i-1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i+1) / 16 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj-1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj+1 ,ii+i) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i-1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i+1) / 8 +
	// 				// 								       DIRECT_A2D_ELEM(tiltImage, jj ,ii+i) / 4);

	// 				// DIRECT_A2D_ELEM(tmpImage, jj ,ii+i) = -99;
	// 			}	
	// 		}
	// 	}	
	// }

	// tiltImage = tmpImage;

// }


// void ProgTomoDetectMisalignmentTrajectory::localAmplitude(MultidimArray<double> &tiltImage, MultidimArray<double> &amplitude)
// {
// 	amplitude = tiltImage;

// 	FourierTransformer transformer1(FFTW_BACKWARD);
// 	MultidimArray<std::complex<double>> fftrx, fftry;
// 	transformer1.FourierTransform(tiltImage, fftrx, true);
// 	fftry = fftrx;

//     double uy;
// 	double ux;
// 	double u;
// 	double uy2;

// 	long n=0;
// 	std::complex<double> J(0,1);

// 	for(size_t i=0; i<YSIZE(fftrx); ++i)
// 	{
// 		FFT_IDX2DIGFREQ(i, ySize, uy);
// 		uy2=uy*uy;

// 		for(size_t j=0; j<XSIZE(fftrx); ++j)
// 		{
// 			FFT_IDX2DIGFREQ(j, xSize, ux);
// 			u=sqrt(uy2+ux*ux)+1e-38;

// 			DIRECT_MULTIDIM_ELEM(fftrx, n) *= -J*(ux/u);

// 			DIRECT_MULTIDIM_ELEM(fftry, n) *= -J*(uy/u);
// 			++n;
// 		}
// 	}

// 	MultidimArray<double> fx, fy;

// 	fx.resizeNoCopy(tiltImage);
// 	fy.resizeNoCopy(tiltImage);

// 	transformer1.inverseFourierTransform(fftrx, fx);
// 	transformer1.inverseFourierTransform(fftry, fy);

// 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(tiltImage)
// 	{
// 		double fxx = DIRECT_MULTIDIM_ELEM(fx,n);
// 		double fyy = DIRECT_MULTIDIM_ELEM(fy,n);
// 		double aa = DIRECT_MULTIDIM_ELEM(amplitude,n);
// 		DIRECT_MULTIDIM_ELEM(amplitude,n) += sqrt(aa*aa + fxx*fxx + fyy*fyy);
// 	}

// 	size_t l = fnOut.find_last_of("\\/");
// 	std::string rawnameBis = fnOut.substr(0, l);
// 	std::string outputFileNameFilteredVolumeBis;
//     outputFileNameFilteredVolumeBis = rawnameBis + "/ts_amplitude.mrcs";

// 	Image<double> saveImageBis;
// 	saveImageBis() = amplitude;
// 	saveImageBis.write(outputFileNameFilteredVolumeBis);

// }



// bool ProgTomoDetectMisalignmentTrajectory::detectMisalignmentFromResiduals()
// {
// 	double mod2Thr = (fiducialSizePx * thrFiducialDistance) * (fiducialSizePx * thrFiducialDistance);
// 	double modThr = sqrt(mod2Thr);

// 	// Analyze residuals out of range distribution
// 	MultidimArray<int> resDistribution;
// 	std::vector<int> resCoordsOutOfRange(numberOfInputCoords, 0); // Array with the number of tilt-images out of rage for each coordinate
// 	std::vector<int> resImagesOutOfRange(nSize, 0); // Array with the number of coordinates out of rage for each tilt-image

// 	resDistribution.initZeros(nSize, numberOfInputCoords);

// 	for (size_t i = 0; i < vCM.size(); i++)
// 	{
// 		double mod2 = (vCM[i].residuals.x * vCM[i].residuals.x) + (vCM[i].residuals.y * vCM[i].residuals.y);

// 		if (mod2 > mod2Thr)
// 		{
// 			DIRECT_A2D_ELEM(resDistribution, (size_t)vCM[i].detectedCoordinate.z, vCM[i].id) = 1;
// 			resCoordsOutOfRange[vCM[i].id] += 1;
// 			resImagesOutOfRange[(size_t)vCM[i].detectedCoordinate.z] += 1;
// 		}
// 	}


// 	// *** potentially move thresholds to .h 
// 	double resCoordsOutOfRangeThr = 0.05;
// 	double resImagesOutOfRangeThr = 0.05;

// 	int resCoordsOutOfRangeThrAbs = std::max(1, (int)(nSize*0.05));

// 	size_t numberMisaliCoords = 0;
// 	size_t numberMisaliImages = 0;

// 	for (size_t i = 0; i < resCoordsOutOfRange.size(); i++)
// 	{
// 		if (resCoordsOutOfRange[i] > resCoordsOutOfRangeThrAbs)
// 		{
// 			std::cout << "COORDINATE " << i << " PRESENTS MISALIGNMENT THROUGH THE SERIES" << std::endl;
// 			numberMisaliCoords += 1;
// 		}
// 	}

// 	for (size_t i = 0; i < resImagesOutOfRange.size(); i++)
// 	{
// 		double pValue = binomialTest(resImagesOutOfRange[i], numberOfInputCoords, 0.5);

// 		#ifdef DEBUG_RESIDUAL_ANALYSIS	
// 		std::cout << "-----------------------------numberOfInputCoords" << numberOfInputCoords << std::endl;
// 		std::cout << "-----------------------------resImagesOutOfRange[i]" << resImagesOutOfRange[i] << std::endl;
// 		std::cout << "-----------------------------pvalue" << pValue << std::endl;
// 		#endif
		
// 		if (pValue < 0.05 && resImagesOutOfRange[i] > (numberOfInputCoords/2))
// 		{
// 			std::cout << "IMAGE " << i << " IN TILT-SERIES PRESENTS MISALIGNMENT" << std::endl;
// 			numberMisaliImages += 1;

// 			// Fill the local alignment vector
// 			localAlignment[i] = false;
// 		}
// 	}

// 	if ((numberMisaliCoords/numberOfInputCoords) > 0.1 && numberMisaliCoords > 1)
// 	{
// 		globalAlignment = false;
// 	}
	

// 	std::cout << "IN TOTAL " << numberMisaliCoords << " OUT OF " << numberOfInputCoords << " INPUT COORDINATES PRESENTS MIALIGNMENT" << std::endl;
// 	std::cout << "IN TOTAL " << numberMisaliImages << " TILT-IMAGES PRESENTS MISALIGNMENT" << std::endl;


// 	size_t lastindex = fnOut.find_last_of("\\/");
// 	std::string rawname = fnOut.substr(0, lastindex);
// 	std::string outFileName = rawname + "/ts_resDistribution.mrc";

// 	Image<int> saveImage;
// 	saveImage() = resDistribution;
// 	saveImage.write(outFileName);


// 	// Run XmippScript for statistical residual analysis

// 	std::cout << "\nRunning residual statistical analysis..." << std::endl;

// 	lastindex = fnOut.find_last_of("\\/");
// 	rawname = fnOut.substr(0, lastindex);
// 	std::string fnVCM;
// 	std::string fnStats;
//     fnVCM = rawname + "/vCM.xmd";
// 	fnStats = rawname + "/residualStatistics.xmd";

// 	std::string cmd;

// 	#ifdef DEBUG_RESIDUAL_ANALYSIS
// 	// Debug command
// 	cmd = "python3 /home/fdeisidro/xmipp_devel/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnVCM + " -o " + fnStats + " --debug ";
// 	// cmd = "python3 /home/fdeisidro/data/scipion3/xmipp-bundle/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnVCM + " -o " + fnStats + " --debug";
// 	#else
// 	// No debug command
// 	cmd = "python3 /home/fdeisidro/xmipp_devel/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnVCM + " -o " + fnStats;
// 	// cmd = "python3 /home/fdeisidro/data/scipion3/xmipp-bundle/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnVCM + " -o " + fnStats;
// 	#endif
	
// 	std::cout << cmd << std::endl;
// 	int systemOut = system(cmd.c_str());

// 	// NEW APPROACH. NOW WE SPLIT IMAGE AND LANDMARK STATISTICS. ALSO SAVE SEPARATELY BY EACH INDIVIDUAL, NO AVERAGE OR PERCENTILES
// 	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// 	std::string fnStats_lm;
// 	std::string fnStats_image;

// 	fnStats_lm = rawname + "/residualStatistics_resid.xmd";
// 	fnStats_image = rawname + "/residualStatistics_image.xmd";

// 	std::string statistic;
// 	std::string statisticName;
// 	std::string residualNumber;
// 	int residualNumber_int;
// 	double value;

// 	// -- Save residual information --
// 	MetaDataVec residualStatsMd;
// 	residualStatsMd.read(fnStats_lm);

// 	// Vector containing stats: avg, std, chArea, chPerim, pvBinX, pvBinY, pvF, pvADF, ImageOutOfRange, LongestMisaliChain
// 	std::vector<double> residualStatsLine (10);
// 	std::vector<std::vector<double>> residualStatsTable;

// 	for (size_t i = 0; i < numberOfInputCoords; i++)
// 	{
// 		residualStatsTable.push_back(residualStatsLine);
// 	}

// 	// Save statistics in table
// 	for(size_t objId : residualStatsMd.ids())
// 	{
// 		residualStatsMd.getValue(MDL_IMAGE, statistic, objId);
// 		residualStatsMd.getValue(MDL_MIN, value, objId);

// 		statisticName = statistic.substr(statistic.find("_")+1);
// 		residualNumber = statistic.substr(0, statistic.find("_"));
// 		residualNumber_int = std::stoi(residualNumber);

// 		#ifdef DEBUG_RESIDUAL_ANALYSIS	
// 		std::cout << "Reading object " << objId << " from metadata" << std::endl;
// 		std::cout << "statistic " << statistic << std::endl;
// 		std::cout << "value " << value << std::endl;
// 		std::cout << "statisticName " << statisticName << std::endl;
// 		std::cout << "residualNumber " << residualNumber << std::endl;
// 		std::cout << "residualNumber_int " << residualNumber_int << std::endl;
// 		#endif

// 		if (strcmp(statisticName.c_str(), "chArea") == 0)
// 		{
// 			residualStatsTable[residualNumber_int][2] = value;
// 		}

// 		else if (strcmp(statisticName.c_str(), "chPerim") == 0)
// 		{
// 			residualStatsTable[residualNumber_int][3] = value;
// 		}
		
// 		else if (strcmp(statisticName.c_str(), "pvBinX") == 0)
// 		{
// 			residualStatsTable[residualNumber_int][4] = value;
// 		}

// 		else if (strcmp(statisticName.c_str(), "pvBinY") == 0)
// 		{
// 			residualStatsTable[residualNumber_int][5] = value;
// 		}

// 		else if (strcmp(statisticName.c_str(), "pvF") == 0)
// 		{
// 			residualStatsTable[residualNumber_int][6] = value;
// 		}

// 		else if (strcmp(statisticName.c_str(), "pvADF") == 0)
// 		{
// 			residualStatsTable[residualNumber_int][7] = value;
// 		}
// 	}

// 	// Complete residual info
// 	for (size_t n = 0; n < numberOfInputCoords; n++)
// 	{
// 		std::vector<CM> CM_fid;
// 		getCMbyFiducial(n, CM_fid);

// 		size_t numberCM = CM_fid.size();

// 		double avg;
// 		double std;
// 		size_t imagesOutOfRange = 0;
// 		size_t longestMisaliChain = 0;
// 		size_t misaliChain = 0;

// 		double sumResid = 0;
// 		double sumResid2 = 0;

// 		for (size_t i = 0; i < numberCM; i++)
// 		{
// 			double sum2 = CM_fid[i].residuals.x*CM_fid[i].residuals.x + CM_fid[i].residuals.y*CM_fid[i].residuals.y;
// 			sumResid2 += sum2;
// 			sumResid += sqrt(sum2);

// 			if (sum2 > mod2Thr)
// 			{
// 				imagesOutOfRange += 1;
// 				misaliChain += 1;
// 			}

// 			else if (misaliChain > longestMisaliChain)
// 			{
// 				longestMisaliChain = misaliChain;
// 				misaliChain = 0;
// 			}
// 		}

// 		#ifdef DEBUG_RESIDUAL_ANALYSIS	
// 		std::cout << "n " << n << std::endl;
// 		std::cout << "numberCM " << numberCM << std::endl;
// 		std::cout << "sumResid " << sumResid << std::endl;
// 		std::cout << "sumResid2 " << sumResid2 << std::endl;
// 		std::cout << "longestMisaliChain " << longestMisaliChain << std::endl;
// 		std::cout << "imagesOutOfRange " << imagesOutOfRange << std::endl;
// 		#endif

// 		avg = sumResid / numberCM;
// 		std = sqrt(sumResid2 / numberCM - avg * avg);

// 		#ifdef DEBUG_RESIDUAL_ANALYSIS	
// 		std::cout << "avg " << avg << std::endl;
// 		std::cout << "std " << std << std::endl;
// 		#endif

// 		residualStatsTable[n][0] = avg;
// 		residualStatsTable[n][1] = std;
// 		residualStatsTable[n][8] = imagesOutOfRange;
// 		residualStatsTable[n][9] = longestMisaliChain;
// 	}

// 	std::cout << " ----------------------------------------------- residualStatsTable" << std::endl;
// 	for (size_t n = 0; n < numberOfInputCoords; n++)
// 	{
// 		for (size_t i = 0; i < 10; i++)
// 		{
// 			std::cout << residualStatsTable[n][i] << " , ";
// 		}
// 		std::cout << "\n" ;
// 	}
// 	std::cout << " ----------------------------------------------- " << std::endl;

// 	// -- Save image information --
// 	residualStatsMd.read(fnStats_image);

// 	// Vector containing stats: avg, std, chArea, chPerim, pvBinX, pvBinY, pvF, ResidOutOfRange
// 	std::vector<double> imageStatsLine (8);
// 	std::vector<std::vector<double>> imageStatsTable;

// 	for (size_t i = 0; i < nSize; i++)
// 	{
// 		imageStatsTable.push_back(imageStatsLine);
// 	}
	
// 	// Save statistics in table
// 	for(size_t objId : residualStatsMd.ids())
// 	{
// 		residualStatsMd.getValue(MDL_IMAGE, statistic, objId);
// 		residualStatsMd.getValue(MDL_MIN, value, objId);

// 		statisticName = statistic.substr(statistic.find("_")+1);
// 		residualNumber = statistic.substr(0, statistic.find("_"));
// 		residualNumber_int = std::stoi(residualNumber);

// 		#ifdef DEBUG_RESIDUAL_ANALYSIS	
// 		std::cout << "Reading object " << objId << " from metadata" << std::endl;
// 		std::cout << "statistic " << statistic << std::endl;
// 		std::cout << "value " << value << std::endl;
// 		std::cout << "statisticName " << statisticName << std::endl;
// 		std::cout << "residualNumber " << residualNumber << std::endl;
// 		std::cout << "residualNumber_int " << residualNumber_int << std::endl;
// 		#endif

// 		if (strcmp(statisticName.c_str(), "chArea") == 0)
// 		{
// 			imageStatsTable[residualNumber_int][2] = value;
// 		}

// 		else if (strcmp(statisticName.c_str(), "chPerim") == 0)
// 		{
// 			imageStatsTable[residualNumber_int][3] = value;
// 		}
		
// 		else if (strcmp(statisticName.c_str(), "pvBinX") == 0)
// 		{
// 			imageStatsTable[residualNumber_int][4] = value;
// 		}

// 		else if (strcmp(statisticName.c_str(), "pvBinY") == 0)
// 		{
// 			imageStatsTable[residualNumber_int][5] = value;
// 		}

// 		else if (strcmp(statisticName.c_str(), "pvF") == 0)
// 		{
// 			imageStatsTable[residualNumber_int][6] = value;
// 		}
// 	}

// 	// Complete residual info
// 	for (size_t n = 0; n < nSize; n++)
// 	{
// 		std::vector<CM> CM_image;
// 		getCMbyImage(n, CM_image);

// 		size_t numberCM = CM_image.size();

// 		if (numberCM > 0)
// 		{
// 			double avg;
// 			double std;
// 			size_t residOutOfRange = 0;

// 			double sumResid = 0;
// 			double sumResid2 = 0;

// 			for (size_t i = 0; i < numberCM; i++)
// 			{
// 				double sum = CM_image[i].residuals.x*CM_image[i].residuals.x + CM_image[i].residuals.y*CM_image[i].residuals.y;
// 				sumResid2 += sum;
// 				sumResid += sqrt(sum);

// 				if (sumResid > mod2Thr)
// 				{
// 					residOutOfRange += 1;
// 				}
// 			}
			
// 			#ifdef DEBUG_RESIDUAL_ANALYSIS	
// 			std::cout << "n " << n << std::endl;
// 			std::cout << "numberCM " << numberCM << std::endl;
// 			std::cout << "sumResid " << sumResid << std::endl;
// 			std::cout << "sumResid2 " << sumResid2 << std::endl;
// 			#endif

// 			avg = sumResid / numberCM;
// 			std = sqrt(sumResid2 / numberCM - avg * avg);

// 			#ifdef DEBUG_RESIDUAL_ANALYSIS	
// 			std::cout << "avg " << avg << std::endl;
// 			std::cout << "std " << std << std::endl;
// 			#endif

// 			imageStatsTable[n][0] = avg;
// 			imageStatsTable[n][1] = std;
// 			imageStatsTable[n][7] = residOutOfRange;
// 		}
// 	}

// 	std::cout << " ----------------------------------------------- imageStatsTable" << std::endl;
// 	for (size_t n = 0; n < nSize; n++)
// 	{
// 		for (size_t i = 0; i < 8; i++)
// 		{
// 			std::cout << imageStatsTable[n][i] << " , ";
// 		}
// 		std::cout << "\n" ;
// 	}
// 	std::cout << " -----------------------------------------------" << std::endl;

// 	// -- Write output file for decision tree training --
// 	std::string decisionTreeStatsFileName_chain;
// 	std::string decisionTreeStatsFileName_image;

// 	size_t li = fnOut.find_last_of("\\/");
// 	std::string fileBaseName = fnOut.substr(0, li);
// 	li = fileBaseName.find_last_of("\\/");
// 	fileBaseName = fileBaseName.substr(0, li);

// 	decisionTreeStatsFileName_chain = fileBaseName + "/decisionTreeStats_chain.txt";
// 	decisionTreeStatsFileName_image = fileBaseName + "/decisionTreeStats_image.txt";

// 	#ifdef DEBUG_RESIDUAL_ANALYSIS	
// 	std::cout << "fileBaseName " << fileBaseName << std::endl;
// 	std::cout << "decisionTreeStatsFileName_chain " << decisionTreeStatsFileName_chain << std::endl;
// 	std::cout << "decisionTreeStatsFileName_image " << decisionTreeStatsFileName_image << std::endl;
// 	#endif
	
// 	std::ofstream myfile;

// 	myfile.open (decisionTreeStatsFileName_chain, std::ios_base::app);
// 	for (size_t n = 0; n < residualStatsTable.size(); n++)  // Landmark decision tree stats
// 	{
// 		myfile << residualStatsTable[n][0];
// 		myfile << ", ";
// 		myfile << residualStatsTable[n][1];
// 		myfile << ", ";
// 		myfile << residualStatsTable[n][2];
// 		myfile << ", ";
// 		myfile << residualStatsTable[n][3];
// 		myfile << ", ";
// 		myfile << residualStatsTable[n][4];
// 		myfile << ", ";
// 		myfile << residualStatsTable[n][5];
// 		myfile << ", ";
// 		myfile << residualStatsTable[n][6];
// 		myfile << ", ";
// 		myfile << residualStatsTable[n][7];
// 		myfile << ", ";
// 		myfile << residualStatsTable[n][8];
// 		myfile << ", ";
// 		myfile << residualStatsTable[n][9];
// 		myfile << "\n";
// 	}
// 	myfile.close();

// 	myfile.open (decisionTreeStatsFileName_image, std::ios_base::app);
// 	for (size_t n = 0; n < imageStatsTable.size(); n++)  // Image decision tree stats
// 	{
// 		myfile << imageStatsTable[n][0];
// 		myfile << ", ";
// 		myfile << imageStatsTable[n][1];
// 		myfile << ", ";
// 		myfile << imageStatsTable[n][2];
// 		myfile << ", ";
// 		myfile << imageStatsTable[n][3];
// 		myfile << ", ";
// 		myfile << imageStatsTable[n][4];
// 		myfile << ", ";
// 		myfile << imageStatsTable[n][5];
// 		myfile << ", ";
// 		myfile << imageStatsTable[n][6];
// 		myfile << ", ";
// 		myfile << imageStatsTable[n][7];	
// 		myfile << "\n";
// 	}
// 	myfile.close();


// 	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// 	// Read results from analysis

// 	// float avgAreaCH = 0;
// 	// float avgPerimeterCH = 0;
// 	// float stdAreaCH = 0;
// 	// float stdPerimeterCH = 0;
// 	// size_t testBinPassed = 0;
// 	// size_t testFvarPassed = 0;
// 	// size_t testRandomWalkPassed = 0;

// 	// std::vector<float> areaCHV;
// 	// std::vector<float> perimCHV;

// 	// MetaDataVec residualStatsMd;
// 	// size_t objId;

// 	// residualStatsMd.read(fnStats);

// 	// int enable;
// 	// double value;

// 	// for(size_t objId : residualStatsMd.ids())
// 	// {
// 	// 	residualStatsMd.getValue(MDL_ENABLED, enable, objId);
// 	// 	residualStatsMd.getValue(MDL_IMAGE, statistic, objId);
// 	// 	residualStatsMd.getValue(MDL_MIN, value, objId);

// 	// 	statisticName = statistic.substr(statistic.find("_")+1);

// 	// 	if (enable == 1)
// 	// 	{
// 	// 		if (strcmp(statisticName.c_str(), "pvBinX") == 0 || strcmp(statisticName.c_str(), "pvBinY") == 0)
// 	// 		{
// 	// 			testBinPassed += 1;
// 	// 		}

// 	// 		else if (strcmp(statisticName.c_str(), "pvF") == 0)
// 	// 		{
// 	// 			testFvarPassed += 1;
// 	// 		}

// 	// 		else if (strcmp(statisticName.c_str(), "pvADF") == 0)
// 	// 		{
// 	// 			testRandomWalkPassed += 1;
// 	// 		}

// 	// 		else if (strcmp(statisticName.c_str(), "chArea") == 0)
// 	// 		{
// 	// 			areaCHV.push_back(value);
// 	// 		}

// 	// 		else if (strcmp(statisticName.c_str(), "chPerim") == 0)
// 	// 		{
// 	// 			perimCHV.push_back(value);
// 	// 		}
// 	// 	}
// 	// }

// 	// float sumAreaCH = 0;
// 	// float sumPerimeterCH = 0;
// 	// float sum2AreaCH = 0;
// 	// float sum2PerimeterCH = 0;
	
// 	// for (size_t i = 0; i < perimCHV.size(); i++)
// 	// {
// 	// 	sumAreaCH += areaCHV[i];
// 	// 	sum2AreaCH += areaCHV[i]*areaCHV[i];
// 	// 	sumPerimeterCH += perimCHV[i];
// 	// 	sum2PerimeterCH += perimCHV[i]*perimCHV[i];
// 	// }

// 	// avgAreaCH = sumAreaCH / numberOfInputCoords;
// 	// avgPerimeterCH = sumPerimeterCH / numberOfInputCoords;

// 	// stdAreaCH = sqrt(sum2AreaCH / numberOfInputCoords - avgAreaCH * avgAreaCH);
// 	// stdPerimeterCH = sqrt(sum2PerimeterCH / numberOfInputCoords - avgPerimeterCH * avgPerimeterCH);

// 	// float rmOutliersAreaCH = 0;
// 	// float rmOutliersPerimCH = 0;
// 	// size_t counterArea = 0;
// 	// size_t counterPerim = 0;

// 	// for (size_t i = 0; i < perimCHV.size(); i++)
// 	// {
// 	// 	if (abs(areaCHV[i]-avgAreaCH)<3*stdAreaCH)
// 	// 	{
// 	// 		rmOutliersAreaCH += areaCHV[i];
// 	// 		counterArea += 1;
// 	// 	}

// 	// 	if (abs(perimCHV[i]-avgPerimeterCH)<3*stdPerimeterCH)
// 	// 	{
// 	// 		rmOutliersPerimCH += perimCHV[i];
// 	// 		counterPerim += 1;
// 	// 	}
// 	// }

// 	// rmOutliersAreaCH /= counterArea;
// 	// rmOutliersPerimCH /= counterPerim;

// 	// std::cout << "-----------------------------------" << std::endl;
// 	// std::cout << "numberOfInputCoords: " << numberOfInputCoords << std::endl;
// 	// std::cout << "avgAreaCH: " << avgAreaCH << std::endl;
// 	// std::cout << "avgPerimeterCH: " << avgPerimeterCH << std::endl;
// 	// std::cout << "sum2AreaCH: " << sum2AreaCH << std::endl;
// 	// std::cout << "sum2PerimeterCH: " << sum2PerimeterCH << std::endl;
// 	// std::cout << "stdAreaCH: " << stdAreaCH << std::endl;
// 	// std::cout << "stdPerimeterCH: " << stdPerimeterCH << std::endl;
// 	// std::cout << "counterArea: " << counterArea << std::endl;
// 	// std::cout << "counterPerim: " << counterPerim << std::endl;
// 	// std::cout << "stdAreaCH: " << stdAreaCH << std::endl;
// 	// std::cout << "stdPerimeterCH: " << stdPerimeterCH << std::endl;
// 	// std::cout << "-----------------------------------" << std::endl;

// 	// #ifdef DEBUG_GLOBAL_MISALI
// 	// std::cout << "Average convex hull area (removing outliers): " << rmOutliersAreaCH << std::endl;
// 	// std::cout << "Average convex hull perimeter (removing outliers): " << rmOutliersPerimCH << std::endl;
// 	// std::cout << "Binomial test passed: " << testBinPassed << "/" << 2 * numberOfInputCoords << std::endl;
// 	// std::cout << "F variance test passed: " << testFvarPassed << "/" << numberOfInputCoords << std::endl;
// 	// std::cout << "Random walk test passed: " << testRandomWalkPassed << "/" << numberOfInputCoords << std::endl;
// 	// #endif

// 	// // 30% of the bead size in pixels
// 	// float maxDeviation = mod2Thr;

// 	// float thrAreaCH = PI*maxDeviation*maxDeviation;
// 	// float thrPerimeterCH = 2*PI*maxDeviation;

// 	// #ifdef DEBUG_GLOBAL_MISALI
// 	// std::cout << "Threshold convex hull area: " << thrAreaCH << std::endl;
// 	// std::cout << "Threshold convex hull perimeter: " << thrPerimeterCH << std::endl;
// 	// #endif


// 	// // Statisticals for decision tree ***
// 	// // Images
// 	// std::vector<double> averageFiducialResidualsInImage;	// Average residual distance per image
// 	// std::vector<double> stdFiducialResidualsInImage;		// STD residual distance per image


// 	// for (size_t i = 0; i < nSize; i++)
// 	// {
// 	// 	std::vector<CM> vCM_image;
// 	// 	getCMbyImage(i, vCM_image);
// 	// 	std::vector<double> residualDistanceInImage;

// 	// 	for (size_t j = 0; j < vCM_image.size(); j++)
// 	// 	{
// 	// 		double distance = sqrt((vCM_image[j].residuals.x*vCM_image[j].residuals.x)+(vCM_image[j].residuals.y*vCM_image[j].residuals.y));
// 	// 		residualDistanceInImage.push_back(distance/fiducialSizePx);
// 	// 	}

// 	// 	double sum = 0;
// 	// 	double sum2 = 0;
// 	// 	double average = 0;
// 	// 	double std = 0;
// 	// 	size_t residualDistanceInImageSize = residualDistanceInImage.size();

// 	// 	for(size_t e = 0; e < residualDistanceInImageSize; e++)
// 	// 	{
// 	// 		double value = residualDistanceInImage[e];
// 	// 		sum += value;
// 	// 		sum2 += value*value;
// 	// 	}

// 	// 	average = sum/residualDistanceInImageSize;
// 	// 	std = sqrt(sum2/residualDistanceInImageSize - average*average);

// 	// 	averageFiducialResidualsInImage.push_back(average);
// 	// 	stdFiducialResidualsInImage.push_back(std);
// 	// }

// 	// // Ficudials
// 	// std::vector<double> averageResidualDistancePerFiducial; 				// Average residual distance per fiducial
// 	// std::vector<double> stdResidualDistancePerFiducial; 					// STD residual distance per fiducial
// 	// std::vector<double> ratioOfImagesOutOfRange(numberOfInputCoords, 0.0);	// Ratio of images out of range in a single gold bead
// 	// std::vector<double> longestMisalignedChain;								// Longest chain of images presenting misalignment (in percentage)


// 	// for (size_t i = 0; i < numberOfInputCoords; i++)
// 	// {
// 	// 	std::vector<CM> vCM_fiducial;
// 	// 	getCMbyFiducial(i, vCM_fiducial);
// 	// 	std::vector<double> residualDistancePerFiducial;

// 	// 	for (size_t j = 0; j < vCM_fiducial.size(); j++)
// 	// 	{
// 	// 		double distance = sqrt((vCM_fiducial[j].residuals.x*vCM_fiducial[j].residuals.x)+(vCM_fiducial[j].residuals.y*vCM_fiducial[j].residuals.y));
// 	// 		residualDistancePerFiducial.push_back(distance/fiducialSizePx);

// 	// 		if (distance > modThr)
// 	// 		{
// 	// 			ratioOfImagesOutOfRange[i] += 1;
// 	// 		}
// 	// 	}

// 	// 	ratioOfImagesOutOfRange[i] /= nSize;
		
// 	// 	double sum = 0;
// 	// 	double sum2 = 0;
// 	// 	double average = 0;
// 	// 	double std = 0;
// 	// 	size_t residualDistancePerFiducialSize = residualDistancePerFiducial.size();

// 	// 	for(size_t e = 0; e < residualDistancePerFiducialSize; e++)
// 	// 	{
// 	// 		double value = residualDistancePerFiducial[e];
// 	// 		sum += value;
// 	// 		sum2 += value*value;
// 	// 	}

// 	// 	average = sum/residualDistancePerFiducialSize;
// 	// 	std = sqrt(sum2/residualDistancePerFiducialSize - average*average);

// 	// 	averageResidualDistancePerFiducial.push_back(average);
// 	// 	stdResidualDistancePerFiducial.push_back(std);
// 	// }

// 	// for (size_t i = 0; i < numberOfInputCoords; i++)
// 	// {
// 	// 	int longestChain = 0;
// 	// 	int chain = 0;

// 	// 	for (size_t j = 0; j < nSize; j++)
// 	// 	{
// 	// 		if (DIRECT_A2D_ELEM(resDistribution, j, i) == 1)
// 	// 		{
// 	// 			chain += 1;
// 	// 		}
// 	// 		else
// 	// 		{
// 	// 			if (chain > longestChain)
// 	// 			{
// 	// 				longestChain = chain;
// 	// 			}

// 	// 			chain = 0;
// 	// 		}
// 	// 	}

// 	// 	if (chain > longestChain)
// 	// 	{
// 	// 		longestChain = chain;
// 	// 	}

// 	// 	longestMisalignedChain.push_back((1.0*longestChain)/(1.0*nSize));
// 	// }

// 	// sort(averageFiducialResidualsInImage.begin(), averageFiducialResidualsInImage.end());
// 	// sort(stdFiducialResidualsInImage.begin(), stdFiducialResidualsInImage.end());
// 	// sort(averageResidualDistancePerFiducial.begin(), averageResidualDistancePerFiducial.end());
// 	// sort(stdResidualDistancePerFiducial.begin(), stdResidualDistancePerFiducial.end());
// 	// sort(ratioOfImagesOutOfRange.begin(), ratioOfImagesOutOfRange.end());
// 	// sort(longestMisalignedChain.begin(), longestMisalignedChain.end());

// 	// std::ofstream myfile;
// 	// myfile.open (decisionTreeStatsFileName_chain, std::ios_base::app);
// 	// myfile << averageFiducialResidualsInImage[0];
// 	// myfile << ", ";
// 	// myfile << averageFiducialResidualsInImage[(int)(averageFiducialResidualsInImage.size()/2)];
// 	// myfile << ", ";
// 	// myfile << averageFiducialResidualsInImage[averageFiducialResidualsInImage.size()-1];
// 	// myfile << ", ";

// 	// myfile << stdFiducialResidualsInImage[0];
// 	// myfile << ", ";
// 	// myfile << stdFiducialResidualsInImage[(int)(stdFiducialResidualsInImage.size()/2)];
// 	// myfile << ", ";
// 	// myfile << stdFiducialResidualsInImage[stdFiducialResidualsInImage.size()-1];
// 	// myfile << ", ";

// 	// myfile << averageResidualDistancePerFiducial[0];
// 	// myfile << ", ";
// 	// myfile << averageResidualDistancePerFiducial[(int)(averageResidualDistancePerFiducial.size()/2)];
// 	// myfile << ", ";
// 	// myfile << averageResidualDistancePerFiducial[averageResidualDistancePerFiducial.size()-1];
// 	// myfile << ", ";

// 	// myfile << stdResidualDistancePerFiducial[0];
// 	// myfile << ", ";
// 	// myfile << stdResidualDistancePerFiducial[(int)(stdResidualDistancePerFiducial.size()/2)];
// 	// myfile << ", ";
// 	// myfile << stdResidualDistancePerFiducial[stdResidualDistancePerFiducial.size()-1];
// 	// myfile << ", ";

// 	// myfile << ratioOfImagesOutOfRange[0];
// 	// myfile << ", ";
// 	// myfile << ratioOfImagesOutOfRange[(int)(ratioOfImagesOutOfRange.size()/2)];
// 	// myfile << ", ";
// 	// myfile << ratioOfImagesOutOfRange[ratioOfImagesOutOfRange.size()-1];
// 	// myfile << ", ";

// 	// myfile << longestMisalignedChain[0];
// 	// myfile << ", ";
// 	// myfile << longestMisalignedChain[(int)(longestMisalignedChain.size()/2)];
// 	// myfile << ", ";
// 	// myfile << longestMisalignedChain[longestMisalignedChain.size()-1];
// 	// myfile << "\n";

// 	// myfile.close();

// 	////////////////////////////////////////////////////////////////////////////////////////////////////////////


// 	// if ((rmOutliersAreaCH < thrAreaCH) && (rmOutliersPerimCH < thrPerimeterCH))
// 	// {
// 	// 	return true;
// 	// }
	
// 	return false;
// }

// void ProgTomoDetectMisalignmentTrajectory::bandPassFilter(MultidimArray<double> &tiltImage, int imageNumber) // *** remove imageNumber only for debugging purposes
// {
// 	// Detect interpolation region
// 	MultidimArray<double> tmpImage = tiltImage;

// 	for (size_t i = 1; i < xSize-1; i++)
// 	{
// 		for (size_t j = 1; j < ySize-1; j++)
// 		{
// 			DIRECT_A2D_ELEM(tmpImage, j ,i) = (-1 * DIRECT_A2D_ELEM(tiltImage, j-1 ,i) +
// 											   -1 * DIRECT_A2D_ELEM(tiltImage, j+1 ,i) +
// 											   -1 * DIRECT_A2D_ELEM(tiltImage, j ,i-1) +
// 											   -1 * DIRECT_A2D_ELEM(tiltImage, j ,i+1) +
// 									 		    4 * DIRECT_A2D_ELEM(tiltImage, j ,i));
// 		}
// 	}

// 	// if (imageNumber == 10)
// 	// {
	
// 	// #ifdef DEBUG_OUTPUT_FILES
// 	// size_t lastindex = fnOut.find_last_of("\\/");
// 	// std::string rawname = fnOut.substr(0, lastindex);
// 	// std::string outputFileNameFilteredVolume;
//     // outputFileNameFilteredVolume = rawname + "/ts_laplace.mrcs";

// 	// Image<double> saveImage;
// 	// saveImage() = tmpImage;
// 	// saveImage.write(outputFileNameFilteredVolume);
// 	// #endif
// 	// }


// 	// int x1 = 0;  			// (x1, 0)
// 	// int x2 = xSize - 1;  	// (x2, 0)
// 	// int x3 = 0;  			// (x3, ySize)
// 	// int x4 = xSize - 1;  	// (x4, ySize)
// 	// int y1 = 0;  			// (y1, 0)
// 	// int y2 = ySize - 1;  	// (xSize, y2)
// 	// int y3 = 0;  			// (0, y3)
// 	// int y4 = ySize -1;  	// (xSize, y4)
	
// 	// Background value as the median of the corners
// 	std::vector<double> corners{DIRECT_A2D_ELEM(tiltImage, 0, 0),
// 								DIRECT_A2D_ELEM(tiltImage, 0, xSize-1),
// 								DIRECT_A2D_ELEM(tiltImage, ySize-1, 0),
// 								DIRECT_A2D_ELEM(tiltImage, ySize-1, xSize-1)};

// 	sort(corners.begin(), corners.end(), std::greater<double>());

// 	double backgroundValue = (corners[1]+corners[2])/2;

// 	// Margin thickness
// 	int marginThickness = (int)(fiducialSizePx * 0.5);

// 	auto epsilon = MINDOUBLE;

// 	std::vector<Point2D<int>> interpolationLimits;

// 	bool firstLimitFound;

// 	int xMin;
// 	int xMax;

// 	for (size_t j = 1; j < ySize-2; j++)
// 	{
// 		for (size_t i = 1; i < xSize-1; i++)
// 		{
// 			if(abs(DIRECT_A2D_ELEM(tmpImage, j, i)) > epsilon)
// 			{
// 				xMin = ((i + marginThickness)>(xSize-1)) ? (xSize-1) : (i + marginThickness);

// 				// Fill margin thickness with background value
// 				for (size_t a = i; a < i + marginThickness; a++)
// 				{
// 					DIRECT_A2D_ELEM(tiltImage, j, a) = backgroundValue;
// 				}
				
// 				break;
// 			}
// 		}


// 		for (size_t i = xSize-1; i > 1; i--)
// 		{
// 			if(abs(DIRECT_A2D_ELEM(tmpImage, j, i)) > epsilon)
// 			{
// 				xMax = ((i - marginThickness)<0) ? 0 : (i - marginThickness);

// 				// Fill margin thickness with background value
// 				for (size_t a = i - marginThickness; a < i; a++)
// 				{
// 					DIRECT_A2D_ELEM(tiltImage, j, a) = backgroundValue;
// 				}

// 				break;
// 			}
// 		}

// 		if (xMin >= xMax)
// 		{
// 			int value = (int) (((xMax+marginThickness)+(xMin-marginThickness))/2);
// 			xMax = value;
// 			xMin = value;
// 		}
		
// 		Point2D<int> limit (xMin, xMax);
// 		interpolationLimits.push_back(limit);
// 	}

// 	interpolationLimitsVector.push_back(interpolationLimits);

// 	// for (size_t i = 1; i < xSize-2; i++)
// 	// {
// 	// 	if(abs(DIRECT_A2D_ELEM(tmpImage, 1, i)) > epsilon)
// 	// 	{
// 	// 		x1=i;
// 	// 		break;
// 	// 	}
// 	// }

// 	// for (size_t i = xSize-2; i > x1; i--)
// 	// {
// 	// 	if(abs(DIRECT_A2D_ELEM(tmpImage, 1, i)) > epsilon)
// 	// 	{
// 	// 		x2=i;
// 	// 		break;
// 	// 	}
// 	// }

// 	// for (size_t i = 1; i < xSize-2; i++)
// 	// {
// 	// 	if(abs(DIRECT_A2D_ELEM(tmpImage, ySize-2, i)) > epsilon)
// 	// 	{
// 	// 		x3=i;
// 	// 		break;
// 	// 	}
// 	// }

// 	// for (size_t i = xSize-2; i > x3; i--)
// 	// {
// 	// 	if(abs(DIRECT_A2D_ELEM(tmpImage, ySize-2, i)) > epsilon)
// 	// 	{
// 	// 		x4=i;
// 	// 		break;
// 	// 	}
// 	// }

// 	// for (size_t j = 1; j < ySize-2; j++)
// 	// {
// 	// 	if(abs(DIRECT_A2D_ELEM(tmpImage, j, 1)) > epsilon)
// 	// 	{
// 	// 		y1=j;
// 	// 		break;
// 	// 	}
// 	// }

// 	// for (size_t j = 1; j < ySize-2; j++)
// 	// {
// 	// 	if(abs(DIRECT_A2D_ELEM(tmpImage, j, xSize-2)) > epsilon)
// 	// 	{
// 	// 		y2=j;
// 	// 		break;
// 	// 	}
// 	// }

// 	// for (size_t j = ySize-2; j > y1; j--)
// 	// {
// 	// 	if(abs(DIRECT_A2D_ELEM(tmpImage, j, 1)) > epsilon)
// 	// 	{
// 	// 		y3=j;
// 	// 		break;
// 	// 	}
// 	// }

// 	// for (size_t j = ySize-2; j > y2; j--)
// 	// {
// 	// 	if(abs(DIRECT_A2D_ELEM(tmpImage, j, xSize-2)) > epsilon)
// 	// 	{
// 	// 		y4=j;
// 	// 		break;
// 	// 	}
// 	// }

// 	// #ifdef DEBUG_PREPROCESS
// 	// std::cout<< "x1: " << x1<<std::endl;
// 	// std::cout<< "x2: " << x2<<std::endl;
// 	// std::cout<< "x3: " << x3<<std::endl;
// 	// std::cout<< "x4: " << x4<<std::endl;
// 	// std::cout<< "y1: " << y1<<std::endl;
// 	// std::cout<< "y2: " << y2<<std::endl;
// 	// std::cout<< "y3: " << y3<<std::endl;
// 	// std::cout<< "y4: " << y4<<std::endl;
// 	// # endif 

// 	// // Remove interpolation edges
// 	// int jmin;
// 	// int jmax;

// 	// double m1 = (double)(-y1)/(x1);
// 	// double m2 = (double)(-y2)/(x2-(double)xSize);
// 	// double m3 = (double)(y3-(double)ySize)/(-x3);
// 	// double m4 = (double)(y4-(double)ySize)/((double)xSize-x4);

// 	// #ifdef DEBUG_PREPROCESS
// 	// std::cout<< "m1: " << m1<<std::endl;
// 	// std::cout<< "m2: " << m2<<std::endl;
// 	// std::cout<< "m3: " << m3<<std::endl;
// 	// std::cout<< "m4: " << m4<<std::endl;
// 	// #endif

// 	// x1 += marginThickness * cos(abs(atan(m1)));  // (x1, 0)
// 	// x2 -= marginThickness * cos(abs(atan(m2)));  // (x2, 0)
// 	// x3 += marginThickness * cos(abs(atan(m3)));  // (x3, ySize)
// 	// x4 -= marginThickness * cos(abs(atan(m4)));  // (x4, ySize)
// 	// y1 += marginThickness * sin(abs(atan(m1)));  // (y1, 0)
// 	// y2 += marginThickness * sin(abs(atan(m2)));  // (xSize, y2)
// 	// y3 -= marginThickness * sin(abs(atan(m3)));  // (0, y3)
// 	// y4 -= marginThickness * sin(abs(atan(m4)));  // (xSize, y4)

// 	// m1 = (double)(-y1)/(x1);
// 	// m2 = (double)(-y2)/(x2-(double)xSize);
// 	// m3 = (double)(y3-(double)ySize)/(-x3);
// 	// m4 = (double)(y4-(double)ySize)/((double)xSize-x4);

// 	// #ifdef DEBUG_PREPROCESS
// 	// std::cout<< "x1: " << x1<<std::endl;
// 	// std::cout<< "x2: " << x2<<std::endl;
// 	// std::cout<< "x3: " << x3<<std::endl;
// 	// std::cout<< "x4: " << x4<<std::endl;
// 	// std::cout<< "y1: " << y1<<std::endl;
// 	// std::cout<< "y2: " << y2<<std::endl;
// 	// std::cout<< "y3: " << y3<<std::endl;
// 	// std::cout<< "y4: " << y4<<std::endl;
// 	// # endif 
	
// 	// #ifdef DEBUG_PREPROCESS
// 	// std::cout<< "m1: " << m1<<std::endl;
// 	// std::cout<< "m2: " << m2<<std::endl;
// 	// std::cout<< "m3: " << m3<<std::endl;
// 	// std::cout<< "m4: " << m4<<std::endl;
// 	// #endif

// 	#ifdef DEBUG_PREPROCESS
// 	std::cout  << "backgroundValue " << backgroundValue << std::endl;
// 	std::cout  << "marginThickness " << marginThickness << std::endl;
// 	// std::cout  << "marginThickness * cos(abs(atan(m1))) " << marginThickness * cos(abs(atan(m1))) << std::endl;
// 	// std::cout  << "marginThickness * sin(abs(atan(m1))) " << marginThickness * sin(abs(atan(m1))) << std::endl;
// 	#endif

// 	// IC ic={x1, x2, x3, x4, y1, y2, y3, y4, m1, m2, m3, m4};
// 	// vIC.push_back(ic);

// 	// for (int i = 0; i < xSize; i++)
// 	// {
// 	// 	// minimum y index for interation
// 	// 	if(i < x1)
// 	// 	{
// 	// 		jmin = (int)(m1*i+y1);
// 	// 	}
// 	// 	else if (i > x2)
// 	// 	{
// 	// 		jmin = (int)(m2*(i-(int)xSize)+y2);
// 	// 	}
// 	// 	else
// 	// 	{
// 	// 		jmin = 1;
// 	// 	}
		
// 	// 	// maximum y index for interation
// 	// 	if(i < x3)
// 	// 	{
// 	// 		jmax = (int)(m3*(i-x3)+(int)ySize);
// 	// 	}
// 	// 	else if (i > x4)
// 	// 	{
// 	// 		jmax = (int)(m4*(i-x4)+(int)ySize);
// 	// 	}
// 	// 	else
// 	// 	{
// 	// 		jmax = (int)(ySize-2);
// 	// 	}

// 	// 	// check range in image size
// 	// 	if(jmin < 1)
// 	// 	{
// 	// 		jmin = 1;
// 	// 	}

// 	// 	if(jmax > (int)(ySize-2))
// 	// 	{
// 	// 		jmax = (int)(ySize-2);
// 	// 	}
		
// 	// 	// Remove edges
// 	// 	for (int j = 0; j <= jmin; j++)
// 	// 	{
// 	// 		DIRECT_A2D_ELEM(tiltImage, j ,i) = backgroundValue;
// 	// 	}

// 	// 	for (int j = jmax; j < ySize; j++)
// 	// 	{
// 	// 		DIRECT_A2D_ELEM(tiltImage, j ,i) = backgroundValue;
// 	// 	}
// 	// }

// 	// Bandpass filter image
// 	FourierTransformer transformer1(FFTW_BACKWARD);
// 	MultidimArray<std::complex<double>> fftImg;
// 	transformer1.FourierTransform(tiltImage, fftImg, true);

// 	normDim = (xSize>ySize) ? xSize : ySize;

// 	// 43.2 = 1440 * 0.03. This 43.2 value makes w = 0.03 (standard value) for an image whose bigger dimension is 1440 px.
// 	double w = 43.2 / normDim;

//     double lowFreqFilt = samplingRate/(1.05*fiducialSize);
// 	double highFreqFilt = samplingRate/(0.95*fiducialSize);

// 	double tail_high = highFreqFilt + w;
//     double tail_low = lowFreqFilt - w;

// 	double delta = PI / w;

//     double uy;
// 	double ux;
// 	double u;
// 	double uy2;

// 	#ifdef DEBUG_PREPROCESS
// 	std::cout << "Filter params: " << std::endl;
// 	std::cout << "samplingRate: " << samplingRate << std::endl;
// 	std::cout << "normDim: " << normDim << std::endl;
// 	std::cout << "w: " << w << std::endl;
// 	std::cout << "lowFreqFilt: " << lowFreqFilt << std::endl;
// 	std::cout << "highFreqFilt: " << highFreqFilt << std::endl;
// 	std::cout << "tail_low: " << tail_low << std::endl;
// 	std::cout << "tail_high: " << tail_high << std::endl;
// 	std::cout << "delta: " << delta << std::endl;
// 	#endif

// 	long n=0;

// 	for(size_t i=0; i<YSIZE(fftImg); ++i)
// 	{
// 		FFT_IDX2DIGFREQ(i, ySize, uy);
// 		uy2=uy*uy;

// 		for(size_t j=0; j<XSIZE(fftImg); ++j)
// 		{
// 			FFT_IDX2DIGFREQ(j, xSize, ux);
// 			u=sqrt(uy2+ux*ux);

// 			if (u > tail_high || u < tail_low)
// 			{
// 				DIRECT_MULTIDIM_ELEM(fftImg, n) = 0;
// 			}
// 			else
// 			{
// 				if (u >= highFreqFilt && u <=tail_high)
// 				{
// 					DIRECT_MULTIDIM_ELEM(fftImg, n) *= 0.5*(1+cos((u-highFreqFilt)*delta));
// 				}

// 				if (u <= lowFreqFilt && u >= tail_low)
// 				{
// 					DIRECT_MULTIDIM_ELEM(fftImg, n) *= 0.5*(1+cos((u-lowFreqFilt)*delta));
// 				}
// 			}

// 			++n;
// 		}
// 	}

// 	transformer1.inverseFourierTransform(fftImg, tiltImage);

//     // Apply Laplacian to tilt-image with kernel:
// 	//     0 -1 0
// 	// k = -1 4 -1
// 	//     0 -1 0

// 	tmpImage = tiltImage;
// 	tiltImage.initZeros(ySize, xSize);

// 	#ifdef DEBUG_PREPROCESS
// 	std::cout << "Laplacian in interpolation limits" << std::endl;
// 	std::cout << "interpolationLimits.size() " << interpolationLimits.size() << std::endl;
// 	#endif

// 	for (size_t j = 1; j < ySize-2; j++)
// 	{
// 		Point2D<int> il = interpolationLimits[j-1];
// 		xMin = il.x;
// 		xMax = il.y;

// 		#ifdef DEBUG_PREPROCESS
// 		std::cout << "j " << j << ", xMax " << xMax << ", xMin " << xMin << std::endl;
// 		#endif

// 		for (size_t i = xMin; i < xMax; i++)
// 		{
// 			DIRECT_A2D_ELEM(tiltImage, j ,i) = (-1 * DIRECT_A2D_ELEM(tmpImage, j-1 ,i) +
// 												-1 * DIRECT_A2D_ELEM(tmpImage, j+1 ,i) +
// 												-1 * DIRECT_A2D_ELEM(tmpImage, j ,i-1) +
// 												-1 * DIRECT_A2D_ELEM(tmpImage, j ,i+1) +
// 												4 * DIRECT_A2D_ELEM(tmpImage, j ,i));
// 		}
// 	}

// // 	for (int i = 1; i < xSize-2; i++)
// // 	{
// // 		// minimum y index for interation
// // 		if(i < x1)
// // 		{
// // 			jmin = (int)(m1*i+y1);
// // 		}
// // 		else if (i > x2)
// // 		{
// // 			jmin = (int)(m2*(i-(int)xSize)+y2);
// // 		}
// // 		else
// // 		{
// // 			jmin = 1;
// // 		}
		
// // 		// maximum y index for interation
// // 		if(i < x3)
// // 		{
// // 			jmax = (int)(m3*(i-x3)+(int)ySize);
// // 		}
// // 		else if (i > x4)
// // 		{
// // 			jmax = (int)(m4*(i-x4)+(int)ySize);
// // 		}
// // 		else
// // 		{
// // 			jmax = (int)(ySize-2);
// // 		}

// // 		// check range in image size
// // 		if(jmin < 1)
// // 		{
// // 			jmin = 1;
// // 		}

// // 		if(jmax > (int)(ySize-2))
// // 		{
// // 			jmax = (int)(ySize-2);
// // 		}
		
// // 		// Apply laplacian in when y belongs to (jmin, jmax)
// // 		for (int j = jmin; j <= jmax; j++)
// // 		{
// // 			DIRECT_A2D_ELEM(tiltImage, j ,i) = (-2 * DIRECT_A2D_ELEM(tmpImage, j-1 ,i) +
// // 											    -2 * DIRECT_A2D_ELEM(tmpImage, j+1 ,i) +
// // 												-2 * DIRECT_A2D_ELEM(tmpImage, j ,i-1) +
// // 												-2 * DIRECT_A2D_ELEM(tmpImage, j ,i+1) +
// // 									 			8 * DIRECT_A2D_ELEM(tmpImage, j ,i));
// // 		}	
// // 	}

// }




// void ProgTomoDetectMisalignmentTrajectory::getHighContrastCoordinates(MultidimArray<double> tiltSeriesFiltered)
// {
// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Picking high contrast coordinates..." << std::endl;
// 	#endif

// 	// *** reutilizar binaryCoordinatesMapSlice slice a slice y descartar labelCoordiantesMap
// 	// OJO perdería mos el debug de la serie con el labelling
//     MultidimArray<double> binaryCoordinatesMapSlice;
//     MultidimArray<double> labelCoordiantesMapSlice;
//     MultidimArray<double> labelCoordiantesMap;

// 	labelCoordiantesMap.initZeros(nSize, zSize, ySize, xSize);

// 	for(size_t k = 0; k < nSize; ++k)
// 	{
// 		#ifdef VERBOSE_OUTPUT
// 		std::cout <<  "Searching for high contrast coordinates in tilt-image " << k << std::endl;
// 		#endif
// 		std::vector<int> sliceVector;

// 		// Calculate threshold value for each image of the series
// 		// IC ic = vIC[k];
// 		// int jmin;
// 		// int jmax;

//         // for(size_t i = 0; i < xSize; ++i)
//         // {
// 		// 	// Search inside the interpolation edges
// 		// 	if(i < ic.x1)
// 		// 	{
// 		// 		jmin = (int)(ic.m1*i+ic.y1);
// 		// 	}
// 		// 	else if (i > ic.x2)
// 		// 	{
// 		// 		jmin = (int)(ic.m2*(i-(int)xSize)+ic.y2);
// 		// 	}
// 		// 	else
// 		// 	{
// 		// 		jmin = 0;
// 		// 	}
			
// 		// 	if(i < ic.x3)
// 		// 	{
// 		// 		jmax = (int)(ic.m3*(i-ic.x3)+(int)ySize);
// 		// 	}
// 		// 	else if (i > ic.x4)
// 		// 	{
// 		// 		jmax = (int)(ic.m4*(i-ic.x4)+(int)ySize);
// 		// 	}
// 		// 	else
// 		// 	{
// 		// 		jmax = (int)(ySize-1);
// 		// 	}

// 		// 	if(jmin < 0)
// 		// 	{
// 		// 		jmin = 0;
// 		// 	}

// 		// 	if(jmax > (int)(ySize))
// 		// 	{
// 		// 		jmax = (int)(ySize);
// 		// 	}
			
// 		// 	for (int j = jmin; j < jmax; j++)
// 		// 	{
// 		// 			/// *** enhance performance: do not use slice vector, sum directly from image
// 		// 			sliceVector.push_back(DIRECT_NZYX_ELEM(tiltSeriesFiltered, k, 0, j ,i));
// 		// 	}	
//         // }

// 		std::vector<Point2D<int>> interpolationLimits = interpolationLimitsVector[k];
// 		int xMin;
// 		int xMax;

// 		for (size_t j = 1; j < ySize-2; j++)
// 		{
// 			Point2D<int> il = interpolationLimits[j-1];
// 			xMin = il.x;
// 			xMax = il.y;

// 			for (size_t i = xMin; i < xMax; i++)
// 			{
// 				sliceVector.push_back(DIRECT_NZYX_ELEM(tiltSeriesFiltered, k, 0, j ,i));
// 			}
// 		}

//         double sum = 0;
// 		double sum2 = 0;
//         int Nelems = 0;
//         double average = 0;
//         double standardDeviation = 0;
//         double sliceVectorSize = sliceVector.size();

// 		// ***TODO: value = (value-min)^2 aplicar no linearidad

// 		// int maximum = *max_element(sliceVector.begin(), sliceVector.end());
// 		// #ifdef DEBUG_HCC
// 		// std::cout << "maximum: " <<  maximum << std::endl;
//         // #endif

// 		// for (size_t i = 0; i < sliceVectorSize; i++)
// 		// {
// 		// 	sliceVector[i] = (sliceVector[i]-maximum)*(sliceVector[i]-maximum);
// 		// }

//         for(size_t e = 0; e < sliceVectorSize; e++)
//         {
//             int value = sliceVector[e];
//             sum += value;
//             sum2 += value*value;
//             ++Nelems;
//         }

//         average = sum / sliceVectorSize;
//         standardDeviation = sqrt(sum2/Nelems - average*average);

//         double thresholdL = average - thrSDHCC * standardDeviation;  // THRESHOLD FOR FILTERING PIXELS
//         double thresholdU = average + thrSDHCC * standardDeviation;  // THRESHOLD FOR FILTERING PIXELS

//         #ifdef DEBUG_HCC
// 		std::cout << "------------------------------------------------------" << std::endl;
// 		std::cout << "Slice: " << k+1 << " Average: " << average << " SD: " << standardDeviation << std::endl;
//         #endif

// 		int numberOfPointsAddedBinaryMap;
// 		bool firstExecution = true;
// 		int numberOfNewPeakedCoordinates;
// 		std::vector<Point3D<double>> newCoordinates3D;

// 		size_t iteration = 0;

// 		while(true)
// 		{
// 			numberOfPointsAddedBinaryMap = 0;
// 			numberOfNewPeakedCoordinates = 0;
// 			newCoordinates3D.clear();
// 			binaryCoordinatesMapSlice.initZeros(ySize, xSize);
// 			labelCoordiantesMapSlice.initZeros(ySize, xSize);

// 			if (!firstExecution)
// 			{
// 				thresholdL -= 0.05 * standardDeviation;
// 				std::cout << "New thresholdL " << thresholdL << std::endl;

// 				thresholdU += 0.05 * standardDeviation;
// 				std::cout << "New thresholdU " << thresholdU << std::endl;
// 			}
			

// 			for(size_t i = 0; i < ySize; i++)
// 			{
// 				for(size_t j = 0; j < xSize; ++j)
// 				{
// 					double value = DIRECT_A3D_ELEM(tiltSeriesFiltered, k, i, j);
				
// 					if (value < thresholdL | value > thresholdU)
// 					{
// 						DIRECT_A2D_ELEM(binaryCoordinatesMapSlice, i, j) = 1.0;
						
// 						numberOfPointsAddedBinaryMap += 1;
// 					}
// 				}
// 			}

// 			#ifdef DEBUG_HCC
// 			std::cout << "Number of points in the binary map: " << numberOfPointsAddedBinaryMap << std::endl;
// 			#endif

// 			iteration +=1;
// 			std::cout << " iteration " <<  iteration << std::endl;
// 			std::cout << " numberOfPointsAddedBinaryMap " <<  numberOfPointsAddedBinaryMap << std::endl;
// 			std::cout << " ((double)numberOfPointsAddedBinaryMap/ (xSize*ySize)) " <<  ((double)numberOfPointsAddedBinaryMap/ (xSize*ySize)) << std::endl;

// 			int colour;

// 			closing2D(binaryCoordinatesMapSlice, 5, 2, 8);
			
// 			colour = labelImage2D(binaryCoordinatesMapSlice, labelCoordiantesMapSlice, 8);  // The value 8 is the neighbourhood

// 			#ifdef DEBUG_HCC
// 			std::cout << "Colour: " << colour << std::endl;
// 			#endif

// 			std::vector<std::vector<int>> coordinatesPerLabelX (colour);
// 			std::vector<std::vector<int>> coordinatesPerLabelY (colour);

// 			for(size_t i = 0; i < ySize; i++)
// 			{
// 				for(size_t j = 0; j < xSize; ++j)
// 				{
// 					int value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);

// 					if(value!=0)
// 					{
// 						coordinatesPerLabelX[value-1].push_back(j);
// 						coordinatesPerLabelY[value-1].push_back(i);
// 					}
// 				}
// 			}

// 			size_t numberOfCoordinatesPerValue;


// 			// std::vector<double> occupancyV;

// 			// Trim coordinates based on the characteristics of the labeled region
// 			for(size_t value = 0; value < colour; value++)
// 			{
// 				numberOfCoordinatesPerValue =  coordinatesPerLabelX[value].size();

// 				int xCoor = 0;
// 				int yCoor = 0;

// 				for(size_t coordinate=0; coordinate < coordinatesPerLabelX[value].size(); coordinate++)
// 				{
// 					xCoor += coordinatesPerLabelX[value][coordinate];
// 					yCoor += coordinatesPerLabelY[value][coordinate];
// 				}

// 				double xCoorCM = xCoor/numberOfCoordinatesPerValue;
// 				double yCoorCM = yCoor/numberOfCoordinatesPerValue;

// 				bool keep = filterLabeledRegions(coordinatesPerLabelX[value], coordinatesPerLabelY[value], xCoorCM, yCoorCM);
			
// 				// double occupancy = filterLabeledRegions(coordinatesPerLabelX[value], coordinatesPerLabelY[value], xCoorCM, yCoorCM);
// 				// occupancyV.push_back(occupancy);

// 				if(keep)
// 				{
// 					Point3D<double> point3D(xCoorCM, yCoorCM, k);
// 					newCoordinates3D.push_back(point3D);

// 					numberOfNewPeakedCoordinates += 1;
// 				}
// 			}

// 			std::cout << " numberOfNewPeakedCoordinates " <<  numberOfNewPeakedCoordinates << std::endl;
// 			std::cout << " newCoordinates3D.size() " <<  newCoordinates3D.size() << std::endl;

// 			if (newCoordinates3D.size() < inputCoords.size()*5)
// 			{
// 				for (size_t i = 0; i < newCoordinates3D.size(); i++)
// 				{
// 					coordinates3D.push_back(newCoordinates3D[i]);
// 				}

// 				for(size_t i = 0; i < ySize; i++)
// 				{
// 					for(size_t j = 0; j < xSize; ++j)
// 					{
// 						double value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);

// 						if (value > 0)
// 						{
// 							DIRECT_NZYX_ELEM(labelCoordiantesMap, k, 0, i, j) = value;
// 						}
// 					}
// 				}

// 				std::cout << " newCoordinates3D.size() " <<  newCoordinates3D.size() << std::endl;
// 				std::cout << " coordinates3D.size() " <<  coordinates3D.size() << std::endl;

// 				break;
// 			}


// 			firstExecution = false;
// 		}

// 		// } while(!(numberOfNewPeakedCoordinates < 1));
	
// 		// std::cout << "Occupancy vector=";
// 		// for (size_t i = 0; i < occupancyV.size(); i++)
// 		// {
// 		// 	std::cout << occupancyV[i] << " ";
// 		// }
// 		// std::cout << "" << std::endl;
		

// 		// sort(occupancyV.begin(), occupancyV.end(), std::greater<double>());

// 		// std::cout << "Occupancy vector sorted=";
// 		// for (size_t i = 0; i < occupancyV.size(); i++)
// 		// {
// 		// 	std::cout << occupancyV[i] << " ";
// 		// }
// 		// std::cout << "" << std::endl;

// 		// double occupancyThr = occupancyV[20];

// 		// std::cout << occupancyThr << std::endl;


// 		// // Add coordinates if occupancy > occupancyThr
// 		// for(size_t value = 0; value < colour; value++)
// 		// {
// 		// 	numberOfCoordinatesPerValue =  coordinatesPerLabelX[value].size();

// 		// 	int xCoor = 0;
// 		// 	int yCoor = 0;

// 		// 	for(size_t coordinate=0; coordinate < coordinatesPerLabelX[value].size(); coordinate++)
// 		// 	{
// 		// 		xCoor += coordinatesPerLabelX[value][coordinate];
// 		// 		yCoor += coordinatesPerLabelY[value][coordinate];
// 		// 	}

// 		// 	double xCoorCM = xCoor/numberOfCoordinatesPerValue;
// 		// 	double yCoorCM = yCoor/numberOfCoordinatesPerValue;

// 		// 	// bool keep = filterLabeledRegions(coordinatesPerLabelX[value], coordinatesPerLabelY[value], xCoorCM, yCoorCM);
			
// 		// 	double occupancy = filterLabeledRegions(coordinatesPerLabelX[value], coordinatesPerLabelY[value], xCoorCM, yCoorCM);

// 		// 	if(occupancy>occupancyThr)
// 		// 	{
// 		// 		Point3D<double> point3D(xCoorCM, yCoorCM, k);
// 		// 		coordinates3D.push_back(point3D);

// 		// 		#ifdef DEBUG_HCC
// 		// 		numberOfNewPeakedCoordinates += 1;
// 		// 		#endif
			
// 		// 	}
// 		// }
		

// 		#ifdef DEBUG_HCC
// 		std::cout << "Number of coordinates added: " << numberOfNewPeakedCoordinates <<std::endl;
// 		std::cout << "Accumulated number of coordinates: " << coordinates3D.size() <<std::endl;
// 		#endif

//     }

// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Number of peaked coordinates: " << coordinates3D.size() << std::endl;
// 	#endif

// 	#ifdef DEBUG_OUTPUT_FILES
// 	size_t lastindex = fnOut.find_last_of("\\/");
// 	std::string rawname = fnOut.substr(0, lastindex);
// 	std::string outputFileNameLabeledVolume;
//     outputFileNameLabeledVolume = rawname + "/ts_labeled.mrcs";

// 	Image<double> saveImage;
// 	saveImage() = labelCoordiantesMap; 
// 	saveImage.write(outputFileNameLabeledVolume);
// 	#endif

// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "High contrast coordinates picked succesfully!" << std::endl;
// 	#endif
// }



// void ProgTomoDetectMisalignmentTrajectory::centerCoordinates(MultidimArray<double> tiltSeriesFiltered)
// {
// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Centering coordinates..." << std::endl;
// 	#endif

// 	size_t numberOfFeatures = coordinates3D.size();

// 	MultidimArray<double> feature;
// 	MultidimArray<double> mirrorFeature;
// 	MultidimArray<double> correlationVolumeR;

// 	int coordHalfX;
// 	int coordHalfY;
// 	int ti;

// 	int boxSize = int(fiducialSizePx);
// 	int doubleBoxSize = fiducialSizePx * 2;

// 	for(size_t n = 0; n < numberOfFeatures; n++)
// 	{
// 		#ifdef DEBUG_CENTER_COORDINATES
// 		std::cout << "-------------------- coordinate " << n << " (" << coordinates3D[n].x << ", " << coordinates3D[n].y << ", " << coordinates3D[n].z << ")" << std::endl;
// 		#endif

// 		// Construct feature and its mirror symmetric. We quadruple the size to include a feature two times
// 		// the box size plus padding to avoid incoherences in the shift sign
// 		feature.initZeros(2 * doubleBoxSize, 2 * doubleBoxSize);
// 		mirrorFeature.initZeros(2 * doubleBoxSize, 2 * doubleBoxSize);

// 		coordHalfX = coordinates3D[n].x - boxSize;
// 		coordHalfY = coordinates3D[n].y - boxSize;
// 		ti = coordinates3D[n].z;

// 		for(int j = 0; j < doubleBoxSize; j++) // xDim
// 		{
// 			for(int i = 0; i < doubleBoxSize; i++) // yDim
// 			{
// 				// Check coordinate is not out of volume
// 				if ((coordHalfY + i) < 0 || (coordHalfY + i) > ySize ||
// 					(coordHalfX + j) < 0 || (coordHalfX + j) > xSize)
// 				{
// 					DIRECT_A2D_ELEM(feature, i + boxSize, j + boxSize) = 0;

// 					DIRECT_A2D_ELEM(mirrorFeature, doubleBoxSize + boxSize -1 - i, doubleBoxSize + boxSize -1 - j) = 0;
// 				}
// 				else
// 				{
// 					DIRECT_A2D_ELEM(feature, i + boxSize, j + boxSize) = DIRECT_A3D_ELEM(tiltSeriesFiltered, 
// 																		ti, 
// 																		coordHalfY + i, 
// 																		coordHalfX + j);

// 					DIRECT_A2D_ELEM(mirrorFeature, doubleBoxSize + boxSize -1 - i, doubleBoxSize + boxSize -1 - j) = 
// 					DIRECT_A3D_ELEM(tiltSeriesFiltered, 
// 									ti, 
// 									coordHalfY + i,
// 									coordHalfX + j);
// 				}
// 			}
// 		}

// 		#ifdef DEBUG_CENTER_COORDINATES
// 		Image<double> image;

// 		std::cout << "Feature dimensions (" << XSIZE(feature) << ", " << YSIZE(feature) << ", " << ZSIZE(feature) << ")" << std::endl;
// 		image() = feature;
// 		size_t lastindex = fnOut.find_last_of(".");
// 		std::string rawname = fnOut.substr(0, lastindex);
// 		std::string outputFileName;
// 		outputFileName = rawname + "_" + std::to_string(n) + "_feature.mrc";
// 		image.write(outputFileName);

// 		std::cout << "Mirror feature dimensions (" << XSIZE(mirrorFeature) << ", " << YSIZE(mirrorFeature) << ", " << ZSIZE(mirrorFeature) << ")" << std::endl;
// 		image() = mirrorFeature;
// 		outputFileName = rawname + "_" + std::to_string(n) + "_mirrorFeature.mrc";
// 		image.write(outputFileName);
// 		#endif

// 		// Shift the particle respect to its symmetric to look for the maximum correlation displacement
// 		CorrelationAux aux;
// 		correlation_matrix(feature, mirrorFeature, correlationVolumeR, aux, true);

// 		auto maximumCorrelation = MINDOUBLE;
// 		double xDisplacement = 0;
// 		double yDisplacement = 0;

// 		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(correlationVolumeR)
// 		{
// 			double value = DIRECT_A2D_ELEM(correlationVolumeR, i, j);

// 			if (value > maximumCorrelation)
// 			{
// 				maximumCorrelation = value;
// 				xDisplacement = j;
// 				yDisplacement = i;
// 			}
// 		}

// 		#ifdef DEBUG_CENTER_COORDINATES
// 		std::cout << "maximumCorrelation " << maximumCorrelation << std::endl;
// 		std::cout << "xDisplacement " << ((int) xDisplacement - doubleBoxSize) / 2 << std::endl;
// 		std::cout << "yDisplacement " << ((int) yDisplacement - doubleBoxSize) / 2 << std::endl;

// 		std::cout << "Correlation volume dimensions (" << XSIZE(correlationVolumeR) << ", " << YSIZE(correlationVolumeR) << ")" << std::endl;
// 		#endif


// 		// Update coordinate and remove if it is moved out of the volume
// 		double updatedCoordinateX = coordinates3D[n].x + ((int) xDisplacement - doubleBoxSize) / 2;
// 		double updatedCoordinateY = coordinates3D[n].y + ((int) yDisplacement - doubleBoxSize) / 2;

// 		int deletedCoordinates = 0;

// 		if (updatedCoordinateY < 0 || updatedCoordinateY > ySize ||
// 			updatedCoordinateX < 0 || updatedCoordinateX > xSize)
// 		{
// 			coordinates3D.erase(coordinates3D.begin()+n-deletedCoordinates);
// 			deletedCoordinates++;
// 		}
// 		else
// 		{
// 			coordinates3D[n].x = updatedCoordinateX;
// 			coordinates3D[n].y = updatedCoordinateY;
// 		}

// 		#ifdef DEBUG_CENTER_COORDINATES
// 		// Construct and save the centered feature
// 		MultidimArray<double> centerFeature;

// 		centerFeature.initZeros(doubleBoxSize, doubleBoxSize);

// 		coordHalfX = coordinates3D[n].x - boxSize;
// 		coordHalfY = coordinates3D[n].y - boxSize;

// 		for(int j = 0; j < doubleBoxSize; j++) // xDim
// 		{
// 			for(int i = 0; i < doubleBoxSize; i++) // yDim
// 			{
// 				// Check coordinate is not out of volume
// 				if ((coordHalfY + i) < 0 || (coordHalfY + i) > ySize ||
// 					(coordHalfX + j) < 0 || (coordHalfX + j) > xSize)
// 				{
// 					DIRECT_A2D_ELEM(centerFeature, i, j) = 0;
// 				}
// 				else
// 				{
// 					DIRECT_A2D_ELEM(centerFeature, i, j) = DIRECT_A3D_ELEM(tiltSeriesFiltered,
// 																				ti,
// 																				coordHalfY + i,
// 																				coordHalfX + j);
// 				}
// 			}
// 		}

// 		std::cout << "Centered feature dimensions (" << XSIZE(centerFeature) << ", " << YSIZE(centerFeature) << ")" << std::endl;

// 		image() = centerFeature;
// 		outputFileName = rawname + "_" + std::to_string(n) + "_centerFeature.mrc";
// 		image.write(outputFileName);
// 		#endif
// 	}

// 	#ifdef DEBUG_CENTER_COORDINATES
// 	std::cout << "3D coordinates after centering: " << std::endl;

// 	for(size_t n = 0; n < numberOfFeatures; n++)
// 	{
// 		std::cout << "Coordinate " << n << " (" << coordinates3D[n].x << ", " << coordinates3D[n].y << ", " << coordinates3D[n].z << ")" << std::endl;

// 	}
// 	#endif

// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Centering of coordinates finished successfully!" << std::endl;
// 	#endif
// }


// bool ProgTomoDetectMisalignmentTrajectory::votingHCC()
// {
// 	std::vector<size_t> coordinatesInSlice;
// 	std::vector<size_t> coordinatesInSlice_left;
// 	std::vector<size_t> coordinatesInSlice_right;

// 	std::vector<size_t> coord3DVotes_V(coordinates3D.size(), 0);

// 	float thrVottingDistance2 = (fiducialSizePx)*(fiducialSizePx);

// 	#ifdef DEBUG_VOTTING
// 	std::cout << "thrVottingDistance2 " << thrVottingDistance2 << std::endl;
// 	#endif

// 	// Votting step	
// 	for (int n = 0; n < nSize; n++)
// 	{
// 		#ifdef DEBUG_VOTTING
// 		std::cout << "votting image " << n << std::endl;
// 		#endif

// 		coordinatesInSlice = getCoordinatesInSliceIndex(n);
		
// 		// Skip for first image in the series
// 		if (n != 0)
// 		{
// 			coordinatesInSlice_left = getCoordinatesInSliceIndex(n-1);
// 		}

// 		// Skip for last image in the series
// 		if (n != (nSize-1))
// 		{		
// 			coordinatesInSlice_right = getCoordinatesInSliceIndex(n+1);
// 		}

// 		for(size_t i = 0; i < coordinatesInSlice.size(); i++)
// 		{
// 			Point3D<double> c = coordinates3D[coordinatesInSlice[i]];

// 			// Skip for first image in the series
// 			if (n != 0)
// 			{
// 				for (size_t j = 0; j < coordinatesInSlice_left.size(); j++)
// 				{
// 					Point3D<double> cl = coordinates3D[coordinatesInSlice_left[j]];
// 					float distance2 = (c.x-cl.x)*(c.x-cl.x)+(c.y-cl.y)*(c.y-cl.y);

// 					if(distance2 < thrVottingDistance2)
// 					{
// 						coord3DVotes_V[coordinatesInSlice[i]] += 1;
// 					}
// 				}
// 			}

// 			// Skip for last image in the series
// 			if (n != (nSize-1))
// 			{		
// 				for (size_t j = 0; j < coordinatesInSlice_right.size(); j++)
// 				{
// 					Point3D<double> cr = coordinates3D[coordinatesInSlice_right[j]];
// 					float distance2 = (c.x-cr.x)*(c.x-cr.x)+(c.y-cr.y)*(c.y-cr.y);

// 					if(distance2 < thrVottingDistance2)
// 					{
// 						coord3DVotes_V[coordinatesInSlice[i]] += 1;
// 					}
// 				}
// 			}
// 		}

// 		#ifdef VERBOSE_OUTPUT
// 		if (coordinatesInSlice.size() == 0)
// 		{
// 			std::cout << "No matching coordinates for slice " << n << std::endl;
// 		}
// 		#endif
		
// 	}

// 	// Trimming step
// 	size_t deletedIndexes = 0;

// 	for (size_t i = 0; i < coord3DVotes_V.size(); i++)
// 	{
// 		if (coord3DVotes_V[i] == 0)
// 		{
// 			#ifdef DEBUG_VOTTING
// 			std::cout << "Deleted coordinate " << i << std::endl;
// 			#endif

// 			coordinates3D.erase(coordinates3D.begin()+i);
// 			coord3DVotes_V.erase(coord3DVotes_V.begin()+i);
// 			deletedIndexes++;
// 			i--;
// 		}
// 	}
	
// 	// Generate output labeled and filtered series
// 	#ifdef DEBUG_OUTPUT_FILES
// 	MultidimArray<int> filteredLabeledTS;
// 	filteredLabeledTS.initZeros(nSize, 1, ySize, xSize);

// 	std::vector<Point2D<double>> cis;

// 	for (size_t n = 0; n < nSize; n++)
// 	{
// 		cis = getCoordinatesInSlice(n);

// 		MultidimArray<int> filteredLabeledTS_Image;
// 		filteredLabeledTS_Image.initZeros(ySize, xSize);

// 		for(size_t i = 0; i < cis.size(); i++)
// 		{
// 			fillImageLandmark(filteredLabeledTS_Image, (int)cis[i].x, (int)cis[i].y, 1);
// 		}

// 		for (size_t i = 0; i < ySize; ++i)
// 		{
// 			for (size_t j = 0; j < xSize; ++j)
// 			{
// 				DIRECT_NZYX_ELEM(filteredLabeledTS, n, 0, i, j) = DIRECT_A2D_ELEM(filteredLabeledTS_Image, i, j);
// 			}
// 		}
// 	}

// 	size_t lastindexBis = fnOut.find_last_of("\\/");
// 	std::string rawnameBis = fnOut.substr(0, lastindexBis);
// 	std::string outputFileNameFilteredVolumeBis;
//     outputFileNameFilteredVolumeBis = rawnameBis + "/ts_labeled_filtered.mrcs";

// 	Image<int> saveImageBis;
// 	saveImageBis() = filteredLabeledTS;
// 	saveImageBis.write(outputFileNameFilteredVolumeBis);
// 	#endif

// 	if (deletedIndexes != 0)
// 	{
// 		return true;
// 	}
// 	else
// 	{
// 		return false;
// 	}
// }


// bool ProgTomoDetectMisalignmentTrajectory::checkProjectedCoordinateInInterpolationEdges(Matrix1D<double> projectedCoordinate, size_t slice)
// {
// 	std::vector<Point2D<int>> interpolationLimits = lmDetector.interpolationLimitsVector[slice];

// 	int x = (int)(XX(projectedCoordinate));
// 	int y = (int)(YY(projectedCoordinate));

// 	if (x >= interpolationLimits[y].x || x <= interpolationLimits[y].y)
// 	{
// 		return true;
// 	}
// 	else
// 	{
// 		return false;
// 	}
	

// 	// for (size_t j = 1; j < ySize-2; j++)
// 	// {
// 	// 	Point2D<int> il = interpolationLimits[j-1];
// 	// 	xMin = il.x;
// 	// 	xMax = il.y;

// 	// 	for (size_t i = xMin; i < xMax; i++)
// 	// 	{
// 	// 		sliceVector.push_back(DIRECT_NZYX_ELEM(tiltSeriesFiltered, k, 0, j ,i));
// 	// 	}
// 	// }


// 	// IC ic = vIC[slice];

// 	// int x = (int)(XX(projectedCoordinate));
// 	// int y = (int)(YY(projectedCoordinate));
// 	// int jmin;
// 	// int jmax;

// 	// if(x < ic.x1)
// 	// {
// 	// 	jmin = (int)(ic.m1*x+ic.y1);
// 	// }
// 	// else if (x > ic.x2)
// 	// {
// 	// 	jmin = (int)(ic.m2*(x-(int)xSize)+ic.y2);
// 	// }
// 	// else
// 	// {
// 	// 	jmin = 0;
// 	// }
	
// 	// if(x < ic.x3)
// 	// {
// 	// 	jmax = (int)(ic.m3*(x-ic.x3)+(int)ySize);
// 	// }
// 	// else if (x > ic.x4)
// 	// {
// 	// 	jmax = (int)(ic.m4*(x-ic.x4)+(int)ySize);
// 	// }
// 	// else
// 	// {
// 	// 	jmax = (int)(ySize);
// 	// }

// 	// if(jmin < 0)
// 	// {
// 	// 	jmin = 0;
// 	// }

// 	// if(jmax > (int)(ySize))
// 	// {
// 	// 	jmax = (int)(ySize);
// 	// }

// 	// if (y > jmax || y < jmin)
// 	// {
// 	// 	return false;
// 	// }
// 	// else
// 	// {
// 	// 	return true;
// 	// }
// }

// double ProgTomoDetectMisalignmentTrajectory::binomialTest(int x, int n, float p)
// {
// 	double fact = 1;

// 	// fact = n! / x!
// 	for (size_t i = n; i < x; i--)
// 	{
// 		fact *= i;
// 	}

// 	// fact = fact / (n-x)!
// 	for (size_t i = n-x; i < 0; i--)
// 	{
// 		fact /= i;
// 	}

// 	// return: fact * p^x * (1-p)^(n-x)
// 	return fact * pow(p, x) * pow((1-p), (n-x));
// }


// void ProgTomoDetectMisalignmentTrajectory::closing2D(MultidimArray<double> binaryImage, int size, int count, int neig)
// {
// 	MultidimArray<double> tmp;
//     int i;

// 	//dilate
//     tmp = binaryImage;
// 	binaryImage.initZeros(ySize, xSize);

// 	size = 1;

//     for (i = 0;i < size;i++)
//     {
// 		double sum = 0;
// 		double dcount=count;
		
// 		for (int i = STARTINGY(tmp) + 1;i < FINISHINGY(tmp); i++)
// 			for (int j = STARTINGX(tmp) + 1;j < FINISHINGX(tmp); j++)
// 			{
// 				if (A2D_ELEM(tmp,i, j) == 0)
// 				{
// 					// 4-environment
// 					A2D_ELEM(binaryImage, i, j) = 0;
// 					sum = A2D_ELEM(tmp,i - 1, j) + A2D_ELEM(tmp,i + 1, j) +
// 						  A2D_ELEM(tmp,i, j - 1) + A2D_ELEM(tmp,i, j + 1);
// 					if (sum > dcount)
// 					{ //change the value to foreground
// 						A2D_ELEM(binaryImage, i, j) = 1;
// 					}
// 					else if (neig == 8)
// 					{ //8-environment
// 						sum +=A2D_ELEM(tmp,i - 1, j - 1) + A2D_ELEM(tmp,i - 1, j + 1) +
// 							  A2D_ELEM(tmp,i + 1, j - 1) + A2D_ELEM(tmp,i + 1, j + 1);
// 						if (sum > dcount)
// 						{ //change the value to foreground
// 							A2D_ELEM(binaryImage, i, j) = 1;
// 						}
// 					}
// 				}
// 				else
// 				{
// 					A2D_ELEM(binaryImage, i, j) = A2D_ELEM(tmp,i, j);
// 				}
//         	}
//     }

// 	// erode
// 	tmp = binaryImage;
// 	binaryImage.initZeros(ySize, xSize);

//     for (i = 0;i < size;i++)
//     {
// 		double sum = 0;
// 		double dcount=count;

// 		for (int i = STARTINGY(tmp) + 1;i < FINISHINGY(tmp); i++)
// 			for (int j = STARTINGX(tmp) + 1;j < FINISHINGX(tmp); j++)
// 			{
// 				if (A2D_ELEM(tmp,i, j) == 1)
// 				{
// 					// 4-environment
// 					A2D_ELEM(binaryImage, i, j) = 1;
// 					sum = A2D_ELEM(tmp,i - 1, j) + A2D_ELEM(tmp,i + 1, j) +
// 						  A2D_ELEM(tmp,i, j - 1) + A2D_ELEM(tmp,i, j + 1);
// 					if ((4 - sum) > dcount)
// 					{ //change the value to foreground
// 						A2D_ELEM(binaryImage, i, j) = 0;
// 					}
// 					else if (neig == 8)
// 					{ //8-environment
// 						sum +=A2D_ELEM(tmp,i - 1, j - 1) + A2D_ELEM(tmp,i - 1, j + 1) +
// 							  A2D_ELEM(tmp,i + 1, j - 1) + A2D_ELEM(tmp,i + 1, j + 1);
// 						if ((neig - sum) > dcount)
// 						{ //change the value to foreground
// 							A2D_ELEM(binaryImage, i, j) = 0;
// 						}
// 					}
// 				}
// 				else
// 				{
// 					A2D_ELEM(binaryImage, i, j) = A2D_ELEM(tmp,i, j);
// 				}
// 			}
//     }
// }


// void ProgTomoDetectMisalignmentTrajectory::detectLandmarkChains()
// {
// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Detecting landmark chains..." << std::endl;
// 	#endif

// 	std::vector<int> counterLinesOfLandmarkAppearance(ySize);

// 	// Calculate the number of landmarks per row (y index)
// 	for(size_t i = 0; i < ySize; i++)
// 	{
// 		for(int j = 0; j < coordinates3D.size(); j++)
// 		{
// 			if(coordinates3D[j].y == i)
// 			{
// 				counterLinesOfLandmarkAppearance[i] += 1;
// 			}

// 			else if (coordinates3D[j-1].y == i && j-1 > 0)
// 			{
// 				counterLinesOfLandmarkAppearance[i] += 1;
// 			}

// 			else if (coordinates3D[j+1].y == i && j+1 < coordinates3D.size())
// 			{
// 				counterLinesOfLandmarkAppearance[i] += 1;
// 			}
// 		}
// 	}

// 	// Calculate poisson lambda
// 	int numberEmptyRows = std::count(counterLinesOfLandmarkAppearance.begin(), counterLinesOfLandmarkAppearance.end(), 0);
// 	std::vector<int> histogramOfLandmarkAppearanceSorted (counterLinesOfLandmarkAppearance.size()-numberEmptyRows); 

// 	size_t sideIndex = 0;
// 	for (size_t i = 0; i < counterLinesOfLandmarkAppearance.size(); i++)
// 	{
// 		if(counterLinesOfLandmarkAppearance[i]!=0)
// 		{
// 			histogramOfLandmarkAppearanceSorted[sideIndex] = counterLinesOfLandmarkAppearance[i];
// 			sideIndex += 1;
// 		}
// 	}
	
// 	// *** TODO: optimize, get n maxima elements without sorting
// 	sort(histogramOfLandmarkAppearanceSorted.begin(), histogramOfLandmarkAppearanceSorted.end(), std::greater<int>());

// 	// Poisson lambda
// 	// for (size_t p = 0; p < counterLinesOfLandmarkAppearance.size(); p++)
// 	// {
// 	// 	std::cout <<  counterLinesOfLandmarkAppearance[p] <<std::endl;
// 	// }


// 	// for (size_t p = 0; p < histogramOfLandmarkAppearanceSorted.size(); p++)
// 	// {
// 	// 	std::cout <<  histogramOfLandmarkAppearanceSorted[p] <<std::endl;
// 	// }

// 	float absolutePossionPercetile = histogramOfLandmarkAppearanceSorted.size()*poissonLandmarkPercentile;

// 	float poissonAverage = histogramOfLandmarkAppearanceSorted[(int)absolutePossionPercetile];
	
// 	std::vector<size_t> chainIndexesY;

// 	// Test possion probability
// 	for (size_t i = 0; i < counterLinesOfLandmarkAppearance.size(); i++)
// 	{
// 		// Normalize the input values (make lambda=100 and make k=100(k'/lambda)) to fix a threshold value for the distribution.
// 		if (testPoissonDistribution(100*(poissonAverage/poissonAverage), 100*(counterLinesOfLandmarkAppearance[i]/poissonAverage)) > 0.001)
// 		{
// 			#ifdef DEBUG_POISSON
// 			std::cout << "Index " << i << " added with testPoissonDistribution=" << testPoissonDistribution(100*(poissonAverage/poissonAverage), 
// 						 100*(counterLinesOfLandmarkAppearance[i]/poissonAverage)) << std::endl;
// 			#endif

// 			chainIndexesY.push_back(i);
// 		}
// 	}

// 	// globalAlignment = detectGlobalAlignmentPoisson(counterLinesOfLandmarkAppearance, chainIndexesY);

// 	#ifdef DEBUG_CHAINS
// 	std::cout << "chainIndexesY.size()=" << chainIndexesY.size() << std::endl;
// 	#endif

// 	// Compose and cluster chains
// 	chain2dMap.initZeros(ySize, xSize);

// 	#ifdef DEBUG_OUTPUT_FILES
// 	MultidimArray<int> clustered2dMap;
// 	clustered2dMap.initZeros(ySize, xSize);
// 	#endif

// 	for (size_t i = 0; i < chainIndexesY.size(); i++)
// 	{
// 		// Compose chains
// 		size_t chainIndexY = chainIndexesY[i];

// 		#ifdef DEBUG_CHAINS
// 		std::cout << "-----------------COMPOSING LINE " << i << "-----------------" << std::endl;
// 		#endif

// 		// Binary vector with one's in the x coordinates belonging to each y coordinate
// 		std::vector<size_t> chainLineY(xSize, 0);

// 		// Vector containing the angles of the selected coordinates
// 		std::vector<float> chainLineYAngles(xSize, 0);
		
// 		for (size_t j = 0; j < chainLineY.size() ; j++)
// 		{
// 			for(int x = 0; x < coordinates3D.size(); x++)
// 			{
// 				Point3D<double> coordinate3D = coordinates3D[x];

// 				if(coordinate3D.y == chainIndexY)
// 				{
// 					chainLineY[coordinate3D.x] = 1;
					
// 					if(abs(tiltAngles[coordinate3D.z])>abs(chainLineYAngles[coordinate3D.x]))
// 					{
// 						chainLineYAngles[coordinate3D.x] = tiltAngles[coordinate3D.z];
// 					}
// 				}

// 				else if(coordinate3D.y == chainIndexY-1 && x-1 > 0)
// 				{
// 					chainLineY[coordinate3D.x] = 1;
					
// 					if(abs(tiltAngles[coordinate3D.z])>abs(chainLineYAngles[coordinate3D.x]))
// 					{
// 						chainLineYAngles[coordinate3D.x] = tiltAngles[coordinate3D.z];
// 					}
// 				}

// 				else if(coordinate3D.y == chainIndexY+1 && x+1 < coordinates3D.size())
// 				{
// 					chainLineY[coordinate3D.x] = 1;

// 					if(abs(tiltAngles[coordinate3D.z])>abs(chainLineYAngles[coordinate3D.x]))
// 					{
// 						chainLineYAngles[coordinate3D.x] = tiltAngles[coordinate3D.z];
// 					}
// 				}	
// 			}
// 		}

// 		// Cluser chains
// 		std::vector<size_t> clusteredChainLineY(xSize, 0);
// 		size_t clusterId = 2;
// 		size_t clusterIdSize = 0;
// 		std::vector<size_t> clusterSizeVector;
// 		int landmarkDisplacementThreshold;

// 		#ifdef DEBUG_CHAINS
// 		std::cout << "-----------------CLUSTERING LINE " << i << "-----------------" << std::endl;
// 		#endif

// 		for (size_t j = 0; j < chainLineY.size(); j++)
// 		{
// 			if(chainLineY[j] != 0){

// 				// Check angle range to calculate landmarkDisplacementThreshold does not go further that the size of the image
// 				if (chainLineYAngles[j]+thrNumberDistanceAngleChain*tiltAngleStep > tiltAngles[tiltAngles.size()-1])
// 				{
// 					#ifdef DEBUG_CHAINS
// 					std::cout << chainLineYAngles[j] << "calculateLandmarkProjectionDiplacement DEFAULT"<< std::endl;
// 					#endif
// 					landmarkDisplacementThreshold = calculateLandmarkProjectionDiplacement(chainLineYAngles[j], tiltAngles[tiltAngles.size()-1], j); 
// 				}
// 				else
// 				{
// 					#ifdef DEBUG_CHAINS
// 					std::cout << chainLineYAngles[j] << "calculateLandmarkProjectionDiplacement CALCULATED"<< std::endl;
// 					#endif
// 					landmarkDisplacementThreshold = calculateLandmarkProjectionDiplacement(chainLineYAngles[j], chainLineYAngles[j]+thrNumberDistanceAngleChain*tiltAngleStep, j);
// 				}

// 				#ifdef DEBUG_CHAINS
// 				std::cout << "landmarkDisplacementThreshold=" << landmarkDisplacementThreshold << std::endl;
// 				#endif

// 				if(chainLineY[j]==1)
// 				{
// 					if(clusterIdSize > 0)
// 					{
// 						#ifdef DEBUG_CHAINS
// 						std::cout << "CASE: chainLineY[j]==1 --> clusterSizeVector.push_back " <<  clusterIdSize << std::endl;
// 						#endif

// 						clusterSizeVector.push_back(clusterIdSize);
// 						clusterIdSize = 0;
// 						clusterId += 1;
// 					}

// 					chainLineY[j] = clusterId;
// 					clusterIdSize += 1;

// 					for (size_t k = 1; k <= landmarkDisplacementThreshold; k++)
// 					{
// 						if(chainLineY[j+k]==1)
// 						{
// 							chainLineY[j+k] = clusterId;
// 							clusterIdSize += 1;
// 						}
// 					}
// 				}

// 				else
// 				{
// 					bool found = false;

// 					for (size_t k = 1; k <= landmarkDisplacementThreshold; k++)
// 					{
// 						// Check for new points added to the clusted
// 						if(chainLineY[j+k]==1)
// 						{
// 							chainLineY[j+k] = clusterId;
// 							clusterIdSize += 1;
// 							found = true;
// 						} 
						
// 						// Check for forward points already belonging to the cluster
// 						else if (chainLineY[j+k]!=0)
// 						{
// 							found = true;
// 						}
// 					}

// 					if (!found)
// 					{
// 						#ifdef DEBUG_CHAINS
// 						std::cout << "CASE: chainLineY[j]!=1 --> claverageusterSizeVector.push_back " <<  clusterIdSize << std::endl;
// 						#endif

// 						clusterSizeVector.push_back(clusterIdSize);
// 						clusterIdSize = 0;
// 						clusterId += 1;
// 					}
// 				}
// 			}
// 		}

// 		if(clusterIdSize>0)
// 		{
// 			#ifdef DEBUG_CHAINS
// 			std::cout << "CASE: clusterIdSize>0 --> clusterSizeVector.push_back " <<  clusterIdSize << std::endl;
// 			#endif

// 			clusterSizeVector.push_back(clusterIdSize);
// 			clusterIdSize = 0;
// 		}

// 		// Complete the overall 2D chain map
// 		#ifdef DEBUG_CHAINS
// 		std::cout << "clusterSizeVector.size()=" << clusterSizeVector.size() << std::endl; 

// 		for (size_t x = 0; x < clusterSizeVector.size(); x++)
// 		{
// 			std::cout << "clusterSizeVector[" << x << "]=" << clusterSizeVector[x] << std::endl;
// 		}
// 		#endif
		
// 		size_t lastClusterId=0, lastIndex=0, firstIndex=0;

// 		for (size_t j = 0; j < chainLineY.size(); j++)
// 		{	
// 			size_t clusterValue=chainLineY[j];

// 			if (clusterSizeVector[clusterValue-2] > numberOfElementsInChainThreshold && clusterValue != 0)
// 			{
// 				#ifdef DEBUG_CHAINS
// 				std::cout << "clusterValue=" << clusterValue << std::endl;
// 				std::cout << "lastClusterId=" << lastClusterId << std::endl;
// 				std::cout << "firstIndex=" << firstIndex << std::endl;
// 				std::cout << "lastIndex=" << lastIndex << std::endl;
// 				std::cout << "-----------------" << std::endl;
// 				#endif

// 				#ifdef DEBUG_OUTPUT_FILES
// 				DIRECT_A2D_ELEM(clustered2dMap, chainIndexY, j) = chainLineY[j];
// 				#endif
				
// 				// New label
// 				if (clusterValue != lastClusterId && lastClusterId != 0)
// 				{
// 					#ifdef DEBUG_CHAINS
// 					std::cout << "Chain with label " << chainLineY[j] << " and label size " << clusterSizeVector[chainLineY[j]-2] << " ADDED" << std::endl;
// 					std::cout << "First index: " << firstIndex << std::endl;
// 					std::cout << "Last index: " << lastIndex << std::endl;
// 					#endif

// 					for (size_t k = firstIndex; k < lastIndex+1; k++)
// 					{
// 						DIRECT_A2D_ELEM(chain2dMap, chainIndexY, k) = 1;
// 					}
					

// 					lastClusterId=clusterValue;
// 					firstIndex = j;
// 					lastIndex = j;
// 				}

// 				// Find first cluster
// 				else if (clusterValue != lastClusterId && lastClusterId == 0)
// 				{
// 					lastClusterId=clusterValue;
// 					firstIndex = j;
// 					lastIndex = j;
// 				}
				

// 				// Add elements to cluster
// 				else if (clusterValue==lastClusterId)
// 				{
// 					lastIndex = j;
// 				}
// 			}

// 			// Add last cluster
// 			if (firstIndex != lastIndex)
// 			{
// 				for (size_t k = firstIndex; k < lastIndex+1; k++)
// 				{
// 					DIRECT_A2D_ELEM(chain2dMap, chainIndexY, k) = 1;
// 				}
// 			}
			
// 		}
		
// 		clusterSizeVector.clear();
// 	}

// 	#ifdef DEBUG_OUTPUT_FILES
// 	size_t lastindex = fnOut.find_last_of("\\/");
// 	std::string rawname = fnOut.substr(0, lastindex);

// 	std::string outputFileNameChain2dMap;
// 	std::string outputFileNameClustered2dMap;

//     outputFileNameChain2dMap = rawname + "/ts_filteredChains.mrc";
//     outputFileNameClustered2dMap = rawname + "/ts_clusteredChains.mrc";

// 	Image<int> saveImageBis;
// 	saveImageBis() = clustered2dMap;
// 	saveImageBis.write(outputFileNameClustered2dMap);
	
// 	Image<int> saveImage;
// 	saveImage() = chain2dMap;
// 	saveImage.write(outputFileNameChain2dMap);
// 	#endif

// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Landmark chains detected succesfully!" << std::endl;
// 	#endif
// }


// void ProgTomoDetectMisalignmentTrajectory::detectMisalignedTiltImages()
// {
// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Detecting misaligned tilt-images..." << std::endl;
// 	#endif

// 	std::vector<Point2D<double>> coordinatesInSlice;
// 	std::vector<size_t> lmOutRange(nSize, 0);

// 	for (size_t n = 0; n < nSize; n++)
// 	{
// 		// Calculate distances
// 		coordinatesInSlice = getCoordinatesInSlice(n);

// 		if(coordinatesInSlice.size() > 0)
// 		{

// 			#ifdef DEBUG_LOCAL_MISALI
// 			// Vector holding the distance of each landmark to its closest chain
// 			std::vector<double> vectorDistance;
// 			#endif

// 			for (size_t coord = 0; coord < coordinatesInSlice.size(); coord++)
// 			{
// 				Point2D<double> coord2D = coordinatesInSlice[coord];
// 				size_t matchCoordX = (size_t)-1; // Maximum possible size_t datatype
// 				size_t matchCoordY = (size_t)-1; // Maximum possible size_t datatype

// 				bool found = false;

// 				// Find distance to closest neighbour
// 				// *** optimizar: cada vez que se aumenta la distancia se revisitan los pixeles ya comprobados en distancias menores
// 				for (int distance = 1; distance < thrChainDistancePx; distance++)
// 				{
// 					for (int i = -distance; i <= distance; i++)
// 					{
// 						for (int j = -distance; j <= distance; j++)
// 						{
// 							if (j + coord2D.y > 0 && i + coord2D.x  > 0 && j + coord2D.y < ySize && i + coord2D.x < xSize && i*i+j*j <= distance*distance)
// 							{
// 								if (DIRECT_A2D_ELEM(chain2dMap, (int)(j + coord2D.y), (int)(i + coord2D.x)) != 0)
// 								{
// 									if(std::min(matchCoordX, matchCoordY) > std::min(i, j))
// 									{
// 										#ifdef DEBUG_LOCAL_MISALI
// 										//std::cout << "Found!! (" <<j<<"+"<<coord2D.y<<", "<<i<<"+"<<coord2D.x<<", "<< n << ")" << std::endl;
// 										#endif

// 										found = true;
// 										matchCoordX = i;
// 										matchCoordX = j;

// 										// Here we could break the loop but we do not to get the minimum distance to a chain (as a measurement of quality)***
// 										// *** Tal vez sería mas interesante medir la distancia media de la coordenadas y no cuantas están más alla de una distancia, nos
// 										// ahorraríamos un threshold.
// 										// break;
// 									}
// 								}
// 							}
// 						}
// 					}
					
// 					if(found)
// 					{
// 						break;
// 					}
// 				}

// 				if(!found)
// 				{
// 					#ifdef DEBUG_LOCAL_MISALI
// 					//std::cout << "Not found!! (" <<coord2D.y<<", "<<coord2D.x<<", "<< n << ")" << std::endl;
// 					vectorDistance.push_back(0);
// 					#endif

// 					lmOutRange[n] += 1;
// 				}
// 				#ifdef DEBUG_LOCAL_MISALI
// 				else
// 				{
// 					vectorDistance.push_back(sqrt(matchCoordX*matchCoordX + matchCoordY*matchCoordY));
// 				}
// 				#endif
// 					}

// 			#ifdef DEBUG_LOCAL_MISALI
// 			for (size_t i = 0; i < vectorDistance.size(); i++)
// 			{
// 				std::cout << vectorDistance[i] << "  ";
// 			}
			
// 			std::cout << "\nlmOutRange[" << n << "]=" << lmOutRange[n] << "/" << coordinatesInSlice.size() << "=" << 
// 			float(lmOutRange[n])/float(coordinatesInSlice.size()) << "\n"<< std::endl;
// 			#endif
// 		}
// 		else
// 		{
// 			#ifdef VERBOSE_OUTPUT
// 			std::cout << "No landmarks detected in slice " << n << ". IMPOSSIBLE TO DETECT POTENTIAL MISALIGNMENT IN THIS IMAGE." << std::endl;
// 			#endif

// 			lmOutRange[n] = 0;
// 			#ifdef DEBUG_LOCAL_MISALI
// 			std::cout << "lmOutRange[" << n << "]=" << lmOutRange[n] << "\n"<< std::endl;
// 			#endif
// 		}
// 	}

// 	// Detect misalignment
// 	double sum = 0, sum2 = 0;
// 	int Nelems = 0;
// 	double m = 0;
// 	double sd = 0;

// 	for(size_t n = 0; n < nSize; n++)
// 	{
// 		int value = lmOutRange[n];
// 		sum += value;
// 		sum2 += value*value;
// 		++Nelems;
// 	}

// 	m = sum / nSize;
// 	sd = sqrt(sum2/Nelems - m*m);

// 	for(size_t n = 0; n < nSize; n++)
// 	{
// 		if(lmOutRange[n] > (m + 3*sd))
// 		{
// 			localAlignment[n] = false;

// 			#ifdef VERBOSE_OUTPUT
// 			std::cout << "MISALIGNMENT DETECTED IN IMAGE " << n << std::endl;
// 			#endif
// 		}
// 	}

// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Misalignment in tilt-images succesfully detected!" << std::endl;
// 	#endif
// }

// void ProgTomoDetectMisalignmentTrajectory::writeOutputCoordinates()
// {
// 	size_t lastindex = fnOut.find_last_of("\\/");
// 	std::string rawname = fnOut.substr(0, lastindex);
// 	std::string outputFileNameLandmarkCoordinates;
//     outputFileNameLandmarkCoordinates = rawname + "/ts_landmarkCoordinates.xmd";

// 	MetaDataVec md;
// 	size_t id;

// 	for(size_t i = 0; i < coordinates3D.size(); i++)
// 	{
// 		id = md.addObject();
// 		md.setValue(MDL_XCOOR, (int)coordinates3D[i].x, id);
// 		md.setValue(MDL_YCOOR, (int)coordinates3D[i].y, id);
// 		md.setValue(MDL_ZCOOR, (int)coordinates3D[i].z, id);
// 	}


// 	md.write(outputFileNameLandmarkCoordinates);
	
// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Output coordinates metadata saved at: " << outputFileNameLandmarkCoordinates << std::endl;
// 	#endif
// }



// void ProgTomoDetectMisalignmentTrajectory::detectInterpolationEdges(MultidimArray<double> &tiltImage)
// {
// 	// Detect interpolation region
// 	MultidimArray<double> tmpImage = tiltImage;

// 	for (size_t i = 1; i < xSize-1; i++)
// 	{
// 		for (size_t j = 1; j < ySize-1; j++)
// 		{
// 			DIRECT_A2D_ELEM(tmpImage, j ,i) = (-1 * DIRECT_A2D_ELEM(tiltImage, j-1 ,i) +
// 											   -1 * DIRECT_A2D_ELEM(tiltImage, j+1 ,i) +
// 											   -1 * DIRECT_A2D_ELEM(tiltImage, j ,i-1) +
// 											   -1 * DIRECT_A2D_ELEM(tiltImage, j ,i+1) +
// 									 		    4 * DIRECT_A2D_ELEM(tiltImage, j ,i));
// 		}
// 	}

	
// 	// Background value as the median of the corners
// 	std::vector<double> corners{DIRECT_A2D_ELEM(tiltImage, 0, 0),
// 								DIRECT_A2D_ELEM(tiltImage, 0, xSize-1),
// 								DIRECT_A2D_ELEM(tiltImage, ySize-1, 0),
// 								DIRECT_A2D_ELEM(tiltImage, ySize-1, xSize-1)};

// 	sort(corners.begin(), corners.end(), std::greater<double>());

// 	double backgroundValue = (corners[1]+corners[2])/2;

// 	// Margin thickness
// 	int marginThickness = (int)(fiducialSizePx * 0.5);

// 	auto epsilon = MINDOUBLE;

// 	std::vector<Point2D<int>> interpolationLimits;

// 	bool firstLimitFound;

// 	int xMin;
// 	int xMax;

// 	for (size_t j = 1; j < ySize-2; j++)
// 	{
// 		for (size_t i = 1; i < xSize-1; i++)
// 		{
// 			if(abs(DIRECT_A2D_ELEM(tmpImage, j, i)) > epsilon)
// 			{
// 				xMin = ((i + marginThickness)>(xSize-1)) ? (xSize-1) : (i + marginThickness);

// 				// Fill margin thickness with background value
// 				for (size_t a = i; a < i + marginThickness; a++)
// 				{
// 					DIRECT_A2D_ELEM(tiltImage, j, a) = backgroundValue;
// 				}
				
// 				break;
// 			}
// 		}


// 		for (size_t i = xSize-1; i > 1; i--)
// 		{
// 			if(abs(DIRECT_A2D_ELEM(tmpImage, j, i)) > epsilon)
// 			{
// 				xMax = ((i - marginThickness)<0) ? 0 : (i - marginThickness);

// 				// Fill margin thickness with background value
// 				for (size_t a = i - marginThickness; a < i; a++)
// 				{
// 					DIRECT_A2D_ELEM(tiltImage, j, a) = backgroundValue;
// 				}

// 				break;
// 			}
// 		}

// 		if (xMin >= xMax)
// 		{
// 			int value = (int) (((xMax+marginThickness)+(xMin-marginThickness))/2);
// 			xMax = value;
// 			xMin = value;
// 		}
		
// 		Point2D<int> limit (xMin, xMax);
// 		interpolationLimits.push_back(limit);
// 	}

// 	interpolationLimitsVector.push_back(interpolationLimits);
// }


// bool ProgTomoDetectMisalignmentTrajectory::filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY)
// {
// 	// Uncomment for phantom
// 	// return true;

// 	// *** TODO: filtering by number of coordinates might not be the best method, i could be better just filter by occupancy and max distance (radius)

// 	// Check number of elements of the label <----------------------------*** ESTO NO HACE NADA CON EL THR ACTUAL
// 	// if(coordinatesPerLabelX.size() < thrNumberCoords)
// 	// {
// 	// 	return false;
// 	// }

// 	// Calculate the furthest point of the region from the centroid
// 	double maxSquareDistance = 0;
// 	double distance;

// 	#ifdef DEBUG_FILTERLABEL
// 	size_t debugN;
// 	#endif

// 	for(size_t n = 0; n < coordinatesPerLabelX.size(); n++)
// 	{
// 		distance = (coordinatesPerLabelX[n]-centroX)*(coordinatesPerLabelX[n]-centroX)+(coordinatesPerLabelY[n]-centroY)*(coordinatesPerLabelY[n]-centroY);

// 		if(distance >= maxSquareDistance)
// 		{
// 			#ifdef DEBUG_FILTERLABEL
// 			debugN = n;
// 			#endif

// 			maxSquareDistance = distance;
// 		}
// 	}

// 	double maxDistace;
// 	maxDistace = sqrt(maxSquareDistance);

// 	// Check sphericity of the labeled region
// 	double circumscribedArea = PI * (maxDistace * maxDistace);;
// 	double area = 0.0 + (double)coordinatesPerLabelX.size();
// 	double ocupation;

// 	ocupation = area / circumscribedArea;

// 	#ifdef DEBUG_FILTERLABEL
// 	std::cout << "debugN " << debugN << std::endl;
// 	std::cout << "x max distance " << coordinatesPerLabelX[debugN] << std::endl;
// 	std::cout << "y max distance " << coordinatesPerLabelY[debugN] << std::endl;
// 	std::cout << "centroX " << centroX << std::endl;
// 	std::cout << "centroY " << centroY << std::endl;
// 	std::cout << "area " << area << std::endl;
// 	std::cout << "circumscribedArea " << circumscribedArea << std::endl;
// 	std::cout << "maxDistace " << maxDistace << std::endl;
// 	std::cout << "ocupation " << ocupation << std::endl;
// 	#endif

// 	if(ocupation < 0.5)
// 	{
// 		#ifdef DEBUG_FILTERLABEL
// 		std::cout << "COORDINATE REMOVED AT " << centroX << " , " << centroY << " BECAUSE OF OCCUPATION"<< std::endl;
// 		#endif
// 		return false;
// 	}

// 	// Check the relative area compared with the expected goldbead
// 	double expectedArea = PI * ((fiducialSizePx/2) * fiducialSizePx/2);
// 	double relativeArea = (4*area)/expectedArea;  // Due to filtering and labelling processes labeled gold beads tend to reduce its radius in half

// 	#ifdef DEBUG_FILTERLABEL
// 	std::cout << "expectedArea " << expectedArea << std::endl;
// 	std::cout << "relativeArea " << relativeArea << std::endl;
// 	std::cout << "-------------------------------------------"  << std::endl;
// 	#endif


// 	if (relativeArea > 4 || relativeArea < 0.1)
// 	{
// 		#ifdef DEBUG_FILTERLABEL
// 		std::cout << "COORDINATE REMOVED AT " << centroX << " , " << centroY << " BECAUSE OF RELATIVE AREA"<< std::endl;
// 		#endif
// 		return false;
// 	}
// 	#ifdef DEBUG_FILTERLABEL
// 	std::cout << "COORDINATE NO REMOVED AT " << centroX << " , " << centroY << std::endl;
// 	#endif
// 	return true;
// }


// std::vector<size_t> ProgTomoDetectMisalignmentTrajectory::getCoordinatesInSliceIndex(size_t slice)
// {
// 	std::vector<size_t> coordinatesInSlice;

// 	for(size_t n = 0; n < coordinates3D.size(); n++)
// 	{
// 		if(slice == coordinates3D[n].z)
// 		{
// 			coordinatesInSlice.push_back(n);
// 		}
// 	}

// 	return coordinatesInSlice;
// }


// float ProgTomoDetectMisalignmentTrajectory::testPoissonDistribution(float lambda, size_t k)
// {
// 	double quotient=1;

// 	#ifdef DEBUG_POISSON
// 	std::cout << "k="<< k <<std::endl;
// 	std::cout << "lambda="<< lambda <<std::endl;
// 	#endif

// 	// Since k! can not be holded we calculate the quotient lambda^k/k!= (lambda/k) * (lambda/(k-1)) * ... * (lambda/1)
// 	for (size_t i = 1; i < k+1; i++)
// 	{
// 		quotient *= lambda / i;
// 	}

// 	#ifdef DEBUG_POISSON
// 	std::cout << "quotient="<< quotient <<std::endl;
// 	std::cout << "quotient*exp(-lambda)="<< quotient*exp(-lambda) <<std::endl;
// 	#endif
	
// 	return quotient*exp(-lambda);

// }


// float ProgTomoDetectMisalignmentTrajectory::calculateLandmarkProjectionDiplacement(float theta1, float theta2, float coordinateProjX)
// {
// 	float xCoor = coordinateProjX - xSize/2;

// 	#ifdef DEBUG_CHAINS
// 	std::cout << "coordinateProjX=" << coordinateProjX << std::endl;
// 	std::cout << "xCoor=" << xCoor << std::endl;
// 	std::cout << "theta1=" << theta1 << std::endl;
// 	std::cout << "theta2=" << theta2 << std::endl;	
// 	std::cout << "theta1 * PI/180.0=" << theta1 * PI/180.0 << std::endl;
// 	std::cout << "theta2 * PI/180.0=" << theta2 * PI/180.0 << std::endl;
// 	#endif

// 	float distance = abs(((cos(theta2 * PI/180.0)/cos(theta1 * PI/180.0))-1)*xCoor);

// 	if (distance<minDistancePx)
// 	{
// 		return (int)minDistancePx;
// 	}
	
// 	return (int)distance;
// }


// void ProgTomoDetectMisalignmentTrajectory::detectMisalignmentFromResidualsBis()
// {
// 	double mod2Thr = (fiducialSizePx * thrFiducialDistance) * (fiducialSizePx * thrFiducialDistance);

// 	// Global alignment analysis
// 	std::vector<bool> globalMialingmentVotting(numberOfInputCoords, true);  // Vector saving status of (mis)aligned chains
// 	float vottingRatio;

// 	for (size_t n = 0; n < numberOfInputCoords; n++)
// 	{
// 		std::vector<CM> CM_fid;
// 		getCMbyFiducial(n, CM_fid);

// 		size_t numberCM = CM_fid.size();

// 		double avg;
// 		double std;
// 		size_t imagesOutOfRange = 0;

// 		double sumResid = 0;
// 		double sumResid2 = 0;

// 		for (size_t i = 0; i < numberCM; i++)
// 		{
// 			double sum2 = CM_fid[i].residuals.x*CM_fid[i].residuals.x + CM_fid[i].residuals.y*CM_fid[i].residuals.y;
// 			sumResid2 += sum2;
// 			sumResid += sqrt(sum2);

// 			if (sum2 > mod2Thr)
// 			{
// 				imagesOutOfRange += 1;
// 			}
// 		}

// 		avg = sumResid / numberCM;
// 		std = sqrt(sumResid2 / numberCM - avg * avg);

// 		#ifdef DEBUG_RESIDUAL_ANALYSIS
// 		std::cout << "n " << n << std::endl;
// 		std::cout << "numberCM " << numberCM << std::endl;
// 		std::cout << "sumResid " << sumResid << std::endl;
// 		std::cout << "sumResid2 " << sumResid2 << std::endl;
// 		std::cout << "imagesOutOfRange " << imagesOutOfRange << std::endl;
// 		std::cout << "avg " << avg << std::endl;
// 		std::cout << "std " << std << std::endl;
// 		#endif

// 		if (imagesOutOfRange > 4.5 && std > 43.5)
// 		{
// 			globalMialingmentVotting[n] = false;
			
// 			#ifdef DEBUG_RESIDUAL_ANALYSIS
// 			std::cout << "Chain number " << n << " present global misalignment with std=" << std << " and imagesOutOfRange=" << imagesOutOfRange << std::endl;
// 			#endif			
// 		}
// 		else if (imagesOutOfRange > 6.5)
// 		{
// 			globalMialingmentVotting[n] = false;

// 			#ifdef DEBUG_RESIDUAL_ANALYSIS
// 			std::cout << "Chain number " << n << " present global misalignment with imagesOutOfRange=" << imagesOutOfRange << std::endl;
// 			#endif
// 		}	
// 	}

// 	#ifdef DEBUG_RESIDUAL_ANALYSIS
// 	std::cout << "globalMialingmentVotting" << std::endl;
// 	#endif

// 	for (size_t n = 0; n < numberOfInputCoords; n++)
// 	{
// 		#ifdef DEBUG_RESIDUAL_ANALYSIS
// 		std::cout << globalMialingmentVotting[n] << " ";
// 		#endif

// 		if (globalMialingmentVotting[n])
// 		{
// 			vottingRatio += 1;
// 		}
		
// 	}

// 	vottingRatio /= numberOfInputCoords;

// 	#ifdef DEBUG_RESIDUAL_ANALYSIS
// 	std::cout  << "\n votting ratio " << vottingRatio << std::endl;
// 	#endif

// 	if (vottingRatio < 0.5)
// 	{
// 		globalAlignment = false;

// 		#ifdef VERBOSE_OUTPUT
// 		std::cout << "GLOBAL MISLAIGNMENT DETECTED" << std::endl;
// 		#endif

// 		return;
// 	}

// 	#ifdef DEBUG_RESIDUAL_ANALYSIS
// 	std::cout << "Output global (chain) alingmnet vector" << std::endl;
// 	for (size_t n = 0; n < numberOfInputCoords; n++)
// 	{
// 		std::cout << globalMialingmentVotting[n] << ", ";
// 	}
// 	std::cout << std::endl;
// 	#endif

// 	// Local alignment analysis
// 	std::vector<float> resid2Vector;  // Vector containing residual^2 values from the aligned landmark chains

// 	for (size_t n = 0; n < numberOfInputCoords; n++)
// 	{
// 		if (globalMialingmentVotting[n])
// 		{
// 			std::vector<CM> CM_fid;
// 			getCMbyFiducial(n, CM_fid);

// 			size_t numberCM = CM_fid.size();

// 			for (size_t i = 0; i < numberCM; i++)
// 			{
// 				resid2Vector.push_back(CM_fid[i].residuals.x*CM_fid[i].residuals.x + CM_fid[i].residuals.y*CM_fid[i].residuals.y);
// 			}
// 		}
// 	}
		
// 	bool elementRemoved;
	

// 	// Remove outliers from resid2Vector
// 	// do
// 	// {
// 	// 	elementRemoved = false;

// 	// 	size_t resid2Vector_size = resid2Vector.size();
// 	// 	double sum2resid2 = 0;
// 	// 	double sumResid2 = 0;

// 	// 	for (size_t i = 0; i < resid2Vector_size; i++)
// 	// 	{
// 	// 		double sum = resid2Vector[i];
// 	// 		sum2resid2 += sum*sum;
// 	// 		sumResid2 += sum;
// 	// 	}

// 	// 	double resid2avg = sumResid2 / resid2Vector_size;
// 	// 	double resid2std = sqrt(sum2resid2 / resid2Vector_size - resid2avg * resid2avg);

// 	// 	double resid2thr_high = resid2avg + 2*resid2std;
// 	// 	double resid2thr_low = resid2avg - 2*resid2std;

// 	// 	std::cout << "resid2thr_high " << resid2thr_high << std::endl;
// 	// 	std::cout << "resid2thr_low " << resid2thr_low << std::endl;

// 	// 	for (size_t i = 0; i < resid2Vector.size(); i++)
// 	// 	{
// 	// 		if (resid2Vector[i] > resid2thr_high || resid2Vector[i] < resid2thr_low)
// 	// 		{
// 	// 			resid2Vector.erase(resid2Vector.begin()+i);
// 	// 			elementRemoved = true;
// 	// 		}
// 	// 	}

// 	// 	#ifdef DEBUG_RESIDUAL_ANALYSIS
// 	// 	std::cout << "RESIDUALS -------------------------------- size: " << resid2Vector_size << std::endl;
// 	// 	for (size_t j = 0; j < resid2Vector_size; j++)
// 	// 	{
// 	// 		// std::cout << resid2Vector[j] << std::endl;
// 	// 	}
// 	// 	#endif

// 	// 	std::cout << "elementRemoved " << elementRemoved <<  std::endl;

// 	// }while(elementRemoved);
	
// 	size_t resid2Vector_size = resid2Vector.size();
// 	sort(resid2Vector.begin(), resid2Vector.end());

// 	#ifdef DEBUG_RESIDUAL_ANALYSIS
// 	std::cout << "SORTED RESIDUALS -------------------------------- size: " << resid2Vector_size << std::endl;
// 	for (size_t j = 0; j < resid2Vector_size; j++)
// 	{
// 		std::cout << resid2Vector[j] << std::endl;
// 	}
// 	#endif
	
// 	for (size_t n = 0; n < nSize; n++)
// 	{
// 		std::vector<CM> CM_image;
// 		getCMbyImage(n, CM_image);

// 		size_t numberCM = CM_image.size();
// 		double vottingRatio = 0;

// 		#ifdef DEBUG_RESIDUAL_ANALYSIS
// 		std::cout << "------------ Analyzing image " << n << ". Presenting " << numberCM << " coordinates." <<  std::endl;
// 		#endif

// 		if (numberCM > 0)
// 		{
// 			for (size_t i = 0; i < numberCM; i++)
// 			{
// 				double resid2 = CM_image[i].residuals.x*CM_image[i].residuals.x + CM_image[i].residuals.y*CM_image[i].residuals.y;

// 				bool found = false;

// 				for (size_t j = 0; j < resid2Vector_size; j++)
// 				{
// 					if(resid2 < resid2Vector[j])
// 					{
// 						#ifdef DEBUG_RESIDUAL_ANALYSIS
// 						std::cout << "residual value " << resid2 << " position " << j << "percentile " << (double(j)/double(resid2Vector_size)) << std::endl;
// 						#endif

// 						if ((double(j)/double(resid2Vector_size)) > avgResidPercentile_LocalAlignment)
// 						{
// 							vottingRatio += 1;
// 						}
// 						found = true;
// 						break;	
// 					}
// 				}

// 				if (!found)  // In case resid2 is bigger than the biggest element in resid2Vector
// 				{
// 					std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!resid2 is bigger than the biggest element in resid2Vector" << std::endl;
// 					vottingRatio += 1;
// 				}
				
// 			}

// 			vottingRatio /= float(numberCM);
// 			std::cout << "-------- For image " << n << " votting ratio=" << vottingRatio << " out of " << numberCM << std::endl;


// 			if (vottingRatio > 0.5)
// 			{
// 				localAlignment[n] = false;

// 				#ifdef VERBOSE_OUTPUT
// 				std::cout << "LOCAL MISLAIGNMENT DETECTED AT TILT-IMAGE " << n << ". Failed residuals ratio: " << vottingRatio << " out of " << numberCM << std::endl;
// 				#endif
// 			}
			
// 		}
// 		else
// 		{
// 			#ifdef VERBOSE_OUTPUT
// 			std::cout << "UNDETECTED COORDINATES IN TILT-IMAGE " << n << ". IMPOSSIBLE TO STUDY MIALIGNMENT" << std::endl;
// 			#endif
// 		}
// 	}
// }