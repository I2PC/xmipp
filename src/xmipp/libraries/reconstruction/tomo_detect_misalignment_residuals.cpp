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

#include "tomo_detect_misalignment_residuals.h"
#include <chrono>



// --------------------------- INFO functions ----------------------------

void ProgTomoDetectMisalignmentResiduals::readParams()
{
	// fnInputTS = getParam("-i");
	fnResidualInfo = getParam("--inputResInfo");
    fnOut = getParam("-o");

	samplingRate = getDoubleParam("--samplingRate");
	fiducialSize = getDoubleParam("--fiducialSize");


	thrFiducialDistance = getDoubleParam("--thrFiducialDistance");
}


void ProgTomoDetectMisalignmentResiduals::defineParams()
{
	addUsageLine("This function determines the location of high contrast features in a volume.");
	// addParamsLine("  -i <mrcs_file=\"\">                   					: Input tilt-series.");
	addParamsLine("  --inputResInfo <input=\"\">							: Input file containing residual information of the detected landmarks.");

	addParamsLine("  [-o <output=\"./alignemntReport.xmd\">]       			: Output file containing the alignemnt report.");

	addParamsLine("  [--samplingRate <samplingRate=1>]						: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--fiducialSize <fiducialSize=100>]					: Fiducial size in Angstroms (A).");

	addParamsLine("  [--thrFiducialDistance <thrFiducialDistance=0.5>]		: Threshold times of fiducial size as maximum distance to consider a match between the 3d coordinate projection and the detected fiducial.");
}


void ProgTomoDetectMisalignmentResiduals::generateSideInfo()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Generating side info..." << std::endl;
	#endif

	// Update thresholds depending on input tilt-series sampling rate
	fiducialSizePx = fiducialSize / samplingRate; 

	#ifdef VERBOSE_OUTPUT
	std::cout << "fiducialSizePx: "<< fiducialSizePx << std::endl;
	#endif

	// Read input residuals file
	readInputResiduals();

	// Get number of inpur coordinates from input residuals
	numberOfInputCoords = 0;

	for (size_t i = 0; i < vResMod.size(); i++)
	{
		if (numberOfInputCoords < vResMod[i].id)
		{
			numberOfInputCoords = vResMod[i].id;
		}
	}

	numberOfInputCoords += 1;

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of input coordinates: " << numberOfInputCoords << std::endl;
	#endif

	// Get number of tilt images from input residuals
	nSize = 0;
	for (size_t i = 0; i < vResMod.size(); i++)
	{
		if (vResMod[i].landmarkCoord.z > nSize)
		{
			nSize = vResMod[i].landmarkCoord.z;
		}
	}

	nSize += 1;

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of tilt-images: " << nSize << std::endl;
	#endif

	// Initialize local alignment vector (depends on the number of acquisition angles)
	localAlignment.resize(nSize, true);
	avgMahalanobisDistanceV.resize(nSize, 0.0);
	stdMahalanobisDistanceV.resize(nSize, 0.0);

	#ifdef VERBOSE_OUTPUT
	std::cout << "Side info generated succesfully!" << std::endl;
	#endif
}



// --------------------------- HEAD functions ----------------------------

void ProgTomoDetectMisalignmentResiduals::detectMisalignmentFromResiduals()
{
	double mod2Thr = (fiducialSizePx * thrFiducialDistance) * (fiducialSizePx * thrFiducialDistance);

	// Global alignment analysis
	std::vector<bool> globalMialingmentVotting(numberOfInputCoords, true);  // Vector saving status of (mis)aligned chains
	float vottingRatio;

	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		std::vector<resMod> resMod_fid;
		getResModByFiducial(n, resMod_fid);

		size_t numberResMod = resMod_fid.size();

		double avg;
		double std;
		size_t imagesOutOfRange = 0;

		double sumResid = 0;
		double sumResid2 = 0;

		for (size_t i = 0; i < numberResMod; i++)
		{
			double sum2 = resMod_fid[i].residuals.x*resMod_fid[i].residuals.x + resMod_fid[i].residuals.y*resMod_fid[i].residuals.y;
			sumResid2 += sum2;
			sumResid += sqrt(sum2);

			if (sum2 > mod2Thr)
			{
				imagesOutOfRange += 1;
			}
		}

		avg = sumResid / numberResMod;
		std = sqrt(sumResid2 / numberResMod - avg * avg);

		#ifdef DEBUG_RESIDUAL_ANALYSIS
		std::cout << "n " << n << std::endl;
		std::cout << "numberResMod " << numberResMod << std::endl;
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
			std::cout << "Chain number " << n << " ssent global misalignment with imagesOutOfRange=" << imagesOutOfRange << std::endl;
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
		std::vector<resMod> resMod_image;
		getResModByImage(n, resMod_image);

		size_t numberResMod = resMod_image.size();
		double vottingRatio = 0;

		#ifdef DEBUG_RESIDUAL_ANALYSIS
		std::cout << "------------ Analyzing image " << n << ". Presenting " << numberResMod << " coordinates." <<  std::endl;
		#endif

		if (numberResMod > 0)
		{
			for (size_t i = 0; i < numberResMod; i++)
			{
				double resid2 = resMod_image[i].residuals.x*resMod_image[i].residuals.x + resMod_image[i].residuals.y*resMod_image[i].residuals.y;

				if(resid2 > mod2Thr)
				{
					vottingRatio += 1;
				}
				
			}

			vottingRatio /= float(numberResMod);
			
			#ifdef DEBUG_RESIDUAL_ANALYSIS
			std::cout << "-------- For image " << n << " votting ratio=" << vottingRatio << " out of " << numberResMod << std::endl;
			#endif

			if (vottingRatio > 0.5)
			{
				localAlignment[n] = false;

				#ifdef VERBOSE_OUTPUT
				std::cout << "LOCAL MISLAIGNMENT DETECTED AT TILT-IMAGE " << n << ". Failed residuals ratio: " << vottingRatio << " out of " << numberResMod << std::endl;
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


void ProgTomoDetectMisalignmentResiduals::detectMisalignmentFromResidualsMahalanobis()
{
	double sigma = fiducialSizePx / 3;	// Sigma for 99% of the points inside the fiducial radius
	double sigma2 = sigma * sigma;

	// Matrix2D<double> covariance_inv;
	// MAT_ELEM(covariance_inv, 0, 0) = 1/sigma2;
	// MAT_ELEM(covariance_inv, 0, 1) = 0;
	// MAT_ELEM(covariance_inv, 1, 0) = 0;
	// MAT_ELEM(covariance_inv, 1, 1) = 1/sigma2;

	// iterate residuals
	for(size_t i = 0; i < vResMod.size(); i++)
	{
		vResMod[i].mahalanobisDistance = sqrt((vResMod[i].residuals.x*vResMod[i].residuals.x)/sigma2 + (vResMod[i].residuals.y*vResMod[i].residuals.y)/sigma2);
	}

	std::cout << "---------------- Distance parameters" << std::endl;
	std::cout << "Sigma2 = " << sigma2 << std::endl;

	// Global alignment analysis
	std::cout << "---------------- Global misalignemnt analysis" << std::endl;

	double rationMisalignedChains = 0;	// 0 -> aligned -- 1 -> misaligned 

	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		std::vector<resMod> resMod_fid;
		getResModByFiducial(n, resMod_fid);

		size_t numberResMod = resMod_fid.size();

		double sumMahaDist = 0;
		double sumMahaDist2 = 0;

		for (size_t i = 0; i < numberResMod; i++)
		{
			sumMahaDist += resMod_fid[i].mahalanobisDistance;
			sumMahaDist2 += resMod_fid[i].mahalanobisDistance * resMod_fid[i].mahalanobisDistance; 
		}

		double avgMahaDist = sumMahaDist / numberResMod;
		double stdMahaDist = sqrt(sumMahaDist2 / numberResMod - avgMahaDist * avgMahaDist);

		if (avgMahaDist > 1)
		{
			rationMisalignedChains += 1;
		}

		std::cout << "Statistics of mahalanobis distances for 3D coordinate " << n << std::endl;
		std::cout << "Average mahalanobis distance: " << avgMahaDist << std::endl;
		std::cout << "STD mahalanobis distance: " << stdMahaDist << std::endl;
	}

	// If more than thrRatioMisalignedChains of coordinates are misaligned, set global misalignment
	double thrRatioMisalignedChains = 0.8;

	if (rationMisalignedChains / numberOfInputCoords > thrRatioMisalignedChains)
	{
		globalAlignment = false;
	}

	std::cout << "------> Global alignment score: " << rationMisalignedChains / numberOfInputCoords << std::endl;
	
	// Local alignment analysis
	std::cout << "---------------- Local misaligment analysis" << std::endl;
	
	for (size_t n = 0; n < nSize; n++)
	{
		std::vector<resMod> resMod_image;
		getResModByImage(n, resMod_image);

		size_t numberResMod = resMod_image.size();

		if (numberResMod > 0)
		{

			double sumMahaDist = 0;
			double sumMahaDist2 = 0;

			for (size_t i = 0; i < numberResMod; i++)
			{
				sumMahaDist += resMod_image[i].mahalanobisDistance;
				sumMahaDist2 += resMod_image[i].mahalanobisDistance * resMod_image[i].mahalanobisDistance; 
			}

			double avgMahaDist = sumMahaDist / numberResMod;
			double stdMahaDist = sqrt(sumMahaDist2 / numberResMod - avgMahaDist * avgMahaDist);

			avgMahalanobisDistanceV[n] = avgMahaDist;
			stdMahalanobisDistanceV[n] = stdMahaDist;

			if (avgMahaDist > 1)
			{
				localAlignment[n] = false;
				std::cout << "------> Local misalignment detected at image: " << n << std::endl;
			}

			std::cout << "Statistics of mahalanobis distances for tilt-image " << n << std::endl;
			std::cout << "Average mahalanobis distance: " << avgMahaDist << std::endl;
			std::cout << "STD mahalanobis distance: " << stdMahaDist << std::endl;
		}
		else
		{
			std::cout << "ERROR: impossible to study misalignment in tilt-image " << n << ". No residuals calculated for this image" << std::endl;
			localAlignment[n] = false;
		}
	}
}


void ProgTomoDetectMisalignmentResiduals::detectMisalignmentFromResidualsMahalanobisRobust()
{
	double sigma = fiducialSizePx / 3;	// Sigma for 99% of the points inside the fiducial radius
	double sigma2 = sigma * sigma;

	double mahaDist;
	double sumMahaDist = 0;
	double sumMahaDist2 = 0;

	double avgMahaDist;
	double stdMahaDist;

	size_t numberResMod = vResMod.size();

	// Calculate a robust threshold so only those residuals < avg + 3*std are considered
	for(size_t i = 0; i < numberResMod; i++)
	{
		mahaDist = sqrt((vResMod[i].residuals.x*vResMod[i].residuals.x)/sigma2 + 
		                (vResMod[i].residuals.y*vResMod[i].residuals.y)/sigma2);
		
		vResMod[i].mahalanobisDistance = mahaDist;
		sumMahaDist += mahaDist;
		sumMahaDist2 += mahaDist * mahaDist; 
	}

	avgMahaDist = sumMahaDist / numberResMod;
	stdMahaDist = sqrt(sumMahaDist2 / numberResMod - avgMahaDist * avgMahaDist);
	double mahaThr = avgMahaDist + 3 * stdMahaDist;

	// Global alignment analysis
	std::cout << "---------------- Global misalignemnt analysis" << std::endl;

	double rationMisalignedChains = 0;	// 0 -> aligned -- 1 -> misaligned 

	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		std::vector<resMod> resMod_fid;
		getResModByFiducial(n, resMod_fid);

		size_t numberResMod = resMod_fid.size();

		sumMahaDist = 0;
		sumMahaDist2 = 0;

		for (size_t i = 0; i < numberResMod; i++)
		{
			if (resMod_fid[i].mahalanobisDistance < mahaThr)
			{		
				sumMahaDist += resMod_fid[i].mahalanobisDistance;
				sumMahaDist2 += resMod_fid[i].mahalanobisDistance * resMod_fid[i].mahalanobisDistance;
			}
		}

		avgMahaDist = sumMahaDist / numberResMod;

		if (avgMahaDist > 1)
		{
			rationMisalignedChains += 1;
		}

		std::cout << "Average mahalanobis distance for 3D cooridinate " << n << ": " << avgMahaDist << std::endl;
	}

	// If more than thrRatioMisalignedChains of coordinates are misaligned, set global misalignment
	double thrRatioMisalignedChains = 0.8;

	if (rationMisalignedChains / numberOfInputCoords > thrRatioMisalignedChains)
	{
		globalAlignment = false;
	}

	std::cout << "------> Global alignment score: " << rationMisalignedChains / numberOfInputCoords << std::endl;
	
	// Local alignment analysis
	std::cout << "---------------- Local misaligment analysis" << std::endl;
	
	for (size_t n = 0; n < nSize; n++)
	{
		std::vector<resMod> resMod_image;
		getResModByImage(n, resMod_image);

		size_t numberResMod = resMod_image.size();

		if (numberResMod > 1)
		{
			sumMahaDist = 0;
			sumMahaDist2 = 0;

			for (size_t i = 0; i < numberResMod; i++)
			{
				if (resMod_image[i].mahalanobisDistance < mahaThr)
				{
					sumMahaDist += resMod_image[i].mahalanobisDistance;
					sumMahaDist2 += resMod_image[i].mahalanobisDistance * resMod_image[i].mahalanobisDistance;
				}
			}

			avgMahaDist = sumMahaDist / numberResMod;
			double stdMahaDist = sqrt(sumMahaDist2 / numberResMod - avgMahaDist * avgMahaDist);

			avgMahalanobisDistanceV[n] = avgMahaDist;
			stdMahalanobisDistanceV[n] = stdMahaDist;

			if (avgMahaDist > 1)
			{
				localAlignment[n] = false;
				std::cout << "------> Local misalignment detected at image: " << n << std::endl;
			}

			std::cout << "Statistics of mahalanobis distances for tilt-image " << n << std::endl;
			std::cout << "Average mahalanobis distance: " << avgMahaDist << std::endl;
			std::cout << "STD mahalanobis distance: " << stdMahaDist << std::endl;
		}
		else
		{
			std::cout << "ERROR: impossible to study misalignment in tilt-image " << n << ". Number of residuals for this image: " << numberResMod << std::endl;
			localAlignment[n] = false;
		}
	}
}


void ProgTomoDetectMisalignmentResiduals::contructResidualMatrix()
{
	Matrix2D<double> xResidMat(numberOfInputCoords, nSize);
	Matrix2D<double> yResidMat(numberOfInputCoords, nSize);

	for (size_t n = 0; n < vResMod.size(); n++)
	{
		resMod rm = vResMod[n];

		MAT_ELEM(xResidMat, (int)rm.id, rm.landmarkCoord.z) = rm.residuals.x;
		MAT_ELEM(yResidMat, (int)rm.id, rm.landmarkCoord.z) = rm.residuals.y;
	}

	// CODE TO SAVE MATRIX TO FILES
	// ---------------------------------------------------------------------
	std::string matrixFileX;
	std::string matrixFileY;

	size_t li = fnOut.find_last_of("\\/");
	std::string fileBaseName = fnOut.substr(0, li);
	fileBaseName = fileBaseName.substr(0, li);

	matrixFileX = fileBaseName + "/matrixX.txt";
	matrixFileY = fileBaseName + "/matrixY.txt";

    std::ofstream outputFileX(matrixFileX);
    std::ofstream outputFileY(matrixFileY);
    
    for (size_t i = 0; i < numberOfInputCoords; ++i) {
        for (size_t j = 0; j < nSize; ++j) {
            outputFileX << MAT_ELEM(xResidMat, i, j);
            outputFileY << MAT_ELEM(yResidMat, i, j);

            // Add a tab separator unless it's the last element in the row
            if (j < nSize-1) {
                outputFileX << '\t';
                outputFileY << '\t';
            }
        }
        outputFileX << '\n'; // Start a new line for the next row
        outputFileY << '\n'; // Start a new line for the next row
    }

    outputFileX.close();
    outputFileY.close();
	// ---------------------------------------------------------------------
}


void ProgTomoDetectMisalignmentResiduals::generateResidualStatiscticsFile()
{
	// CODE FOR GENERATING RESIDUAL STATISTICS FILE FOR DECISION TREE TRAINING
	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	// Run XmippScript for statistical residual analysis
	std::cout << "\nRunning residual statistical analysis..." << std::endl;

	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);

	std::string fnStats;
	fnStats = rawname + "/residualStatistics.xmd";

	std::string cmd;

	#ifdef DEBUG_RESIDUAL_STATISTICS_FILE
	// Debug command
	cmd = "python3 /home/fdeisidro/xmipp_devel/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnResidualInfo + " -o " + fnStats + " --debug ";
	// cmd = "python3 /home/fdeisidro/data/xmipp/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnResidualInfo + " -o " + fnStats + " --debug";
	#else
	// No debug command
	cmd = "python3 /home/fdeisidro/xmipp_devel/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnResidualInfo + " -o " + fnStats;
	// cmd = "python3 /home/fdeisidro/data/xmipp/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py -i " + fnResidualInfo + " -o " + fnStats;
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
		std::vector<resMod> resMod_fid;
		getResModByFiducial(n, resMod_fid);

		size_t numberResMod = resMod_fid.size();

		double avg;
		double std;
		size_t imagesOutOfRange = 0;
		double imagesOutOfRangePercentage;
		size_t longestMisaliChain = 0;
		double longestMisaliChainPercentage;
		size_t misaliChain = 0;

		double sumResid = 0;
		double sumResid2 = 0;

		for (size_t i = 0; i < numberResMod; i++)
		{
			double sum2 = resMod_fid[i].residuals.x*resMod_fid[i].residuals.x + resMod_fid[i].residuals.y*resMod_fid[i].residuals.y;
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

		imagesOutOfRangePercentage = double(imagesOutOfRange) / nSize;
		longestMisaliChainPercentage = double(longestMisaliChain) / nSize;

		#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
		std::cout << "n " << n << std::endl;
		std::cout << "numberResMod " << numberResMod << std::endl;
		std::cout << "sumResid " << sumResid << std::endl;
		std::cout << "sumResid2 " << sumResid2 << std::endl;
		std::cout << "imagesOutOfRange " << imagesOutOfRange << std::endl;
		std::cout << "longestMisaliChain " << longestMisaliChain << std::endl;
		std::cout << "imagesOutOfRangePercentage " << imagesOutOfRangePercentage << std::endl;
		std::cout << "longestMisaliChainPercentage " << longestMisaliChainPercentage << std::endl;
		#endif

		avg = sumResid / numberResMod;
		std = sqrt(sumResid2 / numberResMod - avg * avg);

		#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
		std::cout << "avg " << avg << std::endl;
		std::cout << "std " << std << std::endl;
		#endif

		residualStatsTable[n][0] = avg;
		residualStatsTable[n][1] = std;
		residualStatsTable[n][8] = imagesOutOfRangePercentage;
		residualStatsTable[n][9] = longestMisaliChainPercentage;
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
		std::vector<resMod> resMod_image;
		getResModByImage(n, resMod_image);

		size_t numberResMod = resMod_image.size();

		if (numberResMod > 0)
		{
			double avg;
			double std;
			size_t residOutOfRange = 0;
			double residOutOfRangePercentage = 0;

			double sumResid = 0;
			double sumResid2 = 0;

			for (size_t i = 0; i < numberResMod; i++)
			{
				double sum = resMod_image[i].residuals.x*resMod_image[i].residuals.x + resMod_image[i].residuals.y*resMod_image[i].residuals.y;
				sumResid2 += sum;
				sumResid += sqrt(sum);

				if (sumResid > mod2Thr)
				{
					residOutOfRange += 1;
				}
			}

			residOutOfRangePercentage = double(residOutOfRange) / nSize;
			
			#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
			std::cout << "n " << n << std::endl;
			std::cout << "numberResMod " << numberResMod << std::endl;
			std::cout << "sumResid " << sumResid << std::endl;
			std::cout << "sumResid2 " << sumResid2 << std::endl;
			std::cout << "residOutOfRange " << residOutOfRange << std::endl;
			std::cout << "residOutOfRangePercentage " << residOutOfRangePercentage << std::endl;
			#endif

			avg = sumResid / numberResMod;
			std = sqrt(sumResid2 / numberResMod - avg * avg);

			#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
			std::cout << "avg " << avg << std::endl;
			std::cout << "std " << std << std::endl;
			#endif

			imageStatsTable[n][0] = avg;
			imageStatsTable[n][1] = std;
			imageStatsTable[n][7] = residOutOfRangePercentage;
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

void ProgTomoDetectMisalignmentResiduals::readInputResiduals()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Reading input residuals from " << fnResidualInfo  << std::endl;
	#endif

	MetaDataVec md;
	md.read(fnResidualInfo);

	size_t id;

	double lmX;
	double lmY;
	double lmZ;

	int coord3dx;
	int coord3dy;
	int coord3dz;

	double resX;
	double resY;

	size_t idLM;

	for(size_t id : md.ids())
	{
		md.getValue(MDL_X, lmX, id);
		md.getValue(MDL_Y, lmY, id);
		md.getValue(MDL_Z, lmZ, id);
		md.getValue(MDL_XCOOR, coord3dx, id);
		md.getValue(MDL_YCOOR, coord3dy, id);
		md.getValue(MDL_ZCOOR, coord3dz, id);
		md.getValue(MDL_SHIFT_X, resX, id);
		md.getValue(MDL_SHIFT_Y, resY, id);
		md.getValue(MDL_FRAME_ID, idLM, id);

		Point3D<double> lm(lmX, lmY, lmZ);
		Point3D<double> coord3d(coord3dx, coord3dy, coord3dz);
		Point2D<double> res(resX, resY);
		double mahalanobisDistance = 0;

		resMod rm {lm, coord3d, res, idLM, mahalanobisDistance};
		vResMod.push_back(rm);
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Input residuals vectors read successfully!" << std::endl;
	#endif
}


void ProgTomoDetectMisalignmentResiduals::writeWeightedResiduals()
{
	auto outputResFilename = fnResidualInfo;
    size_t lastDotPos = outputResFilename.find_last_of('.');

    if (lastDotPos != std::string::npos) {
        outputResFilename.erase(lastDotPos);
    }

	outputResFilename += "_weighted.xmd";


	#ifdef VERBOSE_OUTPUT
	std::cout << "Writting weighted output residuals at " << outputResFilename  << std::endl;
	#endif

	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < vResMod.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_X, vResMod[i].landmarkCoord.x, id);
		md.setValue(MDL_Y, vResMod[i].landmarkCoord.y, id);
		md.setValue(MDL_Z, vResMod[i].landmarkCoord.z, id);
		md.setValue(MDL_XCOOR, (int)vResMod[i].coordinate3d.x, id);
		md.setValue(MDL_YCOOR, (int)vResMod[i].coordinate3d.y, id);
		md.setValue(MDL_ZCOOR, (int)vResMod[i].coordinate3d.z, id);
		md.setValue(MDL_SHIFT_X, vResMod[i].residuals.x, id);
		md.setValue(MDL_SHIFT_Y, vResMod[i].residuals.y, id);
		md.setValue(MDL_FRAME_ID, vResMod[i].id, id);
		md.setValue(MDL_COST, vResMod[i].mahalanobisDistance, id);
	}

	md.write(outputResFilename);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Weighted residuals metadata saved successfully" << std::endl;
	#endif
}


void ProgTomoDetectMisalignmentResiduals::writeOutputAlignmentReport()
{
	MetaDataVec md;
	size_t id;


	if(!globalAlignment)
	{
		for(size_t i = 0; i < nSize; i++)
		{
			id = md.addObject();

			// Tilt-image			
			md.setValue(MDL_IDX, i, id);

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
			id = md.addObject();

			// Tilt-image			
			md.setValue(MDL_IDX, i, id);

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



// --------------------------- MAIN ----------------------------------
void ProgTomoDetectMisalignmentResiduals::run()
{
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;	

	auto t1 = high_resolution_clock::now();

	std::cout << "Starting..." << std::endl;

	size_t Xdim, Ydim;

	MetaDataVec tiltseriesmd;
    // ImageGeneric tiltSeriesImages;

    // if (fnInputTS.isMetaData())
    // {
    //     tiltseriesmd.read(fnInputTS);
    // }
    // else
    // {
    //     tiltSeriesImages.read(fnInputTS, HEADER);

    //     size_t Zdim, Ndim;
    //     tiltSeriesImages.getDimensions(Xdim, Ydim, Zdim, Ndim);

    //     if (fnInputTS.getExtension() == "mrc" and Ndim == 1)
    //         Ndim = Zdim;

    //     size_t id;
    //     FileName fn;
    //     for (size_t i = 0; i < Ndim; i++) 
    //     {
    //         id = tiltseriesmd.addObject();
    //         fn.compose(i + FIRST_IMAGE, fnInputTS);
    //         tiltseriesmd.setValue(MDL_IMAGE, fn, id);
    //     }
    // }

	// tiltSeriesImages.getDimensions(xSize, ySize, zSize, nSize);

	// #ifdef DEBUG_DIM
	// std::cout << "Input tilt-series dimensions:" << std::endl;
	// std::cout << "x " << xSize << std::endl;
	// std::cout << "y " << ySize << std::endl;
	// std::cout << "z " << zSize << std::endl;
	// std::cout << "n " << nSize << std::endl;
	// #endif

	generateSideInfo();

	contructResidualMatrix();

	// detectMisalignmentFromResiduals();
	// detectMisalignmentFromResidualsMahalanobis();
	detectMisalignmentFromResidualsMahalanobisRobust();
	
	#ifdef GENERATE_RESIDUAL_STATISTICS
	generateResidualStatiscticsFile();
	#endif

	writeOutputAlignmentReport();
	writeWeightedResiduals();

	auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1); 	// Getting number of milliseconds as an integer
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}



// --------------------------- UTILS functions ----------------------------

void ProgTomoDetectMisalignmentResiduals::getResModByFiducial(size_t fiducialNumber, std::vector<resMod> &vResMod_fiducial)
{
	for (size_t i = 0; i < vResMod.size(); i++)
	{
		if (vResMod[i].id == fiducialNumber)
		{
			vResMod_fiducial.push_back(vResMod[i]);
		}
	}
}


void ProgTomoDetectMisalignmentResiduals::getResModByImage(size_t tiltImageNumber, std::vector<resMod> &vResMod_image)
{
	for (size_t i = 0; i < vResMod.size(); i++)
	{
		if ((size_t)(vResMod[i].landmarkCoord.z) == tiltImageNumber)
		{
			vResMod_image.push_back(vResMod[i]);
		}
	}
}
