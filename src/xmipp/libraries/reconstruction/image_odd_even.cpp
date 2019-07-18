/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano            coss@cnb.csic.es (2019)
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

#include "image_odd_even.h"
//#define DEBUG
//#define DEBUG_MASK

void ProgOddEven::readParams()
{
	fnImg = getParam("--img");
	splitType = getParam("--type");
	sumFrames = checkParam("--sum_frames");
	fnOut_odd = getParam("-o");
	fnOut_even = getParam("-e");
}


void ProgOddEven::defineParams()
{
	addUsageLine("This function splits a set of images or frames in two subsets named odd and even");
	addParamsLine("  --img <img_file=\"\">   : File of input images (movie or images tilt series)");
	addParamsLine("  --type <split_type>  : Type of splitting");
	addParamsLine("                          :+ frames  -  If the frames will be split)");
	addParamsLine("                          :+ images  -  If the images will be split)");
	addParamsLine("[  --sum_frames]           : Sum the set of split frames");
	addParamsLine("  -o <img_file=\"\">      : File of odd images/frames)");
	addParamsLine("  -e <img_file=\"\">      : File of even images/frames)");
}


void ProgOddEven::run()
{
	std::cout << "Starting..." << std::endl;



	if ((splitType != "frames") or (splitType != "frames"))
	{
		std::cout << "ERROR: Please specify the type of splitting in frames or images" << std::endl;
		std::cout << "       --type frames for splitting the set of frames or --type images for splitting"
				"the set of images" << std::endl;
		exit(0);
	}

	MetaData movie;
	movie.read(fnImg);

	if (splitType == "frames")
	{
		if (fnImg.isMetaData())
		{
			movie.read(fnImg);
		}
		else
		{
			ImageGeneric movieStack;
			movieStack.read(fnImg, HEADER);
			size_t Xdim, Ydim, Zdim, Ndim;
			movieStack.getDimensions(Xdim, Ydim, Zdim, Ndim);
			if (fnImg.getExtension() == "mrc" and Ndim == 1)
				Ndim = Zdim;
			size_t id;
			FileName fn;
			for (size_t i = 0; i < Ndim; i++) {
				id = movie.addObject();
				fn.compose(i + FIRST_IMAGE, fnImg);
				movie.setValue(MDL_IMAGE, fn, id);
			}
		}
	}
//
//	if (splitType == "images")
//	{
//		std::cout << "TODO" << std::endl;
//	}

	long  n = 1;
	MetaData movieOdd, movieEven;

	FileName fnFrame;
	size_t objId, objId_odd, objId_even;
	Image<double> frame, imgOdd, imgEven;


	int counter=0;

	FOR_ALL_OBJECTS_IN_METADATA(movie)
	{
		objId = __iter.objId;
		movie.getValue(MDL_IMAGE, fnFrame, objId);
		if (sumFrames)
			frame.read(fnFrame);


		if (objId%2 == 0)
		{
			objId_even = movieEven.addObject();
			movieEven.setValue(MDL_IMAGE, fnFrame, objId_even);
			if (sumFrames)
				if (counter > 1)
					imgEven() += frame();
				else
					imgEven() = frame();
			++counter;
		}
		else
		{
			objId_odd = movieOdd.addObject();
			movieOdd.setValue(MDL_IMAGE, fnFrame, objId_odd);
			if (sumFrames)
				if (counter > 1)
					imgOdd() += frame();
				else
					imgOdd() = frame();
			++counter;
		}
	}

	movieOdd.write(fnOut_odd);
	movieEven.write(fnOut_even);

	if (sumFrames)
	{
		imgOdd.write(fnOut_odd.withoutExtension()+"_aligned.mrc");
		imgEven.write(fnOut_even.withoutExtension()+"_aligned.mrc");
	}
}

