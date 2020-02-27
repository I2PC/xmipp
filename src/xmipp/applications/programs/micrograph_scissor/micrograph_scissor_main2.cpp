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

#include <core/argsparser.h>
#include <core/xmipp_program.h>
#include <data/micrograph.h>
#include <core/args.h>
#include <core/metadata_extension.h>

class ProgMicrographScissor: public XmippProgram
{
private:

protected:
    void defineParams()
    {
        addUsageLine ("Extract particles from a micrograph");
        addSeeAlsoLine("micrograph_mark");

        addParamsLine(" == General Options == ");
        addParamsLine("  -i <input_untilted_micrograph>     : From which the untilted images will be cutted");
        addParamsLine("     alias --untilted;");
        addParamsLine(" -iAlign <iAlign>                     : Input metadata with the alignment params for the movie to process.");
        addParamsLine("  [--orig <original_micrograph>]      : unless this parameter is specified");
        addParamsLine("  -o <output_stack>                   : Name for the particle images");
        addParamsLine("                                      :+ Two files will be created: output_stack with the particles in a Spider stack an output_stack.xmd with the list of image names, the micrograph they were taken from, and their coordinates");
        addParamsLine("     alias --untiltfn;");
        addParamsLine("  [--pos <position_file>]             : file with particle coordinates");
        addParamsLine("     alias --untiltPos;");
        addParamsLine("  [--extractNoise <n=-1>]             : extract noise particles instead of true particles");
        addParamsLine("                                      : n is the number of noise particles to extract");
        addParamsLine("                                      : If n=-1, then the number of noise particles is the number of coordinates in the pos file");
        addParamsLine("                                      : The input posfile WILL be rewritten with the noise coordinates");

        addParamsLine(" == Processing Options == ");
        addParamsLine("  --Xdim <window_X_dim>               : In pixels");
        addParamsLine("  [--downsampling <float=1.>]         : The positions were determined with this downsampling rate");
        addParamsLine("  [--Ydim <window_Y_dim>]             : If not given Ydim=Xdim");
        addParamsLine("  [--invert]                          : Invert contrast");
        addParamsLine("  [--log]                             : Take logarithm (compute transmitance)");
        addParamsLine("  [--appendToStack]                   : The output stack is deleted.");
        addParamsLine("                                      : Use this option to add the new images to the stack");
        addParamsLine("  [--fillBorders]                     : If the box is outside the micrograph, fill the missing pixels");
        addParamsLine("                                      : instead of setting the whole image to blank");

        addParamsLine(" == Options for tilt pairs == ");
        addParamsLine("  [-t <input_tilted_micrograph>]      : From which the   tilted images will be cutted");
        addParamsLine("     alias --tilted;");
        addParamsLine("  [--tiltfn <output_stack>]           : Name for tilted images (Stack FileName)");
        addParamsLine("     requires --untiltfn;                                                         ");
        addParamsLine("  [--tiltAngles <angles_file>]        : Name of the estimated tilt angles");
        addParamsLine("                                      :  Angle from the Y axis to the tilt axis. Both (tilted and untilted) must be cut with its corresponding angle. Cut images are rotated so that the tilt axis is parallel to Y axis.");
        addParamsLine("     requires --tiltfn;                                                         ");
        addParamsLine("  [--tiltPos <position_file>]         : file with particle coordinates");
        addParamsLine("     requires --tiltfn;                                                         ");
        addParamsLine("  [--ctfparam <ctfparam>]             : metadata with ctf parameters for the micrograph.");

        addExampleLine ("   xmipp_micrograph_scissor -i g7107.raw --pos g7107.raw.Common.pos --oroot images --Xdim 64");
        addExampleLine ("   xmipp_micrograph_scissor --untilted Preprocessing/untilt/down1_untilt.raw --tilted Preprocessing/tilt/down1_tilt.raw --untiltfn untilt --tiltfn tilt --Xdim 60 --tiltAngles ParticlePicking/down1_untilt.raw.angles.txt --pos ParticlePicking/down1_untilt.raw.Common.pos --tiltPos ParticlePicking/down1_untilt.raw.tilted.Common.pos");
    }
    FileName fn_micrograph,fn_out;
    FileName fn_tilted, fn_out_tilted, fn_angles, fn_tilt_pos;
    FileName fn_orig, fn_pos;
    FileName fn_iAlign;
    bool     pair_mode;
    int      Ydim, Xdim;
    bool     reverse_endian;
    bool     compute_transmitance;
    bool     compute_inverse ;
    double   down_transform;
    bool     rmStack;
    bool     fillBorders;
    int      Nnoise;
    bool     extractNoise;

    void readParams()
    {
        fn_micrograph = getParam  ("-i");
        fn_iAlign = getParam("-iAlign");
        pair_mode     = checkParam("--tilted");
        fn_out        = getParam  ("-o");
        fn_pos        = getParam  ("--pos");
        Xdim          = getIntParam("--Xdim");
        if (checkParam("--Ydim"))
            Ydim      = getIntParam("--Ydim");
        else
            Ydim = Xdim;

        compute_inverse      = checkParam("--invert");
        compute_transmitance = checkParam("--log");
        rmStack              = !checkParam("--appendToStack");

        if (!pair_mode)
        {
            if (checkParam("--orig"))
                fn_orig       = getParam("--orig");
        }
        else
        {
            fn_out_tilted = getParam("--tiltfn");
            fn_tilted      = getParam("--tilted");
            fn_angles      = getParam("--tiltAngles");
            fn_tilt_pos    = getParam("--tiltPos");
        }
        down_transform = getDoubleParam("--downsampling");
        fillBorders = checkParam("--fillBorders");
        extractNoise = checkParam("--extractNoise");
        if (extractNoise)
        {
        	randomize_random_generator();
        	Nnoise = getIntParam("--extractNoise",0);
        }
    }
public:

    double bspline03(double x) {
        double Argument = fabs(x);
        if (Argument < 1.0)
            return Argument * Argument * (Argument - 2.0) * 0.5 + 2.0 / 3.0;
        else if (Argument < 2.0) {
            Argument -= 2.0;
            return Argument * Argument * Argument * (-1.0 / 6.0);
        } else
            return 0.0;
    }


    void getShift(int lX, int lY, int lN, int xdim, int ydim, int ndim, int x,
            int y, int curFrame, double &shiftY, double &shiftX, const double* coefsX,
            const double* coefsY) {
        double delta = 0.0001;
        // take into account end poits
        double hX = (lX == 3) ? xdim : (xdim / (double) ((lX - 3)));
        double hY = (lY == 3) ? ydim : (ydim / (double) ((lY - 3)));
        double hT = (lN == 3) ? ndim : (ndim / (double) ((lN - 3)));
        // index of the 'cell' where pixel is located (<0, N-3> for N control points)
        double xPos = x / hX;
        double yPos = y / hY;
        double tPos = curFrame / hT;
        // indices of the control points are from -1 .. N-2 for N points
        // pixel in 'cell' 0 may be influenced by points with indices <-1,2>
        for (int idxT = std::max(-1, (int) (tPos) - 1);
                idxT <= std::min((int) (tPos) + 2, lN - 2); ++idxT) {
            double tmpT = bspline03(tPos - idxT);
            for (int idxY = std::max(-1, (int) (yPos) - 1);
                    idxY <= std::min((int) (yPos) + 2, lY - 2); ++idxY) {
                double tmpY = bspline03(yPos - idxY);
                for (int idxX = std::max(-1, (int) (xPos) - 1);
                        idxX <= std::min((int) (xPos) + 2, lX - 2); ++idxX) {
                    double tmpX = bspline03(xPos - idxX);
                    double tmp = tmpX * tmpY * tmpT;
                    if (fabsf(tmp) > delta) {
                        size_t coeffOffset = (idxT + 1) * (lX * lY)
                                + (idxY + 1) * lX + (idxX + 1);
                        shiftX += coefsX[coeffOffset] * tmp;
                        shiftY += coefsY[coeffOffset] * tmp;
                    }
                }
            }
        }
    }

    void run()
    {

        MetaData movieAlignedMd;
    	MDRow rowMd;
    	std::vector<double> coeffsX, coeffsY;
    	std::vector<size_t> ctrlPts;
		if (fn_iAlign.hasMetadataExtension()){
			std::cout<<fn_iAlign<<std::endl;
			movieAlignedMd.read((FileName)("localAlignment@")+fn_iAlign, NULL);
			MDIterator *iterSF = new MDIterator(movieAlignedMd);
			movieAlignedMd.getRow(rowMd, iterSF->objId);
			if(rowMd.containsLabel(MDL_LOCAL_ALIGNMENT_COEFFS_X))
				rowMd.getValue(MDL_LOCAL_ALIGNMENT_COEFFS_X, coeffsX);
			if(rowMd.containsLabel(MDL_LOCAL_ALIGNMENT_COEFFS_Y))
				rowMd.getValue(MDL_LOCAL_ALIGNMENT_COEFFS_Y, coeffsY);
			if(rowMd.containsLabel(MDL_LOCAL_ALIGNMENT_CONTROL_POINTS))
				rowMd.getValue(MDL_LOCAL_ALIGNMENT_CONTROL_POINTS, ctrlPts);
		}
        std::cout<<coeffsX.size()<<" "<<coeffsY.size()<<" "<<ctrlPts.size()<<std::endl;


        if (!pair_mode)
        {
            Micrograph m;
            m.open_micrograph(fn_micrograph);
            m.set_window_size(Xdim, Ydim);
            m.read_coordinates(0, fn_pos);
            if (down_transform != 1.)
                m.scale_coordinates(1./down_transform);
            m.add_label("");
            m.set_transmitance_flag(compute_transmitance);
            m.set_inverse_flag(compute_inverse);

            if (checkParam("--ctfparam"))
            {
              MetaData ctfparam(getParam("--ctfparam"));
              MDRow ctfRow;
              ctfparam.getRow(ctfRow, ctfparam.firstObject());
              m.set_ctfparams(ctfRow);
            }

            //AJ modifying the coordinates to come back to the original position taking into account the movie alignment params
            int imax = m.coords.size();
            for (int i = 0; i < imax; i++)
            {
                if (m.coords[i].valid)
                {
                    m.coords[i].X = (int) (coords[i].X * c);
                    m.coords[i].Y = (int) (coords[i].Y * c);
                }
            }

            m.produce_all_images(0, -1, fn_out, fn_orig, 0.,0.,0., rmStack, fillBorders, extractNoise, Nnoise);
            if (extractNoise)
            {
            	MDRow row=firstRow(fn_pos);
            	size_t micId=0;
            	if (row.containsLabel(MDL_MICROGRAPH_ID))
            		row.getValue(MDL_MICROGRAPH_ID,micId);

            	// Rewrite the input posfile with the true noise coordinates
            	MetaData MD;
            	for (size_t i=0; i<m.coords.size(); i++)
                {
            		size_t id=MD.addObject();
                    MD.setValue(MDL_XCOOR, m.coords[i].X, id);
                    MD.setValue(MDL_YCOOR, m.coords[i].Y, id);
                    MD.setValue(MDL_MICROGRAPH_ID, micId, id);
                }
            	MD.write(fn_pos);
            }
        }
        else
        {
            MetaData auxMd;
            // Read angles
            double alpha_u, alpha_t, tilt_angle;
            auxMd.read(fn_angles);
            size_t objId = auxMd.firstObject();
            auxMd.getValue(MDL_ANGLE_Y, alpha_u, objId);
            auxMd.getValue(MDL_ANGLE_Y2, alpha_t, objId);
            auxMd.getValue(MDL_ANGLE_TILT, tilt_angle, objId);

            // Generate the images for the untilted image
            Micrograph m;
            m.open_micrograph(fn_micrograph);
            m.set_window_size(Xdim, Ydim);
            m.read_coordinates(0, fn_pos);
            m.add_label("");
            m.set_transmitance_flag(compute_transmitance);
            m.set_inverse_flag(compute_inverse);
            m.produce_all_images(0, -1, fn_out, "", alpha_u,0.,0.,rmStack);
            m.close_micrograph();

            // Generate the images for the tilted image
            Micrograph mt;
            mt.open_micrograph(fn_tilted);
            mt.set_window_size(Xdim, Ydim);
            mt.read_coordinates(0, fn_tilt_pos);
            mt.add_label("");
            mt.set_transmitance_flag(compute_transmitance);
            mt.set_inverse_flag(compute_inverse);
            mt.produce_all_images(0, -1, fn_out_tilted, "", 0., tilt_angle, alpha_t, rmStack);
            mt.close_micrograph();
        }
    }
};

RUN_XMIPP_PROGRAM(ProgMicrographScissor)
