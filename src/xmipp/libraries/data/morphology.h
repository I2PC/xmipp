/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *              Pedro A. de Alarc�n     (pedro@cnb.csic.es)
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
#ifndef _CORE_XMIPP_MORPHOLOGY_HH
#define _CORE_XMIPP_MORPHOLOGY_HH

#include <core/multidim_array.h>

/// @defgroup MathematicalMorphology Mathematical morphology
/// @ingroup DataLibrary
//@{
/**@name Morphology 2D processes
    The neighbourhood must be 4 or 8.

    Count is the number of pixels
    meeting the condition so that the operation is performed, by default,
    0. Ie, if more than 0 pixels meet the condition then the corresponding
    operation is applied.

    Example of use:
    @code
    MultidimArray<double> maskDilated;
    maskDilated.initZeros(mask);
    dilate2D(mask,maskDilated,8,0,patchSize);    
    @endcode

    Size is the size of the structuring element (box).

    The output image must be already resized to the desired shape*/
//@{
/** Dilate.
    See the group documentation for the parameter meanings */
void dilate2D(const MultidimArray<double> &in, MultidimArray<double> &out, int neig,
              int count, int size);
/** Erode.
    See the group documentation for the parameter meanings */
void erode2D(const MultidimArray<double> &in, MultidimArray<double> &out, int neig,
             int count, int size);
/** Closing=Dilation+Erosion */
void closing2D(const MultidimArray<double> &in, MultidimArray<double> &out, int neig,
               int count, int size);
/** Opening=Erosion+Dilation */
void opening2D(const MultidimArray<double> &in, MultidimArray<double> &out, int neig,
               int count, int size);
/** Border of a binary image.
    The border is computed as the subtraction of an image and its dilation. */
void border(const MultidimArray<double> &img, MultidimArray<double> &border);
/** Simplify border.
    The border is simplified by removing all points having more than 2
    neighbours. */
void simplify_border(const MultidimArray<double> &border,
                     MultidimArray<double> &simplified_border);

/** Random Convex hull.
    This routine takes a random number (N) of triangles within the image
    and fill these triangles. The effect is like that of creating the convex
    hull of a binary image. */
void random_convex_hull(const MultidimArray<double> &img, MultidimArray<double> &hull,
                        long N = 100);
//@}

/**@name Morphology 3D processes
    The neighbourhood must be 6, 18 or 26.

    Count is the number of voxels
    meeting the condition so that the operation is performed, by default,
    0. Ie, if more than 0 voxels meet the condition then the corresponding
    operation is applied.

    Size is the size of the structuring element (box).

    The output image must be already resized to the desired shape*/
//@{
/** Binary Dilate.
    See the group documentation for the parameter meanings */
void dilate3D(const MultidimArray<double> &in, MultidimArray<double> &out, int neig,
              int count, int size);
/** Binary Erode.
    See the group documentation for the parameter meanings */
void erode3D(const MultidimArray<double> &in, MultidimArray<double> &out, int neig,
             int count, int size);
/** Binary Closing=Dilation+Erosion */
void closing3D(const MultidimArray<double> &in, MultidimArray<double> &out, int neig,
               int count, int size);
/** Binary Opening=Erosion+Dilation */
void opening3D(const MultidimArray<double> &in, MultidimArray<double> &out, int neig,
               int count, int size);

/** Gray dilation.
    The structuring element must be centered at 0. */
void dilate3D(const MultidimArray<double> &in,
              const MultidimArray<double> &structuringElement,
              MultidimArray<double> &out);

/** Gray erosion.
    The structuring element must be centered at 0. */
void erode3D(const MultidimArray<double> &in,
              const MultidimArray<double> &structuringElement,
              MultidimArray<double> &out);

/** Sharpening.
    Width (radius in pixels), strength (as a percentange of the input range).
    
    Implemented according to JGM Schavemaker, MJT Reinders, JJ Gerbrands,
    E Backer. Image sharpening by morphological filtering. Pattern Recognition
    33: 997-1012 (2000). */
void sharpening(const MultidimArray<double> &in, double width, double strength,
              MultidimArray<double> &out);
//@}
//@}
#endif
