/***************************************************************************
 * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
 *
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

#include "tomo_coordinate_tracking.h"
#include "delaunay/delaunay.h"
#include "core/geometry.h"
#include "core/matrix1d.h"
#include "core/matrix2d.h"
#include "core/linear_system_helper.h"
#include "hungarian.h"  // Hungarian algorithm library


void ProgCoordTracking::readParams()
{
	fnuntilt = getParam("--untiltcoor");
	fntilt = getParam("--tiltcoor");
	tiltAng = getDoubleParam("--tiltAngle");
	fndir = getParam("--odir");
}

void ProgCoordTracking::defineParams()
{
	//usage
	addUsageLine("Validate a 3D reconstruction from its projections attending to directionality and spread of the angular assignments from a given significant value");
	//params

	addParamsLine("  [--untiltcoor <md_file=\"\">]    : Untilt coordinates");
	addParamsLine("  [--tiltcoor <md_file=\"\">]    : Tilt coordinates");
	addParamsLine("  [--tiltAngle <s=0>]    : Tilt micrography");
	addParamsLine("  [--odir <outputDir=\".\">]   : Output directory");
}


void ProgCoordTracking::run()
{
	std::cout << "Starting..." << std::endl;

	//LOAD METADATA and TRIANGULATIONS
	MetaDataVec md_untilt, md_tilt, md_matchingUntilted, md_matchingTilted;

	md_untilt.read(fnuntilt);
	md_tilt.read(fntilt);

	int x, y, coor_u=0, coor_t=0;
	size_t numerOfUntilted, numerOfTilted;

	numerOfUntilted = md_untilt.size();
	numerOfTilted = md_tilt.size();

	//storing untilted points and creating Delaunay triangulation
	struct Delaunay_T delaunay_untilt;

	// Coordinates of the vertex of the Delaunay triangulation are (ux, uy) for the untilted and (tx, ty) for the tilted
	Matrix1D<double> ux(numerOfUntilted), uy(numerOfUntilted), tx(numerOfTilted), ty(numerOfTilted);
	init_Delaunay( &delaunay_untilt, numerOfUntilted);

	for (size_t objId : md_untilt.ids())
	{
		md_untilt.getValue(MDL_XCOOR, x, objId);
		md_untilt.getValue(MDL_YCOOR, y, objId);

		VEC_ELEM(ux, coor_u) = x;
		VEC_ELEM(uy, coor_u) = y;

		insert_Point( &delaunay_untilt, x, y);
		coor_u += 1;
	}

	//storing tilted points and creating Delaunay triangulation
	struct Delaunay_T delaunay_tilt;
	init_Delaunay( &delaunay_tilt, numerOfTilted);
	for (size_t objId : md_tilt.ids())
	{
		md_tilt.getValue(MDL_XCOOR, x, objId);
		md_tilt.getValue(MDL_YCOOR, y, objId);

		VEC_ELEM(tx,coor_t) = x;
		VEC_ELEM(ty,coor_t) = y;

		insert_Point( &delaunay_tilt, x, y);
		coor_t += 1;
	}

	// We create the Voronoi Tesselation (untilted an tilted)
	create_Delaunay_Triangulation( &delaunay_untilt, 1);	
	create_Delaunay_Triangulation( &delaunay_tilt, 1);

	struct Point_T u, pred_u, pred_t, t_closest, u_closest;
	double dist;

	// Matrix2D<double> TM;
	// TM.initZeros(2,2);

	double c = cos(tiltAng*PI/180);
	// MAT_ELEM(TM, 0, 0) = cos(tiltAng*PI/180);
	// MAT_ELEM(TM, 1, 1) = 1;
	// MAT_ELEM(TM, 0, 1) = 0;
	// MAT_ELEM(TM, 1, 0) = 0;

	// For each coordinate untilted we predict the tilted one
	for (size_t i = 0; i<numerOfUntilted; i++)
	{
		u.x = VEC_ELEM(ux, i);
		u.y = VEC_ELEM(uy, i);

		//Predicted -> pred_t = TM*u
		pred_t.x = c*u.x;
		pred_t.y = u.y;

		if (!select_Closest_Point(&delaunay_tilt, &pred_t, &t_closest, &dist))
		{ 
			write_DCEL(delaunay_tilt.dcel, 0, "tiltdata_dcel.txt");
			//std::cerr << "WARNING IN TRIANGULATION OR CLOSEST NEIGHBOUR" << std::endl;
			//printf("Maybe the warning involves the tilt coordinates ( %f , %f ) \n", t_dist.x, t_dist.y);
			continue;
		}

		//Predicted -> pred_u = inv(TM)*t_closest
		pred_u.x = t_closest.x/c;
		pred_u.y = t_closest.y;

		if (!select_Closest_Point(&delaunay_untilt, &pred_u, &u_closest, &dist))
		{
			write_DCEL(delaunay_untilt.dcel, 0, "untiltdata_dcel.txt");
			//std::cerr << "WARNING IN TRIANGULATION OR CLOSEST NEIGHBOUR" << std::endl;
			//printf("Maybe the warning involves the tilt coordinates ( %f , %f ) \n", t_dist.x, t_dist.y);
			continue;
		}

		if ((u.x == u_closest.x))	// && (u.y == u_closest.y))
		// if (true)
		{
			MDRowVec rowUntilted, rowTilted;
			rowUntilted.setValue(MDL_XCOOR, (int) VEC_ELEM(ux, i));
			rowUntilted.setValue(MDL_YCOOR, (int) VEC_ELEM(uy, i));
			rowTilted.setValue(MDL_XCOOR, (int) t_closest.x);
			rowTilted.setValue(MDL_YCOOR, (int) t_closest.y);
			
			md_matchingUntilted.addRow(rowUntilted);
			md_matchingTilted.addRow(rowTilted);
		}
	}

	delete_Delaunay(&delaunay_untilt);
	delete_Delaunay(&delaunay_tilt);

	md_matchingUntilted.write(fndir+'/'+fnuntilt.getBaseName() + ".xmd" );
	md_matchingTilted.write(fndir+'/'+fntilt.getBaseName() + ".xmd" );

}

