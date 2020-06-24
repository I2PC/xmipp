/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela                 eramirez@cnb.csic.es
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

#include "volume_3dfsc_decomposition.h"
#include "resolution_directional.h"

void Prog3dDecomp::readParams()
{
        fnVol = getParam("--vol");
        fnHalf1 = getParam("--half1");
        fnHalf2 = getParam("--half2");
        fnMask = getParam("--mask");
        Nthread = getIntParam("-n");
        sampling = getDoubleParam("--sampling");
        icosahedron = checkParam("--ico");
        wfsc = checkParam("--fsc");
        mask_exist = checkParam("--mask");
        local = checkParam("--local");
}

void Prog3dDecomp::defineParams()
{
        addUsageLine("This function performs local sharpening");
        addParamsLine("  --vol <vol_file=\"\">                  : Input volume");
        addParamsLine("  --half1 <half1_file=\"\">              : Input half map 1");
        addParamsLine("  --half2 <half2_file=\"\">              : Input half map 2");
        addParamsLine("  [--mask <vol_file=\"\">]               : Binary mask");
        addParamsLine("  --sampling <s=1>                       : sampling");
        addParamsLine("  [--ico]:                               : Use icosahedron as coverage of the projection sphere");
        addParamsLine("  [--fsc]:                               : Wheigt by directional fsc");
        addParamsLine("  [--local]:                             : Take into account the locality");
        addParamsLine("  [-n <Nthread=1>]                       : Number of threads");
}


void Prog3dDecomp::produceSideInfo()
{
        std::cout << "Starting..." << std::endl;
        Monogenic mono;
        MultidimArray<double> inputVol;

        std::cout << "Reading data..." << std::endl;
        Image<double> V;
        V.read(fnVol);
        V().setXmippOrigin();
        inputVol = V();


        FourierTransformer transformer;

        //TODO: check if its possible to use only one transformer instead of transformer_inv and transformer
        transformer_inv.setThreadsNumber(Nthread);

        transformer.FourierTransform(inputVol, fftV);

        // Frequency volume
        iu = mono.fourierFreqs_3D(fftV, inputVol, freq_fourier_x, freq_fourier_y, freq_fourier_z);
}


void Prog3dDecomp::simpleGeometryFaces(Matrix2D<double> &faces, Matrix2D<double> &limts)
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
    MAT_ELEM(faces, 4,0) = 0;           MAT_ELEM(faces, 4,1) = 0;           MAT_ELEM(faces, 4,2) = 1;

    limts.initZeros(5,4);
    MAT_ELEM(limts, 0,0) = 0.0;   MAT_ELEM(limts, 0,1) = 90.0;    MAT_ELEM(limts, 0,2) = 90-36.86;  MAT_ELEM(limts, 0,3) = 90.0;
    MAT_ELEM(limts, 1,0) = -90;   MAT_ELEM(limts, 1,1) = -180.0;  MAT_ELEM(limts, 1,2) = 90-36.86;  MAT_ELEM(limts, 1,3) = 90.0;
    MAT_ELEM(limts, 2,0) = 0.0;   MAT_ELEM(limts, 2,1) = 90.0;    MAT_ELEM(limts, 2,2) = -90.0;     MAT_ELEM(limts, 2,3) = 90+36.86;
    MAT_ELEM(limts, 3,0) = -90;   MAT_ELEM(limts, 3,1) = -180.0;  MAT_ELEM(limts, 3,2) = -90.0;     MAT_ELEM(limts, 3,3) = 90+36.86;
    MAT_ELEM(limts, 4,0) = -90.0; MAT_ELEM(limts, 4,1) = 90.0;    MAT_ELEM(limts, 4,2) = 0.0;       MAT_ELEM(limts, 4,3) = 90-36.86;

}


void Prog3dDecomp::icosahedronVertex(Matrix2D<double> &vertex)
{
    std::cout << "Defining Icosahedron vertex..." << std::endl;

    //The icosahedron vertex are located in (0, +-1, +-phi), (+-1, +-phi, 0), (+-phi, 0, +-1) with phi = (1+sqrt(5))/2
    double phi =  (1+sqrt(5))/2;

    vertex.initZeros(12,3);

    MAT_ELEM(vertex, 0,0) = 0;          MAT_ELEM(vertex, 0,1) = 1;          MAT_ELEM(vertex, 0,2) = phi;
    MAT_ELEM(vertex, 1,0) = 0;          MAT_ELEM(vertex, 1,1) = 1;          MAT_ELEM(vertex, 1,2) = -phi;
    MAT_ELEM(vertex, 2,0) = 0;          MAT_ELEM(vertex, 2,1) = -1;         MAT_ELEM(vertex, 2,2) = phi;
    MAT_ELEM(vertex, 3,0) = 0;          MAT_ELEM(vertex, 3,1) = -1;         MAT_ELEM(vertex, 3,2) = -phi;
    MAT_ELEM(vertex, 4,0) = 1;          MAT_ELEM(vertex, 4,1) = phi;            MAT_ELEM(vertex, 4,2) = 0;
    MAT_ELEM(vertex, 5,0) = 1;          MAT_ELEM(vertex, 5,1) = -phi;           MAT_ELEM(vertex, 5,2) = 0;
    MAT_ELEM(vertex, 6,0) = -1;         MAT_ELEM(vertex, 6,1) = phi;            MAT_ELEM(vertex, 6,2) = 0;
    MAT_ELEM(vertex, 7,0) = -1;         MAT_ELEM(vertex, 7,1) = -phi;           MAT_ELEM(vertex, 7,2) = 0;
    MAT_ELEM(vertex, 8,0) = phi;            MAT_ELEM(vertex, 8,1) = 0;          MAT_ELEM(vertex, 8,2) = 1;
    MAT_ELEM(vertex, 9,0) = phi;            MAT_ELEM(vertex, 9,1) = 0;          MAT_ELEM(vertex, 9,2) = -1;
    MAT_ELEM(vertex, 10,0) = -phi;          MAT_ELEM(vertex, 10,1) = 0;         MAT_ELEM(vertex, 10,2) = 1;
    MAT_ELEM(vertex, 11,0) = -phi;          MAT_ELEM(vertex, 11,1) = 0;         MAT_ELEM(vertex, 11,2) = -1;

    vertex = vertex*(1/sqrt(1+phi*phi));
}

void Prog3dDecomp::icosahedronFaces(Matrix2D<int> &faces, Matrix2D<double> &vertex)
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

void Prog3dDecomp::getFaceVectorIcosahedron(Matrix2D<int> &faces,
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

void Prog3dDecomp::getFaceVectorSimple(Matrix2D<double> &facesVector, Matrix2D<double> &faces)
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

void Prog3dDecomp::defineIcosahedronCone(int face_number, double &x1, double &y1, double &z1,
        MultidimArray< std::complex<double> > &myfftV, MultidimArray<double> &conefilter, double coneAngle)
{

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
//              DIRECT_MULTIDIM_ELEM(conefilter, n) = 1;
                if (cosine>=cosconeAngle)
                {
                    if (DIRECT_MULTIDIM_ELEM(iu,n) <1.99)
                    {
                        n++;
                        continue;
                    }
                    DIRECT_MULTIDIM_ELEM(conefilter, n) = 1;
//                  DIRECT_MULTIDIM_ELEM(conetest, n) = 0;
                }
                ++n;
            }
        }
    }

//    Image<double> icosahedronMasked;
//    icosahedronMasked = conefilter;
//    FileName fnmasked;
//    fnmasked = formatString("maskConeFourier_%i.mrc",face_number);
//    icosahedronMasked.write(fnmasked);
}

void Prog3dDecomp::defineSimpleCaps(MultidimArray<int> &coneMask, Matrix2D<double> &limits,
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
//    FileName fnmasked;
//    fnmasked = "maskCone.mrc";
//    int m1sizeX = 240;
//    int m1sizeY = 240;
//    int m1sizeZ = 240;

//    MultidimArray<double> fullMap;
//  createFullFourier(coneMask, fnmasked, m1sizeX, m1sizeY, m1sizeZ, fullMap);

//    Image<int> icosahedronMasked;
//    icosahedronMasked = coneMask;
//    icosahedronMasked.write(fnmasked);
}

void cleanFaces2(Matrix2D<int> &faces, Matrix2D<double> &vertex)
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


void Prog3dDecomp::defineComplexCaps(Matrix2D<double> &facesVector,
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

//    FileName fnmasked;
//    fnmasked = "coneMask.mrc";
//    int m1sizeX = 240;
//    int m1sizeY = 240;
//    int m1sizeZ = 240;
//
//
//    Image<int> icosahedronMasked;
//    icosahedronMasked = coneMask;
//    icosahedronMasked.write(fnmasked);
}


void Prog3dDecomp::FilterFunction(size_t &Nfaces, MultidimArray<int> &maskCone, MultidimArray<double> &vol,
        MultidimArray<double> &hmap1, MultidimArray<double> &hmap2, FourierTransformer &transformer_inv)
{

////    ProgVolumeCorrectBfactor bfactor;
//    std::vector<double> snr;
//    std::vector<fit_point2D> guinierweighted;
//    double slope;
//    double intercept = 0.;
//    double BfactorToAplly;
//    xsize = XSIZE(vol);
//    fit_minres = 10;


    MultidimArray<double> Vorig, localV, localH1, localH2;
    MultidimArray< std::complex<double> > fftM1, fftM2;
    Vorig = vol;
    xsize = XSIZE(Vorig);

    //****Final sharpened Volume
    MultidimArray<double> dsharpVolFinal;
    dsharpVolFinal.initZeros(Vorig);

    //******Introduce the locality
    if (local == true)
    {
        transformer.setThreadsNumber(Nthread);

        int boxdim = static_cast<int>(xsize/2);
        std::cerr <<"xsize "<< xsize <<std::endl;
        std::cerr <<"boxdim "<< boxdim <<std::endl;

        localV.initZeros(Vorig);
        localH1.initZeros(hmap1);
        localH2.initZeros(hmap2);


        //***supose cubic map (xsize=ysize=zsize)

        for (int ii=0; ii<xsize; ii+=boxdim)
        {
            for (int jj=0; jj<xsize; jj+=boxdim)
            {
                for (int kk=0; kk<xsize; kk+=boxdim)
                {
                    long n=0;
                    for (int xx=0; xx<xsize; ++xx)
                    {
                        for (int yy=0; yy<xsize; ++yy)
                        {
                            for (int zz=0; zz<xsize; ++zz)
                            {
                                if ( ((xx>=ii) && (xx<ii+boxdim)) && ((yy>=jj) && (yy<jj+boxdim)) && ((zz>=kk) && (zz<kk+boxdim)) )
                                {
                                    DIRECT_MULTIDIM_ELEM(localV,n) = DIRECT_MULTIDIM_ELEM(Vorig,n);
                                    DIRECT_MULTIDIM_ELEM(localH1,n) = DIRECT_MULTIDIM_ELEM(hmap1,n);
                                    DIRECT_MULTIDIM_ELEM(localH2,n) = DIRECT_MULTIDIM_ELEM(hmap2,n);
                                }
                                ++n;

                            }
                        }

                    }
                    std::cerr <<"ii "<< ii <<std::endl;
                    std::cerr <<"jj "<< jj <<std::endl;
                    std::cerr <<"kk "<< kk <<std::endl;
                    transformer.FourierTransform(localV, fftV);
                    transformer.FourierTransform(localH1, fftM1);
                    transformer.FourierTransform(localH2, fftM2);
                    directionalFilter(Nfaces, maskCone, vol, fftV, fftM1, fftM2, transformer_inv, dsharpVolFinal);

//                    FileName fs1;
//                    Image<double> saveImg1;
//                    fs1 = formatString("localMap_%i_%i_%i.vol", ii, jj, kk);
//                    saveImg1() = localV;
//                    saveImg1.write(fs1);
//                    exit(0);

                }

            }
        }

    }
    else
    {

        transformer.setThreadsNumber(Nthread);
        transformer.FourierTransform(vol, fftV);
        transformer.FourierTransform(hmap1, fftM1);
        transformer.FourierTransform(hmap2, fftM2);

    //    MultidimArray<double> dsharpVolFinal;
        dsharpVolFinal.initZeros(Vorig);
        directionalFilter(Nfaces, maskCone, Vorig, fftV, fftM1, fftM2, transformer_inv, dsharpVolFinal);
    }

//    MultidimArray< std::complex<double> > fftVfilter, fftM1filter, fftM2filter;
//    MultidimArray<double> dsharpVolFinal, dsharpVol, filteredMap1, filteredMap2;
//
//    dsharpVolFinal.initZeros(Vorig);

//    for (size_t face_number = 0; face_number<Nfaces; ++face_number)
//    {
//
//        fftVfilter.initZeros(fftV);
//        fftM1filter.initZeros(fftM1);
//        fftM2filter.initZeros(fftM2);
//        dsharpVol.initZeros(Vorig);
//
//        // Filter the input volume and add it to amplitude
//        long n=0;
//        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftV)
//        {
//            if (DIRECT_MULTIDIM_ELEM(maskCone, n) == face_number)
//            {
//
//                DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(fftV, n);
//                DIRECT_MULTIDIM_ELEM(fftM1filter, n) = DIRECT_MULTIDIM_ELEM(fftM1, n);
//                DIRECT_MULTIDIM_ELEM(fftM2filter, n) = DIRECT_MULTIDIM_ELEM(fftM2, n);
//
//            }
//        }
//
//        ///// ***** Determine the FSC between half maps
//        FscCalculation(fftM1filter, fftM2filter, frc);
//        apply_maxres = resol;
//        if (resol < 4)
//        {
//            fit_maxres = 4;
//        }
//        else
//            fit_maxres = resol;
//
//
//        ///// ***** Weight the map and apply b-factor
//
//        if (wfsc == true)
//        {
//            std::cerr<<"calculando fsc"<<std::endl;
//            snrWeights(snr);
//            apply_snrWeights(fftVfilter,snr);
//        }
//        make_guinier_plot(fftVfilter,guinierweighted);
//        least_squares_line_fit(guinierweighted, slope, intercept);
//        BfactorToAplly = 4. * slope;
////        BfactorToAplly = -60;
//        std::cerr<<"Applying B-factor of "<< BfactorToAplly << " squared Angstroms"<<std::endl;
//        apply_bfactor(fftVfilter,BfactorToAplly);
//
//        transformer_inv.inverseFourierTransform(fftVfilter, dsharpVol);
//
//
//        //******** Sum all directions
//
//        if (std::isnan(BfactorToAplly))
//            continue;
//        else
//            dsharpVolFinal += dsharpVol;
//
//        FileName fsk;
//        Image<double> saveImgk;
//        fsk = formatString("sharp_total.vol");
//        saveImgk() = dsharpVolFinal;
//        saveImgk.write(fsk);
//
//    }

}


//************Calculation of the FSC******************

void Prog3dDecomp::FscCalculation(MultidimArray< std::complex<double> > &FThalf1,
                                  MultidimArray< std::complex<double> > &FThalf2,
                                  MultidimArray<double> &frc)
//void Prog3dDecomp::FscCalculation(MultidimArray<double> &FThalf1,
//                                  MultidimArray<double> &FThalf2)

{
    MultidimArray<double> freq, dpr, frc_noise, error_l2;
    double rFactor = -1.;
    double min_samp = 0.;
    double max_samp = 2 * sampling;
    resol = 0;
    frc.clear();

    frc_dpr(FThalf1, FThalf2, sampling, freq, frc, frc_noise, dpr, error_l2,
            false, false, min_samp, sampling/max_samp, &rFactor);

    long i = 0;

    FOR_ALL_ELEMENTS_IN_ARRAY1D(freq)
    {
        if (i>0)
        {
            if (dAi(frc, i) <= 0.143)
            {
                resol = 1/(dAi(freq, i));
                break;
            }
        }
    }
    std::cout <<  " resol " <<  resol << std::endl;

}

//void Prog3dDecomp::VolumesFsc(MultidimArray<double> &FThalf1,
//                              MultidimArray<double> &FThalf2)
void Prog3dDecomp::VolumesFsc(MultidimArray< std::complex<double> > &FT1,
                              MultidimArray< std::complex<double> > &FT2)
{
    FscCalculation(FT1, FT2, frc);
}


//************Calculation of the b-factor******************

void Prog3dDecomp::snrWeights(std::vector<double> &snr)
{
    snr.clear();
    double fsc;

    FOR_ALL_ELEMENTS_IN_ARRAY1D(frc)
    {
        if (i>0)
        {
            fsc=dAi(frc, i);
            double mysnr = XMIPP_MAX( (2*fsc) / (1+fsc), 0.);
            snr.push_back( sqrt(mysnr) );
        }
    }
}

void  Prog3dDecomp::apply_snrWeights(MultidimArray< std::complex< double > > &FT1,
        std::vector<double> &snr)
{

    Matrix1D<double> f(3);
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(FT1)
    {
        FFT_IDX2DIGFREQ(j,xsize,XX(f));
        FFT_IDX2DIGFREQ(i,YSIZE(FT1),YY(f));
        FFT_IDX2DIGFREQ(k,ZSIZE(FT1),ZZ(f));
        double R = f.module();
        if (sampling / R >= apply_maxres)
        {
            int idx=ROUND(R*xsize);
            dAkij(FT1, k, i, j) *= snr[idx];
        }
    }
}

void  Prog3dDecomp::make_guinier_plot(MultidimArray< std::complex< double > > &FT1,
        std::vector<fit_point2D> &guinier)
{
    MultidimArray< int >  radial_count(xsize);
    MultidimArray<double> lnF(xsize);
    Matrix1D<double>      f(3);
    fit_point2D      onepoint;

    lnF.initZeros();
    for (size_t k=0; k<ZSIZE(FT1); k++)
    {
        FFT_IDX2DIGFREQ(k,ZSIZE(FT1),ZZ(f));
        double z2=ZZ(f)*ZZ(f);
        for (size_t i=0; i<YSIZE(FT1); i++)
        {
            FFT_IDX2DIGFREQ(i,YSIZE(FT1),YY(f));
            double y2z2=z2+YY(f)*YY(f);
            for (size_t j=0; j<XSIZE(FT1); j++)
            {
                FFT_IDX2DIGFREQ(j,xsize,XX(f));
                double R2=y2z2+XX(f)*XX(f);
                if (R2>0.25)
                    continue;
                double R=sqrt(R2);
                int idx=ROUND(R*xsize);
                A1D_ELEM(lnF,idx) += abs(dAkij(FT1, k, i, j));
                ++A1D_ELEM(radial_count,idx);
            }
        }
    }

    guinier.clear();
    for (size_t i = 0; i < XSIZE(radial_count); i++)
    {
        double res = (xsize * sampling)/(double)i;
        if (res >= apply_maxres)
        {
            onepoint.x = 1. / (res * res);
            if (lnF(i)>0.)
            {
                onepoint.y = log ( lnF(i) / radial_count(i) );
                if (res <= fit_minres && res >= fit_maxres)
                {
                    onepoint.w = 1.;
                }
                else
                {
                    onepoint.w = 0.;
                }
            }
            else
            {
                onepoint.y = 0.;
                onepoint.w = 0.;
            }
            guinier.push_back(onepoint);
        }
    }
}

void  Prog3dDecomp::apply_bfactor(MultidimArray< std::complex< double > > &FT1,
                                        double bfactor)
{
    Matrix1D<double> f(3);
    double isampling_rate=1.0/sampling;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(FT1)
    {
        FFT_IDX2DIGFREQ(j,xsize,XX(f));
        FFT_IDX2DIGFREQ(i,YSIZE(FT1),YY(f));
        FFT_IDX2DIGFREQ(k,ZSIZE(FT1),ZZ(f));
        double R = f.module() * isampling_rate;
        if (1./R >= apply_maxres)
        {
            dAkij(FT1, k, i, j) *= exp( -0.25* bfactor  * R * R);
        }
    }
}

void  Prog3dDecomp::directionalFilter(size_t &Nfaces, MultidimArray<int> &maskCone, MultidimArray<double> &Vorig,
            MultidimArray< std::complex< double > > &fftV, MultidimArray< std::complex< double > > &fftM1,
            MultidimArray< std::complex< double > > &fftM2, FourierTransformer &transformer_inv, MultidimArray<double> dsharpVolFinal)
{

    //    ProgVolumeCorrectBfactor bfactor;
    std::vector<double> snr;
    std::vector<fit_point2D> guinierweighted;
    double slope;
    double intercept = 0.;
    double BfactorToAplly;
    fit_minres = 10;

    MultidimArray< std::complex<double> > fftVfilter, fftM1filter, fftM2filter;
    MultidimArray<double> dsharpVol;

    for (size_t face_number = 0; face_number<Nfaces; ++face_number)
    {
        fftVfilter.initZeros(fftV);
        fftM1filter.initZeros(fftM1);
        fftM2filter.initZeros(fftM2);
        dsharpVol.initZeros(Vorig);

        // Filter the input volume and add it to amplitude
        long n=0;
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftV)
        {
            if (DIRECT_MULTIDIM_ELEM(maskCone, n) == face_number)
            {

                DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(fftV, n);
                DIRECT_MULTIDIM_ELEM(fftM1filter, n) = DIRECT_MULTIDIM_ELEM(fftM1, n);
                DIRECT_MULTIDIM_ELEM(fftM2filter, n) = DIRECT_MULTIDIM_ELEM(fftM2, n);

            }
        }

        ///// ***** Determine the FSC between half maps
        FscCalculation(fftM1filter, fftM2filter, frc);
        apply_maxres = resol;
        fit_maxres = resol;
//        if (resol < 4)
//        {
//            fit_maxres = 4;
//        }
//        else
//            fit_maxres = resol;

        ///// ***** Weight the map and apply b-factor

        if (wfsc == true)
        {
            std::cerr<<"calculando fsc"<<std::endl;
            snrWeights(snr);
            apply_snrWeights(fftVfilter,snr);
        }
        make_guinier_plot(fftVfilter,guinierweighted);
        least_squares_line_fit(guinierweighted, slope, intercept);
        BfactorToAplly = 4. * slope;
        std::cerr<<"Applying B-factor of "<< BfactorToAplly << " squared Angstroms"<<std::endl;
        apply_bfactor(fftVfilter,BfactorToAplly);

        transformer_inv.inverseFourierTransform(fftVfilter, dsharpVol);


        //******** Sum all directions

        if (std::isnan(BfactorToAplly))
            continue;
        else
            dsharpVolFinal += dsharpVol;

    }
    FileName fsk;
    Image<double> saveImgk;
    fsk = formatString("sharp_total.vol");
    saveImgk() = dsharpVolFinal;
    saveImgk.write(fsk);

}


void Prog3dDecomp::run()
{

    produceSideInfo();

    std::cout << "Reading data..." << std::endl;
    //Defining the number of vertex and faces of the icosahedron
    Matrix2D<double> vertex, facesSimple, limtSimple, faceVector;
    Matrix2D<int> faces;
    double coneAngle = PI/6;
    MultidimArray< std::complex<double> > fftCone;
    MultidimArray<double> conefilter;
    MultidimArray<int> coneMask;
    Monogenic mono;
    size_t Nfaces;

    if (icosahedron == true)
    {
        std::cout << "Using Icosahedron geometry" << std::endl;
        icosahedronVertex(vertex);
        icosahedronFaces(faces, vertex);
        cleanFaces2(faces, vertex);
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

        fftCone = mono.applyMaskFourier(fftV, conefilter);

    }
    t1 = clock();

    double time = (double(t1-t0)/CLOCKS_PER_SEC);
    std::cout << "%Execution Time: " << time << std::endl;
    Image<double> Vin, Hin1, Hin2;
    Vin.read(fnVol);
    Hin1.read(fnHalf1);
    Hin2.read(fnHalf2);

    Vin().setXmippOrigin();
    Hin1().setXmippOrigin();
    Hin2().setXmippOrigin();

    MultidimArray<double> Vorig = Vin();
    MultidimArray<double> Hmap1 = Hin1();
    MultidimArray<double> Hmap2 = Hin2();

    ////*********mask
    if (mask_exist == true)
    {
        Image<int> mask;
        mask.read(fnMask);
        mask().setXmippOrigin();
        MultidimArray<int> pMask=mask();
        mono.applyMask(Hmap1, mask());
        mono.applyMask(Hmap1, mask());
//        mono.applyMask(Vorig, mask());
    }

    FilterFunction(Nfaces, coneMask, Vorig, Hmap1, Hmap2, transformer_inv);

//    transformer.setThreadsNumber(Nthread);
//    MultidimArray< std::complex<double> > fftM1, fftM2;
//
//    FourierTransformer transformer1(FFTW_BACKWARD);
//    FourierTransformer transformer2(FFTW_BACKWARD);
//
//    std::cout << "Hago transformadas" << std::endl;
//    transformer1.FourierTransform(Hmap1, fftM1, false);
//    transformer2.FourierTransform(Hmap2, fftM2, false);
//    std::cout << "Termino transformadas" << std::endl;
//
//    VolumesFsc(fftM1,fftM2);

}



