/***************************************************************************
 *
 * Authors:    Carlos Oscar            coss@cnb.csic.es (2010)
 *             Joaquin Oton      joton@cnb.csic.es (2013)
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

#include <core/args.h>
#include <data/filters.h>
#include <core/xmipp_image_extension.h>
#include "xray_import.h"

// usage ===================================================================
void ProgXrayImport::defineParams()
{
    addUsageLine("Preprocess X-ray micrograph to be used in tomographic reconstruction software");
    addUsageLine(" ");
    addUsageLine("+This program converts a single-axis tilt series coming from X-ray");
    addUsageLine("+microscopy into a stack. Several corrections are applied ");
    addUsageLine("+during this conversion including correction by the beam current, ");
    addUsageLine("+the exposure time, and the slitwidth. The flatfield and a possible ");
    addUsageLine("+darkfield are also corrected. The exact formula applied is");
    addUsageLine("+ ");
    addUsageLine("+                       (I-darkfield)/(expTime*beamCurrent*slitWidth)",true);
    addUsageLine("+Inormalized =  -----------------------------------------------------------",true);
    addUsageLine("+                 Avg{(Iflatfield-darkfield)/(expTime*beamCurrent*slitWidth)}",true);
    addUsageLine("+ ");
    addUsageLine("+Because the intrinsic self-attenuation, X-ray projections are not optimal to be used ");
    addUsageLine("+with EM 3D reconstruction algorithm. To fix that, use --log flag:");
    addUsageLine("+ ");
    addUsageLine("+log = log10(Inormalized)");
    addUsageLine("+ ");
    addUsageLine("+In addition to log correction, to apply a contrast inversion, which allows 3DEM ");
    addUsageLine("+reconstruction algorithms returning volume coefficients close to the real expected ");
    addUsageLine("+absorption values, use --correct flag:");
    addUsageLine("+ ");
    addUsageLine("+Icorrected = -log10(Inormalized)");
    addUsageLine("+ ");
    addUsageLine("+If darkfield/flatfield are not available, then they are not used for the correction.");
    addUsageLine("+ ");
    addUsageLine("+Specific flags to import tomo series from Mistral microscope (Alba synchrotron) and ");
    addUsageLine("+U41-TXM (Bessy).");
    addUsageLine("+ ");
    addUsageLine("+In the general case, the flatfields and the the tilt series may be in any directory.");
    addUsageLine("+ If the darkfield are available, they must be within the flatfield and the tilt");
    addUsageLine("+series under a directory named darkfields (in small letters). For each");
    addUsageLine("+SPE file there must be a positions file. For instance, the file myfile.spe");
    addUsageLine("+must have in the same directory the file myfile-positions.txt. The");
    addUsageLine("+exposure time, the beam current and the slit width are taken from this file.");
    addUsageLine("+ ");
    addUsageLine("+The exposure time is the parameter xm:ccd:exp_time, the beam current");
    addUsageLine("+is sr:current, and the slit width is xm:mono:slitwidth. The tilt angle");
    addUsageLine("+xm:sample:rx.");

    addParamsLine("[--input <input_directory> ]         : Directory with tomograms, position files and, optionally, ");
    addParamsLine("                                     :  darkfields (in a directory called darkfields)");
    addParamsLine("[--flat <ff_rootname=\"\">]         : Directory with SPE images, position files and, optionally, ");
    addParamsLine("                                     : darkfields (in a directory called darkfields).");
    addParamsLine("                                     : This param can also be used with specific microscopes params to ");
    addParamsLine("                                     : use a stack of flatfields to average, or an averaged flatfield image");
    addParamsLine("[--oroot <output_rootname>]         :  Rootname for output files. If empty, input filename is taken.");
    addParamsLine("                                     :+ The following files are created:");
    addParamsLine("                                     :+ rootname.mrc: MRC stack with tilt series fully corrected as described above");
    addParamsLine("                                     :+ rootname_darkfield.xmp: darkfield for the tilt series");
    addParamsLine("                                     :+ rootname_flatfields_darkfield.xmp: darkfield of the flatfields");
    addParamsLine("                                     :+ rootname_flatfields_avg.xmp: average flatfield corrected by its darkfield");
    addParamsLine("                                     :+ rootname.xmd: selfile with the images in the stack");
    addParamsLine("                                     :+ rootname.tlt: list of angles");
    addParamsLine("  [--crop <sizeX=0> <sizeY=sizeX>]   : Number of pixels to crop from each side");
    addParamsLine("                                     : This is used to avoid some black pixels of some cameras");
    addParamsLine("  [--thr  <N=1>]                     : Number of threads");
    addParamsLine("   == Specific microscopes           ");
    addParamsLine("[--bessy <input_directory> <t_ini> <t_end> <f_ini> <f_end> ] : Directory with raw SPE images and position files acquired in ");
    addParamsLine("          : Bessy X-ray microscope:");
    addParamsLine("                                     : t_ini and t_end denotes the range of the tomogram images, and");
    addParamsLine("                                     : f_ini and f_end denotes the range of the flatfield images stored in the same directory");
    addParamsLine("[--mistral <input_file>]       : hdf5 Nexus file acquired in Mistral microscope at Alba, which contains all data");
    addParamsLine("   == Filters                                          ");
    addParamsLine("  [--bad_pixels <type=factor>]    : Apply a boundaries median filter to bad pixels");
    addParamsLine("       where <type>  ");
    addParamsLine("           factor <k=3.>         : Applied at those pixels below [mean - k*std]");
    addParamsLine("           mask <mask_file>        : Applied at those pixels given by mask");
    addParamsLine("           none               : Do not apply the filter");
    addParamsLine("  [--log]                            : Apply log to pixel values");
    addParamsLine("  alias -l;");
    addParamsLine("  [--correct]                        : Correct for the self-attenuation of X-ray projections applying ");
    addParamsLine("                  : log and multiplying by -1");
    addParamsLine("  alias -c;");
    addExampleLine("Standard call for spe files classified in folders is:",false);
    addExampleLine("xmipp_xray_import --data 10s --flat flatfields --oroot ProcessedData/img --crop 7");
    addExampleLine("Importing data from Alba synchrotron:",false);
    addExampleLine("xmipp_xray_import --mistral tomogram.hdf5 --oroot imported_tomogram");
    addExampleLine("Correcting for self-áttenuation and cropping:",false);
    addExampleLine("xmipp_xray_import --mistral tomogram.hdf5 --oroot imported_tomogram --correct --crop 10");
}



void ProgXrayImport::init()
{
    tIni = tEnd = fIni = fEnd = 0;
    extFlat = darkFix = false;
    BPFactor = 0;
    dSource = NONE;
}

// Read arguments ==========================================================
void ProgXrayImport::readParams()
{

    if (checkParam("--mistral"))
    {
        fnInput = getParam("--mistral");
        fnFlat = "NXtomo/instrument/bright_field/data@" + fnInput;
        dSource = MISTRAL;
    }
    else if (checkParam("--bessy"))
    {
        fnInput = getParam("--bessy", 0);
        tIni = getIntParam("--bessy", 1);
        tEnd = getIntParam("--bessy", 2);
        fIni = getIntParam("--bessy", 3);
        fEnd = getIntParam("--bessy", 4);
        fnFlat = fnInput;
        dSource = BESSY;
    }
    else
    {
        fnInput = getParam("--input");
        dSource = GENERIC;
    }
    // If --flat is passed it forces the use of this flatfield
    if (checkParam("--flat"))
    {
        fnFlat = getParam("--flat");
        extFlat = true;
    }

    fnRoot    = getParam("--oroot");
    cropSizeX  = getIntParam("--crop");

    if ( STR_EQUAL(getParam("--crop", 1), "sizeX") )
        cropSizeY = cropSizeX;
    else
        cropSizeY = getIntParam("--crop",1);


    thrNum    = getIntParam("--thr");

    String tempString = getParam("--bad_pixels");
    if (tempString == "factor")
        BPFactor = getDoubleParam("--bad_pixels", 1);
    else if (tempString == "mask")
        fnBPMask  = getParam("--bad_pixels",1);
    else
        BPFactor = -1;

    selfAttFix   = checkParam("--correct");
    logFix   = (selfAttFix)? true : checkParam("--log");
}

// Show ====================================================================
void ProgXrayImport::show() const
{
    std::ostream &out = std::cout;

    out << "Input data directory       : " << fnInput    << std::endl;
    if (!fnFlat.empty())
        out << "Input flatfield directory  : " << fnFlat  << std::endl;
    out << "Output rootname            : " << fnRoot     << std::endl;
    out << "Crop sizes                 : " << formatString("%dx%d",cropSizeX,cropSizeY).c_str()   << std::endl;
    out << "Number of threads          : " << thrNum     << std::endl;
    ;
}

// Really import ==========================================================
void ProgXrayImport::readAndCrop(const FileName &fn, Image<double> &I, int xCropSize, int yCropSize) const
{
    I.read(fn);
    I().selfWindow(yCropSize,xCropSize,
                   (int)(YSIZE(I())-yCropSize-1),(int)(XSIZE(I())-xCropSize-1));

    I().resetOrigin();
}


void ProgXrayImport::readGeoInfo(const FileName &fn, MDRow &rowGeo) const
{
    double tiltAngle;

    switch (dSource)
    {
    case MISTRAL:
        tiltAngle = dMi(anglesArray, fn.getPrefixNumber()-1);
        break;
    case BESSY:
    case GENERIC:
        {
            FileName fnBase = fn.withoutExtension();
            std::ifstream fhPosition;
            fhPosition.open((fnBase+"-positions.txt").c_str());
            if (!fhPosition)
                REPORT_ERROR(ERR_IO_NOTEXIST,fnBase+"-positions.txt");
            int itemsFound=0;
            std::string line;
            std::vector<std::string> tokens;
            while (!fhPosition.eof())
            {
                getline(fhPosition,line);
                splitString(line," ",tokens);
                if (tokens[0]=="xm:sample:rx")
                {
                    tiltAngle=textToFloat(tokens[1]);
                    itemsFound++;
                }
                if (itemsFound==1)
                    break;
            }
            fhPosition.close();
            if (itemsFound!=1)
                REPORT_ERROR(ERR_VALUE_EMPTY,(std::string)"Cannot find tilt angle in "+
                             fnBase+"-positions.txt");
        }
        break;
    }

    rowGeo.setValue(MDL_ANGLE_TILT, tiltAngle);
}

void ProgXrayImport::readCorrectionInfo(const FileName &fn, double &currentBeam,
                                        double &expTime, double &slitWidth) const
{
    FileName fnBase = fn.withoutExtension();
    std::ifstream fhPosition;
    fhPosition.open((fnBase+"-positions.txt").c_str());

    if (!fhPosition)
        REPORT_ERROR(ERR_IO_NOTEXIST,fnBase+"-positions.txt");
    int itemsFound = 0;
    std::string line;
    std::vector<std::string> tokens;

    while (!fhPosition.eof())
    {
        getline(fhPosition,line);
        splitString(line," ",tokens);
        if (tokens[0]=="sr:current")
        {
            currentBeam = textToFloat(tokens[1]);
            itemsFound++;
        }
        else if (tokens[0]=="xm:ccd:exp_time")
        {
            expTime = textToFloat(tokens[1]);
            itemsFound++;
        }
        else if (tokens[0]=="xm:mono:slitwidth")
        {
            slitWidth = textToFloat(tokens[1]);
            itemsFound++;
        }
        if (itemsFound==3)
            break;
    }
    fhPosition.close();

    if ( itemsFound != 3 )
        REPORT_ERROR(ERR_VALUE_EMPTY,(std::string)"Cannot find all parameters in "+
                     fnBase+"-positions.txt");
}

void ProgXrayImport::getDarkfield(const FileName &fnDir, Image<double> &IavgDark)
{
    IavgDark.clear();
    std::vector<FileName> listDir;
    fnDir.getFiles(listDir);

    for (size_t i=0; i<listDir.size(); i++)
        if (listDir[i]=="darkfields")
        {
            std::cout << formatString("Getting darkfield from %s/darkfields",fnDir.c_str()) << " ..." << std::endl;

            darkFix = true;
            std::vector<FileName> listDirDark;
            FileName(fnDir+"/darkfields").getFiles(listDirDark);
            int N = 0;

            for (size_t j=0; j<listDirDark.size(); j++)
            {
                if (!listDirDark[j].hasImageExtension())
                    continue;
                Image<double> Iaux;
                readAndCrop(fnDir+"/darkfields/"+listDirDark[j],Iaux, cropSizeX, cropSizeY);
                if (N==0)
                    IavgDark()=Iaux();
                else
                    IavgDark()+=Iaux();
                N++;
            }
            if (N==0)
                REPORT_ERROR(ERR_IO_NOTEXIST,"darkfields directory is empty");
            IavgDark()*=1.0/N;
            break;
        }
}

void ProgXrayImport::getFlatfield(const FileName &fnFFinput,
                                  Image<double> &Iavg)
{

    // Process the flatfield images


    MultidimArray<double> &mdaIavg = Iavg();

    Matrix1D<double> expTimeArray, cBeamArray, slitWidthArray; // Local vectors

    DataSource ldSource = dSource;

    // Checking if fnFFinput is a single file to obtain the flatfield avg from
    if (extFlat && isImage(fnFFinput))
    {
        fMD.read(fnFFinput);
        ldSource = NONE;
    }
    else
    {
        switch (ldSource)
        {
        case MISTRAL:
            if (!H5File.checkDataset(fnFFinput.getBlockName().c_str()))
                break;
            {
                fMD.read(fnFFinput);
                H5File.getDataset("NXtomo/instrument/bright_field/ExpTimes", expTimeArray, false);
                H5File.getDataset("NXtomo/instrument/bright_field/current", cBeamArray, false);

                // If expTime is empty or only one single value in nexus file then we fill with 1
                if (expTimeArray.size() < 2)
                {
                    reportWarning("Input file does not contains flatfields' exposition time information.");
                    expTimeArray.initConstant(fMD.size(), 1.);
                }
                // If current is empty or only one single value in nexus file then we fill with 1
                if (cBeamArray.size() < 2)
                {
                    reportWarning("Input file does not contains flatfields' current beam information.");
                    cBeamArray.initConstant(fMD.size(), 1.);
                }

            }

            // Since Alba does not provide slit width, we set to ones
            slitWidthArray.initConstant(fMD.size(), 1.);

            break;
        case BESSY:
            {
                size_t objId;

                for (size_t i = fIni; i <= fEnd; ++i)
                {
                    objId = fMD.addObject();
                    fMD.setValue(MDL_IMAGE, fnFFinput + formatString("/img%d.spe", i), objId);
                }
                break;
            }
        case GENERIC:
            {
                // Get Darkfield
                std::cout << "Getting darkfield from "+fnFFinput << " ..." << std::endl;
                getDarkfield(fnFFinput, IavgDark);
                if (darkFix)
                    IavgDark.write(fnRoot+"_"+fnFFinput.removeDirectories()+"_darkfield.xmp");

                std::vector<FileName> listDir;
                fnFFinput.getFiles(listDir);
                size_t objId;

                for (size_t i = 0; i < listDir.size(); ++i)
                {
                    if (!listDir[i].hasImageExtension())
                        continue;
                    objId = fMD.addObject();
                    fMD.setValue(MDL_IMAGE, fnFFinput+"/"+listDir[i], objId);
                }
            }
            break;
        }
    }

    if ( fMD.size() == 0 )
    {
        reportWarning("XrayImport::getFlatfield: No images to process");
        return;
    }

    ImageInfo imFFInfo;

    getImageInfo(fMD, imFFInfo);

    if ( (imFFInfo.adim.xdim != imgInfo.adim.xdim) || (imFFInfo.adim.ydim != imgInfo.adim.ydim) )
    {
        reportWarning(formatString("XrayImport:: Flatfield images size %dx%d different from Tomogram images size %dx%d",
                                   imFFInfo.adim.xdim,imFFInfo.adim.ydim,imgInfo.adim.xdim,imgInfo.adim.ydim));
        std::cout << "Setting crop values to fit the smallest dimensions." <<std::endl;

        // This shift in the crop sizes is exclusive of Mistral data
        cropSizeXi = (imgInfo.adim.xdim - std::min(imgInfo.adim.xdim-cropSizeX*2, imFFInfo.adim.xdim))/2 + 1;
        cropSizeYi = (imgInfo.adim.ydim - std::min(imgInfo.adim.ydim-cropSizeY*2, imFFInfo.adim.ydim))/2 + 1;
        cropSizeXe = cropSizeXi - 2;
        cropSizeYe = cropSizeYi - 2;
    }

    int cropX = (imFFInfo.adim.xdim - (imgInfo.adim.xdim-cropSizeXi-cropSizeXe))/2;// - 1;
    int cropY = (imFFInfo.adim.ydim - (imgInfo.adim.ydim-cropSizeYi-cropSizeYe))/2;// - 1;


    int N  = 0;
    Image<double> Iaux;
    FileName fnImg;

    FOR_ALL_OBJECTS_IN_METADATA(fMD)
    {
        fMD.getValue(MDL_IMAGE, fnImg, __iter.objId);

        readAndCrop(fnImg, Iaux, cropX, cropY);


        if ( darkFix )
        {
            Iaux() -= IavgDark();
            forcePositive(Iaux());
        }

        double currentBeam = 1;
        double expTime = 1;
        double slitWidth = 1;

        switch (ldSource)
        {
        case MISTRAL:
            {
                size_t idx = fnImg.getPrefixNumber();
                currentBeam = dMi(cBeamArray, idx-1);
                expTime = dMi(expTimeArray, idx-1);
                slitWidth = dMi(slitWidthArray, idx-1);
            }
            break;
        case BESSY:
        case GENERIC:
            readCorrectionInfo(fnImg, currentBeam, expTime, slitWidth);
            break;
        default:
            break;
        }

        Iaux() *= 1.0/(currentBeam*expTime*slitWidth);

        if ( N == 0 )
            mdaIavg = Iaux();
        else
            mdaIavg += Iaux();
        N++;
    }

    darkFix = false; // We reset just in case there is no dark field for tomo images

    if (0 == N) {
        REPORT_ERROR(ERR_NUMERICAL, "N is zero (0), which would lead to division by zero");
    }
    mdaIavg/=N;

    /* Create a mask with zero valued pixels to apply boundaries median filter
     * to avoid dividing by zero when normalizing */

    MultidimArray<char> &mdaMask = MULTIDIM_ARRAY(bpMask);

    if ( !fnBPMask.empty() && !mdaIavg.sameShape(mdaMask) )
        REPORT_ERROR(ERR_MULTIDIM_SIZE, "XrayImport: Mask size does not match flat fields size.");

    if (BPFactor > 0)
    {
        double  avg, std;
        mdaIavg.computeAvgStdev(avg, std);
        mdaMask.resize(mdaIavg, false);

        double th = avg - std*BPFactor;
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mdaMask)
        dAi(mdaMask, n) = dAi(mdaIavg, n) < th;
    }
    else if (fnBPMask.empty())
        mdaIavg.equal(0,mdaMask);

    MultidimArray<char> mask = mdaMask;
    boundMedianFilter(mdaIavg,mask);
}

void runThread(ThreadArgument &thArg)
{
    int thread_id = thArg.thread_id;
    ProgXrayImport * ptrProg= (ProgXrayImport *)thArg.workClass;

    MetaData localMD;
    Image<double> Iaux;
    FileName fnImgIn, fnImgOut;
    size_t first = 0, last = 0;
    MultidimArray<char> mask;

    while (ptrProg->td->getTasks(first, last))
    {
        for (size_t i=first; i<=last; i++)
        {
            ptrProg->inMD.getValue(MDL_IMAGE, fnImgIn, ptrProg->objIds[i]);


            MDRow rowGeo;
            ptrProg->readGeoInfo(fnImgIn, rowGeo);
//            ptrProg->readAndCrop(fnImgIn, Iaux, ptrProg->cropSizeX, ptrProg->cropSizeY);

            Iaux.read(fnImgIn);
            Iaux().selfWindow(ptrProg->cropSizeYi,ptrProg->cropSizeXi,
            		(int)(YSIZE(Iaux())-ptrProg->cropSizeYe-1),(int)(XSIZE(Iaux())-ptrProg->cropSizeXe-1));

            Iaux().resetOrigin();




            if (XSIZE(ptrProg->IavgDark())!=0)
            {
                Iaux()-=ptrProg->IavgDark();
                forcePositive(Iaux());
            }


            double currentBeam = 1;
            double expTime = 1;
            double slitWidth = 1;

            if ( ptrProg->dSource == ptrProg->MISTRAL )
            {
                size_t idx = fnImgIn.getPrefixNumber();
                currentBeam = dMi(ptrProg->cBeamArray, idx-1);
                expTime = dMi(ptrProg->expTimeArray, idx-1);
                slitWidth = dMi(ptrProg->slitWidthArray, idx-1);
            }
            else
                ptrProg->readCorrectionInfo(fnImgIn, currentBeam, expTime, slitWidth);

            Iaux() *= 1.0/(currentBeam*expTime*slitWidth);
            if (XSIZE(ptrProg->IavgFlat())!=0)
                Iaux()/=ptrProg->IavgFlat();

            // Assign median filter to zero valued pixels to avoid -inf when applying log10
            Iaux().equal(0,mask);
            mask.resizeNoCopy(Iaux());

            if (XSIZE(ptrProg->bpMask()) != 0)
                mask += ptrProg->bpMask();

            boundMedianFilter(Iaux(), mask);

            if (ptrProg->logFix)
            {
                Iaux().selfLog();
                if (ptrProg->selfAttFix)
                    Iaux() *= -1.;
            }

            fnImgOut.compose(i+1, ptrProg->fnOut);


            size_t objId = localMD.addObject();
            localMD.setValue(MDL_IMAGE,fnImgOut,objId);
            localMD.setRow(rowGeo, objId); //
            //            localMD.setValue(MDL_ANGLE_TILT,Iaux.tilt(),objId);
            Iaux.write(fnImgOut);
            if (thread_id==0)
                progress_bar(i);
        }
    }
    //Lock for update the total counter
    ptrProg->mutex.lock();
    ptrProg->outMD.unionAll(localMD);
    ptrProg->mutex.unlock();
}

void ProgXrayImport::run()
{
    // Delete output stack if it exists
    fnOut = fnRoot + ".mrc";
    fnOut.deleteFile();

    /* Turn off error handling */
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    if (dSource == MISTRAL)
        H5File.openFile(fnInput, H5F_ACC_RDONLY);


    // Reading bad pixels mask
    if ( !fnBPMask.empty() )
    {
        std::cerr << "Reading bad pixels mask from "+fnBPMask << "." << std::endl;
        bpMask.read(fnBPMask);
        if ( (cropSizeX + cropSizeY ) > 0 )
            bpMask().selfWindow(cropSizeY,cropSizeX,
                                (int)(YSIZE(bpMask())-cropSizeY-1),(int)(XSIZE(bpMask())-cropSizeX-1));
        STARTINGX(bpMask()) = STARTINGY(bpMask()) = 0;
    }


    // Setting the image projections list
    switch (dSource)
    {
    case MISTRAL:
        {
            inMD.read(fnInput);
            H5File.getDataset("NXtomo/data/rotation_angle", anglesArray, false);
            H5File.getDataset("NXtomo/instrument/sample/ExpTimes", expTimeArray, false);
            H5File.getDataset("NXtomo/instrument/sample/current", cBeamArray);

            /* In case there is no angles information we set them to to an increasing sequence
             * just to be able to continue importing data */
            if ( anglesArray.size() != inMD.size() )
            {
                reportWarning("Input file does not contains angle information. Default sequence used.");
                anglesArray.resizeNoCopy(inMD.size());
                anglesArray.enumerate();
            }

            // If expTime is empty or only one single value in nexus file then we fill with 1
            if (expTimeArray.size() < 2)
            {
                reportWarning("Input file does not contains tomogram exposition time information.");
                expTimeArray.initConstant(anglesArray.size(), 1.);
            }
            // If current is empty or only one single value in nexus file then we fill with 1
            if (cBeamArray.size() < 2)
            {
                reportWarning("Input file does not contains tomogram current beam information.");
                cBeamArray.initConstant(anglesArray.size(), 1.);
            }
            // Since Alba does not provide slit width, we set to ones
            slitWidthArray.initConstant(anglesArray.size(), 1.);
        }
        break;
    case BESSY:
        {
            size_t objId;

            for (size_t i = tIni; i <= tEnd; ++i)
            {
                objId = inMD.addObject();
                inMD.setValue(MDL_IMAGE, fnInput + formatString("/img%d.spe", i), objId);
            }
            break;
        }
    case GENERIC:
        {
            // Get Darkfield
            std::cerr << "Getting darkfield from "+fnInput << " ..." << std::endl;
            getDarkfield(fnInput, IavgDark);
            if (XSIZE(IavgDark())!=0)
                IavgDark.write(fnRoot+"_darkfield.xmp");


            std::vector<FileName> listDir;
            fnInput.getFiles(listDir);
            size_t objId;

            for (size_t i = 0; i < listDir.size(); ++i)
            {
                if (!listDir[i].hasImageExtension())
                    continue;
                objId = inMD.addObject();
                inMD.setValue(MDL_IMAGE, fnInput+"/"+listDir[i], objId);
            }
        }
        break;
    }

    inMD.findObjects(objIds);
    size_t nIm = inMD.size();

    // Create empty output stack file

    getImageInfo(inMD, imgInfo);


    /* Get the flatfield:: We get the FF after the image list because we need the image size to adapt the FF
     * in case they were already cropped.
     */
    if (!fnFlat.empty())
    {
        std::cout << "Getting flatfield from "+fnFlat << " ..." << std::endl;
        getFlatfield(fnFlat,IavgFlat);
        if ( XSIZE(IavgFlat()) != 0 )
        {
            FileName ffName = fnRoot+"_flatfield_avg.xmp";
            IavgFlat.write(ffName);
            fMD.setValue(MDL_IMAGE, ffName, fMD.addObject());
        }
    }

    createEmptyFile(fnOut, imgInfo.adim.xdim-cropSizeXi-cropSizeXe, imgInfo.adim.ydim-cropSizeYi-cropSizeYe, 1, nIm);

    // Process images
    td = new ThreadTaskDistributor(nIm, XMIPP_MAX(1, nIm/30));
    tm = new ThreadManager(thrNum, this);
    std::cerr << "Getting data from " << fnInput << " ...\n";
    init_progress_bar(nIm);
    tm->run(runThread);
    progress_bar(nIm);

    // Write Metadata and angles
    MetaData MDSorted;
    MDSorted.sort(outMD,MDL_ANGLE_TILT);
    MDSorted.write("tomo@"+fnRoot + ".xmd");
    if ( fMD.size() > 0 )
        fMD.write("flatfield@"+fnRoot + ".xmd", MD_APPEND);

    // We also reference initial and final images at 0 degrees for Mistral tomograms
    if ( dSource == MISTRAL )
    {
        fMD.clear();
        FileName degree0Fn = "NXtomo/instrument/sample/0_degrees_initial_image";
        if ( H5File.checkDataset(degree0Fn.c_str()))
            fMD.setValue(MDL_IMAGE, degree0Fn + "@" + fnInput, fMD.addObject());
        degree0Fn = "NXtomo/instrument/sample/0_degrees_final_image";
        if ( H5File.checkDataset(degree0Fn.c_str()))
            fMD.setValue(MDL_IMAGE, degree0Fn + "@" + fnInput, fMD.addObject());
        if ( fMD.size() > 0 )
            fMD.write("degree0@"+fnRoot + ".xmd", MD_APPEND);
    }

    // Write tlt file for IMOD
    std::ofstream fhTlt;
    fhTlt.open((fnRoot+".tlt").c_str());
    if (!fhTlt)
        REPORT_ERROR(ERR_IO_NOWRITE,fnRoot+".tlt");
    FOR_ALL_OBJECTS_IN_METADATA(MDSorted)
    {
        double tilt;
        MDSorted.getValue(MDL_ANGLE_TILT,tilt,__iter.objId);
        fhTlt << tilt << std::endl;
    }
    fhTlt.close();
    delete td;
    delete tm;
}
