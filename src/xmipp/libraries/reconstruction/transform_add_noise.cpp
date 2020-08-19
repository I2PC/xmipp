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

#include <random>
#include "core/xmipp_metadata_program.h"
#include "core/xmipp_error.h"
#include "core/xmipp_image.h"

class ProgAddNoise: public XmippMetadataProgram
{
private:
    const std::string TYPE_POISSON = "poisson";

protected:
    double param1, param2;
    double df, limit0, limitF;
    bool   do_limit0, do_limitF;
    std::string noise_type;

    void defineParams()
    {
        each_image_produces_an_output = true;
        XmippMetadataProgram::defineParams();
        //Usage
        addUsageLine("Add random noise to the input images.");
        addUsageLine("Noise can be generated using uniform, gaussian or t-student distributions.");
        //Parameters
        addParamsLine("--type <rand_mode>                : Type of noise to add");
        addParamsLine("       where <rand_mode>");
        addParamsLine("              gaussian <stddev> <avg=0.>     :Gaussian distribution parameters");
        addParamsLine("              student <df> <stddev> <avg=0.> :t-student distribution parameters");
        addParamsLine("              uniform  <min> <max>           :Uniform distribution parameters");
        addParamsLine("              " + TYPE_POISSON + "  <min> <max>          :Poission distribution. Each pixel i of output is generated as poisson(ref-input[i])");
        addParamsLine("  [--limit0 <float> ]               :Crop noise histogram below this value ");
        addParamsLine("  [--limitF <float> ]               :Crop noise histogram above this value ");
        //Examples
        addExampleLine("Add noise to a single image, writing in different image:", false);
        addExampleLine("xmipp_transform_add_noise -i cleanImage.spi --type gaussian 10 5 -o noisyGaussian.spi");
        addExampleLine("+++Following =cleanImage.spi= at left and =noisyGaussian.spi= at right: %BR%", false);
        addExampleLine("+++%ATTACHURL%/cleanImage.jpg %ATTACHURL%/noisyGaussian.jpg  %BR%", false);
        addExampleLine("Add uniform noise to a volume, overriding input volume:", false);
        addExampleLine("xmipp_transform_add_noise -i g0ta.vol -uniform -0.1 0.1");

    }

    void readParams()
    {
        XmippMetadataProgram::readParams();
        do_limit0 = checkParam("--limit0");
        if (do_limit0)
            limit0 =  getDoubleParam("-limit0");
        do_limitF = checkParam("--limitF");
        if (do_limitF)
            limitF =  getDoubleParam("--limitF");

        ///Default value of df in addNoise function
        df = 3.;
        noise_type = getParam("--type");

        if (noise_type == "gaussian")
        {
            param2 = getDoubleParam("--type", 1);
            param1 = getDoubleParam("--type", 2);
        }
        else if (noise_type == "student")
        {
            df = getDoubleParam("--type", 1);
            param1 = getDoubleParam("--type", 2);
            param2 = getDoubleParam("--type", 3);
        }
        else if (noise_type == "uniform")
        {
            param1 = getDoubleParam("--type", 1);
            param2 = getDoubleParam("--type", 2);
        }
        else if (TYPE_POISSON == noise_type)
        {
            param1 = getDoubleParam("--type", 1);
            param2 = getDoubleParam("--type", 2);
        }
        else
            REPORT_ERROR(ERR_ARG_INCORRECT, "Unknown noise type");
    }

    void show()
    {
        XmippMetadataProgram::show();
        if (noise_type == "gaussian")
            std::cout << "Noise avg=" << param1 << std::endl
            << "Noise stddev=" << param2 << std::endl;
        else if (noise_type == "student")
            std::cout << "Degrees of freedom= "<< df << std::endl
            << "Noise avg=" << param1 << std::endl
            << "Noise stddev=" << param2 << std::endl;
        else if (noise_type == "uniform")
            std::cout << "Noise min=" << param1 << std::endl
            << "Noise max=" << param2 << std::endl;
        else if (TYPE_POISSON == noise_type)
            std::cout << "Mean background=" << param1 << std::endl
                        << "Mean foreground=" << param2 << std::endl;
        if (do_limit0)
            std::cout << "Crop noise histogram below=" << limit0 << std::endl;
        if (do_limitF)
            std::cout << "Crop noise histogram above=" << limitF << std::endl;
    }

    template<typename T>
    void limit(Image<T> &img) {
        if (do_limit0) {
            const size_t count = img.data.nzyxdim;
            for (size_t i = 0; i < count; ++i) {
                img.data[i] = XMIPP_MAX(img.data[i], limit0);
            }
        }
        if (do_limitF) {
            const size_t count = img.data.nzyxdim;
            for (size_t i = 0; i < count; ++i) {
                img.data[i] = XMIPP_MIN(img.data[i], limitF);
            }
        }
    }

    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
    {

        if (TYPE_POISSON == noise_type) {
            Image<float> img;
            img.readApplyGeo(fnImg, rowIn);
            std::random_device rd;
            std::mt19937 gen(rd());
            Image<int> res(img.data.xdim, img.data.ydim, img.data.zdim, img.data.ndim);
            const size_t count = res.data.nzyxdim;
            const float gap = param1 - param2;
            auto dist = std::poisson_distribution<>(0);
            for (size_t i = 0; i < count; ++i) {
                float mean = param1 - gap * img.data[i];
                if (dist.mean() != mean) { // reuse distribution, if possible
                    dist = std::poisson_distribution<>(mean);
                }
                res.data[i] = dist(gen);
            }
            limit(res);
            res.write(fnImgOut);
        } else {
            Image<double> img;
            img.readApplyGeo(fnImg, rowIn);
            img().addNoise(param1, param2, noise_type, df);
            limit(img);
            img.write(fnImgOut);
        }

    }

}
;//end of class ProgAddNoise
