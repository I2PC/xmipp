/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
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


#include "prog_extract_subset_main.h"

void ProgExtractSubset::defineParams() {
    addUsageLine("Extract a subset of particles into new file. By default, only 'Enabled' particles will be preserved.");

    addParamsLine(" -i <md_file>                : Input metadata");
    addParamsLine(" -o <md_file>                : Output metadata");
    addParamsLine(" [--odir <outputDir=\".\">]  : Output directory");
    addParamsLine(" [--keepDisabled]            : Ignore 'Enable' flag, i.e. include in the subset also disabled particles");
    addParamsLine(" [--first <n=10>]            : Extract first up to n particles");
    addParamsLine(" [--last <n=10>]             : Extract last up to n particles");
    addParamsLine(" [--range <s=0> <e=10>]      : Extract particles within this zero-based range");
}

void ProgExtractSubset::readParams() {
    m_settings.skipDisabled = ! checkParam("--keepDisabled");
    MetaData &md = m_settings.md;
    md.read(getParam("-i"));
    if (m_settings.skipDisabled) {
        md.removeDisabled();
    }
    size_t n = md.size();

    if (checkParam(OPT_FIRST)) {
        printf("opt first\n"); // FIXME DS remove
        m_settings.first = 0;
        m_settings.count = std::min(n, (size_t)getIntParam(OPT_FIRST));
    } else if (checkParam(OPT_LAST)) {
        printf("opt last\n"); // FIXME DS remove
        m_settings.count = std::min(n, (size_t)getIntParam(OPT_LAST));
        m_settings.first = (0 == m_settings.count) ? 0 : n - m_settings.count;
    } else if (checkParam(OPT_RANGE)) {
        printf("opt range\n"); // FIXME DS remove
        size_t reqS = getIntParam(OPT_RANGE, 0);
        size_t reqE = getIntParam(OPT_RANGE, 1);
        size_t last = std::min((0 == n ? 0 : n - 1), reqE);
        m_settings.first = std::min((0 == n ? 0 : n - 1), reqS);
        m_settings.count = last - m_settings.first + 1;
    } else {
        printf("opt active\n"); // FIXME DS remove
        m_settings.first = 0;
        m_settings.count = n;
    }

    setOutput();
}

void ProgExtractSubset::setOutput() {
    auto outDir = FileName(getParam("--odir"));
    if ( ! outDir.exists()) {
        if (outDir.makePath()) {
            REPORT_ERROR(ERR_IO_NOWRITE, "cannot create " + outDir);
        }
    }
    m_settings.outXmd = outDir + "/" + std::string(getParam("-o"));
    m_settings.outStk = outDir + "/" + m_settings.outXmd.getBaseName() + ".stk";
}

void ProgExtractSubset::show() const {
    if (verbose < 1) return;

    std::cout << m_settings;
}

void ProgExtractSubset::run() {
    show();
    ExtractSubset::createSubset(m_settings);
}


RUN_XMIPP_PROGRAM(ProgExtractSubset)
