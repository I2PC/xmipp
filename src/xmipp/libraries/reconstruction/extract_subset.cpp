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

#include "extract_subset.h"
#include <ostream>

void ExtractSubset::createSubset(const Settings &s) {
    s.check();
    auto it = MDIterator(s.md);
    // get to proper position
    for (size_t i = 0; i < s. first; ++i) {
        it.moveNext();
    }
    MDRow row;
    auto destMD = MetaData();
    auto img = Image<float>();
    // iterate through all items
    for (size_t i = 0; i < s.count; it.moveNext(), i++) {
        // orig name
        FileName origName;
        s.md.getValue(MDL_IMAGE, origName, it.objId);
        // new name
        FileName newName;
        newName.compose(i + 1, s.outStk); // within stk file, index images from one (1)
        // copy row, replace name
        s.md.getRow(row, it.objId);
        row.setValue(MDL_IMAGE, newName);
        row.setValue(MDL_ENABLED, 1);
        destMD.addRow(row);
        // copy image
        // FIXME DS maybe we can do this more efficiently
        img.read(origName);
        img.write(newName, i, true, WRITE_APPEND);
    }
    // store metadata
    destMD.write(s.outXmd);
}

void ExtractSubset::Settings::check() const {
    // target directories must exist, so that we can write there
    if ( ! outXmd.getDir().exists()) {
        REPORT_ERROR(ERR_IO_NOTEXIST, "Directory " + outXmd.getDir() + " does not exist");
    }
    if ( ! outStk.getDir().exists()) {
        REPORT_ERROR(ERR_IO_NOTEXIST, "Directory " + outStk.getDir() + " does not exist");
    }
    // target files must not exist, otherwise we will append to them
    if (outXmd.exists()) {
        REPORT_ERROR(ERR_IO_NOTEXIST, "File  " + outXmd + " already exists.");
    }
    if (outStk.exists()) {
        REPORT_ERROR(ERR_IO_NOTEXIST, "File " + outStk + " already exists");
    }
    const size_t n = md.size();
    if (first >= n) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Zero-based index of the first item ("
                + std::to_string(first) + ") is bigger than size of the metadata ("
                + std::to_string(n) + ")");
    }
    if (first + count > n) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Out of range ("
                + std::to_string(first) + " + "
                + std::to_string(count) + " >= "
                + std::to_string(n) + ")");
    }
}

std::ostream& operator<<(std::ostream &os, const ExtractSubset::Settings &s) {
    os << "Input metadata   : " << s.md.getFilename() << "\n";
    os << "Output metadata  : " << s.outXmd << "\n";
    os << "Matching items   : " << s.count << "\n";
    os << "Skip disabled    : " << (s.skipDisabled ? "yes" : "no") << "\n";
    os.flush();
    return os;
}

