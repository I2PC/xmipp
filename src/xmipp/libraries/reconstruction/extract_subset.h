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

#ifndef EXTRACT_SUBSET_H_
#define EXTRACT_SUBSET_H_

#include "core/xmipp_filename.h"
#include "core/metadata.h"
#include <iosfwd>
#include "core/xmipp_image.h"

/**
 * Class responsible for extracting particles from metadata / stack file into
 * a new metadata / stack file
 *
 */
class ExtractSubset final {
public:
    class Settings {
    public:
        MetaData md; // input
        FileName outXmd; // parent directory expected to exist
        FileName outStk; // parent directory expected to exist
        size_t first; // 0-based index
        size_t count; // number of items to be included
        bool skipDisabled;

        void check() const;
    };

    friend std::ostream& operator<<(std::ostream &os, const Settings &s);

    /**
     * This method will extract particles according to the passed settings.
     * New files containing the subset shall be created.
     */
    static void createSubset(const Settings &s);
};

#endif /* EXTRACT_SUBSET_H_ */
