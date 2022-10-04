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

#include <vector>
#include <mutex>

enum class MemType
{
    CPU_PAGE_ALIGNED,
    CUDA_MANAGED,
    CUDA_HOST, // pinned memory on CPU
    CUDA,
};

class BasicMemManager
{

private:
    struct Record
    {
        Record(size_t bytes, MemType type) : ptr(nullptr), bytes(bytes), available(false), type(type){};
        void *ptr;
        size_t bytes;
        bool available;
        MemType type;
    };

public:
    BasicMemManager(const BasicMemManager &) = delete;
    BasicMemManager &operator=(const BasicMemManager &) = delete;
    BasicMemManager(BasicMemManager &&) = delete;
    BasicMemManager &operator=(BasicMemManager &&) = delete;

    ~BasicMemManager();

    static auto &instance()
    {
        static BasicMemManager instance;
        return instance;
    }

    /**
     * Obtain raw memory block.
     * This might lead to memory allocation.
     * Does not ensure that the memory is actually allocated (i.e. you might receive nullptr).
     * Consider zero-ing it before use. This method does not offer it on purpose as
     * e.g. cuda memory can be memset asynchronously in specific stream.
     **/
    void *get(size_t bytes, MemType type);

    /**
     * Return raw memory block.
     * Block is expected to be allocated via this memory manager.
     * Memory might not be released.
     **/
    void give(void *ptr);

    /**
     * Release all available (unused) memory blocks.
     **/
    void release();

    /**
     * Release all available (unused) memory blocks of specific type.
     **/
    void release(MemType type);

private:
    BasicMemManager() = default; // Disallow instantiation outside of the class.

    /**
     * Allocate new memory block of specific size and type and returns pointer to it.
     * Does not check that the allocation was successful.
     **/
    void *alloc(size_t bytes, MemType type) const;

    /**
     * Release existing memory block of specific type.
     **/
    void release(void *ptr, MemType type) const;

    /**
     * Get smallest available memory block of given size (up to 10% more) and type.
     * Returns nullptr if no such block exists.
     * The block is not changed.
     * Not thread safe.
     **/
    Record *find(size_t bytes, MemType type);

    std::mutex mutex;                 // used to support multithreading
    std::vector<Record> memoryBlocks; // all allocated blocks are stored here
};
