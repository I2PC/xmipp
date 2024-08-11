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

#include "reconstruction_adapt_cuda/basic_mem_manager.h"
#include "reconstruction_cuda/cuda_asserts.h"
#include "core/utils/memory_utils.h"
#include <algorithm>
#include <iostream>

void *BasicMemManager::get(size_t bytes, MemType type)
{
    if (0 == bytes) return nullptr;
    std::unique_lock<std::mutex> lock(mutex);
    auto *b = find(bytes, type);
    if (nullptr == b)
    {

        memoryBlocks.emplace_back(bytes, type);
        b = &memoryBlocks.back();
    }
    if (nullptr == b->ptr)
    {
        b->ptr = alloc(bytes, type);
    }
    b->available = false;
    return b->ptr;
}

void BasicMemManager::give(void *ptr)
{
    if (nullptr == ptr) return;
    std::unique_lock<std::mutex> lock(mutex);
    for (auto &b : memoryBlocks)
    {
        if (ptr == b.ptr)
        {
            b.available = true;
            break;
        }
    }
}

void BasicMemManager::release()
{
    std::unique_lock<std::mutex> lock(mutex);
    for (auto &b : memoryBlocks)
    {
        if (b.available)
        {
            release(b.ptr, b.type);
        }
    }
    memoryBlocks.erase(std::remove_if(memoryBlocks.begin(),
                                      memoryBlocks.end(),
                                      [](Record &r)
                                      { return r.available; }),
                       memoryBlocks.end());
}

void BasicMemManager::release(MemType type)
{
    std::unique_lock<std::mutex> lock(mutex);
    for (auto &b : memoryBlocks)
    {
        if ((type == b.type) && b.available)
        {
            release(b.ptr, b.type);
        }
    }
    memoryBlocks.erase(std::remove_if(memoryBlocks.begin(),
                                      memoryBlocks.end(),
                                      [type](Record &r)
                                      { return type == r.type && r.available; }),
                       memoryBlocks.end());
}

BasicMemManager::Record *BasicMemManager::find(size_t bytes, MemType type)
{
    Record *res = nullptr;
    const auto maxBytes = static_cast<size_t>(bytes * 1.1f); // if we overflow size_t, too bad ...
    for (auto &b : memoryBlocks)
    {
        if (b.available && (type == b.type))
        {
            if (b.bytes == bytes)
            {
                return &b;
            }
            else if ((b.bytes >= bytes) && (b.bytes <= maxBytes))
            {
                if ((nullptr == res) || (res->bytes > b.bytes))
                {
                    res = &b;
                }
            }
        }
    }
    return res;
}

void *BasicMemManager::alloc(size_t bytes, MemType type) const
{
    void *ptr = nullptr;
    switch (type)
    {
    case MemType::CPU:
        ptr = malloc(bytes);
        break;        
    case MemType::CPU_PAGE_ALIGNED:
        ptr = aligned_alloc(memoryUtils::PAGE_SIZE, bytes);
        #ifdef MADV_HUGEPAGE
            madvise(ptr, bytes, MADV_HUGEPAGE); // Not available in all platforms
        #endif
        break;
    case MemType::CUDA_MANAGED:
        cudaMallocManaged(&ptr, bytes);
        break;
    case MemType::CUDA_HOST:
        cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
        break;
    case MemType::CUDA:
        cudaMalloc(&ptr, bytes);
        break;
    default:
        break;
    }
    return ptr;
}

void BasicMemManager::release(void *ptr, MemType type) const
{
    switch (type)
    {
    case MemType::CPU:
    case MemType::CPU_PAGE_ALIGNED:
        free(ptr);
        break;
    case MemType::CUDA_HOST:
        gpuErrchk(cudaFreeHost(ptr));
        break;
    case MemType::CUDA_MANAGED:
    case MemType::CUDA:
        gpuErrchk(cudaFree(ptr));
        break;
    default:
        break;
    }
}

std::ostream &operator<<(std::ostream &s, const MemType &t)
{
    switch (t)
    {
    case MemType::CPU:
        s << "CPU";
        break;
    case MemType::CPU_PAGE_ALIGNED:
        s << "CPU_PAGE_ALIGNED";
        break;
    case MemType::CUDA_MANAGED:
        s << "CUDA_MANAGED";
        break;
    case MemType::CUDA_HOST:
        s << "CUDA_HOST";
        break;
    case MemType::CUDA:
        s << "CUDA";
        break;
    default:
        s << "UNKNOWN";
    }
    return s;
}

BasicMemManager::~BasicMemManager()
{
    release();
    for (auto &b : memoryBlocks)
    {
        std::cerr << "Unreleased memory block of at " << b.ptr << " of " << b.bytes << " bytes and type " << b.type << "\n";
    }
}
