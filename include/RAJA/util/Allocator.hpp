/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing an implementation of a memory pool.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_UTIL_ALLOCATOR_HPP
#define RAJA_UTIL_ALLOCATOR_HPP

#include <cstddef>
#include <string>
#include <unordered_set>

#include "RAJA/util/camp_aliases.hpp"

namespace RAJA
{

/*! \class Allocator
 ******************************************************************************
 *
 * \brief  Allocator Provides a generic interface for allocation in RAJA
 *
 * Allocator& device_pool = RAJA::cuda::get_device_allocator();
 *
 * RAJA::cuda::set_device_allocator<SomeOtherDevicePoolAllocator>(args...);
 *
 ******************************************************************************
 */
struct Allocator
{
  Allocator() = default;

  virtual ~Allocator() = default;

  virtual void* allocate(size_t nbytes) = 0;

  template <typename T>
  inline T* allocate(size_t nitems)
  {
    return static_cast<T*>(this->allocate(sizeof(T)*nitems));
  }

  virtual void deallocate(void* ptr) = 0;

  virtual void release() = 0;

  virtual size_t getHighWatermark() const noexcept = 0;

  virtual size_t getCurrentSize() const noexcept = 0;

  virtual size_t getActualSize() const noexcept = 0;

  virtual size_t getAllocationCount() const noexcept = 0;

  virtual const std::string& getName() const noexcept = 0;

  virtual Platform getPlatform() noexcept = 0;
};

namespace detail
{

extern void add_allocator(Allocator* aloc);
extern void remove_allocator(Allocator* aloc);

} /* end namespace detail */


/*!
 * \brief Get the set of allocators used by RAJA internally
 */
extern std::unordered_set<Allocator*> get_allocators();

} /* end namespace RAJA */


#endif /* RAJA_UTIL_ALLOCATOR_HPP */
