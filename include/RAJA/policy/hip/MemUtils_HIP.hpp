/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for HIP reductions and other operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MemUtils_HIP_HPP
#define RAJA_MemUtils_HIP_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <unordered_map>

#include "RAJA/util/Allocator.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"

namespace RAJA
{

namespace hip
{

namespace detail
{

extern void set_device_allocator(RAJA::Allocator* allocator);
extern void set_device_zeroed_allocator(RAJA::Allocator* allocator);
extern void set_pinned_allocator(RAJA::Allocator* allocator);

}

// Sets the allocator used by RAJA internally by making an allocator of
// allocator_type with the given arguments. It is an error to change the
// allocator when any memory is allocated. This routine is not thread safe.
template < typename allocator_type, typename ... Args >
void set_device_allocator(Args&&... args)
{
  detail::set_device_allocator(
      new allocator_type(std::forward<Args>(args)...));
}
template < typename allocator_type, typename ... Args >
void set_device_zeroed_allocator(Args&&... args)
{
  detail::set_device_zeroed_allocator(
      new allocator_type(std::forward<Args>(args)...));
}
template < typename allocator_type, typename ... Args >
void set_pinned_allocator(Args&&... args)
{
  detail::set_pinned_allocator(
      new allocator_type(std::forward<Args>(args)...));
}

// Reset the allocator used by RAJA internally. This will destroy any existing
// allocator and replace it with the kind of allocator used by default.
extern void reset_device_allocator();
extern void reset_device_zeroed_allocator();
extern void reset_pinned_allocator();

// Gets the allocator used by RAJA internally. This allows the user to query
// the memory stats of the allocator.
extern RAJA::Allocator& get_device_allocator();
extern RAJA::Allocator& get_device_zeroed_allocator();
extern RAJA::Allocator& get_pinned_allocator();

namespace detail
{

//! struct containing launch parameters
struct LaunchInfo {
  hip_dim_t   gridDim{0, 0, 0};
  hip_dim_t   blockDim{0, 0, 0};
  size_t      shmem  = 0;
  hipStream_t stream = 0;

  LaunchInfo(hip_dim_t gridDim_, hip_dim_t blockDim_,
             size_t shmem_, hipStream_t stream_)
    : gridDim(gridDim_)
    , blockDim(blockDim_)
    , shmem(shmem_)
    , stream(stream_)
  { }
};

//! class that changes a value on construction then resets it at destruction
template <typename T>
class SetterResetter
{
public:
  SetterResetter(T& val, T new_val) : m_val(val), m_old_val(val)
  {
    m_val = new_val;
  }
  SetterResetter(const SetterResetter&) = delete;
  ~SetterResetter() { m_val = m_old_val; }

private:
  T& m_val;
  T m_old_val;
};

}  // namespace detail

//! Ensure all streams in use are synchronized wrt raja kernel launches
extern void synchronize();

//! Ensure stream is synchronized wrt raja kernel launches
extern void synchronize(hipStream_t stream);

//! Indicate stream is asynchronous
extern void launch(hipStream_t stream);

//! Launch kernel and indicate stream is asynchronous
RAJA_INLINE
void launch(const void* func, detail::LaunchInfo const& launch_info, void** args)
{
  hipErrchk(hipLaunchKernel(func, launch_info.gridDim, launch_info.blockDim,
                            args, launch_info.shmem, launch_info.stream));
  launch(launch_info.stream);
}

//! Check for errors
RAJA_INLINE
void peekAtLastError() { hipErrchk(hipPeekAtLastError()); }

//! get launch info for the current thread
extern detail::LaunchInfo*& get_tl_launch_info();

//! create copy of loop_body that is setup for device execution
template <typename LOOP_BODY>
RAJA_INLINE typename std::remove_reference<LOOP_BODY>::type make_launch_body(
    detail::LaunchInfo& launch_info,
    LOOP_BODY&& loop_body)
{
  detail::SetterResetter<detail::LaunchInfo*> setup_reducers_srer(
      get_tl_launch_info(), &launch_info);

  using return_type = typename std::remove_reference<LOOP_BODY>::type;
  return return_type(std::forward<LOOP_BODY>(loop_body));
}

RAJA_INLINE
hipDeviceProp_t get_device_prop()
{
  int device;
  hipErrchk(hipGetDevice(&device));
  hipDeviceProp_t prop;
  hipErrchk(hipGetDeviceProperties(&prop, device));
  return prop;
}

RAJA_INLINE
hipDeviceProp_t& device_prop()
{
  static hipDeviceProp_t prop = get_device_prop();
  return prop;
}

}  // namespace hip

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP

#endif  // closing endif for header file include guard
