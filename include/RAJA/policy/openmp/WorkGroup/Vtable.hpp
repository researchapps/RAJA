/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA workgroup Vtable.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_openmp_WorkGroup_Vtable_HPP
#define RAJA_openmp_WorkGroup_Vtable_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/policy/loop/WorkGroup/Vtable.hpp"


namespace RAJA
{

namespace detail
{

/*!
* Populate and return a Vtable object
*/
template < typename T, typename Vtable_T >
inline const Vtable_T* get_Vtable(omp_work const&)
{
  return get_Vtable<T, Vtable_T>(loop_work{});
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
