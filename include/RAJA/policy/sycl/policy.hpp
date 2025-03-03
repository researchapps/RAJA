/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA SYCL policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_sycl_HPP
#define policy_sycl_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <CL/sycl.hpp>

#include "RAJA/policy/PolicyBase.hpp"

#include <cstddef>

namespace RAJA
{

struct uint3 {
  unsigned long x, y, z;
};

using sycl_dim_t = cl::sycl::range<1>;

using sycl_dim_3_t = uint3;

namespace detail
{
template <bool Async>
struct get_launch {
  static constexpr RAJA::Launch value = RAJA::Launch::async;
};

template <>
struct get_launch<false> {
  static constexpr RAJA::Launch value = RAJA::Launch::sync;
};
}  // end namespace detail

namespace policy
{
namespace sycl
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////

template <size_t BLOCK_SIZE, bool Async = true>
struct sycl_exec : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::sycl,
                       RAJA::Pattern::forall,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::sycl> {
};

template <size_t BLOCK_SIZE, bool Async = false>
struct sycl_exec_nontrivial : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::sycl,
                       RAJA::Pattern::forall,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::sycl> {
};


struct sycl_reduce
    : make_policy_pattern_t<RAJA::Policy::sycl, RAJA::Pattern::reduce> {
};

}  // namespace sycl
}  // namespace policy

using policy::sycl::sycl_exec;
using policy::sycl::sycl_exec_nontrivial;
using policy::sycl::sycl_reduce;

/*!
 * Maps indices to SYCL global id
 * Optional WORK_GROUP_SIZE to 
 */ 
template<int dim, int WORK_GROUP_SIZE = 1>
struct sycl_global_123{};

template<int WORK_GROUP_SIZE>
using sycl_global_0 = sycl_global_123<0, WORK_GROUP_SIZE>;
template<int WORK_GROUP_SIZE>
using sycl_global_1 = sycl_global_123<1, WORK_GROUP_SIZE>;
template<int WORK_GROUP_SIZE>
using sycl_global_2 = sycl_global_123<2, WORK_GROUP_SIZE>;

/*!
 * Maps segment indices to SYCL group ids.
 * Loops to allow for any value
 */
template<int dim>
struct sycl_group_123_loop{};

using sycl_group_0_loop = sycl_group_123_loop<0>;
using sycl_group_1_loop = sycl_group_123_loop<1>;
using sycl_group_2_loop = sycl_group_123_loop<2>;

/*!
 * Maps segment indices to SYCL local ids.
 * Loops to allow for any value
 */
template<int dim>
struct sycl_local_123_loop{};

using sycl_local_0_loop = sycl_local_123_loop<0>;
using sycl_local_1_loop = sycl_local_123_loop<1>;
using sycl_local_2_loop = sycl_local_123_loop<2>;

/*!
 * Maps segment indices to SYCL group ids.
 */
template<int dim>
struct sycl_group_123_direct{};

using sycl_group_0_direct = sycl_group_123_direct<0>;
using sycl_group_1_direct = sycl_group_123_direct<1>;
using sycl_group_2_direct = sycl_group_123_direct<2>;

/*!
 * Maps segment indices to SYCL local ids.
 */
template<int dim>
struct sycl_local_123_direct{};

using sycl_local_0_direct = sycl_local_123_direct<0>;
using sycl_local_1_direct = sycl_local_123_direct<1>;
using sycl_local_2_direct = sycl_local_123_direct<2>;


namespace internal{

template<int dim>
struct SyclDimHelper;

template<>
struct SyclDimHelper<0>{

  template<typename dim_t>
  inline
  static
  constexpr
  auto get(dim_t const &d) ->
    decltype(d.x)
  {
    return d.x;
  }

  template<typename dim_t>
  inline
  static
  void set(dim_t &d, int value)
  {
    d.x = value;
  }
};

template<>
struct SyclDimHelper<1>{

  template<typename dim_t>
  inline
  static
  constexpr
  auto get(dim_t const &d) ->
    decltype(d.x)
  {
    return d.y;
  }

  template<typename dim_t>
  inline
  static
  void set(dim_t &d, int value)
  {
    d.y = value;
  }
};

template<>
struct SyclDimHelper<2>{

  template<typename dim_t>
  inline
  static
  constexpr
  auto get(dim_t const &d) ->
    decltype(d.x)
  {
    return d.z;
  }

  template<typename dim_t>
  inline
  static
  void set(dim_t &d, int value)
  {
    d.z = value;
  }
};

template<int dim, typename dim_t>
constexpr
auto get_sycl_dim(dim_t const &d) ->
  decltype(d.x)
{
  return SyclDimHelper<dim>::get(d);
}

template<int dim, typename dim_t>
void set_sycl_dim(dim_t &d, int value)
{
  return SyclDimHelper<dim>::set(d, value);
}
} // namespace internal

}  // namespace RAJA

#endif // RAJA_ENABLE_SYCL

#endif
