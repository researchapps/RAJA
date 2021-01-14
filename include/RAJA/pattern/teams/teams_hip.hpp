/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::hip
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_hip_HPP
#define RAJA_pattern_teams_hip_HPP

#include "RAJA/pattern/teams/teams_core.hpp"
#include "RAJA/policy/hip/policy.hpp"

namespace RAJA
{

namespace expt
{

template <bool async, int num_threads = 0>
struct hip_launch_t {
};

template <typename BODY>
__global__ void launch_global_fcn(LaunchContext ctx, BODY body)
{
  body(ctx);
}


template <bool async>
struct LaunchExecute<RAJA::expt::hip_launch_t<async, 0>> {
  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
    dim3 blocks;
    dim3 threads;

    blocks.x = ctx.teams.value[0];
    blocks.y = ctx.teams.value[1];
    blocks.z = ctx.teams.value[2];

    threads.x = ctx.threads.value[0];
    threads.y = ctx.threads.value[1];
    threads.z = ctx.threads.value[2];
    launch_global_fcn<<<blocks, threads>>>(ctx, body);

    if (!async) {
      hipDeviceSynchronize();
    }
  }
};


template <typename BODY, int num_threads>
__launch_bounds__(num_threads, 1) __global__
    void launch_global_fcn_fixed(LaunchContext ctx, BODY body)
{
  body(ctx);
}


template <bool async, int nthreads>
struct LaunchExecute<RAJA::expt::hip_launch_t<async, nthreads>> {
  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
    dim3 blocks;
    dim3 threads;

    blocks.x = ctx.teams.value[0];
    blocks.y = ctx.teams.value[1];
    blocks.z = ctx.teams.value[2];

    threads.x = ctx.threads.value[0];
    threads.y = ctx.threads.value[1];
    threads.z = ctx.threads.value[2];
    launch_global_fcn_fixed<nthreads><<<blocks, threads>>>(ctx, body);

    if (!async) {
      hipDeviceSynchronize();
    }
  }
};

/*
  HIP thread loops with block strides
*/

template <typename SEGMENT, int DIM>
struct LoopExecute<hip_thread_xyz_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_hip_dim<DIM>(threadIdx);
         tx < len;
         tx += internal::get_hip_dim<DIM>(blockDim) )
    {
      body(*(segment.begin() + tx));
    }
  }
};



/*
  HIP thread direct mappings
*/

template <typename SEGMENT, int DIM>
struct LoopExecute<hip_thread_xyz_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_hip_dim<DIM>(threadIdx);
      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};


/*
  HIP block loops with grid strides
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<hip_block_xyz_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = internal::get_hip_dim<DIM>(blockIdx);
         bx < len;
         bx += internal::get_hip_dim<DIM>(gridDim) ) {
      body(*(segment.begin() + bx));
    }
  }
};



/*
  HIP block direct mappings
*/

template <typename SEGMENT, int DIM>
struct LoopExecute<hip_block_xyz_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = internal::get_hip_dim<DIM>(blockIdx);
      if (bx < len) body(*(segment.begin() + bx));
    }
  }
};




// perfectly nested hip direct policies
using hip_block_xy_nested_direct = hip_block_xyz_direct<0,1>;
using hip_block_xz_nested_direct = hip_block_xyz_direct<0,2>;
using hip_block_yx_nested_direct = hip_block_xyz_direct<1,0>;
using hip_block_yz_nested_direct = hip_block_xyz_direct<1,2>;
using hip_block_zx_nested_direct = hip_block_xyz_direct<2,0>;
using hip_block_zy_nested_direct = hip_block_xyz_direct<2,1>;

using hip_block_xyz_nested_direct = hip_block_xyz_direct<0,1,2>;
using hip_block_xzy_nested_direct = hip_block_xyz_direct<0,2,1>;
using hip_block_yxz_nested_direct = hip_block_xyz_direct<1,0,2>;
using hip_block_yzx_nested_direct = hip_block_xyz_direct<1,2,0>;
using hip_block_zxy_nested_direct = hip_block_xyz_direct<2,0,1>;
using hip_block_zyx_nested_direct = hip_block_xyz_direct<2,1,0>;




template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<hip_block_xyz_direct<DIM0, DIM1>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = internal::get_hip_dim<DIM0>(blockIdx);
      const int ty = internal::get_hip_dim<DIM1>(blockIdx);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};



template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<hip_block_xyz_direct<DIM0, DIM1, DIM2>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = internal::get_hip_dim<DIM0>(blockIdx);
      const int ty = internal::get_hip_dim<DIM1>(blockIdx);
      const int tz = internal::get_hip_dim<DIM2>(blockIdx);
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx),
             *(segment1.begin() + ty),
             *(segment2.begin() + tz));
    }
  }
};


// perfectly nested hip loop policies
using hip_block_xy_nested_loop = hip_block_xyz_loop<0,1>;
using hip_block_xz_nested_loop = hip_block_xyz_loop<0,2>;
using hip_block_yx_nested_loop = hip_block_xyz_loop<1,0>;
using hip_block_yz_nested_loop = hip_block_xyz_loop<1,2>;
using hip_block_zx_nested_loop = hip_block_xyz_loop<2,0>;
using hip_block_zy_nested_loop = hip_block_xyz_loop<2,1>;

using hip_block_xyz_nested_loop = hip_block_xyz_loop<0,1,2>;
using hip_block_xzy_nested_loop = hip_block_xyz_loop<0,2,1>;
using hip_block_yxz_nested_loop = hip_block_xyz_loop<1,0,2>;
using hip_block_yzx_nested_loop = hip_block_xyz_loop<1,2,0>;
using hip_block_zxy_nested_loop = hip_block_xyz_loop<2,0,1>;
using hip_block_zyx_nested_loop = hip_block_xyz_loop<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<hip_block_xyz_loop<DIM0, DIM1>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {

      for (int bx = internal::get_hip_dim<DIM0>(blockIdx);
           bx < len0;
           bx += internal::get_hip_dim<DIM0>(gridDim))
      {
        for (int by = internal::get_hip_dim<DIM1>(blockIdx);
             by < len1;
             by += internal::get_hip_dim<DIM1>(gridDim))
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by));
        }
      }
    }
  }
};



template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<hip_block_xyz_loop<DIM0, DIM1, DIM2>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int bx = internal::get_hip_dim<DIM0>(blockIdx);
         bx < len0;
         bx += internal::get_hip_dim<DIM0>(gridDim))
    {

      for (int by = internal::get_hip_dim<DIM1>(blockIdx);
           by < len1;
           by += internal::get_hip_dim<DIM1>(gridDim))
      {

        for (int bz = internal::get_hip_dim<DIM2>(blockIdx);
             bz < len2;
             bz += internal::get_hip_dim<DIM2>(gridDim))
        {

          body(*(segment0.begin() + bx),
               *(segment1.begin() + by),
               *(segment2.begin() + bz));
        }
      }
    }
  }
};





template <typename SEGMENT, int DIM>
struct TileExecute<hip_thread_xyz_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_hip_dim<DIM>(threadIdx) * tile_size;
         tx < len;
         tx += internal::get_hip_dim<DIM>(blockDim) * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};

template <typename SEGMENT, int DIM>
struct TileExecute<hip_thread_xyz_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_hip_dim<DIM>(threadIdx) * tile_size;
    if(tx < len)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<hip_block_xyz_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_hip_dim<DIM>(blockIdx) * tile_size;

         tx < len;

         tx += internal::get_hip_dim<DIM>(gridDim) * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<hip_block_xyz_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_hip_dim<DIM>(blockIdx) * tile_size;
    if(tx < len){
      body(segment.slice(tx, tile_size));
    }
  }
};

}  // namespace expt

}  // namespace RAJA
#endif
