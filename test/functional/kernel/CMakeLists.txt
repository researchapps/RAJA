###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND KERNEL_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND KERNEL_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TBB)
  list(APPEND KERNEL_BACKENDS TBB)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND KERNEL_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND KERNEL_BACKENDS Hip)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND KERNEL_BACKENDS OpenMPTarget)
endif()

add_subdirectory(basic-single-loop)

add_subdirectory(basic-single-icount-loop)

add_subdirectory(nested-loop)

add_subdirectory(nested-loop-segment-types)

add_subdirectory(nested-loop-view-types)

add_subdirectory(reduce-loc)

add_subdirectory(tile)

unset( KERNEL_BACKENDS )

#
# Note: Kernel region tests define their backend list in the region
#       test directory since region constructs are defined for only
#       some of the RAJA back-ends.
#
add_subdirectory(region)

