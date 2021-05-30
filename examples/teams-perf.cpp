//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "camp/resource.hpp"
#include "memoryManager.hpp"
#include <chrono>

using namespace std::chrono;

/*
 * Define host/device launch policies
 */
using launch_policy = RAJA::expt::LaunchPolicy<
    RAJA::expt::seq_launch_t
    ,
    RAJA::expt::hip_launch_t<true>
    >;

using loop_policy = RAJA::loop_exec;

#if defined(RAJA_ENABLE_HIP)
using gpu_thread_x_policy = RAJA::hip_thread_x_loop;
using gpu_thread_y_policy = RAJA::hip_thread_y_loop;
using gpu_thread_z_policy = RAJA::hip_thread_z_loop;
using gpu_block_x_policy = RAJA::hip_block_x_direct;
#endif

/*
  Define RAJA Team/Thread policies, if a device is available add
  a device policy.
*/
using teams_x = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_block_x_policy
#endif
                                       >;


using threads_x = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_thread_x_policy
#endif
                                       >;

using threads_y = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_thread_y_policy
#endif
                                       >;

using threads_z = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_thread_z_policy
#endif
                                       >;

#define dofs1D 8
#define qpts1D 10

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{


  const int Niter = 80000;
  const int N_mats = 1000;
  double *A_ptr = memoryManager::allocate<double>(dofs1D * dofs1D * dofs1D * N_mats);

  double *B_ptr = memoryManager::allocate<double>(dofs1D * dofs1D * qpts1D * N_mats);
  double *C_ptr = memoryManager::allocate<double>(dofs1D * qpts1D);

  RAJA::View<double, RAJA::Layout<4>> Aview(A_ptr, N_mats, dofs1D, dofs1D, dofs1D);
  RAJA::View<double, RAJA::Layout<4>> Bview(B_ptr, N_mats, dofs1D, dofs1D, qpts1D);
  RAJA::View<double, RAJA::Layout<2>> Cview(C_ptr, qpts1D, dofs1D);

  RAJA::forall<RAJA::hip_exec<256>>(RAJA::RangeSegment(0, qpts1D), [=] RAJA_DEVICE (int i) {
      for(int j=0; j<dofs1D; ++j) {
        Cview(i, j) = 0.5;
      }
  });

  RAJA::forall<RAJA::hip_exec<256>>(RAJA::RangeSegment(0, N_mats), [=] RAJA_DEVICE (int i) {
      for(int r=0; r<dofs1D; ++r){
        for(int c=0; c<dofs1D; ++c){
          for(int k=0; k<dofs1D; ++k){
            Aview(i, r, c, k) = 1.0;
          }
        }
      }
  });



  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  for(int i=0; i<Niter; ++i){

    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
      RAJA::expt::Resources(RAJA::expt::Teams(N_mats),
                            RAJA::expt::Threads(dofs1D, dofs1D, dofs1D)),

       [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

       RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, N_mats), [&](int e) {

           RAJA_TEAM_SHARED double Cmem[qpts1D][dofs1D];
           RAJA_TEAM_SHARED double X[dofs1D][dofs1D][dofs1D];

           //Load Cmem
           RAJA::expt::loop<threads_z>(ctx, RAJA::RangeSegment(0, 1), [&](int dz) {
               RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, qpts1D), [&](int q) {
                   RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, dofs1D), [&](int dx) {
                       Cmem[q][dx] = Cview(q,dx);
                     });
                 });
             });


           RAJA::expt::loop<threads_z>(ctx, RAJA::RangeSegment(0, dofs1D), [&](int dz) {
               RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, dofs1D), [&](int dy) {
                   RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, dofs1D), [&](int dx) {
                       X[dz][dy][dx] = Aview(e, dz, dy, dx);
                     });
                 });
             });


           ctx.teamSync();

         RAJA::expt::loop<threads_z>(ctx, RAJA::RangeSegment(0, qpts1D), [&](int q) {
           RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, dofs1D), [&](int dz) {
               RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, dofs1D), [&](int dy) {

                   double dot = 0.0;
                   for(int d = 0; d<dofs1D; ++d) {
                     dot += Cmem[q][d] * X[d][dy][dz];
                   }
                   Bview(e, dz, dy, q) = dot;
                 });
             });
           });


         });//team_x

     });  // outer lambda
       
  }//iter loop

  hipDeviceSynchronize();
                                      
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  
  std::cout << "It took me " << time_span.count() << " seconds.";
  std::cout << std::endl;
  
  RAJA::ReduceSum<RAJA::hip_reduce, double> dot(0.0);
  
  RAJA::forall<RAJA::hip_exec<256>>
    (RAJA::RangeSegment(0, N_mats*dofs1D*dofs1D*dofs1D), [=] RAJA_DEVICE (int i) {
      
      dot += B_ptr[i];
      
    });
  
                                      
  std::cout<<"dot.Get() = "<<dot.get()<<std::endl;
  

}  // Main
