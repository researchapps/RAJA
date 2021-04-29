#include <iostream>
#include <numeric>
#include <RAJA/RAJA.hpp>
#include <RAJA/util/Timer.hpp>

#include "new_reduce/reduce_basic.hpp"
#include "new_reduce/forall_param.hpp"

void my_sum_fun(int* out, int* in) {
  //printf("%d @ %p += %d @ %p\n", *out, out, *in, in);
  *out = *out + *in;
}

int my_init(int* orig) {
  //printf("orig: %d @ %p\n", *orig, orig);
  return *orig;
}

#pragma omp declare reduction(my_sum : int : my_sum_fun(&omp_out, &omp_in)) initializer(omp_priv = my_init(&omp_orig)) 
//#pragma omp declare reduction(my_sum \
//                              : int \
//                              : my_sum_fun(&omp_out, &omp_in)) \
//                              initializer(omp_priv = my_init(&omp_orig)) \

int main(int argc, char *argv[])
{
  if (argc < 2) {
    std::cout << "Execution Format: ./executable N\n";
    std::cout << "Example of usage: ./new_reduce 5000\n";
    exit(0);
  }
  int N = atoi(argv[1]);

  double r = 0;
  double m = 5000;
  double ma = 0;

  double *a = new double[N]();
  double *b = new double[N]();

  std::iota(a, a + N, 0);
  std::iota(b, b + N, 0);

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  {
    std::cout << "OMP Target Reduction NEW\n";
    #pragma omp target enter data map(to : a[:N], b[:N])

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::omp_target_parallel_for_exec_nt>(N,
                 [=](int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > m_ ? a[i] : m_;
                 },
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m),
                 Reduce<RAJA::operators::maximum>(&ma));
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif

#if defined(RAJA_ENABLE_OPENMP)
  {
    std::cout << "OMP Reduction NEW\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::omp_parallel_for_exec>(N,
                 [=](int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];

                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > m_ ? a[i] : m_;
                 },
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m),
                 Reduce<RAJA::operators::maximum>(&ma));
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif

  {
    std::cout << "Sequential Reduction NEW\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::seq_exec>(N,
                 [=](int i, double &r_, double &m_) {
                 //[=](int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   //ma_ = a[i] > m_ ? a[i] : m_;
                 },
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m));//,
                 //Reduce<RAJA::operators::maximum>(&ma));
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    //std::cout << "ma : " << ma <<"\n";
  }

#if defined(RAJA_ENABLE_TBB)
  {
    std::cout << "Basic TBB Reduction RAJA\n";
    RAJA::ReduceSum<RAJA::tbb_reduce, double> rr(0);
    RAJA::ReduceMin<RAJA::tbb_reduce, double> rm(5000);
    RAJA::ReduceMax<RAJA::tbb_reduce, double> rma(0);

    RAJA::Timer t;
    t.start();
    RAJA::forall<RAJA::tbb_for_exec>(RAJA::RangeSegment(0, N),
                                                        [=](int i) {
                                                          rr += a[i] * b[i];
                                                          rm.min(a[i]);
                                                          rma.max(a[i]);
                                                        });
    t.stop();

    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << rr.get() << "\n";
    std::cout << "m : "  << rm.get()  <<"\n";
    std::cout << "ma : " << rma.get() <<"\n";
  }
#endif

  {
    std::cout << "Basic Seq Reduction RAJA\n";
    RAJA::ReduceSum<RAJA::seq_reduce, double> rr(0);
    RAJA::ReduceMin<RAJA::seq_reduce, double> rm(5000);
    RAJA::ReduceMax<RAJA::seq_reduce, double> rma(0);

    RAJA::Timer t;
    t.start();
    RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, N),
                                                        [=](int i) {
                                                          rr += a[i] * b[i];
                                                          rm.min(a[i]);
                                                          rma.max(a[i]);
                                                        });
    t.stop();

    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << rr.get() << "\n";
    std::cout << "m : "  << rm.get()  <<"\n";
    std::cout << "ma : " << rma.get() <<"\n";
  }

  int s = 0;
  #pragma omp parallel for reduction(my_sum : s)
  for(int i = 0; i < 500; i++) s+=1;
  printf("sum: %d\n", s);

  return 0;
}
