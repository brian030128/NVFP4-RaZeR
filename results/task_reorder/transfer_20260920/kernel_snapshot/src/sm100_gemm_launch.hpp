#pragma once

#include <cuda.h>
#include "cutlass/device_kernel.h"
#include "gemm_benchmark.hpp"

namespace mixfp4 {
inline bool sm100_uses_pdl() {
#if defined(CUTLASS_ENABLE_GDC_FOR_SM100)
  char const* value = std::getenv("MIXFP4_PDL");
  return !value || value[0] != '0';
#else
  return false;
#endif
}

// Prepared launch for a static-cluster SM100 GEMM. The adapter has already
// initialized Params and opted into dynamic shared memory. Cache the launch
// geometry and set the cluster attribute once, rather than once per invocation.
// A prepared launch belongs to the device on which its adapter was initialized.
template<class Gemm>
class Sm100GemmLaunch {
  using Kernel = typename Gemm::GemmKernel;
  using Cluster = typename Kernel::DispatchPolicy::ClusterShape;
  static_assert(cute::is_static_v<Cluster>, "Prepared launch requires a static cluster");
  typename Gemm::Params params_;
  CUfunction function_{};
  CUlaunchConfig config_{};
  CUlaunchAttribute attributes_[2]{};

 public:
  explicit Sm100GemmLaunch(Gemm const& gemm) : params_(gemm.params()) {
    benchmark_cuda_check(cudaGetFuncBySymbol(&function_,
        reinterpret_cast<void const*>(cutlass::device_kernel<Kernel>)));
    auto grid = Gemm::get_grid_shape(params_);
    auto block = Kernel::get_block_shape();
    config_.gridDimX = grid.x; config_.gridDimY = grid.y; config_.gridDimZ = grid.z;
    config_.blockDimX = block.x; config_.blockDimY = block.y; config_.blockDimZ = block.z;
    config_.sharedMemBytes = Kernel::SharedStorageSize;
    if constexpr (cute::size(Cluster{}) > 1) {
      benchmark_cuda_check(cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
          cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      attributes_[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attributes_[0].value.clusterDim = {unsigned(cute::size<0>(Cluster{})),
                                   unsigned(cute::size<1>(Cluster{})),
                                   unsigned(cute::size<2>(Cluster{}))};
      config_.numAttrs = 1;
    }
    config_.attrs = attributes_;
    // The CUTLASS kernel waits for the preceding grid before accessing inputs,
    // and signals dependents after its final MMA. PDL overlaps launch/preamble
    // work while retaining those data dependencies. Set MIXFP4_PDL=0 to disable.
    if (sm100_uses_pdl()) {
      auto& attribute = attributes_[config_.numAttrs++];
      attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute.value.programmaticStreamSerializationAllowed = 1;
    }
  }

  Sm100GemmLaunch(Sm100GemmLaunch const&) = delete;
  Sm100GemmLaunch& operator=(Sm100GemmLaunch const&) = delete;

  void operator()(cudaStream_t stream = nullptr) {
    config_.hStream = stream;
    void* args[] = {&params_};
    CUresult status = cuLaunchKernelEx(&config_, function_, args, nullptr);
    if (status != CUDA_SUCCESS) {
      char const* message = "unknown driver error";
      cuGetErrorString(status, &message);
      std::fprintf(stderr, "SM100 GEMM launch failed: %s\n", message);
      std::exit(EXIT_FAILURE);
    }
  }
};
}  // namespace mixfp4
