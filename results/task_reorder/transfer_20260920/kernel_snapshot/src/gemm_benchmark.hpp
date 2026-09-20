#pragma once

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// Use the same timing path for CUTLASS and cuBLAS. Graph replay is optional:
// MIXFP4_BENCH_GRAPH=1 measures device throughput without CPU launch gaps.
// Allocation, descriptor setup, graph construction and warmup are never timed.
namespace mixfp4 {
inline void benchmark_cuda_check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "Benchmark CUDA error: %s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

inline bool benchmark_uses_graph() {
  char const* value = std::getenv("MIXFP4_BENCH_GRAPH");
  return value && value[0] == '1';
}

template<class Launch>
double benchmark_ms(Launch launch, int warmup, int iterations) {
  if (iterations <= 0 || warmup < 0) {
    std::fprintf(stderr, "Benchmark iteration counts must be positive.\n");
    std::exit(EXIT_FAILURE);
  }
  cudaStream_t stream;
  cudaEvent_t start, stop;
  benchmark_cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  benchmark_cuda_check(cudaEventCreate(&start));
  benchmark_cuda_check(cudaEventCreate(&stop));
  // Ten microsecond-scale launches do not give an idle GPU time to raise its
  // clocks. Apply the same minimum warmup time to CUTLASS and every cuBLAS candidate.
  auto warmup_start = std::chrono::steady_clock::now();
  do {
    for (int i = 0; i < warmup; ++i) { launch(stream); }
    benchmark_cuda_check(cudaStreamSynchronize(stream));
  } while (std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - warmup_start).count() < 50.0);

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (benchmark_uses_graph()) {
    benchmark_cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iterations; ++i) { launch(stream); }
    benchmark_cuda_check(cudaStreamEndCapture(stream, &graph));
    benchmark_cuda_check(cudaGraphInstantiate(&executable, graph, 0));
    benchmark_cuda_check(cudaGraphLaunch(executable, stream));
    benchmark_cuda_check(cudaStreamSynchronize(stream));
  }

  benchmark_cuda_check(cudaEventRecord(start, stream));
  if (executable) {
    benchmark_cuda_check(cudaGraphLaunch(executable, stream));
  } else {
    for (int i = 0; i < iterations; ++i) { launch(stream); }
  }
  benchmark_cuda_check(cudaEventRecord(stop, stream));
  benchmark_cuda_check(cudaEventSynchronize(stop));
  float elapsed;
  benchmark_cuda_check(cudaEventElapsedTime(&elapsed, start, stop));
  if (executable) { benchmark_cuda_check(cudaGraphExecDestroy(executable)); }
  if (graph) { benchmark_cuda_check(cudaGraphDestroy(graph)); }
  benchmark_cuda_check(cudaEventDestroy(start));
  benchmark_cuda_check(cudaEventDestroy(stop));
  benchmark_cuda_check(cudaStreamDestroy(stream));
  return double(elapsed) / iterations;
}
}  // namespace mixfp4
