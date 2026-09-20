// A dispatch-policy tag structurally identical to CUTLASS's
// cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled (3rdparty/cutlass/include/cutlass/
// gemm/dispatch_policy.hpp:1070-1083), but a distinct type -- this is what lets
// cutlass::gemm::collective::CollectiveMma<MixedDispatchPolicy, ...> resolve to OUR forked
// mainloop specialization (sm100_blockscaled_mma_mixed.hpp) instead of CUTLASS's vendored one,
// while reusing 100% of CollectiveBuilder's type derivation.
//
// Same trick as the sm_120 side of this repo (collective/dispatch_policy_mixed.hpp); only the
// policy being shadowed differs.
#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

template <
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = cute::Shape<cute::_1, cute::_1, cute::_1>,
  class ArchTag_ = arch::Sm100
>
struct MainloopSm100TmaUmmaWarpSpecializedBlockScaledMixed {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = ArchTag_;
  constexpr static bool IsOverlappingAccum = AccumulatorPipelineStageCount_ == 1;
  using Schedule = KernelTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_,
                                                            AccumulatorPipelineStageCount_>;
};

} // namespace cutlass::gemm
