/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Visitor tree store operations for the sm90 TMA warp-specialized (ws) epilogue
*/


#pragma once
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
namespace cutlass::epilogue::fusion {
template <class Element, class LayoutOrStrideMNL>
struct Sm100PermutationStore {
  static constexpr auto RoundStyle = FloatRoundStyle::round_to_nearest;
  static constexpr bool EnableNullptr = false;
  using ElementAux = Element;
  using StrideMNL = cutlass::gemm::TagToStrideC_t<LayoutOrStrideMNL>;

  struct SharedStorage { };

  struct Arguments {
    Element* ptr_aux = nullptr;
    StrideMNL dAux = {};
    int const* permutation = nullptr;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm100PermutationStore() { }

  CUTLASS_HOST_DEVICE
  Sm100PermutationStore(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<
    class GTensorR2G,
    class RTensor,
    class CTensorR2G,
    class ProblemShapeMNL
  >
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        GTensorR2G&& tC_gAux,
        RTensor&& tC_rAux,
        CTensorR2G&& tC_cAux,
        ProblemShapeMNL problem_shape_mnl,
        Params const* params_ptr)
      : tC_gAux(cute::forward<GTensorR2G>(tC_gAux)),
        tC_rAux(cute::forward<RTensor>(tC_rAux)),
        tC_cAux(cute::forward<CTensorR2G>(tC_cAux)),
        problem_shape_mnl(problem_shape_mnl),
        params_ptr(params_ptr) {}

    GTensorR2G tC_gAux;
    RTensor tC_rAux;
    CTensorR2G tC_cAux;
    ProblemShapeMNL problem_shape_mnl;
    Params const* params_ptr;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      using ConvertInput = NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux));
      tC_rAux_frg(epi_v) = convert_input(frg_input);

      return convert_input(frg_input);
    }

    CUTLASS_DEVICE void
    end_loop(int epi_m, int epi_n) {
      if constexpr (EnableNullptr) {
        if (params_ptr->ptr_aux == nullptr) {
          return;
        }
      }

      auto coords = coalesce(tC_cAux(_,_,_,epi_m,epi_n));
      auto values = coalesce(tC_rAux);
      CUTLASS_PRAGMA_UNROLL
      for (int i=0;i<size(values);++i) {
        auto c=coords(i);
        if (elem_less(c,problem_shape_mnl)) {
          int n=int(get<1>(c));
          int destination=params_ptr->permutation ? params_ptr->permutation[n] : n;
          params_ptr->ptr_aux[int64_t(get<0>(c))*get<0>(params_ptr->dAux)+destination+
                              int64_t(get<2>(c))*get<2>(params_ptr->dAux)]=values(i);
        }
      }

    }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    auto problem_shape_mnl = make_shape(M,N,L);

    // Gmem Tensor
    Tensor mAux = make_tensor(
      make_gmem_ptr(params_ptr->ptr_aux), make_shape(M,N,L), params_ptr->dAux
    );
    Tensor tC_gAux = sm90_partition_for_epilogue<ReferenceSrc>(
                      mAux, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);

    // Register Tensor
    Tensor tC_rAux = make_tensor<Element>(take<0,3>(shape(tC_gAux)));

    // Predication support
    Tensor coordAux = make_identity_tensor(shape(mAux));
    Tensor tC_cAux = sm90_partition_for_epilogue<ReferenceSrc>(
                      coordAux, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);

    return ConsumerStoreCallbacks<decltype(tC_gAux), decltype(tC_rAux), decltype(tC_cAux), decltype(problem_shape_mnl)>(
      cute::move(tC_gAux),
      cute::move(tC_rAux),
      cute::move(tC_cAux),
      problem_shape_mnl,
      params_ptr
    );

  }

};

}
