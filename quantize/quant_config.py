from typing import Optional

from .utils import parse_type_block


class QuantConfig(dict):
    def __init__(
        self,
        # general quantization parameters
        w_bits: int=16,
        w_dtype: str="fp16",
        w_outlier: float=8.0,
        w_type_block: str="1x16",  # MixFP4 type-block shape "<M>x<K>" for weights
        a_type_block: str="1x16",  # MixFP4 type-block shape "<M>x<K>" for activations
        a_bits: int=16,
        a_dtype: str="fp16",
        k_bits: int=16,
        v_bits: int=16,
        w_groupsize: int=-1,
        a_groupsize: int=-1,
        k_groupsize: int=-1,
        v_groupsize: int=-1,
        kv_quant: bool=False,  # If True, then quantize KV-cache
    ):
        for nbits in [w_bits, k_bits, v_bits]:
            assert (nbits is None) or (nbits in [3, 4, 5, 6, 8, 16]), \
                f'Invalid precision \"{nbits}\" provided for weight / KV-cache. Allowed precisions are {{3, 4, 6, 8, 16}}'
        for nbits in [a_bits]:
            assert (nbits is None) or (nbits in [4, 8, 16]), \
                f'Invalid precision \"{nbits}\" provided for activation / query. Allowed precisions are {{8, 16}}'

        # MixFP4 type-block shapes are validated eagerly so that a bad sweep argument fails fast
        for dtype, type_block in [(w_dtype, w_type_block), (a_dtype, a_type_block)]:
            if isinstance(dtype, str) and dtype.lower() in ("mixfp4", "mix_4_6"):
                parse_type_block(type_block)

        self.w_bits = w_bits
        self.w_dtype = w_dtype
        self.w_outlier = w_outlier
        self.w_type_block = w_type_block
        self.a_type_block = a_type_block
        self.a_bits = a_bits
        self.a_dtype = a_dtype
        self.k_bits = k_bits
        self.v_bits = v_bits

        self.w_groupsize = w_groupsize
        self.a_groupsize = a_groupsize
        self.k_groupsize = k_groupsize
        self.v_groupsize = v_groupsize

        self.kv_quant = kv_quant

        # Fixed activation channel permutations, keyed by the tensor's last-dim size.
        #
        # The activation tag grid -- unlike the weight one -- carries real per-channel structure
        # (col_share 0.169 against 0.004 for weights), because E0M3/E2M1 is decided by block
        # peakedness and activation outliers sit in FIXED channels, so the same channels make their
        # blocks peaked for every token. Reordering the channel axis therefore groups genuinely
        # like with like: +0.081 of the 1x16 ceiling over a cell-shuffle control at 16x64, against
        # +0.003 on weights.
        #
        # The permutation must be FIXED and calibration-derived, never computed from the tensor in
        # flight: a per-batch permutation is not a deployable transform and would re-run the search
        # on every forward pass. It moves whole 16-channel chunks, so the scale blocks -- and hence
        # the WEIGHT quantization of the same axis -- are untouched; permuting only the activation
        # here is therefore exactly equivalent to permuting both GEMM operands.
        #
        # Keying by channel count is a simulation shortcut: `quant_act` is handed a bare tensor with
        # no module identity, and the three activation axes in a Llama block (residual, FF
        # intermediate, head dim) have distinct widths. A real implementation would attach the
        # permutation to the module.
        self.a_perm = None

    def __repr__(self):
        return repr(self.__dict__)
