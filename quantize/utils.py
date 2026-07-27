import torch
from typing import Tuple, Union


def parse_type_block(type_block: Union[str, Tuple[int, int], list, None]) -> Tuple[int, int]:
    """
        Parse a MixFP4 type-block shape into a (rows, cols) tuple.

        A type block is the granularity at which the FP4 element data type (E2M1 or E0M3) is
        selected. It is specified as "<M>x<K>", e.g. "1x16", "16x16", "256x16", "32x64", "32x128",
        where M spans the outer dimension (output channels for weights, tokens for activations)
        and K spans the reduction dimension. K MUST be a multiple of the NVFP4 scale-block size 16,
        because a type block always contains a whole number of scale blocks.
    """
    if type_block is None:
        return (1, 16)
    if isinstance(type_block, (tuple, list)):
        block_m, block_k = int(type_block[0]), int(type_block[1])
    else:
        tokens = str(type_block).lower().replace("*", "x").replace(",", "x").split("x")
        assert len(tokens) == 2, \
            f'Invalid type-block shape \"{type_block}\". Expected the format \"<M>x<K>\", e.g. \"32x128\".'
        block_m, block_k = int(tokens[0]), int(tokens[1])

    assert block_m > 0 and block_k > 0, \
        f'Invalid type-block shape \"{type_block}\". Both dimensions must be positive.'
    assert block_k % 16 == 0, \
        f'Invalid type-block shape \"{type_block}\". The K dimension must be a multiple of the ' \
        f'NVFP4 scale-block size 16, but got {block_k}.'

    return (block_m, block_k)


def format_type_block(type_block: Union[str, Tuple[int, int], list, None]) -> str:
    """
        Canonical string form of a type-block shape, used for result file names.
    """
    block_m, block_k = parse_type_block(type_block)
    return f"{block_m}x{block_k}"


def quant_scale(scale_fp, exp_bits, man_bits, exp_min=None):
    if exp_min is None:
        exp_min = -2**(exp_bits-1) + 2
    scale_sign = scale_fp.sign()
    assert (scale_sign == -1).any().logical_not(), "The scaling factor CANNOT be negative. Something is WRONG..."
    scale_exp  = (
        scale_fp + (scale_fp == 0).type(scale_fp.dtype)
    ).log2().floor().clamp_(min=exp_min)
    scale_man  = torch.round(
        scale_fp / 2**scale_exp * 2**man_bits
    ) / (2**man_bits)
    scale_dq   = scale_sign * 2**scale_exp * scale_man 

    return scale_dq

