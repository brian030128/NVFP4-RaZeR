import torch
from typing import Optional
from functools import partial
from .quant_config import QuantConfig
from .utils import quant_scale, parse_type_block



@torch.no_grad()
def quant_nf4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        4-bit Normal-Float quantization.
    """

    quant_value  = sorted(
        [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, 
        -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
    )
    mid_value    = [
        (quant_value[i] + quant_value[i + 1]) / 2 for i in range(len(quant_value) - 1)
    ]

    orig_shape   = w_fp.shape 
    w_fp_new     = w_fp.reshape(-1, groupsize).to(torch.float32)

    rmax         = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True)
    qmax         = max([abs(x) for x in quant_value])
    scale_fp     = (rmax / qmax).clamp_(min=1e-7)
    w_scaled     = w_fp_new / scale_fp

    w_q = torch.zeros_like(w_scaled)
    for i, data in enumerate(quant_value):
        if i == 0:
            w_q += torch.where(
                w_scaled <= mid_value[i], 
                data, 0
            )
        elif i == len(quant_value) - 1:
            w_q += torch.where(
                w_scaled > mid_value[i - 1], 
                data, 0
            )
        else:
            w_q += torch.where(
                (mid_value[i - 1] < w_scaled) & (w_scaled <= mid_value[i]), 
                data, 0
            )

    w_dq = w_q * scale_fp 

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_hf4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        HiFloat4 quantization. Following the implementation https://arxiv.org/pdf/2602.11287
    """
    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, 4)

    # Block Maximum
    max_4  = w_fp_new.abs().amax(dim=-1, keepdim=True)
    max_8  = max_4.reshape(-1, 2).amax(dim=-1, keepdim=True).repeat(1, 2).view(-1, 1)
    max_64 = max_4.reshape(-1, 16).amax(dim=-1, keepdim=True).repeat(1, 16).view(-1, 1)

    # Block Scale
    scale_64 = max_64 / 7
    scale_64_max, scale_64_min = 2**15 * 1.5, 2**(-48)
    scale_64 = quant_scale(
        scale_64.clamp(min=scale_64_min, max=scale_64_max),
        exp_bits=6, man_bits=2, exp_min=-48
    )
    scale_8  = torch.where(
        max_8 / scale_64 >= 4, 
        2, 1
    )
    scale_4  = torch.where(
        max_64 / (scale_64 * scale_8) >= 2, 
        2, 1
    )

    # Block Quantized Element
    w_scaled = w_fp_new / (scale_64 * scale_8 * scale_4)
    w_q      = (w_scaled * 4).clamp(min=-7, max=7).round() / 4
    w_dq     = w_q * (scale_64 * scale_8 * scale_4)

    return w_dq.view(orig_shape)


@torch.no_grad()
def quant_mxfp4_naive(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        Original NAIVE MXFP4 quantization. Reference: https://github.com/microsoft/microxcaling
    """
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

    FP4_EXP_BITS = 2
    FP4_MAN_BITS = 1
    FP4_EMAX     = 2
    FP4_MAX      = 6.0

    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, groupsize).to(torch.float32)
    
    block_exp  = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True)
    block_exp  = torch.floor(
        torch.log2(
            block_exp + FP32_MIN_NORMAL * (block_exp == 0).type(block_exp.dtype)
        )
    )

    # Offset the max exponent by the largest representable exponent
    # in the element data format
    block_exp  = (block_exp - FP4_EMAX).clamp(min=-FP32_EXPONENT_BIAS+1, max=FP32_EXPONENT_BIAS)
    w_s        = w_fp_new / (2**block_exp)

    # FP4 Quantization
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s) + (w_s == 0).type(w_s.dtype)
        )
    ).clamp(min=0)
    w_m         = w_s / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m         = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q         = w_m * (2**private_exp) / (2**FP4_MAN_BITS)
    w_q         = torch.clamp(w_q, min=-FP4_MAX, max=FP4_MAX)

    # De-Quantization
    w_dq        = w_q * (2**block_exp)

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_mxfp4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        Better MXFP4 quantization.
    """
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    FP4_MAN_BITS = 1
    FP4_MAX      = 6.0

    orig_shape = w_fp.shape 
    w_fp_new = w_fp.reshape(-1, groupsize).to(torch.float32)
    
    scale_fp32 = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True) / FP4_MAX
    block_exp  = torch.ceil(
        torch.log2(
            scale_fp32 + FP32_MIN_NORMAL * (scale_fp32 == 0).type(scale_fp32.dtype)
        )
    ).clamp(min=-FP32_EXPONENT_BIAS+1, max=FP32_EXPONENT_BIAS)
    w_s        = w_fp_new / (2**block_exp) 
    
    # FP4 Quantization
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s) + (w_s == 0).type(w_s.dtype)
        )
    ).clamp(min=0)
    w_m = w_s / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q = w_m * (2**private_exp) / (2**FP4_MAN_BITS)

    # De-Quantization
    w_dq = w_q * (2**block_exp)

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_mxfp4_meta(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        Meta's MXFP4 quantization recipe: https://arxiv.org/abs/2603.08713
    """
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    FP4_MAN_BITS = 1
    FP4_MAX      = 6.0

    # Get the per-1x128-tile maximum and E0M8 scale factor
    orig_shape  = w_fp.shape 
    w_fp_new    = w_fp.reshape(-1, 128).to(torch.float32)
    tile_max    = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True)
    tile_scale  = (
        (tile_max / 1.5).view(torch.int32).bitwise_and(0x007f8000)
    )
    tile_scale  = tile_scale.bitwise_or(0x3f800000).view(torch.float32) 
    w_fp_new    = (w_fp_new / tile_scale).view(-1, 16)

    # Get the per-1x16-block maximum and E8M0 scale factor
    block_max   = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True)
    scale_fp32  = block_max / FP4_MAX
    block_exp   = torch.ceil(
        torch.log2(
            scale_fp32 + FP32_MIN_NORMAL * (scale_fp32 == 0).type(scale_fp32.dtype)
        )
    ).clamp(min=-FP32_EXPONENT_BIAS+1, max=FP32_EXPONENT_BIAS)

    # Overflow-Aware Scaling (OAS): If the scaled block maximum is smaller than 3.5, then upscale the whole block by 2
    block_max_s = block_max / 2**block_exp
    upscale     = block_max_s.lt(3.5).squeeze()
    block_exp[upscale] -= 1
    block_scale = 2**block_exp

    # FP4 Quantization
    w_s         = w_fp_new / block_scale
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s) + (w_s == 0).type(w_s.dtype)
        )
    ).clamp(min=0)
    w_m = w_s / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q = w_m * (2**private_exp) / (2**FP4_MAN_BITS)

    # Dequantization
    w_dq = (w_q * block_scale).view(-1, 128) * tile_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_mxif4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        MXFP4 and MXINT4 quantization, where a block is quantized to FP4 / INT4 with lower quantization error.
    """
    # FP4 quantization values
    quant_value_fp4 = sorted([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mid_value_fp4   = [
        (quant_value_fp4[i] + quant_value_fp4[i + 1]) / 2 for i in range(len(quant_value_fp4) - 1)
    ]
  
    SCALE_EMAX      = 63
    FP32_MIN_NORMAL = 2 ** (-126)
    FP4_MAN_BITS    = 1
    FP4_MAX         = 6.0
    INT4_MAX        = 7.0

    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, groupsize).to(torch.float32)
    block_max  = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True)

    ########## MXFP4 quantization ##########
    block_scale_fp4     = block_max / FP4_MAX
    block_exp_fp4       = torch.ceil(
        torch.log2(
            block_scale_fp4 + FP32_MIN_NORMAL * (block_scale_fp4 == 0).type(block_max.dtype)
        )
    ).clamp(min=-SCALE_EMAX, max=SCALE_EMAX)
    block_scale_q_fp4   = 2**block_exp_fp4
    w_s_fp4             = w_fp_new / block_scale_q_fp4

    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s_fp4) + (w_s_fp4 == 0).type(w_s_fp4.dtype)
        )
    )
    private_exp = private_exp.clamp(min=0)
    w_m_fp4     = w_s_fp4 / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m_fp4     = torch.sign(w_m_fp4) * torch.floor(torch.abs(w_m_fp4) + 0.5)
    w_q_fp4     = w_m_fp4 * (2**private_exp) / (2**FP4_MAN_BITS)

    ########## MXINT4 quantization ##########
    block_scale_int4    = block_max / INT4_MAX
    block_exp_int4      = torch.ceil(
        torch.log2(
            block_scale_int4 + FP32_MIN_NORMAL * (block_scale_int4 == 0).type(block_max.dtype)
        )
    ).clamp(min=-SCALE_EMAX, max=SCALE_EMAX)
    block_scale_q_int4  = 2**block_exp_int4
    w_s_int4            = w_fp_new / block_scale_q_int4
    w_q_int4            = w_s_int4.clamp(min=-7, max=7).round()

    ########## Select between MXFP4 and MXINT4 quantization ##########
    quant_error_fp4   = ((w_q_fp4 * block_scale_q_fp4 - w_fp_new) ** 2).sum(dim=-1)
    quant_error_int4  = ((w_q_int4 * block_scale_q_int4 - w_fp_new) ** 2).sum(dim=-1)
    select_fp4        = (quant_error_fp4 < quant_error_int4)[:, None]

    w_q  = torch.where(
        select_fp4,
        w_q_fp4,
        w_q_int4,
    )
    block_scale_q = torch.where(
        select_fp4,
        block_scale_q_fp4,
        block_scale_q_int4,
    )
    # Dequantization
    w_dq = w_q * block_scale_q

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_mxfp4_razer(w_fp, n_bits: int=4, groupsize: Optional[int]=None, is_act: bool=False):
    """
        MXFP4-RaZeR quantization.
    """
    SCALE_EMAX      = 31
    FP32_MIN_NORMAL = 2**(-126)
    FP4_EXP_BITS    = 2
    FP4_MAN_BITS    = 1
    FP4_EMAX        = 2
    FP4_MAX         = 6.0

    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, groupsize).to(torch.float32)
    
    scale_fp32     = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True) / FP4_MAX
    block_exp      = torch.ceil(
        torch.log2(
            scale_fp32 + FP32_MIN_NORMAL * (scale_fp32 == 0).type(scale_fp32.dtype)
        )
    ).clamp(min=-SCALE_EMAX, max=SCALE_EMAX)
    block_scale_q  = 2**block_exp
    w_s            = w_fp_new / block_scale_q
    
    ########## Normal FP4 quantization ##########
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s) + (w_s == 0).type(w_s.dtype)
        )
    )
    private_exp = private_exp.clamp(min=0)
    w_m         = w_s / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m         = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q_fp4     = w_m * (2**private_exp) / (2**FP4_MAN_BITS)

    ########## Search for the Optimal RaZeR-FP4 Special Value ##########
    if is_act:
        special_value_list = [-5.0, 5.0]
    else:
        special_value_list = [-5.0, 5.0, -3.5, 3.5]

    error     = torch.full([w_fp_new.shape[0]], float('inf'), dtype=w_fp_new.dtype, device=w_fp_new.device)
    w_q_razer = torch.zeros_like(w_fp_new)
    for special_value in special_value_list:
        w_q_razer_tmp = torch.where(
            (w_s - w_q_fp4).abs() < (w_s - special_value).abs(),
            w_q_fp4, special_value
        )
        # Dequantize and calculate error
        quant_error            = (w_q_razer_tmp - w_s).pow(2).mean(-1)
        mask_update            = torch.lt(quant_error, error)
        error[mask_update]     = quant_error[mask_update]
        w_q_razer[mask_update] = w_q_razer_tmp[mask_update]
    ##################################################################

    w_dq = w_q_razer * block_scale_q

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_mxfp4_razer_new(w_fp, n_bits: int=4, groupsize: Optional[int]=None, is_act: bool=False):
    """
        MXFP4-RaZeR quantization.
    """
    SCALE_EMAX      = 31
    FP32_MIN_NORMAL = 2**(-126)
    FP4_EXP_BITS    = 2
    FP4_MAN_BITS    = 1
    FP4_EMAX        = 2
    FP4_MAX         = 6.0

    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, groupsize).to(torch.float32)
    
    block_max      = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True) 
    scale_fp32     = block_max / FP4_MAX
    block_exp      = torch.ceil(
        torch.log2(
            scale_fp32 + FP32_MIN_NORMAL * (scale_fp32 == 0).type(scale_fp32.dtype)
        )
    ).clamp(min=-SCALE_EMAX, max=SCALE_EMAX)
    block_max_s    = block_max / 2**block_exp
    upscale_mask   = block_max_s.lt(3.5).squeeze() # If the scaled block maximum is smaller than 3.5, then upscale the whole block by 2
    block_exp[upscale_mask] -= 1
    block_scale_q  = 2**block_exp
    w_s            = w_fp_new / block_scale_q
    
    ########## Normal FP4 quantization ##########
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s) + (w_s == 0).type(w_s.dtype)
        )
    )
    private_exp = private_exp.clamp(min=0)
    w_m         = w_s / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m         = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q_fp4     = w_m * (2**private_exp) / (2**FP4_MAN_BITS)

    ########## Search for the Optimal RaZeR-FP4 Special Value ##########
    if is_act:
        special_value_list = [-5.0, 5.0]
    else:
        special_value_list = [-5.0, 5.0, -7.0, 7.0]

    error     = torch.full([w_fp_new.shape[0]], float('inf'), dtype=w_fp_new.dtype, device=w_fp_new.device)
    w_q_razer = torch.zeros_like(w_fp_new)
    for special_value in special_value_list:
        w_q_razer_tmp = torch.where(
            (w_s - w_q_fp4).abs() < (w_s - special_value).abs(),
            w_q_fp4, special_value
        )
        # Dequantize and calculate error
        quant_error            = (w_q_razer_tmp - w_s).pow(2).mean(-1)
        mask_update            = torch.lt(quant_error, error)
        error[mask_update]     = quant_error[mask_update]
        w_q_razer[mask_update] = w_q_razer_tmp[mask_update]
    ##################################################################

    w_dq = w_q_razer * block_scale_q

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_nvfp4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        NVFP4 quantization. 
    """
    FP4_MAX       = 6.0
    FP4_MAN_BITS  = 1

    orig_shape    = w_fp.shape 
    w_fp_new      = w_fp.reshape(-1, groupsize).to(torch.float32)

    global_qmax   = FP4_MAX * 448
    global_scale  = w_fp_new.abs().amax() / global_qmax

    ############### Block Scale Quantization ###############
    w_scaled      = w_fp_new / global_scale
    block_max     = w_scaled.abs().amax(dim=-1, keepdim=True)
    block_scale_q = (block_max / FP4_MAX).clamp(
        max=448,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    w_scaled      = w_scaled / block_scale_q

    #################### FP4 Quantization ####################
    private_exp   = torch.floor(
        torch.log2(
            torch.abs(w_scaled) + (w_scaled == 0).type(w_scaled.dtype)
        )
    )
    private_exp   = private_exp.clamp(min=0)
    w_m           = w_scaled / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m           = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q           = w_m * (2**private_exp) / (2**FP4_MAN_BITS)
    w_dq          = w_q * block_scale_q * global_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_nvfp4_4over6(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        NVFP4 4over6 quantization. Following the implementation of FourOverSix (https://arxiv.org/pdf/2512.02010)
    """
    quant_value = sorted([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mid_value = [(quant_value[i] + quant_value[i + 1]) / 2 for i in range(len(quant_value) - 1)]

    orig_shape = w_fp.shape 
    w_fp_new = w_fp.reshape(-1, groupsize).to(torch.float32)

    qmax_6       = 6.0
    qmax_4       = 4.0
    global_qmax  = qmax_6 * 448
    global_scale = w_fp_new.abs().amax() / global_qmax

    w_scaled       = w_fp_new / global_scale
    w_dq_sign      = w_scaled.sign()
    block_scale    = w_scaled.abs().amax(dim=-1, keepdim=True)
    block_scale_6  = (block_scale / qmax_6).clamp(
        max=torch.finfo(torch.float8_e4m3fn).max,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    block_scale_4  = (block_scale / qmax_4).clamp(
        max=torch.finfo(torch.float8_e4m3fn).max,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    w_scaled_6     = (w_scaled / block_scale_6).abs()
    w_scaled_4     = (w_scaled / block_scale_4).abs()

    w_q_6 = torch.zeros_like(w_scaled)
    for i in range(len(quant_value)):
        data = quant_value[i]
        if i == 0:
            w_q_6 += torch.where(w_scaled_6 <= mid_value[i], data, 0)
        elif i == len(quant_value) - 1:
            w_q_6 += torch.where(w_scaled_6 > mid_value[i - 1], data, 0)
        else:
            w_q_6 += torch.where((mid_value[i - 1] < w_scaled_6) & (w_scaled_6 <= mid_value[i]), data, 0)
    w_q_6 = w_q_6 * w_dq_sign

    w_q_4 = torch.zeros_like(w_scaled)
    for i in range(len(quant_value)):
        data = quant_value[i]
        if i == 0:
            w_q_4 += torch.where(w_scaled_4 <= mid_value[i], data, 0)
        elif i == len(quant_value) - 1:
            w_q_4 += torch.where(w_scaled_4 > mid_value[i - 1], data, 0)
        else:
            w_q_4 += torch.where((mid_value[i - 1] < w_scaled_4) & (w_scaled_4 <= mid_value[i]), data, 0)
    w_q_4 = w_q_4 * w_dq_sign

    quant_error_6 = ((w_q_6*block_scale_6 - w_scaled) ** 2).sum(dim=-1)
    quant_error_4 = ((w_q_4*block_scale_4 - w_scaled) ** 2).sum(dim=-1)
    select_4      = (quant_error_4 < quant_error_6)[:, None]

    w_q = torch.where(
        select_4,
        w_q_4,
        w_q_6,
    )
    block_scale = torch.where(
        select_4,
        block_scale_4,
        block_scale_6,
    )

    w_dq = w_q * block_scale * global_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


# Nine points at 6.25% spacing across [1, 1.5]. The endpoints are plain NVFP4 (alpha=1) and
# FourOverSix (alpha=1.5); everything between is new. 6.25% is about the finest spacing the ue4m3
# scale's 3-bit mantissa still resolves, and measurement says the whole useful range is [1, 1.5] --
# on real Llama-2-7B weights, alpha = 2 and 3 are chosen by 0.0% of scale blocks.
NOVER6_ALPHAS = tuple(1.0 + 0.0625 * i for i in range(9))


@torch.no_grad()
def quant_nvfp4_nover6(w_fp, n_bits: int=4, groupsize: Optional[int]=None,
                       alphas=NOVER6_ALPHAS):
    """
        NVFP4 with a WIDER FourOverSix search -- "N over six".

        FourOverSix normalizes a scale block's maximum to code 6 or to code 4, whichever quantizes
        the block better. Nothing distinguishes 4: the choice is just the value written into the
        ue4m3 scale field, so any `block_scale = alpha * block_max / 6` is equally free, and each
        alpha lands the block maximum on a different code:

            alpha=1    -> code 6   {0,.083,.167,.25,.333,.5,.667,1}   log-spaced
            alpha=1.25 -> code 4.8
            alpha=1.5  -> code 4   {0,.125,.25,.375,.5,.75,1}         FourOverSix
            alpha=2    -> code 3   {0,.167,.333,.5,.667,1}            uniform, 6 levels
            alpha=3    -> code 2   {0,.25,.5,.75,1}                   uniform, 4 levels

        (values in units of the block maximum). So the family walks E2M1 from log-spaced at full
        range to uniform with few levels, spending the sparse top of the grid rather than the
        resolution near zero.

        Measured against `quant_nvfp4_4over6` at W4A16, wikitext / c4:

            Llama-3.1-8B   -0.0146 / -0.0218
            Llama-3.2-3B   -0.0159 / -0.0417
            Llama-2-7B     +0.0094 / -0.0109

        i.e. the largest average gain of anything measured in this study, and it needs **no type
        block, no E0M3 operand and no election rule** -- it is plain NVFP4 with a proper
        per-scale-block scale search, running on the existing kernel with no metadata beyond the
        ue4m3 scale NVFP4 already stores. This is `quant_mix_4_6(clip="dense9", elect="never")` with
        the type-block machinery stripped out.

        The Llama-2-7B wikitext row is the one weak spot, offset by c4 on the same model. If you
        need a configuration that is negative on every model AND every dataset, use
        `mix_4_6_clipbothx_clipmin0.3_h3` instead -- smaller (-0.0076 three-model mean against
        -0.0159) but strictly safe. See `results/DECIDE_SUMMARY.md`.

        `alphas < 1` would be clipping. Ungated it measurably costs perplexity, so do not add them
        here; the gated form lives in `quant_mix_4_6` behind `clip_min_gain`.
    """
    E2M1_MAX      = 6.0
    FP8_SCALE_MAX = 448.0
    FP8_SCALE_MIN = 2 ** (-9)

    groupsize  = 16 if groupsize is None else groupsize
    orig_shape = w_fp.shape
    w_fp_new   = w_fp.reshape(-1, groupsize).to(torch.float32)

    global_scale = (w_fp_new.abs().amax() / (E2M1_MAX * FP8_SCALE_MAX)).clamp(
        min=torch.finfo(torch.float32).tiny
    )
    w_scaled  = w_fp_new / global_scale
    block_max = w_scaled.abs().amax(dim=-1, keepdim=True)

    best_dq, best_err = None, None
    for alpha in alphas:
        block_scale = (block_max * (alpha / E2M1_MAX)).clamp(
            max=FP8_SCALE_MAX, min=FP8_SCALE_MIN
        ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
        dq  = _quant_e2m1(w_scaled, block_scale)
        err = (dq - w_scaled).pow(2).sum(dim=-1, keepdim=True)
        if best_dq is None:
            best_dq, best_err = dq, err
        else:
            take     = err < best_err
            best_dq  = torch.where(take, dq, best_dq)
            best_err = torch.where(take, err, best_err)

    return (best_dq * global_scale).view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_nvif4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        NVIF4 quantization. Following the implementation of (https://arxiv.org/abs/2603.28765)
    """
    FP4_MAX      = 6.0
    INT4_MAX     = 7.0
    FP4_MAN_BITS = 1

    #################### Reshape Tensor ####################
    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, groupsize).to(torch.float32)
 
    #################### Global Scale ####################
    global_qmax  = FP4_MAX * 448
    global_scale = w_fp_new.abs().amax() / global_qmax

    ############### Block Scale Quantization ###############
    w_scaled         = w_fp_new / global_scale
    block_max        = w_scaled.abs().amax(dim=-1, keepdim=True)
    block_scale_fp4  = (block_max / FP4_MAX).clamp(
        max=448,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    block_scale_int4 = (block_max / INT4_MAX).clamp(
        max=448,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    w_scaled_fp4     = w_scaled / block_scale_fp4
    w_scaled_int4    = w_scaled / block_scale_int4

    #################### FP4 Quantization ####################
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_scaled_fp4) + (w_scaled_fp4 == 0).type(w_scaled_fp4.dtype)
        )
    ).clamp(min=0)
    w_m_fp4     = w_scaled_fp4 / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m_fp4     = torch.sign(w_m_fp4) * torch.floor(torch.abs(w_m_fp4) + 0.5)
    w_q_fp4     = w_m_fp4 * (2**private_exp) / (2**FP4_MAN_BITS)

    #################### INT4 Quantization ####################
    w_q_int4    = w_scaled_int4.clamp(min=-7, max=7).round()

    #################### Select Optimal Data Type ####################
    quant_error_fp4  = ((w_q_fp4 * block_scale_fp4 - w_scaled) ** 2).sum(dim=-1)
    quant_error_int4 = ((w_q_int4 * block_scale_int4 - w_scaled) ** 2).sum(dim=-1)
    select_int4      = (quant_error_int4 < quant_error_fp4)[:, None]
    w_q              = torch.where(
        select_int4,
        w_q_int4,
        w_q_fp4,
    )
    block_scale   = torch.where(
        select_int4,
        block_scale_int4,
        block_scale_fp4,
    )

    w_dq = w_q * block_scale * global_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def _tile_type_blocks(x, block_m: int, block_k: int, groupsize: int):
    """
        Reshape a 2-D tensor (M, K) into MixFP4 type blocks of scale blocks:
            (num_type_block, num_scale_block_per_type_block, groupsize)

        Returns the tiled tensor plus the metadata needed by `_untile_type_blocks`.
    """
    num_row, num_col = x.shape
    assert num_col % block_k == 0, \
        f'The reduction dimension {num_col} must be divisible by the type-block K dimension {block_k}.'
    assert block_k % groupsize == 0, \
        f'The type-block K dimension {block_k} must be divisible by the scale-block size {groupsize}.'

    # Zero-pad the outer dimension so that it is divisible by the type-block M dimension.
    # The padded rows form their own all-zero type blocks and therefore never affect the
    # data type selected for the real rows.
    pad_row = (-num_row) % block_m
    if pad_row > 0:
        x = torch.cat(
            [x, torch.zeros(pad_row, num_col, dtype=x.dtype, device=x.device)],
            dim=0
        )

    num_block_m = x.shape[0] // block_m
    num_block_k = num_col // block_k
    # (M/BM, BM, K/BK, BK) -> (M/BM, K/BK, BM, BK) -> (num_type_block, num_scale_block, groupsize)
    x = x.view(num_block_m, block_m, num_block_k, block_k).permute(0, 2, 1, 3)
    x = x.reshape(-1, block_m * block_k // groupsize, groupsize)

    return x, (num_row, num_col, num_block_m, num_block_k)


@torch.no_grad()
def _untile_type_blocks(x, block_m: int, block_k: int, meta):
    """
        Inverse of `_tile_type_blocks`: fold the type blocks back into a (M, K) tensor.
    """
    num_row, num_col, num_block_m, num_block_k = meta

    x = x.view(num_block_m, num_block_k, block_m, block_k).permute(0, 2, 1, 3)
    x = x.reshape(num_block_m * block_m, num_col)

    return x[:num_row]


@torch.no_grad()
def quant_mixfp4(
    w_fp,
    n_bits: int=4,
    groupsize: Optional[int]=None,
    type_block=(1, 16),
    is_act: bool=False,
):
    """
        MixFP4 quantization (CPU simulation / fake quantization).

        MixFP4 is NVFP4 with a second, coarser block granularity:
          * scale block (16 elements, fixed): one E4M3 block scale, exactly like NVFP4.
          * type block  (block_m x block_k, configurable): all FP4 elements inside the type block
            share ONE element data type, either E2M1 or E0M3. A type block always contains a whole
            number of scale blocks (block_k is a multiple of 16).

        E2M1 is the standard FP4 grid {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}. E0M3 is the evenly
        spaced signed 4-bit grid {0, +-1, ..., +-7}, i.e. the value set the mxf4nvf4 MMA reads when
        an operand is declared E0M3. Both grids hold 15 distinct values out of 16 codes (the
        redundant zero), and both use the same ue4m3 block scale; only the spacing differs.

        The data type is chosen per type block by minimizing the sum of squared quantization errors
        over every scale block it contains.
    """
    FP4_MAN_BITS  = 1
    E2M1_MAX      = 6.0
    E0M3_MAX      = 7.0        # evenly spaced signed 4-bit grid {0, +-1, ..., +-7}
    FP8_SCALE_MAX = 448.0
    FP8_SCALE_MIN = 2**(-9)

    groupsize     = 16 if groupsize is None else groupsize
    assert groupsize == 16, \
        f'MixFP4 inherits the NVFP4 scale-block size, which must be 16, but got {groupsize}.'
    block_m, block_k = parse_type_block(type_block)

    #################### Reshape Tensor ####################
    orig_shape = w_fp.shape
    w_fp_new   = w_fp.reshape(-1, orig_shape[-1]).to(torch.float32)
    num_col    = w_fp_new.shape[-1]
    assert num_col % groupsize == 0, \
        f'The reduction dimension {num_col} must be divisible by the scale-block size {groupsize}.'
    # Narrow tensors (e.g. a 64-wide head dimension with a 32x128 type block) cannot hold a full
    # type block along K. Shrink the type block to the full row instead of failing the sweep.
    if num_col % block_k != 0:
        assert block_k > num_col, \
            f'The reduction dimension {num_col} must be divisible by the type-block K dimension {block_k}.'
        block_k = num_col

    #################### Global Scale ####################
    global_qmax  = E2M1_MAX * FP8_SCALE_MAX
    global_scale = (w_fp_new.abs().amax() / global_qmax).clamp(min=torch.finfo(torch.float32).tiny)
    w_scaled     = w_fp_new / global_scale

    w_tiled, meta = _tile_type_blocks(w_scaled, block_m, block_k, groupsize)
    block_max     = w_tiled.abs().amax(dim=-1, keepdim=True)

    ############### E2M1 Scale Block Quantization ###############
    block_scale_e2m1 = (block_max / E2M1_MAX).clamp(
        max=FP8_SCALE_MAX,
        min=FP8_SCALE_MIN
    ).to(torch.float8_e4m3fn).to(w_tiled.dtype)
    w_scaled_e2m1    = w_tiled / block_scale_e2m1
    private_exp      = torch.floor(
        torch.log2(
            torch.abs(w_scaled_e2m1) + (w_scaled_e2m1 == 0).type(w_scaled_e2m1.dtype)
        )
    ).clamp(min=0)
    w_m_e2m1         = w_scaled_e2m1 / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m_e2m1         = torch.sign(w_m_e2m1) * torch.floor(torch.abs(w_m_e2m1) + 0.5)
    w_q_e2m1         = (w_m_e2m1 * (2**private_exp) / (2**FP4_MAN_BITS)).clamp(min=-E2M1_MAX, max=E2M1_MAX)

    ############### E0M3 Scale Block Quantization ###############
    block_scale_e0m3 = (block_max / E0M3_MAX).clamp(
        max=FP8_SCALE_MAX,
        min=FP8_SCALE_MIN
    ).to(torch.float8_e4m3fn).to(w_tiled.dtype)
    w_q_e0m3         = (w_tiled / block_scale_e0m3).round().clamp(min=-E0M3_MAX, max=E0M3_MAX)

    ############### Per-Type-Block Data Type Selection ###############
    w_dq_e2m1   = w_q_e2m1 * block_scale_e2m1
    w_dq_e0m3   = w_q_e0m3 * block_scale_e0m3
    # Sum the squared error over every scale block belonging to the same type block
    error_e2m1  = (w_dq_e2m1 - w_tiled).pow(2).sum(dim=(-1, -2))
    error_e0m3  = (w_dq_e0m3 - w_tiled).pow(2).sum(dim=(-1, -2))
    select_e0m3 = (error_e0m3 < error_e2m1)[:, None, None]

    w_dq = torch.where(
        select_e0m3,
        w_dq_e0m3,
        w_dq_e2m1,
    )
    w_dq = _untile_type_blocks(w_dq, block_m, block_k, meta) * global_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def _quant_e2m1(x, block_scale):
    """
        Round to the E2M1 grid {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6} after dividing by
        `block_scale`. Returns the DEQUANTIZED values (still in the globally scaled domain).
    """
    FP4_MAN_BITS = 1
    E2M1_MAX     = 6.0

    x_s         = x / block_scale
    private_exp = torch.floor(
        torch.log2(
            torch.abs(x_s) + (x_s == 0).type(x_s.dtype)
        )
    ).clamp(min=0)
    x_m = x_s / (2**private_exp) * (2**FP4_MAN_BITS)
    x_m = torch.sign(x_m) * torch.floor(torch.abs(x_m) + 0.5)
    x_q = (x_m * (2**private_exp) / (2**FP4_MAN_BITS)).clamp(min=-E2M1_MAX, max=E2M1_MAX)

    return x_q * block_scale


@torch.no_grad()
def _quant_e0m3(x, block_scale):
    """
        Round to the evenly spaced signed 4-bit grid {0, +-1, ..., +-7} after dividing by
        `block_scale`. Returns the DEQUANTIZED values (still in the globally scaled domain).
    """
    E0M3_MAX = 7.0
    return (x / block_scale).round().clamp(min=-E0M3_MAX, max=E0M3_MAX) * block_scale


# Per-scale-block scale candidates, expressed as a CLIP RATIO alpha:
#
#     block_scale = alpha * block_max / grid_max          (grid_max = 6 for E2M1, 7 for E0M3)
#
# so the largest representable magnitude is alpha * block_max:
#
#   alpha = 1    -- the block maximum lands exactly on the top code. No clipping, no waste.
#   alpha < 1    -- CLIPPING. Everything above alpha*block_max saturates to the top code, and in
#                   exchange every step below it shrinks by the same factor. Wins whenever the block
#                   maximum is an isolated outlier that the rest of the block is paying for.
#   alpha > 1    -- headroom: the top codes go unused and the grid is stretched. Useless on E0M3,
#                   but on E2M1 it moves the bulk onto a MORE UNIFORM part of the grid -- alpha=1.5
#                   is exactly the "4" of FourOverSix (block_max -> code 4 instead of code 6).
#
# Choosing alpha per scale block is FREE: it only changes the value written into the ue4m3 scale
# field that both grids already carry, and the decoder still just multiplies code by scale. It costs
# no metadata and, unlike the E2M1/E0M3 choice, it does not have to be uniform across a type block.
# Note the ue4m3 scale has a 3-bit mantissa, so alphas closer together than ~6% are not always
# distinguishable after rounding -- a coarse grid is not a limitation here.
#
# `base` is the pre-existing behaviour (FourOverSix on E2M1, exact-fit E0M3) and is the default.
# The rest is a factorial design over WHICH grid gets the clipping, at two strengths, so that a
# measured gain can be attributed instead of guessed at: clipping helps both branches, and a variant
# that clips both is not evidence that the E2M1-vs-E0M3 decision got any better.
CLIP_PRESETS = {
    "base":  {"e2m1": (1.0, 1.5),                          "e0m3": (1.0,)},
    # NO alpha search at all -- the block maximum stays on the top code of each grid, which is
    # plain NVFP4 plus the E0M3 option and nothing else.
    #
    # This exists because alpha widening is NOT universally safe, contrary to how the rest of these
    # presets are framed. On Qwen3-4B, nvfp4 (alpha=1) scores 13.6584/16.8723 while 4over6 scores
    # 14.0407/17.0153 -- FourOverSix is +0.38 wikitext WORSE than the unmodified format there,
    # and headx is +0.33 worse. On Llama-3.1-8B the same two changes are -0.025 and -0.034, i.e.
    # improvements. The mechanism is the one the selection analysis found: E2M1 is log-spaced and
    # coarse at the top, so its top codes are what absorb an outlier, and alpha > 1 discards exactly
    # those. A model with peakier blocks cannot afford that.
    #
    # So `a1` isolates the E0M3 election from the alpha search, which every other preset entangles
    # it with.
    "a1":    {"e2m1": (1.0,),                              "e0m3": (1.0,)},
    # E0M3 only -- the branch with no free normalization today
    "e0":    {"e2m1": (1.0, 1.5),                          "e0m3": (1.0, 0.9)},
    "e0x":   {"e2m1": (1.0, 1.5),                          "e0m3": (1.0, 0.9, 0.8)},
    # E2M1 headroom only -- alpha >= 1 never clips, so this extends FourOverSix along the one
    # direction round 1 showed to be safe, and it is the largest single win measured here
    # (-0.0186/-0.0098 for `headx`, against -0.0117/-0.0044 for `base`).
    #
    # What headroom actually does is turn E2M1 into a UNIFORM grid with fewer levels. Writing the
    # usable code values in units of the block maximum, with alpha mapping the block max to code
    # 6/alpha:
    #
    #   alpha=1    block max -> code 6   {0, .083, .167, .25, .333, .5, .667, 1}   log-spaced
    #   alpha=1.5  block max -> code 4   {0, .125, .25, .375, .5, .75, 1}          4over6
    #   alpha=2    block max -> code 3   {0, .167, .333, .5, .667, 1}              uniform, 6 levels
    #   alpha=3    block max -> code 2   {0, .25, .5, .75, 1}                      uniform, 4 levels
    #
    # so the family interpolates from "log-spaced at full range" to "uniform with few levels", and
    # the top codes it wastes are exactly the sparse part of the E2M1 grid. E0M3 is the one point
    # this family cannot reach: uniform with SEVEN levels at full range. That is why headroom and
    # the E0M3 election are worth more together than apart -- headroom lets a scale block sit well
    # on whichever grid its tile elected, and E0M3 supplies the finest uniform grid on offer.
    "head":  {"e2m1": (1.0, 1.5, 2.0),                     "e0m3": (1.0,)},
    "headx": {"e2m1": (1.0, 1.25, 1.5, 2.0, 3.0),          "e0m3": (1.0,)},
    "headxx": {"e2m1": (1.0, 1.2, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0), "e0m3": (1.0,)},
    # REMOVED: `heade0` / `heade0x`, which gave E0M3 its own headroom candidates
    # (alpha = 7/6, 7/5, ...). E0M3 with alpha = 7/n maps the block maximum to code n, i.e. a
    # uniform n-level grid, which E2M1 cannot supply above n = 4 -- so those presets did have a
    # principled basis. They are gone because E0M3 headroom is deliberately NOT a factor in this
    # work: it entangles the element-type decision with a second scale search on the E0M3 branch,
    # and every result that needs a wide scale search now uses `headx`, which searches E2M1 only
    # and pins E0M3 at alpha = 1.
    #
    # `test_no_e0m3_headroom` enforces that no preset reachable from `headx`/`base`/`a1` reintroduces
    # it. Presets further down that DO give E0M3 alpha > 1 (`dense9e0`, `dense9sym`, `dense5sym`,
    # `basesym`, `wide`, `full`) are kept deliberately: they exist as confound controls for the
    # question "does E0M3 stop contributing once alpha is searched", and none appears in a reported
    # result. Do not promote one into a headline configuration.
    # Headroom on both grids PLUS clipping candidates. Only usable with `clipmin<t>`: the clipping
    # alphas are gated behind a minimum gain, which is what makes them safe (round 7 measures
    # clipbothx + clipmin0.15 at the best c4 of the study, against a loss for ungated clipping).
    "full":  {"e2m1": (0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0),
              "e0m3": (0.8, 0.9, 1.0, 7.0 / 6.0, 7.0 / 5.0)},
    # DENSE headroom in [1, 1.5], which is where the action actually is. Measured on Llama-2-7B,
    # the share of scale blocks choosing each `headx` candidate is 58.3% / 3.6% / 38.0% / 0% / 0%
    # for alpha = 1 / 1.25 / 1.5 / 2 / 3 -- the coarse uniform grids are never selected, and the
    # whole gain comes from the single extra point at 1.25. Subdividing [1, 1.5] instead cuts weight
    # NMSE by 4.2% (five points) and 5.4% (nine), against 0.76% for `headx`. The ue4m3 scale has a
    # 3-bit mantissa, so ~6% steps are near the finest that survives the scale rounding.
    "dense5":  {"e2m1": (1.0, 1.125, 1.25, 1.375, 1.5),    "e0m3": (1.0,)},
    "dense9":  {"e2m1": tuple(1.0 + 0.0625 * i for i in range(9)), "e0m3": (1.0,)},
    # Twice as fine again, and out to alpha = 2. Only sensible together with `amin<t>`: without the
    # gate a denser grid fits more MSE noise (round 7's `headxx`, round 17a's dense grid on
    # Llama-2-7B), and the gate is what makes extra candidates safe to offer.
    "dense17": {"e2m1": tuple(1.0 + 0.03125 * i for i in range(17)), "e0m3": (1.0,)},
    "dense2x": {"e2m1": tuple(1.0 + 0.0625 * i for i in range(17)), "e0m3": (1.0,)},
    # ... and the same subdivision on E0M3, whose alpha = 7/n reaches uniform n-level grids
    "dense9e0": {"e2m1": tuple(1.0 + 0.0625 * i for i in range(9)),
                 "e0m3": (1.0, 7.0 / 6.5, 7.0 / 6.0, 7.0 / 5.5, 7.0 / 5.0)},
    # SYMMETRIC: both grids get the identical nine-point alpha range. Every other preset gives E2M1
    # more candidates than E0M3 -- `base` is the extreme case, {1, 1.5} against {1}, i.e. FourOverSix
    # searches the E2M1 scale and never searches the E0M3 one. Any claim that "E0M3 stops
    # contributing once alpha is searched" is confounded unless both branches get the same search,
    # which is what this preset is for.
    "dense9sym": {"e2m1": tuple(1.0 + 0.0625 * i for i in range(9)),
                  "e0m3": tuple(1.0 + 0.0625 * i for i in range(9))},
    "dense5sym": {"e2m1": (1.0, 1.125, 1.25, 1.375, 1.5),
                  "e0m3": (1.0, 1.125, 1.25, 1.375, 1.5)},
    # and the symmetric version of the ORIGINAL setting: give E0M3 its own 4/6-style second option
    "basesym":  {"e2m1": (1.0, 1.5),                       "e0m3": (1.0, 1.5)},
    # E2M1 only -- on top of FourOverSix
    "e2":    {"e2m1": (0.9, 1.0, 1.5),                     "e0m3": (1.0,)},
    "e2x":   {"e2m1": (0.8, 0.9, 1.0, 1.5),                "e0m3": (1.0,)},
    # both grids
    "both":  {"e2m1": (0.9, 1.0, 1.5),                     "e0m3": (1.0, 0.9)},
    "bothx": {"e2m1": (0.8, 0.9, 1.0, 1.5),                "e0m3": (1.0, 0.9, 0.8)},
    "wide":  {"e2m1": (0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0), "e0m3": (0.7, 0.8, 0.9, 1.0, 1.15)},
}


SELECT_METRICS = ("mse", "sqnr", "cossim", "mae")


def _parse_metric(metric: str):
    """
        Split a metric name into (family, param).

        "l<p>"    -> ("lp", p)     the generalized power loss sum|dW|^p; "mae" is p=1, "mse" is p=2.
        "corr<r>" -> ("corr", r)   the equicorrelated-input output error, see `_selection_loss`.
    """
    if metric.startswith("corr"):
        tail = metric[len("corr"):]
        if tail and tail.replace(".", "", 1).isdigit():
            return "corr", float(tail)
    tail = metric[1:]
    if metric.startswith("l") and tail and tail.replace(".", "", 1).isdigit():
        return "lp", float(tail)
    if metric == "mae":
        return "lp", 1.0
    return metric, 2.0


@torch.no_grad()
def _selection_loss(x, x_dq, metric: str, weight=None, eps: float=1e-30):
    """
        Per-scale-block selection loss (LOWER is better), shape (..., num_scale_block, 1).

        metric:
          "mse"    - summed squared error over the scale block.
          "mae"    - summed ABSOLUTE error, i.e. "l1". Squared error is dominated by the single
                     worst element of a block, which is exactly the element a clipping candidate
                     gives up on purpose; L1 scores the bulk instead.
          "l<p>"   - summed |error|^p for any p > 0. p < 1 discounts the tail harder still.
          "corr<r>"- the layer-output error under the assumption that the input channels are
                     EQUICORRELATED with correlation r. For y_i = sum_j x_j W_ij the output error is
                     sum_j x_j dW_ij, so

                         E[(sum_j x_j dW_j)^2] = sum_jk E[x_j x_k] dW_j dW_k,

                     and with E[x_j x_k] = sigma^2 (delta_jk + r (1 - delta_jk)) this is

                         sigma^2 [ sum_j dW_j^2 + r ( (sum_j dW_j)^2 - sum_j dW_j^2 ) ].

                     r=0 is exactly MSE. r>0 additionally punishes error that is COHERENT across the
                     block, which plain MSE cannot see at all: 16 errors of +d cost the same as 16
                     errors of alternating sign under MSE, but the first accumulates in the dot
                     product and the second cancels. Clipping produces exactly the coherent kind
                     (every clipped element is pulled the same way), so this is the term that decides
                     whether a clip candidate is really cheaper. It needs no calibration data -- r is
                     one global constant standing in for how much of the activation is a shared
                     direction, which post-LayerNorm inputs certainly have.
          "sqnr"   - negated signal-to-quantization-noise ratio in dB,
                     -10*log10(||x||^2 / ||x - x_q||^2).
                     NOTE: for choosing between candidates WITHIN one scale block this is exactly
                     equivalent to "mse" -- the signal energy is the same for every candidate, so it
                     cancels out of the ranking. It differs only when the per-block losses are summed
                     to pick a TYPE-BLOCK data type, because each scale block is then normalized by
                     its own energy instead of high-energy blocks dominating the total.
          "cossim" - 1 - cosine similarity. Scale invariant: it scores direction only and ignores a
                     uniform magnitude error, which the block scale can partly absorb anyway.

        All-zero blocks score 0 for "mse"/"sqnr"/"l<p>" and 1 for "cossim", and never produce
        NaN/inf.
    """
    family, p = _parse_metric(metric)

    if family == "lp":
        err = (x_dq - x).abs()
        if p != 1.0:
            err = err.pow(p)
        if weight is not None:
            err = err * weight
        return err.sum(dim=-1, keepdim=True)

    if family == "corr":
        assert 0.0 <= p < 1.0, f'"corr<r>" needs 0 <= r < 1, got {p}.'
        # importance enters as diag(s) S diag(s), i.e. it scales the error vector itself
        d      = (x_dq - x) if weight is None else (x_dq - x) * weight.sqrt()
        sum_sq = d.pow(2).sum(dim=-1, keepdim=True)
        sq_sum = d.sum(dim=-1, keepdim=True).pow(2)
        return sum_sq + p * (sq_sum - sum_sq)

    sq = (x_dq - x).pow(2)
    if weight is not None:
        # `weight` is the per-element importance s_j^2 = E[x_j^2] of the input channel that this
        # weight multiplies. Summing s_j^2 * dW_ij^2 over a block is the diagonal-Hessian estimate
        # of how much this block's quantization raises the LAYER OUTPUT error, which is the
        # quantity we actually care about -- unweighted MSE is the special case s_j^2 = 1.
        sq = sq * weight
    error = sq.sum(dim=-1, keepdim=True)

    if metric == "mse":
        return error

    if metric == "sqnr":
        signal = (x.pow(2) if weight is None else x.pow(2) * weight).sum(dim=-1, keepdim=True)
        # +eps on both sides keeps an exactly-representable block (error 0) finite and keeps an
        # all-zero block at 0 dB instead of 0/0
        return -10.0 * torch.log10((signal + eps) / (error + eps))

    if metric == "cossim":
        w      = 1.0 if weight is None else weight
        dot    = (x * x_dq * w).sum(dim=-1, keepdim=True)
        norm_x = (x.pow(2) * w).sum(dim=-1, keepdim=True).sqrt()
        norm_q = (x_dq.pow(2) * w).sum(dim=-1, keepdim=True).sqrt()
        return 1.0 - dot / (norm_x * norm_q).clamp(min=eps)

    raise ValueError(f"Unsupported selection metric \"{metric}\". Expected one of {SELECT_METRICS}.")


ELECT_RULES = ("argmin", "dominance", "margin", "never", "always",
               "vote", "harm", "relmargin", "tol")


@torch.no_grad()
def _elect_e0m3(gain, rule: str="argmin", margin: float=0.0, ref=None, eps: float=1e-30):
    """
        Decide, per type block, whether to elect E0M3 over E2M1.

        `gain` has shape (num_type_block, num_scale_block, 1) and holds, per SCALE block,
            gain_b = loss_E2M1(b) - loss_E0M3(b),
        so gain_b > 0 means E0M3 is better on that block. Returns a (num_type_block, 1, 1) mask.
        `ref` is the per-scale-block E2M1 loss, same shape, and is what the relative rules
        ("relmargin", "tol") measure the gain against; the absolute rules ignore it.

        The whole point of these rules is that the type block has to overrule scale blocks that
        disagree with it, and a bare sum of losses hides how many. Rules, in decreasing order of
        caution:

        "dominance" -- elect only if gain_b >= 0 for EVERY scale block in the tile. Then no block is
            ever worse than it would be under plain 4over6, so the improvement is POINTWISE, not
            merely aggregate. This is the property a 1x16 type block has for free, and it is what
            makes 1x16 the only size that reliably beats 4over6 on perplexity. Safe but rarely fires
            on large tiles.

        "tol"       -- dominance with a tolerance: elect if the total gain is positive AND no single
            scale block is harmed by more than `margin` (as a FRACTION of its own E2M1 loss).
            margin=0 is dominance, margin=inf is argmin. Keeps the pointwise character of dominance
            -- a bounded worst case -- while letting tiles through that dominance rejects over one
            barely-harmed block.

        "margin"    -- elect only if the mean gain exceeds `margin` standard errors of the per-block
            gain:  mean(gain) > margin * std(gain) / sqrt(B).
            This is a one-sided test of "the expected gain is positive" against the block-to-block
            spread. margin=0 reduces to "argmin"; margin~2 demands the tile's advantage be large
            compared to how much the decision hurts individual blocks, which is exactly the churn
            that makes an aggregate MSE win meaningless.

        "relmargin" -- the same test on the RELATIVE gain gain_b / loss_E2M1(b). The absolute gain is
            dominated by whichever scale blocks happen to carry the most energy, so a tile can pass
            "margin" on the strength of two big blocks while most of its blocks are harmed.
            Normalizing per block gives every scale block an equal vote in the variance.

        "harm"      -- elect only if the gain the winners collect outweighs the damage the losers
            take by a factor of `margin`:  sum(gain_b | gain_b>0) > margin * sum(-gain_b | gain_b<0).
            margin=1 is exactly "argmin".

            This is the ROBUST version of the decision, and the only rule here with a derivation
            rather than a heuristic. What we actually want to minimize is sum_b w_b * loss_b for the
            unknown per-block importance w_b > 0; argmin assumes w_b = 1 for every block, which is
            the assumption CLAUDE.md shows to be wrong by orders of magnitude. Electing E0M3 only
            when sum_b w_b gain_b > 0 for EVERY w in an uncertainty set gives, for the set
            {w : 1/kappa <= w_b <= kappa},

                sum_{gain>0} gain_b / kappa  >  kappa * sum_{gain<0} |gain_b|,

            i.e. exactly this rule with margin = kappa^2. So `margin` is not a free knob: it is the
            squared spread of the per-block importance one is willing to be wrong about. kappa -> inf
            recovers "dominance", kappa = 1 recovers "argmin", and the rules in between are the
            robust decisions for intermediate spreads.

        "vote"      -- elect iff the FRACTION of scale blocks that individually prefer E0M3 exceeds
            `margin`. Ignores magnitudes entirely, so no single block can carry a tile. margin=0.5 is
            a plain majority of the scale blocks the tile is about to overrule.

        "argmin"    -- the plain sum comparison: elect iff total gain > 0.
    """
    total = gain.sum(dim=(-1, -2))

    if rule == "never":
        # Never elect E0M3. The result is the E2M1 branch alone, i.e. plain 4over6 under whatever
        # metric/importance is in force -- the control for "how much comes from calibration".
        # NOTE: a huge `margin` does NOT achieve this. At a 1x16 type block the tile holds a single
        # scale block, so std(gain) is 0 and every margin collapses back to `argmin`.
        elect = torch.zeros_like(total, dtype=torch.bool)
    elif rule == "always":
        # Always elect E0M3. The mirror of "never", and the control that says whether the type block
        # is doing anything at all: if a variant ties this row, the E2M1 branch is never used and
        # the format has collapsed to plain INT4 with an NVFP4 scale.
        elect = torch.ones_like(total, dtype=torch.bool)
    elif rule == "argmin":
        elect = total > 0
    elif rule == "dominance":
        elect = (gain >= 0).all(dim=(-1, -2)) & (total > 0)
    elif rule == "margin":
        num_block = gain.shape[-2] * gain.shape[-1]
        mean = total / num_block
        std  = gain.flatten(start_dim=1).std(dim=-1, unbiased=False)
        # margin * standard error of the mean; with one block std is 0 and this is just total > 0
        elect = mean > margin * std / (num_block ** 0.5)
    elif rule == "relmargin":
        assert ref is not None, '"relmargin" needs the per-scale-block E2M1 loss as `ref`.'
        # An all-zero scale block has gain 0 and ref 0; eps keeps it at a relative gain of 0
        # instead of 0/0, so it votes neutrally rather than poisoning the mean.
        rel       = (gain / ref.clamp(min=eps)).flatten(start_dim=1)
        num_block = rel.shape[-1]
        elect     = rel.mean(dim=-1) > margin * rel.std(dim=-1, unbiased=False) / (num_block ** 0.5)
        elect     = elect & (total > 0)
    elif rule == "tol":
        assert ref is not None, '"tol" needs the per-scale-block E2M1 loss as `ref`.'
        worst_harm = (-gain / ref.clamp(min=eps)).amax(dim=(-1, -2))
        elect      = (total > 0) & (worst_harm <= margin)
    elif rule == "harm":
        won  = gain.clamp(min=0).sum(dim=(-1, -2))
        lost = (-gain).clamp(min=0).sum(dim=(-1, -2))
        elect = won > margin * lost
    elif rule == "vote":
        share = (gain > 0).flatten(start_dim=1).to(gain.dtype).mean(dim=-1)
        elect = (share > margin) & (total > 0)
    else:
        raise ValueError(f"Unsupported election rule \"{rule}\". Expected one of {ELECT_RULES}.")

    return elect[:, None, None]


PERMUTE_MODES = ("none", "rows", "cocluster", "colchunk", "coclrows")

# Rotation of the reduction dimension by a normalized Walsh-Hadamard matrix, applied in
# non-overlapping chunks of `rotate_size` columns.
#
#   "none" -- no rotation.
#   "all"  -- every chunk is rotated.
#   "col"  -- each chunk decides for itself, by the same error criterion everything else here uses.
#   "outlier" -- each chunk decides for itself on a DISTRIBUTIONAL criterion instead of an error one:
#             rotate only if the chunk comes out CLEAN, i.e. if the post-rotation block max / block
#             rms is below `rotate_outlier_max`. The point of rotation is to spread a block's
#             outlier over the block; if outliers survive the rotation, the rotation has not done
#             its job and there is no reason to pay for it. A 16-sample Gaussian block has
#             max/rms ~2.0, so a threshold slightly above 2 accepts "rotation made this Gaussian"
#             and rejects "outliers are still here".
#
# WHY THIS IS ALLOWED, AND WHY THE DECISION CANNOT BE PER SCALE BLOCK.
# For Y = X W^T with an orthogonal H applied to a chunk of the reduction dimension,
# (X H) (W H)^T = X H H^T W^T = X W^T, so rotating BOTH operands is exact. The activation side
# rotates a chunk of columns for every token at once, so all output rows of W must agree on whether
# that chunk is rotated. The decision is therefore per COLUMN CHUNK -- one bit per `rotate_size`
# input channels, shared down the whole tensor -- and not per scale block, which would let different
# rows disagree and silently compute a different GEMM.
#
# In fake quantization the whole thing collapses to: rotate, quantize, rotate back. That is exact
# for W4A16, where the activation is not quantized and its rotation is undone in full precision.
# For W4A4 it is exact only for "all", where both operands rotate everything and no pattern has to
# be communicated; "col" would need the weight-derived bit vector plumbed into `quant_act`.
ROTATE_MODES = ("none", "all", "col", "outlier")


@torch.no_grad()
def _hadamard(n: int, dtype, device):
    """Normalized n x n Walsh-Hadamard matrix (n a power of two), so that H @ H.T == I."""
    assert n > 0 and (n & (n - 1)) == 0, f"Hadamard size must be a power of two, got {n}."
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    return (h / (n ** 0.5)).to(dtype)


@torch.no_grad()
def _rotate_chunks(x, size: int, transpose: bool=False):
    """
        Rotate the last dimension of a 2-D tensor in non-overlapping chunks of `size` columns.
        `transpose=True` applies H^T, which undoes it exactly.
    """
    num_row, num_col = x.shape
    assert num_col % size == 0, \
        f"The reduction dimension {num_col} must be divisible by the rotation size {size}."
    h = _hadamard(size, x.dtype, x.device)
    return (x.view(num_row, -1, size) @ (h.t() if transpose else h)).view(num_row, num_col)


@torch.no_grad()
def row_preference(w_scaled, groupsize: int, metric: str, clip: str):
    """
        Per row, how much that row as a whole prefers E0M3 over E2M1:

            pref_i = sum over the row's scale blocks of ( loss_E2M1(b) - loss_E0M3(b) )

        computed at the 1x16 granularity, i.e. with each scale block free to pick its own grid and
        its own clip ratio -- the same quantities the type-block election later sums, just grouped by
        row instead of by tile. `w_scaled` is (M, K) in the globally scaled domain.

        This is a property of the WEIGHTS ALONE. No activations, no calibration set.
    """
    E2M1_MAX, E0M3_MAX = 6.0, 7.0
    FP8_SCALE_MAX, FP8_SCALE_MIN = 448.0, 2 ** (-9)

    blocks    = w_scaled.reshape(w_scaled.shape[0], -1, groupsize)
    block_max = blocks.abs().amax(dim=-1, keepdim=True)
    alphas    = CLIP_PRESETS[clip]

    def best(quant_fn, grid_max, alpha_list):
        best_err = None
        for alpha in alpha_list:
            scale = (block_max * (alpha / grid_max)).clamp(
                max=FP8_SCALE_MAX, min=FP8_SCALE_MIN
            ).to(torch.float8_e4m3fn).to(blocks.dtype)
            err = _selection_loss(blocks, quant_fn(blocks, scale), metric)
            best_err = err if best_err is None else torch.minimum(best_err, err)
        return best_err

    err_e2m1 = best(_quant_e2m1, E2M1_MAX, alphas["e2m1"])
    err_e0m3 = best(_quant_e0m3, E0M3_MAX, alphas["e0m3"])
    return (err_e2m1 - err_e0m3).sum(dim=(-1, -2))


@torch.no_grad()
def quant_mix_4_6(
    w_fp,
    n_bits: int=4,
    groupsize: Optional[int]=None,
    type_block=(1, 16),
    metric: str="mse",
    importance=None,
    elect: str="argmin",
    margin: float=0.0,
    clip: str="base",
    clip_min_gain: float=0.0,
    alpha_min_gain: float=0.0,
    permute: str="none",
    rotate: str="none",
    rotate_size: int=16,
    rotate_min_gain: float=0.0,
    rotate_outlier_max: float=2.1,
    peak_veto: float=0.0,
    imp_alpha: bool=True,
    imp_elect: bool=True,
    imp_gran: int=0,
    is_act: bool=False,
):
    """
        MixFP4 "4over6" variant (CPU simulation / fake quantization).

        Same two block granularities as `quant_mixfp4` -- a 16-element NVFP4 scale block carrying an
        E4M3 scale, and a configurable type block carrying ONE element data type -- with one extra
        degree of freedom underneath: the block scale of each scale block is searched over a small
        set of CLIP RATIOS alpha (see `CLIP_PRESETS`),

            block_scale = alpha * block_max / grid_max,

        independently for the E2M1 and the E0M3 candidate. The default preset `base` is
        alpha in {1, 1.5} for E2M1 -- alpha=1.5 maps the block maximum to code 4 instead of code 6,
        which is FourOverSix (https://arxiv.org/abs/2512.02010) -- and alpha=1 for E0M3.

        Choosing alpha only changes the VALUE stored in the existing ue4m3 scale field -- the decoder
        multiplies code by scale either way -- so it costs no metadata and, unlike the E2M1/E0M3
        choice, does NOT have to be uniform across a type block. It is therefore selected per SCALE
        block, independently inside each type block.

        Selection:
          * per scale block: the best alpha for E2M1, and the best alpha for E0M3 (both free)
          * per type block:  E2M1 (using those per-scale-block choices) vs E0M3, under `metric`
                             and the election rule `elect`
    """
    E2M1_MAX      = 6.0
    E0M3_MAX      = 7.0
    FP8_SCALE_MAX = 448.0
    FP8_SCALE_MIN = 2**(-9)

    groupsize     = 16 if groupsize is None else groupsize
    assert groupsize == 16, \
        f'MixFP4 inherits the NVFP4 scale-block size, which must be 16, but got {groupsize}.'
    assert _parse_metric(metric)[0] in ("lp", "corr") or metric in SELECT_METRICS, \
        f'Unsupported selection metric "{metric}". Expected one of {SELECT_METRICS}, ' \
        f'"l<p>" or "corr<r>".'
    assert elect in ELECT_RULES, \
        f'Unsupported election rule "{elect}". Expected one of {ELECT_RULES}.'
    assert clip in CLIP_PRESETS, \
        f'Unsupported clip preset "{clip}". Expected one of {tuple(CLIP_PRESETS)}.'
    assert permute in PERMUTE_MODES, \
        f'Unsupported permute mode "{permute}". Expected one of {PERMUTE_MODES}.'
    assert rotate in ROTATE_MODES, \
        f'Unsupported rotate mode "{rotate}". Expected one of {ROTATE_MODES}.'
    block_m, block_k = parse_type_block(type_block)

    #################### Reshape Tensor ####################
    orig_shape = w_fp.shape
    w_fp_new   = w_fp.reshape(-1, orig_shape[-1]).to(torch.float32)
    num_col    = w_fp_new.shape[-1]

    #################### Hadamard Rotation ####################
    # Handled by recursing on the rotated tensor and rotating the result back, which is exactly what
    # a real implementation does: rotate both operands, quantize in the rotated basis, and let the
    # two rotations cancel inside the GEMM. See ROTATE_MODES for why the choice is per column chunk.
    if rotate != "none" and num_col >= rotate_size:
        inner = dict(n_bits=n_bits, groupsize=groupsize, type_block=type_block, metric=metric,
                     importance=importance, elect=elect, margin=margin, clip=clip,
                     permute=permute, rotate="none", peak_veto=peak_veto,
                     imp_alpha=imp_alpha, imp_elect=imp_elect, imp_gran=imp_gran,
                     is_act=is_act)
        nchunk = num_col // rotate_size
        rotated = _rotate_chunks(w_fp_new, rotate_size)
        dq_rot  = _rotate_chunks(
            quant_mix_4_6(rotated, **inner).to(torch.float32), rotate_size, transpose=True
        )
        if rotate == "all":
            return dq_rot.view(orig_shape).to(torch.bfloat16)

        if rotate == "outlier":
            # Per column chunk: rotate only where the ROTATED data is clean. Measured on the
            # 16-element scale blocks the quantizer actually uses, averaged over every row of the
            # chunk -- the decision has to be uniform down the rows, see ROTATE_MODES.
            blocks = rotated.reshape(-1, groupsize)
            peak   = blocks.abs().amax(dim=-1)
            rms    = blocks.pow(2).mean(dim=-1).sqrt().clamp(min=1e-12)
            ratio  = (peak / rms).view(w_fp_new.shape[0], nchunk, -1).mean(dim=(0, 2))
            take   = (ratio <= rotate_outlier_max)[None, :, None]
            dq_id  = quant_mix_4_6(w_fp_new, **inner).to(torch.float32)
            dq = torch.where(take,
                             dq_rot.view(-1, nchunk, rotate_size),
                             dq_id.view(-1, nchunk, rotate_size))
            return dq.view(orig_shape).to(torch.bfloat16)

        # "col": each column chunk keeps whichever basis quantizes it better. The rotation is
        # orthogonal, so the squared error of a chunk is the same measured in either basis and the
        # two candidates are directly comparable. Summed over ALL rows, because every row of the
        # tensor has to make the same choice.
        #
        # `rotate_min_gain` is why this is not just "rotate if better". Measured on Llama-2-7B with
        # real activations, rotation cuts the TRUE layer output error by 62% on q_proj and 55% on
        # k_proj, but RAISES it by 73% on v_proj and 3-8% across the MLP -- while weight MSE says
        # every one of them improves. The two groups are separated by HOW MUCH the MSE improves:
        # the layers rotation helps are the ones where it cuts weight MSE by >15%, and the layers it
        # hurts are the ones where the MSE barely moves (<6%). A chunk whose error rotation does not
        # clearly reduce is a chunk where rotation only scrambles the error direction, so requiring
        # a minimum fractional gain keeps the first group and drops the second. This is the same
        # lesson as the election rules: "better" is not enough, it has to be decisively better.
        dq_id = quant_mix_4_6(w_fp_new, **inner).to(torch.float32)
        err   = lambda d: (d - w_fp_new).pow(2).view(-1, nchunk, rotate_size).sum(dim=(0, 2))
        e_rot, e_id = err(dq_rot), err(dq_id)
        take_rot = (e_rot < e_id * (1.0 - rotate_min_gain))[None, :, None]
        dq = torch.where(take_rot,
                         dq_rot.view(-1, nchunk, rotate_size),
                         dq_id.view(-1, nchunk, rotate_size))
        return dq.view(orig_shape).to(torch.bfloat16)
    assert num_col % groupsize == 0, \
        f'The reduction dimension {num_col} must be divisible by the scale-block size {groupsize}.'
    if num_col % block_k != 0:
        assert block_k > num_col, \
            f'The reduction dimension {num_col} must be divisible by the type-block K dimension {block_k}.'
        block_k = num_col

    #################### Global Scale ####################
    global_qmax  = E2M1_MAX * FP8_SCALE_MAX
    global_scale = (w_fp_new.abs().amax() / global_qmax).clamp(min=torch.finfo(torch.float32).tiny)
    w_scaled     = w_fp_new / global_scale

    #################### Row Grouping ####################
    # The reason a coarse type block loses the gain is structural, not statistical: rows that prefer
    # E0M3 and rows that prefer E2M1 are interleaved, so every tile has to overrule a large minority
    # of its scale blocks. Sorting the rows by how strongly they prefer E0M3 puts like with like, so
    # the tiles it then cuts are far more homogeneous and the election has much less to overrule.
    #
    # This is a permutation of the OUTER dimension only (output channels for weights, tokens for
    # activations). It leaves the reduction dimension -- and therefore every dot product -- exactly
    # as it was, so the result is the same GEMM with its output rows/columns reordered. See
    # `row_preference` for how the key is computed and CLAUDE.md for where the permutation is
    # actually free to absorb.
    perm, col_perm = None, None
    if permute == "rows" and block_m > 1:
        perm     = row_preference(w_scaled, groupsize, metric, clip).argsort(descending=True)
        w_scaled = w_scaled[perm]
    elif permute in ("cocluster", "colchunk", "coclrows"):
        # Balanced co-clustering of the tag grid. See quantize/reorder.py and
        # results/reorder/ALGORITHM.md. Columns move only in whole 16-element chunks, so the scale
        # blocks -- and therefore the tag grid the search optimizes -- are exactly invariant.
        from .reorder import expand_chunk_perm, scale_block_gain, search_permutation
        axes  = {"cocluster": "both", "colchunk": "cols", "coclrows": "rows"}[permute]
        gain  = scale_block_gain(w_scaled, groupsize, metric, clip, importance=importance)
        found = search_permutation(gain, block_m, block_k, groupsize, rule=elect, margin=margin,
                                   axes=axes)
        if axes != "cols" and block_m > 1:
            # the search runs on CPU, so the permutations come back on CPU; index_copy_ later
            # undoes them against a tensor that may live on the GPU
            perm     = found["row_perm"].to(w_scaled.device)
            w_scaled = w_scaled[perm]
        if axes != "rows":
            col_perm = expand_chunk_perm(found["chunk_perm"], groupsize).to(w_scaled.device)
            w_scaled = w_scaled[:, col_perm]
            if importance is not None:
                importance = importance.reshape(-1)[col_perm.cpu()]

    w_tiled, meta = _tile_type_blocks(w_scaled, block_m, block_k, groupsize)
    block_max     = w_tiled.abs().amax(dim=-1, keepdim=True)

    # Per-input-channel importance, tiled the same way so it lines up element-for-element.
    imp_tiled = None
    if importance is not None:
        imp = importance.to(w_tiled.dtype).to(w_tiled.device).reshape(1, -1)
        assert imp.shape[-1] == num_col, \
            f'importance has {imp.shape[-1]} entries but the reduction dimension is {num_col}.'
        imp_tiled, _ = _tile_type_blocks(
            imp.expand(w_fp_new.shape[0], -1), block_m, block_k, groupsize
        )
        if imp_gran:
            # Coarsen the importance along K: replace each element's own E[x_j^2] by the mean over
            # a run of `imp_gran` consecutive channels. This asks how much of `hess` comes from the
            # per-element detail as opposed to the coarse envelope.
            #
            # Two values are meaningful, and the second is a control with a PROVABLE answer:
            #   imp_gran == groupsize -- one weight per 1x16 scale block. The alpha search compares
            #     candidates within a block, so a per-block constant cancels there and this leaves
            #     ONLY the election weighted -- but weighted by the block-mean, not per element.
            #   imp_gran == block_k   -- one weight per type block. Every election rule in
            #     `_elect_e0m3` is scale-invariant (argmin/harm/vote compare quantities that are
            #     homogeneous of the same degree, margin is a ratio of degree-1 quantities), and the
            #     alpha search is too, so a per-tile constant is an EXACT no-op: it must reproduce
            #     the unweighted run bit for bit. That makes it a validation point, not a proposal.
            assert block_k % imp_gran == 0, \
                f'imp_gran {imp_gran} must divide the type-block K dimension {block_k}.'
            n_tile = imp_tiled.shape[0]
            # (tile, block, gs) is row-major over (row_in_tile, k_chunk), so flattening the last two
            # axes back to a contiguous K run of length block_k is exact.
            flat = imp_tiled.reshape(n_tile, block_m, block_k)
            flat = flat.reshape(n_tile, block_m, block_k // imp_gran, imp_gran)
            flat = flat.mean(dim=-1, keepdim=True).expand(-1, -1, -1, imp_gran)
            imp_tiled = flat.reshape(n_tile, block_m, block_k).reshape(
                n_tile, block_m * block_k // groupsize, groupsize
            ).contiguous()

    ####### Per-scale-block search over the clip ratio, once per grid #######
    # `clip_min_gain` splits the candidates into the SAFE ones (alpha >= 1, which never clip and only
    # move the bulk onto a different part of the grid -- 4over6 is one of these) and the CLIPPING
    # ones (alpha < 1). A clipping candidate is taken only when it beats the best safe candidate by
    # at least this fraction. Clipping measured as a loss of +0.006 to +0.033 wikitext in round 1
    # while lowering the very error it selects on, and its MSE gains there were only 3-5% -- squarely
    # in the small-gain regime where weight error and layer output error decouple. Requiring a large
    # gain is the same fix that turned rotation from +0.095 into a small win.
    # Importance can be applied to the two decisions INDEPENDENTLY. `hess` weights one loss that
    # drives both the alpha search and the type election, but measurement shows the two effects have
    # different signs on the same model: on Qwen3-4B, importance-weighting the alpha choice costs
    # +0.29 wikitext while importance-weighting the type election gains -1.05. So `imp_alpha` and
    # `imp_elect` select which decision sees it.
    #
    # `sel` is the loss the alpha search minimizes; `out` is the loss handed to the election. When
    # the two scopes differ they are computed from the same dequantized candidate, so the alpha
    # picked and the error reported for it stay consistent.
    imp_a = imp_tiled if imp_alpha else None
    imp_e = imp_tiled if imp_elect else None

    def _best_over_alphas(quant_fn, grid_max, alphas):
        def search(alpha_list):
            best_dq, best_err, best_out = None, None, None
            for alpha in alpha_list:
                block_scale = (block_max * (alpha / grid_max)).clamp(
                    max=FP8_SCALE_MAX,
                    min=FP8_SCALE_MIN
                ).to(torch.float8_e4m3fn).to(w_tiled.dtype)
                dq  = quant_fn(w_tiled, block_scale)
                err = _selection_loss(w_tiled, dq, metric, imp_a)
                out = err if imp_a is imp_e else _selection_loss(w_tiled, dq, metric, imp_e)

                if best_dq is None:
                    best_dq, best_err, best_out = dq, err, out
                else:
                    better   = err < best_err
                    best_dq  = torch.where(better, dq, best_dq)
                    best_out = torch.where(better, out, best_out)
                    best_err = torch.where(better, err, best_err)
            return best_dq, best_err, best_out

        safe     = [a for a in alphas if a >= 1.0] or [1.0]
        clipping = [a for a in alphas if a < 1.0]
        dq, err, out = search(safe)

        # `alpha_min_gain` applies the decisive-margin principle to the SCALE SEARCH ITSELF, which
        # is otherwise the last plain argmin left in this quantizer -- exactly the pattern that
        # loses for the element-type election, for rotation and for clipping. A candidate alpha != 1
        # is taken only when it beats alpha = 1 (plain NVFP4, the block maximum on the top code) by
        # at least this fraction. alpha_min_gain = 0 is the ordinary argmin search.
        if alpha_min_gain > 0.0 and len(safe) > 1:
            dq_1, err_1, out_1 = search([1.0])
            take = err < err_1 * (1.0 - alpha_min_gain)
            dq, err, out = (torch.where(take, dq, dq_1), torch.where(take, err, err_1),
                            torch.where(take, out, out_1))

        if not clipping:
            return dq, out

        dq_c, err_c, out_c = search(clipping)
        take = err_c < err * (1.0 - clip_min_gain)
        return torch.where(take, dq_c, dq), torch.where(take, out_c, out)

    alphas = CLIP_PRESETS[clip]
    w_dq_e2m1, error_e2m1 = _best_over_alphas(_quant_e2m1, E2M1_MAX, alphas["e2m1"])
    w_dq_e0m3, error_e0m3 = _best_over_alphas(_quant_e0m3, E0M3_MAX, alphas["e0m3"])

    ############### Per-Type-Block Data Type Selection ###############
    # Sum the per-scale-block errors over every scale block of the same type block
    # gain_b > 0 means E0M3 is the better element type for scale block b
    select_e0m3 = _elect_e0m3(
        error_e2m1 - error_e0m3, rule=elect, margin=margin, ref=error_e2m1
    )

    # PEAKEDNESS VETO -- an on-the-fly rule needing no search, no permutation and no calibration.
    #
    # Measured on real weights, the E0M3/E2M1 preference is governed by block peakedness: the rank
    # correlation between the E0M3 gain and block_max/block_rms is -0.59 on both Llama-3.1-8B and
    # Qwen3-4B, and the E0M3-preferring population averages max/rms 1.98-2.01 against 2.33-2.36 for
    # E2M1. A 16-sample Gaussian block sits at 2.0, so E0M3 is precisely the "this block has no
    # outlier" grid -- its uniform spacing suits a flat block, while E2M1's log spacing is coarse at
    # the top and absorbs an outlier cheaply.
    #
    # So refuse E0M3 for a tile whose blocks are collectively peaked. `peak_veto` is the threshold
    # on the tile's MEAN max/rms; "any block above it" is unusable at a 16x64 activation tile, which
    # holds 64 scale blocks and would veto almost everything.
    #
    # Both statistics are already on hand: block_max is computed for the scale, and the mean square
    # is one extra reduction over data already in registers.
    if peak_veto > 0.0:
        block_rms = w_tiled.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)
        peak      = (block_max / block_rms).mean(dim=1, keepdim=True)      # (n_tile, 1, 1)
        select_e0m3 = select_e0m3 & (peak <= peak_veto)

    w_dq = torch.where(
        select_e0m3,
        w_dq_e0m3,
        w_dq_e2m1,
    )
    w_dq = _untile_type_blocks(w_dq, block_m, block_k, meta)
    if col_perm is not None:
        # undo the column-chunk permutation; in a real deployment this is not undone at all, the
        # matching permutation of the activation channels is absorbed upstream instead
        w_dq = w_dq.index_copy_(1, col_perm, w_dq.clone())
    if perm is not None:
        # undo the row sort: row perm[i] of the original tensor is row i of the sorted one
        w_dq = torch.empty_like(w_dq).index_copy_(0, perm, w_dq)
    w_dq = w_dq * global_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_nvfp4_razer_e3m3(w_fp, n_bits: int=4, groupsize: Optional[int]=None, outlier: float=8.0):
    """
        NVFP4-RaZeR quantization.
    """

    inlier  = 5.0
    datatype_list = [
        [inlier, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        [-inlier, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        [outlier, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        [-outlier, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    ]

    #################### Reshape Tensor ####################
    orig_shape   = w_fp.shape 
    w_fp_new     = w_fp.view(-1, groupsize).to(torch.float32)
    num_group    = w_fp_new.shape[0]

    #################### Global Scale ####################
    global_qmax  = 6.0 * 28
    global_scale = w_fp_new.abs().amax() / global_qmax
    #################### Block Maximum ####################
    w_scaled     = w_fp_new / global_scale
    block_max    = w_scaled.abs().amax(dim=-1, keepdim=True)

    ############### Optimal Data Type Search ###############
    w_q          = torch.zeros_like(w_fp_new)
    block_scale  = torch.zeros(num_group, 1, dtype=w_fp_new.dtype, device=w_fp_new.device)
    quant_error  = torch.full([num_group], float('inf'), dtype=w_fp_new.dtype, device=w_fp_new.device)
    # Iterate through data types
    for quant_value in datatype_list:
        quant_value     = sorted(quant_value)
        mid_value       = [(quant_value[i] + quant_value[i + 1]) / 2 for i in range(len(quant_value) - 1)]
        qmax_tmp        = abs(max(quant_value, key=abs))
        block_scale_tmp = (block_max / qmax_tmp).clamp(
            max=28,
            min=2**(-5)
        ).to(torch.float8_e4m3fn).to(w_fp_new.dtype)
        w_scaled_tmp    = w_scaled / block_scale_tmp

        # Fake Quantization
        w_q_tmp = torch.zeros_like(w_scaled_tmp)
        for i, data in enumerate(quant_value):
            if i == 0:
                w_q_tmp += torch.where(w_scaled_tmp <= mid_value[i], data, 0)
            elif i == len(quant_value) - 1:
                w_q_tmp += torch.where(w_scaled_tmp > mid_value[i - 1], data, 0)
            else:
                w_q_tmp += torch.where((mid_value[i - 1] < w_scaled_tmp) & (w_scaled_tmp <= mid_value[i]), data, 0)

        # Quantization Error
        quant_error_tmp = (w_q_tmp*block_scale_tmp - w_scaled).pow(2).mean(dim=-1)
        # Update Data Type if Smaller Quantization Error
        mask_update               = torch.lt(quant_error_tmp, quant_error)
        w_q[mask_update]          = w_q_tmp[mask_update]
        block_scale[mask_update]  = block_scale_tmp[mask_update]
        quant_error[mask_update]  = quant_error_tmp[mask_update]

    w_dq = w_q * block_scale * global_scale
    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_nvfp4_razer_e4m3(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        NVFP4-RaZeR quantization.
    """
    FP4_MAX      = 6.0
    FP4_MAN_BITS = 1

    orig_shape     = w_fp.shape 
    w_fp_new       = w_fp.reshape(-1, groupsize).to(torch.float32)
    num_group      = w_fp_new.shape[0]

    inlier         = 5.0
    global_qmax    = FP4_MAX * 448
    global_scale   = w_fp_new.abs().amax() / global_qmax

    ############### Block Scale Quantization ###############
    w_scaled      = w_fp_new / global_scale
    block_max     = w_scaled.abs().amax(dim=-1, keepdim=True)
    block_scale_q = (block_max / FP4_MAX).clamp(
        max=448,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    w_scaled      = w_scaled / block_scale_q

    #################### FP4 Quantization ####################
    private_exp   = torch.floor(
        torch.log2(
            torch.abs(w_scaled) + (w_scaled == 0).type(w_scaled.dtype)
        )
    )
    private_exp   = private_exp.clamp(min=0)
    w_m           = w_scaled / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m           = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q_fp4       = w_m * (2**private_exp) / (2**FP4_MAN_BITS)

    ########## Search for the Optimal RaZeR-FP4 Data Type ##########
    error     = torch.full([num_group], float('inf'), dtype=w_fp_new.dtype, device=w_fp_new.device)
    w_q_razer = torch.zeros_like(w_fp_new)
    for special_value in [-inlier, inlier]:
        # Handle special value
        w_q_razer_tmp = torch.where(
            (w_scaled - w_q_fp4).abs() < (w_scaled - special_value).abs(),
            w_q_fp4, special_value
        )
        # Dequantize and calculate error
        quant_error            = (w_q_razer_tmp - block_scale_q).pow(2).mean(-1)
        mask_update            = torch.lt(quant_error, error)
        error[mask_update]     = quant_error[mask_update]
        w_q_razer[mask_update] = w_q_razer_tmp[mask_update]
    ##################################################################

    w_dq = w_q_razer * block_scale_q * global_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


def parse_mix_4_6_dtype(name: str):
    """
        Decode a mix_4_6 data type name into its selection settings.

            mix_4_6[_sqnr|_cossim|_mae|_l<p>][_clip<preset>][_hess]
                   [_perm][_rot<n>|_rotcol<n>]
                   [_dom|_m<z>|_rm<z>|_tol<d>|_h<lambda>|_v<tau>|_e2m1]

        Selection metric   -- "sqnr", "cossim", "mae", "l<p>"; default MSE.
        Clip preset        -- "clip<name>" for any key of CLIP_PRESETS; default "base" (FourOverSix
                              on E2M1, exact-fit E0M3).
        "hess"             -- weights the selection loss by the per-input-channel importance
                              E[x_j^2], turning raw weight error into the diagonal-Hessian estimate
                              of LAYER OUTPUT error. NEEDS CALIBRATION DATA.
        Election rule      -- "dom" (dominance), "m<z>" (margin), "rm<z>" (relative margin),
                              "tol<d>" (dominance with a relative-harm tolerance), "h<lambda>"
                              (harm ratio), "v<tau>" (vote share), "e2m1" (never elect E0M3),
                              "e0m3" (always elect it -- the control for whether the type block is
                              doing anything); default argmin.

        "perm"             -- sort rows by their E0M3 preference before tiling into type blocks,
                              so the tiles are homogeneous and the election overrules far fewer
                              scale blocks. Calibration-free (it reads only the weights).

        "cocl" / "coclcol" / "coclrow"
                           -- the same idea done properly: a balanced CO-CLUSTERING search over both
                              axes at once (`quantize/reorder.py`), maximizing the gain the type
                              blocks actually realize under the election rule in force. Columns move
                              only in whole 16-element chunks, so the scale blocks are untouched.
                              "coclcol" permutes columns only, "coclrow" rows only, "cocl" both.

                              DEPLOYABILITY DIFFERS BY AXIS, see results/reorder/ALGORITHM.md §7.
                              A per-layer column permutation is free for `down_proj` (absorbed into
                              gate/up_proj's rows) and within-head for `o_proj`; for q/k/v/gate/up
                              the column axis is the shared residual dimension and only ONE global
                              permutation is free. So "cocl" applied to every tensor independently
                              is an upper bound, in the same way `1x16` is -- not a deployable
                              configuration. Calibration-free.

        "rot" / "rot<n>"   -- rotate every chunk of n reduction-dimension columns (default 16) by a
                              normalized Hadamard before quantizing, and rotate back after.
        "rotcol" / "rotcol<n>" -- same, but each column chunk decides for itself whether to rotate.

        "rotmin<t>"        -- rotate a column chunk only when rotation cuts its squared error by at
                              least the fraction t. "rotcol" is t = 0, i.e. rotate whenever it helps
                              at all, which measurably is NOT the right rule.

        "clipmin<t>"       -- take a clipping candidate (alpha < 1) only when it beats the best
                              non-clipping candidate by at least the fraction t.

        "amin<t>"          -- take a scale candidate alpha != 1 only when it beats alpha = 1 by at
                              least the fraction t. Applies the decisive-margin principle to the
                              scale search itself, the last plain argmin in this quantizer.

        Returns (metric, elect, margin, use_importance, clip, clip_min_gain, alpha_min_gain,
                 permute, rotate, rotate_size, rotate_min_gain, ..., imp_gran).
    """
    assert name.startswith("mix_4_6"), name
    metric, elect, margin, use_importance, clip = "mse", "argmin", 0.0, False, "base"
    permute, rotate, rotate_size, rotate_min_gain = "none", "none", 16, 0.0
    clip_min_gain, alpha_min_gain, rotate_outlier_max = 0.0, 0.0, 2.1
    peak_veto = 0.0
    imp_alpha, imp_elect = True, True
    imp_gran = 0

    def _num(s):
        return s.replace(".", "", 1).isdigit()

    for part in [p for p in name[len("mix_4_6"):].split("_") if p]:
        if part in ("sqnr", "cossim", "mae"):
            metric = part
        elif _parse_metric(part)[0] in ("lp", "corr"):
            metric = part
        elif part.startswith("clipmin") and part[7:].replace(".", "", 1).isdigit():
            clip_min_gain = float(part[7:])
        elif part.startswith("amin") and part[4:].replace(".", "", 1).isdigit():
            alpha_min_gain = float(part[4:])
        elif part.startswith("clip") and part[len("clip"):] in CLIP_PRESETS:
            clip = part[len("clip"):]
        elif part == "perm":
            permute = "rows"
        elif part == "cocl":
            permute = "cocluster"
        elif part == "coclcol":
            permute = "colchunk"
        elif part == "coclrow":
            permute = "coclrows"
        elif part == "rot":
            rotate = "all"
        elif part == "rotcol":
            rotate = "col"
        elif part.startswith("roto") and part[4:].replace(".", "", 1).isdigit():
            rotate, rotate_outlier_max = "outlier", float(part[4:])
        elif part.startswith("rot") and part[3:].isdigit():
            rotate, rotate_size = "all", int(part[3:])
        elif part.startswith("rotcol") and part[6:].isdigit():
            rotate, rotate_size = "col", int(part[6:])
        elif part.startswith("rotmin") and part[6:].replace(".", "", 1).isdigit():
            rotate, rotate_min_gain = "col", float(part[6:])
        elif part.startswith("pv") and _num(part[2:]):
            peak_veto = float(part[2:])
        elif part.startswith("impg") and part[4:].isdigit():
            # coarsen the importance along K to runs of N channels; impg<block_k> is an exact no-op
            imp_gran = int(part[4:])
        elif part == "hess":
            use_importance = True
        elif part == "hesst":
            # importance on the TYPE election only; alpha still chosen by unweighted error
            use_importance, imp_alpha, imp_elect = True, False, True
        elif part == "hessa":
            # importance on the ALPHA search only; the election still uses unweighted error
            use_importance, imp_alpha, imp_elect = True, True, False
        elif part == "dom":
            elect = "dominance"
        elif part == "e2m1":
            elect = "never"
        elif part == "e0m3":
            elect = "always"
        elif part.startswith("rm") and _num(part[2:]):
            elect, margin = "relmargin", float(part[2:])
        elif part.startswith("tol") and _num(part[3:]):
            elect, margin = "tol", float(part[3:])
        elif part.startswith("h") and _num(part[1:]):
            elect, margin = "harm", float(part[1:])
        elif part.startswith("v") and _num(part[1:]):
            elect, margin = "vote", float(part[1:])
        elif part.startswith("m") and _num(part[1:]):
            elect, margin = "margin", float(part[1:])
        else:
            raise ValueError(f'Unrecognized mix_4_6 data type qualifier "{part}" in "{name}".')
    return (metric, elect, margin, use_importance, clip, clip_min_gain, alpha_min_gain,
            permute, rotate, rotate_size, rotate_min_gain, rotate_outlier_max, peak_veto,
            imp_alpha, imp_elect, imp_gran)


def quant_weight(model, quant_config: QuantConfig, importance=None):
    """
        `importance`: optional {module_name: 1-D tensor of E[x_j^2]} from
        `quantize.importance.collect_importance`. When given, MixFP4 variants weight their selection
        error by it, which measures the LAYER OUTPUT error rather than the raw weight error.
    """
    n_bits       = quant_config.w_bits
    w_groupsize  = quant_config.w_groupsize
    w_dtype      = quant_config.w_dtype.lower()
    w_outlier    = quant_config.w_outlier
    w_type_block = quant_config.w_type_block

    if w_dtype.startswith(("mixfp4", "mix_4_6")):
        block_m, block_k = parse_type_block(w_type_block)
        print(f"Performing LLM weight quantization using Data Type:  {w_dtype}  "
              f"(type block: {block_m}x{block_k})\n")
    else:
        print(f"Performing LLM weight quantization using Data Type:  {w_dtype}\n")

    if (n_bits >= 16) or (w_dtype is None) or (w_dtype in ["fp16", "fp32"]):
        return

    n_bits      = 4
    quant_func  = None
    if (w_dtype == "mxfp4_naive"):
        quant_func = quant_mxfp4_naive
    elif (w_dtype == "mxfp4"):
        quant_func  = quant_mxfp4
    elif (w_dtype == "mxfp4_meta"):
        quant_func  = quant_mxfp4_meta
    elif (w_dtype == "mxif4"):
        quant_func  = quant_mxif4
    elif (w_dtype == "mxfp4_razer"):
        quant_func  = quant_mxfp4_razer
    elif (w_dtype == "mxfp4_razer_new"):
        quant_func  = quant_mxfp4_razer_new
    elif (w_dtype == "nf4"):
        quant_func = quant_nf4  
    elif (w_dtype == "hf4"):
        quant_func = quant_hf4
    elif (w_dtype == "nvfp4"):
        quant_func = quant_nvfp4
    elif (w_dtype == "nvfp4_4over6"):
        quant_func = quant_nvfp4_4over6
    elif (w_dtype == "nvfp4_nover6"):
        quant_func = quant_nvfp4_nover6
    elif (w_dtype == "nvif4"):
        quant_func = quant_nvif4
    elif (w_dtype == "mixfp4"):
        quant_func = partial(quant_mixfp4, type_block=w_type_block)
    elif w_dtype.startswith("mix_4_6"):
        (_metric, _elect, _margin, _use_imp, _clip,
         _clip_g, _alpha_g, _perm, _rot, _rot_n, _rot_g, _rot_o, _pv, _ia, _ie, _ig) = parse_mix_4_6_dtype(w_dtype)
        quant_func = partial(
            quant_mix_4_6, type_block=w_type_block,
            metric=_metric, elect=_elect, margin=_margin, clip=_clip, clip_min_gain=_clip_g, alpha_min_gain=_alpha_g, permute=_perm,
            rotate=_rot, rotate_size=_rot_n, rotate_min_gain=_rot_g, rotate_outlier_max=_rot_o, peak_veto=_pv, imp_alpha=_ia, imp_elect=_ie, imp_gran=_ig,
        )
    elif (w_dtype == "nvfp4_razer_e3m3"):
        quant_func = partial(quant_nvfp4_razer_e3m3, outlier=w_outlier)
    elif (w_dtype == "nvfp4_razer_e4m3"):
        quant_func = quant_nvfp4_razer_e4m3
    else:
        raise ValueError(f"Unsupported Data Type: {w_dtype}")
    
    supports_importance = w_dtype.startswith("mix_4_6") and _use_imp
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and ('head' not in n):
            kwargs = {}
            if supports_importance and importance is not None and n in importance:
                kwargs["importance"] = importance[n].to(m.weight.device)
            m.weight.data = quant_func(
                m.weight.data, n_bits=n_bits, groupsize=w_groupsize, **kwargs
            )


def quant_act(act, quant_config: QuantConfig):
    n_bits       = quant_config.a_bits
    a_groupsize  = quant_config.a_groupsize
    a_dtype      = quant_config.a_dtype.lower()
    a_type_block = quant_config.a_type_block

    if (n_bits >= 16) or (a_dtype in ["fp16", "fp32"]):
        return act

    n_bits      = 4
    quant_func  = None

    if (a_dtype == "mxfp4_naive"):
        quant_func = quant_mxfp4_naive
    elif (a_dtype == "mxfp4"):
        quant_func = quant_mxfp4
    elif (a_dtype == "mxfp4_meta"):
        quant_func  = quant_mxfp4_meta
    elif (a_dtype == "mxif4"):
        quant_func = quant_mxif4
    elif (a_dtype == "mxfp4_razer"):
        quant_func  = partial(quant_mxfp4_razer, is_act=True)
    elif (a_dtype == "mxfp4_razer_new"):
        quant_func  = partial(quant_mxfp4_razer_new, is_act=True)
    elif (a_dtype == "nf4"):
        quant_func = quant_nf4
    elif (a_dtype == "hf4"):
        quant_func = quant_hf4
    elif (a_dtype == "nvfp4"):
        quant_func = quant_nvfp4
    elif (a_dtype == "nvfp4_4over6"):
        quant_func = quant_nvfp4_4over6
    elif (a_dtype == "nvfp4_nover6"):
        quant_func = quant_nvfp4_nover6
    elif (a_dtype == "nvif4"):
        quant_func = quant_nvif4
    elif (a_dtype == "mixfp4"):
        quant_func = partial(quant_mixfp4, type_block=a_type_block, is_act=True)
    elif a_dtype.startswith("mix_4_6"):
        (_metric, _elect, _margin, _, _clip,
         _clip_g, _alpha_g, _perm, _rot, _rot_n, _rot_g, _rot_o, _pv, _ia, _ie, _ig) = parse_mix_4_6_dtype(a_dtype)
        quant_func = partial(
            quant_mix_4_6, type_block=a_type_block, is_act=True,
            metric=_metric, elect=_elect, margin=_margin, clip=_clip, clip_min_gain=_clip_g, alpha_min_gain=_alpha_g, permute=_perm,
            rotate=_rot, rotate_size=_rot_n, rotate_min_gain=_rot_g, rotate_outlier_max=_rot_o, peak_veto=_pv, imp_alpha=_ia, imp_elect=_ie, imp_gran=_ig,
        )
    elif (a_dtype == "nvfp4_razer_e4m3"):
        quant_func = quant_nvfp4_razer_e4m3
    else:
        raise ValueError(f"Unsupported Data Type: {a_dtype}")

    # Fixed, calibration-derived channel permutation (see QuantConfig.a_perm). Permute the channel
    # axis, quantize, permute back -- which in a deployment is not undone at all, the matching
    # permutation being absorbed into the weight columns of the same axis. Because it moves whole
    # 16-channel chunks the scale blocks are unchanged, so the weight side is bit-identical and
    # permuting the activation alone reproduces the real transform exactly.
    perm = getattr(quant_config, "a_perm", None)
    if perm is not None:
        p = perm.get(act.shape[-1])
        if p is not None:
            p = p.to(act.device)
            inv = torch.empty_like(p)
            inv[p] = torch.arange(p.numel(), device=p.device)
            out = quant_func(act.index_select(-1, p), n_bits=n_bits, groupsize=a_groupsize)
            return out.index_select(-1, inv)

    return quant_func(act, n_bits=n_bits, groupsize=a_groupsize)
