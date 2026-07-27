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
    # E0M3 only -- the branch with no free normalization today
    "e0":    {"e2m1": (1.0, 1.5),                          "e0m3": (1.0, 0.9)},
    "e0x":   {"e2m1": (1.0, 1.5),                          "e0m3": (1.0, 0.9, 0.8)},
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


ELECT_RULES = ("argmin", "dominance", "margin", "never",
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


PERMUTE_MODES = ("none", "rows")


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
    permute: str="none",
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
    block_m, block_k = parse_type_block(type_block)

    #################### Reshape Tensor ####################
    orig_shape = w_fp.shape
    w_fp_new   = w_fp.reshape(-1, orig_shape[-1]).to(torch.float32)
    num_col    = w_fp_new.shape[-1]
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
    perm = None
    if permute == "rows" and block_m > 1:
        perm     = row_preference(w_scaled, groupsize, metric, clip).argsort(descending=True)
        w_scaled = w_scaled[perm]

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

    ####### Per-scale-block search over the clip ratio, once per grid #######
    def _best_over_alphas(quant_fn, grid_max, alphas):
        best_dq, best_err = None, None
        for alpha in alphas:
            block_scale = (block_max * (alpha / grid_max)).clamp(
                max=FP8_SCALE_MAX,
                min=FP8_SCALE_MIN
            ).to(torch.float8_e4m3fn).to(w_tiled.dtype)
            dq  = quant_fn(w_tiled, block_scale)
            err = _selection_loss(w_tiled, dq, metric, imp_tiled)

            if best_dq is None:
                best_dq, best_err = dq, err
            else:
                better   = err < best_err
                best_dq  = torch.where(better, dq, best_dq)
                best_err = torch.where(better, err, best_err)
        return best_dq, best_err

    alphas = CLIP_PRESETS[clip]
    w_dq_e2m1, error_e2m1 = _best_over_alphas(_quant_e2m1, E2M1_MAX, alphas["e2m1"])
    w_dq_e0m3, error_e0m3 = _best_over_alphas(_quant_e0m3, E0M3_MAX, alphas["e0m3"])

    ############### Per-Type-Block Data Type Selection ###############
    # Sum the per-scale-block errors over every scale block of the same type block
    # gain_b > 0 means E0M3 is the better element type for scale block b
    select_e0m3 = _elect_e0m3(
        error_e2m1 - error_e0m3, rule=elect, margin=margin, ref=error_e2m1
    )

    w_dq = torch.where(
        select_e0m3,
        w_dq_e0m3,
        w_dq_e2m1,
    )
    w_dq = _untile_type_blocks(w_dq, block_m, block_k, meta)
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
                   [_perm][_dom|_m<z>|_rm<z>|_tol<d>|_h<lambda>|_v<tau>|_e2m1]

        Selection metric   -- "sqnr", "cossim", "mae", "l<p>"; default MSE.
        Clip preset        -- "clip<name>" for any key of CLIP_PRESETS; default "base" (FourOverSix
                              on E2M1, exact-fit E0M3).
        "hess"             -- weights the selection loss by the per-input-channel importance
                              E[x_j^2], turning raw weight error into the diagonal-Hessian estimate
                              of LAYER OUTPUT error. NEEDS CALIBRATION DATA.
        Election rule      -- "dom" (dominance), "m<z>" (margin), "rm<z>" (relative margin),
                              "tol<d>" (dominance with a relative-harm tolerance), "h<lambda>"
                              (harm ratio), "v<tau>" (vote share), "e2m1" (never elect E0M3);
                              default argmin.

        "perm"             -- sort rows by their E0M3 preference before tiling into type blocks,
                              so the tiles are homogeneous and the election overrules far fewer
                              scale blocks. Calibration-free (it reads only the weights).

        Returns (metric, elect, margin, use_importance, clip, permute).
    """
    assert name.startswith("mix_4_6"), name
    metric, elect, margin, use_importance, clip = "mse", "argmin", 0.0, False, "base"
    permute = "none"

    def _num(s):
        return s.replace(".", "", 1).isdigit()

    for part in [p for p in name[len("mix_4_6"):].split("_") if p]:
        if part in ("sqnr", "cossim", "mae"):
            metric = part
        elif _parse_metric(part)[0] in ("lp", "corr"):
            metric = part
        elif part.startswith("clip") and part[len("clip"):] in CLIP_PRESETS:
            clip = part[len("clip"):]
        elif part == "perm":
            permute = "rows"
        elif part == "hess":
            use_importance = True
        elif part == "dom":
            elect = "dominance"
        elif part == "e2m1":
            elect = "never"
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
    return metric, elect, margin, use_importance, clip, permute


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
    elif (w_dtype == "nvif4"):
        quant_func = quant_nvif4
    elif (w_dtype == "mixfp4"):
        quant_func = partial(quant_mixfp4, type_block=w_type_block)
    elif w_dtype.startswith("mix_4_6"):
        _metric, _elect, _margin, _use_imp, _clip, _perm = parse_mix_4_6_dtype(w_dtype)
        quant_func = partial(
            quant_mix_4_6, type_block=w_type_block,
            metric=_metric, elect=_elect, margin=_margin, clip=_clip, permute=_perm,
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
    elif (a_dtype == "nvif4"):
        quant_func = quant_nvif4
    elif (a_dtype == "mixfp4"):
        quant_func = partial(quant_mixfp4, type_block=a_type_block, is_act=True)
    elif a_dtype.startswith("mix_4_6"):
        _metric, _elect, _margin, _, _clip, _perm = parse_mix_4_6_dtype(a_dtype)
        quant_func = partial(
            quant_mix_4_6, type_block=a_type_block, is_act=True,
            metric=_metric, elect=_elect, margin=_margin, clip=_clip, permute=_perm,
        )
    elif (a_dtype == "nvfp4_razer_e4m3"):
        quant_func = quant_nvfp4_razer_e4m3
    else:
        raise ValueError(f"Unsupported Data Type: {a_dtype}")
    
    return quant_func(act, n_bits=n_bits, groupsize=a_groupsize)
