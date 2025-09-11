# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ATen operation registrations for mycelya device backend.

This module contains only the PyTorch library registrations that connect
the ATen operations to their mycelya implementations. The actual operation
implementations are organized in separate modules by functionality.
"""

import torch

from .copy import _copy_from
from .dispatch import _remote_kernel_fallback
from .scalar import _equal, _local_scalar_dense

# Register the fallback kernel for all unspecified operations
_mycelya_lib = torch.library.Library("_", "IMPL")
_mycelya_lib.fallback(_remote_kernel_fallback, dispatch_key="PrivateUse1")

# Register specific ATen operation implementations
_mycelya_lib_aten = torch.library.Library("aten", "IMPL")
_mycelya_lib_aten.impl("_copy_from", _copy_from, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl(
    "_local_scalar_dense", _local_scalar_dense, dispatch_key="PrivateUse1"
)
_mycelya_lib_aten.impl("equal", _equal, dispatch_key="PrivateUse1")

# Core ATen operators with automatic wrappers around _remote_kernel_fallback
# Total: 431 operators (base + inplace + out variants)
# All operations sorted alphabetically

def __adaptive_avg_pool2d_wrapper(*args, **kwargs):
    """Wrapper for aten._adaptive_avg_pool2d"""
    return _remote_kernel_fallback(torch.ops.aten._adaptive_avg_pool2d, *args, **kwargs)
def __adaptive_avg_pool2d_out_wrapper(*args, **kwargs):
    """Wrapper for aten._adaptive_avg_pool2d.out"""
    return _remote_kernel_fallback(torch.ops.aten._adaptive_avg_pool2d.out, *args, **kwargs)
def __adaptive_avg_pool2d_backward_wrapper(*args, **kwargs):
    """Wrapper for aten._adaptive_avg_pool2d_backward"""
    return _remote_kernel_fallback(torch.ops.aten._adaptive_avg_pool2d_backward, *args, **kwargs)
def __adaptive_avg_pool2d_backward_out_wrapper(*args, **kwargs):
    """Wrapper for aten._adaptive_avg_pool2d_backward.out"""
    return _remote_kernel_fallback(torch.ops.aten._adaptive_avg_pool2d_backward.out, *args, **kwargs)
def __adaptive_avg_pool3d_wrapper(*args, **kwargs):
    """Wrapper for aten._adaptive_avg_pool3d"""
    return _remote_kernel_fallback(torch.ops.aten._adaptive_avg_pool3d, *args, **kwargs)
def __adaptive_avg_pool3d_out_wrapper(*args, **kwargs):
    """Wrapper for aten._adaptive_avg_pool3d.out"""
    return _remote_kernel_fallback(torch.ops.aten._adaptive_avg_pool3d.out, *args, **kwargs)
def __cdist_forward_wrapper(*args, **kwargs):
    """Wrapper for aten._cdist_forward"""
    return _remote_kernel_fallback(torch.ops.aten._cdist_forward, *args, **kwargs)
def __cdist_forward_out_wrapper(*args, **kwargs):
    """Wrapper for aten._cdist_forward.out"""
    return _remote_kernel_fallback(torch.ops.aten._cdist_forward.out, *args, **kwargs)
def __embedding_bag_wrapper(*args, **kwargs):
    """Wrapper for aten._embedding_bag"""
    return _remote_kernel_fallback(torch.ops.aten._embedding_bag, *args, **kwargs)
def __embedding_bag_out_wrapper(*args, **kwargs):
    """Wrapper for aten._embedding_bag.out"""
    return _remote_kernel_fallback(torch.ops.aten._embedding_bag.out, *args, **kwargs)
def __fft_r2c_wrapper(*args, **kwargs):
    """Wrapper for aten._fft_r2c"""
    return _remote_kernel_fallback(torch.ops.aten._fft_r2c, *args, **kwargs)
def __fft_r2c_out_wrapper(*args, **kwargs):
    """Wrapper for aten._fft_r2c.out"""
    return _remote_kernel_fallback(torch.ops.aten._fft_r2c.out, *args, **kwargs)
def __log_softmax_wrapper(*args, **kwargs):
    """Wrapper for aten._log_softmax"""
    return _remote_kernel_fallback(torch.ops.aten._log_softmax, *args, **kwargs)
def __log_softmax_out_wrapper(*args, **kwargs):
    """Wrapper for aten._log_softmax.out"""
    return _remote_kernel_fallback(torch.ops.aten._log_softmax.out, *args, **kwargs)
def __native_batch_norm_legit_wrapper(*args, **kwargs):
    """Wrapper for aten._native_batch_norm_legit"""
    return _remote_kernel_fallback(torch.ops.aten._native_batch_norm_legit, *args, **kwargs)
def __native_batch_norm_legit_no_stats_wrapper(*args, **kwargs):
    """Wrapper for aten._native_batch_norm_legit.no_stats"""
    return _remote_kernel_fallback(torch.ops.aten._native_batch_norm_legit.no_stats, *args, **kwargs)
def __native_batch_norm_legit_no_stats_out_wrapper(*args, **kwargs):
    """Wrapper for aten._native_batch_norm_legit.no_stats_out"""
    return _remote_kernel_fallback(torch.ops.aten._native_batch_norm_legit.no_stats_out, *args, **kwargs)
def __native_batch_norm_legit_out_wrapper(*args, **kwargs):
    """Wrapper for aten._native_batch_norm_legit.out"""
    return _remote_kernel_fallback(torch.ops.aten._native_batch_norm_legit.out, *args, **kwargs)
def __native_batch_norm_legit_no_training_wrapper(*args, **kwargs):
    """Wrapper for aten._native_batch_norm_legit_no_training"""
    return _remote_kernel_fallback(torch.ops.aten._native_batch_norm_legit_no_training, *args, **kwargs)
def __native_batch_norm_legit_no_training_out_wrapper(*args, **kwargs):
    """Wrapper for aten._native_batch_norm_legit_no_training.out"""
    return _remote_kernel_fallback(torch.ops.aten._native_batch_norm_legit_no_training.out, *args, **kwargs)
def __pdist_forward_wrapper(*args, **kwargs):
    """Wrapper for aten._pdist_forward"""
    return _remote_kernel_fallback(torch.ops.aten._pdist_forward, *args, **kwargs)
def __pdist_forward_out_wrapper(*args, **kwargs):
    """Wrapper for aten._pdist_forward.out"""
    return _remote_kernel_fallback(torch.ops.aten._pdist_forward.out, *args, **kwargs)
def __softmax_wrapper(*args, **kwargs):
    """Wrapper for aten._softmax"""
    return _remote_kernel_fallback(torch.ops.aten._softmax, *args, **kwargs)
def __softmax_out_wrapper(*args, **kwargs):
    """Wrapper for aten._softmax.out"""
    return _remote_kernel_fallback(torch.ops.aten._softmax.out, *args, **kwargs)
def _abs_wrapper(*args, **kwargs):
    """Wrapper for aten.abs"""
    return _remote_kernel_fallback(torch.ops.aten.abs, *args, **kwargs)
def _abs_out_wrapper(*args, **kwargs):
    """Wrapper for aten.abs.out"""
    return _remote_kernel_fallback(torch.ops.aten.abs.out, *args, **kwargs)
def _abs__wrapper(*args, **kwargs):
    """Wrapper for aten.abs_"""
    return _remote_kernel_fallback(torch.ops.aten.abs_, *args, **kwargs)
def _acos_wrapper(*args, **kwargs):
    """Wrapper for aten.acos"""
    return _remote_kernel_fallback(torch.ops.aten.acos, *args, **kwargs)
def _acos_out_wrapper(*args, **kwargs):
    """Wrapper for aten.acos.out"""
    return _remote_kernel_fallback(torch.ops.aten.acos.out, *args, **kwargs)
def _acos__wrapper(*args, **kwargs):
    """Wrapper for aten.acos_"""
    return _remote_kernel_fallback(torch.ops.aten.acos_, *args, **kwargs)
def _acosh_wrapper(*args, **kwargs):
    """Wrapper for aten.acosh"""
    return _remote_kernel_fallback(torch.ops.aten.acosh, *args, **kwargs)
def _acosh_out_wrapper(*args, **kwargs):
    """Wrapper for aten.acosh.out"""
    return _remote_kernel_fallback(torch.ops.aten.acosh.out, *args, **kwargs)
def _acosh__wrapper(*args, **kwargs):
    """Wrapper for aten.acosh_"""
    return _remote_kernel_fallback(torch.ops.aten.acosh_, *args, **kwargs)
def _adaptive_avg_pool1d_wrapper(*args, **kwargs):
    """Wrapper for aten.adaptive_avg_pool1d"""
    return _remote_kernel_fallback(torch.ops.aten.adaptive_avg_pool1d, *args, **kwargs)
def _adaptive_avg_pool1d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.adaptive_avg_pool1d.out"""
    return _remote_kernel_fallback(torch.ops.aten.adaptive_avg_pool1d.out, *args, **kwargs)
def _add_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.add.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.add.Scalar, *args, **kwargs)
def _add_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.add.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.add.Scalar_out, *args, **kwargs)
def _add_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.add.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.add.Tensor, *args, **kwargs)
def _add_out_wrapper(*args, **kwargs):
    """Wrapper for aten.add.out"""
    return _remote_kernel_fallback(torch.ops.aten.add.out, *args, **kwargs)
def _add__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.add_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.add_.Scalar, *args, **kwargs)
def _add__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.add_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.add_.Tensor, *args, **kwargs)
def _addmm_wrapper(*args, **kwargs):
    """Wrapper for aten.addmm"""
    return _remote_kernel_fallback(torch.ops.aten.addmm, *args, **kwargs)
def _addmm_out_wrapper(*args, **kwargs):
    """Wrapper for aten.addmm.out"""
    return _remote_kernel_fallback(torch.ops.aten.addmm.out, *args, **kwargs)
def _addmm__wrapper(*args, **kwargs):
    """Wrapper for aten.addmm_"""
    return _remote_kernel_fallback(torch.ops.aten.addmm_, *args, **kwargs)
def _amax_wrapper(*args, **kwargs):
    """Wrapper for aten.amax"""
    return _remote_kernel_fallback(torch.ops.aten.amax, *args, **kwargs)
def _amax_out_wrapper(*args, **kwargs):
    """Wrapper for aten.amax.out"""
    return _remote_kernel_fallback(torch.ops.aten.amax.out, *args, **kwargs)
def _amin_wrapper(*args, **kwargs):
    """Wrapper for aten.amin"""
    return _remote_kernel_fallback(torch.ops.aten.amin, *args, **kwargs)
def _amin_out_wrapper(*args, **kwargs):
    """Wrapper for aten.amin.out"""
    return _remote_kernel_fallback(torch.ops.aten.amin.out, *args, **kwargs)
def _any_wrapper(*args, **kwargs):
    """Wrapper for aten.any"""
    return _remote_kernel_fallback(torch.ops.aten.any, *args, **kwargs)
def _any_dim_wrapper(*args, **kwargs):
    """Wrapper for aten.any.dim"""
    return _remote_kernel_fallback(torch.ops.aten.any.dim, *args, **kwargs)
def _any_dims_wrapper(*args, **kwargs):
    """Wrapper for aten.any.dims"""
    return _remote_kernel_fallback(torch.ops.aten.any.dims, *args, **kwargs)
def _any_dims_out_wrapper(*args, **kwargs):
    """Wrapper for aten.any.dims_out"""
    return _remote_kernel_fallback(torch.ops.aten.any.dims_out, *args, **kwargs)
def _any_out_wrapper(*args, **kwargs):
    """Wrapper for aten.any.out"""
    return _remote_kernel_fallback(torch.ops.aten.any.out, *args, **kwargs)
def _arange_out_wrapper(*args, **kwargs):
    """Wrapper for aten.arange.out"""
    return _remote_kernel_fallback(torch.ops.aten.arange.out, *args, **kwargs)
def _arange_start_step_wrapper(*args, **kwargs):
    """Wrapper for aten.arange.start_step"""
    return _remote_kernel_fallback(torch.ops.aten.arange.start_step, *args, **kwargs)
def _argmax_wrapper(*args, **kwargs):
    """Wrapper for aten.argmax"""
    return _remote_kernel_fallback(torch.ops.aten.argmax, *args, **kwargs)
def _argmax_out_wrapper(*args, **kwargs):
    """Wrapper for aten.argmax.out"""
    return _remote_kernel_fallback(torch.ops.aten.argmax.out, *args, **kwargs)
def _argmin_wrapper(*args, **kwargs):
    """Wrapper for aten.argmin"""
    return _remote_kernel_fallback(torch.ops.aten.argmin, *args, **kwargs)
def _argmin_out_wrapper(*args, **kwargs):
    """Wrapper for aten.argmin.out"""
    return _remote_kernel_fallback(torch.ops.aten.argmin.out, *args, **kwargs)
def _asin_wrapper(*args, **kwargs):
    """Wrapper for aten.asin"""
    return _remote_kernel_fallback(torch.ops.aten.asin, *args, **kwargs)
def _asin_out_wrapper(*args, **kwargs):
    """Wrapper for aten.asin.out"""
    return _remote_kernel_fallback(torch.ops.aten.asin.out, *args, **kwargs)
def _asin__wrapper(*args, **kwargs):
    """Wrapper for aten.asin_"""
    return _remote_kernel_fallback(torch.ops.aten.asin_, *args, **kwargs)
def _asinh_wrapper(*args, **kwargs):
    """Wrapper for aten.asinh"""
    return _remote_kernel_fallback(torch.ops.aten.asinh, *args, **kwargs)
def _asinh_out_wrapper(*args, **kwargs):
    """Wrapper for aten.asinh.out"""
    return _remote_kernel_fallback(torch.ops.aten.asinh.out, *args, **kwargs)
def _asinh__wrapper(*args, **kwargs):
    """Wrapper for aten.asinh_"""
    return _remote_kernel_fallback(torch.ops.aten.asinh_, *args, **kwargs)
def _atan_wrapper(*args, **kwargs):
    """Wrapper for aten.atan"""
    return _remote_kernel_fallback(torch.ops.aten.atan, *args, **kwargs)
def _atan_out_wrapper(*args, **kwargs):
    """Wrapper for aten.atan.out"""
    return _remote_kernel_fallback(torch.ops.aten.atan.out, *args, **kwargs)
def _atan2_wrapper(*args, **kwargs):
    """Wrapper for aten.atan2"""
    return _remote_kernel_fallback(torch.ops.aten.atan2, *args, **kwargs)
def _atan2_out_wrapper(*args, **kwargs):
    """Wrapper for aten.atan2.out"""
    return _remote_kernel_fallback(torch.ops.aten.atan2.out, *args, **kwargs)
def _atan2__wrapper(*args, **kwargs):
    """Wrapper for aten.atan2_"""
    return _remote_kernel_fallback(torch.ops.aten.atan2_, *args, **kwargs)
def _atan__wrapper(*args, **kwargs):
    """Wrapper for aten.atan_"""
    return _remote_kernel_fallback(torch.ops.aten.atan_, *args, **kwargs)
def _atanh_wrapper(*args, **kwargs):
    """Wrapper for aten.atanh"""
    return _remote_kernel_fallback(torch.ops.aten.atanh, *args, **kwargs)
def _atanh_out_wrapper(*args, **kwargs):
    """Wrapper for aten.atanh.out"""
    return _remote_kernel_fallback(torch.ops.aten.atanh.out, *args, **kwargs)
def _atanh__wrapper(*args, **kwargs):
    """Wrapper for aten.atanh_"""
    return _remote_kernel_fallback(torch.ops.aten.atanh_, *args, **kwargs)
def _avg_pool1d_wrapper(*args, **kwargs):
    """Wrapper for aten.avg_pool1d"""
    return _remote_kernel_fallback(torch.ops.aten.avg_pool1d, *args, **kwargs)
def _avg_pool1d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.avg_pool1d.out"""
    return _remote_kernel_fallback(torch.ops.aten.avg_pool1d.out, *args, **kwargs)
def _avg_pool2d_wrapper(*args, **kwargs):
    """Wrapper for aten.avg_pool2d"""
    return _remote_kernel_fallback(torch.ops.aten.avg_pool2d, *args, **kwargs)
def _avg_pool2d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.avg_pool2d.out"""
    return _remote_kernel_fallback(torch.ops.aten.avg_pool2d.out, *args, **kwargs)
def _avg_pool2d_backward_wrapper(*args, **kwargs):
    """Wrapper for aten.avg_pool2d_backward"""
    return _remote_kernel_fallback(torch.ops.aten.avg_pool2d_backward, *args, **kwargs)
def _avg_pool3d_wrapper(*args, **kwargs):
    """Wrapper for aten.avg_pool3d"""
    return _remote_kernel_fallback(torch.ops.aten.avg_pool3d, *args, **kwargs)
def _avg_pool3d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.avg_pool3d.out"""
    return _remote_kernel_fallback(torch.ops.aten.avg_pool3d.out, *args, **kwargs)
def _bitwise_and_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_and.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_and.Scalar, *args, **kwargs)
def _bitwise_and_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_and.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_and.Scalar_out, *args, **kwargs)
def _bitwise_and_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_and.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_and.Tensor, *args, **kwargs)
def _bitwise_and_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_and.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_and.Tensor_out, *args, **kwargs)
def _bitwise_and__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_and_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_and_.Scalar, *args, **kwargs)
def _bitwise_and__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_and_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_and_.Tensor, *args, **kwargs)
def _bitwise_not_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_not"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_not, *args, **kwargs)
def _bitwise_not_out_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_not.out"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_not.out, *args, **kwargs)
def _bitwise_not__wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_not_"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_not_, *args, **kwargs)
def _bitwise_or_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_or.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_or.Scalar, *args, **kwargs)
def _bitwise_or_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_or.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_or.Scalar_out, *args, **kwargs)
def _bitwise_or_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_or.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_or.Tensor, *args, **kwargs)
def _bitwise_or_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_or.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_or.Tensor_out, *args, **kwargs)
def _bitwise_or__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_or_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_or_.Scalar, *args, **kwargs)
def _bitwise_or__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_or_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_or_.Tensor, *args, **kwargs)
def _bitwise_xor_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_xor.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_xor.Scalar, *args, **kwargs)
def _bitwise_xor_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_xor.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_xor.Scalar_out, *args, **kwargs)
def _bitwise_xor_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_xor.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_xor.Tensor, *args, **kwargs)
def _bitwise_xor_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_xor.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_xor.Tensor_out, *args, **kwargs)
def _bitwise_xor__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_xor_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_xor_.Scalar, *args, **kwargs)
def _bitwise_xor__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.bitwise_xor_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.bitwise_xor_.Tensor, *args, **kwargs)
def _bmm_wrapper(*args, **kwargs):
    """Wrapper for aten.bmm"""
    return _remote_kernel_fallback(torch.ops.aten.bmm, *args, **kwargs)
def _bmm_out_wrapper(*args, **kwargs):
    """Wrapper for aten.bmm.out"""
    return _remote_kernel_fallback(torch.ops.aten.bmm.out, *args, **kwargs)
def _cat_wrapper(*args, **kwargs):
    """Wrapper for aten.cat"""
    return _remote_kernel_fallback(torch.ops.aten.cat, *args, **kwargs)
def _cat_out_wrapper(*args, **kwargs):
    """Wrapper for aten.cat.out"""
    return _remote_kernel_fallback(torch.ops.aten.cat.out, *args, **kwargs)
def _ceil_wrapper(*args, **kwargs):
    """Wrapper for aten.ceil"""
    return _remote_kernel_fallback(torch.ops.aten.ceil, *args, **kwargs)
def _ceil_out_wrapper(*args, **kwargs):
    """Wrapper for aten.ceil.out"""
    return _remote_kernel_fallback(torch.ops.aten.ceil.out, *args, **kwargs)
def _ceil__wrapper(*args, **kwargs):
    """Wrapper for aten.ceil_"""
    return _remote_kernel_fallback(torch.ops.aten.ceil_, *args, **kwargs)
def _clamp_wrapper(*args, **kwargs):
    """Wrapper for aten.clamp"""
    return _remote_kernel_fallback(torch.ops.aten.clamp, *args, **kwargs)
def _clamp_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.clamp.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.clamp.Tensor, *args, **kwargs)
def _clamp_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.clamp.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.clamp.Tensor_out, *args, **kwargs)
def _clamp_out_wrapper(*args, **kwargs):
    """Wrapper for aten.clamp.out"""
    return _remote_kernel_fallback(torch.ops.aten.clamp.out, *args, **kwargs)
def _clamp__wrapper(*args, **kwargs):
    """Wrapper for aten.clamp_"""
    return _remote_kernel_fallback(torch.ops.aten.clamp_, *args, **kwargs)
def _clamp__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.clamp_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.clamp_.Tensor, *args, **kwargs)
def _clone_wrapper(*args, **kwargs):
    """Wrapper for aten.clone"""
    return _remote_kernel_fallback(torch.ops.aten.clone, *args, **kwargs)
def _clone_out_wrapper(*args, **kwargs):
    """Wrapper for aten.clone.out"""
    return _remote_kernel_fallback(torch.ops.aten.clone.out, *args, **kwargs)
def _col2im_wrapper(*args, **kwargs):
    """Wrapper for aten.col2im"""
    return _remote_kernel_fallback(torch.ops.aten.col2im, *args, **kwargs)
def _col2im_out_wrapper(*args, **kwargs):
    """Wrapper for aten.col2im.out"""
    return _remote_kernel_fallback(torch.ops.aten.col2im.out, *args, **kwargs)
def _constant_pad_nd_wrapper(*args, **kwargs):
    """Wrapper for aten.constant_pad_nd"""
    return _remote_kernel_fallback(torch.ops.aten.constant_pad_nd, *args, **kwargs)
def _constant_pad_nd_out_wrapper(*args, **kwargs):
    """Wrapper for aten.constant_pad_nd.out"""
    return _remote_kernel_fallback(torch.ops.aten.constant_pad_nd.out, *args, **kwargs)
def _convolution_wrapper(*args, **kwargs):
    """Wrapper for aten.convolution"""
    return _remote_kernel_fallback(torch.ops.aten.convolution, *args, **kwargs)
def _convolution_out_wrapper(*args, **kwargs):
    """Wrapper for aten.convolution.out"""
    return _remote_kernel_fallback(torch.ops.aten.convolution.out, *args, **kwargs)
def _convolution_backward_wrapper(*args, **kwargs):
    """Wrapper for aten.convolution_backward"""
    return _remote_kernel_fallback(torch.ops.aten.convolution_backward, *args, **kwargs)
def _convolution_backward_out_wrapper(*args, **kwargs):
    """Wrapper for aten.convolution_backward.out"""
    return _remote_kernel_fallback(torch.ops.aten.convolution_backward.out, *args, **kwargs)
def _cos_wrapper(*args, **kwargs):
    """Wrapper for aten.cos"""
    return _remote_kernel_fallback(torch.ops.aten.cos, *args, **kwargs)
def _cos_out_wrapper(*args, **kwargs):
    """Wrapper for aten.cos.out"""
    return _remote_kernel_fallback(torch.ops.aten.cos.out, *args, **kwargs)
def _cos__wrapper(*args, **kwargs):
    """Wrapper for aten.cos_"""
    return _remote_kernel_fallback(torch.ops.aten.cos_, *args, **kwargs)
def _cosh_wrapper(*args, **kwargs):
    """Wrapper for aten.cosh"""
    return _remote_kernel_fallback(torch.ops.aten.cosh, *args, **kwargs)
def _cosh_out_wrapper(*args, **kwargs):
    """Wrapper for aten.cosh.out"""
    return _remote_kernel_fallback(torch.ops.aten.cosh.out, *args, **kwargs)
def _cosh__wrapper(*args, **kwargs):
    """Wrapper for aten.cosh_"""
    return _remote_kernel_fallback(torch.ops.aten.cosh_, *args, **kwargs)
def _cumsum_wrapper(*args, **kwargs):
    """Wrapper for aten.cumsum"""
    return _remote_kernel_fallback(torch.ops.aten.cumsum, *args, **kwargs)
def _cumsum_out_wrapper(*args, **kwargs):
    """Wrapper for aten.cumsum.out"""
    return _remote_kernel_fallback(torch.ops.aten.cumsum.out, *args, **kwargs)
def _cumsum__wrapper(*args, **kwargs):
    """Wrapper for aten.cumsum_"""
    return _remote_kernel_fallback(torch.ops.aten.cumsum_, *args, **kwargs)
def _diagonal_wrapper(*args, **kwargs):
    """Wrapper for aten.diagonal"""
    return _remote_kernel_fallback(torch.ops.aten.diagonal, *args, **kwargs)
def _div_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.div.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.div.Scalar, *args, **kwargs)
def _div_Scalar_mode_wrapper(*args, **kwargs):
    """Wrapper for aten.div.Scalar_mode"""
    return _remote_kernel_fallback(torch.ops.aten.div.Scalar_mode, *args, **kwargs)
def _div_Scalar_mode_out_wrapper(*args, **kwargs):
    """Wrapper for aten.div.Scalar_mode_out"""
    return _remote_kernel_fallback(torch.ops.aten.div.Scalar_mode_out, *args, **kwargs)
def _div_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.div.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.div.Scalar_out, *args, **kwargs)
def _div_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.div.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.div.Tensor, *args, **kwargs)
def _div_Tensor_mode_wrapper(*args, **kwargs):
    """Wrapper for aten.div.Tensor_mode"""
    return _remote_kernel_fallback(torch.ops.aten.div.Tensor_mode, *args, **kwargs)
def _div_out_wrapper(*args, **kwargs):
    """Wrapper for aten.div.out"""
    return _remote_kernel_fallback(torch.ops.aten.div.out, *args, **kwargs)
def _div__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.div_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.div_.Scalar, *args, **kwargs)
def _div__Scalar_mode_wrapper(*args, **kwargs):
    """Wrapper for aten.div_.Scalar_mode"""
    return _remote_kernel_fallback(torch.ops.aten.div_.Scalar_mode, *args, **kwargs)
def _div__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.div_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.div_.Tensor, *args, **kwargs)
def _div__Tensor_mode_wrapper(*args, **kwargs):
    """Wrapper for aten.div_.Tensor_mode"""
    return _remote_kernel_fallback(torch.ops.aten.div_.Tensor_mode, *args, **kwargs)
def _elu_wrapper(*args, **kwargs):
    """Wrapper for aten.elu"""
    return _remote_kernel_fallback(torch.ops.aten.elu, *args, **kwargs)
def _elu_out_wrapper(*args, **kwargs):
    """Wrapper for aten.elu.out"""
    return _remote_kernel_fallback(torch.ops.aten.elu.out, *args, **kwargs)
def _elu__wrapper(*args, **kwargs):
    """Wrapper for aten.elu_"""
    return _remote_kernel_fallback(torch.ops.aten.elu_, *args, **kwargs)
def _embedding_wrapper(*args, **kwargs):
    """Wrapper for aten.embedding"""
    return _remote_kernel_fallback(torch.ops.aten.embedding, *args, **kwargs)
def _embedding_out_wrapper(*args, **kwargs):
    """Wrapper for aten.embedding.out"""
    return _remote_kernel_fallback(torch.ops.aten.embedding.out, *args, **kwargs)
def _embedding_dense_backward_wrapper(*args, **kwargs):
    """Wrapper for aten.embedding_dense_backward"""
    return _remote_kernel_fallback(torch.ops.aten.embedding_dense_backward, *args, **kwargs)
def _embedding_dense_backward_out_wrapper(*args, **kwargs):
    """Wrapper for aten.embedding_dense_backward.out"""
    return _remote_kernel_fallback(torch.ops.aten.embedding_dense_backward.out, *args, **kwargs)
def _eq_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.eq.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.eq.Scalar, *args, **kwargs)
def _eq_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.eq.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.eq.Scalar_out, *args, **kwargs)
def _eq_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.eq.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.eq.Tensor, *args, **kwargs)
def _eq_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.eq.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.eq.Tensor_out, *args, **kwargs)
def _eq__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.eq_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.eq_.Scalar, *args, **kwargs)
def _eq__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.eq_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.eq_.Tensor, *args, **kwargs)
def _erf_wrapper(*args, **kwargs):
    """Wrapper for aten.erf"""
    return _remote_kernel_fallback(torch.ops.aten.erf, *args, **kwargs)
def _erf_out_wrapper(*args, **kwargs):
    """Wrapper for aten.erf.out"""
    return _remote_kernel_fallback(torch.ops.aten.erf.out, *args, **kwargs)
def _erf__wrapper(*args, **kwargs):
    """Wrapper for aten.erf_"""
    return _remote_kernel_fallback(torch.ops.aten.erf_, *args, **kwargs)
def _exp_wrapper(*args, **kwargs):
    """Wrapper for aten.exp"""
    return _remote_kernel_fallback(torch.ops.aten.exp, *args, **kwargs)
def _exp_out_wrapper(*args, **kwargs):
    """Wrapper for aten.exp.out"""
    return _remote_kernel_fallback(torch.ops.aten.exp.out, *args, **kwargs)
def _exp__wrapper(*args, **kwargs):
    """Wrapper for aten.exp_"""
    return _remote_kernel_fallback(torch.ops.aten.exp_, *args, **kwargs)
def _expand_wrapper(*args, **kwargs):
    """Wrapper for aten.expand"""
    return _remote_kernel_fallback(torch.ops.aten.expand, *args, **kwargs)
def _expm1_wrapper(*args, **kwargs):
    """Wrapper for aten.expm1"""
    return _remote_kernel_fallback(torch.ops.aten.expm1, *args, **kwargs)
def _expm1_out_wrapper(*args, **kwargs):
    """Wrapper for aten.expm1.out"""
    return _remote_kernel_fallback(torch.ops.aten.expm1.out, *args, **kwargs)
def _expm1__wrapper(*args, **kwargs):
    """Wrapper for aten.expm1_"""
    return _remote_kernel_fallback(torch.ops.aten.expm1_, *args, **kwargs)
def _fill_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.fill.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.fill.Scalar, *args, **kwargs)
def _fill_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.fill.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.fill.Scalar_out, *args, **kwargs)
def _fill__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.fill_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.fill_.Scalar, *args, **kwargs)
def _flip_wrapper(*args, **kwargs):
    """Wrapper for aten.flip"""
    return _remote_kernel_fallback(torch.ops.aten.flip, *args, **kwargs)
def _flip_out_wrapper(*args, **kwargs):
    """Wrapper for aten.flip.out"""
    return _remote_kernel_fallback(torch.ops.aten.flip.out, *args, **kwargs)
def _floor_wrapper(*args, **kwargs):
    """Wrapper for aten.floor"""
    return _remote_kernel_fallback(torch.ops.aten.floor, *args, **kwargs)
def _floor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.floor.out"""
    return _remote_kernel_fallback(torch.ops.aten.floor.out, *args, **kwargs)
def _floor__wrapper(*args, **kwargs):
    """Wrapper for aten.floor_"""
    return _remote_kernel_fallback(torch.ops.aten.floor_, *args, **kwargs)
def _fmod_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.fmod.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.fmod.Scalar, *args, **kwargs)
def _fmod_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.fmod.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.fmod.Scalar_out, *args, **kwargs)
def _fmod_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.fmod.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.fmod.Tensor, *args, **kwargs)
def _fmod_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.fmod.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.fmod.Tensor_out, *args, **kwargs)
def _fmod__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.fmod_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.fmod_.Scalar, *args, **kwargs)
def _fmod__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.fmod_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.fmod_.Tensor, *args, **kwargs)
def _full_wrapper(*args, **kwargs):
    """Wrapper for aten.full"""
    return _remote_kernel_fallback(torch.ops.aten.full, *args, **kwargs)
def _full_out_wrapper(*args, **kwargs):
    """Wrapper for aten.full.out"""
    return _remote_kernel_fallback(torch.ops.aten.full.out, *args, **kwargs)
def _full_like_wrapper(*args, **kwargs):
    """Wrapper for aten.full_like"""
    return _remote_kernel_fallback(torch.ops.aten.full_like, *args, **kwargs)
def _full_like_out_wrapper(*args, **kwargs):
    """Wrapper for aten.full_like.out"""
    return _remote_kernel_fallback(torch.ops.aten.full_like.out, *args, **kwargs)
def _gather_wrapper(*args, **kwargs):
    """Wrapper for aten.gather"""
    return _remote_kernel_fallback(torch.ops.aten.gather, *args, **kwargs)
def _gather_out_wrapper(*args, **kwargs):
    """Wrapper for aten.gather.out"""
    return _remote_kernel_fallback(torch.ops.aten.gather.out, *args, **kwargs)
def _ge_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.ge.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.ge.Scalar, *args, **kwargs)
def _ge_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.ge.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.ge.Scalar_out, *args, **kwargs)
def _ge_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.ge.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.ge.Tensor, *args, **kwargs)
def _ge_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.ge.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.ge.Tensor_out, *args, **kwargs)
def _ge__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.ge_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.ge_.Scalar, *args, **kwargs)
def _ge__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.ge_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.ge_.Tensor, *args, **kwargs)
def _gelu_wrapper(*args, **kwargs):
    """Wrapper for aten.gelu"""
    return _remote_kernel_fallback(torch.ops.aten.gelu, *args, **kwargs)
def _gelu_out_wrapper(*args, **kwargs):
    """Wrapper for aten.gelu.out"""
    return _remote_kernel_fallback(torch.ops.aten.gelu.out, *args, **kwargs)
def _gelu__wrapper(*args, **kwargs):
    """Wrapper for aten.gelu_"""
    return _remote_kernel_fallback(torch.ops.aten.gelu_, *args, **kwargs)
def _grid_sampler_2d_wrapper(*args, **kwargs):
    """Wrapper for aten.grid_sampler_2d"""
    return _remote_kernel_fallback(torch.ops.aten.grid_sampler_2d, *args, **kwargs)
def _grid_sampler_2d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.grid_sampler_2d.out"""
    return _remote_kernel_fallback(torch.ops.aten.grid_sampler_2d.out, *args, **kwargs)
def _gt_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.gt.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.gt.Scalar, *args, **kwargs)
def _gt_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.gt.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.gt.Scalar_out, *args, **kwargs)
def _gt_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.gt.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.gt.Tensor, *args, **kwargs)
def _gt_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.gt.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.gt.Tensor_out, *args, **kwargs)
def _gt__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.gt_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.gt_.Scalar, *args, **kwargs)
def _gt__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.gt_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.gt_.Tensor, *args, **kwargs)
def _hardtanh_wrapper(*args, **kwargs):
    """Wrapper for aten.hardtanh"""
    return _remote_kernel_fallback(torch.ops.aten.hardtanh, *args, **kwargs)
def _hardtanh_out_wrapper(*args, **kwargs):
    """Wrapper for aten.hardtanh.out"""
    return _remote_kernel_fallback(torch.ops.aten.hardtanh.out, *args, **kwargs)
def _hardtanh__wrapper(*args, **kwargs):
    """Wrapper for aten.hardtanh_"""
    return _remote_kernel_fallback(torch.ops.aten.hardtanh_, *args, **kwargs)
def _index_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.index.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.index.Tensor, *args, **kwargs)
def _index_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.index.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.index.Tensor_out, *args, **kwargs)
def _index_put_wrapper(*args, **kwargs):
    """Wrapper for aten.index_put"""
    return _remote_kernel_fallback(torch.ops.aten.index_put, *args, **kwargs)
def _index_put_out_wrapper(*args, **kwargs):
    """Wrapper for aten.index_put.out"""
    return _remote_kernel_fallback(torch.ops.aten.index_put.out, *args, **kwargs)
def _index_put__wrapper(*args, **kwargs):
    """Wrapper for aten.index_put_"""
    return _remote_kernel_fallback(torch.ops.aten.index_put_, *args, **kwargs)
def _index_select_wrapper(*args, **kwargs):
    """Wrapper for aten.index_select"""
    return _remote_kernel_fallback(torch.ops.aten.index_select, *args, **kwargs)
def _index_select_out_wrapper(*args, **kwargs):
    """Wrapper for aten.index_select.out"""
    return _remote_kernel_fallback(torch.ops.aten.index_select.out, *args, **kwargs)
def _isinf_wrapper(*args, **kwargs):
    """Wrapper for aten.isinf"""
    return _remote_kernel_fallback(torch.ops.aten.isinf, *args, **kwargs)
def _isinf_out_wrapper(*args, **kwargs):
    """Wrapper for aten.isinf.out"""
    return _remote_kernel_fallback(torch.ops.aten.isinf.out, *args, **kwargs)
def _isnan_wrapper(*args, **kwargs):
    """Wrapper for aten.isnan"""
    return _remote_kernel_fallback(torch.ops.aten.isnan, *args, **kwargs)
def _isnan_out_wrapper(*args, **kwargs):
    """Wrapper for aten.isnan.out"""
    return _remote_kernel_fallback(torch.ops.aten.isnan.out, *args, **kwargs)
def _le_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.le.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.le.Scalar, *args, **kwargs)
def _le_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.le.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.le.Scalar_out, *args, **kwargs)
def _le_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.le.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.le.Tensor, *args, **kwargs)
def _le_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.le.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.le.Tensor_out, *args, **kwargs)
def _le__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.le_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.le_.Scalar, *args, **kwargs)
def _le__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.le_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.le_.Tensor, *args, **kwargs)
def _leaky_relu_wrapper(*args, **kwargs):
    """Wrapper for aten.leaky_relu"""
    return _remote_kernel_fallback(torch.ops.aten.leaky_relu, *args, **kwargs)
def _leaky_relu_out_wrapper(*args, **kwargs):
    """Wrapper for aten.leaky_relu.out"""
    return _remote_kernel_fallback(torch.ops.aten.leaky_relu.out, *args, **kwargs)
def _leaky_relu__wrapper(*args, **kwargs):
    """Wrapper for aten.leaky_relu_"""
    return _remote_kernel_fallback(torch.ops.aten.leaky_relu_, *args, **kwargs)
def _log_wrapper(*args, **kwargs):
    """Wrapper for aten.log"""
    return _remote_kernel_fallback(torch.ops.aten.log, *args, **kwargs)
def _log_out_wrapper(*args, **kwargs):
    """Wrapper for aten.log.out"""
    return _remote_kernel_fallback(torch.ops.aten.log.out, *args, **kwargs)
def _log10_wrapper(*args, **kwargs):
    """Wrapper for aten.log10"""
    return _remote_kernel_fallback(torch.ops.aten.log10, *args, **kwargs)
def _log10_out_wrapper(*args, **kwargs):
    """Wrapper for aten.log10.out"""
    return _remote_kernel_fallback(torch.ops.aten.log10.out, *args, **kwargs)
def _log10__wrapper(*args, **kwargs):
    """Wrapper for aten.log10_"""
    return _remote_kernel_fallback(torch.ops.aten.log10_, *args, **kwargs)
def _log1p_wrapper(*args, **kwargs):
    """Wrapper for aten.log1p"""
    return _remote_kernel_fallback(torch.ops.aten.log1p, *args, **kwargs)
def _log1p_out_wrapper(*args, **kwargs):
    """Wrapper for aten.log1p.out"""
    return _remote_kernel_fallback(torch.ops.aten.log1p.out, *args, **kwargs)
def _log1p__wrapper(*args, **kwargs):
    """Wrapper for aten.log1p_"""
    return _remote_kernel_fallback(torch.ops.aten.log1p_, *args, **kwargs)
def _log2_wrapper(*args, **kwargs):
    """Wrapper for aten.log2"""
    return _remote_kernel_fallback(torch.ops.aten.log2, *args, **kwargs)
def _log2_out_wrapper(*args, **kwargs):
    """Wrapper for aten.log2.out"""
    return _remote_kernel_fallback(torch.ops.aten.log2.out, *args, **kwargs)
def _log2__wrapper(*args, **kwargs):
    """Wrapper for aten.log2_"""
    return _remote_kernel_fallback(torch.ops.aten.log2_, *args, **kwargs)
def _log__wrapper(*args, **kwargs):
    """Wrapper for aten.log_"""
    return _remote_kernel_fallback(torch.ops.aten.log_, *args, **kwargs)
def _logical_and_wrapper(*args, **kwargs):
    """Wrapper for aten.logical_and"""
    return _remote_kernel_fallback(torch.ops.aten.logical_and, *args, **kwargs)
def _logical_and_out_wrapper(*args, **kwargs):
    """Wrapper for aten.logical_and.out"""
    return _remote_kernel_fallback(torch.ops.aten.logical_and.out, *args, **kwargs)
def _logical_and__wrapper(*args, **kwargs):
    """Wrapper for aten.logical_and_"""
    return _remote_kernel_fallback(torch.ops.aten.logical_and_, *args, **kwargs)
def _logical_not_wrapper(*args, **kwargs):
    """Wrapper for aten.logical_not"""
    return _remote_kernel_fallback(torch.ops.aten.logical_not, *args, **kwargs)
def _logical_not_out_wrapper(*args, **kwargs):
    """Wrapper for aten.logical_not.out"""
    return _remote_kernel_fallback(torch.ops.aten.logical_not.out, *args, **kwargs)
def _logical_not__wrapper(*args, **kwargs):
    """Wrapper for aten.logical_not_"""
    return _remote_kernel_fallback(torch.ops.aten.logical_not_, *args, **kwargs)
def _logical_or_wrapper(*args, **kwargs):
    """Wrapper for aten.logical_or"""
    return _remote_kernel_fallback(torch.ops.aten.logical_or, *args, **kwargs)
def _logical_or_out_wrapper(*args, **kwargs):
    """Wrapper for aten.logical_or.out"""
    return _remote_kernel_fallback(torch.ops.aten.logical_or.out, *args, **kwargs)
def _logical_or__wrapper(*args, **kwargs):
    """Wrapper for aten.logical_or_"""
    return _remote_kernel_fallback(torch.ops.aten.logical_or_, *args, **kwargs)
def _logical_xor_wrapper(*args, **kwargs):
    """Wrapper for aten.logical_xor"""
    return _remote_kernel_fallback(torch.ops.aten.logical_xor, *args, **kwargs)
def _logical_xor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.logical_xor.out"""
    return _remote_kernel_fallback(torch.ops.aten.logical_xor.out, *args, **kwargs)
def _logical_xor__wrapper(*args, **kwargs):
    """Wrapper for aten.logical_xor_"""
    return _remote_kernel_fallback(torch.ops.aten.logical_xor_, *args, **kwargs)
def _lt_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.lt.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.lt.Scalar, *args, **kwargs)
def _lt_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.lt.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.lt.Scalar_out, *args, **kwargs)
def _lt_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.lt.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.lt.Tensor, *args, **kwargs)
def _lt_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.lt.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.lt.Tensor_out, *args, **kwargs)
def _lt__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.lt_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.lt_.Scalar, *args, **kwargs)
def _lt__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.lt_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.lt_.Tensor, *args, **kwargs)
def _masked_scatter_wrapper(*args, **kwargs):
    """Wrapper for aten.masked_scatter"""
    return _remote_kernel_fallback(torch.ops.aten.masked_scatter, *args, **kwargs)
def _masked_scatter_out_wrapper(*args, **kwargs):
    """Wrapper for aten.masked_scatter.out"""
    return _remote_kernel_fallback(torch.ops.aten.masked_scatter.out, *args, **kwargs)
def _masked_scatter__wrapper(*args, **kwargs):
    """Wrapper for aten.masked_scatter_"""
    return _remote_kernel_fallback(torch.ops.aten.masked_scatter_, *args, **kwargs)
def _max_dim_wrapper(*args, **kwargs):
    """Wrapper for aten.max.dim"""
    return _remote_kernel_fallback(torch.ops.aten.max.dim, *args, **kwargs)
def _max_out_wrapper(*args, **kwargs):
    """Wrapper for aten.max.out"""
    return _remote_kernel_fallback(torch.ops.aten.max.out, *args, **kwargs)
def _max_pool2d_with_indices_wrapper(*args, **kwargs):
    """Wrapper for aten.max_pool2d_with_indices"""
    return _remote_kernel_fallback(torch.ops.aten.max_pool2d_with_indices, *args, **kwargs)
def _max_pool2d_with_indices_out_wrapper(*args, **kwargs):
    """Wrapper for aten.max_pool2d_with_indices.out"""
    return _remote_kernel_fallback(torch.ops.aten.max_pool2d_with_indices.out, *args, **kwargs)
def _max_pool2d_with_indices_backward_wrapper(*args, **kwargs):
    """Wrapper for aten.max_pool2d_with_indices_backward"""
    return _remote_kernel_fallback(torch.ops.aten.max_pool2d_with_indices_backward, *args, **kwargs)
def _max_pool3d_with_indices_wrapper(*args, **kwargs):
    """Wrapper for aten.max_pool3d_with_indices"""
    return _remote_kernel_fallback(torch.ops.aten.max_pool3d_with_indices, *args, **kwargs)
def _max_pool3d_with_indices_out_wrapper(*args, **kwargs):
    """Wrapper for aten.max_pool3d_with_indices.out"""
    return _remote_kernel_fallback(torch.ops.aten.max_pool3d_with_indices.out, *args, **kwargs)
def _maximum_wrapper(*args, **kwargs):
    """Wrapper for aten.maximum"""
    return _remote_kernel_fallback(torch.ops.aten.maximum, *args, **kwargs)
def _maximum_out_wrapper(*args, **kwargs):
    """Wrapper for aten.maximum.out"""
    return _remote_kernel_fallback(torch.ops.aten.maximum.out, *args, **kwargs)
def _mean_wrapper(*args, **kwargs):
    """Wrapper for aten.mean"""
    return _remote_kernel_fallback(torch.ops.aten.mean, *args, **kwargs)
def _mean_dim_wrapper(*args, **kwargs):
    """Wrapper for aten.mean.dim"""
    return _remote_kernel_fallback(torch.ops.aten.mean.dim, *args, **kwargs)
def _mean_out_wrapper(*args, **kwargs):
    """Wrapper for aten.mean.out"""
    return _remote_kernel_fallback(torch.ops.aten.mean.out, *args, **kwargs)
def _min_dim_wrapper(*args, **kwargs):
    """Wrapper for aten.min.dim"""
    return _remote_kernel_fallback(torch.ops.aten.min.dim, *args, **kwargs)
def _min_out_wrapper(*args, **kwargs):
    """Wrapper for aten.min.out"""
    return _remote_kernel_fallback(torch.ops.aten.min.out, *args, **kwargs)
def _minimum_wrapper(*args, **kwargs):
    """Wrapper for aten.minimum"""
    return _remote_kernel_fallback(torch.ops.aten.minimum, *args, **kwargs)
def _minimum_out_wrapper(*args, **kwargs):
    """Wrapper for aten.minimum.out"""
    return _remote_kernel_fallback(torch.ops.aten.minimum.out, *args, **kwargs)
def _mm_wrapper(*args, **kwargs):
    """Wrapper for aten.mm"""
    return _remote_kernel_fallback(torch.ops.aten.mm, *args, **kwargs)
def _mm_out_wrapper(*args, **kwargs):
    """Wrapper for aten.mm.out"""
    return _remote_kernel_fallback(torch.ops.aten.mm.out, *args, **kwargs)
def _mul_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.mul.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.mul.Scalar, *args, **kwargs)
def _mul_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.mul.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.mul.Scalar_out, *args, **kwargs)
def _mul_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.mul.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.mul.Tensor, *args, **kwargs)
def _mul_out_wrapper(*args, **kwargs):
    """Wrapper for aten.mul.out"""
    return _remote_kernel_fallback(torch.ops.aten.mul.out, *args, **kwargs)
def _mul__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.mul_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.mul_.Scalar, *args, **kwargs)
def _mul__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.mul_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.mul_.Tensor, *args, **kwargs)
def _native_dropout_wrapper(*args, **kwargs):
    """Wrapper for aten.native_dropout"""
    return _remote_kernel_fallback(torch.ops.aten.native_dropout, *args, **kwargs)
def _native_dropout_out_wrapper(*args, **kwargs):
    """Wrapper for aten.native_dropout.out"""
    return _remote_kernel_fallback(torch.ops.aten.native_dropout.out, *args, **kwargs)
def _native_group_norm_wrapper(*args, **kwargs):
    """Wrapper for aten.native_group_norm"""
    return _remote_kernel_fallback(torch.ops.aten.native_group_norm, *args, **kwargs)
def _native_group_norm_out_wrapper(*args, **kwargs):
    """Wrapper for aten.native_group_norm.out"""
    return _remote_kernel_fallback(torch.ops.aten.native_group_norm.out, *args, **kwargs)
def _native_group_norm_backward_wrapper(*args, **kwargs):
    """Wrapper for aten.native_group_norm_backward"""
    return _remote_kernel_fallback(torch.ops.aten.native_group_norm_backward, *args, **kwargs)
def _native_group_norm_backward_out_wrapper(*args, **kwargs):
    """Wrapper for aten.native_group_norm_backward.out"""
    return _remote_kernel_fallback(torch.ops.aten.native_group_norm_backward.out, *args, **kwargs)
def _native_layer_norm_wrapper(*args, **kwargs):
    """Wrapper for aten.native_layer_norm"""
    return _remote_kernel_fallback(torch.ops.aten.native_layer_norm, *args, **kwargs)
def _native_layer_norm_out_wrapper(*args, **kwargs):
    """Wrapper for aten.native_layer_norm.out"""
    return _remote_kernel_fallback(torch.ops.aten.native_layer_norm.out, *args, **kwargs)
def _native_layer_norm_backward_wrapper(*args, **kwargs):
    """Wrapper for aten.native_layer_norm_backward"""
    return _remote_kernel_fallback(torch.ops.aten.native_layer_norm_backward, *args, **kwargs)
def _native_layer_norm_backward_out_wrapper(*args, **kwargs):
    """Wrapper for aten.native_layer_norm_backward.out"""
    return _remote_kernel_fallback(torch.ops.aten.native_layer_norm_backward.out, *args, **kwargs)
def _ne_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.ne.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.ne.Scalar, *args, **kwargs)
def _ne_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.ne.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.ne.Scalar_out, *args, **kwargs)
def _ne_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.ne.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.ne.Tensor, *args, **kwargs)
def _ne_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.ne.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.ne.Tensor_out, *args, **kwargs)
def _ne__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.ne_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.ne_.Scalar, *args, **kwargs)
def _ne__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.ne_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.ne_.Tensor, *args, **kwargs)
def _neg_wrapper(*args, **kwargs):
    """Wrapper for aten.neg"""
    return _remote_kernel_fallback(torch.ops.aten.neg, *args, **kwargs)
def _neg_out_wrapper(*args, **kwargs):
    """Wrapper for aten.neg.out"""
    return _remote_kernel_fallback(torch.ops.aten.neg.out, *args, **kwargs)
def _neg__wrapper(*args, **kwargs):
    """Wrapper for aten.neg_"""
    return _remote_kernel_fallback(torch.ops.aten.neg_, *args, **kwargs)
def _nonzero_wrapper(*args, **kwargs):
    """Wrapper for aten.nonzero"""
    return _remote_kernel_fallback(torch.ops.aten.nonzero, *args, **kwargs)
def _nonzero_out_wrapper(*args, **kwargs):
    """Wrapper for aten.nonzero.out"""
    return _remote_kernel_fallback(torch.ops.aten.nonzero.out, *args, **kwargs)
def _permute_wrapper(*args, **kwargs):
    """Wrapper for aten.permute"""
    return _remote_kernel_fallback(torch.ops.aten.permute, *args, **kwargs)
def _pow_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.pow.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.pow.Scalar, *args, **kwargs)
def _pow_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.pow.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.pow.Scalar_out, *args, **kwargs)
def _pow_Tensor_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.pow.Tensor_Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.pow.Tensor_Scalar, *args, **kwargs)
def _pow_Tensor_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.pow.Tensor_Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.pow.Tensor_Scalar_out, *args, **kwargs)
def _pow_Tensor_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.pow.Tensor_Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.pow.Tensor_Tensor, *args, **kwargs)
def _pow_Tensor_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.pow.Tensor_Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.pow.Tensor_Tensor_out, *args, **kwargs)
def _pow__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.pow_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.pow_.Scalar, *args, **kwargs)
def _prod_wrapper(*args, **kwargs):
    """Wrapper for aten.prod"""
    return _remote_kernel_fallback(torch.ops.aten.prod, *args, **kwargs)
def _prod_dim_int_wrapper(*args, **kwargs):
    """Wrapper for aten.prod.dim_int"""
    return _remote_kernel_fallback(torch.ops.aten.prod.dim_int, *args, **kwargs)
def _prod_out_wrapper(*args, **kwargs):
    """Wrapper for aten.prod.out"""
    return _remote_kernel_fallback(torch.ops.aten.prod.out, *args, **kwargs)
def _rand_wrapper(*args, **kwargs):
    """Wrapper for aten.rand"""
    return _remote_kernel_fallback(torch.ops.aten.rand, *args, **kwargs)
def _rand_out_wrapper(*args, **kwargs):
    """Wrapper for aten.rand.out"""
    return _remote_kernel_fallback(torch.ops.aten.rand.out, *args, **kwargs)
def _randn_wrapper(*args, **kwargs):
    """Wrapper for aten.randn"""
    return _remote_kernel_fallback(torch.ops.aten.randn, *args, **kwargs)
def _randn_out_wrapper(*args, **kwargs):
    """Wrapper for aten.randn.out"""
    return _remote_kernel_fallback(torch.ops.aten.randn.out, *args, **kwargs)
def _randperm_wrapper(*args, **kwargs):
    """Wrapper for aten.randperm"""
    return _remote_kernel_fallback(torch.ops.aten.randperm, *args, **kwargs)
def _randperm_out_wrapper(*args, **kwargs):
    """Wrapper for aten.randperm.out"""
    return _remote_kernel_fallback(torch.ops.aten.randperm.out, *args, **kwargs)
def _reciprocal_wrapper(*args, **kwargs):
    """Wrapper for aten.reciprocal"""
    return _remote_kernel_fallback(torch.ops.aten.reciprocal, *args, **kwargs)
def _reciprocal_out_wrapper(*args, **kwargs):
    """Wrapper for aten.reciprocal.out"""
    return _remote_kernel_fallback(torch.ops.aten.reciprocal.out, *args, **kwargs)
def _reciprocal__wrapper(*args, **kwargs):
    """Wrapper for aten.reciprocal_"""
    return _remote_kernel_fallback(torch.ops.aten.reciprocal_, *args, **kwargs)
def _reflection_pad1d_wrapper(*args, **kwargs):
    """Wrapper for aten.reflection_pad1d"""
    return _remote_kernel_fallback(torch.ops.aten.reflection_pad1d, *args, **kwargs)
def _reflection_pad1d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.reflection_pad1d.out"""
    return _remote_kernel_fallback(torch.ops.aten.reflection_pad1d.out, *args, **kwargs)
def _reflection_pad2d_wrapper(*args, **kwargs):
    """Wrapper for aten.reflection_pad2d"""
    return _remote_kernel_fallback(torch.ops.aten.reflection_pad2d, *args, **kwargs)
def _reflection_pad2d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.reflection_pad2d.out"""
    return _remote_kernel_fallback(torch.ops.aten.reflection_pad2d.out, *args, **kwargs)
def _reflection_pad3d_wrapper(*args, **kwargs):
    """Wrapper for aten.reflection_pad3d"""
    return _remote_kernel_fallback(torch.ops.aten.reflection_pad3d, *args, **kwargs)
def _reflection_pad3d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.reflection_pad3d.out"""
    return _remote_kernel_fallback(torch.ops.aten.reflection_pad3d.out, *args, **kwargs)
def _relu_wrapper(*args, **kwargs):
    """Wrapper for aten.relu"""
    return _remote_kernel_fallback(torch.ops.aten.relu, *args, **kwargs)
def _relu_out_wrapper(*args, **kwargs):
    """Wrapper for aten.relu.out"""
    return _remote_kernel_fallback(torch.ops.aten.relu.out, *args, **kwargs)
def _relu__wrapper(*args, **kwargs):
    """Wrapper for aten.relu_"""
    return _remote_kernel_fallback(torch.ops.aten.relu_, *args, **kwargs)
def _remainder_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.remainder.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.remainder.Scalar, *args, **kwargs)
def _remainder_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.remainder.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.remainder.Scalar_out, *args, **kwargs)
def _remainder_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.remainder.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.remainder.Tensor, *args, **kwargs)
def _remainder_Tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.remainder.Tensor_out"""
    return _remote_kernel_fallback(torch.ops.aten.remainder.Tensor_out, *args, **kwargs)
def _remainder__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.remainder_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.remainder_.Scalar, *args, **kwargs)
def _remainder__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.remainder_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.remainder_.Tensor, *args, **kwargs)
def _repeat_wrapper(*args, **kwargs):
    """Wrapper for aten.repeat"""
    return _remote_kernel_fallback(torch.ops.aten.repeat, *args, **kwargs)
def _repeat_out_wrapper(*args, **kwargs):
    """Wrapper for aten.repeat.out"""
    return _remote_kernel_fallback(torch.ops.aten.repeat.out, *args, **kwargs)
def _replication_pad2d_wrapper(*args, **kwargs):
    """Wrapper for aten.replication_pad2d"""
    return _remote_kernel_fallback(torch.ops.aten.replication_pad2d, *args, **kwargs)
def _replication_pad2d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.replication_pad2d.out"""
    return _remote_kernel_fallback(torch.ops.aten.replication_pad2d.out, *args, **kwargs)
def _replication_pad3d_wrapper(*args, **kwargs):
    """Wrapper for aten.replication_pad3d"""
    return _remote_kernel_fallback(torch.ops.aten.replication_pad3d, *args, **kwargs)
def _replication_pad3d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.replication_pad3d.out"""
    return _remote_kernel_fallback(torch.ops.aten.replication_pad3d.out, *args, **kwargs)
def _round_wrapper(*args, **kwargs):
    """Wrapper for aten.round"""
    return _remote_kernel_fallback(torch.ops.aten.round, *args, **kwargs)
def _round_out_wrapper(*args, **kwargs):
    """Wrapper for aten.round.out"""
    return _remote_kernel_fallback(torch.ops.aten.round.out, *args, **kwargs)
def _round__wrapper(*args, **kwargs):
    """Wrapper for aten.round_"""
    return _remote_kernel_fallback(torch.ops.aten.round_, *args, **kwargs)
def _rsqrt_wrapper(*args, **kwargs):
    """Wrapper for aten.rsqrt"""
    return _remote_kernel_fallback(torch.ops.aten.rsqrt, *args, **kwargs)
def _rsqrt_out_wrapper(*args, **kwargs):
    """Wrapper for aten.rsqrt.out"""
    return _remote_kernel_fallback(torch.ops.aten.rsqrt.out, *args, **kwargs)
def _rsqrt__wrapper(*args, **kwargs):
    """Wrapper for aten.rsqrt_"""
    return _remote_kernel_fallback(torch.ops.aten.rsqrt_, *args, **kwargs)
def _scalar_tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.scalar_tensor"""
    return _remote_kernel_fallback(torch.ops.aten.scalar_tensor, *args, **kwargs)
def _scalar_tensor_out_wrapper(*args, **kwargs):
    """Wrapper for aten.scalar_tensor.out"""
    return _remote_kernel_fallback(torch.ops.aten.scalar_tensor.out, *args, **kwargs)
def _scatter_src_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter.src"""
    return _remote_kernel_fallback(torch.ops.aten.scatter.src, *args, **kwargs)
def _scatter_src_out_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter.src_out"""
    return _remote_kernel_fallback(torch.ops.aten.scatter.src_out, *args, **kwargs)
def _scatter_value_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter.value"""
    return _remote_kernel_fallback(torch.ops.aten.scatter.value, *args, **kwargs)
def _scatter_value_out_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter.value_out"""
    return _remote_kernel_fallback(torch.ops.aten.scatter.value_out, *args, **kwargs)
def _scatter__src_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter_.src"""
    return _remote_kernel_fallback(torch.ops.aten.scatter_.src, *args, **kwargs)
def _scatter__value_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter_.value"""
    return _remote_kernel_fallback(torch.ops.aten.scatter_.value, *args, **kwargs)
def _scatter_add_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter_add"""
    return _remote_kernel_fallback(torch.ops.aten.scatter_add, *args, **kwargs)
def _scatter_add_out_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter_add.out"""
    return _remote_kernel_fallback(torch.ops.aten.scatter_add.out, *args, **kwargs)
def _scatter_add__wrapper(*args, **kwargs):
    """Wrapper for aten.scatter_add_"""
    return _remote_kernel_fallback(torch.ops.aten.scatter_add_, *args, **kwargs)
def _scatter_reduce_two_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter_reduce.two"""
    return _remote_kernel_fallback(torch.ops.aten.scatter_reduce.two, *args, **kwargs)
def _scatter_reduce_two_out_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter_reduce.two_out"""
    return _remote_kernel_fallback(torch.ops.aten.scatter_reduce.two_out, *args, **kwargs)
def _scatter_reduce__two_wrapper(*args, **kwargs):
    """Wrapper for aten.scatter_reduce_.two"""
    return _remote_kernel_fallback(torch.ops.aten.scatter_reduce_.two, *args, **kwargs)
def _select_int_wrapper(*args, **kwargs):
    """Wrapper for aten.select.int"""
    return _remote_kernel_fallback(torch.ops.aten.select.int, *args, **kwargs)
def _select_scatter_wrapper(*args, **kwargs):
    """Wrapper for aten.select_scatter"""
    return _remote_kernel_fallback(torch.ops.aten.select_scatter, *args, **kwargs)
def _select_scatter_out_wrapper(*args, **kwargs):
    """Wrapper for aten.select_scatter.out"""
    return _remote_kernel_fallback(torch.ops.aten.select_scatter.out, *args, **kwargs)
def _sigmoid_wrapper(*args, **kwargs):
    """Wrapper for aten.sigmoid"""
    return _remote_kernel_fallback(torch.ops.aten.sigmoid, *args, **kwargs)
def _sigmoid_out_wrapper(*args, **kwargs):
    """Wrapper for aten.sigmoid.out"""
    return _remote_kernel_fallback(torch.ops.aten.sigmoid.out, *args, **kwargs)
def _sigmoid__wrapper(*args, **kwargs):
    """Wrapper for aten.sigmoid_"""
    return _remote_kernel_fallback(torch.ops.aten.sigmoid_, *args, **kwargs)
def _sign_wrapper(*args, **kwargs):
    """Wrapper for aten.sign"""
    return _remote_kernel_fallback(torch.ops.aten.sign, *args, **kwargs)
def _sign_out_wrapper(*args, **kwargs):
    """Wrapper for aten.sign.out"""
    return _remote_kernel_fallback(torch.ops.aten.sign.out, *args, **kwargs)
def _sign__wrapper(*args, **kwargs):
    """Wrapper for aten.sign_"""
    return _remote_kernel_fallback(torch.ops.aten.sign_, *args, **kwargs)
def _sin_wrapper(*args, **kwargs):
    """Wrapper for aten.sin"""
    return _remote_kernel_fallback(torch.ops.aten.sin, *args, **kwargs)
def _sin_out_wrapper(*args, **kwargs):
    """Wrapper for aten.sin.out"""
    return _remote_kernel_fallback(torch.ops.aten.sin.out, *args, **kwargs)
def _sin__wrapper(*args, **kwargs):
    """Wrapper for aten.sin_"""
    return _remote_kernel_fallback(torch.ops.aten.sin_, *args, **kwargs)
def _sinh_wrapper(*args, **kwargs):
    """Wrapper for aten.sinh"""
    return _remote_kernel_fallback(torch.ops.aten.sinh, *args, **kwargs)
def _sinh_out_wrapper(*args, **kwargs):
    """Wrapper for aten.sinh.out"""
    return _remote_kernel_fallback(torch.ops.aten.sinh.out, *args, **kwargs)
def _sinh__wrapper(*args, **kwargs):
    """Wrapper for aten.sinh_"""
    return _remote_kernel_fallback(torch.ops.aten.sinh_, *args, **kwargs)
def _slice_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.slice.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.slice.Tensor, *args, **kwargs)
def _slice_scatter_wrapper(*args, **kwargs):
    """Wrapper for aten.slice_scatter"""
    return _remote_kernel_fallback(torch.ops.aten.slice_scatter, *args, **kwargs)
def _slice_scatter_out_wrapper(*args, **kwargs):
    """Wrapper for aten.slice_scatter.out"""
    return _remote_kernel_fallback(torch.ops.aten.slice_scatter.out, *args, **kwargs)
def _sort_wrapper(*args, **kwargs):
    """Wrapper for aten.sort"""
    return _remote_kernel_fallback(torch.ops.aten.sort, *args, **kwargs)
def _split_with_sizes_wrapper(*args, **kwargs):
    """Wrapper for aten.split_with_sizes"""
    return _remote_kernel_fallback(torch.ops.aten.split_with_sizes, *args, **kwargs)
def _sqrt_wrapper(*args, **kwargs):
    """Wrapper for aten.sqrt"""
    return _remote_kernel_fallback(torch.ops.aten.sqrt, *args, **kwargs)
def _sqrt_out_wrapper(*args, **kwargs):
    """Wrapper for aten.sqrt.out"""
    return _remote_kernel_fallback(torch.ops.aten.sqrt.out, *args, **kwargs)
def _sqrt__wrapper(*args, **kwargs):
    """Wrapper for aten.sqrt_"""
    return _remote_kernel_fallback(torch.ops.aten.sqrt_, *args, **kwargs)
def _squeeze_dim_wrapper(*args, **kwargs):
    """Wrapper for aten.squeeze.dim"""
    return _remote_kernel_fallback(torch.ops.aten.squeeze.dim, *args, **kwargs)
def _squeeze_dims_wrapper(*args, **kwargs):
    """Wrapper for aten.squeeze.dims"""
    return _remote_kernel_fallback(torch.ops.aten.squeeze.dims, *args, **kwargs)
def _squeeze__dim_wrapper(*args, **kwargs):
    """Wrapper for aten.squeeze_.dim"""
    return _remote_kernel_fallback(torch.ops.aten.squeeze_.dim, *args, **kwargs)
def _squeeze__dims_wrapper(*args, **kwargs):
    """Wrapper for aten.squeeze_.dims"""
    return _remote_kernel_fallback(torch.ops.aten.squeeze_.dims, *args, **kwargs)
def _sub_Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.sub.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.sub.Scalar, *args, **kwargs)
def _sub_Scalar_out_wrapper(*args, **kwargs):
    """Wrapper for aten.sub.Scalar_out"""
    return _remote_kernel_fallback(torch.ops.aten.sub.Scalar_out, *args, **kwargs)
def _sub_Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.sub.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.sub.Tensor, *args, **kwargs)
def _sub_out_wrapper(*args, **kwargs):
    """Wrapper for aten.sub.out"""
    return _remote_kernel_fallback(torch.ops.aten.sub.out, *args, **kwargs)
def _sub__Scalar_wrapper(*args, **kwargs):
    """Wrapper for aten.sub_.Scalar"""
    return _remote_kernel_fallback(torch.ops.aten.sub_.Scalar, *args, **kwargs)
def _sub__Tensor_wrapper(*args, **kwargs):
    """Wrapper for aten.sub_.Tensor"""
    return _remote_kernel_fallback(torch.ops.aten.sub_.Tensor, *args, **kwargs)
def _sum_dim_IntList_wrapper(*args, **kwargs):
    """Wrapper for aten.sum.dim_IntList"""
    return _remote_kernel_fallback(torch.ops.aten.sum.dim_IntList, *args, **kwargs)
def _sum_out_wrapper(*args, **kwargs):
    """Wrapper for aten.sum.out"""
    return _remote_kernel_fallback(torch.ops.aten.sum.out, *args, **kwargs)
def _sym_numel_wrapper(*args, **kwargs):
    """Wrapper for aten.sym_numel"""
    return _remote_kernel_fallback(torch.ops.aten.sym_numel, *args, **kwargs)
def _sym_size_int_wrapper(*args, **kwargs):
    """Wrapper for aten.sym_size.int"""
    return _remote_kernel_fallback(torch.ops.aten.sym_size.int, *args, **kwargs)
def _sym_storage_offset_wrapper(*args, **kwargs):
    """Wrapper for aten.sym_storage_offset"""
    return _remote_kernel_fallback(torch.ops.aten.sym_storage_offset, *args, **kwargs)
def _sym_stride_int_wrapper(*args, **kwargs):
    """Wrapper for aten.sym_stride.int"""
    return _remote_kernel_fallback(torch.ops.aten.sym_stride.int, *args, **kwargs)
def _tan_wrapper(*args, **kwargs):
    """Wrapper for aten.tan"""
    return _remote_kernel_fallback(torch.ops.aten.tan, *args, **kwargs)
def _tan_out_wrapper(*args, **kwargs):
    """Wrapper for aten.tan.out"""
    return _remote_kernel_fallback(torch.ops.aten.tan.out, *args, **kwargs)
def _tan__wrapper(*args, **kwargs):
    """Wrapper for aten.tan_"""
    return _remote_kernel_fallback(torch.ops.aten.tan_, *args, **kwargs)
def _tanh_wrapper(*args, **kwargs):
    """Wrapper for aten.tanh"""
    return _remote_kernel_fallback(torch.ops.aten.tanh, *args, **kwargs)
def _tanh_out_wrapper(*args, **kwargs):
    """Wrapper for aten.tanh.out"""
    return _remote_kernel_fallback(torch.ops.aten.tanh.out, *args, **kwargs)
def _tanh__wrapper(*args, **kwargs):
    """Wrapper for aten.tanh_"""
    return _remote_kernel_fallback(torch.ops.aten.tanh_, *args, **kwargs)
def _topk_wrapper(*args, **kwargs):
    """Wrapper for aten.topk"""
    return _remote_kernel_fallback(torch.ops.aten.topk, *args, **kwargs)
def _trunc_wrapper(*args, **kwargs):
    """Wrapper for aten.trunc"""
    return _remote_kernel_fallback(torch.ops.aten.trunc, *args, **kwargs)
def _trunc_out_wrapper(*args, **kwargs):
    """Wrapper for aten.trunc.out"""
    return _remote_kernel_fallback(torch.ops.aten.trunc.out, *args, **kwargs)
def _trunc__wrapper(*args, **kwargs):
    """Wrapper for aten.trunc_"""
    return _remote_kernel_fallback(torch.ops.aten.trunc_, *args, **kwargs)
def _unsqueeze_wrapper(*args, **kwargs):
    """Wrapper for aten.unsqueeze"""
    return _remote_kernel_fallback(torch.ops.aten.unsqueeze, *args, **kwargs)
def _unsqueeze__wrapper(*args, **kwargs):
    """Wrapper for aten.unsqueeze_"""
    return _remote_kernel_fallback(torch.ops.aten.unsqueeze_, *args, **kwargs)
def _upsample_bilinear2d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.upsample_bilinear2d.out"""
    return _remote_kernel_fallback(torch.ops.aten.upsample_bilinear2d.out, *args, **kwargs)
def _upsample_bilinear2d_vec_wrapper(*args, **kwargs):
    """Wrapper for aten.upsample_bilinear2d.vec"""
    return _remote_kernel_fallback(torch.ops.aten.upsample_bilinear2d.vec, *args, **kwargs)
def _upsample_bilinear2d_vec_out_wrapper(*args, **kwargs):
    """Wrapper for aten.upsample_bilinear2d.vec_out"""
    return _remote_kernel_fallback(torch.ops.aten.upsample_bilinear2d.vec_out, *args, **kwargs)
def _upsample_nearest2d_out_wrapper(*args, **kwargs):
    """Wrapper for aten.upsample_nearest2d.out"""
    return _remote_kernel_fallback(torch.ops.aten.upsample_nearest2d.out, *args, **kwargs)
def _upsample_nearest2d_vec_wrapper(*args, **kwargs):
    """Wrapper for aten.upsample_nearest2d.vec"""
    return _remote_kernel_fallback(torch.ops.aten.upsample_nearest2d.vec, *args, **kwargs)
def _upsample_nearest2d_vec_out_wrapper(*args, **kwargs):
    """Wrapper for aten.upsample_nearest2d.vec_out"""
    return _remote_kernel_fallback(torch.ops.aten.upsample_nearest2d.vec_out, *args, **kwargs)
def _var_correction_wrapper(*args, **kwargs):
    """Wrapper for aten.var.correction"""
    return _remote_kernel_fallback(torch.ops.aten.var.correction, *args, **kwargs)
def _var_correction_out_wrapper(*args, **kwargs):
    """Wrapper for aten.var.correction_out"""
    return _remote_kernel_fallback(torch.ops.aten.var.correction_out, *args, **kwargs)
def _var_dim_wrapper(*args, **kwargs):
    """Wrapper for aten.var.dim"""
    return _remote_kernel_fallback(torch.ops.aten.var.dim, *args, **kwargs)
def _var_out_wrapper(*args, **kwargs):
    """Wrapper for aten.var.out"""
    return _remote_kernel_fallback(torch.ops.aten.var.out, *args, **kwargs)
def _where_self_wrapper(*args, **kwargs):
    """Wrapper for aten.where.self"""
    return _remote_kernel_fallback(torch.ops.aten.where.self, *args, **kwargs)
def _where_self_out_wrapper(*args, **kwargs):
    """Wrapper for aten.where.self_out"""
    return _remote_kernel_fallback(torch.ops.aten.where.self_out, *args, **kwargs)

# Register all core ATen operators (sorted alphabetically)
_mycelya_lib_aten.impl("_adaptive_avg_pool2d", __adaptive_avg_pool2d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_adaptive_avg_pool2d.out", __adaptive_avg_pool2d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_adaptive_avg_pool2d_backward", __adaptive_avg_pool2d_backward_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_adaptive_avg_pool2d_backward.out", __adaptive_avg_pool2d_backward_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_adaptive_avg_pool3d", __adaptive_avg_pool3d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_adaptive_avg_pool3d.out", __adaptive_avg_pool3d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_cdist_forward", __cdist_forward_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_cdist_forward.out", __cdist_forward_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_embedding_bag", __embedding_bag_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_embedding_bag.out", __embedding_bag_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_fft_r2c", __fft_r2c_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_fft_r2c.out", __fft_r2c_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_log_softmax", __log_softmax_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_log_softmax.out", __log_softmax_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_native_batch_norm_legit", __native_batch_norm_legit_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_native_batch_norm_legit.no_stats", __native_batch_norm_legit_no_stats_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_native_batch_norm_legit.no_stats_out", __native_batch_norm_legit_no_stats_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_native_batch_norm_legit.out", __native_batch_norm_legit_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_native_batch_norm_legit_no_training", __native_batch_norm_legit_no_training_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_native_batch_norm_legit_no_training.out", __native_batch_norm_legit_no_training_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_pdist_forward", __pdist_forward_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_pdist_forward.out", __pdist_forward_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_softmax", __softmax_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("_softmax.out", __softmax_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("abs", _abs_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("abs.out", _abs_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("abs_", _abs__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("acos", _acos_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("acos.out", _acos_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("acos_", _acos__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("acosh", _acosh_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("acosh.out", _acosh_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("acosh_", _acosh__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("adaptive_avg_pool1d", _adaptive_avg_pool1d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("adaptive_avg_pool1d.out", _adaptive_avg_pool1d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("add.Scalar", _add_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("add.Scalar_out", _add_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("add.Tensor", _add_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("add.out", _add_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("add_.Scalar", _add__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("add_.Tensor", _add__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("addmm", _addmm_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("addmm.out", _addmm_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("addmm_", _addmm__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("amax", _amax_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("amax.out", _amax_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("amin", _amin_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("amin.out", _amin_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("any", _any_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("any.dim", _any_dim_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("any.dims", _any_dims_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("any.dims_out", _any_dims_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("any.out", _any_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("arange.out", _arange_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("arange.start_step", _arange_start_step_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("argmax", _argmax_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("argmax.out", _argmax_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("argmin", _argmin_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("argmin.out", _argmin_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("asin", _asin_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("asin.out", _asin_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("asin_", _asin__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("asinh", _asinh_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("asinh.out", _asinh_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("asinh_", _asinh__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("atan", _atan_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("atan.out", _atan_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("atan2", _atan2_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("atan2.out", _atan2_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("atan2_", _atan2__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("atan_", _atan__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("atanh", _atanh_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("atanh.out", _atanh_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("atanh_", _atanh__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("avg_pool1d", _avg_pool1d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("avg_pool1d.out", _avg_pool1d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("avg_pool2d", _avg_pool2d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("avg_pool2d.out", _avg_pool2d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("avg_pool2d_backward", _avg_pool2d_backward_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("avg_pool3d", _avg_pool3d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("avg_pool3d.out", _avg_pool3d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_and.Scalar", _bitwise_and_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_and.Scalar_out", _bitwise_and_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_and.Tensor", _bitwise_and_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_and.Tensor_out", _bitwise_and_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_and_.Scalar", _bitwise_and__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_and_.Tensor", _bitwise_and__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_not", _bitwise_not_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_not.out", _bitwise_not_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_not_", _bitwise_not__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_or.Scalar", _bitwise_or_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_or.Scalar_out", _bitwise_or_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_or.Tensor", _bitwise_or_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_or.Tensor_out", _bitwise_or_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_or_.Scalar", _bitwise_or__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_or_.Tensor", _bitwise_or__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_xor.Scalar", _bitwise_xor_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_xor.Scalar_out", _bitwise_xor_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_xor.Tensor", _bitwise_xor_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_xor.Tensor_out", _bitwise_xor_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_xor_.Scalar", _bitwise_xor__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bitwise_xor_.Tensor", _bitwise_xor__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bmm", _bmm_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("bmm.out", _bmm_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cat", _cat_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cat.out", _cat_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ceil", _ceil_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ceil.out", _ceil_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ceil_", _ceil__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("clamp", _clamp_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("clamp.Tensor", _clamp_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("clamp.Tensor_out", _clamp_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("clamp.out", _clamp_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("clamp_", _clamp__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("clamp_.Tensor", _clamp__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("clone", _clone_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("clone.out", _clone_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("col2im", _col2im_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("col2im.out", _col2im_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("constant_pad_nd", _constant_pad_nd_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("constant_pad_nd.out", _constant_pad_nd_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("convolution", _convolution_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("convolution.out", _convolution_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("convolution_backward", _convolution_backward_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("convolution_backward.out", _convolution_backward_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cos", _cos_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cos.out", _cos_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cos_", _cos__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cosh", _cosh_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cosh.out", _cosh_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cosh_", _cosh__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cumsum", _cumsum_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cumsum.out", _cumsum_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("cumsum_", _cumsum__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("diagonal", _diagonal_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div.Scalar", _div_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div.Scalar_mode", _div_Scalar_mode_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div.Scalar_mode_out", _div_Scalar_mode_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div.Scalar_out", _div_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div.Tensor", _div_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div.Tensor_mode", _div_Tensor_mode_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div.out", _div_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div_.Scalar", _div__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div_.Scalar_mode", _div__Scalar_mode_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div_.Tensor", _div__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("div_.Tensor_mode", _div__Tensor_mode_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("elu", _elu_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("elu.out", _elu_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("elu_", _elu__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("embedding", _embedding_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("embedding.out", _embedding_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("embedding_dense_backward", _embedding_dense_backward_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("embedding_dense_backward.out", _embedding_dense_backward_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("eq.Scalar", _eq_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("eq.Scalar_out", _eq_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("eq.Tensor", _eq_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("eq.Tensor_out", _eq_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("eq_.Scalar", _eq__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("eq_.Tensor", _eq__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("erf", _erf_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("erf.out", _erf_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("erf_", _erf__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("exp", _exp_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("exp.out", _exp_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("exp_", _exp__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("expand", _expand_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("expm1", _expm1_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("expm1.out", _expm1_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("expm1_", _expm1__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("fill.Scalar", _fill_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("fill.Scalar_out", _fill_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("fill_.Scalar", _fill__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("flip", _flip_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("flip.out", _flip_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("floor", _floor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("floor.out", _floor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("floor_", _floor__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("fmod.Scalar", _fmod_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("fmod.Scalar_out", _fmod_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("fmod.Tensor", _fmod_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("fmod.Tensor_out", _fmod_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("fmod_.Scalar", _fmod__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("fmod_.Tensor", _fmod__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("full", _full_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("full.out", _full_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("full_like", _full_like_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("full_like.out", _full_like_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gather", _gather_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gather.out", _gather_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ge.Scalar", _ge_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ge.Scalar_out", _ge_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ge.Tensor", _ge_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ge.Tensor_out", _ge_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ge_.Scalar", _ge__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ge_.Tensor", _ge__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gelu", _gelu_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gelu.out", _gelu_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gelu_", _gelu__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("grid_sampler_2d", _grid_sampler_2d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("grid_sampler_2d.out", _grid_sampler_2d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gt.Scalar", _gt_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gt.Scalar_out", _gt_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gt.Tensor", _gt_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gt.Tensor_out", _gt_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gt_.Scalar", _gt__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("gt_.Tensor", _gt__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("hardtanh", _hardtanh_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("hardtanh.out", _hardtanh_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("hardtanh_", _hardtanh__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("index.Tensor", _index_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("index.Tensor_out", _index_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("index_put", _index_put_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("index_put.out", _index_put_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("index_put_", _index_put__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("index_select", _index_select_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("index_select.out", _index_select_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("isinf", _isinf_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("isinf.out", _isinf_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("isnan", _isnan_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("isnan.out", _isnan_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("le.Scalar", _le_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("le.Scalar_out", _le_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("le.Tensor", _le_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("le.Tensor_out", _le_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("le_.Scalar", _le__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("le_.Tensor", _le__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("leaky_relu", _leaky_relu_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("leaky_relu.out", _leaky_relu_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("leaky_relu_", _leaky_relu__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log", _log_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log.out", _log_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log10", _log10_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log10.out", _log10_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log10_", _log10__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log1p", _log1p_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log1p.out", _log1p_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log1p_", _log1p__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log2", _log2_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log2.out", _log2_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log2_", _log2__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("log_", _log__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_and", _logical_and_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_and.out", _logical_and_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_and_", _logical_and__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_not", _logical_not_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_not.out", _logical_not_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_not_", _logical_not__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_or", _logical_or_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_or.out", _logical_or_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_or_", _logical_or__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_xor", _logical_xor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_xor.out", _logical_xor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("logical_xor_", _logical_xor__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("lt.Scalar", _lt_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("lt.Scalar_out", _lt_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("lt.Tensor", _lt_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("lt.Tensor_out", _lt_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("lt_.Scalar", _lt__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("lt_.Tensor", _lt__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("masked_scatter", _masked_scatter_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("masked_scatter.out", _masked_scatter_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("masked_scatter_", _masked_scatter__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("max.dim", _max_dim_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("max.out", _max_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("max_pool2d_with_indices", _max_pool2d_with_indices_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("max_pool2d_with_indices.out", _max_pool2d_with_indices_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("max_pool2d_with_indices_backward", _max_pool2d_with_indices_backward_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("max_pool3d_with_indices", _max_pool3d_with_indices_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("max_pool3d_with_indices.out", _max_pool3d_with_indices_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("maximum", _maximum_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("maximum.out", _maximum_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mean", _mean_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mean.dim", _mean_dim_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mean.out", _mean_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("min.dim", _min_dim_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("min.out", _min_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("minimum", _minimum_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("minimum.out", _minimum_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mm", _mm_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mm.out", _mm_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mul.Scalar", _mul_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mul.Scalar_out", _mul_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mul.Tensor", _mul_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mul.out", _mul_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mul_.Scalar", _mul__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("mul_.Tensor", _mul__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("native_dropout", _native_dropout_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("native_dropout.out", _native_dropout_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("native_group_norm", _native_group_norm_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("native_group_norm.out", _native_group_norm_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("native_group_norm_backward", _native_group_norm_backward_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("native_group_norm_backward.out", _native_group_norm_backward_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("native_layer_norm", _native_layer_norm_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("native_layer_norm.out", _native_layer_norm_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("native_layer_norm_backward", _native_layer_norm_backward_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("native_layer_norm_backward.out", _native_layer_norm_backward_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ne.Scalar", _ne_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ne.Scalar_out", _ne_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ne.Tensor", _ne_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ne.Tensor_out", _ne_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ne_.Scalar", _ne__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("ne_.Tensor", _ne__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("neg", _neg_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("neg.out", _neg_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("neg_", _neg__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("nonzero", _nonzero_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("nonzero.out", _nonzero_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("permute", _permute_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("pow.Scalar", _pow_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("pow.Scalar_out", _pow_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("pow.Tensor_Scalar", _pow_Tensor_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("pow.Tensor_Scalar_out", _pow_Tensor_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("pow.Tensor_Tensor", _pow_Tensor_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("pow.Tensor_Tensor_out", _pow_Tensor_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("pow_.Scalar", _pow__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("prod", _prod_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("prod.dim_int", _prod_dim_int_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("prod.out", _prod_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("rand", _rand_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("rand.out", _rand_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("randn", _randn_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("randn.out", _randn_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("randperm", _randperm_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("randperm.out", _randperm_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("reciprocal", _reciprocal_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("reciprocal.out", _reciprocal_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("reciprocal_", _reciprocal__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("reflection_pad1d", _reflection_pad1d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("reflection_pad1d.out", _reflection_pad1d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("reflection_pad2d", _reflection_pad2d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("reflection_pad2d.out", _reflection_pad2d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("reflection_pad3d", _reflection_pad3d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("reflection_pad3d.out", _reflection_pad3d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("relu", _relu_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("relu.out", _relu_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("relu_", _relu__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("remainder.Scalar", _remainder_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("remainder.Scalar_out", _remainder_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("remainder.Tensor", _remainder_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("remainder.Tensor_out", _remainder_Tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("remainder_.Scalar", _remainder__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("remainder_.Tensor", _remainder__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("repeat", _repeat_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("repeat.out", _repeat_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("replication_pad2d", _replication_pad2d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("replication_pad2d.out", _replication_pad2d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("replication_pad3d", _replication_pad3d_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("replication_pad3d.out", _replication_pad3d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("round", _round_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("round.out", _round_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("round_", _round__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("rsqrt", _rsqrt_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("rsqrt.out", _rsqrt_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("rsqrt_", _rsqrt__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scalar_tensor", _scalar_tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scalar_tensor.out", _scalar_tensor_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter.src", _scatter_src_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter.src_out", _scatter_src_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter.value", _scatter_value_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter.value_out", _scatter_value_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter_.src", _scatter__src_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter_.value", _scatter__value_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter_add", _scatter_add_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter_add.out", _scatter_add_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter_add_", _scatter_add__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter_reduce.two", _scatter_reduce_two_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter_reduce.two_out", _scatter_reduce_two_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("scatter_reduce_.two", _scatter_reduce__two_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("select.int", _select_int_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("select_scatter", _select_scatter_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("select_scatter.out", _select_scatter_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sigmoid", _sigmoid_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sigmoid.out", _sigmoid_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sigmoid_", _sigmoid__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sign", _sign_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sign.out", _sign_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sign_", _sign__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sin", _sin_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sin.out", _sin_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sin_", _sin__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sinh", _sinh_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sinh.out", _sinh_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sinh_", _sinh__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("slice.Tensor", _slice_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("slice_scatter", _slice_scatter_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("slice_scatter.out", _slice_scatter_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sort", _sort_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("split_with_sizes", _split_with_sizes_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sqrt", _sqrt_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sqrt.out", _sqrt_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sqrt_", _sqrt__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("squeeze.dim", _squeeze_dim_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("squeeze.dims", _squeeze_dims_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("squeeze_.dim", _squeeze__dim_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("squeeze_.dims", _squeeze__dims_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sub.Scalar", _sub_Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sub.Scalar_out", _sub_Scalar_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sub.Tensor", _sub_Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sub.out", _sub_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sub_.Scalar", _sub__Scalar_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sub_.Tensor", _sub__Tensor_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sum.dim_IntList", _sum_dim_IntList_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sum.out", _sum_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sym_numel", _sym_numel_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sym_size.int", _sym_size_int_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sym_storage_offset", _sym_storage_offset_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("sym_stride.int", _sym_stride_int_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("tan", _tan_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("tan.out", _tan_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("tan_", _tan__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("tanh", _tanh_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("tanh.out", _tanh_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("tanh_", _tanh__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("topk", _topk_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("trunc", _trunc_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("trunc.out", _trunc_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("trunc_", _trunc__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("unsqueeze", _unsqueeze_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("unsqueeze_", _unsqueeze__wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("upsample_bilinear2d.out", _upsample_bilinear2d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("upsample_bilinear2d.vec", _upsample_bilinear2d_vec_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("upsample_bilinear2d.vec_out", _upsample_bilinear2d_vec_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("upsample_nearest2d.out", _upsample_nearest2d_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("upsample_nearest2d.vec", _upsample_nearest2d_vec_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("upsample_nearest2d.vec_out", _upsample_nearest2d_vec_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("var.correction", _var_correction_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("var.correction_out", _var_correction_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("var.dim", _var_dim_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("var.out", _var_out_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("where.self", _where_self_wrapper, dispatch_key="PrivateUse1")
_mycelya_lib_aten.impl("where.self_out", _where_self_out_wrapper, dispatch_key="PrivateUse1")
