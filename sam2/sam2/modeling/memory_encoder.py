# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import DropPath, get_clones, LayerNorm2d

from ytools.bench import test_torch_cuda_time
from ytools.executor import ModelExectuor
from ytools.onnxruntime import OnnxRuntimeExecutor


class MaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))

    def forward(self, x):
        return self.encoder(x)


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,  # in_dim of pix_feats
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        # --- 新增代码 ---
        # 1. 初始化 backend_contexts 列表
        self.backend_contexts: List[ModelExectuor] = []

        # 2. 初始化方法指针，默认指向 PyTorch 实现
        self.inference_memory = self.inference_memory_torch

        # 3. 设置默认后端
        self.set_runtime_backend(backend="torch")

    def set_runtime_backend(self, backend="torch", args: dict = None):
        """
        动态设置 MemoryEncoder 的运行时后端 (torch 或 onnxruntime)。
        """
        self.backend_contexts = []
        if backend.lower() == "torch":
            self.inference_memory = self.inference_memory_torch
        elif backend.lower() == "onnxruntime":
            self.inference_memory = self.inference_memory_onnxruntime
            assert args and "model_paths" in args, '需要提供 "model_paths" 参数来指定 ONNX 模型路径'

            model_path = args["model_paths"][0]
            providers = args.get("providers", None)
            executor = OnnxRuntimeExecutor(model_path, providers=providers)

            print(f"Warming up ONNX Runtime for MemoryEncoder ({model_path})...")
            try:
                warmup_device = torch.device("cuda" if torch.cuda.is_available() and "CUDAExecutionProvider" in (
                            providers or ["CUDAExecutionProvider"]) else "cpu")
                pixel_features_warmup = torch.randn(1, 256, 64, 64, device=warmup_device)
                mask_for_memory_warmup = torch.rand(1, 1, 1024, 1024, device=warmup_device)
                warmup_inputs = [pixel_features_warmup, mask_for_memory_warmup]
                executor.warmup(warmup_inputs)
                print("MemoryEncoder warmup successful.")
            except Exception as e:
                print(f"[Warning] MemoryEncoder ONNX warmup failed: {e}")

            self.backend_contexts.append(executor)
        else:
            raise ValueError(f"不支持的后端: {backend}")

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)

        x,pos = self.inference_memory(pix_feat, masks)

        return {"vision_features": x, "vision_pos_enc": [pos]}

        # --- 新增: PyTorch 版本的核心逻辑 ---
    @test_torch_cuda_time()
    def inference_memory_torch(self, pix_feat: torch.Tensor, masks: torch.Tensor):
        masks_embedded = self.mask_downsampler(masks)
        pix_feat_processed = pix_feat.to(masks_embedded.device)
        x = self.pix_feat_proj(pix_feat_processed)
        x = x + masks_embedded
        x = self.fuser(x)
        x = self.out_proj(x)
        pos = self.position_encoding(x).to(x.dtype)
        return x, pos

    # --- 新增: ONNX 版本的核心逻辑 ---
    @test_torch_cuda_time()
    def inference_memory_onnxruntime(self, pix_feat: torch.Tensor, masks: torch.Tensor):
        inputs = [pix_feat, masks]
        executor = self.backend_contexts[0]
        outputs = executor.Inference(inputs, output_type="torch")

        x, pos = outputs[0], outputs[1]
        device = pix_feat.device
        return x.to(device), pos.to(device)






