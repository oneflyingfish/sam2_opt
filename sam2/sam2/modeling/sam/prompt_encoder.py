# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type, List

import torch
from torch import nn

from sam2.modeling.position_encoding import PositionEmbeddingRandom

from sam2.modeling.sam2_utils import LayerNorm2d

from ytools.bench import test_torch_cuda_time
from ytools.executor import ModelExectuor
from ytools.onnxruntime import OnnxRuntimeExecutor


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        # --- 新增: 后端切换逻辑初始化 ---
        self.backend_contexts: List[ModelExectuor] = []
        self.inference_prompt = self.inference_prompt_torch
        self.set_runtime_backend(backend="torch")

    def set_runtime_backend(self, backend="torch", args: dict = None):
        """
        动态设置 PromptEncoder 的运行时后端 (torch 或 onnxruntime)。
        """
        self.backend_contexts = []
        if backend.lower() == "torch":
            self.inference_prompt = self.inference_prompt_torch
        elif backend.lower() == "onnxruntime":
            self.inference_prompt = self.inference_prompt_onnxruntime
            assert args and "model_paths" in args, '需要提供 "model_paths" 参数来指定 ONNX 模型路径'

            model_path = args["model_paths"][0]  # PromptEncoder 只需一个模型
            providers = args.get("providers", None)
            executor = OnnxRuntimeExecutor(model_path, providers=providers)

            print(f"Warming up ONNX Runtime for PromptEncoder ({model_path})...")
            try:
                warmup_device = torch.device("cuda" if torch.cuda.is_available() and "CUDAExecutionProvider" in (
                            providers or ["CUDAExecutionProvider"]) else "cpu")
                points_coords_warmup = torch.randint(0, 1024, (1, 2, 2), dtype=torch.float, device=warmup_device)
                points_labels_warmup = torch.tensor([[1, 0]], dtype=torch.int32, device=warmup_device)

                # ONNX 模型的输入是扁平化的张量列表
                warmup_inputs = [points_coords_warmup, points_labels_warmup]
                executor.warmup(warmup_inputs)
                print("PromptEncoder warmup successful.")
            except Exception as e:
                print(f"[Warning] PromptEncoder ONNX warmup failed: {e}")

            self.backend_contexts.append(executor)
        else:
            raise ValueError(f"不支持的后端: {backend}")

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 0).unsqueeze(-1),
            point_embedding + self.point_embeddings[0].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 1).unsqueeze(-1),
            point_embedding + self.point_embeddings[1].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 2).unsqueeze(-1),
            point_embedding + self.point_embeddings[2].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 3).unsqueeze(-1),
            point_embedding + self.point_embeddings[3].weight,
            point_embedding,
        )
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inference_prompt(points, boxes, masks)
        # """
        # Embeds different types of prompts, returning both sparse and dense
        # embeddings.
        #
        # Arguments:
        #   points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
        #     and labels to embed.
        #   boxes (torch.Tensor or none): boxes to embed
        #   masks (torch.Tensor or none): masks to embed
        #
        # Returns:
        #   torch.Tensor: sparse embeddings for the points and boxes, with shape
        #     BxNx(embed_dim), where N is determined by the number of input points
        #     and boxes.
        #   torch.Tensor: dense embeddings for the masks, in the shape
        #     Bx(embed_dim)x(embed_H)x(embed_W)
        # """
        # bs = self._get_batch_size(points, boxes, masks)
        # sparse_embeddings = torch.empty(
        #     (bs, 0, self.embed_dim), device=self._get_device()
        # )
        # if points is not None:
        #     coords, labels = points
        #     point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
        #     sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        # if boxes is not None:
        #     box_embeddings = self._embed_boxes(boxes)
        #     sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        #
        # if masks is not None:
        #     dense_embeddings = self._embed_masks(masks)
        # else:
        #     dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
        #         bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        #     )
        #
        # return sparse_embeddings, dense_embeddings

# --- 新增: PyTorch 版本的核心逻辑 ---
    @test_torch_cuda_time()
    def inference_prompt_torch(self, points, boxes, masks) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1,
                                                                                     self.image_embedding_size[0],
                                                                                     self.image_embedding_size[1])
        return sparse_embeddings, dense_embeddings


    @test_torch_cuda_time()
        # --- 新增: ONNX 版本的核心逻辑 ---
    def inference_prompt_onnxruntime(self, points, boxes, masks) -> Tuple[torch.Tensor, torch.Tensor]:
        # 对于 ONNX，我们假设 box prompt 已经被转换并合并到 points 中
        # 并且 mask 输入要么存在，要么为 None (但不会被传入)
        # 我们的 ONNX 模型只接收 points_coords 和 points_labels
        assert boxes is None, "ONNX backend for PromptEncoder does not support 'boxes' directly. Convert them to points."
        assert masks is None, "ONNX backend for PromptEncoder does not support 'masks' directly. It's designed for point prompts."

        executor = self.backend_contexts[0]
        device = self._get_device()

        point_coords, point_labels = points
        inputs = [point_coords, point_labels]

        outputs = executor.Inference(inputs, output_type="torch")

        sparse_embeddings, dense_embeddings = outputs[0], outputs[1]

        return sparse_embeddings.to(device), dense_embeddings.to(device)
