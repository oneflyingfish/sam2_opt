# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones
from ytools.executor import ModelExectuor

class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt,
        memory,
        query_pos,
        pos,
        num_k_exclude_rope: Tensor = None,
    ):
        # Cross-Attention
        tgt2 = self.norm2(tgt)
        if isinstance(self.cross_attn_image, RoPEAttention):
            tgt2 = self.cross_attn_image(
                q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
                k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
                v=memory,
                num_k_exclude_rope=num_k_exclude_rope,
            )
        else:
            tgt2 = self.cross_attn_image(
                q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
                k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
                v=memory,
            )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: Tensor = torch.tensor([0], dtype=torch.int32),
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        # --- new ---
        self.backend_contexts: List[ModelExectuor] = []
        self.inference_memory_attention_exclude = self.inference_memory_attention_torch
        self.inference_memory_attention_none = self.inference_memory_attention_torch
        self.set_runtime_backend(backend="torch")

    def set_runtime_backend(self, backend="torch", args: dict = None):
        """
        Dynamically sets the runtime backend for MemoryAttention (torch or onnxruntime).
        """
        if self.backend_contexts is not None:
            for context in self.backend_contexts:
                context.Release()
                
        self.backend_contexts = []
        if backend.lower() == "torch":
            self.inference_memory_attention_exclude = (
                self.inference_memory_attention_torch
            )
            self.inference_memory_attention_none = self.inference_memory_attention_torch
        elif backend.lower() == "onnxruntime":
            self.inference_memory_attention_exclude = (
                self.inference_memory_attention_speedup_exclude
            )
            self.inference_memory_attention_none = (
                self.inference_memory_attention_speedup_none
            )
            assert (
                args and "model_paths" in args
            ), "The 'model_paths' argument is required to specify the ONNX model path"

            from ytools.onnxruntime import OnnxRuntimeExecutor

            model_none, model_exclude = args["model_paths"]
            providers = args.get("providers", None)
            executor_none = OnnxRuntimeExecutor(model_none, providers=providers)
            executor_exclude = OnnxRuntimeExecutor(model_exclude, providers=providers)

            # warmup
            curr_warmup = torch.randn(4096, 1, 256)
            curr_pos_warmup = torch.randn(4096, 1, 256)

            for L in [1, 7]:
                warmup_inputs = [
                    curr_warmup,
                    torch.randn(L, 4096, 1, 64),
                    curr_pos_warmup,
                    torch.randn(L, 4096, 1, 64),
                    torch.randn(0, 1, 64),
                    torch.randn(0, 1, 64),
                ][
                    : len(executor_none.GetModelInputDesc())
                ]  # may remove last input while export onnx because it get no use
                executor_none.warmup(warmup_inputs)

                warmup_inputs = [
                    curr_warmup,
                    torch.randn(L, 4096, 1, 64),
                    curr_pos_warmup,
                    torch.randn(L, 4096, 1, 64),
                    torch.randn(64, 1, 64),
                    torch.randn(64, 1, 64),
                ]
                executor_exclude.warmup(warmup_inputs)

            self.backend_contexts.append(executor_none)
            self.backend_contexts.append(executor_exclude)

        elif backend.lower() in ["tensorrt", "trt"]:
            assert (
                "model_paths" in args
            ), 'need args["model_paths"] to set *.engine path'

            model_paths = args["model_paths"]
            if isinstance(model_paths, str):
                model_paths = [model_paths]

            from ytools.tensorrt import TensorRTExecutor
            # image_encoder
            if model_paths[0] is None:
                self.inference_memory_attention_exclude = (
                    self.inference_memory_attention_torch
                )
                self.inference_memory_attention_none = (
                    self.inference_memory_attention_torch
                )
            else:
                self.inference_memory_attention_exclude = (
                    self.inference_memory_attention_speedup_exclude
                )
                self.inference_memory_attention_none = (
                    self.inference_memory_attention_speedup_none
                )

                model_none, model_exclude = args["model_paths"]
                providers = args.get("providers", None)
                executor_none = TensorRTExecutor(
                    model_none, build_args=args.get("build_args", {})
                )
                executor_exclude = TensorRTExecutor(
                    model_exclude, build_args=args.get("build_args", {})
                )

                # warmup
                curr_warmup = torch.randn(4096, 1, 256)
                curr_pos_warmup = torch.randn(4096, 1, 256)

                for L in [1, 7]:
                    warmup_inputs = [
                        curr_warmup,
                        torch.randn(L, 4096, 1, 64),
                        curr_pos_warmup,
                        torch.randn(L, 4096, 1, 64),
                        torch.randn(0, 1, 64),
                        torch.randn(0, 1, 64),
                    ][
                        : len(executor_none.GetModelInputDesc())
                    ]  # may remove last input while export onnx because it get no use
                    executor_none.warmup(warmup_inputs)

                    warmup_inputs = [
                        curr_warmup,
                        torch.randn(L, 4096, 1, 64),
                        curr_pos_warmup,
                        torch.randn(L, 4096, 1, 64),
                        torch.randn(64, 1, 64),
                        torch.randn(64, 1, 64),
                    ]
                    executor_exclude.warmup(warmup_inputs)

                self.backend_contexts.append(executor_none)
                self.backend_contexts.append(executor_exclude)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def forward(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        num_obj_ptr_tokens: int = 0,
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        valid_len = memory.size(0) - num_obj_ptr_tokens

        inputs = (
            curr,
            memory[:valid_len, ...].unflatten(0, (-1, curr.size(0))),
            curr_pos,
            memory_pos[:valid_len, ...].unflatten(0, (-1, curr.size(0))),
            memory[valid_len:, ...],
            memory_pos[valid_len:, ...],
        )

        if num_obj_ptr_tokens > 0:
            return self.inference_memory_attention_exclude(*inputs)
        else:
            return self.inference_memory_attention_none(*inputs)
        
    def inference_memory_attention_torch(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        memory_exclude: Optional[Tensor] = None,
        memory_pos_exclude: Optional[Tensor] = None,
    ) -> Tensor:
        memory = memory.flatten(0, 1)
        memory_pos = memory_pos.flatten(0, 1)

        if memory_exclude.size(0) > 0:
            memory = torch.cat([memory, memory_exclude], dim=0)
            memory_pos = torch.cat([memory_pos, memory_pos_exclude], dim=0)

        num_k_exclude_rope = torch.tensor(
            [memory_exclude.size(0)], dtype=torch.int32, device=memory.device
        )

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            if isinstance(layer.cross_attn_image, RoPEAttention):
                output = layer(
                    tgt=output,
                    memory=memory,
                    pos=memory_pos,
                    query_pos=curr_pos,
                    num_k_exclude_rope=num_k_exclude_rope,
                )

            else:
                output = layer(
                    tgt=output, memory=memory, pos=memory_pos, query_pos=curr_pos
                )
        normed_output = self.norm(output)

        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            # curr_pos = curr_pos.transpose(0, 1) # This line was in your original code but seems unused

        return normed_output

    def inference_memory_attention_speedup_exclude(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        memory_exclude: Optional[Tensor] = None,
        memory_pos_exclude: Optional[Tensor] = None,
    ) -> Tensor:
        results = []
        for i in range(curr.shape[1]):
            outs = self.backend_contexts[1].Inference(
                [curr.narrow(dim=1,start=i,length=1), 
                memory.narrow(dim=2,start=i,length=1), 
                curr_pos.narrow(dim=1,start=i,length=1), 
                memory_pos.narrow(dim=2,start=i,length=1), 
                memory_exclude.narrow(dim=1,start=i,length=1), 
                memory_pos_exclude.narrow(dim=1,start=i,length=1)],
                output_type="torch",
            )
            results.append(outs[0].to(memory.device))
        return torch.cat(results, dim=1)


    def inference_memory_attention_speedup_none(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        memory_exclude: Optional[Tensor] = None,
        memory_pos_exclude: Optional[Tensor] = None,
    ) -> Tensor:
        outs = self.backend_contexts[0].Inference(
            [curr, memory, curr_pos, memory_pos, memory_exclude, memory_pos_exclude][
                : len(self.backend_contexts[0].GetModelInputDesc())
            ],
            output_type="torch",
        )
        return outs[0].to(memory.device)
