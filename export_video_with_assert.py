import sys
import os
import torch
import onnx
import numpy as np


sys.path.insert(0, os.path.abspath("sam2"))

from sam2.build_sam import build_sam2_video_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
onnx_path = "models_video"

sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


def simplify_and_save(onnx_model, save_path):
    """辅助函数，用于简化和保存 ONNX 模型。"""
    try:
        from onnxsim import simplify
        print(f"Simplifying ONNX model: {save_path}")
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, save_path)
        print(f"Successfully simplified and saved to {save_path}")
    except Exception as e:
        print(f"Error during simplification: {e}. Saving the original model.")
        onnx.save(onnx_model, save_path)


@torch.no_grad()
def export_image_encoder_with_assertions(onnx_name="video_image_encoder.onnx", simplify_onnx=True, override=False):
    """
    导出 Image Encoder，并验证其对输入批次大小变化的响应。
    """
    global predictor, onnx_path
    print("\n--- [Image Encoder] Assertion and Export ---")
    os.makedirs(onnx_path, exist_ok=True)
    save_path = os.path.join(onnx_path, onnx_name)
    if os.path.exists(save_path) and not override:
        print(f"Skipping, {save_path} already exists.")
        return

    original_forward = predictor.forward
    predictor.forward = predictor.inference_image
    #  上下文：在导出图像编码器之前，代码首先用不同批次大小的输入来测试模型。
    #  base_input 的批次大小是 1（shape: [1, 3, 1024, 1024]）。
    #  variant_input 的批次大小是 3（shape: [3, 3, 1024, 1024]）。
    #  具体解释：这个断言检查两件事：
    #  var_shape[0] == 3：确认当输入批次为3时，输出的批次维度也相应地变成了3。
    #  base_shape[1:] == var_shape[1:]：确认除了批次维度之外，输出的其他所有维度（如通道数、特征图尺寸）都保持不变。
    #  目的：这个assert确保了模型的第一维度（批次大小）是动态的，而其他维度是静态的。
    print(">>> 1. Assertion Phase: Verifying dynamic batch size...")
    base_input = torch.randn(1, 3, 1024, 1024).to(device)
    base_outputs = predictor(base_input)
    base_shapes = [o.shape for o in base_outputs]
    # vision_features: [1, 256, 64, 64]
    # vision_pos_enc0: [1, 256, 64, 64] ...
    # backbone_fpn0: [1, 256, 64, 64]  ...

    variant_input = torch.randn(3, 3, 1024, 1024).to(device)
    variant_outputs = predictor(variant_input)
    variant_shapes = [o.shape for o in variant_outputs]

    for i, (base_shape, var_shape) in enumerate(zip(base_shapes, variant_shapes)):
        assert var_shape[0] == 3 and base_shape[1:] == var_shape[1:]
        f"Mismatch at output {i}: base shape = {base_shape}, variant shape = {var_shape}"
    print("  - Assertions PASSED: Batch dimension is dynamic, other dimensions are static.")

    print("\n>>> 2. Export Phase...")
    input_names = ["image"]
    output_names = ["vision_features", "vision_pos_enc0", "vision_pos_enc1", "vision_pos_enc2",
                    "backbone_fpn0", "backbone_fpn1", "backbone_fpn2"]
    dynamic_axes = {name: {0: "N"} for name in input_names + output_names}

    torch.onnx.export(predictor, base_input, save_path, export_params=True, opset_version=17, do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)

    predictor.forward = original_forward
    print(f"Exported to {save_path}")
    if simplify_onnx:
        simplify_and_save(onnx.load(save_path), save_path.replace(".onnx", "_opt.onnx"))


@torch.no_grad()
def export_prompt_encoder_with_assertions(onnx_name="video_prompt_encoder.onnx", simplify_onnx=True, override=False):
    """
    导出 Prompt Encoder，并验证其对不同类型和数量提示的响应。
    """
    global predictor, onnx_path
    print("\n--- [Prompt Encoder] Assertion and Export ---")
    os.makedirs(onnx_path, exist_ok=True)
    save_path = os.path.join(onnx_path, onnx_name)
    if os.path.exists(save_path) and not override:
        print(f"Skipping, {save_path} already exists.")
        return

    prompt_encoder = predictor.sam_prompt_encoder

    # --- 验证阶段 ---
    print(">>> 1. Assertion Phase: Verifying token generation logic based on source code...")

    # 场景 1: 只有点时
    print("  - Running with points only (3 points)...")
    p_coords_var1 = torch.randint(0, 1024, (1, 3, 2), dtype=torch.float, device=device)
    p_labels_var1 = torch.tensor([[1, 0, 1]], dtype=torch.int, device=device)
    args_var1 = ((p_coords_var1, p_labels_var1), None, None)
    var1_sparse_emb, _ = prompt_encoder(*args_var1)
    print(f"    Variant 1 shapes (3 points): sparse_emb={var1_sparse_emb.shape}")
    expected_tokens_1 = 3 + 1
    assert var1_sparse_emb.shape[1] == expected_tokens_1

    # 只有点输入时，模型会为每个点生成一个 token，并额外增加一个 "padding" token
    print("  - Running with points only (4 points)...")
    p_coords_var1 = torch.randint(0, 1024, (1, 4, 2), dtype=torch.float, device=device)
    p_labels_var1 = torch.tensor([[1, 0, 1, 0]], dtype=torch.int, device=device)
    args_var1 = ((p_coords_var1, p_labels_var1), None, None)
    var1_sparse_emb, _ = prompt_encoder(*args_var1)
    print(f"    Variant 1 shapes (3 points): sparse_emb={var1_sparse_emb.shape}")
    expected_tokens_1 = 4 + 1
    assert var1_sparse_emb.shape[1] == expected_tokens_1

    # 场景 2: 只有框时。boxes 的形状应为 (B, 4)，其中 B 是框的数量，也被视作批次大小。
    print("  - Running with boxes only (2 boxes as a batch of 2)...")
    # 形状 (2, 4) 代表一个批次包含2个元素，每个元素是一个框。
    box_coords_var2 = torch.tensor([[100, 100, 400, 400], [50, 50, 150, 150]], dtype=torch.float, device=device)
    args_var2 = (None, box_coords_var2, None)
    var2_sparse_emb, _ = prompt_encoder(*args_var2)
    print(f"    Variant 2 shapes (2 boxes): sparse_emb={var2_sparse_emb.shape}")
    # 对于 (2, 4) 的输入，输出的批次大小也应为 2。每个框产生2个token。
    assert var2_sparse_emb.shape == (2, 2, 256), f"Shape mismatch for boxes-only. Got {var2_sparse_emb.shape}"

    print("  - Running with boxes only (1 box)...")
    # 形状 (1, 4) 每个元素是一个框。
    box_coords_var2 = torch.tensor([[100, 100, 400, 400]], dtype=torch.float, device=device)
    args_var2 = (None, box_coords_var2, None)
    var2_sparse_emb, _ = prompt_encoder(*args_var2)
    print(f"    Variant 2 shapes (2 boxes): sparse_emb={var2_sparse_emb.shape}")
    # 对于 (1, 4) 的输入，输出的批次大小也应为 1。每个框产生2个token。
    assert var2_sparse_emb.shape == (1, 2, 256), f"Shape mismatch for boxes-only. Got {var2_sparse_emb.shape}"

    # 场景 3: 单批次混合输入 (N=1, 2 points + 1 box)
    print("  - Running with points AND boxes (N=1, 2 points + 1 box)...")
    p_coords_var3 = torch.randint(0, 1024, (1, 2, 2), dtype=torch.float, device=device)
    p_labels_var3 = torch.tensor([[1, 0]], dtype=torch.int, device=device)
    box_coords_var3 = torch.tensor([[200, 200, 500, 500]], dtype=torch.float, device=device)  # Shape (1, 4)
    args_var3 = ((p_coords_var3, p_labels_var3), box_coords_var3, None)
    var3_sparse_emb, _ = prompt_encoder(*args_var3)
    expected_tokens_3 = 2 + 2  # num_points (2) + num_boxes (1) * 2
    assert var3_sparse_emb.shape == (1, expected_tokens_3, 256)

    # 场景 4: 多批次混合输入 (N=2)
    # 每个批次元素都有 3 个点和 1 个框
    print("  - Running with batched points AND boxes (N=2, 3 points + 1 box per element)...")
    p_coords_var4 = torch.randint(0, 1024, (2, 3, 2), dtype=torch.float, device=device)
    p_labels_var4 = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.int, device=device)
    box_coords_var4 = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float,
                                   device=device)  # Shape (2, 4)
    args_var4 = ((p_coords_var4, p_labels_var4), box_coords_var4, None)
    var4_sparse_emb, _ = prompt_encoder(*args_var4)
    expected_tokens_4 = 3 + 2  # num_points (3) + num_boxes (1) * 2
    assert var4_sparse_emb.shape == (2, expected_tokens_4, 256)

    print("  - Assertions PASSED: PromptEncoder token logic is fully verified against source code.")

    # --- 导出阶段 ---
    print("\n>>> 2. Export Phase (using points-only mode)...")
    torch.onnx.export(prompt_encoder,
                      args_var1,
                      save_path,
                      export_params=True, opset_version=17, do_constant_folding=True,
                      input_names=["point_coords", "point_labels"],
                      output_names=["sparse_embeddings", "dense_embeddings"],
                      dynamic_axes={
                          "point_coords": {0: "N", 1: "num_points"},
                          "point_labels": {0: "N", 1: "num_points"},
                          "sparse_embeddings": {0: "N", 1: "num_prompts"},
                          "dense_embeddings": {0: "N"}
                      })
    print(f"Exported to {save_path}")
    if simplify_onnx:
        simplify_and_save(onnx.load(save_path), save_path.replace(".onnx", "_opt.onnx"))


# 输入 tokens 的数量是动态的，但 MaskDecoder 的输出形状是静态的
@torch.no_grad()
def export_mask_decoder_with_assertions(onnx_name="video_mask_decoder.onnx", simplify_onnx=True, override=False):
    """
    导出 Mask Decoder，并验证其对输入 token 数量变化的响应。
    模拟 MaskDecoder 的输入 tokens，并确认其输出形状是静态的。
    """
    global predictor, onnx_path
    print("\n--- [Mask Decoder] Assertion and Export ---")
    os.makedirs(onnx_path, exist_ok=True)
    save_path = os.path.join(onnx_path, onnx_name)
    if os.path.exists(save_path) and not override:
        print(f"Skipping, {save_path} already exists.")
        return

    mask_decoder = predictor.sam_mask_decoder
    original_forward = mask_decoder.forward
    mask_decoder.forward = mask_decoder.inference_predict_masks

    # --- 验证阶段 ---
    print(">>> 1. Assertion Phase: Verifying dynamic number of tokens...")

    src = torch.randn(1, 256, 64, 64, device=device)
    pos_src = torch.randn(1, 256, 64, 64, device=device)
    high_res_feature0 = torch.randn(1, 32, 256, 256, device=device)
    high_res_feature1 = torch.randn(1, 64, 128, 128, device=device)

    num_output_tokens = 1 + 1 + mask_decoder.num_mask_tokens  # obj_score_token + iou_token + mask_tokens
    output_tokens = torch.randn(1, num_output_tokens, predictor.hidden_dim, device=device)

    # 场景 1: 2个外部prompt tokens
    print("  - Running with baseline input (2 external prompt tokens)...")
    prompt_tokens_base = torch.randn(1, 2, predictor.hidden_dim, device=device)
    tokens_base = torch.cat([output_tokens, prompt_tokens_base], dim=1)
    args_base = (src, tokens_base, pos_src, high_res_feature0, high_res_feature1)
    base_outputs = mask_decoder.forward(*args_base)
    base_shapes = [o.shape for o in base_outputs]
    print(f"    Total input tokens: {tokens_base.shape[1]}. Output shapes: {base_shapes}")

    # 场景 2: 6个外部prompt tokens
    print("  - Running with variant input (6 external prompt tokens)...")
    prompt_tokens_variant = torch.randn(1, 6, predictor.hidden_dim, device=device)
    tokens_variant = torch.cat([output_tokens, prompt_tokens_variant], dim=1)
    args_variant = (src, tokens_variant, pos_src, high_res_feature0, high_res_feature1)
    variant_outputs = mask_decoder.forward(*args_variant)
    variant_shapes = [o.shape for o in variant_outputs]
    print(f"    Total input tokens: {tokens_variant.shape[1]}. Output shapes: {variant_shapes}")

    # 断言验证：输出形状应是静态的，因为它们由固定的 num_mask_tokens 决定。
    print("  - Asserting shape changes...")
    for i in range(len(base_shapes)):
        assert base_shapes[i] == variant_shapes[i], \
            f"Output {i} shape should be static, but changed from {base_shapes[i]} to {variant_shapes[i]}"
    print("  - Assertions PASSED: All output shapes are static, correctly handling dynamic number of prompt tokens.")

    # --- 导出阶段 ---
    print("\n>>> 2. Export Phase...")
    output_names = ["masks", "iou_pred", "mask_tokens_out", "object_score_logits"]
    torch.onnx.export(mask_decoder,
                      args_variant,
                      save_path,
                      export_params=True,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=["src", "tokens", "pos_src", "high_res_feature0", "high_res_feature1"],
                      output_names=output_names,
                      dynamic_axes={
                          "src": {0: "N"},
                          "tokens": {0: "N", 1: "num_total_tokens"},  # 总token数是动态的
                          "pos_src": {0: "N"},
                          "high_res_feature0": {0: "N"},
                          "high_res_feature1": {0: "N"},
                          # 输出维度是固定的（除了批次N）
                          "masks": {0: "N"},
                          "iou_pred": {0: "N"},
                          "mask_tokens_out": {0: "N"},
                          "object_score_logits": {0: "N"}
                      })

    mask_decoder.forward = original_forward
    print(f"Exported to {save_path}")
    if simplify_onnx:
        simplify_and_save(onnx.load(save_path), save_path.replace(".onnx", "_opt.onnx"))



# SAM2VideoPredictor 总是会将多物体/多掩码的情况转换成一个更大的批次（batch），
# 对于 MemoryEncoder 这个独立的模块而言，它永远只会接收到批处理化的单通道掩码输入。
# 这意味着，在 MemoryEncoder 的视角里，不存在“单掩码 vs 多掩码”这两种模式的区别。
# 它看到的永远是 (N, 1, H, W) 格式的 masks 和 (N, C, H, W) 格式的 pix_feat，其中 N 是一个可变的批次大小。
@torch.no_grad()
def export_memory_encoder_with_assertions(onnx_name="video_memory_encoder.onnx", simplify_onnx=True, override=False):
    """
    导出 Memory Encoder，并验证其对输入批次大小变化的响应。
    【最终无Wrapper、无新方法版】：通过修改原始 forward 并直接导出 inference_memory 实现。
    """
    global predictor, onnx_path
    print("\n--- [Memory Encoder] Assertion and Export ---")
    os.makedirs(onnx_path, exist_ok=True)
    save_path = os.path.join(onnx_path, onnx_name)
    if os.path.exists(save_path) and not override:
        print(f"Skipping, {save_path} already exists.")
        return

    module = predictor.memory_encoder
    original_forward = module.forward
    module.forward = module.inference_memory

    # --- 验证阶段 ---
    print(">>> 1. Assertion Phase: Verifying dynamic batch size...")

    # 场景 1: 基准输入 (批次大小 N=1)
    print("  - Running with baseline input (N=1)...")
    pix_feat_base = torch.randn(1, 256, 64, 64, device=device)
    mask_base = torch.rand(1, 1, 1024, 1024, device=device)
    args_base = (pix_feat_base, mask_base)
    base_mem_feat, _ = module.forward(*args_base)  # 调用的是 inference_memory
    print(f"    Baseline output shape: {base_mem_feat.shape}")

    # 场景 2: 可变输入 (批次大小 N=3)
    print("  - Running with variant input (N=3)...")
    pix_feat_variant = torch.randn(3, 256, 64, 64, device=device)
    mask_variant = torch.rand(3, 1, 1024, 1024, device=device)
    args_variant = (pix_feat_variant, mask_variant)
    variant_mem_feat, _ = module.forward(*args_variant)  # 调用的是 inference_memory
    print(f"    Variant output shape: {variant_mem_feat.shape}")

    # 断言验证
    print("  - Asserting shape changes...")
    assert variant_mem_feat.shape[0] == 3
    assert base_mem_feat.shape[1:] == variant_mem_feat.shape[1:]
    print("  - Assertions PASSED: Model correctly handles dynamic batch size.")

    # --- 导出阶段 ---
    print("\n>>> 2. Export Phase...")

    # 直接导出 module，因为它的 forward 现在指向 inference_memory，是完全 ONNX 友好的
    torch.onnx.export(
        module,
        args_variant,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_features", "mask_for_memory"],
        output_names=["mask_memory_features", "mask_memory_pos_enc"],
        dynamic_axes={
            "pixel_features": {0: "N"},
            "mask_for_memory": {0: "N"},
            "mask_memory_features": {0: "N"},
            "mask_memory_pos_enc": {0: "N"}
        }
    )

    # 清理现场，恢复原始的 forward 方法
    module.forward = original_forward

    print(f"Exported to {save_path}")
    if simplify_onnx:
        simplify_and_save(onnx.load(save_path), save_path.replace(".onnx", "_opt.onnx"))



# 主执行逻辑
if __name__ == "__main__":
    print("正在构建和加载 SAM2VideoPredictor 模型...")
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    predictor.eval()
    print("模型加载完成。")

    export_functions_to_run = [
        # export_image_encoder_with_assertions,
        # export_prompt_encoder_with_assertions,
        # export_mask_decoder_with_assertions,
        # export_memory_encoder_with_assertions,
        export_memory_attention_with_assertions,  # <-- 在这里添加新函数
    ]

    for export_func in export_functions_to_run:
        try:
            export_func(override=True, simplify_onnx=True)
        except Exception as e:
            import traceback

            print(f"\n[FATAL ERROR] Failed to run {export_func.__name__}: {e}")
            traceback.print_exc()
            print("Skipping this module and continuing with the next one.")

    print(f"\n--- ONNX Rigorous Export Process Finished ---")
    print(f"All attempted models have been exported to the '{onnx_path}' directory.")