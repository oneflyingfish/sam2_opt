# import sys
# import os
# import torch
# import onnx
#
# # 将 sam2 目录添加到 Python 路径中
# sys.path.insert(0, "sam2")
#
# from sam2.build_sam import build_sam2_video_predictor
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# onnx_path = "models_video"
#
# # 模型配置
# sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
#
# print("正在构建和加载 SAM2VideoPredictor 模型...")
# predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
# predictor.eval()
# print("模型加载完成。")
#
#
# def simplify_and_save(onnx_model, save_path):
#     """辅助函数，用于简化和保存 ONNX 模型。"""
#     try:
#         from onnxsim import simplify
#         print(f"Simplifying ONNX model: {save_path}")
#         model_simp, check = simplify(onnx_model)
#         assert check, "Simplified ONNX model could not be validated"
#         onnx.save(model_simp, save_path)
#         print(f"Successfully simplified and saved to {save_path}")
#     except Exception as e:
#         print(f"Error during simplification: {e}. Saving the original model.")
#         onnx.save(onnx_model, save_path)
#
#
# @torch.no_grad()
# def export_prompt_encoder(onnx_name="video_prompt_encoder.onnx", simplify_onnx=True, override=False):
#     global predictor, onnx_path
#     os.makedirs(onnx_path, exist_ok=True)
#     save_path = os.path.join(onnx_path, onnx_name)
#     if os.path.exists(save_path) and not override: return
#     # 直接导出 predictor 的子模块 sam_prompt_encoder
#     prompt_encoder = predictor.sam_prompt_encoder
#     # 创建假的提示输入：1个批次，2个点，每个点2个坐标 (x, y)
#     points_coords = torch.randint(0, 1024, (1, 2, 2), dtype=torch.float, device=device)
#     # 创建对应的标签：1 表示前景点，0 表示背景点
#     points_labels = torch.tensor([[1, 0]], dtype=torch.int, device=device)
#     # prompt_encoder 的输入是一个元组，这里只提供了点提示，所以其他提示（如 box, mask）为 None
#     args = ((points_coords, points_labels), None, None)
#     torch.onnx.export(prompt_encoder, args, save_path, export_params=True, opset_version=17, do_constant_folding=True,
#                       input_names=["point_coords", "point_labels"],
#                       output_names=["sparse_embeddings", "dense_embeddings"],
#                       dynamic_axes={"point_coords": {0: "N", 1: "num_points"},
#                                     "point_labels": {0: "N", 1: "num_points"}, "sparse_embeddings": {0: "N"},
#                                     "dense_embeddings": {0: "N"}})
#     if simplify_onnx:
#         simplify_and_save(onnx.load(save_path), save_path.replace(".onnx", "_opt.onnx"))
#
#
#
# @torch.no_grad()
# def export_memory_encoder(onnx_name="video_memory_encoder.onnx", simplify_onnx=True, override=False):
#     global predictor, onnx_path
#     os.makedirs(onnx_path, exist_ok=True)
#     save_path = os.path.join(onnx_path, onnx_name)
#     if os.path.exists(save_path) and not override: return
#
#     memory_encoder_module = predictor.memory_encoder
#
#     original_forward = memory_encoder_module.forward
#
#     memory_encoder_module.forward = memory_encoder_module.inference_memory
#
#     pixel_features = torch.randn(1, 256, 64, 64, device=device)
#     mask_for_memory = torch.rand(1, 1, 1024, 1024, device=device)
#     args = (pixel_features, mask_for_memory)
#
#     torch.onnx.export(
#         memory_encoder_module,  # 直接导出子模块
#         args,                   # 传入匹配的参数
#         save_path,
#         export_params=True,
#         opset_version=17,
#         do_constant_folding=True,
#         input_names=["pixel_features", "mask_for_memory"],
#         output_names=["mask_memory_features", "mask_memory_pos_enc"], # 与 inference_memory 的返回值对应
#         dynamic_axes={
#             "pixel_features": {0: "N"},
#             "mask_for_memory": {0: "N"},
#             "mask_memory_features": {0: "N"},
#             "mask_memory_pos_enc": {0: "N"}
#         }
#     )
#
#     memory_encoder_module.forward = original_forward
#
#     if simplify_onnx:
#         simplify_and_save(onnx.load(save_path), save_path.replace(".onnx", "_opt.onnx"))
#
# @torch.no_grad()
# def export_memory_attention(onnx_name="video_memory_attention.onnx", simplify_onnx=True, override=False):
#     global predictor, onnx_path
#     os.makedirs(onnx_path, exist_ok=True)
#     save_path = os.path.join(onnx_path, onnx_name)
#     if os.path.exists(save_path) and not override: return
#
#     memory_encoder_module = predictor.memory_encoder
#
#     original_forward = memory_encoder_module.forward
#
#     memory_encoder_module.forward = memory_encoder_module.inference_memory
#
#     pixel_features = torch.randn(1, 256, 64, 64, device=device)
#     mask_for_memory = torch.rand(1, 1, 1024, 1024, device=device)
#     args = (pixel_features, mask_for_memory)
#
#     torch.onnx.export(
#         memory_encoder_module,  # 直接导出子模块
#         args,                   # 传入匹配的参数
#         save_path,
#         export_params=True,
#         opset_version=17,
#         do_constant_folding=True,
#         input_names=["pixel_features", "mask_for_memory"],
#         output_names=["mask_memory_features", "mask_memory_pos_enc"], # 与 inference_memory 的返回值对应
#         dynamic_axes={
#             "pixel_features": {0: "N"},
#             "mask_for_memory": {0: "N"},
#             "mask_memory_features": {0: "N"},
#             "mask_memory_pos_enc": {0: "N"}
#         }
#     )
#
#     memory_encoder_module.forward = original_forward
#
#     if simplify_onnx:
#         simplify_and_save(onnx.load(save_path), save_path.replace(".onnx", "_opt.onnx"))
#
#
# if __name__ == "__main__":
#     export_functions = [
#         # export_image_encoder,
#         export_prompt_encoder,
#         # export_mask_decoder,
#         export_memory_encoder,
#     ]
#
#     for export_func in export_functions:
#         try:
#             print(f"\n>>> Running export: {export_func.__name__}")
#             export_func(override=True)
#         except Exception as e:
#             import traceback
#
#             print(f"\n[FATAL ERROR] Failed to run {export_func.__name__}: {e}")
#             traceback.print_exc()
#             print("Skipping this module and continuing with the next one.")
#
#     print("\n--- ONNX Export Process Finished ---")
#     print(f"All attempted models have been exported to the '{onnx_path}' directory.")