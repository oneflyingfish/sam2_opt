
# video.py

import sys
import time

from ytools.bench import test_torch_cuda_time

# 将 sam2 目录添加到 Python 路径中，以便能顺利导入其模块
sys.path.insert(0, "sam2")

import os
import shutil
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor


# =================================================================
# 辅助函数 (保持不变)
# =================================================================

def draw_mask(img, mask, color=[0, 0, 255], alpha=0.6):
    squeezed_mask = np.squeeze(mask)
    if squeezed_mask.ndim != 2:
        raise ValueError(
            f"无法将输入掩码转换为二维 (H, W) 形状。原始形状: {mask.shape}, squeeze后: {squeezed_mask.shape}")
    bool_mask = squeezed_mask > 0
    out_img = img.copy()
    color_layer = np.zeros_like(out_img, dtype=np.uint8)
    color_layer[bool_mask] = color
    out_img = cv2.addWeighted(out_img, 1, color_layer, alpha, 0)
    return out_img


def save_video_masks(video_path, masks_dict, output_fold="data/test_video"):
    if os.path.exists(output_fold):
        shutil.rmtree(output_fold)
    os.makedirs(output_fold, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="保存视频帧")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx in masks_dict:
            mask = masks_dict[frame_idx]
            masked_frame = draw_mask(frame.copy(), mask)
            bw_mask = (np.squeeze(mask) > 0).astype(np.uint8)
            mask_save_path = os.path.join(output_fold, f"{frame_idx:05d}_mask.png")
            cv2.imwrite(mask_save_path, bw_mask * 255)
            gen_save_path = os.path.join(output_fold, f"{frame_idx:05d}_gen.png")
            cv2.imwrite(gen_save_path, masked_frame)
        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()


@test_torch_cuda_time()
def run_segmentation(predictor, video_path, frame_idx, obj_id, points=None, labels=None, box=None):
    print("步骤 1: 初始化推理状态...")
    inference_state = predictor.init_state(video_path)
    print(f"步骤 2: 在第 {frame_idx} 帧添加初始提示...")
    frame_idx_out, obj_ids_out, masks_out = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
        box=box
    )
    initial_mask = masks_out[0:1, ...].cpu().numpy()
    all_masks = {frame_idx_out: initial_mask}
    print("获取初始掩码完成。")
    print("步骤 3: 传播掩码...")
    propagation_generator = predictor.propagate_in_video(inference_state)
    for f_idx, o_ids, m_out in propagation_generator:
        all_masks[f_idx] = m_out[0:1, ...].cpu().numpy()
    print("掩码传播完成。")
    return all_masks


# =================================================================
# 主程序
# =================================================================
if __name__ == "__main__":

    # --- 1. 模型和数据准备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    print("正在构建和加载 SAM2VideoPredictor 模型...")
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print("模型加载完成。")

    # --- 2. 后端配置 ---

    # 选项 A: 默认 PyTorch 后端
    print("\n--- 使用 PyTorch 后端 ---")
    predictor.set_runtime_backend(backend="torch")

    # 选项 B: ONNX Runtime 后端 (只加速 MemoryEncoder 和 PromptEncoder)
    # print("\n--- 使用 ONNX Runtime 后端 ---")
    # # 根据你的需求，只配置 memory 和 prompt encoder
    # onnx_paths = [
    #     "models_video/video_memory_encoder_opt.onnx",  # [0] for memory_encoder
    #     "models_video/video_prompt_encoder_opt.onnx",  # [1] for prompt_encoder
    # ]

    # 检查 ONNX 文件是否存在
    # for path in onnx_paths:
    #     if not os.path.exists(path):
    #         raise FileNotFoundError(f"ONNX 模型文件未找到: {path}. 请先运行导出脚本。")

    # print("成功set_onnxruntime_backend")
    # predictor.set_runtime_backend(
    #     backend="onnxruntime",
    #     args={
    #         "model_paths": onnx_paths,
    #         "providers": [
    #             "TensorrtExecutionProvider",
    #             "CUDAExecutionProvider",
    #             "CPUExecutionProvider",
    #         ],
    #     },
    # )

    # --- 3. 运行推理 ---
    video_path = "./sam2/notebooks/videos/bedroom.mp4"
    initial_frame_idx = 0
    object_id = 1

    print("\n测试模式: 边界框 + 前景点")
    input_points = np.array([[257, 176]])
    input_labels = np.array([1])
    input_box = np.array([161, 138, 291, 415])

    final_masks = run_segmentation(
        predictor, video_path, initial_frame_idx, object_id,
        points=input_points, labels=input_labels, box=input_box
    )

    # --- 4. 保存结果和计时 ---
    save_video_masks(video_path, final_masks, output_fold="data/test_video")