import sys
import time

# 将 sam2 目录添加到 Python 路径中，以便能顺利导入其模块
sys.path.insert(0, "sam2")

import os
import shutil
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor


def draw_mask(img, mask, color=[0, 0, 255], alpha=0.6):  # 默认颜色改为红色 BGR
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


def save_video_masks(
        video_path,
        masks_dict,
        output_fold="data/test_video",
):
    """
    将跟踪到的掩码逐帧保存为图片和视频。
    """
    if os.path.exists(output_fold):
        shutil.rmtree(output_fold)
    os.makedirs(output_fold, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    frame_idx = 0
    print(f"正在保存掩码到文件夹: {output_fold}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="保存视频帧")

    while cap.isOpened():
        ret, frame = cap.read()  # frame 是 BGR 格式
        if not ret:
            break

        if frame_idx in masks_dict:
            mask = masks_dict[frame_idx]

            masked_frame = draw_mask(frame.copy(), mask)

            # 为了保存黑白掩码，我们再次确保它是二维的 0/1 数组
            bw_mask = (np.squeeze(mask) > 0).astype(np.uint8)

            mask_save_path = os.path.join(output_fold, f"{frame_idx:05d}_mask.png")
            cv2.imwrite(mask_save_path, bw_mask * 255)

            gen_save_path = os.path.join(output_fold, f"{frame_idx:05d}_gen.png")
            cv2.imwrite(gen_save_path, masked_frame)

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print("所有帧处理完毕，结果已保存。")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

print("正在构建和加载 SAM2VideoPredictor 模型...")
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
print("模型加载完成。")

video_path = "./sam2/notebooks/videos/bedroom.mp4"
initial_frame_idx = 0
input_point = np.array([[261, 170]])
input_label = np.array([1])
object_id = 1



def run_segmentation(predictor, video_path, frame_idx, obj_id, points=None, labels=None, box=None):
    """
    一个统一的函数来处理所有类型的提示（点、框、或组合）。
    """
    # 步骤 1: 初始化
    print("步骤 1: 初始化推理状态...")
    inference_state = predictor.init_state(video_path)

    # 步骤 2: 添加初始提示并获取初始掩码
    print(f"步骤 2: 在第 {frame_idx} 帧添加初始提示...")
    frame_idx_out, obj_ids_out, masks_out = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
        box=box
    )

    # 只关心第一个（也是唯一一个）对象的掩码
    initial_mask = masks_out[0:1, ...].cpu().numpy()

    # 将初始掩码存入结果字典
    all_masks = {frame_idx_out: initial_mask}
    print("获取初始掩码完成。")

    # 步骤 3: 传播掩码
    print("步骤 3: 传播掩码...")
    propagation_generator = predictor.propagate_in_video(inference_state)

    for f_idx, o_ids, m_out in propagation_generator:
        # 同样，只取第一个对象的掩码
        all_masks[f_idx] = m_out[0:1, ...].cpu().numpy()

    print("掩码传播完成。")
    return all_masks


if __name__ == "__main__":
    # --- 模型和数据准备 ---
    start_time = time.perf_counter() # <--- 开始计时
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    video_path = r"D:\Mongo\sam2_opt\sam2\notebooks\videos\bedroom.mp4"  # <--- 你的视频路径
    initial_frame_idx = 0
    object_id = 1
    # 选择要测试的模式

    # --- 1: 多点输入 (前景+背景) ---
    # print("测试模式: 多点输入")
    # input_points = np.array([[257, 176], [235, 286]])  # 前景点 + 背景点
    # input_labels = np.array([1, 0])  # 前景标签 + 背景标签
    # final_masks = run_segmentation(
    #     predictor, video_path, initial_frame_idx, object_id,
    #     points=input_points, labels=input_labels
    # )

    # --- 2: 边界框输入 ---
    # print("测试模式: 边界框输入")
    # input_box = np.array([161, 138, 291, 415]) # [x1, y1, x2, y2]
    # final_masks = run_segmentation(
    #     predictor, video_path, initial_frame_idx, object_id,
    #     box=input_box
    # )

    # # --- 3: 边界框 + 一个前景点 ---
    print("测试模式: 边界框 + 前景点")
    input_points = np.array([[257, 176]]) # 小孩身上的点
    input_labels = np.array([1])
    input_box = np.array([161, 138, 291, 415]) # 大致框住小孩的框
    final_masks = run_segmentation(
        predictor, video_path, initial_frame_idx, object_id,
        points=input_points, labels=input_labels, box=input_box
    )

    # =================================================================

    save_video_masks(video_path, final_masks, output_fold="data/test_video")
    end_time = time.perf_counter()  # <--- 结束计时
    duration = end_time - start_time
    print(f"\n方法总耗时: {duration:.2f} 秒")
    print("\n模拟运行成功！请检查 'data/test_video' 文件夹中的输出结果。")