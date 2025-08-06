# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

from sam2.modeling.sam2_base import SAM2Base

from sam2.utils.transforms import SAM2Transforms
from torch import nn
from ytools.executor import ModelExectuor
import os


class SAM2ImagePredictor(nn.Module):
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        self.backend_contexts = []  # type: List[ModelExectuor]
        self.set_image_e2e = self.set_image_e2e_torch
        
        # Add a forward attribute for compatibility with ONNX export scripts
        self.forward = None


    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    def release(self):
        self.speedup("torch")

    def speedup(self, backend="tensorrt", use_cache=True, model_root_path=None):
        """
        only support for large model version

        you can set backend with ["torch", "onnxruntime", "tensorrt"]

        backend=="torch" means raw code
        """
        if model_root_path is None:
            model_root_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../checkpoints/opts"
            )

        if backend in ["torch"]:
            self.set_runtime_backend(backend="torch")
        elif backend in ["onnxruntime", "ort", "onnxrt"]:
            self.set_runtime_backend(
                backend="onnxruntime",
                args={
                    "model_paths": [
                        os.path.join(model_root_path, "set_image_e2e_opt.onnx")
                    ],
                    "providers": [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                },
            )
        elif backend in ["tensorrt", "trt"]:
            self.set_runtime_backend(
                backend="tensorrt",
                args={
                    "model_paths": [
                        os.path.join(model_root_path, "set_image_e2e_opt.onnx")
                    ],
                    "build_args": {
                        "dynamic_axes": {
                            "image": {"min": {0: 1}, "opt": {0: 1}, "max": {0: 1}}
                        },
                        "use_cache": use_cache,
                    },
                },
            )
        else:
            raise f"Unknown backend={backend}"

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. The behavior of this
        method is unchanged for the user.
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
            image = np.array(image)
        else:
            raise NotImplementedError("Image format not supported")

        # Wrap the single image in a list to use the generalized _set_image_ method
        self._set_image_([image]) 

    @torch.no_grad()
    def _set_image_(self, images: List[np.ndarray]):
        """
        Computes image embeddings for a list of images by processing them sequentially
        in a loop to manage memory usage.
        """
        num_images = len(images)
        if num_images == 0:
            self.reset_predictor()
            logging.warning("Input image list is empty. Predictor has been reset.")
            return

        logging.info(f"Processing {num_images} image(s) sequentially...")

        all_feats = []
        img_tensor = []
        for i in range(num_images):
            img_tensor.append(self._transforms.resize(torch.from_numpy(images[i]).to(self.device).movedim(-1, 0)))
   
        img_tensor = torch.stack(img_tensor,dim=0) / 255.0

        for i in range(img_tensor.size(0)):
            current_feats = self.set_image_e2e(img_tensor[i:i+1])  
            for j, feat in enumerate(current_feats):
                while j >= len(all_feats):
                    all_feats.append([])
                all_feats[j].append(feat.squeeze(0))   

        all_feats = [torch.stack(feat_list, dim=0) for feat_list in all_feats]
        
        self._features = {"image_embed": all_feats[-1],            
                          "high_res_feats": all_feats[:-1]}     

        self._is_image_set = True
        logging.info("Image embeddings computed for all images.")

        
    def set_runtime_backend(self, backend="torch", args: dict = None):
        if self.backend_contexts is not None:
            for context in self.backend_contexts:
                context.Release()

        self.backend_contexts = []
        if backend.lower() == "torch":
            self.set_image_e2e = self.set_image_e2e_torch
        elif backend.lower() in ["onnxruntime", "ort", "onnxrt"]:
            assert "model_paths" in args, 'need args["model_paths"] to set *.onnx path'

            from ytools.onnxruntime import OnnxRuntimeExecutor

            model_paths = args["model_paths"]
            if isinstance(model_paths, str):
                model_paths = [model_paths]

            if model_paths[0] is None:
                self.set_image_e2e = self.set_image_e2e_torch
            else:
                self.set_image_e2e = self.set_image_e2e_onnxruntime
                forward_image_executor = OnnxRuntimeExecutor(
                    model_paths[0], providers=args.get("providers", None)
                )
                forward_image_executor.warmup([torch.randn(1, 3, 1024, 1024)])
                self.backend_contexts.append(forward_image_executor)
        elif backend.lower() in ["tensorrt", "trt"]:
            assert (
                "model_paths" in args
            ), 'need args["model_paths"] to set *.engine path'
            from ytools.tensorrt import TensorRTExecutor

            model_paths = args["model_paths"]
            if isinstance(model_paths, str):
                model_paths = [model_paths]

            if model_paths[0] is None:
                self.set_image_e2e = self.set_image_e2e_torch
            else:
                self.set_image_e2e = self.set_image_e2e_tensorrt
                forward_image_executor = TensorRTExecutor(
                    model_paths[0], build_args=args.get("build_args", {})
                )
                forward_image_executor.warmup([torch.randn(1, 3, 1024, 1024)])
                self.backend_contexts.append(forward_image_executor)
        else:
            raise Exception(f"unsupported runtime backend={backend}")

    def set_image_e2e_torch(self, img_batch: torch.Tensor):
        img_batch = self._transforms.norm(img_batch)
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(
            backbone_out
        )
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).unflatten(2, feat_size)
            for feat, feat_size in zip(vision_feats, feat_sizes)
        ]
        return feats[0], feats[1], feats[2]

    def set_image_e2e_onnxruntime(self, img_batch: torch.Tensor):
        outs = self.backend_contexts[0].Inference([img_batch], output_type="torch")
        outputs = [o.to(img_batch.device) for o in outs]
        return tuple(outputs)

    def set_image_e2e_tensorrt(self, img_batch: torch.Tensor):
        outs = self.backend_contexts[0].Inference([img_batch], output_type="torch")
        outputs = [o.to(img_batch.device) for o in outs]
        return tuple(outputs)

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[np.ndarray],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.
        """
        self.reset_predictor()
        if not isinstance(image_list, list) or not image_list:
             raise ValueError("image_list must be a non-empty list of numpy arrays.")
             
        self._orig_hw = []
        for image in image_list:
            assert isinstance(image, np.ndarray), "Images must be numpy arrays in HWC format."
            self._orig_hw.append(image.shape[:2])
            
        # Call the generalized _set_image_ method
        self._set_image_(image_list)

        # Set the batch flag to true after setting the image features
        if self._is_image_set:
            self._is_batch = True

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        if not self._is_batch:
            if self._is_image_set:
                raise RuntimeError("Predictor was set with a single image via `set_image`. Use `predict` instead of `predict_batch`.")
            raise RuntimeError("This function should only be used when in batched mode. Call `set_image_batch` first.")

        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
            
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks

    @torch.no_grad()
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )
        if self._is_batch:
            raise RuntimeError(
                "Predictor was set with a batch of images via `set_image_batch`. Use `predict_batch` instead of `predict`."
            )

        # Transform input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx: int = -1
    ):
        # In batch mode, img_idx must be valid.
        # In single image mode, self._orig_hw has one element, so img_idx=0 is correct.
        current_img_idx = img_idx if self._is_batch else 0
        if self._is_batch and img_idx == -1:
            raise ValueError("In batch mode, a valid `img_idx` must be provided.")

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[current_img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[current_img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # In batch mode, img_idx must be valid.
        # In single image mode, feature tensors have a batch dim of 1, so img_idx=0 is correct.
        current_img_idx = img_idx if self._is_batch else 0
        if self._is_batch and img_idx == -1:
            raise ValueError("In batch mode, a valid `img_idx` must be provided.")
            
        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        
        # Select the features for the correct image index
        high_res_features = [
            feat_level[current_img_idx].unsqueeze(0)            # (1, C, H, W)
            for feat_level in self._features["high_res_feats"]      
        ]
        image_embed = self._features["image_embed"][current_img_idx].unsqueeze(0)  

        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[current_img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False