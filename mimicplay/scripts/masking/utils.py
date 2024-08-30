import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import h5py
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TVF
import matplotlib.pyplot as plt
import mimicplay
import os
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from mimicplay.scripts.aloha_process.simarUtils import AlohaFK, ee_pose_to_cam_pixels, ARIA_INTRINSICS, EXTRINSICS
import cv2


class Inpainter:
    def __init__(self):
        pipeline = AutoPipelineForInpainting.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        pipeline.enable_model_cpu_offload()

        self.pipeline = pipeline
    
    def inpaint(self, image, mask):
        """
            image: tensor (B, C, H, W)
            mask: tensor (B, 1, H, W)
        """
        generator = torch.Generator("cuda").manual_seed(92)

        im_out = self.pipeline(prompt="wooden table, highly detailed, 8k", negative_prompt="human arm, poor details, blurry, disfigured", image=image, mask_image=mask, generator=generator)

        return im_out


def inpaint_hdf5(hdf5_file, inpainter):
    with h5py.File(hdf5_file, "r+") as data:
        for i in tqdm(range(2, len(data["data"].keys()))):
            demo = data[f"data/demo_{i}"]
            imgs = demo["obs/front_img_1"]
            masks = demo["obs/front_img_1_mask"]
            inpainted_imgs = np.zeros_like(imgs)

            for k in range(imgs.shape[0]):
                img = imgs[k]
                mask = masks[k]

                # init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
                # mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
                mask = torch.from_numpy(mask).int()[None, :]

                img = TVF.resize(img, (512, 512))
                mask = TVF.resize(mask, (512, 512)).to(torch.uint8)
                # dilate the mask
                mask = torch.nn.functional.max_pool2d(mask.float(), 3, stride=1, padding=1) > 0.5
                mask = mask.to(torch.uint8)


                inpainted_img = np.array(inpainter.inpaint(img, mask).images[0])

                plt.imsave(f"./scripts/masking/overlays/demo_{i}_inpaint_{k}.png", inpainted_img)
                inpainted_img = TVF.resize(torch.from_numpy(inpainted_img).permute(2, 0, 1), (480, 640)).numpy()
                inpainted_imgs[k] = inpainted_img.transpose((1, 2, 0))

                # breakpoint()

class SAM:
    def __init__(self):
        sam2_checkpoint = os.path.join(
            os.path.dirname(mimicplay.__file__), "resources/sam2_hiera_tiny.pt"
        )
        model_cfg = "sam2_hiera_t.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

        predictor = SAM2ImagePredictor(sam2_model)

        self.predictor = predictor
        self.fk = AlohaFK()
    
    def get_robot_mask(self, images, qpos):
        """
            images: tensor (B, H, W, 3)
            qpos: B, 7
        """
        
        joint_pos = self.fk.chain.forward_kinematics(qpos[:, :-1], end_only=False)
        
        gripper_positions = joint_pos['vx300s/ee_gripper_link'].get_matrix()[:, :3, 3]
        elbow_positions = joint_pos['vx300s/upper_forearm_link'].get_matrix()[:, :3, 3]
        lower_forearm_positions = joint_pos['vx300s/lower_forearm_link'].get_matrix()[:, :3, 3]

        px_val_gripper = ee_pose_to_cam_pixels(gripper_positions, EXTRINSICS["ariaJul29R"], ARIA_INTRINSICS)
        px_val_elbow = ee_pose_to_cam_pixels(elbow_positions, EXTRINSICS["ariaJul29R"], ARIA_INTRINSICS)
        px_val_lower_forearm = ee_pose_to_cam_pixels(lower_forearm_positions, EXTRINSICS["ariaJul29R"], ARIA_INTRINSICS)

        line_images = np.zeros_like(images)
        mask_images = np.zeros_like(images)

        for i, image in enumerate(images):
            pt1, pt2 = px_val_lower_forearm[[i]][:, :2], px_val_gripper[[i]][:, :2]
            pt3 = (pt1 + pt2)/2
            input_point = np.concatenate([pt1, pt2, pt3], axis=0)

            input_label = np.array([1, 1, 1])

            self.predictor.set_image(image)

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            
            masked_img = image.copy()
            masked_img[masks[0] == 1] = 0
            mask_images[i] = masked_img

            line_img = cv2.line(masked_img.copy(), (int(px_val_gripper[i,0]),int(px_val_gripper[i,1])),(int(px_val_elbow[i,0]),int(px_val_elbow[i,1])),color=(255,0,0), thickness=25)
            # line_images[i] = image
            # plt.imsave("robo_overlay/masked.png", image)
            # breakpoint()

            line_images[i] = line_img

        return mask_images, line_images

    def get_mask(self, images, pos_prompts, neg_prompts=None):
        """
            images: tensor (B, H, W, 3)
            pos_prompts: tensor (B, 2)

            returns: raw_masks
        """
        masked_imgs = np.zeros_like(images)
        raw_masks = np.zeros((images.shape[0], 480, 640)).astype(bool)

        for k in range(images.shape[0]):
            img = images[k]
            input_point = pos_prompts[[k]]
            if input_point[0, 0] > 640 or input_point[0, 1] > 480 or input_point[0, 0] < 0 or input_point[0, 1] < 0:
                print("skipping image", k)
                masked_imgs[k] = img
                continue
            input_label = np.array([1])
            if neg_prompts is not None:
                input_point = np.concatenate([input_point, neg_prompts[[k]]], axis=0)
                input_label = np.array([1, 0])

            self.predictor.set_image(img)

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            
            masked_img = img.copy()
            masked_img[masks[0] == 1] = 0
            raw_masks[k] = masks[0].astype(bool)
            masked_imgs[k] = masked_img

        return masked_imgs, raw_masks