import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import h5py
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TVF
import matplotlib.pyplot as plt

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