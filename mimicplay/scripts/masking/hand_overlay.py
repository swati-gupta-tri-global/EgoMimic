from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from mimicplay.scripts.aloha_process.simarUtils import nds, cam_frame_to_cam_pixels, ARIA_INTRINSICS
import argparse
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"mask_{i}.png")
        plt.close()


def get_bounds(binary_image):
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to create a binary image
    # _,binary_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store max and min x and y values
    max_x = max_y = 0
    min_x = min_y = float('inf')

    # Loop through all contours to find max and min x and y values
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    return min_x, max_x, min_y, max_y

def main(args):
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    sam2_checkpoint = "/coc/flash9/skareer6/Projects/EgoPlay/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    predictor = SAM2ImagePredictor(sam2_model)

    with h5py.File(args.dataset, "r+") as data:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in tqdm(range(len(data["data"].keys()))):
                demo = data[f"data/demo_{i}"]
                imgs = demo["obs/front_img_1"]
                ee_poses = demo["obs/ee_pose"]
                prompts = cam_frame_to_cam_pixels(ee_poses[:], ARIA_INTRINSICS)[:, :2]
                masked_imgs = np.zeros_like(imgs)
                overlayed_imgs = np.zeros_like(imgs)


                # TODO: this can be batched
                for k in range(imgs.shape[0]):
                    img = imgs[k]
                    input_point = prompts[[k]]
                    input_label = np.array([1])

                    predictor.set_image(img)

                    masks, scores, logits = predictor.predict(
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
                    masked_imgs[k] = masked_img

                    if args.debug:
                        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(f"./overlays/demo_{i}_masked_{k}.png", masked_img)

                    min_x, max_x, min_y, max_y = get_bounds(masks[0].astype(np.uint8))
                    gamma = 0.8
                    alpha = 0.2
                    scale = max_y - min_y
                    min_x = int(max_x + gamma * (min_x - max_x))
                    min_y = int(max_y + gamma * (min_y - max_y))
                    max_x = int(max_x - scale * alpha)

                    line_image = cv2.line(masked_img.copy(), (min_x,min_y),(max_x,max_y),color=(255,0,0), thickness=25)
                    overlayed_imgs[k] = line_image

                    if args.debug:
                        cv2.imwrite(f"./overlays/demo_{i}_line_{k}.png", line_image)

                    # show_masks(img, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
                    # breakpoint()
                
                if "front_img_1_masked" in demo["obs"]:
                    print("Deleting existing masked images")
                    del demo["obs/front_img_1_masked"]
                if "front_img_1_line" in demo["obs"]:
                    print("Deleting existing line images")
                    del demo["obs/front_img_1_line"]

                demo["obs"].create_dataset("front_img_1_masked", data=masked_imgs, chunks=(1, 480, 640, 3))
                demo["obs"].create_dataset("front_img_1_line", data=overlayed_imgs, chunks=(1, 480, 640, 3))

                # batched
                # imgs_batch = [imgs[i] for i in range(imgs.shape[0])][:50]
                # labels_batch = [np.array([[1]]) for i in range(imgs.shape[0])][:50]
                # prompts_batch = [prompts[[i]][None, :] for i in range(imgs.shape[0])][:50]

                # predictor.set_image_batch(imgs_batch)


                # masks_batch, scores_batch, _ = predictor.predict_batch(prompts_batch, labels_batch, box_batch=None, multimask_output=True)

                # # Select the best single mask per object
                # best_masks = []
                # for masks, scores in zip(masks_batch,scores_batch):
                #     # best_masks.append(masks[range(len(masks)), np.argmax(scores, axis=-1)])
                #     best_masks.append(masks[np.argmax(scores, axis=-1)])
                
                # count = 0
                # # show the first mask and image
                # for image, points, labels, best_mask in zip(imgs_batch, prompts_batch, labels_batch, best_masks):
                #     plt.figure(figsize=(10, 10))
                #     plt.imshow(image)   
                #     # for mask in masks:
                #     show_mask(best_mask, plt.gca(), random_color=True)
                #     show_points(points, labels, plt.gca())
                #     plt.savefig(f"best_mask_{count}.png")
                #     plt.close()
                #     count += 1
                
                # breakpoint()


if __name__ == '__main__':
    '''
    usage:     
    python hand_overlay.py --hdf5_file /coc/flash7/datasets/egoplay/_OBOO_ARIA/oboo_yellow_jun12/converted/oboo_yellow_jun12_ACTGMMCompat_masked.hdf5 --debug
    '''
    parser = argparse.ArgumentParser(description='Process an HDF5 file.')
    parser.add_argument('--dataset', type=str, help='HDF5 file name')
    # parser.add_argument('--change_type', type=int, help='Type of change: 0-masked image 1-line on unmasked image 2-line on masked image')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    main(args)