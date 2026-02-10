#!/usr/bin/env python3
"""
Validation script to generate videos with ground truth trajectory dots overlaid
on the processed EgoDx dataset to verify the dataset generation is working correctly.
"""

from builtins import print
import argparse
import h5py
import numpy as np
import torch
import os
import cv2
from egomimic.utils.val_utils import draw_both_actions_on_frame, write_video_safe
from egomimic.utils.egomimicUtils import nds


def validate_egodx_dataset(hdf5_file, output_dir, num_demos=3, max_frames_per_demo=200):
    """
    Generate validation videos from processed EgoDx dataset with ground truth dots.
    
    Args:
        hdf5_file: Path to processed EgoDx HDF5 file
        output_dir: Directory to save validation videos
        num_demos: Number of demonstrations to validate
        max_frames_per_demo: Maximum frames per demo to process (for speed)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf5_file, 'r') as f:
        print(f"Dataset keys: {list(f.keys())}")
        
        # Get demo names from the 'data' group
        data_group = f['data']
        demo_names = [key for key in data_group.keys() if key.startswith('demo_')]
        demo_names = sorted(demo_names)[:num_demos]
        
        print(f"Processing demos: {demo_names}")
        
        for demo_name in demo_names:
            demo = data_group[demo_name]
            print(f"\nProcessing {demo_name}")
            print(f"Demo keys: {list(demo.keys())}")
            
            # Load data - EgoDx only has XYZ actions (hand trajectories)
            actions_xyz = demo['actions_xyz_act'][:]       # (N, 100, 6)
            front_img = demo['obs/front_img_1'][:]         # (N, H, W, 3)
            ee_pose = demo['obs/ee_pose'][:]                 # (N, 2, 3)
            print(f"Actions XYZ shape: {actions_xyz.shape}")
            print(f"Front images shape: {front_img.shape}")
            
            try:
                intrinsics_t = demo['obs/intrinsics'][:]         # (N, 3, 4)
            except:
                # EgoDex hardcoded intrinsics
                # Create basic 3x4 intrinsics matrix (you may need to adjust these values)
                intrinsics_t = np.array([
                    [[736.6339,   0.,     960.,     0.    ],
                    [  0.,     736.6339, 540.,     0.    ],
                    [  0.,       0.,       1.,     0.    ]]
                ])
                 
                print(f"Using hardcoded intrinsics: {intrinsics_t.shape}")
    
            
            # Limit frames for speed
            N = min(actions_xyz.shape[0], max_frames_per_demo)
            
            # Create video with trajectory dots
            video_frames = []
            
            for t in range(N):
                # Get frame
                img = front_img[t]  # (H, W, 3)
                if img.max() <= 1.0:  # Normalize if needed
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                
                # Get actions for this timestep
                xyz_t = actions_xyz[t]        # (100, 6) - hand trajectory for this timestep
                
                # Draw hand XYZ trajectories as green dots
                # img = draw_both_actions_on_frame(
                #     img,
                #     type="xyz",
                #     color="Reds", 
                #     actions=xyz_t,
                #     arm="both",  # EgoDx has bimanual hand data (6D = 3D left + 3D right)
                #     intrinsics=intrinsics_t,
                #     extrinsics=None
                # )

                img = draw_both_actions_on_frame(
                    img,
                    type="xyz",
                    color="Reds", 
                    actions=ee_pose,
                    arm="both",  # EgoDx has bimanual hand data (6D = 3D left + 3D right)
                    intrinsics=intrinsics_t,
                    extrinsics=None
                )
                
                # Add frame info text
                cv2.putText(img, f'Frame {t}/{N}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f'Red: Hand XYZ trajectory', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  #RGB
                cv2.putText(img, f'EgoDx Dataset', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                video_frames.append(img)
            
            # Save video
            if video_frames:
                video_tensor = torch.stack([torch.from_numpy(frame) for frame in video_frames])
                video_path = os.path.join(output_dir, f'{demo_name}_egodx_validation.mp4')
                print(f"Saving video: {video_path}")
                write_video_safe(video_path, video_tensor, fps=10)
            
            print(f"Completed {demo_name}")


def main():
    parser = argparse.ArgumentParser(description='Validate EgoDx dataset by generating videos with trajectory dots')
    parser.add_argument('--hdf5_file', type=str, required=True,
                        help='Path to processed EgoDx HDF5 file')
    parser.add_argument('--output_dir', type=str, default='./validation_videos_egodx',
                        help='Output directory for validation videos')
    parser.add_argument('--num_demos', type=int, default=3,
                        help='Number of demos to validate')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Maximum frames per demo to process')
    
    args = parser.parse_args()
    
    print(f"Validating EgoDx dataset: {args.hdf5_file}")
    print(f"Output directory: {args.output_dir}")
    
    validate_egodx_dataset(
        args.hdf5_file, 
        args.output_dir, 
        args.num_demos, 
        args.max_frames
    )
    
    print("EgoDx validation complete!")

if __name__ == "__main__":
    main()