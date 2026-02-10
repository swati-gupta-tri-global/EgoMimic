#!/usr/bin/env python3
"""
Validation script to generate videos with ground truth trajectory dots overlaid
on the processed LBM dataset to verify the dataset generation is working correctly.

docker exec -it swati-egomimic python /workspace/externals/EgoMimic/validate_lbm_dataset.py --hdf5_file /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/converted/BimanualHangMugsOnMugHolderFromDryingRack.hdf5 --output_dir /workspace/externals/EgoMimic/validation_videos --num_demos 1 --max_frames 200
"""

import argparse
import h5py
import numpy as np
import torch
import os
import cv2
from egomimic.utils.val_utils import draw_both_actions_on_frame, write_video_safe

def validate_lbm_dataset(hdf5_file, output_dir, num_demos=3, max_frames_per_demo=200):
    """
    Generate validation videos from processed LBM dataset with ground truth dots.
    
    Args:
        hdf5_file: Path to processed LBM HDF5 file
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
            
            # Load data
            actions_joints = demo['actions_joints_act'][:]  # (N, 100, 14)
            actions_xyz = demo['actions_xyz_act'][:]       # (N, 100, 6)
            front_img = demo['obs/front_img_1'][:]         # (N, H, W, 3)
            intrinsics = demo['obs/intrinsics'][:]         # (3, 4)
            extrinsics = demo['obs/extrinsics'][:]         # (4, 4) 
            ee_pose = demo['obs/ee_pose'][:]                 # (N, 2, 3)
            joint_positions = demo['obs/joint_positions'][:]  # (N, 2, 7)
            
            print(f"Actions joints shape: {actions_joints.shape}")
            print(f"Actions XYZ shape: {actions_xyz.shape}")
            print(f"Front images shape: {front_img.shape}")
            print(f"Intrinsics shape: {intrinsics.shape}")
            print(f"Extrinsics shape: {extrinsics.shape}")
            print(f"End-effector pose shape: {ee_pose.shape}")
            print(f"Joint positions shape: {joint_positions.shape}")
            import ipdb; ipdb.set_trace()
            
            # Limit frames for speed
            N = min(actions_joints.shape[0], max_frames_per_demo)
            
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
                joints_t = actions_joints[t]  # (100, 14) - trajectory for this timestep
                xyz_t = actions_xyz[t]        # (100, 6) - trajectory for this timestep
                
                # Get intrinsics and extrinsics for this timestep
                if len(intrinsics.shape) == 3:
                    intrinsics_t = intrinsics[t]  # Get intrinsics for this timestep: (3, 3)
                else:
                    intrinsics_t = intrinsics  # Use single intrinsics matrix
                    
                if len(extrinsics.shape) == 3:
                    extrinsics_t = extrinsics[t]  # Get extrinsics for this timestep: (4, 4)
                else:
                    extrinsics_t = extrinsics  # Use single extrinsics matrix
                
                # Convert 3x3 intrinsics to 3x4 projection matrix if needed
                if intrinsics_t.shape == (3, 3):
                    # Add zero column to make it (3, 4) for projection
                    intrinsics_t = np.hstack([intrinsics_t, np.zeros((3, 1))])
                
                # Add safety check for intrinsics shape
                if intrinsics_t.ndim == 3 and intrinsics_t.shape[0] == 1:
                    intrinsics_t = intrinsics_t.squeeze(0)
                
                # import ipdb; ipdb.set_trace()
                # Draw joint trajectories (converted to XYZ via FK) as red dots
                print (extrinsics_t)
                print (intrinsics_t)
                img = draw_both_actions_on_frame(
                    img, 
                    type="xyz", 
                    color="Reds", 
                    actions=ee_pose[t, None, :],  # (N, 6) -> (N, 1, )
                    arm="both", 
                    intrinsics=intrinsics_t,
                    extrinsics=extrinsics_t
                )
                # img = draw_both_actions_on_frame(
                #     img, 
                #     type="joints", 
                #     color="Reds", 
                #     actions=joint_positions[t, None, :],
                #     arm="both", 
                #     intrinsics=intrinsics_t,
                #     extrinsics=extrinsics_t
                # )

                # Draw actions
                # print (joints_t.shape, joint_positions[t, None, :].shape)
                # img = draw_both_actions_on_frame(
                #     img, 
                #     type="joints", 
                #     color="Greens", 
                #     actions=joints_t,
                #     arm="both", 
                #     intrinsics=intrinsics_t,
                #     extrinsics=extrinsics_t
                # )
                
                
                # # Draw ground truth XYZ trajectories as red dots for comparison
                # img = draw_both_actions_on_frame(
                #     img,
                #     type="xyz",
                #     color="Reds", 
                #     actions=xyz_t,
                #     arm="both",
                #     intrinsics=intrinsics_t,
                #     extrinsics=extrinsics_t
                # )
                    
                # Add frame info text
                cv2.putText(img, f'Frame {t}/{N}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # cv2.putText(img, f'Green: FK from joints', (10, 60), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f'Red: Ground truth XYZ', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                video_frames.append(img)
            
            # Save video
            if video_frames:
                video_tensor = torch.stack([torch.from_numpy(frame) for frame in video_frames])
                video_path = os.path.join(output_dir, f'{demo_name}_validation.mp4')
                print(f"Saving video: {video_path}")
                write_video_safe(video_path, video_tensor, fps=10)
            
            print(f"Completed {demo_name}")


def main():
    parser = argparse.ArgumentParser(description='Validate LBM dataset by generating videos with trajectory dots')
    parser.add_argument('--hdf5_file', type=str, required=True,
                        help='Path to processed LBM HDF5 file')
    parser.add_argument('--output_dir', type=str, default='./validation_videos',
                        help='Output directory for validation videos')
    parser.add_argument('--num_demos', type=int, default=3,
                        help='Number of demos to validate')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Maximum frames per demo to process')
    
    args = parser.parse_args()
    
    print(f"Validating dataset: {args.hdf5_file}")
    print(f"Output directory: {args.output_dir}")
    
    validate_lbm_dataset(
        args.hdf5_file, 
        args.output_dir, 
        args.num_demos, 
        args.max_frames
    )
    
    print("Validation complete!")


if __name__ == "__main__":
    main()