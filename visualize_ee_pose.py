#!/usr/bin/env python3
"""
Script to visualize end-effector poses projected onto images from EgoDex/LBM HDF5 dataset.
Visualizes both left and right hand EE poses as colored circles on the front camera image.
You can also visualize the action_xyz as dots on the image.

Usage:
python3 visualize_ee_pose.py --hdf5 /workspace/externals/EgoMimic/datasets/LBM_sim_egocentric/processed/BimanualHangMugsOnMugHolderFromTable.hdf5 --demo 0 --visualize-ee
python3 visualize_ee_pose.py --hdf5 /workspace/externals/EgoMimic/datasets/egodex/processed/part1/add_remove_lid.hdf5 --demo 0 --visualize-ee
python3 visualize_ee_pose.py --hdf5 /workspace/externals/EgoMimic/datasets/AVP/processed/egoPutKiwiInCenterOfTable.hdf5 --demo 0 --visualize-actions
"""

import h5py
import numpy as np
import cv2
import argparse
import os


def project_3d_to_2d(point_3d, intrinsics):
    """
    Project a 3D point in camera frame to 2D image coordinates.
    
    Args:
        point_3d: (3,) array [x, y, z] in camera frame (meters)
        intrinsics: (3, 3) camera intrinsic matrix
    
    Returns:
        (u, v): 2D pixel coordinates
    """
    # point_3d is already in camera frame, so just apply projection
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x, y, z = point_3d
    
    if z <= 0:
        return None, None  # Point is behind camera
    
    u = (fx * x / z) + cx
    v = (fy * y / z) + cy
    
    return int(u), int(v)


def visualize_ee_poses(hdf5_path, demo_idx=0, output_dir="./ee_pose_visualization", skip_frames=5, create_video=True, visualize_ee=False, visualize_actions=True):
    """
    Visualize end-effector poses and/or actions projected onto front camera images.
    
    Args:
        hdf5_path: Path to HDF5 file
        demo_idx: Index of demonstration to visualize
        output_dir: Directory to save visualization images
        skip_frames: Skip every N frames for faster visualization
        create_video: Whether to create video from frames
        visualize_ee: Whether to visualize EE pose observations (green/blue circles)
        visualize_actions: Whether to visualize action trajectories (cyan/magenta circles)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract task name from HDF5 file path
    task_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    
    with h5py.File(hdf5_path, 'r') as f:
        # List available demos
        demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
        print(f"Available demos: {len(demo_keys)}")
        
        if demo_idx >= len(demo_keys):
            print(f"Error: demo_idx {demo_idx} out of range. Max: {len(demo_keys)-1}")
            return
        
        demo_key = f"demo_{demo_idx}"
        demo = f['data'][demo_key]
        
        # Load data
        images = demo['obs/front_img_1'][:]  # (N, H, W, 3)
        ee_poses = demo['obs/ee_pose'][:]    # (N, 6) - [left_x, left_y, left_z, right_x, right_y, right_z]
        intrinsics = demo['obs/intrinsics'][:]  # (N, 3, 3)
        actions_xyz = demo['actions_xyz_act'][:]  # (N, 100, 6)
        
        num_samples = demo.attrs['num_samples']
        
        print(f"\nDemo {demo_idx} info:")
        print(f"  Number of samples: {num_samples}")
        print(f"  Images shape: {images.shape}")
        print(f"  EE poses shape: {ee_poses.shape}")
        print(f"  Intrinsics shape: {intrinsics.shape}")
        print(f"  Image dtype: {images.dtype}, range: [{images.min()}, {images.max()}]")
        
        # Visualize frames
        print(f"\nVisualizing every {skip_frames} frames...")
        for i in range(0, len(images), skip_frames):
            img = images[i].copy()
            
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Ensure RGB format
            if img.shape[-1] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
            
            ee_pose = ee_poses[i]
            
            # Left hand (first 3 values) - Green
            left_ee = ee_pose[:3]
            u_left, v_left = project_3d_to_2d(left_ee, intrinsics[0])
            
            # Right hand (last 3 values) - Blue
            right_ee = ee_pose[3:]
            u_right, v_right = project_3d_to_2d(right_ee, intrinsics[0])
            
            # Draw EE pose projected points (current observation)
            if visualize_ee:
                if u_left is not None and v_left is not None:
                    if 0 <= u_left < img_bgr.shape[1] and 0 <= v_left < img_bgr.shape[0]:
                        cv2.circle(img_bgr, (u_left, v_left), 10, (0, 255, 0), -1)  # Green for left
                        cv2.putText(img_bgr, "L_EE", (u_left + 15, v_left), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if u_right is not None and v_right is not None:
                    if 0 <= u_right < img_bgr.shape[1] and 0 <= v_right < img_bgr.shape[0]:
                        cv2.circle(img_bgr, (u_right, v_right), 10, (255, 0, 0), -1)  # Blue for right
                        cv2.putText(img_bgr, "R_EE", (u_right + 15, v_right), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Visualize action_xyz trajectory (subsample for clarity)
            out_of_bounds_count = 0
            valid_action_count = 0
            
            if visualize_actions:
                # if i+1 >= len(actions_xyz):
                #     continue  # No future actions available
                action_traj = actions_xyz[i]  # (100, 6) - [left_x, left_y, left_z, right_x, right_y, right_z]
                subsample_step = 1  # Show every 10th action point
                
                for j in range(0, len(action_traj), subsample_step):
                    action = action_traj[j]
                    
                    # Left action (first 3 values) - Cyan
                    left_action = action[:3]
                    # Check if action has valid depth
                    if left_action[2] > 0.01:  # Only project if z > 0.01m
                        u_left_act, v_left_act = project_3d_to_2d(left_action, intrinsics[0])
                        
                        if u_left_act is not None and v_left_act is not None:
                            if 0 <= u_left_act < img_bgr.shape[1] and 0 <= v_left_act < img_bgr.shape[0]:
                                cv2.circle(img_bgr, (u_left_act, v_left_act), 3, (255, 255, 0), -1)  # Cyan for left actions
                                valid_action_count += 1
                            else:
                                out_of_bounds_count += 1
                    
                    # Right action (last 3 values) - Magenta
                    right_action = action[3:]
                    # Check if action has valid depth
                    if right_action[2] > 0.01:  # Only project if z > 0.01m
                        u_right_act, v_right_act = project_3d_to_2d(right_action, intrinsics[0])
                        
                        if u_right_act is not None and v_right_act is not None:
                            if 0 <= u_right_act < img_bgr.shape[1] and 0 <= v_right_act < img_bgr.shape[0]:
                                cv2.circle(img_bgr, (u_right_act, v_right_act), 3, (255, 0, 255), -1)  # Magenta for right actions
                                valid_action_count += 1
                            else:
                                out_of_bounds_count += 1
            
            # Print stats occasionally
            if visualize_actions and i % (skip_frames * 10) == 0 and out_of_bounds_count > 0:
                print(f"  Frame {i}: {valid_action_count} valid actions, {out_of_bounds_count} out of bounds")
            
            # Add frame info
            cv2.putText(img_bgr, f"Frame {i}/{len(images)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add legend
            legend_y = 60
            legend_offset = 25
            cv2.putText(img_bgr, "Legend:", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if visualize_ee:
                cv2.circle(img_bgr, (20, legend_y + legend_offset), 5, (0, 255, 0), -1)
                cv2.putText(img_bgr, "Left EE (obs)", (35, legend_y + legend_offset + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                legend_offset += 25
                cv2.circle(img_bgr, (20, legend_y + legend_offset), 5, (255, 0, 0), -1)
                cv2.putText(img_bgr, "Right EE (obs)", (35, legend_y + legend_offset + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                legend_offset += 25
                # Add 3D coordinates
                text_left = f"L: [{left_ee[0]:.3f}, {left_ee[1]:.3f}, {left_ee[2]:.3f}]"
                text_right = f"R: [{right_ee[0]:.3f}, {right_ee[1]:.3f}, {right_ee[2]:.3f}]"
                cv2.putText(img_bgr, text_left, (10, img_bgr.shape[0] - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img_bgr, text_right, (10, img_bgr.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            if visualize_actions:
                cv2.circle(img_bgr, (20, legend_y + legend_offset), 3, (255, 255, 0), -1)
                cv2.putText(img_bgr, "Left Actions", (35, legend_y + legend_offset + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                legend_offset += 25
                cv2.circle(img_bgr, (20, legend_y + legend_offset), 3, (255, 0, 255), -1)
                cv2.putText(img_bgr, "Right Actions", (35, legend_y + legend_offset + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            

            
            # Save individual frames
            output_path = os.path.join(output_dir, f"{task_name}_demo_{demo_idx}_frame_{i:04d}.png")
            cv2.imwrite(output_path, img_bgr)
            
            if i % (skip_frames * 10) == 0:
                print(f"  Processed frame {i}/{len(images)}")
        
        print(f"\nVisualization complete! Saved to {output_dir}")
        print(f"Total frames saved: {len(range(0, len(images), skip_frames))}")
        
        # Create a video from frames (optional)
        if create_video:
            print("\nCreating video from frames...")
            ret = create_video_from_frames(output_dir, task_name, demo_idx)
            if ret != 0:
                print("Video creation failed.")
            else:
                # delete individual frames if video creation succeeded
                rm_cmd = f"rm {os.path.join(output_dir, f'{task_name}_demo_{demo_idx}_frame_*.png')}"
                os.system(rm_cmd)
        else:
            print("\nSkipping video creation (use without --no-video to create video)")


def create_video_from_frames(frame_dir, task_name, demo_idx, fps=30):
    """Create MP4 video from saved frames."""
    import glob
    import subprocess
    
    frame_files = sorted(glob.glob(os.path.join(frame_dir, f"{task_name}_demo_{demo_idx}_frame_*.png")))
    
    if not frame_files:
        print("No frames found to create video")
        return
    
    print(f"Found {len(frame_files)} frames to process")
    
    output_video = os.path.join(frame_dir, f"{task_name}_demo_{demo_idx}_visualization.mp4")
    
    # Try using ffmpeg directly (most reliable)
    print("Attempting to create video using ffmpeg...")
    
    # Create a temporary file list for ffmpeg
    filelist_path = os.path.join(frame_dir, f"{task_name}_demo_{demo_idx}_filelist.txt")
    with open(filelist_path, 'w') as f:
        for frame_file in frame_files:
            # ffmpeg expects duration per frame
            f.write(f"file '{os.path.basename(frame_file)}'\n")
            f.write(f"duration {1.0/fps}\n")
        # Add last frame again for proper duration
        f.write(f"file '{os.path.basename(frame_files[-1])}'\n")
    
    # Run ffmpeg command
    # Use basename for paths when using cwd
    filelist_basename = os.path.basename(filelist_path)
    output_basename = os.path.basename(output_video)
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # overwrite output
        '-f', 'concat',
        '-safe', '0',
        '-i', filelist_basename,
        '-vf', f'fps={fps}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_basename
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, cwd=frame_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Clean up temp file
            os.remove(filelist_path)
            
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                print(f"✓ Video saved successfully using ffmpeg: {output_video}")
                print(f"  File size: {os.path.getsize(output_video) / (1024*1024):.2f} MB")
                return 0
            else:
                print("ffmpeg completed but no valid video file created")
                return 1
        else:
            print(f"ffmpeg failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("ffmpeg not found, trying OpenCV fallback...")
    except Exception as e:
        print(f"ffmpeg error: {e}")
    
    # Fallback to OpenCV
    print("Attempting to create video using OpenCV...")
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Error: Could not read first frame: {frame_files[0]}")
        return 1
    
    height, width = first_frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")
    
    # Try different codecs
    codecs_to_try = [
        ('avc1', '.mp4', 'H264'),
        ('mp4v', '.mp4', 'mp4v')
    ]
    
    for codec, ext, name in codecs_to_try:
        output_path = output_video if ext == '.mp4' else output_video.replace('.mp4', ext)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if out.isOpened():
            print(f"Using {name} codec...")
            for i, frame_file in enumerate(frame_files):
                frame = cv2.imread(frame_file)
                if frame is not None:
                    out.write(frame)
                if (i + 1) % 10 == 0:
                    print(f"  Wrote {i + 1}/{len(frame_files)} frames")
            
            out.release()
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✓ Video saved successfully: {output_path}")
                print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
                return 0
    
    print("✗ ERROR: Could not create video with any method!")
    print("Frames are saved as individual PNG files in:", frame_dir)
    return 1


def main():
    parser = argparse.ArgumentParser(description="Visualize EE poses and/or actions projected onto images")
    parser.add_argument("--hdf5", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--demo", type=int, nargs='+', default=[0], 
                       help="Demo index/indices to visualize (can specify multiple: --demo 0 1 2)")
    parser.add_argument("--output-dir", type=str, default="./ee_pose_visualization", 
                       help="Output directory for visualizations")
    parser.add_argument("--skip-frames", type=int, default=5, 
                       help="Skip every N frames for faster visualization")
    parser.add_argument("--no-video", action="store_true", 
                       help="Don't create video output")
    parser.add_argument("--visualize-ee", action="store_true", 
                       help="Visualize end-effector pose observations (green/blue circles)")
    parser.add_argument("--visualize-actions", action="store_true", default=True,
                       help="Visualize action trajectories (cyan/magenta circles, enabled by default)")
    parser.add_argument("--no-actions", action="store_true",
                       help="Don't visualize action trajectories")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5):
        print(f"Error: HDF5 file not found: {args.hdf5}")
        return
    
    # Handle action visualization flag (default is True unless --no-actions is set)
    visualize_actions = not args.no_actions
    
    # Process each demo index
    demo_indices = args.demo if isinstance(args.demo, list) else [args.demo]
    print(f"\nProcessing {len(demo_indices)} demo(s): {demo_indices}")
    
    for demo_idx in demo_indices:
        print(f"\n{'='*60}")
        print(f"Processing demo {demo_idx}")
        print(f"{'='*60}")
        
        # Pass create_video flag (opposite of no_video)
        visualize_ee_poses(args.hdf5, demo_idx, args.output_dir, args.skip_frames, 
                          create_video=not args.no_video,
                          visualize_ee=args.visualize_ee,
                          visualize_actions=visualize_actions)
    
    print(f"\n{'='*60}")
    print(f"All {len(demo_indices)} demo(s) processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
