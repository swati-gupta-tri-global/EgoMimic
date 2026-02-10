"""
Test script to verify hand tracking extraction from AVP dataset.
This loads a single episode and tests the parse_to_se3 and extract_hand_poses_from_response functions.
"""

import os
import sys

# Add AVP python_client to path for tracker_pb2 import
avp_client_path = "/workspace/externals/EgoMimic/external/avp_teleop/python_client"
if avp_client_path not in sys.path:
    sys.path.insert(0, avp_client_path)

import numpy as np
import gzip
import pickle
from scipy.spatial.transform import Rotation

def parse_to_se3(pose):
    """
    Parse translation and rotation data into SE(3) format (4x4 homogeneous matrix).
    Based on client_utils.py from AVP teleop code.
    """
    # Extract translation values
    t = np.array([pose.translation.x, pose.translation.y, pose.translation.z])
    
    # Extract rotation values (quaternion as x, y, z, w)
    q = np.array([pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w])
    
    # Convert quaternion (x, y, z, w) to rotation matrix
    rot = Rotation.from_quat(q)
    R = rot.as_matrix()
    
    # Create SE(3) matrix (4x4)
    se3 = np.eye(4)
    se3[:3, :3] = R
    se3[:3, 3] = t
    
    return se3

def extract_hand_poses_from_response(response):
    """
    Extract hand pose from tracker response object.
    """
    try:
        # Extract the anchor transform (base of the hand)
        if not hasattr(response, 'hand') or not hasattr(response.hand, 'anchor_transform'):
            print(f"[WARNING] Response missing hand.anchor_transform")
            return None
        
        anchor_transform = response.hand.anchor_transform
        anchor_transform_se3 = parse_to_se3(anchor_transform)
        
        return anchor_transform_se3
        
    except Exception as e:
        print(f"[ERROR] Failed to extract hand pose: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Test with first episode from egoPutKiwiInCenterOfTable
    dataset_dir = "/workspace/externals/EgoMimic/datasets/AVP/egoPutKiwiInCenterOfTable"
    
    # Find timestamp directory
    timestamp_dirs = [d for d in os.listdir(dataset_dir) 
                     if os.path.isdir(os.path.join(dataset_dir, d))]
    
    if not timestamp_dirs:
        print(f"No timestamp directories found in {dataset_dir}")
        return
    
    # Use first timestamp directory
    timestamp_dir = os.path.join(dataset_dir, timestamp_dirs[0])
    print(f"Using timestamp directory: {timestamp_dir}")
    
    # Find episode directories
    episode_dirs = sorted([d for d in os.listdir(timestamp_dir) 
                          if os.path.isdir(os.path.join(timestamp_dir, d)) and d.startswith("episode_")])
    
    if not episode_dirs:
        print(f"No episodes found in {timestamp_dir}")
        return
    
    first_episode = os.path.join(timestamp_dir, episode_dirs[0])
    episode_pkl = os.path.join(first_episode, "episode.pkl")
    
    print(f"Testing with: {first_episode}")
    print(f"Loading: {episode_pkl}")
    
    # Load episode data
    try:
        with gzip.open(episode_pkl, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded pickle file")
    except Exception as e:
        print(f"✗ Failed to load pickle: {e}")
        return
    
    # Check data structure
    print(f"\nData keys: {data.keys()}")
    print(f"Number of frames: {len(data['frame_timestamps'])}")
    print(f"Success: {data.get('success', 'N/A')}")
    
    # Test hand pose extraction from first frame
    if 'pose_snapshots' in data and len(data['pose_snapshots']) > 0:
        pose_snapshot = data['pose_snapshots'][0]
        print(f"\nPose snapshot keys: {pose_snapshot.keys()}")
        
        # Test left hand
        print("\n--- Testing Left Hand ---")
        if 'left' in pose_snapshot:
            left_data = pose_snapshot['left']
            print(f"Left data keys: {left_data.keys()}")
            
            if 'response' in left_data:
                left_response = left_data['response']
                print(f"Left response type: {type(left_response)}")
                print(f"Left response has 'hand' attr: {hasattr(left_response, 'hand')}")
                
                if hasattr(left_response, 'hand'):
                    print(f"Left hand has 'anchor_transform': {hasattr(left_response.hand, 'anchor_transform')}")
                    
                    left_pose = extract_hand_poses_from_response(left_response)
                    if left_pose is not None:
                        print(f"✓ Successfully extracted left hand pose")
                        print(f"  Shape: {left_pose.shape}")
                        print(f"  Translation: {left_pose[:3, 3]}")
                        print(f"  Rotation matrix:\n{left_pose[:3, :3]}")
                    else:
                        print(f"✗ Failed to extract left hand pose")
        
        # Test right hand
        print("\n--- Testing Right Hand ---")
        if 'right' in pose_snapshot:
            right_data = pose_snapshot['right']
            print(f"Right data keys: {right_data.keys()}")
            
            if 'response' in right_data:
                right_response = right_data['response']
                print(f"Right response type: {type(right_response)}")
                print(f"Right response has 'hand' attr: {hasattr(right_response, 'hand')}")
                
                if hasattr(right_response, 'hand'):
                    print(f"Right hand has 'anchor_transform': {hasattr(right_response.hand, 'anchor_transform')}")
                    
                    right_pose = extract_hand_poses_from_response(right_response)
                    if right_pose is not None:
                        print(f"✓ Successfully extracted right hand pose")
                        print(f"  Shape: {right_pose.shape}")
                        print(f"  Translation: {right_pose[:3, 3]}")
                        print(f"  Rotation matrix:\n{right_pose[:3, :3]}")
                    else:
                        print(f"✗ Failed to extract right hand pose")
        
        # Test multiple frames to ensure consistency
        print("\n--- Testing Multiple Frames ---")
        num_test_frames = min(10, len(data['pose_snapshots']))
        successful_extractions = 0
        
        for i in range(num_test_frames):
            pose_snapshot = data['pose_snapshots'][i]
            left_response = pose_snapshot['left']['response']
            right_response = pose_snapshot['right']['response']
            
            left_pose = extract_hand_poses_from_response(left_response)
            right_pose = extract_hand_poses_from_response(right_response)
            
            if left_pose is not None and right_pose is not None:
                successful_extractions += 1
        
        print(f"Successfully extracted poses from {successful_extractions}/{num_test_frames} frames")
        
        if successful_extractions == num_test_frames:
            print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
            print("The hand tracking extraction is working correctly!")
        else:
            print(f"\n✗✗✗ SOME TESTS FAILED ✗✗✗")
            print(f"Only {successful_extractions}/{num_test_frames} frames had successful extraction")
    
    else:
        print("\n✗ No pose_snapshots found in data")

if __name__ == "__main__":
    main()
