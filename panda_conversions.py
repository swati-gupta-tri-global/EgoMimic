import numpy as np
import torch
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R
from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    ee_pose_to_cam_frame,
)

def panda_fk_corrected(joint_angles):
    """Corrected FK using proper Panda DH parameters."""
    # Correct Panda DH parameters [a, alpha, d, theta_offset]
    dh = np.array([
        [0,      0,       0.333, 0],        # Joint 1
        [0,      -np.pi/2, 0,    0],        # Joint 2  
        [0,      np.pi/2,  0.316, 0],       # Joint 3
        [0.0825, np.pi/2,  0,     0],       # Joint 4
        [-0.0825, -np.pi/2, 0.384, 0],     # Joint 5
        [0,      np.pi/2,  0,     0],       # Joint 6
        [0.088,  np.pi/2,  0.107, 0]       # Joint 7
    ])
    
    N = joint_angles.shape[0]
    positions = {}
    
    for n in range(N):
        T = np.eye(4)  # Base transform
        
        # Add base offset (Panda base is 0.333m high)
        T_base = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, 0.333],
            [0, 0, 0, 1]
        ])
        T = T_base
        
        for i in range(7):
            a, alpha, d, offset = dh[i]
            theta = joint_angles[n, i] + offset
            
            # DH transformation
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            
            T_i = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])
            T = T @ T_i
            
            # Store link positions
            if i == 3:  # Elbow (link 4)
                positions['elbow'] = T[:3, 3].copy()
            elif i == 6:  # Wrist (link 7) 
                positions['wrist'] = T[:3, 3].copy()
        
        # Final end-effector position
        positions['gripper'] = T[:3, 3].copy()
    
    return positions

def update_project_single_joint_position_to_image(joint_positions, 
                                                  extrinsics, 
                                                  intrinsics, 
                                                  arm="right"):
    """Updated with corrected FK."""
    if joint_positions.shape[0] == 1:
        # Use corrected forward kinematics
        pos = panda_fk_corrected(joint_positions)
        
        gripper_pos = pos['gripper'].reshape(1, -1)
        wrist_pos = pos['wrist'].reshape(1, -1) 
        elbow_pos = pos['elbow'].reshape(1, -1)
        
        return gripper_pos, wrist_pos, elbow_pos
    
    # Handle batch processing
    all_positions = {'gripper': [], 'wrist': [], 'elbow': []}
    
    for i in range(joint_positions.shape[0]):
        pos = panda_fk_corrected(joint_positions[i:i+1])
        all_positions['gripper'].append(pos['gripper'])
        all_positions['wrist'].append(pos['wrist'])
        all_positions['elbow'].append(pos['elbow'])
    
    gripper_pos = np.array(all_positions['gripper'])
    wrist_pos = np.array(all_positions['wrist'])
    elbow_pos = np.array(all_positions['elbow'])
    
    return gripper_pos, wrist_pos, elbow_pos

# def joint_angles_to_xyz_positions(joint_angles, urdf_path=None):
#     """
#     Convert joint angles to 3D XYZ positions for Panda arm.
    
#     Args:
#         joint_angles: (N, 7) array of joint angles in radians
#         urdf_path: path to panda URDF file
    
#     Returns:
#         positions: dict of joint positions in 3D space
#     """
#     # Setup kinematic chain
#     if urdf_path is None:
#         # Use default Panda URDF
#         chain = pk.build_serial_chain_from_urdf(
#             open("panda.urdf").read(), "panda_hand"
#         )
#     else:
#         chain = pk.build_serial_chain_from_urdf(
#             open(urdf_path).read(), "panda_hand"
#         )
    
#     # Convert to torch tensor
#     q = torch.tensor(joint_angles, dtype=torch.float32)
    
#     # Forward kinematics for all joints
#     fk_result = chain.forward_kinematics(q, end_only=False)
    
#     # Extract positions for key joints
#     positions = {}
    
#     for joint_name in fk_result:
#             # Extract 4x4 transformation matrix
#             transform = fk_result[joint_name].get_matrix()
#             # Get XYZ position (translation part)
#             positions[joint_name] = transform[:, :3, 3].numpy()
    
#     return positions

# def get_end_effector_pose(joint_angles):
#     """Get end-effector position and orientation."""
#     positions = joint_angles_to_xyz_positions(joint_angles)
    
#     # End-effector position
#     ee_pos = positions['panda_hand']  # (N, 3)
    
#     return ee_pos

# Usage in your masking utils:
# def update_project_single_joint_position_to_image(joint_positions, extrinsics, intrinsics, arm="right"):
#     """Updated version using proper FK."""
#     # Get all joint positions in 3D
#     # import ipdb; ipdb.set_trace()
#     joint_positions_xyz = joint_angles_to_xyz_positions(joint_positions)  
#     # import ipdb; ipdb.set_trace()
#     # Extract specific joint positions
#     gripper_pos = joint_positions_xyz['panda_hand']
#     wrist_pos = joint_positions_xyz['panda_link7'] 
#     elbow_pos = joint_positions_xyz['panda_link6']
    
#     # Transform to camera frame
#     gripper_cam = ee_pose_to_cam_frame(gripper_pos, extrinsics)
#     wrist_cam = ee_pose_to_cam_frame(wrist_pos, extrinsics)
#     elbow_cam = ee_pose_to_cam_frame(elbow_pos, extrinsics)

#     return gripper_cam, wrist_cam, elbow_cam

# Example joint angles for Panda arm (7 values)
if __name__ == "__main__":
    joint_positions = np.array([[-0.61086524, -0.61086524, 0.0, -2.35619449,
                                0.0, 1.83259571, 0.78539816]])
    # print (joint_positions.shape)
    extrinsics = np.array([[ 0.02003261, -0.69701915,  0.71677263, -0.82825142],
        [-0.9996949 , -0.00360283,  0.02443628, -0.01548906],
        [-0.01445014, -0.71704346, -0.69687866,  0.87998277],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    intrinsics = np.array([[616.12902832,   0.        , 321.26901245,   0.        ],
        [  0.        , 615.75799561, 247.86399841,   0.        ],
        [  0.        ,   0.        ,   1.        ,   0.        ]])
    a, b, c = update_project_single_joint_position_to_image(joint_positions, extrinsics, intrinsics, arm="left")
#     ee_poses[0]
# array([-0.246299  , -0.05888327,  0.70566622,  0.31499997, -0.0853572 ,
#         0.70584114])
    print ("a[0]: ", a[0])
    expected = np.array([-0.246299, -0.05888327, 0.70566622])
    assert np.allclose(a[0], expected, atol=1e-6), f"Got {a[0]}, expected {expected}"

# print ("a: ", a.shape, b.shape, c.shape)