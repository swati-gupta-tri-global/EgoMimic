import numpy as np
import cv2
import os

hand_tracking_file = open("hand_tracking_data.txt", "r+")
hand_tracking_file.seek(0)

numData = len(hand_tracking_file.readlines())

poses = np.zeros((numData, 7))
images = np.zeros((numData, 480, 640, 3))

failed_to_read = False
prev_pose = None

hand_tracking_file.seek(0)


def project_3d_to_2d(x, y, z, camera_intrinsics):
    focal_length_x, focal_length_y, principal_point_x, principal_point_y = (
        camera_intrinsics
    )

    # Apply camera intrinsics to project 3D point to 2D
    u = (focal_length_x * x / z) + principal_point_x
    v = (focal_length_y * y / z) + principal_point_y

    return u, v


# focal_length_x, focal_length_y, principal_point_x, principal_point_y
camera_intrinsics = (616.16, 615.714, 313.42, 236.67)

for idx, line in enumerate(hand_tracking_file.readlines()):
    pose = line.split()
    if os.path.isfile("images/frame_" + str(pose[0])):
        image = cv2.imread("images/frame_" + str(pose[0]))
        x, y, z = np.array(pose[3:6], dtype=float)
        image = np.array(image)
        z -= 175

        # Project 3D to 2D
        u, v = project_3d_to_2d(-x, z, y, camera_intrinsics)
        print(f"2D Point: ({u:.2f}, {v:.2f})")
        point_color = (0, 255, 0)

        # Define the point radius
        point_radius = 10

        # Draw the point on the image
        cv2.circle(image, (int(u), int(v)), point_radius, point_color, -1)

        # Show the image with the point
        cv2.imshow("Image with Point", image)
        cv2.waitKey(20)

cv2.destroyAllWindows()

print("NumImages:", images.shape)
print("NumPoses:", poses.shape)
