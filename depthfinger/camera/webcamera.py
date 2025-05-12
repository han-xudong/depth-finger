#!/usr/bin/env python

import argparse
import sys
from typing import Tuple
import time
import cv2
import yaml
import numpy as np
from typing import Optional
from collections import deque
from scipy.spatial.transform import Rotation as R
from ..modules.zmq import CameraSubscriber


class WebCamera:
    """
    WebCamera class.

    The class is to get the image from a web camera and detect the ArUco markers.
    The poses of the markers are estimated and filtered.
    """

    def __init__(
        self,
        name: str,
        camera_params: dict,
        detector_params: Optional[dict] = None,
    ) -> None:
        """
        Initialize the WebCamera.

        Args:
            name: The name of the camera.
            camera_params: The camera parameters.
            detector_params: The detector parameters.
        """

        # Camera initialization
        self.name = name
        print("{:-^80}".format(f" {self.name} Initialization "))
        # Set the camera parameters
        self.mode = "web"
        print(f"Camera mode: {self.mode.upper()}")
        self.width = camera_params["width"]
        self.height = camera_params["height"]
        self.mtx = np.array(camera_params["mtx"])
        self.dist = np.array(camera_params["dist"])
        self.camera = CameraSubscriber(ip=camera_params["ip"], port=camera_params["port"])
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Camera matrix:\n{self.mtx}")
        print(f"Camera distortion:\n{self.dist}")
        print("Camera Initialization Done.")
        print("{:-^80}".format(""))

        # ArUco detector initialization
        print("{:-^80}".format(" ArUco Detector Initialization "))
        # Set the detector parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        aruco_detector_params = cv2.aruco.DetectorParameters()
        if detector_params is not None:
            for key in detector_params:
                aruco_detector_params.__setattr__(key, detector_params[key])
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_detector_params)
        self.aruco_estimate_params = cv2.aruco.EstimateParameters()
        self.aruco_estimate_params.solvePnPMethod = cv2.SOLVEPNP_IPPE_SQUARE
        # Set the marker size
        self.marker_size = camera_params["marker_size"]
        print(f"Marker size: {self.marker_size}")
        # Set the translation and rotation from marker frame to global frame
        self.transfer_tvec = np.array(camera_params["marker2global"]["translation"])
        self.transfer_rmat = np.array(camera_params["marker2global"]["rotation"])
        # Set the initial pose
        self.init_pose = np.zeros(6)
        # Set the pose
        self.pose = np.zeros(6)
        # Set the filter parameters
        self.filter_on = camera_params["filter_on"]
        self.filter_frame = camera_params["filter_frame"]
        print(f"Pose Filter: {self.filter_on}")
        if self.filter_on:
            print(f"Filter frame: {self.filter_frame}")
        self.last_pose = np.zeros(6)
        self.img = np.zeros((self.height, self.width, 3))
        self.first_frame = True

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # self.sharpen_kernel = np.ones((5, 5), np.float32) / 25

        # Init pose
        print(f"Calculating the initial pose of {self.name} ...")
        self.init_pose = self._calculateInitPose()
        self.init_tvec = self.init_pose[:, :3]
        self.init_rvec = self.init_pose[:, 3:]
        self.init_rmat = [R.from_rotvec(rvec).as_matrix() for rvec in self.init_rvec]

        print(f"Initial pose: {self.init_pose}")
        print(f"The finger motion will be calculated as reference pose base on this initial pose.")
        print("ArUco Detector Initialization Done.")
        print("{:-^80}".format(""))

    def _calculateInitPose(self) -> np.ndarray:
        """
        Calculate the initial pose.

        After 60 frames, the function will calculate the mean of the pose and return it.

        Args:
            None

        Returns:
            pose (np.ndarray([n, 6])): The pose vector.
        """

        # Create lists to store the tvec and rvec
        tvec_list = []
        rvec_list = []

        # Get the pose for 60 frames
        for i in range(60):
            pose, _ = self.readImageAndPose()
            tvec_list.append(pose[:, :3])
            rvec_list.append(pose[:, 3:])

        # Calculate the mean of n poses
        tvec_list = np.array(tvec_list).reshape(-1, 3)
        tvec = np.mean(tvec_list, axis=0)
        rvec_list = np.array(rvec_list).reshape(-1, 3)
        rvec = R.from_rotvec(rvec_list).mean().as_rotvec()
        pose = np.hstack((tvec, rvec)).reshape(-1, 6)

        return pose

    def _poseFilter(self, pose: np.ndarray) -> np.ndarray:
        """
        Filter the pose.

        The function is to filter the pose by the mean of the pose list.
        If the pose list is less than the frame, the pose will be appended to the pose list directly.
        Otherwise, the first pose will be popped and the pose will be appended to the pose list.

        Args:
            pose (np.ndarray([n, 6])): The pose vector.

        Returns:
            filtered_pose (np.ndarray([n, 6])): The filtered pose vector.
        """

        # Initialize the pose deque if not already initialized
        if not hasattr(self, "pose_history"):
            self.pose_history = deque(maxlen=self.filter_frame)

        # Append the pose to the pose history
        self.pose_history.append(pose)

        filtered_pose = []
        for i in range(len(pose)):
            # Calculate the mean of the pose list
            pose_list = np.array([pose[i] for pose in self.pose_history])
            # Calculate the mean of the tvec and rvec
            tvec_list = pose_list[:, :3]
            tvec = np.mean(tvec_list, axis=0)
            rvec_list = pose_list[:, 3:]
            rvec = R.from_rotvec(rvec_list).mean().as_rotvec()

            # Append the filtered pose to the filtered pose list
            filtered_pose.append(np.hstack((tvec, rvec)))

        filtered_pose = np.array(filtered_pose).reshape(-1, 6)
        # Copy the filtered pose to the last pose
        self.pose_history[-1] = filtered_pose

        # Return the filtered pose
        return filtered_pose

    def _imageToPose(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the pose and image from the camera.

        The function is to get the pose and image from the camera.
        First, the image is converted to the gray image and filtered by the kernel.
        Then, the ArUco markers are detected and the pose is estimated.
        Finally, the pose is filtered and the markers are drawn on the image.

        Args:
            img (np.ndarray([height, width, 3])): The image captured by the camera.

        Returns:
            pose (np.ndarray([n, 6])): The pose vector.
            img (np.ndarray([height, width, 3])): The image with the markers.
        """

        # Convert the image to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply the bilateral filter to the gray image
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
        # Apply the CLAHE to the gray image
        gray = self.clahe.apply(gray)
        # Apply the filter to the gray image
        # gray = cv2.filter2D(gray, -1, self.sharpen_kernel)

        # Detect the markers
        corners, ids, _ = self.detector.detectMarkers(gray)

        # Check if the markers are detected
        if ids is None:
            return np.zeros([1, 6]), img

        # Estimate the pose using IPPE_SQUARE
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.mtx, self.dist)
        pose = np.hstack((tvec * 1000, rvec)).reshape(-1, 6)

        # Filter the pose
        if self.filter_on:
            pose = self._poseFilter(pose)

        # Draw the markers
        color_image_result = cv2.aruco.drawDetectedMarkers(img, corners, ids)
        color_image_result = cv2.drawFrameAxes(img, self.mtx, self.dist, rvec, tvec, self.marker_size)
        # Return the pose and image
        return pose, color_image_result

    def release(self) -> None:
        """
        Release the camera.

        The function will close the window.
        """

        # Release the camera
        self.camera.close()
        # Close the window
        cv2.destroyAllWindows()

    def readImage(self) -> np.ndarray:
        """
        Read the image from the camera.

        Using the OpenCV, the function will read the image from the camera.
        If the image is not read, the function will raise an error.

        Args:
            None

        Returns:
            img (np.ndarray([height, width, 3])): The image captured by the camera.

        Raises:
            ValueError: Cannot read the image from the camera.
        """

        # Read the image from the camera
        try:
            img = cv2.imdecode(self.camera.subscribeMessage(), cv2.IMREAD_COLOR)
        except ValueError:
            print("\033[31mCannot read the image from the camera.\033[0m")
            sys.exit()
        return img

    def readImageAndPose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the pose and image from the camera.

        The function is to read the pose and image from the camera.
        The pose and image are got from the _imageToPose function.

        Returns:
            pose (np.ndarray([n, 6])): The pose vector.
            img (np.ndarray([height, width, 3])): The image with the markers.
        """

        # Read the image from the camera
        img = self.readImage()

        # Get the pose and image from the camera
        pose, img = self._imageToPose(img)

        # Check if the first frame
        if self.first_frame:
            self.first_frame = False
            self.last_pose = pose

        # Check if the pose is valid
        if np.linalg.norm(pose[:3] - self.last_pose[:3]) > 20:
            self.pose = self.last_pose
        else:
            self.pose = pose
        self.last_pose = self.pose
        self.img = img

        # Return the pose and image
        return self.pose, self.img

    def poseToReferece(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert the pose to the reference pose.

        The function is to calculate the reference pose by the initial pose.
        The tvec is the difference between the current tvec and the initial tvec.
        The rvec is the matrix multiplication of the inverse of the initial rvec and the current rvec.

        Args:
            pose (np.ndarray([n, 6])): The pose vector.

        Returns:
            ref_pose (np.ndarray([n, 6])): The reference pose vector.
        """

        pose = pose.reshape(-1, 6)
        ref_pose = []
        for i in range(len(pose)):
            # Convert rvec to rotation matrix
            rvec = pose[i, 3:]  # rx, ry, rz
            rmat = R.from_rotvec(rvec).as_matrix()  # Current rotation matrix

            # Calculate the relative rotation (initial_rmat is the initial rotation matrix)
            rmat = np.linalg.inv(self.init_rmat[i]) @ rmat
            rvec = R.from_matrix(rmat).as_rotvec()  # Convert back to rotation vector

            # Convert tvec (translation) to the reference frame
            tvec = np.linalg.inv(self.init_rmat[i]) @ (
                pose[i, :3] - self.init_tvec[i]
            )  # Compute the relative translation

            # Combine tvec and rvec into the reference pose
            ref_pose.append(np.hstack((tvec, rvec)))

        # Return the reference pose
        return np.array(ref_pose).reshape(-1, 6)

    def poseVectorToEuler(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert the pose to the euler angles.

        The function is to convert the rotation vector of the pose to the euler angles.
        The unit of the euler angles is radian.

        Args:
            pose (np.ndarray([n, 6])): The pose vector.

        Returns:
            pose_euler (np.ndarray([n, 6])): The pose with euler angles.
        """

        pose = pose.reshape(-1, 6)
        pose_euler = []
        for i in range(len(pose)):
            # Convert rvec to rotation matrix
            rvec = pose[i, 3:]
            rr = R.from_rotvec(rvec)
            rpy = rr.as_euler("xyz", degrees=False)
            # Create the euler pose
            pose_euler.append(np.hstack((pose[i, :3], rpy)))

        # Return the euler pose
        return np.array(pose_euler).reshape(-1, 6)

    def poseVectorToQuaternion(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert the pose to the quaternion.

        The function is to convert the pose to the quaternion.
        The quaternion is represented by [x, y, z, qx, qy, qz, qw].

        Args:
            pose (np.ndarray([n, 6])): The pose vector.

        Returns:
            pose_quat (np.ndarray([n, 7])): The pose quaternion.
        """

        pose = pose.reshape(-1, 6)
        pose_quat = []
        for i in range(len(pose)):
            # Convert rvec to rotation matrix
            rvec = pose[i, 3:]
            rr = R.from_rotvec(rvec)
            quat = rr.as_quat()
            # Create the quaternion pose
            pose_quat.append(np.hstack((pose[i, :3], quat)))

        # Return the quaternion pose
        return np.array(pose_quat).reshape(-1, 7)

    def poseVectorToMatrix(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert the pose to the matrix.

        The function is to convert the pose to the matrix.
        The matrix is represented by
        [[r11, r12, r13, x],
         [r21, r22, r23, y],
         [r31, r32, r33, z],
         [  0,   0,   0, 1]].

        Args:
            pose (np.ndarray([n, 6])): The pose vector.

        Returns:
            pose_matrix (np.ndarray([n, 4, 4])): The pose matrix.
        """

        pose = pose.reshape(-1, 6)
        pose_matrix = []
        for i in range(len(pose)):
            # Convert rvec to rotation matrix
            rvec = pose[i, 3:]
            rr = R.from_rotvec(rvec)

            # Create the matrix pose
            pose_matrix.append(np.eye(4))
            pose_matrix[i][:3, :3] = rr.as_matrix()
            pose_matrix[i][:3, 3] = pose[i][:3]
            pose_matrix[i][3, 3] = 1.0

        # Return the matrix pose
        return np.array(pose_matrix).reshape(-1, 4, 4)

    def poseAxisTransfer(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert the pose from the marker frame to the global frame.

        Args:
            pose (np.ndarray([n, 6])): The pose vector in the marker frame.

        Returns:
            pose_transfer (np.ndarray([n, 6])): The pose vector in the global frame.
        """

        # Reshape the pose to [n, 6]
        pose_ori = pose.reshape(-1, 6)
        pose_transfer = []
        for i in range(len(pose_ori)):
            # Convert rvec to rotation matrix
            rotation_matrix = R.from_rotvec(pose_ori[i, 3:]).as_matrix()
            rvec = R.from_matrix(
                self.transfer_rmat[i] @ rotation_matrix @ self.transfer_rmat[i].T
            ).as_rotvec()

            # Convert tvec to the global frame
            tvec = self.transfer_rmat[i] @ pose_ori[i, :3]

            # Combine tvec and rvec into the global pose
            pose_transfer.append(np.hstack((tvec, rvec)))

        # Return the global pose
        return np.array(pose_transfer).reshape(-1, 6)


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="Camera",
        help="The name of the camera.",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default="./config/maixcam-88a7.yaml",
        help="The path of the camera parameters.",
    )
    parser.add_argument(
        "--show_img",
        action="store_true",
        help="Show the image captured by the camera (default: False).",
    )
    args = parser.parse_args()

    # Read the camera parameters
    with open(args.params_path, "r") as f:
        camera_params = yaml.load(f.read(), Loader=yaml.Loader)

    with open("./config/detector.yaml", "r") as f:
        detector_params = yaml.load(f.read(), Loader=yaml.Loader)

    try:
        # Create a camera
        camera = WebCamera(
            name=args.name,
            camera_params=camera_params,
            detector_params=detector_params,
        )
        show_img = args.show_img

        frame_count = 0
        start_time = time.time()
        # Start the loop
        while True:
            # Get the pose and image from the camera
            pose, frame = camera.readImageAndPose()
            # Convert the pose to the reference pose
            pose = camera.poseToReferece(pose)
            # Convert the pose to the euler angles
            pose = camera.poseVectorToEuler(pose)

            # Print the FPS
            frame_count += 1
            if frame_count == 50:
                print(
                    f"FPS: %.2f" % (frame_count / (time.time() - start_time)),
                )
                start_time = time.time()
                frame_count = 0

            if show_img:
                # Show the image
                cv2.imshow("Camera_0", frame)

                # Break the loop
                if cv2.waitKey(10) == 27:
                    break

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stop the camera.")
    finally:
        # Release the camera
        camera.release()
        print("Camera released.")
        print("Program terminated.")
