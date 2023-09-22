import cv2
import numpy as np


def EulerAngles(R):
    # Calculate rotation angles from rotation matrix
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


class ArucoPoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        return corners, ids

    def estimate_pose(self, corners, marker_length=1.0):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, self.camera_matrix,
                                                              self.dist_coeffs)
        return rvecs, tvecs

    def estimate_distance(self, corners, marker_length):
        # Assuming square marker and only one detected for simplicity
        corner = corners[0][0]

        # Calculate the size of the marker in pixels
        d_pixels = np.linalg.norm(corner[0] - corner[1])

        # Focal length in pixels can be obtained from the camera matrix
        # It's the element at row 1, column 1
        focal_length = self.camera_matrix[0, 0]

        # Calculate distance in meters
        distance_meters = (focal_length * marker_length) / d_pixels

        return distance_meters * 100  # Convert to cm

    def draw(self, frame, corners, ids, rvecs, tvecs):
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(rvecs)):
            cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.23)
            rotM = cv2.Rodrigues(rvecs[i])[0]
            angles = EulerAngles(rotM) * (180 / np.pi)
            cv2.putText(frame, f"Pitch: {angles[0]:.2f}, Yaw: {angles[1]:.2f}, Roll: {angles[2]:.2f}",
                        (50, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if len(corners) > 0:
                distance_cm = self.estimate_distance(corners, 0.1)  # 0.1 meters or 10 cm as the actual marker length
                cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (50, 100 + i * 100), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)


if __name__ == "__main__":
    cap = cv2.VideoCapture(-1)
    out = cv2.VideoWriter('results/result.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    camera_matrix = np.array([[532.46305079, 0, 332.79571817], [0, 533.80458011, 221.00816556], [0, 0, 1]],
                             dtype=np.float32)
    dist_coeffs = np.array([-0.39688559, 0.17036189, 0.00482907, 0.0006105, -0.00245277], dtype=np.float32)
    aruco_estimator = ArucoPoseEstimator(camera_matrix, dist_coeffs)

    while True:
        ret, frame = cap.read()
        corners, ids = aruco_estimator.detect_markers(frame)

        if ids is not None:
            rvecs, tvecs = aruco_estimator.estimate_pose(corners)
            aruco_estimator.draw(frame, corners, ids, rvecs, tvecs)
        out.write(frame)
        cv2.imshow("Aruco Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
