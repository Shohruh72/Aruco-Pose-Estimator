# Aruco Pose Estimator
### This Python application uses OpenCV to detect ArUco markers and estimate their 3D pose relative to the camera.


## Table of Contents

1. [Features](https://github.com/RakhmatovShohruh/#Features)
2. [Installation](https://github.com/RakhmatovShohruh/Camera_Calibration#Installation)
3. [Usage](https://github.com/RakhmatovShohruh/Camera_Calibration#Usage)
4. [Generating](https://github.com/RakhmatovShohruh/#Features)

## Features

1. Detects ArUco markers in real-time video feed.
2. Estimates the 3D pose (orientation and position) of the markers.
3. Calculates and displays the pitch, yaw, and roll of the detected markers.
4. Estimates the distance from the camera to the marker.

## Installation
To run this project, you need to have Python and OpenCV installed. You can install OpenCV via pip:
```bash
$ conda env create -f environment.yml
$ conda activate xvision
```
## Usage
Clone this repository to your local machine.
Modify the `camera_matrix` and `dist_coeffs` variables with the camera calibration parameters specific to your camera.
Run `main.py`:
```bash
python main.py
```

## Generating Compatible ArUco Markers
To generate compatible ArUco markers, you can use the provided marker_generator.py script. To run the script:
```bash
python marker_generator.py
```
**By default, this will generate an ArUco marker using the 6x6_250 dictionary. You can modify the aruco_type variable in the script to generate different types of ArUco markers.**



The program will open a window displaying the real-time video feed with the detected ArUco markers(cv2.aruco.DICT_6X6_250). To stop the program, press 'q'.
