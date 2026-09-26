import cv2
import numpy as np
import os
import math

video_path = "sample.mp4"

if not os.path.exists(video_path):
    print("ERROR: sample.mp4 not found!")
    print("Place sample.mp4 in the same folder as this Python file.")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open sample.mp4")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

print("Video opened successfully")
print("Width:", width)
print("Height:", height)
print("FPS:", fps)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output = cv2.VideoWriter(
    "object_tracking_output.mp4",
    fourcc,
    fps,
    (width, height)
)

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

ret, old_frame = cap.read()

if not ret:
    print("ERROR: Could not read video.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(
    old_frame,
    cv2.COLOR_BGR2GRAY
)

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

trajectory = np.zeros_like(old_frame)
previous_center = None
total_distance = 0

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    if p0 is not None and len(p0) > 0:

        p1, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
            **lk_params
        )

        if p1 is not None:

            good_new = p1[status == 1]
            good_old = p0[status == 1]

            if len(good_new) > 0:

                center_x = int(
                    np.mean(good_new[:, 0])
                )

                center_y = int(
                    np.mean(good_new[:, 1])
                )

                current_center = (
                    center_x,
                    center_y
                )

                if previous_center is not None:

                    dx = (
                        current_center[0]
                        - previous_center[0]
                    )

                    dy = (
                        current_center[1]
                        - previous_center[1]
                    )

                    magnitude = math.sqrt(
                        dx * dx + dy * dy
                    )

                    total_distance += magnitude
                    cv2.arrowedLine(
                        frame,
                        previous_center,
                        current_center,
                        (0, 255, 255),
                        2,
                        tipLength=0.3
                    )

                    angle = math.degrees(
                        math.atan2(dy, dx)
                    )

                    if -22.5 <= angle < 22.5:
                        direction = "RIGHT"

                    elif 22.5 <= angle < 67.5:
                        direction = "DOWN-RIGHT"

                    elif 67.5 <= angle < 112.5:
                        direction = "DOWN"

                    elif 112.5 <= angle < 157.5:
                        direction = "DOWN-LEFT"

                    elif angle >= 157.5 or angle < -157.5:
                        direction = "LEFT"

                    elif -157.5 <= angle < -112.5:
                        direction = "UP-LEFT"

                    elif -112.5 <= angle < -67.5:
                        direction = "UP"

                    else:
                        direction = "UP-RIGHT"
                    cv2.putText(
                        frame,
                        "Direction: " + direction,
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Motion: {:.2f} pixels".format(
                            magnitude
                        ),
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                cv2.circle(
                    frame,
                    current_center,
                    8,
                    (0, 0, 255),
                    -1
                )
                if previous_center is not None:

                    trajectory = cv2.line(
                        trajectory,
                        previous_center,
                        current_center,
                        (255, 255, 255),
                        2
                    )

                frame = cv2.add(
                    frame,
                    trajectory
                )

                previous_center = current_center

                p0 = good_new.reshape(-1, 1, 2)

    flow = cv2.calcOpticalFlowFarneback(
        old_gray,
        frame_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    magnitude_dense, angle_dense = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    hsv = np.zeros_like(frame)

    hsv[..., 1] = 255

    hsv[..., 0] = (
        angle_dense * 180 / np.pi / 2
    )

    hsv[..., 2] = cv2.normalize(
        magnitude_dense,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    dense_flow = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


    cv2.putText(
        frame,
        "Lucas-Kanade Object Tracking",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    output.write(frame)
    if frame_count == 0:

        dense_output = cv2.VideoWriter(
            "farneback_tracking_output.mp4",
            fourcc,
            fps,
            (width, height)
        )

    dense_output.write(dense_flow)

    old_gray = frame_gray.copy()

    frame_count += 1
    if p0 is None or len(p0) < 10:

        p0 = cv2.goodFeaturesToTrack(
            old_gray,
            mask=None,
            **feature_params
        )

    if frame_count % 20 == 0:

        print(
            "Processed frames:",
            frame_count
        )

cap.release()
output.release()
dense_output.release()

print()
print("===================================")
print("Experiment 8 completed!")
print("===================================")
print()
print("Output files:")
print("1. object_tracking_output.mp4")
print("2. farneback_tracking_output.mp4")
print()
print("Total frames:", frame_count)
print(
    "Total tracked displacement: {:.2f} pixels".format(
        total_distance
    )
)