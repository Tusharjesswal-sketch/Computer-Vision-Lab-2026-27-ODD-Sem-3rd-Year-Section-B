import cv2
import numpy as np
import os

video_path = "sample.mp4"
if not os.path.exists(video_path):
    print("ERROR: sample.mp4 not found!")
    print("Put sample.mp4 in the same folder as optical_flow.py")
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

lk_output = cv2.VideoWriter(
    "lucas_kanade_output.mp4",
    fourcc,
    fps,
    (width, height)
)

farneback_output = cv2.VideoWriter(
    "farneback_output.mp4",
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

mask = np.zeros_like(old_frame)

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    lk_frame = frame.copy()

    if p0 is not None:

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
            **lk_params
        )

        if p1 is not None:

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for new, old in zip(
                good_new,
                good_old
            ):

                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()

                x_new = int(x_new)
                y_new = int(y_new)

                x_old = int(x_old)
                y_old = int(y_old)

                mask = cv2.line(
                    mask,
                    (x_new, y_new),
                    (x_old, y_old),
                    (255, 255, 255),
                    2
                )
                lk_frame = cv2.circle(
                    lk_frame,
                    (x_new, y_new),
                    4,
                    (0, 255, 0),
                    -1
                )

            lk_frame = cv2.add(
                lk_frame,
                mask
            )

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

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )
    hsv = np.zeros_like(frame)

    hsv[..., 1] = 255

    hsv[..., 0] = (
        angle * 180 / np.pi / 2
    )

    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    farneback_frame = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    cv2.putText(
        lk_frame,
        "Lucas-Kanade Sparse Optical Flow",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        farneback_frame,
        "Farneback Dense Optical Flow",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    lk_output.write(lk_frame)
    farneback_output.write(farneback_frame)

    old_gray = frame_gray.copy()

    frame_number += 1

    if frame_number % 20 == 0:
        print(
            "Processed frames:",
            frame_number
        )

cap.release()
lk_output.release()
farneback_output.release()

print()
print("================================")
print("Processing completed!")
print("================================")
print()
print("Output files created:")
print("1. lucas_kanade_output.mp4")
print("2. farneback_output.mp4")