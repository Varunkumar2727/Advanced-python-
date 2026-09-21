"""
Draw lines on a live webcam feed using OpenCV.

Controls:
  - Left-click and drag  : draw a line
  - 'c'                  : clear all drawn lines
  - 's'                  : save current frame (with drawings) as image
  - 'q' / ESC            : quit

The drawings persist across frames (they stay on screen) because they are
stored in a separate transparent-style canvas and re-composited onto every
new frame.
"""

import cv2
import numpy as np

# ---------- Configuration ----------
CAM_INDEX = 0            # change if you have multiple cameras
LINE_COLOR = (0, 0, 255) # BGR - red
LINE_THICKNESS = 3
WINDOW_NAME = "Draw on Webcam"

# ---------- Globals for mouse handling ----------
drawing = False
start_point = None
end_point = None
lines = []  # list of (pt1, pt2) completed lines


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point, lines

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        lines.append((start_point, end_point))


def main():
    global lines

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("Left-click and drag to draw. Press 'c' to clear, 's' to save, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # mirror view, feels more natural

        # Draw all completed lines
        for pt1, pt2 in lines:
            cv2.line(frame, pt1, pt2, LINE_COLOR, LINE_THICKNESS)

        # Draw the line currently being dragged
        if drawing and start_point and end_point:
            cv2.line(frame, start_point, end_point, LINE_COLOR, LINE_THICKNESS)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('c'):
            lines = []
        elif key == ord('s'):
            filename = "saved_frame.png"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
