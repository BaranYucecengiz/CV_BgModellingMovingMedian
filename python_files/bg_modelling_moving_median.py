import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()

frame_queue = deque(maxlen=5)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_queue.append(gray_frame)

    if len(frame_queue) == frame_queue.maxlen:
        median_background = np.median(np.array(frame_queue), axis=0).astype(dtype=np.uint8)

        foreground_mask = cv2.absdiff(gray_frame, median_background)
        _, foreground_mask = cv2.threshold(foreground_mask, 30, 255, cv2.THRESH_BINARY)

        cv2.imshow('Original Frame', frame)
        cv2.imshow('Median Background Model', median_background)
        cv2.imshow('Foreground Mask', foreground_mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()