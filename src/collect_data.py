# collect_data.py
import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)
outfile = os.path.join(OUT_DIR, f"hand_data_{int(time.time())}.csv")

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        # header: timestamp, fingertip_x_norm, fingertip_y_norm, landmark0_x,...landmark20_y, optional label
        header = ["ts", "idx_x", "idx_y"] + [f"lm{i}_x" for i in range(21)] + [f"lm{i}_y" for i in range(21)] + ["label"]
        writer.writerow(header)
        print("Recording to", outfile)
        label = ""  # if you want to add gesture labels during recording
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            h, w, _ = frame.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # index fingertip is landmark 8
                    idx = hand_landmarks.landmark[8]
                    idx_x, idx_y = idx.x, idx.y
                    lmx = [lm.x for lm in hand_landmarks.landmark]
                    lmy = [lm.y for lm in hand_landmarks.landmark]
                    ts = time.time()
                    row = [ts, idx_x, idx_y] + lmx + lmy + [label]
                    writer.writerow(row)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, "Press q to quit, l to set label", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('l'):
                label = input("Enter label for next records (empty to clear): ")

cap.release()
cv2.destroyAllWindows()
