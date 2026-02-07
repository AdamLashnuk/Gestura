import cv2
import mediapipe as mp
import csv
import os

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

DATA_PATH = "data/asl_letters.csv"

TARGET_SAMPLES = 80
labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

counts = {label: 0 for label in labels}

# Create CSV if it doesn't exist
if not os.path.exists(DATA_PATH):
    with open(DATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    normalized = []
    for lm in landmarks:
        normalized.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return normalized

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        key = cv2.waitKey(1) & 0xFF
        char = chr(key).upper() if key != 255 else None

        if char in labels and counts[char] < TARGET_SAMPLES:
            row = normalize_landmarks(hand_landmarks.landmark)
            row.append(char)

            with open(DATA_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            counts[char] += 1

    # Display counters
    y = 30
    for label in labels:
        cv2.putText(
            frame,
            f"{label}: {counts[label]} / {TARGET_SAMPLES}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        y += 30

    cv2.imshow("ASL Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
