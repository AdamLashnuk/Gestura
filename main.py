import cv2
import mediapipe as mp
import time
import random
from classifier import SignClassifier

# ---------------- MODES ----------------
MODE_SELECT = 0
MODE_EDUCATION = 1
MODE_FREEWRITE = 2

mode = MODE_SELECT
score = 0

practice_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
target_letter = random.choice(practice_letters)

# ---------------- HAND TRACKING ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

classifier = SignClassifier()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- UX TIMING ----------------
CONFIRM_FRAMES = 12
COOLDOWN_TIME = 1.0

confirm_count = 0
last_success_time = 0

LETTER_HOLD_FRAMES = 30
hold_count = 0
last_letter = None

letter_buffer = []
final_text = ""

# ---------------- UI ----------------
edu_button = (170, 220, 470, 280)
free_button = (170, 310, 470, 370)

mouse_x, mouse_y = 0, 0
pTime = 0

# ---------------- BUTTON DRAW ----------------
def draw_button(frame, rect, text, hover=False):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1 + 4, y1 + 4), (x2 + 4, y2 + 4), (0, 0, 0), -1)
    color = (70, 180, 255) if hover else (40, 140, 220)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    tx = x1 + (x2 - x1 - text_size[0]) // 2
    ty = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# ---------------- MOUSE ----------------
def mouse_handler(event, x, y, flags, param):
    global mode, mouse_x, mouse_y
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        if edu_button[0] <= x <= edu_button[2] and edu_button[1] <= y <= edu_button[3]:
            mode = MODE_EDUCATION
        elif free_button[0] <= x <= free_button[2] and free_button[1] <= y <= free_button[3]:
            mode = MODE_FREEWRITE

cv2.namedWindow("ASL Translation")
cv2.setMouseCallback("ASL Translation", mouse_handler)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    key = cv2.waitKey(1) & 0xFF

    # -------- MENU --------
    if mode == MODE_SELECT:
        frame[:] = (20, 20, 20)

        cv2.putText(frame, "ASL Learning Assistant",
                    (120, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.putText(frame, "Choose a mode",
                    (210, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        draw_button(frame, edu_button, "Education Mode",
                    edu_button[0] <= mouse_x <= edu_button[2] and edu_button[1] <= mouse_y <= edu_button[3])
        draw_button(frame, free_button, "Free Write Mode",
                    free_button[0] <= mouse_x <= free_button[2] and free_button[1] <= mouse_y <= free_button[3])

        cv2.imshow("ASL Translation", frame)
        if key == 27:
            break
        continue

    # -------- HAND PROCESSING --------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        letter = classifier.predict(hand_landmarks.landmark)
        cv2.putText(frame, f"Detected: {letter}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # -------- EDUCATION MODE --------
        if mode == MODE_EDUCATION:
            current_time = time.time()
            cv2.putText(frame, f"Target: {target_letter}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)

            if current_time - last_success_time < COOLDOWN_TIME:
                status, color = "Nice!", (0, 200, 0)
            elif letter == target_letter:
                confirm_count += 1
                status, color = f"Hold... ({confirm_count}/{CONFIRM_FRAMES})", (0, 255, 255)
                if confirm_count >= CONFIRM_FRAMES:
                    score += 1
                    target_letter = random.choice(practice_letters)
                    confirm_count = 0
                    last_success_time = current_time
                    status, color = "Correct!", (0, 255, 0)
            else:
                confirm_count = 0
                status, color = "Incorrect", (0, 0, 255)

            cv2.putText(frame, status, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(frame, f"Score: {score}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # -------- FREE WRITE MODE --------
        elif mode == MODE_FREEWRITE:
            if key == 127 and letter_buffer:  # BACKSPACE
                letter_buffer.pop()

            if key == 32:  # SPACE commits
                final_text += "".join(letter_buffer)
                letter_buffer.clear()

            if letter == last_letter:
                hold_count += 1
            else:
                hold_count = 0
                last_letter = letter

            if hold_count >= LETTER_HOLD_FRAMES:
                if not letter_buffer or letter_buffer[-1] != letter:
                    letter_buffer.append(letter)
                hold_count = 0
                last_letter = None

    # -------- FREE WRITE UI --------
    if mode == MODE_FREEWRITE:
        cv2.putText(frame, "Current: " + "".join(letter_buffer), (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(frame, "Text: " + final_text, (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Holding: {hold_count}/{LETTER_HOLD_FRAMES}", (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

    # -------- FPS --------
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime else 0
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("ASL Translation", frame)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
