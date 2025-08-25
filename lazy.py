import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui

# === System Volume Setup ===
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume_ctrl.GetVolumeRange()[:2]

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

# === Mouse Screen Dimensions ===
screen_w, screen_h = pyautogui.size()

# === Tilt Detection Memory ===
tilt_history = deque(maxlen=2)

def fingers_status(lm):
    return [
        1 if lm[4][0] > lm[3][0] else 0,   # Thumb
        1 if lm[8][1] < lm[6][1] else 0,   # Index
        1 if lm[12][1] < lm[10][1] else 0, # Middle
        1 if lm[16][1] < lm[14][1] else 0, # Ring
        1 if lm[20][1] < lm[18][1] else 0  # Pinky
    ]

def get_head_tilt(landmarks):
    left_eye = np.array([landmarks[33].x, landmarks[33].y])
    right_eye = np.array([landmarks[263].x, landmarks[263].y])
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return np.degrees(np.arctan2(dy, dx))

def trigger_volume_action():
    if len(tilt_history) < 2:
        return None
    (dir1, t1), (dir2, t2) = tilt_history
    if dir1 == dir2 and abs(t2 - t1) < 1.5:
        return dir1
    return None

# === Main Webcam Loop ===
cap = cv2.VideoCapture(0)
scroll_cooldown = 0.3
click_cooldown = 1.0
last_scroll = last_click = last_back = time.time()

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process face and hand landmarks
    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)
    now = time.time()

    # === Face Mesh: Volume Control ===
    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0].landmark
        angle = get_head_tilt(face_landmarks)

        # Track head tilts
        if angle > 15:
            tilt_history.append(("right", now))
        elif angle < -15:
            tilt_history.append(("left", now))

        action = trigger_volume_action()
        if action == "right":
            volume_ctrl.SetMasterVolumeLevelScalar(0.6, None)
            tilt_history.clear()
            print("Volume set to 60%")
        elif action == "left":
            curr_vol = volume_ctrl.GetMasterVolumeLevelScalar()
            volume_ctrl.SetMasterVolumeLevelScalar(max(0.0, curr_vol - 0.3), None)
            tilt_history.clear()
            print("Volume reduced by 30%")

    # === Hand Gestures: Mouse & Scroll ===
    if hand_result.multi_hand_landmarks:
        lmList = []
        for lm in hand_result.multi_hand_landmarks[0].landmark:
            lmList.append((int(lm.x * w), int(lm.y * h)))

        fingers = fingers_status(lmList)

        # Palm center for mouse movement
        cx = int((lmList[0][0] + lmList[5][0] + lmList[17][0]) / 3)
        cy = int((lmList[0][1] + lmList[5][1] + lmList[17][1]) / 3)

        screen_x = np.interp(cx, [0, w], [0, screen_w])
        screen_y = np.interp(cy, [0, h], [0, screen_h])
        pyautogui.moveTo(screen_x, screen_y)

        # Scroll and Click Gestures
        if fingers == [0,1,0,0,0] and now - last_scroll > scroll_cooldown:
            pyautogui.scroll(30)
            last_scroll = now
            print("Scroll Up")
        elif fingers == [0,0,0,0,1] and now - last_scroll > scroll_cooldown:
            pyautogui.scroll(-30)
            last_scroll = now
            print("Scroll Down")
        elif sum(fingers) == 0 and now - last_click > click_cooldown:
            pyautogui.click()
            last_click = now
            print("Click")
        elif fingers == [1,0,0,0,0] and now - last_back > click_cooldown:
            pyautogui.press('backspace')
            last_back = now
            print("Back")

    # Display Webcam Feed
    cv2.imshow("Face + Hand Gesture Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
