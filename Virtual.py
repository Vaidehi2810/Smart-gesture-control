import cv2
import mediapipe as mp
import pyautogui
import time
import pyttsx3
import math
import threading

# Speak function (runs in background thread)
def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cooldown = 1.0
last_action_time = 0
is_muted = False
cursor_mode = False

screen_w, screen_h = pyautogui.size()

def distance(point1, point2):
    return math.hypot(point1.x - point2.x, point1.y - point2.y)

def fingertips_close(hand_landmarks, threshold=0.07):
    tips = [4, 8, 12, 16, 20]
    tip_points = [hand_landmarks.landmark[i] for i in tips]
    for i in range(len(tip_points)):
        for j in range(i + 1, len(tip_points)):
            if distance(tip_points[i], tip_points[j]) > threshold:
                return False
    return True

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    dips = [3, 6, 10, 14, 18]
    fingers = []

    for tip, dip in zip(tips, dips):
        tip_y = hand_landmarks.landmark[tip].y
        dip_y = hand_landmarks.landmark[dip].y
        if tip_y < dip_y - 0.03:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def detect_gesture(fingers):
    if fingers == [1, 1, 1, 1, 1]:
        return "start_presentation"
    if fingers == [0, 0, 0, 0, 0]:
        return "stop_presentation"
    elif fingers == [1, 1, 1, 1, 0]:
        return "next_slide"
    elif fingers == [1, 1, 0, 0, 1]:
        return "previous_slide"
    elif fingers == [1, 0, 0, 0, 0]:
        return "unmute"
    elif fingers == [0, 0, 0, 0, 1]:
        return "mute"
    elif fingers == [1, 1, 0, 0, 0]:
        return "volume_up"
    elif fingers == [1, 0, 0, 0, 1]:
        return "volume_down"
    elif fingers == [1, 1, 0, 0, 0]:
        return "single_click"
    elif fingers == [1, 1, 1, 0, 0]:
        return "double_click"
    elif fingers == [0, 0, 1, 1, 1]:
        return "right_click"
    elif fingers == [0, 1, 0, 0, 0]:
        return "move_cursor"
    else:
        return None

# Start webcam
cap = cv2.VideoCapture(0)
print("Starting Gesture Control. Press 'q' to quit.")
speak("Gesture Control Started")

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    img_h, img_w, _ = img.shape
    gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw fingertip circles for visual feedback
            for tip_id in [4, 8, 12, 16, 20]:
                x = int(hand_landmarks.landmark[tip_id].x * img_w)
                y = int(hand_landmarks.landmark[tip_id].y * img_h)
                cv2.circle(img, (x, y), 10, (0, 255, 255), cv2.FILLED)

            fingers = fingers_up(hand_landmarks)

            # Check for "stop_presentation" using fingertip proximity
            if fingertips_close(hand_landmarks):
                gesture = "stop_presentation"
            else:
                gesture = detect_gesture(fingers)

            current_time = time.time()

            # Move Cursor
            if gesture == "move_cursor":
                cursor_mode = True
                if (current_time - last_action_time) > cooldown:
                    speak("Cursor control activated")
                    last_action_time = current_time

                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * screen_w)
                y = int(index_finger_tip.y * screen_h)
                pyautogui.moveTo(x, y)
                continue

            # Reset cursor_mode if not in gesture
            cursor_mode = False

            if gesture and (current_time - last_action_time) > cooldown:
                print(f"Gesture detected: {gesture}")
                last_action_time = current_time

                if gesture == "start_presentation":
                    pyautogui.press('f5')
                    speak("Starting presentation")
                elif gesture == "stop_presentation":
                    pyautogui.press('esc')
                    speak("Stopping presentation")
                elif gesture == "next_slide":
                    pyautogui.press('right')
                    speak("Next slide")
                elif gesture == "previous_slide":
                    pyautogui.press('left')
                    speak("Previous slide")
                elif gesture == "mute":
                    if not is_muted:
                        pyautogui.press('volumemute')
                        is_muted = True
                        speak("Muted")
                elif gesture == "unmute":
                    if is_muted:
                        pyautogui.press('volumemute')
                        is_muted = False
                        speak("Unmuted")
                elif gesture == "volume_up":
                    pyautogui.press('volumeup')
                    speak("Volume up")
                elif gesture == "volume_down":
                    pyautogui.press('volumedown')
                    speak("Volume down")
                elif gesture == "single_click":
                    pyautogui.click()
                    speak("Click")
                elif gesture == "double_click":
                    pyautogui.doubleClick()
                    speak("Double click")
                elif gesture == "right_click":
                    pyautogui.rightClick()
                    speak("Right click")

    cv2.imshow("Smart Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        speak("Exiting gesture control")
        break

cap.release()
cv2.destroyAllWindows()
