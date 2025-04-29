# --------------------- IMPORTS ---------------------
import tkinter as tk  # For GUI creation
from tkinter import messagebox  # For pop-up info boxes
import threading  # For running video capture in background
import cv2  # OpenCV for video stream handling
import mediapipe as mp  # MediaPipe for hand/face detection
import pyautogui  # For simulating keyboard input
import time  # For time tracking (cooldown)
import math  # For angle calculations if needed

# --------------------- COMMON FUNCTIONS ---------------------

# Initializes webcam with fallback to multiple backends
def init_capture(device_index=0):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(device_index, backend)
        if cap.isOpened():
            # Set video properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        else:
            cap.release()
    raise RuntimeError("Could not open webcam with any backend")

# Returns the status of all fingers (1=up, 0=down) based on hand landmarks
def get_fingers_status(hand_landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

    # Thumb - x-based comparison due to orientation
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers - y-based comparison
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Optional utility to calculate angle between 3 points (not used directly)
def calculate_angle(a, b, c):
    ang = math.degrees(math.atan2(c.y - b.y, c.x - b.x) -
                       math.atan2(a.y - b.y, a.x - b.x))
    return ang + 360 if ang < 0 else ang

# --------------------- GESTURE DETECTION ---------------------

# Maps finger status to specific gesture based on mode
def detect_gesture(hand_landmarks, mode='youtube'):
    fingers = get_fingers_status(hand_landmarks)

    if mode == 'youtube':
        if fingers == [1, 1, 0, 0, 1]: return "volume_up"
        if fingers == [0, 1, 0, 0, 1]: return "volume_down"
        if fingers == [1, 1, 1, 1, 1]: return "play_pause"
        if fingers == [0, 1, 0, 0, 0]: return "forward"
        if fingers == [0, 0, 0, 0, 1]: return "backward"
        if fingers == [0, 1, 1, 0, 0]: return "next_video"
        if fingers == [0, 1, 1, 1, 0]: return "mute_unmute"
        if fingers == [1, 0, 0, 0, 0]: return "full_screen"

    elif mode == 'ott':
        if fingers == [1, 1, 0, 0, 1]: return "volume_up"
        if fingers == [0, 1, 0, 0, 1]: return "volume_down"
        if fingers == [1, 1, 1, 1, 1]: return "play_pause"
        if fingers == [0, 1, 0, 0, 0]: return "forward"
        if fingers == [0, 0, 0, 0, 1]: return "backward"
        if fingers == [1, 0, 0, 0, 0]: return "full_screen"

    return None  # No gesture matched

# --------------------- ACTION PERFORM ---------------------

# Performs a keyboard action if cooldown allows
def perform_action(gesture, mode='youtube'):
    now = time.time()
    if now - perform_action.last_action_time < perform_action.action_cooldown:
        return  # Skip if within cooldown

    print("Action:", gesture)

    # Key mappings for YouTube
    yt_mapping = {
        "volume_up": lambda: pyautogui.press('up'),
        "volume_down": lambda: pyautogui.press('down'),
        "forward": lambda: pyautogui.press('l'),
        "backward": lambda: pyautogui.press('j'),
        "play_pause": lambda: pyautogui.press('k'),
        "next_video": lambda: pyautogui.hotkey('shift', 'n'),
        "mute_unmute": lambda: pyautogui.press('m'),
        "full_screen": lambda: pyautogui.press('f'),
    }

    # Key mappings for OTT platform
    ott_mapping = {
        "volume_up": lambda: pyautogui.press('up'),
        "volume_down": lambda: pyautogui.press('down'),
        "forward": lambda: pyautogui.press('right'),
        "backward": lambda: pyautogui.press('left'),
        "play_pause": lambda: pyautogui.press('space'),
        "full_screen": lambda: pyautogui.press('f11'),
        "exit_full_screen": lambda: pyautogui.press('esc'),
    }

    # Select mapping based on mode
    mapping = yt_mapping if mode == 'youtube' else ott_mapping

    if gesture in mapping:
        mapping[gesture]()  # Perform the action
        perform_action.last_action_time = now  # Update last action time

# Initialize cooldown variables
perform_action.last_action_time = 0
perform_action.action_cooldown = 1.0  # 1 second

# --------------------- MAIN CONTROL FUNCTION ---------------------

# Starts the controller loop
def start_controller(mode):
    cap = init_capture()  # Start camera
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Face detection init
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    prev_gesture = None
    gesture_confirm_count = 0
    gesture_confirm_threshold = 3
    face_missing = False  # Track if face is gone

    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.flip(img, 1)  # Mirror image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect face
            face_result = face_detection.process(img_rgb)
            if not face_result.detections:
                if not face_missing:
                    pyautogui.press('k' if mode == 'youtube' else 'space')  # Auto pause
                    face_missing = True
                continue
            else:
                face_missing = False

            # Detect hands
            result = hands.process(img_rgb)
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                    gesture = detect_gesture(handLms, mode)

                    if gesture == prev_gesture:
                        gesture_confirm_count += 1
                    else:
                        gesture_confirm_count = 0
                        prev_gesture = gesture

                    # Only act on gesture if detected several times consecutively
                    if gesture_confirm_count >= gesture_confirm_threshold and gesture:
                        perform_action(gesture, mode)
                        gesture_confirm_count = 0
                        prev_gesture = None

            # Show video
            cv2.imshow(f"{mode.capitalize()} Gesture Controller", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print("Error:", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()

# --------------------- GUI ---------------------

def launch_gui():
    window = tk.Tk()
    window.title("Gesture Controller")
    window.geometry("600x420")
    window.configure(bg="#0f0f1a")  # Dark theme
    window.resizable(True, True)

    # Hover effects for buttons
    def on_enter(e, btn, hover_color):
        btn['bg'] = hover_color

    def on_leave(e, btn, original_color):
        btn['bg'] = original_color

    # Start controller in background
    def start_youtube():
        messagebox.showinfo("Starting", "Launching YouTube Gesture Controller...")
        threading.Thread(target=start_controller, args=("youtube",), daemon=True).start()

    def start_ott():
        messagebox.showinfo("Starting", "Launching OTT Gesture Controller...")
        threading.Thread(target=start_controller, args=("ott",), daemon=True).start()

    # ---------------- Header ----------------
    heading = tk.Label(
        window, text="🖐️ Gesture Control Hub",
        font=("Helvetica", 28, "bold"),
        fg="#00ffe0", bg="#0f0f1a"
    )
    heading.pack(pady=(40, 10))

    subtitle = tk.Label(
        window, text="Seamlessly control media using just your hand!",
        font=("Helvetica", 13),
        fg="#cccccc", bg="#0f0f1a"
    )
    subtitle.pack()

    # ---------------- Button Creation ----------------
    def create_styled_button(text, bg_color, hover_color, command):
        btn = tk.Button(
            window, text=text,
            font=("Segoe UI", 16, "bold"),
            width=28, height=2,
            bg=bg_color, fg="white",
            activebackground=hover_color,
            bd=0, relief="flat",
            command=command
        )
        btn.pack(pady=15)
        btn.bind("<Enter>", lambda e: on_enter(e, btn, hover_color))
        btn.bind("<Leave>", lambda e: on_leave(e, btn, bg_color))
        return btn

    # ---------------- Buttons ----------------
    create_styled_button("🎬  YouTube Controller", "#1565C0", "#1E88E5", start_youtube)
    create_styled_button("📺  OTT Controller", "#8E24AA", "#AB47BC", start_ott)

    # ---------------- Footer ----------------
    footer = tk.Label(
        window,
        text="Made with ❤️ using OpenCV & Mediapipe",
        font=("Courier", 10),
        fg="#888888",
        bg="#0f0f1a"
    )
    footer.pack(side="bottom", pady=20)

    window.mainloop()

# --------------------- RUN GUI ---------------------
if __name__ == "__main__":
    launch_gui()
