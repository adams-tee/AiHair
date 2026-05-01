import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# 1. GLOBAL SETUP (Outside the callback to save memory)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
hair_img = cv2.imread('pngwing.com.png', cv2.IMREAD_UNCHANGED)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def overlay_transparent(background, overlay, x, y, size=None):
    if size:
        overlay = cv2.resize(overlay, size)
    hb, wb = background.shape[:2]
    ho, wo = overlay.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + wo, wb), min(y + ho, hb)
    overlay_x1, overlay_y1 = x1 - x, y1 - y
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)
    if x1 >= x2 or y1 >= y2:
        return background
    visible_hair = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    if visible_hair.shape[2] == 4:
        alpha = visible_hair[:, :, 3] / 255.0
        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (1.0 - alpha) * background[y1:y2, x1:x2, c] + alpha * visible_hair[:, :, c]
    return background

# 2. THE CALLBACK
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) 
    h, w, _ = img.shape

    # Process AI
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # LIPS
            top_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
            bot_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
            top_pts = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in top_idx])
            bot_pts = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in bot_idx])

            cv2.polylines(img, [top_pts], False, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.polylines(img, [bot_pts], False, (0, 0, 0), 2, cv2.LINE_AA)

            # HAIR
            if hair_img is not None:
                forehead = face_landmarks.landmark[10]
                face_width = abs(face_landmarks.landmark[234].x - face_landmarks.landmark[454].x) * w
                hair_w = int(face_width * 3.0) 
                hair_h = int(hair_w * (hair_img.shape[0] / hair_img.shape[1]))
                x_pos = int(forehead.x * w - hair_w / 2)
                y_pos = int(forehead.y * h - hair_h / 3) 
                img = overlay_transparent(img, hair_img, x_pos, y_pos, (hair_w, hair_h))

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. UI
st.title("💄 AI Virtual Makeover")

# Use a specific key and disable session state tracking for this component
webrtc_streamer(
    key="makeup-app-v3",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)