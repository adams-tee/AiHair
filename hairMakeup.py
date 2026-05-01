import cv2
import mediapipe as mp
import numpy as np

# 1. SETUP
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

# Load hair image
hair_img = cv2.imread('pngwing.com.png', cv2.IMREAD_UNCHANGED)

if hair_img is None:
    print("❌ Error: Could not find 'pngwing.com.png'. Check the folder!")
else:
    print("✅ Hair image loaded successfully!")

def overlay_transparent(background, overlay, x, y, size=None):
    if size:
        overlay = cv2.resize(overlay, size)
    
    hb, wb = background.shape[:2]
    ho, wo = overlay.shape[:2]
    
    # Clipping logic to keep hair on screen
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + wo, wb), min(y + ho, hb)
    overlay_x1, overlay_y1 = x1 - x, y1 - y
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return background

    visible_hair = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    
    # Handle transparency if the image has an alpha channel
    if visible_hair.shape[2] == 4:
        alpha = visible_hair[:, :, 3] / 255.0
        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (1.0 - alpha) * background[y1:y2, x1:x2, c] + \
                                          alpha * visible_hair[:, :, c]
    else:
        # If no alpha, just paste it (for debugging)
        background[y1:y2, x1:x2] = visible_hair[:, :, :3]
        
    return background

# 2. MAIN LOOP
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Flip frame for a mirror effect (easier to apply makeup)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # --- LIPS LOGIC ---
            top_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
            bot_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

            top_pts = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in top_idx])
            bot_pts = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in bot_idx])

            # Draw Liner & Gloss
            cv2.polylines(frame, [top_pts], False, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.polylines(frame, [bot_pts], False, (0, 0, 0), 2, cv2.LINE_AA)
            
            full_lip_pts = np.vstack((top_pts, bot_pts[::-1]))
            mask = np.zeros_like(frame)
            cv2.fillPoly(mask, [full_lip_pts], (25, 255, 25))
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            frame = cv2.addWeighted(frame, 1.0, mask, 0.2, 0)

            # --- HAIR LOGIC ---
            if hair_img is not None:
                forehead = face_landmarks.landmark[10]
                
                # Dynamic Scaling
                face_width = abs(face_landmarks.landmark[234].x - face_landmarks.landmark[454].x) * w
                hair_w = int(face_width * 3.0) 
                hair_h = int(hair_w * (hair_img.shape[0] / hair_img.shape[1]))
                
                x_pos = int(forehead.x * w - hair_w / 2)
                # Lowered the Y just to make sure it's visible on your face first
                y_pos = int(forehead.y * h - hair_h / 3) 

                frame = overlay_transparent(frame, hair_img, x_pos, y_pos, (hair_w, hair_h))

    cv2.imshow('Makeup & Hair App', frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()