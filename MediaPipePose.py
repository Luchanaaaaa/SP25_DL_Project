import cv2
import mediapipe as mp
import numpy as np

# === Path Configuration ===
video_path = 'dance_demo.mp4'
dress_path = 'dress.png'
output_path = 'output_dress_with_arm_occlusion_fixed.mp4'

# === Initialize MediaPipe Pose and Segmentation ===
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

pose = mp_pose.Pose(static_image_mode=False)
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# === Video Read & Output Config ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Load Dress Image (with Alpha Channel) ===
dress_img = cv2.imread(dress_path, cv2.IMREAD_UNCHANGED)

# === Overlay with Transparency Support ===
def overlay_alpha(img, overlay, pos):
    x, y = pos
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return img
    overlay_rgb = overlay[:, :, :3]
    alpha = overlay[:, :, 3:] / 255.0
    img[y:y+h, x:x+w] = (1 - alpha) * img[y:y+h, x:x+w] + alpha * overlay_rgb
    return img

# === Arm Extraction: Use Pose Keypoints to Get Bounding Box, Then Use Segmentation Mask to Extract Arm Layer ===
def extract_arm_from_mask(frame, seg_mask, pt1, pt2, scale=1.5):
    # Center and estimated width/height
    cx = int((pt1[0] + pt2[0]) / 2)
    cy = int((pt1[1] + pt2[1]) / 2)
    w = int(abs(pt1[0] - pt2[0]) * scale)
    h = int(abs(pt1[1] - pt2[1]) * scale)
    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(width, cx + w // 2)
    y2 = min(height, cy + h // 2)

    # Extract region from mask
    seg_crop = seg_mask[y1:y2, x1:x2]
    frame_crop = frame[y1:y2, x1:x2]

    # Mask the arm using segmentation
    arm_mask = (seg_crop * 255).astype(np.uint8)
    arm_mask_3c = cv2.merge([arm_mask]*3)
    arm_img = cv2.bitwise_and(frame_crop, arm_mask_3c)

    return arm_img, (x1, y1), arm_mask

# === Main Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    seg_result = segmentor.process(image_rgb)
    seg_mask = (seg_result.segmentation_mask > 0.5).astype(np.uint8)

    frame_with_dress = frame.copy()

    if results.pose_landmarks:
        try:
            lm = results.pose_landmarks.landmark
            def to_px(idx):
                pt = lm[idx]
                return int(pt.x * width), int(pt.y * height)

            # Shoulder / Hip / Wrist coordinates
            ls, rs = to_px(mp_pose.PoseLandmark.LEFT_SHOULDER), to_px(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            lh, rh = to_px(mp_pose.PoseLandmark.LEFT_HIP), to_px(mp_pose.PoseLandmark.RIGHT_HIP)
            lw, rw = to_px(mp_pose.PoseLandmark.LEFT_WRIST), to_px(mp_pose.PoseLandmark.RIGHT_WRIST)

            #  Dress overlay
            center_x = int((ls[0] + rs[0] + lh[0] + rh[0]) / 4)
            center_y = int((ls[1] + rs[1] + lh[1] + rh[1]) / 4)
            dress_w = int(np.linalg.norm(np.array(rs) - np.array(ls)) * 1.5)
            dress_h = int(np.linalg.norm(np.array(rh) - np.array(ls)) * 2.0)
            dress_resized = cv2.resize(dress_img, (dress_w, dress_h), interpolation=cv2.INTER_AREA)
            top_left = (center_x - dress_w // 2, center_y - dress_h // 3)
            frame_with_dress = overlay_alpha(frame_with_dress, dress_resized, top_left)

            # Extract left and right arm regions
            left_arm_img, left_pos, _ = extract_arm_from_mask(frame, seg_mask, ls, lw)
            right_arm_img, right_pos, _ = extract_arm_from_mask(frame, seg_mask, rs, rw)

            # Paste arms back (on top of the dress)
            lx, ly = left_pos
            rx, ry = right_pos
            try:
                h1, w1 = left_arm_img.shape[:2]
                frame_with_dress[ly:ly+h1, lx:lx+w1] = np.where(left_arm_img > 0, left_arm_img, frame_with_dress[ly:ly+h1, lx:lx+w1])
                h2, w2 = right_arm_img.shape[:2]
                frame_with_dress[ry:ry+h2, rx:rx+w2] = np.where(right_arm_img > 0, right_arm_img, frame_with_dress[ry:ry+h2, rx:rx+w2])
            except:
                pass

        except Exception as e:
            print(" Pose detection failed. Skipping frame.")
            pass

    # Display and Save
    cv2.imshow('Virtual Try-on with Arm Occlusion', frame_with_dress)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(frame_with_dress)

# === Cleanup ===
cap.release()
out.release()
pose.close()
segmentor.close()
cv2.destroyAllWindows()

print(" Final output with proper arm occlusion:", output_path)
