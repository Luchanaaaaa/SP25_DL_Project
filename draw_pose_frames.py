# draw_pose_frames.py
import cv2
import mediapipe as mp
import os

# Input path (previously extracted frame images)
input_folder = 'frames'
output_folder = 'pose_condition_frames'
os.makedirs(output_folder, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Process each frame
for filename in sorted(os.listdir(input_folder)):
    if not filename.endswith('.png'):
        continue

    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    # Create a black canvas
    black = image.copy()
    black[:] = 0

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            black,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, black)

print(f" Generated {len(os.listdir(output_folder))} pose condition images at: {output_folder}/")
