import os
import cv2
import numpy as np
from function import *
datasets = [
    ("ASL", IMAGE_PATH_ASL, DATA_PATH_ASL, actions_asl),
    ("ISL", IMAGE_PATH_ISL, DATA_PATH_ISL, actions_isl)
]
valid_exts = (".jpg", ".jpeg", ".png")
for lang, IMAGE_PATH, DATA_PATH, actions in datasets:
    if not os.path.exists(IMAGE_PATH):
        print(f"{lang} Image folder not found")
        continue
    actions = np.array(sorted(os.listdir(IMAGE_PATH)))
    print(f"{lang} Actions found:", actions)
    if len(actions) == 0:
        print(f"No gesture folders inside {IMAGE_PATH}")
        continue
    for action in actions:
        for sequence in range(no_sequences):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
    mp_hands, mp_drawing = get_mediapipe()
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hands:
        for action in actions:
            image_folder = os.path.join(IMAGE_PATH, action)
            images = sorted([
                f for f in os.listdir(image_folder)
                if f.lower().endswith(valid_exts)
            ])
            print(f"Processing {lang}-{action}: {len(images)} images")
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    idx = sequence * sequence_length + frame_num
                    if idx >= len(images):
                        continue
                    frame_path = os.path.join(image_folder, images[idx])
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print("Failed to read:", frame_path)
                        continue
                    image, results = mediapipe_detection(frame, hands)
                    keypoints = extract_keypoints(results)
                    np.save(
                        os.path.join(DATA_PATH, action, str(sequence), str(frame_num)),
                        keypoints
                    )
print("MP_Data_ASL and MP_Data_ISL created successfully")