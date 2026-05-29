import cv2
import numpy as np
import os
import mediapipe as mp

DATA_PATH_ASL = os.path.join("MP_Data_ASL")
DATA_PATH_ISL = os.path.join("MP_Data_ISL")
IMAGE_PATH_ASL = os.path.join("Image_ASL")
IMAGE_PATH_ISL = os.path.join("Image_ISL")

if os.path.exists(DATA_PATH_ASL):
    actions_asl = np.array(sorted(os.listdir(DATA_PATH_ASL)))
else:
    actions_asl = np.array([])
if os.path.exists(DATA_PATH_ISL):
    actions_isl = np.array(sorted(os.listdir(DATA_PATH_ISL)))
else:
    actions_isl = np.array([])
no_sequences = 30
sequence_length = 5
def get_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    return mp_hands, mp_drawing
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
def draw_styled_landmarks(image, results):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        # Collect up to 2 hands
        for hand in results.multi_hand_landmarks[:2]:
            keypoints.extend([[res.x, res.y, res.z] for res in hand.landmark])
        # If only one hand detected, pad with zeros for the second hand
        if len(results.multi_hand_landmarks) == 1:
            keypoints.extend([[0, 0, 0]] * 21)
    # If no hands detected, return all zeros
    if len(keypoints) == 0:
        keypoints = [[0, 0, 0]] * 42
    return np.array(keypoints).flatten()