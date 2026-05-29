from function import *
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
def train_model(lang, DATA_PATH, actions):
    if len(actions) == 0:
        print(f"No actions found for {lang}")
        return
    print(f"\nTraining {lang} model...")
    print("Classes:", actions)
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                if not os.path.exists(file_path):
                    print(f"⚠ Missing file: {file_path}")
                    continue
                res = np.load(file_path)
                if res.shape[0] != 126:
                    print(f"⚠ Wrong shape in {file_path}: {res.shape}")
                    continue
                window.append(res)
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
    if len(sequences) == 0:
        print(f"No valid sequences for {lang}")
        return
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    print("Data Loaded")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu',
                   input_shape=(sequence_length, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Training started...")
    model.fit(X_train, y_train, epochs=50)
    with open(f"model_{lang}.json", "w") as f:
        f.write(model.to_json())
    model.save_weights(f"model_{lang}.h5")
    print(f"{lang} model trained and saved!")
train_model("ASL", DATA_PATH_ASL, actions_asl)
train_model("ISL", DATA_PATH_ISL, actions_isl)