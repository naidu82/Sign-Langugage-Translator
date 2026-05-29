import cv2
import os
import shutil
lang = input("Enter language (ASL/ISL): ").upper()
if lang not in ["ASL", "ISL"]:
    print("Invalid language. Please enter ASL or ISL.")
    exit()
label = input("Enter sign name: ").upper()
directory = f"Image_{lang}/{label}/"
if os.path.exists(directory):
    print(f"Directory exists. Deleting old {lang}-{label} data...")
    shutil.rmtree(directory)
os.makedirs(directory, exist_ok=True)
cap = cv2.VideoCapture(0)
count = 0
max_images = 300
if lang == "ASL":
    roi_width, roi_height = 300, 360
else:  # ISL
    roi_width, roi_height = 600, 440
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    center_x = (frame.shape[1] - roi_width) // 2
    x1, y1 = center_x, 40
    x2, y2 = x1 + roi_width, y1 + roi_height
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"{lang}-{label}: {count}/{max_images}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and count < max_images:
        roi_resized = cv2.resize(roi, (224, 224))
        cv2.imwrite(f"{directory}/{count}.png", roi_resized)
        count += 1
    if count >= max_images:
        print(f"Capture completed for {lang}-{label}")
        break
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()