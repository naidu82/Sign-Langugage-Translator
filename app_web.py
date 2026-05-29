from flask import Flask, render_template, Response, request, jsonify, send_file, session, redirect, url_for
from functools import wraps
import cv2, os, numpy as np, time, sys, subprocess, shutil
from keras.models import model_from_json
from function import *
from gtts import gTTS
app = Flask(__name__)
app.secret_key = 'sign_language_secret_key_2024'
VALID_USERS = {
    'admin': 'password123',
    'Naidu': '53822358',
    'Uday': '12345678',
    'Peter': '12345678',
    'Thanvi': '12345678'
}
SEQUENCE_LENGTH = 5
max_images = 300
os.makedirs("audio_ASL", exist_ok=True)
os.makedirs("audio_ISL", exist_ok=True)
os.makedirs("Image_ASL", exist_ok=True)
os.makedirs("Image_ISL", exist_ok=True)
model = None
actions = np.array([])
current_lang = "ASL"
def load_model(lang="ASL"):
    global model, actions, current_lang
    current_lang = lang.upper()
    try:
        model_file = f"model_{current_lang}.json"
        weights_file = f"model_{current_lang}.h5"
        if not os.path.exists(model_file) or not os.path.exists(weights_file):
            print(f"Model files not found for {current_lang}")
            model = None
            actions = np.array([])
            return False
        with open(model_file, "r") as f:
            model = model_from_json(f.read())
        model.load_weights(weights_file)
        data_path = f"MP_Data_{current_lang}"
        if os.path.exists(data_path):
            actions = np.array(sorted(os.listdir(data_path)))
        else:
            actions = np.array([])
        print(f"Loaded {current_lang} model with actions: {actions}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        return False
load_model("ASL")
sequence = []
current_word = ""
predicting = False
streaming = True
capture_label = ""
capture_count = 0
frame_count = 0
cap = cv2.VideoCapture(0)
mp_hands, mp_drawing = get_mediapipe()
def generate_frames():
    global sequence, current_word, predicting, capture_label, capture_count, frame_count
    with mp_hands.Hands() as hands:
        while True:
            if not cap.isOpened():
                cap.open(0)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (640,480))
            if current_lang == "ASL":
                roi_width, roi_height = 300, 360
            else:  # ISL
                roi_width, roi_height = 600, 440
            center_x = (frame.shape[1] - roi_width) // 2
            x1, y1 = center_x, 40
            x2, y2 = x1 + roi_width, y1 + roi_height
            crop = frame[y1:y2, x1:x2]
            _, results = mediapipe_detection(crop, hands)
            frame_count += 1
            if not predicting:
                current_word = "-"
                sequence = []
            if predicting and model is not None:
                keypoints = extract_keypoints(results)
                if keypoints is not None:
                    sequence.append(keypoints)
                    sequence = sequence[-SEQUENCE_LENGTH:]
                if len(sequence) == SEQUENCE_LENGTH:
                    try:
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        pred = np.argmax(res)
                        if pred < len(actions):
                            current_word = actions[pred]
                    except:
                        sequence = []
            else:
                sequence = []
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            if capture_label:
                cv2.putText(frame,f"{capture_label}:{capture_count}/{max_images}",
                            (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,f"{'ON' if predicting else 'OFF'} | {current_word} | {current_lang}",
                            (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        if 'username' in session:
            return redirect(url_for('index'))
        return render_template('login.html')
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        if username in VALID_USERS and VALID_USERS[username] == password:
            session['username'] = username
            return jsonify({"status": "success", "message": "Login successful"})
        else:
            return jsonify({
                "status": "error", 
                "message": "Invalid username or password"
            }), 401
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or len(username) < 3:
        return jsonify({
            "status": "error",
            "message": "Username must be at least 3 characters"
        }), 400
    if not password or len(password) < 6:
        return jsonify({
            "status": "error",
            "message": "Password must be at least 6 characters"
        }), 400
    if username in VALID_USERS:
        return jsonify({
            "status": "error",
            "message": "Username already exists"
        }), 409
    VALID_USERS[username] = password
    return jsonify({
        "status": "success",
        "message": "Registration successful"
    }), 201
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
@app.route('/')
@login_required
def index():
    username = session.get('username', 'User')
    return render_template('index.html', username=username)
@app.route('/video')
@login_required
def video():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/get_word')
@login_required
def get_word():
    return jsonify({"word": current_word})
@app.route('/get_language')
@login_required
def get_language():
    return jsonify({"language": current_lang})
@app.route('/switch_language/<lang>')
@login_required
def switch_language(lang):
    global sequence, current_word, predicting
    predicting = False
    sequence = []
    current_word = "-"
    if load_model(lang):
        return jsonify({"status": "success", "language": current_lang})
    else:
        return jsonify({"status": "error", "message": f"Failed to load {lang} model"})
@app.route('/reset')
@login_required
def reset():
    global predicting, sequence, current_word
    predicting = False
    sequence = []
    current_word = "-"
    return jsonify({"status": "reset"})
@app.route('/toggle_prediction')
@login_required
def toggle_prediction():
    global predicting, sequence
    predicting = not predicting
    sequence = []
    return jsonify({"status": predicting})
@app.route('/start_capture', methods=['POST'])
@login_required
def start_capture():
    global capture_label, capture_count
    capture_label = request.form['label'].upper()
    capture_count = 0
    image_dir = f"Image_{current_lang}/{capture_label}/"
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)

    os.makedirs(image_dir, exist_ok=True)
    return jsonify({"status": "created"})
@app.route('/capture_frame', methods=['POST'])
@login_required
def capture_frame():
    global capture_count, capture_label
    ret, frame = cap.read()
    if not ret:
        return jsonify({"status":"error"})
    frame = cv2.resize(frame, (640, 480))
    if current_lang == "ASL":
        roi_width, roi_height = 300, 360
    else:
        roi_width, roi_height = 600, 440
    center_x = (frame.shape[1] - roi_width) // 2
    x1, y1 = center_x, 40
    x2, y2 = x1 + roi_width, y1 + roi_height
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (224, 224))
    cv2.imwrite(f"Image_{current_lang}/{capture_label}/{capture_count}.png", roi)
    capture_count += 1
    if capture_count >= max_images:
        capture_label = ""
        return jsonify({"status":"done"})
    return jsonify({"status":"capturing","count":capture_count})
@app.route('/train')
@login_required
def train():
    global sequence, current_word, predicting
    predicting = False
    sequence = []
    current_word = ""
    subprocess.call([sys.executable,"data.py"])
    subprocess.call([sys.executable,"trainmodel.py"])
    subprocess.call([sys.executable,"audiocreator.py"])
    load_model(current_lang)
    return jsonify({"status":"trained"})
@app.route('/speak/<word>')
@login_required
def speak(word):
    audio_dir = f"audio_{current_lang}"
    path = f"{audio_dir}/{word}.mp3"
    if not os.path.exists(path):
        try:
            gTTS(word).save(path)
        except:
            pass
    if os.path.exists(path):
        return send_file(path, mimetype='audio/mpeg')
    return jsonify({"error":"Audio not found"})
@app.route('/shutdown')
def shutdown():
    os._exit(0)
if __name__ == "__main__":
    app.run(debug=True)