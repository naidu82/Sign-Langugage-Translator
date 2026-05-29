import os
from gtts import gTTS
from function import IMAGE_PATH_ASL, IMAGE_PATH_ISL
def generate_audio(lang, IMAGE_PATH):
    if not os.path.exists(IMAGE_PATH):
        print(f"{lang} Image folder not found")
        return
    actions = os.listdir(IMAGE_PATH)
    audio_dir = f"audio_{lang}"
    os.makedirs(audio_dir, exist_ok=True)
    for action in actions:
        path = f"{audio_dir}/{action}.mp3"
        if not os.path.exists(path):
            try:
                tts = gTTS(text=action, lang='en')  # Default English, can extend to multilingual
                tts.save(path)
                print(f"{lang} audio created for: {action}")
            except Exception as e:
                print(f"Failed to create audio for {lang}-{action}: {e}")
    print(f"{lang} Audio ready")
generate_audio("ASL", IMAGE_PATH_ASL)
generate_audio("ISL", IMAGE_PATH_ISL)