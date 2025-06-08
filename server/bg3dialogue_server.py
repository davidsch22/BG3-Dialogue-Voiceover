import os
import torch
import torchaudio
from flask import Flask, request, make_response, send_file
from waitress import serve
from PIL import Image
from pytesseract import pytesseract
from tts import TTS

TTS_FILE = "tts.wav"
CLIENT_EXE = "BG3_Dialogue_Voiceover.zip"

app = Flask(__name__)

# Used to save the last text that was sent
previous_text = ""

# Providing the tesseract executable location to pytesseract library
pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize the TTS class
tts = TTS()


@app.route("/", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return {"error": "No image provided in the request"}, 400
    img_file = request.files["image"]

    frame = Image.open(img_file.stream)

    # Read the text from the frame
    text = pytesseract.image_to_string(
        frame, config='--oem 1 --psm 6')

    text = text.replace('’', '\'')
    text = text.replace('\n', ' ')
    text = text.replace('|', 'I')

    # If text is detected and it's not repeating the same one
    global previous_text
    if text != "" and text != previous_text and len(text) < 250:
        # Save text so it doesn't repeat while highlighted
        previous_text = text
        # Pass the text to the TTS engine
        audio = tts.infer(text)
        # Save tts audio as wav file
        torchaudio.save(TTS_FILE, torch.tensor(
            audio).unsqueeze(0), 24000)
        with open(TTS_FILE, 'rb') as f:
            audio_data = f.read()
            f.close()
        response = make_response(audio_data)
        response.headers["Content-Type"] = "audio/wav"
        response.headers["text"] = text
        response.status_code = 200
        return response

    response = make_response()
    response.headers["text"] = text
    response.status_code = 304
    return response


@app.route("/client", methods=["GET"])
def download_client():
    return send_file(CLIENT_EXE, as_attachment=True)


if __name__ == "__main__":
    try:
        serve(app, host='0.0.0.0', port=8080)
    finally:
        if os.path.isfile(TTS_FILE):
            os.remove(TTS_FILE)
