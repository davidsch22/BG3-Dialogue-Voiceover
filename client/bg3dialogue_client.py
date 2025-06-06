import os
import keyboard
import numpy as np
import cv2 as cv
import dxcam
import requests
from time import time
from pygame import mixer
from PIL import Image
from ui import UI
from vision import Vision
from filters import HsvFilter

IMAGE_FILE = "text.png"
TTS_FILE = "tts.wav"


class BG3DialogueVoiceoverClient:

    PAUSE_KEY = "F6"

    pause = True
    was_pressed = False

    def __init__(self):
        # Initialize and start the screen capture thread
        self.camera = dxcam.create(output_color="BGR")
        self.camera.start()

        self.ui = UI()

        # Initialize the Vision class
        self.vision = Vision()

        # Initialize the audio player
        mixer.init()

        self.loop_time = time()

    def main_loop(self):
        while True:
            if self.ui.is_closed():
                break

            if keyboard.is_pressed(self.PAUSE_KEY) and not self.was_pressed:
                self.was_pressed = True
                self.pause = not self.pause
                if self.pause:
                    self.ui.set_pause_status("Paused")
                    self.ui.set_fps(0)
                else:
                    self.ui.set_pause_status("Running")
            elif not keyboard.is_pressed(self.PAUSE_KEY) and self.was_pressed:
                self.was_pressed = False

            if not self.pause:
                # Get an updated image of the game
                screenshot = self.camera.get_latest_frame()

                # Crop screenshot to see just the dialogue options area
                if np.size(screenshot, axis=0) == 1080:
                    screenshot = screenshot[795:1035, 480:1440]
                elif np.size(screenshot, axis=0) == 1440:
                    screenshot = screenshot[1060:1380, 640:1920]
                else:
                    print("ERROR: Screen resolution not compatible")
                    break

                # Filter and crop everything out except the highlighted option
                hsv_filter = HsvFilter(19, 122, 77, 20, 140, 218, 0, 0, 0, 0)
                cropped = self.vision.crop_text(
                    screenshot, hsv_filter)

                if not cropped is None:
                    # Filter the cropped text block for input to OCR
                    hsv_filter = HsvFilter(
                        18, 0, 100, 21, 255, 255, 0, 0, 0, 0)
                    text_img = self.vision.filter_text(cropped, hsv_filter)

                    # cv.imshow('Processed', text_img)

                    # If audio isn't already playing
                    if not mixer.music.get_busy():
                        # Convert the image from nparray to PIL
                        frame = Image.fromarray(text_img)
                        if os.path.isfile(IMAGE_FILE):
                            os.remove(IMAGE_FILE)
                        frame.save(IMAGE_FILE)

                        # Send image to TTS API
                        IMAGE_TO_AUDIO_URL = "http://75.184.112.22:8080/"
                        with open(IMAGE_FILE, 'rb') as f:
                            files = {"image": f}
                            response = requests.post(
                                IMAGE_TO_AUDIO_URL, files=files)
                            f.close()

                        # If response has audio
                        if response.status_code == 200:
                            text = response.headers["text"]
                            self.ui.set_dialogue_text(text)
                            mixer.music.unload()
                            # Save returned audio file
                            if os.path.isfile(TTS_FILE):
                                os.remove(TTS_FILE)
                            with open(TTS_FILE, 'wb') as f:
                                f.write(response.content)
                                f.close()
                            # Play TTS audio
                            mixer.music.load(TTS_FILE)
                            mixer.music.play()

                # Debug the loop rate
                self.fps = 1 / (time() - self.loop_time)
                self.ui.set_fps(self.fps)
                self.loop_time = time()
        self.ui.close_window()


def main():
    app = BG3DialogueVoiceoverClient()
    try:
        app.main_loop()
    finally:
        cv.destroyAllWindows()
        app.camera.stop()
        mixer.music.stop()
        mixer.music.unload()
        if os.path.isfile(IMAGE_FILE):
            os.remove(IMAGE_FILE)
        if os.path.isfile(TTS_FILE):
            os.remove(TTS_FILE)


if __name__ == "__main__":
    main()
