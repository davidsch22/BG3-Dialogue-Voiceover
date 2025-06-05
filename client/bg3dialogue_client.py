import os
import keyboard
import numpy as np
import cv2 as cv
import dxcam
from time import time
from pygame import mixer
from PIL import Image
from ui import UI
from vision import Vision
from filters import HsvFilter

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

        # Initialize the trackbar window
        # self.vision.init_control_gui()

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

                # Filter everything out except the highlighted option
                hsv_filter = HsvFilter(20, 22, 145, 33, 137, 255, 0, 0, 0, 0)
                filtered_cont = self.vision.crop_text(
                    screenshot, hsv_filter)

                # Cropping the text block for giving input to OCR
                if not filtered_cont is None:
                    text_img = screenshot.copy()
                    mask = np.zeros_like(screenshot)
                    cv.drawContours(mask, [filtered_cont],
                                    0, (255, 255, 255), -1)

                    text_img = cv.bitwise_and(screenshot, mask)

                    (x, y, w, h) = cv.boundingRect(filtered_cont)
                    text_img = text_img[y:y+h, x:x+w]

                    text_img = cv.bilateralFilter(text_img, 3, 50, 50)

                    hsv_filter = HsvFilter(
                        19, 0, 164, 32, 255, 255, 0, 0, 0, 0)
                    text_img = self.vision.apply_hsv_filter(
                        text_img, hsv_filter)

                    text_img = self.vision.filter_text(text_img)

                    # cv.imshow('Processed', text_img)

                    # Convert the image from nparray to PIL
                    frame = Image.fromarray(text_img)

                    # TODO: Send processed image to server
                    text = ""
                    audio = 0

                    # If response has audio and audio isn't already playing
                    if not mixer.music.get_busy():  # TODO: and response has audio
                        self.ui.set_dialogue_text(text)
                        mixer.music.unload()
                        if os.path.isfile(TTS_FILE):
                            os.remove(TTS_FILE)
                        # TODO: Save returned audio file
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
        if os.path.isfile(TTS_FILE):
            os.remove(TTS_FILE)


if __name__ == "__main__":
    main()
