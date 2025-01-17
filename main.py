import os
from time import time
import cv2 as cv
import dxcam
from PIL import Image
from pytesseract import pytesseract
from gtts import gTTS
from pygame import mixer
from vision import Vision
from hsvfilter import HsvFilter

# Initialize and start the screen capture thread
camera = dxcam.create(output_color="RGB")
camera.start()

# Initialize the Vision class
vision = Vision()

# Initialize the trackbar window
# vision.init_control_gui()

# HSV filter
hsv_filter = HsvFilter(17, 29, 178, 32, 133, 255, 0, 0, 0, 0)

# Providing the tesseract executable location to pytesseract library
pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize the audio player
mixer.init()

TTS_FILE = "tts.wav"

# Saving the last text that was read from the screen
previous_text = ""

loop_time = time()

while True:
    # Get an updated image of the game
    screenshot = camera.get_latest_frame()

    # Crop screenshot to see just the dialogue options area
    screenshot = screenshot[1000:1380, 660:1900]

    # Filter everything out except the highlighted option
    filtered = vision.apply_hsv_filter(screenshot, hsv_filter)

    # Display the filtered frame
    cv.imshow('Processed', filtered)

    # Convert the image from nparray to PIL
    frame = Image.fromarray(filtered)

    # Read the text from the frame
    text = pytesseract.image_to_string(frame)

    # If text is detected and it's not repeating the same one
    if text != "" and text != previous_text:
        # Interrupt previous audio if already playing
        mixer.music.stop()
        mixer.music.unload()
        if os.path.isfile(TTS_FILE):
            os.remove(TTS_FILE)
        # Save text so it doesn't repeat while highlighted
        previous_text = text
        # Passing the text and language to the TTS engine
        text_to_speech = gTTS(
            text=text, lang="en-GB", tld="co.uk")
        # Save TTS audio and play it
        text_to_speech.save(TTS_FILE)
        mixer.music.load(TTS_FILE)
        mixer.music.play()

    # Debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # Press 'q' with the output window focused to exit.
    # Waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        camera.stop()
        mixer.music.stop()
        mixer.music.unload()
        if os.path.isfile(TTS_FILE):
            os.remove(TTS_FILE)
        break

print('Done')
