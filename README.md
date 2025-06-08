# BG3 Dialogue Voiceover

A program that reads Baldur's Gate 3 dialogue options before they're selected for people with reading difficulties.

To offload the heavy lifting of the image-to-text and text-to-speech from the user's machine (and to make the client files smaller and make compiling it easier), the program has been split into a client executable that does the image filtering and audio playback and an API server that takes the filtered image as input and returns the TTS audio.

The server can be hosted on another machine so that the player's PC and game performance isn't affected by the image-to-text and TTS processes. It would just require some extra network setup.

## Server Setup

If you would like to host the server yourself, there are a few things you'll need to install to make this work:

- [Python 3.12](https://www.python.org/downloads/) (Make sure you download 3.12)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract/releases) (Install to its default folder)
- [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) (Optional, Used for GPU acceleration if you have an NVIDIA GPU. 12.4 is necessary to work for Windows with this version of Python and PyTorch)

After CUDA Toolkit is installed, you'll need to manually set the CUDA_HOME environment variable to the path where it was installed (most likely `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`)

If you have an NVIDIA GPU and would like to speed up the process, you'll need the CUDA Toolkit linked above as well as a special version of PyTorch to use GPU acceleration. With the desired Python environment active, run this command:

`pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124`

## Running

If you're just using the client, run the `BG3DialogueVoiceover.exe` file.

If you're also running the server, run the following command in a terminal from the server folder:

`python bg3dialogue_server.py`

When you run the server the first time, it will take a while to download the XTTS model.
