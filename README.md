# BG3 Dialogue Voiceover

A program that reads Baldur's Gate 3 dialogue options before you select them for people with reading difficulties.

## Setup

There are a few things you'll need to install to make this work:

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract/releases) (install to its default folder)
- [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-1-download-archive) (12.1 is necessary to work for Windows)

After CUDA Toolkit is installed, you'll need to manually set the CUDA_HOME environment variable to the path where it was installed (most likely `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`)

## Running

When you run it the first time, it will take a while to download the XTTS model.

While focused on the program window, press `Q` to stop the program.
