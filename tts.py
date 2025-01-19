import os
import time
import torch
import torchaudio
from TTS.api import TTS as cTTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class TTS:
    XTTS_FOLDER = "tts/tts_models--multilingual--multi-dataset--xtts_v2/"

    model = None
    gpt_cond_latent = None
    speaker_embedding = None

    def __init__(self):
        print("Loading model...")
        os.environ.setdefault(
            "TTS_HOME", os.path.dirname(os.path.abspath(__file__)))
        cTTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        config = XttsConfig()
        config.load_json(
            self.XTTS_FOLDER + "config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config, checkpoint_dir=self.XTTS_FOLDER, use_deepspeed=True)
        self.model.cuda()

        print("Computing speaker latents...")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=[
            "bg3_tav5_voice.wav"])

    def infer(self, text, save_file):
        print("Inference...")
        # t0 = time.time()
        # chunks = self.model.inference_stream(
        out = self.model.inference(
            text,
            "en",
            self.gpt_cond_latent,
            self.speaker_embedding
        )

        # wav_chuncks = []
        # for i, chunk in enumerate(chunks):
        #     if i == 0:
        #         print(f"Time to first chunck: {time.time() - t0}")
        #     print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        #     wav_chuncks.append(chunk)
        # wav = torch.cat(wav_chuncks, dim=0)
        torchaudio.save(save_file, torch.tensor(
            out["wav"]).unsqueeze(0), 24000)
        # wav.squeeze().unsqueeze(0).cpu(), 24000)
