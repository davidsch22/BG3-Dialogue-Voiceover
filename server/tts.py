import os
from TTS.api import TTS as cTTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class TTS:
    XTTS_FOLDER = "/tts/tts_models--multilingual--multi-dataset--xtts_v2/"

    model = None
    gpt_cond_latent = None
    speaker_embedding = None

    def __init__(self):
        print("Loading model...")
        file_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ.setdefault("TTS_HOME", file_dir)
        cTTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        config = XttsConfig()
        config.load_json(
            file_dir + self.XTTS_FOLDER + "config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config, checkpoint_dir=file_dir + self.XTTS_FOLDER, use_deepspeed=True)
        self.model.cuda()

        print("Computing speaker latents...")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=[
            "bg3_tav5_voice.wav"])

    def infer(self, text):
        print("Inferring...")
        out = self.model.inference(
            text,
            "en",
            self.gpt_cond_latent,
            self.speaker_embedding,
            speed=1.25
        )
        return out["wav"]
