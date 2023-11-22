from enum import Enum
import random
import re
import time
import typing

random.seed(0)
import numpy as np
import numpy.typing as npt
np.random.seed(0)
import yaml
from rich import print

import prompt_toolkit
from prompt_toolkit.layout.containers import Window, Container
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.cursor_shapes import CursorShape
from Modules.ui import REFERENCE_PATH, choose_reference, depend_zip, play_audio, write_audio

import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from torch import Tensor
import torchaudio
import librosa

import nltk
from nltk.tokenize import word_tokenize
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert
from phonemizer.backend import EspeakBackend

from models import *
from utils import *
from text_utils import TextCleaner
from Configs.app import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
nltk.data.path = [NLTK_DATA_PATH]
text_cleaner = TextCleaner()
phonemizer = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')
model_config = yaml.safe_load(open(MODEL_PATH + CONFIG_FILENAME))
model_params = typing.cast(Munch, recursive_munch(model_config['model_params']))
model = None # initialized by initialize()
sampler = None # initialized by initialize()

def compute_style(path: str) -> Tensor:
    """
    Categorise a given audio file's speech style

    :param path: filepath to the audio file
    :returns: Tensor describing speech style
    """

    if model is None:
        raise RuntimeError("Expected model to be initialized before computing styles")

    wave, sr = librosa.load(path, sr=int(model_config['preprocess_params']['sr']))
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != model_config['preprocess_params']['sr']:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=int(model_config['preprocess_params']['sr']))

    # pre-process wave
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80,
        n_fft=2048,
        win_length=1200,
        hop_length=300
    )
    mean, std = -4, 4
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    mel_tensor = mel_tensor.to(DEVICE)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)

def initialize():
    # initialize model
    text_aligner = load_ASR_models(model_config.get('ASR_path', False), model_config.get('ASR_config', False))
    pitch_extractor = load_F0_models(model_config.get('F0_path', False))
    plbert = load_plbert(model_config.get('PLBERT_dir', False))
    global model
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(DEVICE) for key in model]

    # load params
    params_whole = torch.load(PRETRAINED_MODEL_PATH, map_location='cpu')
    params = params_whole['net']
    for key in model:
        if key in params:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                
                model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

    # initialize sampler
    global sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )

def inference(text, reference_style, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1) -> npt.NDArray:
    if model is None or sampler is None or model_params is None:
        raise RuntimeError("Expected model to be initialized before performing inference")

    phonemes = phonemizer.phonemize([text.strip()])
    phonemes = word_tokenize(phonemes[0])
    phonemes = ' '.join(phonemes)
    tokens = text_cleaner(phonemes)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(DEVICE)
        text_mask = torch.arange(typing.cast(int, input_lengths.max())).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1))
        text_mask = text_mask.to(DEVICE)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(DEVICE), 
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=reference_style, # reference from the same speaker as the embedding
            num_steps=diffusion_steps
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * reference_style[:, :128]
        s = beta * s + (1 - beta)  * reference_style[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(dim=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(DEVICE))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(DEVICE))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

class State(Enum):
    CHOOSE_REFERENCE = 1
    SYNTHESIZE = 2
    DONE = 3

# TODO: Use up/down arrow keys to cycle prompt history
# TODO: Add LJSpeech support

def main():
    depend_zip('LibriTTS pre-trained model', PRETRAINED_MODEL_PATH, MODEL_URL)
    depend_zip('Punkt tokenizer', PUNKT_PATH, PUNKT_URL, TOKENIZERS_PATH)

    initialize()
    print('[green]Successfully initialized model.[/green]')

    state = State.CHOOSE_REFERENCE
    style_cache: typing.Dict[str, Tensor] = {}
    reference_file = None
    filepath = ''
    while state != State.DONE:
        if state == State.CHOOSE_REFERENCE:
            reference_file = choose_reference()
            if reference_file is None:
                state = State.DONE
            else:
                print('Enter text for synthesis (Ctrl+C to go back)')
                state = State.SYNTHESIZE
        elif state == State.SYNTHESIZE:
            bindings = KeyBindings()

            @bindings.add('c-c')
            def _(event):
                " Exit when `c-c` is pressed. "
                event.app.exit()
            
            @bindings.add('c-p')
            def _(event):
                " Play the most recently synthesized sound when `c-p` is pressed. "
                play_audio(filepath)

            print(f"[deep_sky_blue1]({reference_file})[/deep_sky_blue1] ", end="")
            text = prompt_toolkit.prompt("> ", key_bindings=bindings, cursor=CursorShape.UNDERLINE)
            if text is None:
                state = State.CHOOSE_REFERENCE
            elif len(text) > model_config['model_params']['max_conv_dim']:
                print(f"[red]Sorry, StyleTTS2 is limited to {model_config['model_params']['max_conv_dim']} characters.[/red] Please shorten your input and try again.")
            else:
                reference_file = typing.cast(str, reference_file)
                start = time.time()
                if reference_file not in style_cache:
                    style_cache[reference_file] = compute_style(REFERENCE_PATH + reference_file)
                audio = inference(text, style_cache[reference_file])
                processing_time = time.time() - start
                duration = len(audio) / model_config['preprocess_params']['sr']
                filename = re.sub(r'\W', '', text)[:20]
                filepath = write_audio(audio, filename)
                print(f"[green]Synthesized {filepath}[/green]")
                print(f"Wrote {duration:2f}s of data in {processing_time:2f}s (Ctrl+P to play)")

if __name__ == "__main__":
    main()
