import random
import re
import typing

random.seed(0)
import numpy as np
np.random.seed(0)
import time
import yaml
from munch import Munch
import os
import typer
import pytermgui as ptg
from rich import print
import gdown
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import asyncio

import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from torch import Tensor, nn
import torch.nn.functional as F
import torchaudio
import librosa

import nltk
from nltk.tokenize import word_tokenize
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert

from models import *
from utils import *
from text_utils import TextCleaner

from phonemizer.backend import EspeakBackend
from scipy.io import wavfile

SAMPLE_RATE = 24000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG_FILENAME = 'config.yml'
EPOCH_FILENAME = 'epochs_2nd_00020.pth'
MODEL_URL = 'https://drive.google.com/uc?id=1jK_VV3TnGM9dkrIMsdQ_upov8FrIymr7'
MODEL_PATH = 'Models/LibriTTS/'
NLTK_DATA_PATH = 'Data/nltk/'
# TODO: fix hardcoded path
nltk.data.path = ["/Users/andrew/Documents/StyleTTS2/" + NLTK_DATA_PATH]
TOKENIZERS_PATH = NLTK_DATA_PATH + 'tokenizers/'
PUNKT_PATH = TOKENIZERS_PATH + 'punkt/PY3/english.pickle'
PUNKT_URL = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip'
REFERENCE_PATH = 'Data/reference_audio/'
OUTPUT_PATH = 'Output/'
text_cleaner = TextCleaner()
phonemizer = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')
config = None # initialized by initialize()
model_params = None # initialized by initialize()
model = None # initialized by initialize()
sampler = None # initialized by initialize()

def compute_style(path: str) -> Tensor:
    """
    Categorise a given audio file's speech style

    :param path: filepath to the audio file
    :returns: Tensor describing speech style
    """
    wave, sr = librosa.load(path, sr=SAMPLE_RATE)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, sr, SAMPLE_RATE)

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

    # load config
    global config
    config = yaml.safe_load(open(MODEL_PATH + CONFIG_FILENAME))
    global model_params
    model_params = recursive_munch(config['model_params'])

    # initialize model
    text_aligner = load_ASR_models(config.get('ASR_path', False), config.get('ASR_config', False))
    pitch_extractor = load_F0_models(config.get('F0_path', False))
    plbert = load_plbert(config.get('PLBERT_dir', False))
    global model
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(DEVICE) for key in model]

    # load params
    params_whole = torch.load(MODEL_PATH + EPOCH_FILENAME, map_location='cpu')
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

def inference(text, reference_style, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    phonemes = phonemizer.phonemize([text.strip()])
    phonemes = word_tokenize(phonemes[0])
    phonemes = ' '.join(phonemes)
    tokens = text_cleaner(phonemes)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(DEVICE)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
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

        duration = torch.sigmoid(duration).sum(axis=-1)
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

def synthesize(reference_file: str, text: str, filename: str = ""):

    # start = time.time()
    audio = inference(text, compute_style(REFERENCE_PATH + reference_file))
    # rtf = (time.time() - start) / (len(wav) / SAMPLE_RATE)
    # print(f"RTF = {rtf:5f}")
    # Convert to (little-endian) 16 bit integers.
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if not filename:
        filename = re.sub(r'\W', '', text)[:20] + '.wav'
    wavfile.write(f"{OUTPUT_PATH}{filename}", SAMPLE_RATE, audio)

def depend_zip(name: str, check_path: str, url: str, extract_path: str | None = None):
    if not os.path.isfile(check_path):
        download_model = typer.confirm(f"ℹ️  It appears you are missing the {name}. Would you like to download it now?")
        manual_instructions = f"For manual installation, download the {name} from {url}, and extract it into {extract_path if extract_path else 'the project root'}."
        if not download_model:
            print(manual_instructions)
            raise typer.Exit()
        
        try:
            if url.startswith('https://drive.google.com'):
                gdown.cached_download(url=url, path=extract_path, quiet=False, postprocess=gdown.extractall)
            else:
                with urlopen(url) as res:
                    with ZipFile(BytesIO(res.read())) as zipfile:
                        zipfile.extractall(extract_path)
        except Exception as e:
            print(f"[red]There was a problem downloading the {name}.[/red]")
            print(e)
            print(manual_instructions)
            raise typer.Abort()
        if not os.path.isfile(check_path):
            print("[red]Extracted files did not have the expected file structure![/red]")
            raise typer.Abort()
        
        print(f"[green]{name} successfully downloaded and extracted.[/green]")

# def choose_reference(manager: ptg.WindowManager) -> asyncio.Future[str]:
#     os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
#     filenames = next(os.walk(REFERENCE_PATH), (None, None, []))[2]
#     wavfiles = [f for f in filenames if f.endswith('.wav')]

#     if not wavfiles:
#         if wavfiles:
#             print(f"Reference audio clips in {REFERENCE_PATH} must be .wav files")
#             typer.Abort()
#         else:
#             print(f"You do not have any reference audio clips. Place a .wav file {REFERENCE_PATH} containing a ~3 second voice clip to use as a reference.")
#             typer.Exit()

#     future = asyncio.Future()

#     def on_choose(value: str):
#         if future.done:
#             return
#         manager.remove(window)
#         future.set_result(value)
#     window = ptg.Window(
#             "[bold accent]This is my example",
#             *[[f[:-4], lambda _: on_choose(f)] for f in wavfiles],
#             is_noblur = True,
#             is_noresize = True,
#             is_static = True,
#     )

#     manager.add(window)

#     return future

from prompt_toolkit.widgets import RadioList, Label
# from prompt_toolkit.widgets.base import _DialogList, _T
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets.base import _T, E as Event
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window, Container
from prompt_toolkit.layout.margins import ConditionalMargin, ScrollbarMargin
from prompt_toolkit.key_binding.key_bindings import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.formatted_text import AnyFormattedText, to_formatted_text, StyleAndTextTuples
from prompt_toolkit.keys import Keys


class SelectList(typing.Generic[_T]):
    """
    List of selectable values. Only one can be chosen, then its value is returned

    :param values: List of (label, value) tuples.
    """

    open_character: str = ""
    close_character: str = ""
    # ⏺ • ○ ⊙ ⋅ 🞅 ⦿
    selected_character: str = "⦿"
    unselected_character: str = "○"
    container_style: str = ""
    default_style: str = ""
    selected_style: str = ""
    show_scrollbar: bool = True

    def __init__(
        self,
        values: typing.Sequence[tuple[AnyFormattedText, _T]],
    ) -> None:
        assert len(values) > 0

        self.choices = sorted(values)
        self.value: _T | None = None
        self._selected_index = 0

        # Key bindings.
        kb = KeyBindings()

        @kb.add("up")
        def _up(event: Event) -> None:
            self._selected_index = max(0, self._selected_index - 1)

        @kb.add("down")
        def _down(event: Event) -> None:
            self._selected_index = min(len(self.choices) - 1, self._selected_index + 1)

        @kb.add("pageup")
        def _pageup(event: Event) -> None:
            w = event.app.layout.current_window
            if w.render_info:
                self._selected_index = max(
                    0, self._selected_index - len(w.render_info.displayed_lines)
                )

        @kb.add("pagedown")
        def _pagedown(event: Event) -> None:
            w = event.app.layout.current_window
            if w.render_info:
                self._selected_index = min(
                    len(self.choices) - 1,
                    self._selected_index + len(w.render_info.displayed_lines),
                )

        @kb.add("enter")
        @kb.add(" ")
        def _click(event: Event) -> None:
            self.value = self.choices[self._selected_index][1]
            event.app.exit(result=self.value)  

        @kb.add('c-c')
        def exit_(event):
            """
            Pressing Ctrl-c will exit the user interface.
            """
            event.app.exit()

        @kb.add(Keys.Any)
        def _find(event: Event) -> None:
            # We first check values after the selected value, then all values.
            values = list(self.choices)
            for value in values[self._selected_index + 1 :] + values:
                text = fragment_list_to_text(to_formatted_text(value[0])).lower()

                if text.startswith(event.data.lower()):
                    self._selected_index = self.choices.index(value)
                    return

        # Control and window.
        self.control = FormattedTextControl(
            self._get_text_fragments, key_bindings=kb, focusable=True
        )

        self.window = Window(
            content=self.control,
            style=self.container_style,
            right_margins=[
                ConditionalMargin(
                    margin=ScrollbarMargin(display_arrows=True),
                    filter=Condition(lambda: self.show_scrollbar),
                ),
            ],
            dont_extend_height=True,
        )

    def _get_text_fragments(self) -> StyleAndTextTuples:
        result: StyleAndTextTuples = []
        for i, value in enumerate(self.choices):
            selected = i == self._selected_index

            style = ""
            if selected:
                style += " " + self.selected_style

            result.append((style, self.open_character))

            if selected:
                result.append(("[SetCursorPosition]", ""))

            result.append((style, self.selected_character if selected else self.unselected_character))
            result.append((style, self.close_character))
            result.append((self.default_style, " "))
            result.extend(to_formatted_text(value[0], style=self.default_style))
            result.append(("", "\n"))

        result.pop()  # Remove last newline.
        return result

    def __pt_container__(self) -> Container:
        return self.window

def main():

    depend_zip('LibriTTS pre-trained model', MODEL_PATH + CONFIG_FILENAME, MODEL_URL)
    depend_zip('Punkt tokenizer', PUNKT_PATH, PUNKT_URL, TOKENIZERS_PATH)

    initialize()
    print('[green]Successfully initialized model.[/green]')

    # with ptg.WindowManager() as manager:
    #     manager.layout.add_slot("Body")
        
    #     reference = choose_reference(manager)



    #     await reference

    #     window = ptg.Window(
    #         f"[bold accent] Chose reference: {reference.result}",
    #         is_noblur = True,
    #         is_noresize = True,
    #         is_static = True,
    #     )

    #     manager.add(window)

    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    filenames = next(os.walk(REFERENCE_PATH), (None, None, []))[2]
    wavfiles = [f for f in filenames if f.endswith('.wav')]

    if not wavfiles:
        if filenames:
            print(f"Reference audio clips in {REFERENCE_PATH} must be .wav files")
            typer.Abort()
        else:
            print(f"You do not have any reference audio clips. Place a .wav file {REFERENCE_PATH} containing a ~3 second voice clip to use as a reference.")
            typer.Exit()

    select_list =  SelectList([(f[:-4], f) for f in wavfiles])
    application = Application(
        layout=Layout(HSplit([ Label('Choose reference speaker:'), select_list])),
        key_bindings=load_key_bindings(),
    )

    application.output.show_cursor = lambda:None
    
    choice = application.run()
    if choice is None:
        return

    synthesize(choice, "Thus did this humane and right minded father comfort his unhappy daughter, and her mother embracing her again, did all she could to soothe her feelings.")
    print(f"[green]Synthesized {1} items.[/green]")

# def wrap_main():
#     asyncio.run(main())

if __name__ == "__main__":
    # typer.run(wrap_main)
    typer.run(main)

