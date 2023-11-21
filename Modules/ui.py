import os
import typing
import gdown
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
import numpy.typing as npt
from scipy.io import wavfile
import pyaudio  
import wave

import typer
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets.base import _T, E as Event, _DialogList
from prompt_toolkit.layout.containers import Window, Container
from prompt_toolkit.layout.margins import ConditionalMargin, ScrollbarMargin
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.formatted_text import AnyFormattedText, to_formatted_text, StyleAndTextTuples
from prompt_toolkit.keys import Keys
from prompt_toolkit.widgets import Label
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window, Container
from prompt_toolkit.key_binding.defaults import load_key_bindings

from config import * 

# TODO: This works ok, but _DialogList does not accept keybinds, so Ctrl+C to interrupt is impossible
class SelectList(_DialogList[_T]):
    """
    List of selectable values. Only one can be chosen, then its value is returned

    :param values: List of (label, value) tuples.
    """
    selected_character: str = "⦿"
    unselected_character: str = "○"
    on_choose: typing.Callable[[_T], None] = lambda x: None

    def __init__(
        self,
        values: typing.Sequence[tuple[_T, AnyFormattedText]],
        on_choose: typing.Callable[[_T], None],
    ) -> None:
        super(SelectList, self).__init__(values)
        self.on_choose = on_choose

    def _handle_enter(self) -> None:
        self.on_choose(self.values[self._selected_index][0])

    def _get_text_fragments(self) -> StyleAndTextTuples:
        result: StyleAndTextTuples = []

        for i, value in enumerate(self.values):
            selected = i == self._selected_index

            if selected:
                result.append(("[SetCursorPosition]", ""))

            result.append(("", (self.selected_character if selected else self.unselected_character) + " "))
            result.extend(to_formatted_text(value[1]))
            result.append(("", "\n"))

        result.pop()  # Remove last newline.
        return result
    
def depend_zip(name: str, check_path: str, url: str, extract_path: str | None = None):
    """ Check for a dependency, and download and extract it if it is missing """
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

def choose_reference() -> str | None:
    """ Prompt user to choose a reference audio file. Searches config.REFERENCE_PATH. """
    # Ensure there is at least one reference speaker
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

    result = ''
    def set_result(x: str):
        result = x
    select_list = SelectList([(f[:-4], f) for f in wavfiles], set_result)
    application = Application(
        layout=Layout(HSplit([ Label('Choose reference speaker (Ctrl+C to quit)'), select_list])),
        key_bindings=load_key_bindings(),
    )

    # hackily hide the cursor by overwriting the show_cursor method
    show_cursor = application.output.show_cursor
    application.output.show_cursor = lambda: None
    application.run()
    # we have to restore it afterwards
    application.output.show_cursor = show_cursor
    return result

def write_audio(audio: npt.NDArray[np.float64], filename: str):
    """
    Write an audio stream as a list of floats between -1 and 1 to .wav file
    
    :return: path to the file that was written
    """
    # Convert to (little-endian) 16 bit integers.
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)

    # create the output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # pick a suitable filename
    filepath = OUTPUT_PATH + filename
    counter = 0
    version = ""
    while os.path.isfile(filepath + version + '.wav'):
        counter += 1
        version = f"_{counter:03d}"
    filepath = f"{OUTPUT_PATH}{filename}{version}.wav"
    wavfile.write(filepath, SAMPLE_RATE, audio)

    return filepath

def play_audio(filepath: str):
    if not os.path.isfile(filepath):
        return False
    
    f = wave.open(filepath, "rb")
    p = pyaudio.PyAudio()
    chunk = 1024
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  
    data = f.readframes(chunk)  
    
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  
    
    stream.stop_stream()
    stream.close()  
    p.terminate()
    f.close()