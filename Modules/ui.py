import typing
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets.base import _T, E as Event
from prompt_toolkit.layout.containers import Window, Container
from prompt_toolkit.layout.margins import ConditionalMargin, ScrollbarMargin
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.formatted_text import AnyFormattedText, to_formatted_text, StyleAndTextTuples
from prompt_toolkit.keys import Keys

# Modified version of prompt_toolkit.widgets.base._DialogList
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