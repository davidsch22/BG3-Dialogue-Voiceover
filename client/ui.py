from PySimpleGUI import Text, Window, WIN_CLOSED


class UI:
    def __init__(self):
        # Initialize main window
        normal_style = ("Arial", 18, "normal")
        bold_style = ("Arial", 18, "bold")
        italic_style = ("Arial", 18, "italic")
        layout = [[Text("FPS: ", font=bold_style), Text("", key="fps", font=normal_style)],
                  [Text("Press F6 to pause and resume the screen reader",
                        font=italic_style)],
                  [Text("Paused Status: ", font=bold_style),
                      Text("Paused", key="pause_status", font=normal_style)],
                  [Text("Dialogue Text: ", font=bold_style),
                      Text("", key="dialogue_text", font=normal_style)]]
        self.window = Window("BG3 Dialogue Voiceover", layout)

    def set_fps(self, text):
        self.window["fps"].update(text)

    def set_pause_status(self, text):
        self.window["pause_status"].update(text)

    def set_dialogue_text(self, text):
        self.window["dialogue_text"].update(text)

    def is_closed(self):
        event, values = self.window.read(timeout=10)
        if event == WIN_CLOSED:
            return True
        return False

    def close_window(self):
        self.window.close()
