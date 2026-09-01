from dataclasses import dataclass

import carb.input
import omni.appwindow
import weakref


@dataclass
class KeyboardState:
    save_episode: bool = False
    discard_episode: bool = False
    quit: bool = False


class KeyboardControl:
    """
    Keyboard controls

    RIGHT : Save episode
    LEFT  : Discard episode
    Q     : Quit
    """

    def __init__(self):
        self.state = KeyboardState()

        self._keyboard = None
        self._sub_key = None

        self._setup_keyboard()

    def _setup_keyboard(self):
        app_window = omni.appwindow.get_default_app_window()

        if app_window is None:
            raise RuntimeError(
                "No default app window found. Keyboard input cannot be initialized."
            )

        self._keyboard = app_window.get_keyboard()

        if self._keyboard is None:
            raise RuntimeError(
                "Failed to get keyboard from app window."
            )

        self._input = carb.input.acquire_input_interface()

        self._sub_key = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                event, *args
            )
        )

        print("[INFO]: Keyboard control initialized")

    def destroy(self):
        if self._sub_key is not None:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard,
                self._sub_key,
            )
            self._sub_key = None

    def _on_keyboard_event(self, event, *args, **kwargs):

        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        key_name = event.input.name

        if key_name == "RIGHT":
            self.state.save_episode = True

        elif key_name == "LEFT":
            self.state.discard_episode = True

        elif key_name == "Q":
            self.state.quit = True

        return True


    def consume_save(self):
        value = self.state.save_episode
        self.state.save_episode = False
        return value

    def consume_discard(self):
        value = self.state.discard_episode
        self.state.discard_episode = False
        return value

    def should_quit(self):
        return self.state.quit