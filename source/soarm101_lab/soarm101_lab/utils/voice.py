import platform
import subprocess


def say(text: str, blocking: bool = False) -> None:
    if platform.system() == "Linux":
        command = ["spd-say", text]
    elif platform.system() == "Darwin":
        command = ["say", text]
    else:
        raise RuntimeError("Unsupported operating system")

    if blocking:
        subprocess.run(command, check=False)
    else:
        subprocess.Popen(command)


def log_say(text: str, blocking: bool = False) -> None:
    print(text, flush=True)
    say(text, blocking=blocking)