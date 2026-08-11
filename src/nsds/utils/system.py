import subprocess
import sys
import warnings


def _escape_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def show_mac_notification(message: str, title: str = "Notification"):
    """
    No-op on anything but macOS, so that the same notebook can run on a cluster.
    """
    if sys.platform != "darwin":
        warnings.warn(
            f"Desktop notifications are only supported on macOS: {title} - {message}",
            stacklevel=2,
        )
        return

    script = (
        f'display notification "{_escape_applescript(message)}" '
        f'with title "{_escape_applescript(title)}"'
    )
    subprocess.run(["osascript", "-e", script])
