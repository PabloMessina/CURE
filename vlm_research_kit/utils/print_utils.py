from typing import Any, Optional
from termcolor import colored, ATTRIBUTES, COLORS

def print_colored(
    *args: Any,
    color: Optional[str] = None,
    bold: bool = False,
    end: str = "\n",
    **kwargs: Any,
):
    """
    Prints arguments to the console, formatting the entire line.

    Args:
        *args: Objects to print. They will be converted to strings and joined
               with spaces.
        color (Optional[str]): Name of the color (e.g., 'red', 'blue').
                               See termcolor.COLORS for options. Defaults to None.
        bold (bool): Apply bold style. Defaults to False.
        end (str): String appended after the last value, default a newline.
        **kwargs: Additional keyword arguments passed to termcolor.colored
                  (e.g., on_color, other attrs).
    """
    text = " ".join(map(str, args))
    attrs = kwargs.pop("attrs", [])
    if bold and "bold" not in attrs:
        attrs.append("bold")

    # Filter out invalid color/attrs before passing to colored
    valid_color = color if color in COLORS else None
    valid_attrs = [a for a in attrs if a in ATTRIBUTES]

    if valid_color or valid_attrs:
        # Only apply colored() if there's actually formatting to do
        formatted_text = colored(
            text, color=valid_color, attrs=valid_attrs, **kwargs
        )
        print(formatted_text, end=end)
    else:
        # Otherwise, print plain text
        print(text, end=end)


# --- Specific Helper Functions ---


def print_bold(*args: Any, end: str = "\n", **kwargs: Any):
    """Prints text in bold (using the terminal's default color)."""
    print_colored(*args, color=None, bold=True, end=end, **kwargs)


def print_red(*args: Any, bold: bool = False, end: str = "\n", **kwargs: Any):
    """Prints text in red, optionally bold."""
    print_colored(*args, color="red", bold=bold, end=end, **kwargs)


def print_green(*args: Any, bold: bool = False, end: str = "\n", **kwargs: Any):
    """Prints text in green, optionally bold."""
    print_colored(*args, color="green", bold=bold, end=end, **kwargs)


def print_yellow(
    *args: Any, bold: bool = False, end: str = "\n", **kwargs: Any
):
    """Prints text in yellow, optionally bold."""
    print_colored(*args, color="yellow", bold=bold, end=end, **kwargs)


def print_blue(*args: Any, bold: bool = False, end: str = "\n", **kwargs: Any):
    """Prints text in blue, optionally bold."""
    print_colored(*args, color="blue", bold=bold, end=end, **kwargs)


def print_magenta(
    *args: Any, bold: bool = False, end: str = "\n", **kwargs: Any
):
    """Prints text in magenta, optionally bold."""
    print_colored(*args, color="magenta", bold=bold, end=end, **kwargs)


def print_cyan(*args: Any, bold: bool = False, end: str = "\n", **kwargs: Any):
    """Prints text in cyan, optionally bold."""
    print_colored(*args, color="cyan", bold=bold, end=end, **kwargs)


def print_white(*args: Any, bold: bool = False, end: str = "\n", **kwargs: Any):
    """Prints text in white, optionally bold."""
    print_colored(*args, color="white", bold=bold, end=end, **kwargs)


def print_grey(*args: Any, bold: bool = False, end: str = "\n", **kwargs: Any):
    """Prints text in grey, optionally bold."""
    print_colored(*args, color="grey", bold=bold, end=end, **kwargs)