"""
Helper functions for the Insights Engine
"""

from typing import Optional


def print_header(text: str, char: str = "═", width: int = 80):
    """Print a formatted header"""
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}\n")


def print_section(text: str, char: str = "─", width: int = 60):
    """Print a formatted section header"""
    print(f"\n{char * width}")
    print(f"📌 {text}")
    print(f"{char * width}")


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}m {secs}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal value as percentage"""
    return f"{value:.{decimals}f}%"


def format_number(value: int) -> str:
    """Format a number with thousand separators"""
    return f"{value:,}"

