"""
File utility functions
"""

import os


def save_last_chunking_method(method):
    """Save the last used chunking method to a file"""
    with open(".last_chunking_method", "w") as f:
        f.write(method)


def get_last_chunking_method():
    """Get the last used chunking method from file"""
    try:
        if os.path.exists(".last_chunking_method"):
            with open(".last_chunking_method", "r") as f:
                return f.read().strip()
    except Exception:
        pass
    return "semantic"  # default
