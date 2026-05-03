"""Minimal utils module required by run-fisher.py (jload is used only for Alpaca dataset path)."""
import io
import json


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        with open(f, mode=mode) as fh:
            return json.load(fh)
    return json.load(f)
