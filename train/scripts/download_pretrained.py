"""Fetch the DAC 44.1 kHz checkpoint into data/pretrained/dac_44khz.pth.

Run by `make dac`. The file lands at the exact path the configs already point
to (`dac.checkpoint_path`), so no config editing is needed.
"""

import shutil
from pathlib import Path

TARGET = Path("data/pretrained/dac_44khz.pth")


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    if TARGET.exists():
        print(f"DAC checkpoint already present: {TARGET}")
        return

    try:
        import dac
    except ImportError as e:
        raise SystemExit(
            "descript-audio-codec not installed. Run `make deps` first."
        ) from e

    print("Downloading DAC 44.1 kHz checkpoint...")
    src = Path(dac.utils.download(model_type="44khz"))
    shutil.copy2(src, TARGET)
    print(f"DAC checkpoint ready: {TARGET}")


if __name__ == "__main__":
    main()
