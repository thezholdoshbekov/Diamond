"""Fetch the DNSMOS-P.835 ONNX model into data/pretrained/dnsmos_p835.onnx.

Run by `make dnsmos` (and by `make setup`). ~1.1 MB, from the public
DNS-Challenge repo. It lands at the path `make eval` already looks at, so no
flags or config edits are needed.
"""

import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/"
       "DNSMOS/DNSMOS/sig_bak_ovr.onnx")
TARGET = Path("data/pretrained/dnsmos_p835.onnx")
MIN_BYTES = 500_000   # a git-lfs pointer stub would be a few hundred bytes


def main() -> None:
    if TARGET.exists() and TARGET.stat().st_size > MIN_BYTES:
        print(f"DNSMOS model already present: {TARGET}")
        return

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DNSMOS-P.835 from {URL} ...")
    tmp = TARGET.with_suffix(".onnx.part")
    urllib.request.urlretrieve(URL, tmp)

    size = tmp.stat().st_size
    if size < MIN_BYTES:
        tmp.unlink()
        raise SystemExit(f"Downloaded file is only {size} B — expected a real ONNX. "
                         f"Fetch it manually from {URL}")
    tmp.replace(TARGET)
    print(f"DNSMOS model ready: {TARGET} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
