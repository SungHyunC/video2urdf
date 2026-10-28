"""Download a benchmark subset of the RBO dataset (TU Berlin, CC BY 4.0) into data/rbo/.

R. Martín-Martín, C. Eppner, O. Brock, "The RBO Dataset of Articulated Objects and
Interactions", IJRR 2019. https://zenodo.org/records/1036660 (CC BY 4.0).

Usage: .venv/bin/python scripts/rbo_download.py [laptop02 book03 ...]
Without names, downloads SUBSET: static camera, plain background, large interactions.
Each sequence is extracted to data/rbo/x/<name>/ and each object model to data/rbo/x/<object>_model/.
"""

import shutil
import ssl
import sys
import tarfile
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parent.parent
RECORD = "https://zenodo.org/api/records/1036660/files/{}/content"
SUBSET = ["laptop02", "laptop04", "laptop05", "book03", "book05", "book13", "treasurebox02", "treasurebox03",
          "treasurebox05", "cardboardbox02", "cardboardbox03", "cardboardbox06", "pliers02", "pliers03", "pliers04",
          "foldingrule01", "foldingrule02"]


def fetch(name: str, raw: Path) -> Path:
    dst = raw / name
    if not dst.exists():
        tmp = dst.with_suffix(".part")
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(RECORD.format(name), context=ctx) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(dst)
    return dst


def extract(archive: Path, out: Path):
    if out.exists():
        return
    out.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")  # refuses absolute paths and links out of the folder


def main():
    raw, x = ROOT / "data" / "rbo" / "raw", ROOT / "data" / "rbo" / "x"
    raw.mkdir(parents=True, exist_ok=True)
    for seq in sys.argv[1:] or SUBSET:
        obj = seq.rstrip("0123456789")
        extract(fetch(f"{obj}.tar.gz", raw), x / f"{obj}_model")
        extract(fetch(f"{seq}_o.tar.gz", raw), x / seq)
        print(seq, "ok", flush=True)


if __name__ == "__main__":
    main()
