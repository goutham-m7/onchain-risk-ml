"""Fetch immutable public CSVs; verify content before replacing local files."""

import csv
import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

from .graph import contexts

ROOT = Path("data")
SOURCE = "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/d075df05ace1db9a826512fe71806aaecb76f97e/"
HASHES = {
    "classes": "93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493",
    "edgelist": "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",
    "features": "fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0",
}


def path(kind):
    return ROOT / f"elliptic_txs_{kind}.csv"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch():
    ROOT.mkdir(exist_ok=True)
    for kind, expected in HASHES.items():
        target = path(kind)
        if target.is_file() and digest(target) == expected:
            print(f"verified {target}")
            continue
        with tempfile.NamedTemporaryFile(dir=ROOT, delete=False) as f:
            tmp = Path(f.name)
        try:
            with (
                urlopen(SOURCE + target.name + "?download=true", timeout=90) as r,
                tmp.open("wb") as out,
            ):
                while chunk := r.read(1024 * 1024):
                    out.write(chunk)
            if digest(tmp) != expected:
                raise ValueError(f"checksum mismatch: {target.name}")
            tmp.replace(target)
            print(f"fetched {target}")
        finally:
            tmp.unlink(missing_ok=True)


def load():
    for k, expected in HASHES.items():
        if digest(path(k)) != expected:
            raise ValueError(f"unexpected {k} CSV; run python -m risk.data")
    with path("features").open("rb") as f:
        # Ignore all precomputed covariates: only the ID and time step are needed.
        steps = {int(n): int(t) for n, t, _ in (line.split(b",", 2) for line in f)}
    with path("classes").open() as f:
        labels = {
            int(row["txId"]): {"1": 1, "2": 0, "unknown": -1}[row["class"]]
            for row in csv.DictReader(f)
        }
    if labels.keys() != steps.keys():
        raise ValueError("incomplete label join")
    with path("edgelist").open() as f:
        graph = contexts(
            steps, ((int(r["txId1"]), int(r["txId2"])) for r in csv.DictReader(f))
        )
    return steps, labels, graph


if __name__ == "__main__":
    fetch()
