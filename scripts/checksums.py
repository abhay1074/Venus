"""Write and verify scripts/checksums.txt for the trained checkpoints.

    python scripts/checksums.py            # verify every checkpoint present
    python scripts/checksums.py --write    # regenerate checksums.txt (after training)

The checkpoints are ~1 GB and not in git, so a corrupt or truncated download is
the likeliest way a demo machine fails. checksums.txt is sha256sum format
(`<hash>  <name>`), so `sha256sum -c scripts/checksums.txt` works in WSL too.

Exit status: 0 all present files match, 1 a mismatch (corrupt file), 2 a
required checkpoint is missing. Required = the grader and the modality gate,
without which nothing can be graded; the quality CNN and lesion U-Net are
optional (the pipeline falls back and each result says which path ran).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "backend" / "weights"
CHECKSUMS = ROOT / "scripts" / "checksums.txt"

# name -> required. Mirrors backend/venus/config.py; the served pipeline needs
# the first two, degrades in a stated way without the others.
CHECKPOINTS = {
    "grader_v2.weights.h5": True,
    "eye_modality_gate.weights.h5": True,
    "quality_cnn.weights.h5": False,
    "lesion_unet.weights.h5": False,
    "eye_best.weights.h5": False,          # legacy v1 grader, only used without grader_v2
    "lesion_unet_1024.weights.h5": False,  # experiment, not served (config/experiments/)
}


def sha256_bytes(path: Path) -> str:
    """Raw byte hash - never the line-ending-normalised text hash in config.py."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums() -> dict[str, str]:
    if not CHECKSUMS.exists():
        return {}
    out = {}
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        out[name.strip()] = digest
    return out


def write() -> int:
    lines = ["# SHA-256 of the trained checkpoints (raw bytes). Regenerate with:",
             "#   python scripts/checksums.py --write",
             "# Verify with:  python scripts/checksums.py   (or: sha256sum -c scripts/checksums.txt)"]
    for name in CHECKPOINTS:
        path = WEIGHTS / name
        if not path.exists():
            print(f"skip {name}: not present")
            continue
        digest = sha256_bytes(path)
        lines.append(f"{digest}  {name}")
        print(f"{digest}  {name}  ({path.stat().st_size / 1e6:.0f} MB)")
    CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {CHECKSUMS}")
    return 0


def verify() -> int:
    expected = read_checksums()
    if not expected:
        print(f"no {CHECKSUMS}; run python scripts/checksums.py --write on a machine with the checkpoints", file=sys.stderr)
        return 2
    status = 0
    for name, required in CHECKPOINTS.items():
        path = WEIGHTS / name
        want = expected.get(name)
        if not path.exists():
            if required:
                print(f"MISSING  {name}  (required; see docs/DEMO.md 'If the checkpoints will not download')", file=sys.stderr)
                status = max(status, 2)
            else:
                print(f"absent   {name}  (optional)")
            continue
        if want is None:
            print(f"unknown  {name}  (not in checksums.txt; regenerate with --write)")
            continue
        got = sha256_bytes(path)
        if got != want:
            size = path.stat().st_size
            print(f"CORRUPT  {name}\n         expected {want}\n         actual   {got}\n"
                  f"         size on disk {size:,} bytes - re-download or re-copy this file", file=sys.stderr)
            status = 1
        else:
            print(f"ok       {name}")
    if status == 0:
        print("all checkpoints verified")
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="regenerate checksums.txt from the files on disk")
    args = parser.parse_args(argv)
    return write() if args.write else verify()


if __name__ == "__main__":
    sys.exit(main())
