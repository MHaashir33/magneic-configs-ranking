from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "database_snapshot"
FILENAME = "magnetic-orderings-database-snapshot.part1.json.gz"
URL = (
    "https://raw.githubusercontent.com/ncfrey/"
    "magnetic-topological-materials/master/data/database_snapshot/"
    + FILENAME
)
DESTINATION = DATA_DIR / FILENAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gzip_json(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("Downloaded payload is not a JSON list.")
    return len(payload)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(DESTINATION.suffix + ".partial")
    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "Altermagnet-ALIGNN-reproducibility-script/1.0"},
    )
    print("Downloading only:", FILENAME)
    print("Source:", URL)

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        record_count = validate_gzip_json(temporary)
        if record_count != 23155:
            raise RuntimeError(
                f"Expected 23,155 records in part1; found {record_count}."
            )

        # Once a release checksum exists, reject a changed upstream file.
        checksum_file = DATA_DIR / "SHA256SUMS.txt"
        if checksum_file.is_file():
            expected_hash = None
            for line in checksum_file.read_text(encoding="utf-8").splitlines():
                fields = line.strip().split()
                if len(fields) >= 2 and fields[-1].lstrip("*") == FILENAME:
                    expected_hash = fields[0].lower()
                    break
            if expected_hash is None:
                raise RuntimeError(
                    f"{checksum_file} does not contain a checksum for {FILENAME}."
                )
            actual_hash = sha256(temporary)
            if actual_hash.lower() != expected_hash:
                raise RuntimeError(
                    "Downloaded part1 SHA-256 does not match the repository checksum. "
                    f"Expected {expected_hash}; received {actual_hash}."
                )

        temporary.replace(DESTINATION)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print("Saved unchanged compressed file:", DESTINATION)
    print("JSON records:", record_count)
    print("SHA256:", sha256(DESTINATION))
    print("The gzip file itself was not rewritten or recompressed.")


if __name__ == "__main__":
    main()
