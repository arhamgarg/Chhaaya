"""Medicine lexicon (design sections 5.2 and 8): NLEM 2022 generics and brand names.

`uv run python -m chhaaya.lexicon` downloads the sources in
data/lexicon_sources.csv to data/lexicon/ (kept out of git), checks their
SHA-256, and writes data/lexicon.csv: one row per medicine in each section of
the National List of Essential Medicines 2022, with its dosage forms and
strengths, and one row per brand in the Common Drug Codes for India, with the
generic medicine it contains. `Lexicon.load().lookup(text)` returns the entry
closest to a medicine name read off a prescription, and its rapidfuzz score.
"""

import csv
import hashlib
import re
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import httpx2
import pymupdf
from rapidfuzz import fuzz, process, utils

DATA = Path(__file__).resolve().parents[2] / "data"
SOURCES = DATA / "lexicon_sources.csv"
DOWNLOADS = DATA / "lexicon"
LEXICON = DATA / "lexicon.csv"

_ITEM = re.compile(r"\d+(\.\d+)+")
# A dosage line starts a new form ("Tablet 5 mg", "a) Liposomal ...") unless it
# continues a wrapped one: "(p)", "mg/vial", "+ 200 mg (C)", or follows a line
# cut after "+", "-", "/" or a joining word ("Contains 52 mg of", "As").
_NEW_FORM = re.compile(r"[A-Z]|[a-z]\) ")
_CONTINUED = re.compile(r"([-+–/(]|\b(of|as|and|in|with|for))$", re.IGNORECASE)
# Prescription shorthand for the dosage form, ahead of the medicine name. A
# lone "T" or "C" stays: it starts brand names such as T-Bact and C-Tax.
_FORM_PREFIX = re.compile(
    r"^(tabs?|tablets?|caps?|capsules?|syp|syr|syrup|inj|injection|oint|drops?) "
)
# A strength written without a space: "650mg", "1gm".
_GLUED_UNIT = re.compile(r"(\d)([a-z])")
_GRAM = re.compile(r"\bgm\b")


@dataclass(frozen=True)
class Entry:
    name: str  # as a prescription would write it: a generic or a brand
    generic: str  # brands: the generic with its strength and dose form
    dosage: str  # NLEM only: its dosage forms and strengths, "; "-separated
    source: str


@dataclass(frozen=True)
class Match:
    entry: Entry
    score: float  # rapidfuzz token-set ratio, 0 to 100


class Lexicon:
    def __init__(self, entries: Sequence[Entry]) -> None:
        self.entries = list(entries)
        self._names = [_normalise(e.name) for e in self.entries]

    @classmethod
    def load(cls, path: Path = LEXICON) -> "Lexicon":
        with path.open(encoding="utf-8", newline="") as f:
            return cls([Entry(**row) for row in csv.DictReader(f)])

    def lookup(self, text: str) -> Match:
        """The entry whose name best contains the words of `text`."""
        query = _normalise(text)
        # Sorted best first; every entry tied with the best is rescored below.
        scored = process.extract(
            query, self._names, scorer=fuzz.token_set_ratio, processor=None, limit=None
        )
        ties = [c for c in scored if c[1] == scored[0][1]]
        # "Telma 40" contains every word of both "Telma 40 mg oral tablet" and
        # "Telma-Beta ER 25 mg + 40 mg ...": prefer the closer whole name.
        _, score, index = max(ties, key=lambda c: fuzz.ratio(query, c[0]))
        return Match(self.entries[index], score)


def _normalise(name: str) -> str:
    """Lower case, punctuation to spaces, "650mg" to "650 mg", no form word."""
    name = _GRAM.sub("g", _GLUED_UNIT.sub(r"\1 \2", utils.default_process(name)))
    return _FORM_PREFIX.sub("", " ".join(name.split()))


def read_nlem(pdf: Path) -> list[Entry]:
    """One entry per numbered medicine in each therapeutic section's table."""
    entries = []
    seen = set()
    # A table continued from the previous page has no header row of its own.
    columns = None
    with pymupdf.open(pdf) as doc:
        for page in doc:
            text = " ".join(page.get_text().split())
            if text.startswith("Alphabetical List of Medicines"):
                break  # the section tables end here; the indexes follow
            for table in page.find_tables().tables:
                for row in table.rows:
                    cells = [page.get_textbox(c) if c else "" for c in row.cells]
                    if any(c.strip().startswith("Medicine") for c in cells):
                        columns = [c for c in row.cells if c]
                        continue
                    if columns is None:
                        continue
                    # Read each column across the row's full height: merged
                    # cells otherwise split one medicine over several rows.
                    top, bottom = row.bbox[1], row.bbox[3]
                    number, name, _level, dosage = (
                        page.get_textbox(pymupdf.Rect(c[0], top, c[2], bottom))
                        for c in columns
                    )
                    number = number.strip()
                    if not _ITEM.fullmatch(number) or number in seen:
                        continue
                    seen.add(number)
                    name = " ".join(name.replace("*", "").split())
                    entries.append(
                        Entry(
                            name=name,
                            generic=name,
                            dosage="; ".join(dosage_forms(dosage)),
                            source=f"NLEM 2022 {number}, p. {page.number + 1}",
                        )
                    )
    return entries


def dosage_forms(cell: str) -> list[str]:
    forms: list[str] = []
    for line in (" ".join(line.split()) for line in cell.splitlines()):
        if not line:
            continue
        if forms and (not _NEW_FORM.match(line) or _CONTINUED.search(forms[-1])):
            forms[-1] = f"{forms[-1]} {line}"
        else:
            forms.append(line)
    return forms


def read_cdci(package: Path) -> tuple[list[Entry], int]:
    """Every CDCI brand with its generic, and how many name an unknown generic."""
    with zipfile.ZipFile(package) as zf:
        folder = Path(zf.namelist()[0]).parts[0]

        def rows(name: str) -> list[dict[str, str]]:
            lines = zf.read(f"{folder}/{name}").decode("utf-8").splitlines()
            return list(csv.DictReader(lines, delimiter="\t"))

        generics = {
            r["Identifier"]: r["Generic Name"] for r in rows("GenericMaster.txt")
        }
        brands = rows("BrandMaster.txt")
    entries = [
        Entry(
            name=b["Brand Name"],
            generic=generics[b["Generic Identifier"]],
            dosage="",
            source=f"CDCI brand {b['Identifier']}",
        )
        for b in brands
        if b["Generic Identifier"] in generics
    ]
    return entries, len(brands) - len(entries)


class DownloadError(RuntimeError):
    pass


def download(directory: Path = DOWNLOADS, sources: Path = SOURCES) -> dict[str, Path]:
    """Fetch each source not already in `directory`, checking its SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    with (
        sources.open(encoding="utf-8", newline="") as f,
        httpx2.Client(follow_redirects=True, timeout=120) as http,
    ):
        for source in csv.DictReader(f):
            target = directory / Path(source["url"]).name
            if not target.exists():
                response = http.get(source["url"])
                response.raise_for_status()
                partial = target.with_suffix(".part")
                partial.write_bytes(response.content)
                partial.replace(target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != source["sha256"]:
                raise DownloadError(
                    f"{source['id']}: {target} has SHA-256 {digest}, not the "
                    f"{source['sha256']} in {sources}; review the new edition "
                    "before updating the checksum"
                )
            paths[source["id"]] = target
    return paths


def write(entries: Sequence[Entry], path: Path = LEXICON) -> None:
    partial = path.with_suffix(".part")
    with partial.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        out.writerow(field.name for field in fields(Entry))
        out.writerows(astuple(e) for e in entries)
    partial.replace(path)


def build() -> None:
    paths = download()
    nlem = read_nlem(paths["nlem_2022"])
    brands, skipped = read_cdci(paths["cdci"])
    write(nlem + brands)
    print(
        f"{len(nlem)} NLEM 2022 entries and {len(brands)} CDCI brands -> {LEXICON}; "
        f"skipped {skipped} brands whose generic is missing from the package",
        file=sys.stderr,
    )


if __name__ == "__main__":
    build()
