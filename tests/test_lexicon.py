import re
import time
import zipfile
from pathlib import Path

import pymupdf
import pytest

from chhaaya.lexicon import DOWNLOADS, LEXICON, Entry, Lexicon, build, read_cdci

INDEX = "Alphabetical List of Medicines in NLEM 2022"

ENTRIES = [
    # Look-alikes come first: a tie must go to the closer name, not the first.
    Entry(
        "Paracetamol (paracetamol) 500 mg oral tablet Agron Remedies Private Limited",
        "Acetaminophen 500 mg oral tablet",
        "",
        "CDCI brand 3",
    ),
    Entry(
        "Paracetamol",
        "Paracetamol",
        "Tablet 500 mg; Tablet 650 mg",
        "NLEM 2022 2.1.5, p. 6",
    ),
    Entry(
        "Telma-Beta ER 25 mg + 40 mg prolonged-release oral tablet",
        "Metoprolol tartrate 25 mg and Telmisartan 40 mg prolonged-release oral tablet",
        "",
        "CDCI brand 5",
    ),
    Entry(
        "Telma 40 mg oral tablet", "Telmisartan 40 mg oral tablet", "", "CDCI brand 4"
    ),
    Entry(
        "Amoxicillin",
        "Amoxicillin",
        "Capsule 250 mg; Capsule 500 mg",
        "NLEM 2022 6.2.1.1, p. 20",
    ),
    Entry(
        "Dolo 650 mg oral tablet",
        "Acetaminophen 650 mg oral tablet",
        "",
        "CDCI brand 1",
    ),
    Entry(
        "Dolod 650 mg oral tablet",
        "Acetaminophen 650 mg oral tablet",
        "",
        "CDCI brand 2",
    ),
    Entry(
        "T Pan 40 mg oral tablet",
        "Pantoprazole sodium 40 mg oral tablet",
        "",
        "CDCI brand 6",
    ),
    Entry(
        "Pantimo 40 mg/vial powder for solution for injection",
        "Pantoprazole 40 mg injection",
        "",
        "CDCI brand 7",
    ),
    Entry(
        "Calpol 120 mg/5 mL oral suspension",
        "Acetaminophen 120 mg/5 mL oral suspension",
        "",
        "CDCI brand 8",
    ),
]


@pytest.mark.parametrize(
    ("written", "name"),
    [
        ("Tab Dolo 650", "Dolo 650 mg oral tablet"),
        ("Dolo-650", "Dolo 650 mg oral tablet"),
        ("TAB. TELMA 40", "Telma 40 mg oral tablet"),
        ("T. Pan 40", "T Pan 40 mg oral tablet"),
        ("Syp Calpol", "Calpol 120 mg/5 mL oral suspension"),
        ("Paracetamol", "Paracetamol"),
        ("amoxycillin 500", "Amoxicillin"),
    ],
)
def test_lookup_reads_through_prescription_shorthand(written: str, name: str):
    assert Lexicon(ENTRIES).lookup(written).entry.name == name


def test_a_medicine_missing_from_the_lexicon_scores_low():
    lexicon = Lexicon(ENTRIES)
    assert lexicon.lookup("Inj Insulin").score < 60
    assert lexicon.lookup("Tab Dolo 650").score == 100


def _tsv(*rows: list[str]) -> str:
    return "".join("\t".join(row) + "\t\n" for row in rows)


def test_cdci_brands_carry_their_generic_and_skip_unknown_generics(tmp_path: Path):
    package = tmp_path / "cdci.zip"
    with zipfile.ZipFile(package, "w") as zf:
        zf.writestr(
            "CommonDrugCodesForIndia_FlatFilePackage/GenericMaster.txt",
            _tsv(
                ["Identifier", "Generic Name", "Substance Identifier"],
                ["10", "Acetaminophen 650 mg oral tablet", "1"],
                ["11", "Telmisartan 40 mg oral tablet", "2"],
            ),
        )
        zf.writestr(
            "CommonDrugCodesForIndia_FlatFilePackage/BrandMaster.txt",
            _tsv(
                [
                    "Identifier",
                    "Brand Name",
                    "Product Identifier",
                    "Generic Identifier",
                ],
                ["901", "Dolo 650 mg oral tablet", "5", "10"],
                ["902", "Telma 40 mg oral tablet", "6", "11"],
                ["903", "Mystery 5 mg oral tablet", "7", "99"],
            ),
        )

    entries, skipped = read_cdci(package)

    assert entries == [
        Entry(
            "Dolo 650 mg oral tablet",
            "Acetaminophen 650 mg oral tablet",
            "",
            "CDCI brand 901",
        ),
        Entry(
            "Telma 40 mg oral tablet",
            "Telmisartan 40 mg oral tablet",
            "",
            "CDCI brand 902",
        ),
    ]
    assert skipped == 1


@pytest.mark.lexicon
def test_built_lexicon_holds_nlem_2022_and_reads_real_brand_names():
    """Downloads NLEM 2022 and CDCI into data/lexicon/ unless already there."""
    started = time.perf_counter()
    build()
    print(f"\nbuilt {LEXICON} in {time.perf_counter() - started:.0f} s")
    lexicon = Lexicon.load()
    nlem = [e for e in lexicon.entries if e.source.startswith("NLEM")]
    brands = [e for e in lexicon.entries if e.source.startswith("CDCI")]
    print(f"{len(nlem)} NLEM 2022 entries, {len(brands)} CDCI brands")

    # The PDF ends with an alphabetical index of its 384 medicines; each must
    # be in a section table (spellings differ: "Hydroxocobalamine").
    with pymupdf.open(DOWNLOADS / "nlem2022.pdf") as doc:
        pages = [" ".join(page.get_text().split()) for page in doc.pages(85, 101)]
    index = re.findall(r"\d+\. (.+?)(?= \d+\. |$)", " ".join(pages).replace(INDEX, ""))
    assert len(index) == 384
    sections = Lexicon(nlem)
    scores = sorted((sections.lookup(name).score, name) for name in index)
    print(f"weakest index matches: {scores[:3]}")
    assert scores[0][0] >= 70

    by_name = {e.name: e for e in nlem}
    assert by_name["Paracetamol"].dosage.startswith("Tablet 500 mg; Tablet 650 mg")
    # Lines wrapped inside a table cell stay one dosage form.
    assert "Capsule/Tablet 50000 IU (including Chewable Tablet)" in (
        by_name["Vitamin A"].dosage
    )
    assert by_name["Hormone releasing IUD"].dosage.startswith(
        "Contains 52 mg of Levonorgestrel"
    )
    assert by_name["Oxaliplatin"].dosage == (
        "Injection 5 mg/mL in 10 mL vial; Injection 5 mg/mL in 20 mL vial"
    )
    assert all(e.dosage and e.source for e in nlem)

    latencies = []
    for written, generic in [
        ("Tab Dolo 650", "Acetaminophen 650 mg"),
        ("Glycomet 500", "Metformin hydrochloride 500 mg"),
        ("Cap Omez 20", "Omeprazole 20 mg"),
        ("Tab Amlong 5", "amlodipine besylate) 5 mg"),
        ("Ecosprin 75", "Aspirin 75 mg"),
        ("Thyronorm 50", "Levothyroxine sodium 50 microgram"),
    ]:
        started = time.perf_counter()
        match = lexicon.lookup(written)
        latencies.append(time.perf_counter() - started)
        entry = match.entry
        print(f"{written:14} {match.score:5.1f} {entry.name} -> {entry.generic}")
        assert generic in match.entry.generic
    print(f"lookup latency: max {max(latencies) * 1000:.0f} ms")
