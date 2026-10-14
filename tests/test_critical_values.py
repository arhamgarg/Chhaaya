import csv
from pathlib import Path

CRITICAL_VALUES = Path(__file__).parents[1] / "data" / "critical_values.csv"


def read() -> list[dict[str, str]]:
    with CRITICAL_VALUES.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_every_row_is_complete_and_cites_a_page_and_url():
    for row in read():
        assert row["test"] and row["unit"], row
        assert float(row["threshold"]) > 0, row
        assert row["direction"] in {"below", "above"}, row
        assert row["population"] in {"adult", "pregnant", "child"}, row
        assert " p. " in row["source"] or " pp. " in row["source"], row
        assert "https://" in row["source"], row
        low, high = row["min_age_years"], row["max_age_years"]
        if low and high:
            assert float(low) < float(high), row


def test_haemoglobin_covers_every_population_and_child_ages_do_not_overlap():
    haemoglobin = [r for r in read() if r["test"] == "haemoglobin"]
    assert {r["population"] for r in haemoglobin} == {"adult", "pregnant", "child"}
    children = sorted(
        (float(r["min_age_years"]), float(r["max_age_years"]))
        for r in haemoglobin
        if r["population"] == "child"
    )
    for (_, end), (start, _) in zip(children, children[1:], strict=False):
        assert end <= start
    adults = [r for r in haemoglobin if r["population"] == "adult"]
    assert float(adults[0]["min_age_years"]) == children[-1][1]
