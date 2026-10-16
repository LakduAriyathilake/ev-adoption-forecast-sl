"""Sanity checks for the hand-built policy events table (no network needed)."""
from pathlib import Path
import pandas as pd

F = Path(__file__).resolve().parents[1] / "data" / "external" / "policy_events_lk.csv"
OK_STATUS = {"checked_in_source", "secondary_only", "headline_only", "date_inferred"}

def load():
    return pd.read_csv(F, dtype=str, keep_default_na=False)

def test_ids_unique_and_fields_present():
    d = load()
    assert d.event_id.is_unique
    for c in ["event_id", "date", "event", "source_url", "verification"]:
        assert (d[c].str.len() > 0).all(), c

def test_verification_values_and_urls():
    d = load()
    assert set(d.verification) <= OK_STATUS
    assert d.source_url.str.startswith("https://").all()

def test_dates_parse_and_are_ordered():
    d = load()
    start = pd.to_datetime(d.date.str.slice(0, 10).where(d.date.str.len() >= 10, d.date + "-01"), errors="coerce", format="mixed")
    assert start.notna().all()
    # year-only rows (precision 'year') have no real position inside the year, so skip them for ordering
    exact = start[d.date_precision != "year"]
    assert exact.is_monotonic_increasing