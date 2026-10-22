import csv
from app.model_check import LABEL, LEARNED, mae_table

def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["origin", "target", "h", "method", "y_true", "y_pred", "observed", "period"])
        w.writerows(rows)

def test_average_miss_by_period_and_method(tmp_path):
    p = tmp_path / "t.csv"
    _write(p, [("2022-01", "2022-02", 1, "naive", 10, 14, True, "development"),
               ("2022-02", "2022-03", 1, "naive", 10, 8, True, "development"),
               ("2026-01", "2026-02", 1, "naive", 100, 130, True, "holdout"),
               ("2022-01", "2022-03", 2, "naive", 10, 99, True, "development")])   # other horizon is ignored
    t = mae_table(str(p))
    assert t["development"]["naive"] == (3.0, 2)
    assert t["holdout"]["naive"] == (30.0, 1)

def test_labels_cover_every_method():
    assert LEARNED <= set(LABEL) and {"naive", "naive3", "snaive", "cagr"} <= set(LABEL)