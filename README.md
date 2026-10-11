# Adaptive EV Adoption Forecasting & Spatial Intelligence

**SLIIT IT4010 Research Project — Individual Component 1**
**Group:** J26-DS-331 | **Student:** Ariyathilake P U L N (IT23413788)

## What This Does

Forecasts electric vehicle (EV) adoption across Sri Lanka's 25 districts (2026–2035)
using an adaptive, regime-aware framework. District forecasts are spatially disaggregated
to city/town level. Outputs feed into the group's charging demand, grid impact, and
station placement components.

## Project Structure

| Folder | Purpose |
|---|---|
| `data/synthetic/` | Synthetic development datasets (safe to share) |
| `data/external/` | External market data (IEA, Indonesia, etc.) |
| `data/raw/` | Real DMT data — local only, never committed |
| `src/` | Reusable Python pipeline modules |
| `notebooks/` | Exploratory analysis and EDA |
| `outputs/` | Generated forecasts and figures |
| `dashboard/` | Streamlit demonstration interface |
| `docs/` | Research documentation |
| `tests/` | Unit tests |

## Setup

```powershell
git clone https://github.com/LakduAriyathilake/ev-adoption-forecast-sl.git
cd ev-adoption-forecast-sl
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Notice

Real DMT registration data is stored locally only and excluded from this repository
via `.gitignore`. Synthetic datasets in `data/synthetic/` are used for development
and are clearly labelled. They do not represent official DMT statistics.

## Research Progress

- [x] Phase 1: Repository setup and environment
- [ ] Phase 2: Data audit and EDA
- [ ] Phase 3: Regime analysis and baseline models
- [ ] Phase 4: Adaptive selection and uncertainty
- [ ] Phase 5: Spatial disaggregation
- [ ] Phase 6: PP1 prototype and evidence