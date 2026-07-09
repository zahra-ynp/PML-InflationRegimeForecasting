# Inflation Regime Forecasting

This project investigates how macroeconomic data and financial news sentiment can be used to identify latent inflation regimes. Regime identification is performed using Hidden Markov Models (HMM) and Autoregressive HMMs (AR-HMM).

**The current scope covers US data only.** European analysis is planned for a future iteration.

---


## US Analysis

### `01_Data_Prep.ipynb` — Data Preparation

This notebook handles two tasks: downloading macroeconomic time series from the FRED API, and scoring a corpus of financial news articles through FinBERT to produce a monthly sentiment signal. Its output — `data/processed/US/us_macro_2007_2023.csv` — feeds directly into the modelling notebook.

#### Macroeconomic Indicators (FRED API, 2007–2023)

| Indicator | FRED ID | Description 
|---|---|---|---|
| CPI | `CPIAUCSL` | Consumer Price Index |
| Unemployment | `UNRATE` | Civilian unemployment rate |
| Interest Rate | `FEDFUNDS` | Effective Federal Funds Rate |
| Industrial Production | `INDPRO` | Industrial Production & Capacity Utilization |

#### News Sentiment — FinBERT

The sentiment pipeline uses `FinSen_US_Categorized_Timestamp.csv`, a dataset of approximately 15,000 financial news articles covering the same sample period. Each article is passed through [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) — a BERT-based model fine-tuned on financial text.

FinBERT outputs class probabilities for *positive*, *negative*, and *neutral* labels. These are converted to a net sentiment score per article:

```
score = P(positive) - P(negative)     ∈ [-1, +1]
```

Scores are averaged across all articles within each calendar month to produce a single `News_Sentiment` signal, which is merged with the macroeconomic indicators.


### `02_AR_HMM_and_Evaluation.ipynb` — HMM & AR-HMM Regime Detection

This notebook identifies US inflation regimes using two classes of Markov Switching models fitted to the `Inflation_MoM` series. The key methodological contribution is **incremental feature selection guided by BIC**: macro indicators are added one at a time to quantify each variable's contribution to model fit.

#### Incremental Feature Selection

Starting from a baseline (inflation only), macro indicators are added sequentially. BIC is recorded at each step:

| Step | Model Specification |
|---|---|
| M1 | Baseline — no exogenous variables |
| M2 | + News Sentiment (FinBERT monthly average) |
| M3 | + Unemployment Rate |
| M4 | + Industrial Production MoM |

The specification with the lowest BIC is selected as the best model within each family.


## Europe

Planned for a future version.

---

## Setup

Requires Python 3.8 or higher. Install dependencies with:

```bash
pip install pandas numpy matplotlib statsmodels fredapi transformers torch tqdm
```

A free FRED API key is required and can be obtained at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html). Set the key in the `FRED_API_KEY` variable at the top of `01_Data_Prep.ipynb`.

**Notebooks must be run in order**: `01_Data_Prep` first, then `02_AR_HMM_and_Evaluation`.
