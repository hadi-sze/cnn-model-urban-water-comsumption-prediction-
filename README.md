# Water Consumption Prediction

A Python and PyTorch project for predicting daily water consumption with **1D convolutional neural networks (CNNs)**. Train a next-day predictor, generate recursive multi-day forecasts, and compare performance against previous-day and previous-week baselines.

The project includes an original CNN and an improved residual CNN ensemble with calendar features, chronological validation, and explicit multi-day backtesting. The urban district experiment uses `Daily_use`, interpreted as **cubic meters** for this dataset.

## Quick start

Use Python 3.11 or newer with a compatible PyTorch release. Download or clone this repository, open a terminal in its root directory, and create a virtual environment:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install dependencies and run the synthetic example:

```sh
python -m pip install -r requirements.txt
python water_cnn.py demo
python water_cnn.py train --csv data/demo_water_usage.csv
python water_cnn.py predict --csv data/demo_water_usage.csv --days 7
```

This creates demo data in `data/` and model, evaluation, and forecast files in `outputs/`. Synthetic results demonstrate the workflow; they do not establish accuracy on real consumption data.

The commands below use the Windows virtual-environment Python directly. On macOS/Linux, or with the environment activated, replace `.\.venv\Scripts\python.exe` with `python`.

## Project files

| File | Purpose |
|---|---|
| `water_cnn.py` | Original CNN, synthetic data generation, training, and prediction |
| `improved_cnn.py` | Residual CNN ensemble, model selection, prediction, and backtesting |
| `prepare_urban_data.py` | Prepare the urban district CSV and write a data-quality audit |
| `plot_results.py` | Generate PNG and SVG charts from saved results |
| `test_water_cnn.py`, `test_improved_cnn.py` | Automated model and data-processing checks |
| `requirements.txt` | Core NumPy and PyTorch dependencies |
| `requirements-charts.txt` | Chart-generation dependencies |
| `requirements-lock.txt` | Recorded dependency versions for the original environment |
| [RESULTS.md](RESULTS.md) | Original experiment results |
| [IMPROVED_RESULTS.md](IMPROVED_RESULTS.md) | Improved model results and evaluation limitations |

Generated checkpoints, forecasts, and charts live in `outputs/`, which is excluded from Git. Run training and prediction locally to create them after cloning.

## Recorded results

Rolling next-day predictions on the same **179 eligible test dates**:

| Model | MAE (m³) | RMSE (m³) |
|---|---:|---:|
| Improved residual CNN ensemble | **68,450.57** | **98,739.35** |
| Previous-day baseline | 71,250.90 | 100,202.94 |
| Original CNN | 163,221.61 | 269,454.90 |
| Previous-week baseline | 172,113.42 | 312,579.39 |

Lower error is better. The improved model reduced MAE by approximately **58.1%** compared with the original CNN and **3.9%** compared with the previous-day baseline. The test period was already inspected during the original experiment, so this is a reused comparison rather than a fresh independent holdout. See [IMPROVED_RESULTS.md](IMPROVED_RESULTS.md) for multi-day results and limitations.

## Improved model: start here

`improved_cnn.py` is the preferred version. It predicts a correction to yesterday's consumption, uses calendar features, and selects among 7-, 14-, and 30-day histories. The selected **14-day ensemble** achieved next-day MAE **68,450.57 m³**, versus **163,221.61 m³** for the original CNN and **71,250.90 m³** for the previous-day baseline, on the same 179 eligible dates. These are comparisons on the previously inspected test period; newer observations are needed for a fresh independent check.

With `data/urban_daily.csv` available, generate the improved model, forecast, and backtest after installing the requirements:

```powershell
.\.venv\Scripts\python.exe improved_cnn.py train
.\.venv\Scripts\python.exe improved_cnn.py predict --days 7
.\.venv\Scripts\python.exe improved_cnn.py backtest --days 7
```

To include the original CNN in the comparison, first train it using the urban dataset commands below, then add `--original-model outputs/urban/model.pt` to improved training. Training uses the prepared `data/urban_daily.csv` by default. To use another prepared file, pass `--csv path/to/data.csv --unit "your unit"` to training, and the same `--csv` to prediction/backtesting. The improved version expects `date,water_used` columns and at least 250 calendar days with 20 complete windows in each training/validation/test segment. Reusing the default output directory replaces the previous improved run; choose `--output` and matching `--model` paths to keep multiple runs.

Changes in the improved model:

- Predicts `yesterday + learned change`; starts from an exact previous-day baseline at epoch zero.
- Uses past consumption plus sine/cosine weekday and annual calendar features, including the known target date.
- Uses training-only median/IQR normalization. Normalized consumption inputs are clipped to [-10, 10] for numerical robustness; original records, target values, and evaluation errors remain unchanged.
- Trains a small CNN with Huber loss (delta 0.2 in normalized change units), AdamW, dropout, gradient clipping, and early stopping by validation MAE in m³.
- Compares three history lengths over two expanding validation folds (55%→70% and 70%→85% of the calendar). All candidates use the same dates with complete 30-day history. Averages three reproducible seeds (11, 22, 33) for each candidate.
- Selects the lowest pooled validation MAE, saves the selection and ensemble, then evaluates the last 15%. The saved models are the selected candidate's second-fold checkpoints, trained through April 2, 2024 and selected using data through November 26, 2024. There is no refit on test data.
- Separately evaluates recursive forecasts for horizons 1–7 at common complete origins. The shorter history means inference needs 14 complete recent days for the selected model; fair-comparison evaluation still requires 30.

See [IMPROVED_RESULTS.md](IMPROVED_RESULTS.md) for the full comparison and limitations. `selection.json` records all candidates, scores, and selected epochs. The original implementation and its reproduction instructions are retained for comparison.

## Urban district dataset

The supplied CSV contains 1,555 rows dated March 20, 2021 through July 23, 2025. The date column is `Day`, in month/day/year format; the consumption target is `Daily_use`. Weather and other fields are not used in this univariate model.

Preparation preserves the source file and writes `data/urban_daily.csv` plus a detailed `data/urban_daily.audit.json`. There are 34 absent calendar dates and two duplicate dates with conflicting values. All four records on those duplicate dates are excluded. Training skips any 30-day input window or target affected by an unavailable day. It does not interpolate, compress calendar gaps, or guess date corrections. Extreme recorded values are retained and listed in the audit for review, including 32,189,800 on September 29, 2024.

Reproduce the dataset preparation, training, and forecast:

```powershell
.\.venv\Scripts\python.exe prepare_urban_data.py "path/to/Urban district water consumption.csv" --unit "cubic meters"
.\.venv\Scripts\python.exe water_cnn.py train --csv data/urban_daily.csv --allow-gaps --unit "cubic meters" --output outputs/urban
.\.venv\Scripts\python.exe water_cnn.py predict --model outputs/urban/model.pt --csv data/urban_daily.csv --days 7 --output outputs/urban/forecast.csv
```

The forecast begins July 24, 2025, immediately after the dataset's last observation. It is **not a forecast for today's date**. Provide newer observations to forecast a newer period. Evaluation and saved weights are in `outputs/urban/`. See `RESULTS.md` for the measured results and limitations.

## Use your own dataset

Provide a CSV containing **daily consumption**, in one consistent unit, for one household, building, or service area:

```csv
date,water_used
2025-01-01,620.5
2025-01-02,590.0
2025-01-03,610.2
```

Use dates in `YYYY-MM-DD` format. Provide at least 100 consecutive daily readings with the default lookback; several months or years are preferable. Rows are sorted by date. Missing days are rejected by default; `--allow-gaps` instead excludes every affected input/target window, without filling the gaps. Duplicate dates, empty values, negative values, and non-finite values are rejected. Resolve missing observations deliberately; do not replace unknown consumption with zero. Cumulative meter readings must first be converted to daily consumption, accounting for resets or rollover.

```powershell
.\.venv\Scripts\python.exe water_cnn.py train --csv data/my_water.csv --unit liters
.\.venv\Scripts\python.exe water_cnn.py predict --csv data/my_water.csv --days 7
```

For other column names, add `--date-column Date --target-column WaterUsed` to training. These names are saved with the model. `--unit` labels results; it does not convert values. Prediction uses the latest 30 observations in the supplied CSV; update that file as new measurements arrive. Use distinct `--output` directories when training different datasets, since training overwrites its output files.

## Original model and evaluation

- Input shape: `(batch, 1, 30)`; usage is the only input feature.
- Architecture: Conv1D (16 filters, kernel 3), ReLU, Conv1D (32 filters, kernel 3), ReLU, flatten, dense 32, ReLU, dense 1.
- Adam optimization with mean squared error and validation-based early stopping.
- Chronological 70% training / 15% validation / 15% test split. Normalization uses only the training period. Validation and test windows can use previously observed days, never the target or future days.
- With `--allow-gaps`, splits use the full calendar; metrics cover only complete windows and the report includes their counts. The latest 30 days must be complete to produce a future forecast.
- Test MAE and RMSE are reported in the original consumption unit, alongside last-day and last-week baselines. Lower is better; a CNN is not guaranteed to beat a simple baseline.
- The checkpoint retains the best validation epoch, without refitting on the held-out periods. Test scores evaluate rolling **one-day-ahead** forecasts, with actual prior observations available. Future multi-day forecasts feed predictions back into the model, so errors can accumulate; those test scores do not measure multi-day accuracy.
- Predictions are clipped at zero. No weather, occupancy, holidays, or prediction intervals are modeled.

Outputs in `outputs/`: `model.pt`, `metrics.json`, `test_predictions.csv`, `training_history.csv`, and (after prediction) `forecast.csv`. Synthetic demo results should never be presented as validated real-water forecasts.

## Tests

Run the automated checks from the project root:

```sh
python -m unittest -v
```

The checks cover chronological windows, missing-day handling, invalid inputs, training-only normalization, baseline initialization, saved-model equivalence, and recursive prediction behavior.

## Result charts

Charts are saved as PNG images and editable SVG files in `outputs/improved/charts/`:

- `model_results`: actual versus predicted consumption, next-day MAE comparison, and recursive forecast error by horizon.
- `seven_day_forecast`: the latest observed consumption followed by the seven predicted days.

Regenerate charts from the saved evaluation files without retraining:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-charts.txt
.\.venv\Scripts\python.exe plot_results.py
```
