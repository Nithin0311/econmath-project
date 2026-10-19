# EconMath Student Performance Analysis

A Python project for exploring student performance data with descriptive
statistics, visualizations, multivariate analysis, time-series methods, and
clustering.

## Files

- `econmath.csv` — dataset used by the analysis.
- `econmath_project_small(2).py` — analysis script.

## Requirements

Python 3 and the following packages:

```text
matplotlib
numpy
pandas
scipy
seaborn
```

Install them with:

```bash
python -m pip install matplotlib numpy pandas scipy seaborn
```

## Run

Run every section:

```bash
python "econmath_project_small(2).py"
```

Run selected sections by number (for example, sections 1, 3, and 5):

```bash
python "econmath_project_small(2).py" 1 3 5
```

Use `--show` to display plots interactively, or `--pause` to pause between
sections:

```bash
python "econmath_project_small(2).py" --show
```

The script looks for `econmath.csv` in the project folder or its `data`
subfolder. Generated figures and tables are saved under `outputs`.
