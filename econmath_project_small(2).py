"""
EconMath Project - Student Performance Analysis Using Python

Usage:
    python econmath_project_small.py            (run all sections)
    python econmath_project_small.py 1
    python econmath_project_small.py 3 5
    python econmath_project_small.py --show
    python econmath_project_small.py --pause

Sections:
    1. Dataset
    2. Experiment 4
    3. CO2 - Cleaning and Univariate Visualisation
    4. Lab 4 - Bivariate Analysis
    5. Lab 5 - Multivariate Analysis
    6. 1D / 2D / ND Analysis
    7. Time Series
    8. Clustering
"""

import sys
import warnings
from pathlib import Path

import matplotlib

SHOW = "--show" in sys.argv
PAUSE = "--pause" in sys.argv

if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy

warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

NUM = [
    "age", "work", "study", "colgpa", "hsgpa",
    "acteng", "actmth", "act", "mathscr", "score"
]

ATT = ["Excellent", "Good", "Fair/Poor"]
GEN = ["#f1948a", "#2e86c1"]


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def q(n, text):
    print(f"\n{'=' * 80}\nQ{n}. {text}\n{'-' * 80}")


def banner(title):
    print("\n" + "#" * 80)
    print("#  " + title)
    print("#" * 80)


def save(name, folder, title=None, xl=None, yl=None, size=None):
    """Save the current figure and optionally display it."""

    if size:
        plt.gcf().set_size_inches(*size)

    if title:
        plt.title(title)

    if xl is not None:
        plt.xlabel(xl)

    if yl is not None:
        plt.ylabel(yl)

    output_folder = FIG / folder
    output_folder.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    output_file = output_folder / f"{name}.png"
    plt.savefig(output_file, dpi=130, bbox_inches="tight")

    print(f"   [figure saved] {output_file.relative_to(ROOT)}")

    if SHOW:
        plt.show()

    plt.close("all")


def save_table(df, name):
    """Save a table as CSV in outputs/tables."""
    path = TAB / f"{name}.csv"
    df.to_csv(path)
    print(f"   [table saved] {path.relative_to(ROOT)}")


def find_dataset():
    """Look for the CSV in the data folder, next to the script, or the current folder."""

    folders = [DATA_DIR, ROOT, Path.cwd(), Path.cwd() / "data"]
    names = ["econmath.csv", "econmath (1).csv", "econmath.CSV"]

    for folder in folders:
        for name in names:
            candidate = folder / name
            if candidate.exists():
                return candidate

    # Last resort: any CSV with "econmath" in its name
    for folder in folders:
        if folder.exists():
            matches = sorted(folder.glob("*econmath*.csv"))
            if matches:
                return matches[0]

    return None


def load_raw():
    """Load the CSV (searches several locations)."""

    file_path = find_dataset()

    if file_path is None:
        raise FileNotFoundError(
            "\nDataset not found!\n"
            "Put econmath.csv in ANY of these places:\n"
            f"  1. {DATA_DIR}\n"
            f"  2. {ROOT}   (same folder as this script)\n"
            f"  3. {Path.cwd()}   (current folder)\n"
        )

    print(f"[dataset] {file_path}")
    return pd.read_csv(file_path)


def add_labels(df):
    """Create readable categories from the dataset variables."""

    df = df.copy()

    df["gender"] = df["male"].map({1: "Male", 0: "Female"})
    df["calculus_taken"] = df["calculus"].map({1: "Yes", 0: "No"})
    df["econ_hs"] = df["econhs"].map({1: "Yes", 0: "No"})

    df["attitude"] = np.select(
        [df["attexc"] == 1, df["attgood"] == 1],
        ["Excellent", "Good"],
        default="Fair/Poor"
    )

    df["parent_college"] = (
        df["fathcoll"] + df["mothcoll"]
    ).map({0: "Neither", 1: "One parent", 2: "Both parents"})

    df["act_level"] = pd.cut(
        df["act"],
        bins=[0, 20, 25, 40],
        labels=["Low (<=20)", "Medium (21-25)", "High (>25)"]
    ).astype("object")

    df["score_band"] = pd.qcut(
        df["score"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop"
    ).astype("object")

    df["work_group"] = pd.cut(
        df["work"],
        bins=[-1, 0, 10, 100],
        labels=["No job", "Part-time (<=10h)", "Heavy (>10h)"]
    ).astype("object")

    return df


def load_clean():
    """Load labelled data and replace missing ACT values with medians."""

    df = add_labels(load_raw())

    for column in ["acteng", "actmth", "act"]:
        df[column] = df[column].fillna(df[column].median())

    df["act_level"] = pd.cut(
        df["act"],
        bins=[0, 20, 25, 40],
        labels=["Low (<=20)", "Medium (21-25)", "High (>25)"]
    ).astype("object")

    return df


def show_table(title, data):
    print(f"\n{title}")
    print(data)


# ------------------------------------------------------------------
# Section 1: Dataset
# ------------------------------------------------------------------

def s1_dataset():
    banner("SECTION 1 - DATASET")

    raw = load_raw()

    print("Dataset shape:", raw.shape)
    print("Number of rows:", raw.shape[0])
    print("Number of columns:", raw.shape[1])
    print("\nColumn names:")
    print(list(raw.columns))

    labelled = add_labels(raw)

    print("\nFirst eight records with labelled variables:")
    columns = [
        "male", "gender", "calculus", "calculus_taken",
        "attexc", "attgood", "attitude", "score", "score_band"
    ]

    print(labelled[columns].head(8))


# ------------------------------------------------------------------
# Section 2: Experiment 4
# ------------------------------------------------------------------

def s2_exp4():
    banner("SECTION 2 - EXPERIMENT 4: IMPORT, TRANSFORM, BASIC STATISTICS")

    df = load_raw()

    q(1, "Import the dataset")
    print("Loaded econmath.csv")

    q(2, "First 10 records")
    print(df.head(10))

    q(3, "Last 10 records")
    print(df.tail(10))

    q(4, "Rows and columns")
    print("Rows:", df.shape[0], "Columns:", df.shape[1])

    q(5, "Variable names")
    print(list(df.columns))

    q(6, "Data types")
    df.info()

    q(7, "Summary statistics")
    print(df.describe(include="all").T)

    q(8, "Missing values per column")
    print(df.isna().sum())

    q(9, "Total missing values")
    print(int(df.isna().sum().sum()))

    q(10, "Columns with missing values and percentages")
    missing = df.isna().sum()
    percentage = missing / len(df) * 100

    print(pd.DataFrame({
        "missing": missing,
        "percent": percentage.round(2)
    })[missing > 0])

    q(11, "Rows with at least one missing value")
    missing_rows = df[df.isna().any(axis=1)]
    print("Number of rows:", len(missing_rows))
    print(missing_rows.head(10))

    q(12, "Remove rows with missing values")
    complete = df.dropna()
    print(f"Rows: {len(df)} -> {len(complete)}")
    print(complete.head())

    q(13, "Fill missing acteng values with the mean")
    d13 = df.copy()
    mean_value = d13["acteng"].mean()
    d13["acteng"] = d13["acteng"].fillna(mean_value)
    print("Mean:", round(mean_value, 4))
    print("Missing left:", d13["acteng"].isna().sum())

    q(14, "Fill missing actmth values with the median")
    median_value = d13["actmth"].median()
    d13["actmth"] = d13["actmth"].fillna(median_value)
    print("Median:", median_value)
    print("Missing left:", d13["actmth"].isna().sum())

    q(15, "Fill missing categorical act_level values with the mode")
    labelled = add_labels(df)
    mode_values = labelled["act_level"].mode()

    if not mode_values.empty:
        mode_value = mode_values.iloc[0]
        labelled["act_level"] = labelled["act_level"].fillna(mode_value)
        print("Mode:", mode_value)

    print(labelled["act_level"].value_counts(dropna=False))

    q(16, "Forward fill: equivalent to na.locf()")
    print("Missing after forward fill:", df["act"].ffill().isna().sum())

    q(17, "Backward fill: equivalent to na.locf(fromLast=TRUE)")
    print("Missing after backward fill:", df["act"].bfill().isna().sum())

    q(18, "Constant fill with zero")
    filled = df.fillna(0)
    print("Missing before:", int(df.isna().sum().sum()))
    print("Missing after:", int(filled.isna().sum().sum()))

    q(19, "Duplicate records")
    print("Duplicate rows:", df.duplicated().sum())

    if "rownames" in df.columns:
        print(
            "Duplicates ignoring ID:",
            df.drop(columns="rownames").duplicated().sum()
        )

    print(f"Rows: {len(df)} -> {len(df.drop_duplicates())}")

    q(20, "Convert variables to categorical type")
    d20 = add_labels(d13)

    categorical_columns = [
        "econhs", "male", "calculus", "attexc", "attgood",
        "fathcoll", "mothcoll", "gender", "calculus_taken",
        "econ_hs", "attitude", "parent_college", "act_level"
    ]

    for column in categorical_columns:
        if column in d20.columns:
            d20[column] = d20[column].astype("category")

    print(d20.dtypes[d20.dtypes == "category"])

    q(21, "Rename columns")
    d21 = d20.rename(columns={
        "colgpa": "college_gpa",
        "hsgpa": "highschool_gpa",
        "mathscr": "math_score"
    })

    print(list(d21.columns))

    q(22, "Create derived columns")
    d21["weekly_load_hrs"] = d21["work"] + d21["study"]
    d21["gpa_change"] = d21["college_gpa"] - d21["highschool_gpa"]

    print(d21[[
        "work", "study", "weekly_load_hrs",
        "highschool_gpa", "college_gpa", "gpa_change"
    ]].head(10))

    num = d13[NUM]

    q(23, "Descriptive statistics")
    print(d13.describe().T)

    q(24, "Minimum and maximum")
    print(pd.DataFrame({"min": num.min(), "max": num.max()}))

    q(25, "Range")
    print((num.max() - num.min()).round(3))

    q(26, "Variance")
    print(num.var().round(4))

    q(27, "Standard deviation")
    print(num.std().round(4))

    q(28, "Interquartile range")
    print(pd.DataFrame({
        "Q1": num.quantile(0.25),
        "Q3": num.quantile(0.75),
        "IQR": num.quantile(0.75) - num.quantile(0.25)
    }).round(4))


# ------------------------------------------------------------------
# Section 3: CO2 - Cleaning and Univariate Visualisation
# ------------------------------------------------------------------

def s3_co2():
    banner("SECTION 3 - CO2: CLEANING AND UNIVARIATE VISUALISATION")
    F = "co2"

    raw = add_labels(load_raw())

    q(1, "Import dataset")
    print("Shape:", raw.shape)

    q(2, "First and last six records")
    print(raw.head(6))
    print(raw.tail(6))

    q(3, "Total missing values")
    print("Total:", int(raw.isna().sum().sum()))
    print(raw.isna().sum()[raw.isna().sum() > 0])

    q(4, "Records with missing values")
    missing = raw[raw.isna().any(axis=1)]
    print("Number of records:", len(missing))

    show_columns = [
        c for c in ["rownames", "acteng", "actmth", "act", "act_level"]
        if c in missing.columns
    ]
    print(missing[show_columns].head(8))

    q(5, "Remove columns with more than 70% missing values")
    pct = raw.isna().mean() * 100
    drop_columns = pct[pct > 70].index.tolist()

    if drop_columns:
        print("Dropped columns:", drop_columns)
    else:
        print(f"No columns dropped. Maximum missing percentage: {pct.max():.2f}%")

    df = raw.drop(columns=drop_columns)

    q(6, "Numerical missing values: mean imputation")
    numerical_missing = [
        c for c in df.select_dtypes(include="number")
        if df[c].isna().any()
    ]

    print("Columns:", numerical_missing)

    mean_filled = df.copy()
    for column in numerical_missing:
        mean_filled[column] = mean_filled[column].fillna(
            mean_filled[column].mean()
        )

    print("Missing values after mean imputation:",
          int(mean_filled.isna().sum().sum()))

    q(7, "Numerical missing values: median imputation")
    print(pd.DataFrame({
        "mean": df[numerical_missing].mean(),
        "median": df[numerical_missing].median()
    }).round(3))

    for column in numerical_missing:
        df[column] = df[column].fillna(df[column].median())

    q(8, "Categorical missing values: mode imputation")
    for column in df.select_dtypes(include=["object", "string", "category"]):
        modes = df[column].mode(dropna=True)
        if df[column].isna().any() and not modes.empty:
            df[column] = df[column].fillna(modes.iloc[0])
            print("Filled:", column)

    print("Missing values now:", int(df.isna().sum().sum()))

    q(9, "Duplicate rows")
    key_columns = [c for c in df.columns if c != "rownames"]
    print("Duplicates ignoring ID:", df.duplicated(subset=key_columns).sum())

    q(10, "Remove duplicate rows")
    old_length = len(df)
    df = df.drop_duplicates(subset=key_columns)
    print(f"Rows: {old_length} -> {len(df)}")

    q(11, "Check incorrect data types")
    binary = [
        c for c in df.select_dtypes(include="number")
        if set(df[c].dropna().unique()).issubset({0, 1})
    ]

    print("Binary columns stored as numbers:", binary)
    print("'rownames', if present, is an ID column.")

    q(12, "Convert variables to categories")
    for column in binary:
        df[column] = df[column].astype("category")

    for column in ["gender", "calculus_taken", "econ_hs",
                   "attitude", "parent_college", "act_level",
                   "score_band", "work_group"]:
        if column in df.columns:
            df[column] = df[column].astype("category")

    print(df.dtypes[df.dtypes == "category"])

    q(13, "Lowercase and uppercase")
    if "attitude" in df.columns:
        attitude = df["attitude"].astype("string")
        print("Original:", list(attitude.unique()))
        print("Lowercase:", list(attitude.str.lower().unique()))
        print("Uppercase:", list(attitude.str.upper().unique()))

    q(14, "Remove extra whitespace from text columns")
    for column in ["gender", "attitude", "parent_college"]:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip().astype("category")
    print("Whitespace stripped from text columns.")

    # Plain numeric copy (no categories) for plotting and statistics
    clean = load_clean()

    q(15, "Outlier detection with the IQR rule")
    rows = []
    for column in NUM:
        q1, q3 = clean[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((clean[column] < low) | (clean[column] > high)).sum())
        rows.append([column, round(low, 3), round(high, 3), n_out])
    outliers = pd.DataFrame(
        rows, columns=["variable", "lower", "upper", "n_outliers"]
    ).set_index("variable")
    print(outliers)
    save_table(outliers, "co2_outliers")

    q(16, "Cap outliers (winsorise) on the work variable")
    q1, q3 = clean["work"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    capped = clean["work"].clip(upper=upper)
    print(f"Upper limit: {upper:.2f}; max before: {clean['work'].max()}, "
          f"max after: {capped.max()}")

    q(17, "Histogram of college GPA")
    plt.figure(figsize=(7, 4.5))
    sns.histplot(clean["colgpa"], bins=20, kde=True, color="#2e86c1")
    save("hist_colgpa", F, "Distribution of College GPA", "College GPA", "Count")

    q(18, "Histogram of score")
    plt.figure(figsize=(7, 4.5))
    sns.histplot(clean["score"], bins=20, kde=True, color="#27ae60")
    save("hist_score", F, "Distribution of Course Score", "Score", "Count")

    q(19, "Histograms of all numeric variables")
    clean[NUM].hist(bins=15, figsize=(14, 9), color="#5dade2", edgecolor="white")
    save("hist_all", F)

    q(20, "Boxplot of numeric variables")
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=clean[["colgpa", "hsgpa", "work", "study", "age"]])
    save("box_numeric", F, "Boxplots of Selected Numeric Variables")

    q(21, "Boxplot of score")
    plt.figure(figsize=(5, 5))
    sns.boxplot(y=clean["score"], color="#f5b041")
    save("box_score", F, "Boxplot of Score", yl="Score")

    q(22, "Bar chart of gender")
    plt.figure(figsize=(5, 4.5))
    counts = clean["gender"].value_counts().reindex(["Female", "Male"])
    plt.bar(counts.index, counts.values, color=GEN)
    save("bar_gender", F, "Gender Distribution", "Gender", "Count")

    q(23, "Bar chart of attitude")
    plt.figure(figsize=(6, 4.5))
    counts = clean["attitude"].value_counts().reindex(ATT).fillna(0)
    plt.bar(counts.index, counts.values, color=["#58d68d", "#5dade2", "#f0b27a"])
    save("bar_attitude", F, "Attitude Towards Economics", "Attitude", "Count")

    q(24, "Pie chart of parents' college education")
    plt.figure(figsize=(5.5, 5.5))
    counts = clean["parent_college"].value_counts()
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    save("pie_parent", F, "Parents' College Education")

    q(25, "Count plots for calculus and economics in high school")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(data=clean, x="calculus_taken", order=["No", "Yes"],
                  color="#5dade2", ax=axes[0])
    axes[0].set_title("Calculus Taken")
    sns.countplot(data=clean, x="econ_hs", order=["No", "Yes"],
                  color="#58d68d", ax=axes[1])
    axes[1].set_title("Economics in High School")
    save("count_calc_econ", F)

    q(26, "Frequency table of work groups")
    print(clean["work_group"].value_counts())


# ------------------------------------------------------------------
# Section 4: Lab 4 - Bivariate Analysis
# ------------------------------------------------------------------

def s4_lab4():
    banner("SECTION 4 - LAB 4: BIVARIATE ANALYSIS")
    F = "lab4"
    df = load_clean()

    q(1, "Scatter plot: high school GPA vs college GPA")
    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(data=df, x="hsgpa", y="colgpa", alpha=0.6)
    save("scatter_hs_col", F, "HS GPA vs College GPA", "HS GPA", "College GPA")

    q(2, "Regression plot: ACT vs score")
    plt.figure(figsize=(6.5, 5))
    sns.regplot(data=df, x="act", y="score",
                scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
    save("reg_act_score", F, "ACT vs Course Score", "ACT", "Score")

    q(3, "Pearson correlations with score")
    corr = df[NUM].corr()
    print(corr["score"].sort_values(ascending=False).round(3))

    q(4, "Correlation test: colgpa vs score (Pearson and Spearman)")
    r, p = stats.pearsonr(df["colgpa"], df["score"])
    rho, p2 = stats.spearmanr(df["colgpa"], df["score"])
    print(f"Pearson r = {r:.4f}, p = {p:.4g}")
    print(f"Spearman rho = {rho:.4f}, p = {p2:.4g}")

    q(5, "Score by gender: summary and boxplot")
    print(df.groupby("gender")["score"].agg(["count", "mean", "std", "median"]).round(3))
    plt.figure(figsize=(5.5, 5))
    sns.boxplot(data=df, x="gender", y="score", order=["Female", "Male"])
    save("box_score_gender", F, "Score by Gender", "Gender", "Score")

    q(6, "Independent t-test: score by gender")
    male = df.loc[df["gender"] == "Male", "score"]
    female = df.loc[df["gender"] == "Female", "score"]
    t, p = stats.ttest_ind(male, female, equal_var=False)
    print(f"Welch t = {t:.4f}, p = {p:.4g}")

    q(7, "Score by calculus: t-test")
    yes = df.loc[df["calculus_taken"] == "Yes", "score"]
    no = df.loc[df["calculus_taken"] == "No", "score"]
    t, p = stats.ttest_ind(yes, no, equal_var=False)
    print(f"Mean (calculus) = {yes.mean():.3f}, mean (no calculus) = {no.mean():.3f}")
    print(f"Welch t = {t:.4f}, p = {p:.4g}")

    q(8, "One-way ANOVA: score by attitude")
    groups = [df.loc[df["attitude"] == a, "score"] for a in ATT]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        f, p = stats.f_oneway(*groups)
        print(f"F = {f:.4f}, p = {p:.4g}")
    plt.figure(figsize=(6.5, 5))
    sns.boxplot(data=df, x="attitude", y="score", order=ATT)
    save("box_score_attitude", F, "Score by Attitude", "Attitude", "Score")

    q(9, "Cross-tabulation: gender vs attitude")
    ct = pd.crosstab(df["gender"], df["attitude"])
    print(ct)
    print(pd.crosstab(df["gender"], df["attitude"], normalize="index").round(3))

    q(10, "Chi-square test of independence: gender vs attitude")
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"chi2 = {chi2:.4f}, dof = {dof}, p = {p:.4g}")

    q(11, "Stacked bar chart: attitude by gender")
    prop = pd.crosstab(df["gender"], df["attitude"], normalize="index")
    prop = prop.reindex(columns=[a for a in ATT if a in prop.columns])
    prop.plot(kind="bar", stacked=True, figsize=(6.5, 5), rot=0)
    save("stacked_att_gender", F, "Attitude by Gender", "Gender", "Proportion")

    q(12, "Mean score by work group")
    print(df.groupby("work_group")["score"].mean().round(3))
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="work_group", y="score",
                order=["No job", "Part-time (<=10h)", "Heavy (>10h)"],
                color="#5dade2")
    save("bar_score_work", F, "Mean Score by Work Group", "Work group", "Mean score")

    q(13, "Covariance of study hours and score")
    print(round(df["study"].cov(df["score"]), 4))


# ------------------------------------------------------------------
# Section 5: Lab 5 - Multivariate Analysis
# ------------------------------------------------------------------

def ols(X, y, names):
    """Simple OLS using numpy; returns a coefficient table and R-squared."""
    X = np.column_stack([np.ones(len(X)), X])
    names = ["intercept"] + list(names)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    table = pd.DataFrame(
        {"coef": beta, "std_err": se, "t": t, "p_value": p}, index=names
    )
    return table.round(4), r2


def s5_lab5():
    banner("SECTION 5 - LAB 5: MULTIVARIATE ANALYSIS")
    F = "lab5"
    df = load_clean()

    q(1, "Correlation matrix")
    corr = df[NUM].corr().round(3)
    print(corr)
    save_table(corr, "lab5_correlation")

    q(2, "Correlation heatmap")
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    save("heatmap_corr", F, "Correlation Heatmap")

    q(3, "Pair plot of key variables")
    sns.pairplot(df[["colgpa", "hsgpa", "act", "study", "score"]],
                 corner=True, plot_kws={"alpha": 0.5, "s": 20})
    save("pairplot", F)

    q(4, "Scatter: colgpa vs score coloured by gender")
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="colgpa", y="score", hue="gender", alpha=0.7)
    save("scatter_gender", F, "College GPA vs Score by Gender", "College GPA", "Score")

    q(5, "Scatter: study vs score coloured by attitude")
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="study", y="score", hue="attitude",
                    hue_order=ATT, alpha=0.7)
    save("scatter_attitude", F, "Study Hours vs Score by Attitude", "Study", "Score")

    q(6, "Facet grid: score vs colgpa by calculus")
    g = sns.FacetGrid(df, col="calculus_taken", height=4, aspect=1.1)
    g.map_dataframe(sns.scatterplot, x="colgpa", y="score", alpha=0.6)
    save("facet_calculus", F)

    q(7, "Pivot table: mean score by gender and calculus")
    pivot = df.pivot_table(values="score", index="gender",
                           columns="calculus_taken", aggfunc="mean").round(3)
    print(pivot)
    save_table(pivot, "lab5_pivot")

    q(8, "Heatmap of the pivot table")
    plt.figure(figsize=(5.5, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    save("heatmap_pivot", F, "Mean Score by Gender and Calculus")

    q(9, "Group means by parental education and attitude")
    print(df.groupby(["parent_college", "attitude"])["score"].mean().round(3).unstack())

    q(10, "Simple regression: score ~ colgpa")
    t1, r2 = ols(df[["colgpa"]].values, df["score"].values, ["colgpa"])
    print(t1)
    print("R-squared:", round(r2, 4))

    q(11, "Multiple regression: score ~ colgpa + hsgpa + act + study + work + male + calculus")
    X = df[["colgpa", "hsgpa", "act", "study", "work", "male", "calculus"]]
    t2, r2 = ols(X.values.astype(float), df["score"].values, X.columns)
    print(t2)
    print("R-squared:", round(r2, 4))
    save_table(t2, "lab5_regression")

    q(12, "Residual plot of the multiple regression")
    Xm = np.column_stack([np.ones(len(X)), X.values.astype(float)])
    beta, *_ = np.linalg.lstsq(Xm, df["score"].values, rcond=None)
    fitted = Xm @ beta
    resid = df["score"].values - fitted
    plt.figure(figsize=(6.5, 5))
    plt.scatter(fitted, resid, alpha=0.5)
    plt.axhline(0, color="red")
    save("residuals", F, "Residuals vs Fitted", "Fitted", "Residual")


# ------------------------------------------------------------------
# Section 6: 1D / 2D / ND Analysis
# ------------------------------------------------------------------

def s6_nd():
    banner("SECTION 6 - 1D / 2D / ND ANALYSIS")
    F = "nd"
    df = load_clean()

    q(1, "1D analysis of score: summary, skewness, kurtosis")
    s = df["score"]
    print(s.describe().round(3))
    print("Skewness:", round(stats.skew(s), 4))
    print("Kurtosis:", round(stats.kurtosis(s), 4))

    q(2, "1D normality test (Shapiro-Wilk) and Q-Q plot")
    sample = s.sample(min(len(s), 5000), random_state=1)
    w, p = stats.shapiro(sample)
    print(f"W = {w:.4f}, p = {p:.4g}")
    plt.figure(figsize=(5.5, 5))
    stats.probplot(s, dist="norm", plot=plt)
    save("qq_score", F, "Q-Q Plot of Score")

    q(3, "1D confidence interval for mean score")
    mean, sem = s.mean(), stats.sem(s)
    lo, hi = stats.t.interval(0.95, df=len(s) - 1, loc=mean, scale=sem)
    print(f"Mean = {mean:.3f}, 95% CI = ({lo:.3f}, {hi:.3f})")

    q(4, "2D analysis: colgpa vs score")
    print(df[["colgpa", "score"]].corr().round(4))
    plt.figure(figsize=(6.5, 5))
    sns.kdeplot(data=df, x="colgpa", y="score", fill=True, cmap="Blues")
    save("kde_2d", F, "2D Density: College GPA vs Score", "College GPA", "Score")

    q(5, "2D joint plot")
    g = sns.jointplot(data=df, x="hsgpa", y="colgpa", kind="reg")
    g.figure.set_size_inches(6.5, 6)
    save("joint_hs_col", F)

    q(6, "ND analysis: covariance matrix")
    cols = ["colgpa", "hsgpa", "act", "study", "score"]
    print(df[cols].cov().round(4))

    q(7, "ND analysis: standardised data and PCA (via SVD)")
    Z = (df[cols] - df[cols].mean()) / df[cols].std()
    U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
    explained = S ** 2 / np.sum(S ** 2)
    print(pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(explained))],
        "explained_variance_ratio": explained.round(4),
        "cumulative": np.cumsum(explained).round(4)
    }).set_index("component"))
    loadings = pd.DataFrame(
        Vt.T, index=cols, columns=[f"PC{i + 1}" for i in range(len(cols))]
    ).round(3)
    print("\nLoadings:")
    print(loadings)

    q(8, "Scree plot")
    plt.figure(figsize=(6, 4.5))
    plt.plot(range(1, len(explained) + 1), explained, "o-")
    plt.xticks(range(1, len(explained) + 1))
    save("scree", F, "Scree Plot", "Component", "Explained variance ratio")

    q(9, "PCA scatter coloured by gender")
    scores = Z.values @ Vt.T
    plt.figure(figsize=(6.5, 5))
    for g_name, colour in zip(["Female", "Male"], GEN):
        mask = (df["gender"] == g_name).values
        plt.scatter(scores[mask, 0], scores[mask, 1], alpha=0.6,
                    label=g_name, color=colour)
    plt.legend()
    save("pca_scatter", F, "PCA: PC1 vs PC2", "PC1", "PC2")


# ------------------------------------------------------------------
# Section 7: Time Series
# ------------------------------------------------------------------

def s7_ts():
    banner("SECTION 7 - TIME SERIES")
    F = "ts"
    df = load_clean()

    print("NOTE: econmath is cross-sectional (no date column).")
    print("Observation order is used as the time index for demonstration.")

    n = len(df)
    index = pd.date_range("2000-01-01", periods=n, freq="D")
    ts = pd.Series(df["score"].values, index=index, name="score")

    q(1, "Time series plot")
    plt.figure(figsize=(11, 4.5))
    ts.plot(alpha=0.6)
    save("ts_plot", F, "Score in Observation Order", "Time index", "Score")

    q(2, "Rolling mean and standard deviation")
    window = max(5, n // 20)
    roll_mean = ts.rolling(window).mean()
    roll_std = ts.rolling(window).std()
    print(f"Window = {window}")
    print(pd.DataFrame({"rolling_mean": roll_mean, "rolling_std": roll_std}).dropna().head())
    plt.figure(figsize=(11, 4.5))
    ts.plot(alpha=0.35, label="Score")
    roll_mean.plot(label="Rolling mean", color="red")
    plt.legend()
    save("ts_rolling", F, "Rolling Mean", "Time index", "Score")

    q(3, "Expanding mean")
    print(ts.expanding().mean().tail().round(4))

    q(4, "First difference")
    diff = ts.diff().dropna()
    print(diff.describe().round(4))
    plt.figure(figsize=(11, 4.5))
    diff.plot()
    save("ts_diff", F, "First Difference of Score", "Time index", "Change")

    q(5, "Autocorrelation function")
    max_lag = min(30, n - 2)
    acf = [ts.autocorr(lag=k) for k in range(1, max_lag + 1)]
    acf_table = pd.Series(acf, index=range(1, max_lag + 1), name="acf").round(4)
    print(acf_table.head(10))
    plt.figure(figsize=(9, 4.5))
    plt.bar(acf_table.index, acf_table.values, color="#5dade2")
    bound = 1.96 / np.sqrt(n)
    plt.axhline(bound, color="red", linestyle="--")
    plt.axhline(-bound, color="red", linestyle="--")
    save("ts_acf", F, "Autocorrelation Function", "Lag", "ACF")

    q(6, "Linear trend")
    x = np.arange(n)
    slope, intercept, r, p, se = stats.linregress(x, ts.values)
    print(f"Slope = {slope:.6f}, intercept = {intercept:.4f}, "
          f"R^2 = {r ** 2:.4f}, p = {p:.4g}")

    q(7, "Exponential smoothing (alpha = 0.2)")
    smooth = ts.ewm(alpha=0.2, adjust=False).mean()
    print(smooth.head().round(4))
    plt.figure(figsize=(11, 4.5))
    ts.plot(alpha=0.35, label="Score")
    smooth.plot(color="green", label="EWMA")
    plt.legend()
    save("ts_ewma", F, "Exponential Smoothing", "Time index", "Score")


# ------------------------------------------------------------------
# Section 8: Clustering
# ------------------------------------------------------------------

def kmeans(X, k, n_iter=100, seed=0):
    """Plain numpy k-means. Returns labels, centres and inertia."""
    rng = np.random.default_rng(seed)
    centres = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)

    for _ in range(n_iter):
        dist = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        new_centres = centres.copy()
        for j in range(k):
            if np.any(new_labels == j):
                new_centres[j] = X[new_labels == j].mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centres, centres):
            break
        labels, centres = new_labels, new_centres

    dist = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    labels = dist.argmin(axis=1)
    inertia = dist[np.arange(len(X)), labels].sum()
    return labels, centres, inertia


def silhouette(X, labels):
    """Mean silhouette score (numpy)."""
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        a = D[i, same].sum() / max(same.sum() - 1, 1)
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = 0 if same.sum() == 1 else (b - a) / max(a, b)
    return s.mean()


def s8_cluster():
    banner("SECTION 8 - CLUSTERING")
    F = "cluster"
    df = load_clean()
    cols = ["colgpa", "hsgpa", "act", "study", "work", "score"]
    Z = ((df[cols] - df[cols].mean()) / df[cols].std()).values

    q(1, "Elbow method")
    ks = list(range(1, 9))
    inertias = [min(kmeans(Z, k, seed=s)[2] for s in range(5)) for k in ks]
    print(pd.DataFrame({"k": ks, "inertia": np.round(inertias, 2)}).set_index("k"))
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(ks, inertias, "o-")
    save("elbow", F, "Elbow Plot", "k", "Inertia")

    q(2, "Silhouette scores")
    sil = {}
    for k in range(2, 7):
        lab, _, _ = kmeans(Z, k, seed=0)
        sil[k] = silhouette(Z, lab)
    print(pd.Series(sil, name="silhouette").round(4))

    q(3, "K-means with k = 3")
    labels, centres, _ = kmeans(Z, 3, seed=0)
    df["cluster"] = labels
    print(df["cluster"].value_counts().sort_index())

    q(4, "Cluster profiles (means of original variables)")
    profile = df.groupby("cluster")[cols].mean().round(3)
    print(profile)
    save_table(profile, "cluster_profile")

    q(5, "Cluster scatter on first two principal components")
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pcs = Z @ Vt.T[:, :2]
    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=labels,
                    palette="deep", alpha=0.7)
    save("cluster_pca", F, "K-means Clusters (PCA view)", "PC1", "PC2")

    q(6, "Score by cluster")
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="cluster", y="score")
    save("cluster_score", F, "Score by Cluster", "Cluster", "Score")

    q(7, "Hierarchical clustering (Ward) dendrogram")
    sample_n = min(len(Z), 60)
    idx = np.random.default_rng(1).choice(len(Z), size=sample_n, replace=False)
    link = hierarchy.linkage(Z[idx], method="ward")
    plt.figure(figsize=(12, 5))
    hierarchy.dendrogram(link, no_labels=True)
    save("dendrogram", F, f"Ward Dendrogram (random sample of {sample_n})")

    hc = hierarchy.fcluster(hierarchy.linkage(Z, method="ward"), t=3, criterion="maxclust")
    print("Hierarchical cluster sizes:")
    print(pd.Series(hc).value_counts().sort_index())


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

SECTIONS = {
    1: s1_dataset,
    2: s2_exp4,
    3: s3_co2,
    4: s4_lab4,
    5: s5_lab5,
    6: s6_nd,
    7: s7_ts,
    8: s8_cluster,
}


def main():
    requested = [int(a) for a in sys.argv[1:] if a.isdigit()]
    chosen = requested if requested else list(SECTIONS)

    invalid = [n for n in chosen if n not in SECTIONS]
    if invalid:
        print(f"Invalid section number(s): {invalid}. Choose from 1-8.")
        sys.exit(1)

    for n in chosen:
        SECTIONS[n]()
        if PAUSE:
            input("\nPress Enter to continue...")

    print("\nDone. Figures are in outputs/figures, tables in outputs/tables.")


if __name__ == "__main__":
    main()
