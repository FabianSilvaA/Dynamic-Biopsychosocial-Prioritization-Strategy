# ============================================================
# Dynamic Biopsychosocial Prioritization Strategy
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 1. GENERAL CONFIGURATION
# ============================================================

N_PATIENTS = 1000
N_REPLICATIONS = 30
CAPACITY_PER_DAY = 10
BASE_SEED = 20260706

# Calibrated parameters 
CURRENT_WAIT_LOW = 330
CURRENT_WAIT_HIGH = 510

TMAX_BASE = 489.5
TMAX_URGENCY_EFFECT = 4.5
TMAX_COMORB_EFFECT = 7.0
TMAX_DIAG_EFFECT = 3.0

PRIORITY_WAIT_WEIGHT = 1.05
PRIORITY_URGENCY_WEIGHT = 5.0
PRIORITY_COMORB_WEIGHT = 25.0
PRIORITY_DIAG_WEIGHT = 10.0
PRIORITY_NOISE_SD = 50.0

URG_MAX = 10


# ============================================================
# 2. STATISTICAL FUNCTIONS
# ============================================================

def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return mean, ci


def fmt(mean, ci, digits=1):
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


# ============================================================
# 3. MATHEMATICAL MODEL FORMULAS
# ============================================================

# Static score:
# S_p = sum(C_i,p * W_i) + sum(V_i,p * A_i) + gamma + x

# Diagnosis and comorbidity component:
# gamma = C_a + C_d * sum(W_d)

# Age adjustment:
# x = [1 + (E_p - E_min)/(E_max - E_min)] * 9, if 18 <= E_p <= 65
# x = 10, if E_p > 65

# Vulnerability index:
# V_p(t) = (t - f_p) / T_max

# Vulnerability multiplier:
# lambda(t) =
# V_p(t) * exp(Urg / Urg_max), if V_p(t) >= 1
# V_p(t) * log10(1 + Urg) / log10(1 + Urg_max) + 1, if V_p(t) < 1

# Dynamic score:
# S'_p(t) = [sum(C'_i,p * J_i,p(t)) + sum(V'_i,p * H_i,p(t))] * lambda(t)

# Total score:
# P_p(t) = S_p + S'_p(t)


def compute_age_adjustment(age):
    return np.where(
        age >= 65,
        10,
        (1 + (age - 18) / (65 - 18)) * 9
    )


def compute_gamma(comorbidities, diagnoses, diagnosis_weight_sum=0.80):
    ca = np.select(
        [comorbidities == 0, comorbidities == 1, comorbidities == 2, comorbidities >= 3],
        [0, 1.25, 2.50, 3.75]
    )

    cd = np.select(
        [diagnoses == 1, diagnoses == 2, diagnoses >= 3],
        [1.25, 2.50, 3.75]
    )

    return ca + cd * diagnosis_weight_sum


def compute_vulnerability(current_wait, tmax):
    return current_wait / tmax


def compute_lambda(vulnerability, urgency):
    return np.where(
        vulnerability >= 1,
        vulnerability * np.exp(urgency / URG_MAX),
        vulnerability * (np.log10(1 + urgency) / np.log10(1 + URG_MAX)) + 1
    )


def compute_static_score(urgency, comorbidities, diagnoses, age):
    age_adjustment = compute_age_adjustment(age)
    gamma = compute_gamma(comorbidities, diagnoses)

    clinical_component = (
        0.35 * urgency
        + 0.45 * comorbidities
        + 0.47 * diagnoses
    )

    psychosocial_component = 3.60

    return clinical_component + psychosocial_component + gamma + age_adjustment


def compute_dynamic_score(urgency, vulnerability):
    lambda_t = compute_lambda(vulnerability, urgency)

    clinical_dynamic_component = 5.40
    psychosocial_dynamic_component = 3.20

    return (clinical_dynamic_component + psychosocial_dynamic_component) * lambda_t


def compute_total_score(static_score, dynamic_score):
    return static_score + dynamic_score


# ============================================================
# 4. DATASET GENERATION
# ============================================================

def generate_patients(seed,
                      urgency_distribution="uniform",
                      comorbidity_lambda=1.2,
                      tmax_scale=1.0):

    rng = np.random.default_rng(seed)

    age = rng.normal(50, 15, N_PATIENTS)
    age = np.clip(age, 18, 65)

    if urgency_distribution == "uniform":
        urgency = rng.uniform(1, 10, N_PATIENTS)
    elif urgency_distribution == "low":
        urgency = 1 + 9 * rng.beta(2, 5, N_PATIENTS)
    elif urgency_distribution == "high":
        urgency = 1 + 9 * rng.beta(5, 2, N_PATIENTS)
    else:
        raise ValueError("urgency_distribution must be uniform, low, or high")

    comorbidities = np.minimum(rng.poisson(comorbidity_lambda, N_PATIENTS), 3)
    diagnoses = np.minimum(rng.poisson(1.2, N_PATIENTS) + 1, 3)

    current_wait = rng.uniform(CURRENT_WAIT_LOW, CURRENT_WAIT_HIGH, N_PATIENTS)

    tmax = (
        TMAX_BASE
        - TMAX_URGENCY_EFFECT * urgency
        - TMAX_COMORB_EFFECT * comorbidities
        - TMAX_DIAG_EFFECT * diagnoses
    ) * tmax_scale

    tmax = np.clip(tmax, 180, 700)

    vulnerability = compute_vulnerability(current_wait, tmax)

    static_score = compute_static_score(
        urgency=urgency,
        comorbidities=comorbidities,
        diagnoses=diagnoses,
        age=age
    )

    dynamic_score = compute_dynamic_score(
        urgency=urgency,
        vulnerability=vulnerability
    )

    total_score = compute_total_score(static_score, dynamic_score)

    priority_score = (
        PRIORITY_WAIT_WEIGHT * current_wait
        + PRIORITY_URGENCY_WEIGHT * urgency
        + PRIORITY_COMORB_WEIGHT * comorbidities
        + PRIORITY_DIAG_WEIGHT * diagnoses
        + rng.normal(0, PRIORITY_NOISE_SD, N_PATIENTS)
    )

    return pd.DataFrame({
        "patient_id": np.arange(1, N_PATIENTS + 1),
        "age": age,
        "urgency": urgency,
        "comorbidities": comorbidities,
        "diagnoses": diagnoses,
        "current_wait": current_wait,
        "tmax": tmax,
        "vulnerability": vulnerability,
        "static_score": static_score,
        "dynamic_score": dynamic_score,
        "total_score": total_score,
        "priority_score": priority_score
    })


# ============================================================
# 5. SCHEDULING STRATEGIES
# ============================================================

def evaluate_strategy(df, strategy):

    if strategy == "FIFO":
        order = np.argsort(-df["current_wait"].to_numpy())

    elif strategy == "Prioritization":
        order = np.argsort(-df["priority_score"].to_numpy())

    else:
        raise ValueError("strategy must be FIFO or Prioritization")

    surgery_day = np.arange(len(df)) // CAPACITY_PER_DAY

    assigned_day = np.empty(len(df))
    assigned_day[order] = surgery_day

    final_wait = df["current_wait"].to_numpy() + assigned_day

    return {
        "mean_wait": np.mean(final_wait),
        "median_wait": np.median(final_wait),
        "over_tmax": 100 * np.mean(final_wait > df["tmax"].to_numpy()),
        "final_wait": final_wait
    }


def run_replication(seed,
                    urgency_distribution="uniform",
                    comorbidity_lambda=1.2,
                    tmax_scale=1.0):

    df = generate_patients(
        seed=seed,
        urgency_distribution=urgency_distribution,
        comorbidity_lambda=comorbidity_lambda,
        tmax_scale=tmax_scale
    )

    fifo = evaluate_strategy(df, "FIFO")
    prioritization = evaluate_strategy(df, "Prioritization")

    return df, fifo, prioritization


# ============================================================
# 6. TABLE IV
# ============================================================

table_iv = pd.DataFrame({
    "Patient": ["Patient 1", "Patient 2", "Patient 3", "Patient 4", "Patient 5"],
    "Pp(0)": [86.30, 158.76, 189.40, 37.23, 153.54],
    "Pp(30)": [118.35, 186.23, 234.88, 107.28, 180.90],
    "Pp(60)": [214.48, 268.66, 371.33, 418.05, 262.98],
    "Pp(90)": [374.71, 406.03, 703.05, 894.07, 399.77],
    "Pp(120)": [599.02, 598.35, 1102.54, 1560.50, 591.28],
    "Pp(150)": [887.43, 845.62, 1616.18, 2417.34, 837.51],
})

print("\nTABLE IV")
print(table_iv.to_string(index=False))


# ============================================================
# 7. TABLE V
# ============================================================

table_v = pd.DataFrame({
    "Urgency": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Static Score": [28.63, 29.48, 28.88, 28.88, 29.21, 28.82, 29.52, 29.46, 29.38, 28.59],
    "Dynamic Score": [11.34, 13.85, 13.24, 13.24, 13.92, 14.26, 15.64, 18.76, 16.33, 16.80],
    "Total Score": [39.97, 43.33, 42.12, 42.12, 43.14, 43.09, 45.16, 48.22, 45.72, 45.39],
})

print("\nTABLE V")
print(table_v.to_string(index=False))


# ============================================================
# 8. TABLE VI
# ============================================================

def run_table_vi():

    fifo_mean_wait = []
    fifo_median_wait = []
    fifo_over_tmax = []

    pri_mean_wait = []
    pri_median_wait = []
    pri_over_tmax = []

    for r in range(N_REPLICATIONS):
        _, fifo, pri = run_replication(BASE_SEED + r)

        fifo_mean_wait.append(fifo["mean_wait"])
        fifo_median_wait.append(fifo["median_wait"])
        fifo_over_tmax.append(fifo["over_tmax"])

        pri_mean_wait.append(pri["mean_wait"])
        pri_median_wait.append(pri["median_wait"])
        pri_over_tmax.append(pri["over_tmax"])

    rows = []

    variables = [
        ("Mean Wait Time", fifo_mean_wait, pri_mean_wait, "days"),
        ("Median Wait Time", fifo_median_wait, pri_median_wait, "days"),
        ("% Over Tmax", fifo_over_tmax, pri_over_tmax, "%")
    ]

    for variable, fifo_values, pri_values, unit in variables:

        fifo_mean, fifo_ci = mean_ci95(fifo_values)
        pri_mean, pri_ci = mean_ci95(pri_values)

        diff = np.asarray(fifo_values) - np.asarray(pri_values)
        diff_mean, diff_ci = mean_ci95(diff)

        ttest = stats.ttest_rel(fifo_values, pri_values)
        wilcoxon = stats.wilcoxon(fifo_values, pri_values)

        rows.append({
            "Variable": variable,
            "FIFO": fmt(fifo_mean, fifo_ci),
            "Prioritization": fmt(pri_mean, pri_ci),
            "Difference": fmt(diff_mean, diff_ci),
            "Unit": unit,
            "Paired t-test p-value": ttest.pvalue,
            "Wilcoxon p-value": wilcoxon.pvalue
        })

    return pd.DataFrame(rows)


table_vi = run_table_vi()

print("\nTABLE VI")
print(table_vi.to_string(index=False))


# ============================================================
# 9. TABLE VII - SENSITIVITY ANALYSIS
# ============================================================

def run_sensitivity_scenario(label,
                             urgency_distribution="uniform",
                             comorbidity_lambda=1.2,
                             tmax_scale=1.0):

    fifo_values = []
    pri_values = []

    for r in range(N_REPLICATIONS):
        _, fifo, pri = run_replication(
            seed=BASE_SEED + r,
            urgency_distribution=urgency_distribution,
            comorbidity_lambda=comorbidity_lambda,
            tmax_scale=tmax_scale
        )

        fifo_values.append(fifo["over_tmax"])
        pri_values.append(pri["over_tmax"])

    fifo_mean, fifo_ci = mean_ci95(fifo_values)
    pri_mean, pri_ci = mean_ci95(pri_values)

    diff = np.asarray(fifo_values) - np.asarray(pri_values)
    diff_mean, diff_ci = mean_ci95(diff)

    ttest = stats.ttest_rel(fifo_values, pri_values)

    return {
        "Scenario": label,
        "Urgency distribution": urgency_distribution,
        "Comorbidity rate": comorbidity_lambda,
        "Tmax scale": tmax_scale,
        "FIFO (% over Tmax)": fmt(fifo_mean, fifo_ci) + "%",
        "Prioritization (% over Tmax)": fmt(pri_mean, pri_ci) + "%",
        "Improvement (percentage points)": fmt(diff_mean, diff_ci),
        "Paired t-test p-value": ttest.pvalue
    }


def run_table_vii():

    scenarios = [
        ("Baseline", "uniform", 1.2, 1.00),
        ("Low urgency", "low", 1.2, 1.00),
        ("High urgency", "high", 1.2, 1.00),
        ("Low comorbidity", "uniform", 0.8, 1.00),
        ("High comorbidity", "uniform", 1.6, 1.00),
        ("0.95 x Tmax", "uniform", 1.2, 0.95),
        ("1.05 x Tmax", "uniform", 1.2, 1.05),
    ]

    rows = []

    for label, urg_dist, com_lambda, scale in scenarios:
        rows.append(
            run_sensitivity_scenario(
                label=label,
                urgency_distribution=urg_dist,
                comorbidity_lambda=com_lambda,
                tmax_scale=scale
            )
        )

    return pd.DataFrame(rows)


table_vii = run_table_vii()

print("\nTABLE VII")
print(table_vii.to_string(index=False))


# ============================================================
# 10. FIGURE 1
# ============================================================

def plot_figure_1():

    time_points = [0, 30, 60, 90, 120, 150]
    score_columns = ["Pp(0)", "Pp(30)", "Pp(60)", "Pp(90)", "Pp(120)", "Pp(150)"]

    plt.figure(figsize=(7, 5))

    for _, row in table_iv.iterrows():
        plt.plot(time_points, row[score_columns].values, marker="o", label=row["Patient"])

    plt.xlabel("Time (days)")
    plt.ylabel("Prioritization Score")
    plt.title("Evolution of Prioritization Scores for Five Patients")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_1_prioritization_evolution.png", dpi=300)
    plt.show()


plot_figure_1()


# ============================================================
# 11. FIGURE 2
# ============================================================

def plot_figure_2():

    df = generate_patients(BASE_SEED)

    plt.figure(figsize=(7, 5))
    plt.hist(df["total_score"], bins=30)

    plt.xlabel("Total Prioritization Score Pp")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Total Prioritization Scores at Day 150")
    plt.tight_layout()
    plt.savefig("figure_2_score_distribution.png", dpi=300)
    plt.show()


plot_figure_2()


# ============================================================
# 12. EXPORTS
# ============================================================

dataset_baseline = generate_patients(BASE_SEED)

dataset_baseline.to_csv("synthetic_patients_baseline.csv", index=False)
dataset_baseline.to_excel("synthetic_patients_baseline.xlsx", index=False)

table_iv.to_csv("table_iv.csv", index=False)
table_iv.to_excel("table_iv.xlsx", index=False)

table_v.to_csv("table_v.csv", index=False)
table_v.to_excel("table_v.xlsx", index=False)

table_vi.to_csv("table_vi.csv", index=False)
table_vi.to_excel("table_vi.xlsx", index=False)

table_vii.to_csv("table_vii_sensitivity_analysis.csv", index=False)
table_vii.to_excel("table_vii_sensitivity_analysis.xlsx", index=False)

print("\nEXPORTS COMPLETED")
print("synthetic_patients_baseline.csv")
print("synthetic_patients_baseline.xlsx")
print("table_iv.csv")
print("table_iv.xlsx")
print("table_v.csv")
print("table_v.xlsx")
print("table_vi.csv")
print("table_vi.xlsx")
print("table_vii_sensitivity_analysis.csv")
print("table_vii_sensitivity_analysis.xlsx")
print("figure_1_prioritization_evolution.png")
print("figure_2_score_distribution.png")
