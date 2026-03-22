import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "data/segment_alerts_all_airports_train.csv"
OUTPUT_DIR = Path("catboost_global_local_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

USE_PISE_2016 = False
RANDOM_SEED = 42

TRAIN_END_YEAR = 2020
VALID_YEAR = 2021
TEST_YEAR = 2022

GLOBAL_WEIGHT = 0.7
LOCAL_WEIGHT = 0.3

MIN_LOCAL_TRAIN_ROWS = 500
MIN_LOCAL_POSITIVES = 20

# ============================================================
# UTILS
# ============================================================
def month_to_season(m):
    if m in [12, 1, 2]:
        return "winter"
    if m in [3, 4, 5]:
        return "spring"
    if m in [6, 7, 8]:
        return "summer"
    return "autumn"

def evaluate_split(name, y_true, proba):
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    ll = log_loss(y_true, proba, labels=[0, 1])
    bs = brier_score_loss(y_true, proba)
    print(f"\n{name}")
    print(f"AUC    : {auc:.6f}")
    print(f"AP     : {ap:.6f}")
    print(f"LogLoss: {ll:.6f}")
    print(f"Brier  : {bs:.6f}")
    return {"split": name, "auc": auc, "ap": ap, "logloss": ll, "brier": bs}

def slice_metrics(df_slice, pred_col="pred"):
    if df_slice["target"].nunique() < 2:
        return pd.Series({
            "n": len(df_slice),
            "target_rate": df_slice["target"].mean(),
            "auc": np.nan,
            "ap": np.nan
        })
    return pd.Series({
        "n": len(df_slice),
        "target_rate": df_slice["target"].mean(),
        "auc": roc_auc_score(df_slice["target"], df_slice[pred_col]),
        "ap": average_precision_score(df_slice["target"], df_slice[pred_col]),
    })

def make_catboost():
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=8,
        random_seed=RANDOM_SEED,
        auto_class_weights="Balanced",
        early_stopping_rounds=200,
        verbose=100
    )

# ============================================================
# LOAD
# ============================================================
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"], utc=True)

if "alert_airport_id" in df.columns and "airport_alert_id" not in df.columns:
    df = df.rename(columns={"alert_airport_id": "airport_alert_id"})

df = df.sort_values(["airport", "date", "lightning_airport_id"]).reset_index(drop=True)

df["airport"] = df["airport"].astype("category")
df["icloud"] = df["icloud"].astype("boolean")
df["is_last_lightning_cloud_ground"] = df["is_last_lightning_cloud_ground"].astype("boolean")

# ============================================================
# BASE FEATURES
# ============================================================
df["in_alert_zone"] = df["airport_alert_id"].notna()
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["hour"] = df["date"].dt.hour
df["dayofyear"] = df["date"].dt.dayofyear

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

df["season"] = df["month"].map(month_to_season).astype("category")
df["is_cg"] = (~df["icloud"].fillna(False)).astype(int)
df["is_ic"] = (df["icloud"].fillna(False)).astype(int)

az_rad = np.deg2rad(df["azimuth"])
df["x_rel"] = df["dist"] * np.sin(az_rad)
df["y_rel"] = df["dist"] * np.cos(az_rad)

df["is_lt20"] = (df["dist"] <= 20).astype(int)
df["is_20_30"] = ((df["dist"] > 20) & (df["dist"] <= 30)).astype(int)

df["alert_key"] = np.where(
    df["in_alert_zone"],
    df["airport"].astype(str) + "__" + df["airport_alert_id"].astype("Int64").astype(str),
    pd.NA
)

# ============================================================
# ROBUSTESSE PISE 2016
# ============================================================
if not USE_PISE_2016:
    mask_drop = (df["airport"].astype(str) == "Pise") & (df["year"] == 2016)
    print(f"Suppression Pise 2016: {mask_drop.sum()} lignes")
    df = df.loc[~mask_drop].copy()

# ============================================================
# FEATURES HISTORIQUES PAR AEROPORT
# ============================================================
parts = []

for airport, g in df.groupby("airport", observed=True):
    g = g.sort_values("date").copy()
    g["row_id"] = np.arange(len(g))
    g = g.set_index("date")

    for win in ["5min", "10min", "20min", "30min", "60min"]:
        # volumes
        g[f"cnt_all_{win}"] = g["lightning_id"].rolling(win).count()
        g[f"cnt_cg_{win}"] = g["is_cg"].rolling(win).sum()
        g[f"cnt_ic_{win}"] = g["is_ic"].rolling(win).sum()
        g[f"cnt_lt20_{win}"] = g["is_lt20"].rolling(win).sum()
        g[f"cnt_20_30_{win}"] = g["is_20_30"].rolling(win).sum()

        # spatial
        g[f"min_dist_{win}"] = g["dist"].rolling(win).min()
        g[f"mean_dist_{win}"] = g["dist"].rolling(win).mean()
        g[f"max_dist_{win}"] = g["dist"].rolling(win).max()

        # intensité / précision
        g[f"mean_abs_amp_{win}"] = g["amplitude"].abs().rolling(win).mean()
        g[f"max_abs_amp_{win}"] = g["amplitude"].abs().rolling(win).max()
        g[f"mean_maxis_{win}"] = g["maxis"].rolling(win).mean()

        # position relative moyenne
        g[f"mean_x_rel_{win}"] = g["x_rel"].rolling(win).mean()
        g[f"mean_y_rel_{win}"] = g["y_rel"].rolling(win).mean()

    g = g.reset_index()
    parts.append(g)

df_feat = pd.concat(parts, ignore_index=True)

# ============================================================
# GAPS TEMPORELS
# ============================================================
df_feat = df_feat.sort_values(["airport", "date", "lightning_airport_id"]).reset_index(drop=True)

df_feat["gap_prev_sec_airport"] = (
    df_feat.groupby("airport")["date"].diff().dt.total_seconds()
)

df_feat["date_lt20"] = df_feat["date"].where(df_feat["is_lt20"] == 1)
df_feat["last_lt20_date"] = df_feat.groupby("airport")["date_lt20"].ffill()
df_feat["gap_since_last_lt20_sec"] = (
    df_feat["date"] - df_feat["last_lt20_date"]
).dt.total_seconds()

df_feat["date_cg_lt20"] = df_feat["date"].where((df_feat["is_lt20"] == 1) & (df_feat["is_cg"] == 1))
df_feat["last_cg_lt20_date"] = df_feat.groupby("airport")["date_cg_lt20"].ffill()
df_feat["gap_since_last_cg_lt20_sec"] = (
    df_feat["date"] - df_feat["last_cg_lt20_date"]
).dt.total_seconds()

# ============================================================
# FEATURES D'ALERTE
# ============================================================
model_df = df_feat[df_feat["in_alert_zone"]].copy()
model_df = model_df.sort_values(["alert_key", "date", "lightning_airport_id"]).reset_index(drop=True)

model_df["target"] = (model_df["is_last_lightning_cloud_ground"] == True).astype(int)

model_df["rank_in_alert"] = model_df.groupby("alert_key").cumcount() + 1

model_df["alert_start_time"] = model_df.groupby("alert_key")["date"].transform("min")
model_df["elapsed_since_alert_start_sec"] = (
    model_df["date"] - model_df["alert_start_time"]
).dt.total_seconds()

model_df["gap_prev_sec_alert"] = model_df.groupby("alert_key")["date"].diff().dt.total_seconds()

# ============================================================
# RATIOS / TENDANCES
# ============================================================
for base in ["all", "cg", "lt20", "20_30"]:
    for short, long in [("5min", "20min"), ("5min", "30min"), ("10min", "30min")]:
        s = f"cnt_{base}_{short}"
        l = f"cnt_{base}_{long}"
        if s in model_df.columns and l in model_df.columns:
            model_df[f"ratio_{base}_{short}_over_{long}"] = model_df[s] / (model_df[l] + 1.0)

for short, long in [("5min", "20min"), ("5min", "30min"), ("10min", "30min")]:
    for prefix in ["mean_dist", "mean_abs_amp"]:
        s = f"{prefix}_{short}"
        l = f"{prefix}_{long}"
        if s in model_df.columns and l in model_df.columns:
            model_df[f"diff_{prefix}_{short}_minus_{long}"] = model_df[s] - model_df[l]

for win in ["5min", "10min", "20min", "30min"]:
    model_df[f"share_lt20_{win}"] = model_df[f"cnt_lt20_{win}"] / (model_df[f"cnt_all_{win}"] + 1.0)
    model_df[f"share_20_30_{win}"] = model_df[f"cnt_20_30_{win}"] / (model_df[f"cnt_all_{win}"] + 1.0)
    model_df[f"share_cg_{win}"] = model_df[f"cnt_cg_{win}"] / (model_df[f"cnt_all_{win}"] + 1.0)

# ============================================================
# FEATURES TRANSITION DE FIN D'ALERTE
# ============================================================
for win in ["5min", "10min", "20min", "30min"]:
    model_df[f"ratio_far_vs_close_{win}"] = (
        model_df[f"cnt_20_30_{win}"] / (model_df[f"cnt_lt20_{win}"] + 1.0)
    )

for short, long in [("5min", "20min"), ("5min", "30min"), ("10min", "30min")]:
    model_df[f"trend_all_{short}_minus_{long}"] = (
        model_df[f"cnt_all_{short}"] - model_df[f"cnt_all_{long}"]
    )
    model_df[f"trend_lt20_{short}_minus_{long}"] = (
        model_df[f"cnt_lt20_{short}"] - model_df[f"cnt_lt20_{long}"]
    )
    model_df[f"trend_20_30_{short}_minus_{long}"] = (
        model_df[f"cnt_20_30_{short}"] - model_df[f"cnt_20_30_{long}"]
    )
    model_df[f"trend_cg_{short}_minus_{long}"] = (
        model_df[f"cnt_cg_{short}"] - model_df[f"cnt_cg_{long}"]
    )

for short, long in [("5min", "20min"), ("5min", "30min"), ("10min", "30min")]:
    model_df[f"norm_trend_lt20_{short}_over_{long}"] = (
        model_df[f"cnt_lt20_{short}"] / (model_df[f"cnt_lt20_{long}"] + 1.0)
    )
    model_df[f"norm_trend_all_{short}_over_{long}"] = (
        model_df[f"cnt_all_{short}"] / (model_df[f"cnt_all_{long}"] + 1.0)
    )

for win in ["5min", "10min", "20min", "30min"]:
    model_df[f"close_minus_far_{win}"] = (
        model_df[f"cnt_lt20_{win}"] - model_df[f"cnt_20_30_{win}"]
    )

for short, long in [("5min", "20min"), ("5min", "30min"), ("10min", "30min")]:
    model_df[f"trend_dist_{short}_minus_{long}"] = (
        model_df[f"mean_dist_{short}"] - model_df[f"mean_dist_{long}"]
    )
    model_df[f"trend_min_dist_{short}_minus_{long}"] = (
        model_df[f"min_dist_{short}"] - model_df[f"min_dist_{long}"]
    )

model_df["gap_ratio_lt20_vs_alert"] = (
    model_df["gap_since_last_lt20_sec"] / (model_df["gap_prev_sec_alert"] + 1.0)
)
model_df["gap_ratio_cg_lt20_vs_alert"] = (
    model_df["gap_since_last_cg_lt20_sec"] / (model_df["gap_prev_sec_alert"] + 1.0)
)
model_df["gap_ratio_cg_lt20_vs_airport"] = (
    model_df["gap_since_last_cg_lt20_sec"] / (model_df["gap_prev_sec_airport"] + 1.0)
)

model_df["elapsed_per_rank"] = (
    model_df["elapsed_since_alert_start_sec"] / model_df["rank_in_alert"].clip(lower=1)
)

for win in ["5min", "10min", "20min"]:
    model_df[f"close_activity_score_{win}"] = (
        model_df[f"cnt_lt20_{win}"] / (model_df[f"mean_dist_{win}"] + 1.0)
    )
    model_df[f"far_activity_score_{win}"] = (
        model_df[f"cnt_20_30_{win}"] * model_df[f"mean_dist_{win}"]
    )

for win in ["5min", "10min"]:
    model_df[f"lt20_density_per_min_{win}"] = (
        model_df[f"cnt_lt20_{win}"] / float(win.replace("min", ""))
    )

# ============================================================
# NORMALISATIONS PAR AEROPORT
# ============================================================
norm_base_cols = [
    "cnt_all_5min", "cnt_all_10min", "cnt_all_20min", "cnt_all_30min",
    "cnt_lt20_5min", "cnt_lt20_10min", "cnt_lt20_20min", "cnt_lt20_30min",
    "cnt_20_30_5min", "cnt_20_30_10min", "cnt_20_30_20min", "cnt_20_30_30min",
    "cnt_cg_5min", "cnt_cg_10min", "cnt_cg_20min", "cnt_cg_30min",
    "mean_dist_5min", "mean_dist_10min", "mean_dist_20min", "mean_dist_30min",
    "min_dist_5min", "min_dist_10min", "min_dist_20min", "min_dist_30min",
    "gap_prev_sec_airport", "gap_since_last_lt20_sec", "gap_since_last_cg_lt20_sec",
]

for col in norm_base_cols:
    if col in model_df.columns:
        airport_mean = model_df.groupby("airport")[col].transform("mean")
        airport_std = model_df.groupby("airport")[col].transform("std")
        model_df[f"{col}_airport_mean_ratio"] = model_df[col] / (airport_mean + 1.0)
        model_df[f"{col}_airport_z"] = (model_df[col] - airport_mean) / (airport_std + 1e-6)

# ============================================================
# LOGS
# ============================================================
extra_log_cols = [
    "ratio_far_vs_close_5min", "ratio_far_vs_close_10min", "ratio_far_vs_close_20min", "ratio_far_vs_close_30min",
    "gap_ratio_lt20_vs_alert", "gap_ratio_cg_lt20_vs_alert", "gap_ratio_cg_lt20_vs_airport",
    "elapsed_per_rank",
    "close_activity_score_5min", "close_activity_score_10min", "close_activity_score_20min",
    "far_activity_score_5min", "far_activity_score_10min", "far_activity_score_20min",
]

for c in extra_log_cols:
    if c in model_df.columns:
        model_df[f"log1p_{c}"] = np.log1p(
            model_df[c].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        )

log_cols = [
    "cnt_all_5min", "cnt_all_10min", "cnt_all_20min", "cnt_all_30min", "cnt_all_60min",
    "cnt_lt20_5min", "cnt_lt20_10min", "cnt_lt20_20min", "cnt_lt20_30min",
    "cnt_20_30_5min", "cnt_20_30_10min", "cnt_20_30_20min", "cnt_20_30_30min",
    "cnt_cg_5min", "cnt_cg_10min", "cnt_cg_20min", "cnt_cg_30min",
    "gap_prev_sec_airport", "gap_since_last_lt20_sec", "gap_since_last_cg_lt20_sec",
    "elapsed_since_alert_start_sec", "gap_prev_sec_alert"
]

for c in log_cols:
    if c in model_df.columns:
        model_df[f"log1p_{c}"] = np.log1p(
            model_df[c].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        )

# ============================================================
# FEATURES FINALES
# ============================================================
candidate_features = [
    # contexte
    "airport", "season", "month", "hour",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",

    # info courant
    "dist", "azimuth", "amplitude", "maxis", "x_rel", "y_rel",

    # alerte courante
    "rank_in_alert", "elapsed_since_alert_start_sec", "gap_prev_sec_alert",
    "log1p_elapsed_since_alert_start_sec", "log1p_gap_prev_sec_alert",

    # gaps globaux
    "gap_prev_sec_airport", "gap_since_last_lt20_sec", "gap_since_last_cg_lt20_sec",
    "log1p_gap_prev_sec_airport", "log1p_gap_since_last_lt20_sec", "log1p_gap_since_last_cg_lt20_sec",

    # volumes
    "cnt_all_5min", "cnt_all_10min", "cnt_all_20min", "cnt_all_30min", "cnt_all_60min",
    "cnt_cg_5min", "cnt_cg_10min", "cnt_cg_20min", "cnt_cg_30min",
    "cnt_lt20_5min", "cnt_lt20_10min", "cnt_lt20_20min", "cnt_lt20_30min",
    "cnt_20_30_5min", "cnt_20_30_10min", "cnt_20_30_20min", "cnt_20_30_30min",

    # logs volumes
    "log1p_cnt_all_5min", "log1p_cnt_all_10min", "log1p_cnt_all_20min", "log1p_cnt_all_30min", "log1p_cnt_all_60min",
    "log1p_cnt_lt20_5min", "log1p_cnt_lt20_10min", "log1p_cnt_lt20_20min", "log1p_cnt_lt20_30min",
    "log1p_cnt_20_30_5min", "log1p_cnt_20_30_10min", "log1p_cnt_20_30_20min", "log1p_cnt_20_30_30min",
    "log1p_cnt_cg_5min", "log1p_cnt_cg_10min", "log1p_cnt_cg_20min", "log1p_cnt_cg_30min",

    # spatial rolling
    "min_dist_5min", "min_dist_10min", "min_dist_20min", "min_dist_30min",
    "mean_dist_5min", "mean_dist_10min", "mean_dist_20min", "mean_dist_30min",
    "max_dist_5min", "max_dist_10min", "max_dist_20min", "max_dist_30min",

    # amplitude / qualité
    "mean_abs_amp_5min", "mean_abs_amp_10min", "mean_abs_amp_20min", "mean_abs_amp_30min",
    "max_abs_amp_5min", "max_abs_amp_10min", "max_abs_amp_20min", "max_abs_amp_30min",
    "mean_maxis_5min", "mean_maxis_10min", "mean_maxis_20min", "mean_maxis_30min",

    # ratios / shares
    "ratio_all_5min_over_20min", "ratio_all_5min_over_30min", "ratio_all_10min_over_30min",
    "ratio_lt20_5min_over_20min", "ratio_lt20_5min_over_30min", "ratio_lt20_10min_over_30min",
    "ratio_20_30_5min_over_20min", "ratio_20_30_5min_over_30min", "ratio_20_30_10min_over_30min",
    "ratio_cg_5min_over_20min", "ratio_cg_5min_over_30min", "ratio_cg_10min_over_30min",
    "share_lt20_5min", "share_lt20_10min", "share_lt20_20min", "share_lt20_30min",
    "share_20_30_5min", "share_20_30_10min", "share_20_30_20min", "share_20_30_30min",
    "share_cg_5min", "share_cg_10min", "share_cg_20min", "share_cg_30min",

    # tendances
    "diff_mean_dist_5min_minus_20min", "diff_mean_dist_5min_minus_30min", "diff_mean_dist_10min_minus_30min",
    "diff_mean_abs_amp_5min_minus_20min", "diff_mean_abs_amp_5min_minus_30min", "diff_mean_abs_amp_10min_minus_30min",

    # transition
    "ratio_far_vs_close_5min", "ratio_far_vs_close_10min", "ratio_far_vs_close_20min", "ratio_far_vs_close_30min",
    "log1p_ratio_far_vs_close_5min", "log1p_ratio_far_vs_close_10min", "log1p_ratio_far_vs_close_20min", "log1p_ratio_far_vs_close_30min",

    "trend_all_5min_minus_20min", "trend_all_5min_minus_30min", "trend_all_10min_minus_30min",
    "trend_lt20_5min_minus_20min", "trend_lt20_5min_minus_30min", "trend_lt20_10min_minus_30min",
    "trend_20_30_5min_minus_20min", "trend_20_30_5min_minus_30min", "trend_20_30_10min_minus_30min",
    "trend_cg_5min_minus_20min", "trend_cg_5min_minus_30min", "trend_cg_10min_minus_30min",

    "norm_trend_lt20_5min_over_20min", "norm_trend_lt20_5min_over_30min", "norm_trend_lt20_10min_over_30min",
    "norm_trend_all_5min_over_20min", "norm_trend_all_5min_over_30min", "norm_trend_all_10min_over_30min",

    "close_minus_far_5min", "close_minus_far_10min", "close_minus_far_20min", "close_minus_far_30min",

    "trend_dist_5min_minus_20min", "trend_dist_5min_minus_30min", "trend_dist_10min_minus_30min",
    "trend_min_dist_5min_minus_20min", "trend_min_dist_5min_minus_30min", "trend_min_dist_10min_minus_30min",

    "gap_ratio_lt20_vs_alert", "gap_ratio_cg_lt20_vs_alert", "gap_ratio_cg_lt20_vs_airport",
    "log1p_gap_ratio_lt20_vs_alert", "log1p_gap_ratio_cg_lt20_vs_alert", "log1p_gap_ratio_cg_lt20_vs_airport",

    "elapsed_per_rank", "log1p_elapsed_per_rank",

    "close_activity_score_5min", "close_activity_score_10min", "close_activity_score_20min",
    "far_activity_score_5min", "far_activity_score_10min", "far_activity_score_20min",
    "log1p_close_activity_score_5min", "log1p_close_activity_score_10min", "log1p_close_activity_score_20min",
    "log1p_far_activity_score_5min", "log1p_far_activity_score_10min", "log1p_far_activity_score_20min",

    "lt20_density_per_min_5min", "lt20_density_per_min_10min",
]

# ajouter les normalisations par aéroport
for col in norm_base_cols:
    candidate_features.append(f"{col}_airport_mean_ratio")
    candidate_features.append(f"{col}_airport_z")

# garder seulement les colonnes existantes
features = [c for c in candidate_features if c in model_df.columns]

# ajouter automatiquement tous les logs créés mais oubliés
for c in model_df.columns:
    if c.startswith("log1p_") and c not in features:
        features.append(c)

# imputation simple
for col in features:
    if str(model_df[col].dtype) in ["category", "object"]:
        model_df[col] = model_df[col].astype(str).fillna("MISSING")
    else:
        model_df[col] = model_df[col].replace([np.inf, -np.inf], np.nan)

# ============================================================
# SPLIT TEMPOREL
# ============================================================
train_df = model_df[model_df["year"] <= TRAIN_END_YEAR].copy()
valid_df = model_df[model_df["year"] == VALID_YEAR].copy()
test_df = model_df[model_df["year"] == TEST_YEAR].copy()

print("Train shape:", train_df.shape, "| target rate:", train_df["target"].mean())
print("Valid shape:", valid_df.shape, "| target rate:", valid_df["target"].mean())
print("Test  shape:", test_df.shape, "| target rate:", test_df["target"].mean() if len(test_df) else "NA")

if len(valid_df) == 0:
    raise ValueError("Validation vide. Vérifie les années présentes dans le fichier.")
if len(test_df) == 0:
    print("Attention: pas de TEST_YEAR trouvé, on évaluera seulement sur la validation.")

X_train = train_df[features].copy()
y_train = train_df["target"].copy()

X_valid = valid_df[features].copy()
y_valid = valid_df["target"].copy()

if len(test_df):
    X_test = test_df[features].copy()
    y_test = test_df["target"].copy()

cat_features = [i for i, c in enumerate(features) if c in ["airport", "season"]]

# ============================================================
# MODELE GLOBAL
# ============================================================
global_model = make_catboost()

train_pool_global = Pool(X_train, y_train, cat_features=cat_features)
valid_pool_global = Pool(X_valid, y_valid, cat_features=cat_features)

if len(test_df):
    test_pool_global = Pool(X_test, y_test, cat_features=cat_features)

print("\n================ GLOBAL MODEL ================\n")
global_model.fit(
    train_pool_global,
    eval_set=valid_pool_global,
    use_best_model=True
)

valid_pred_global = global_model.predict_proba(valid_pool_global)[:, 1]
if len(test_df):
    test_pred_global = global_model.predict_proba(test_pool_global)[:, 1]

# ============================================================
# MODELES LOCAUX PAR AEROPORT
# ============================================================
local_models = {}
valid_pred_local = np.full(len(valid_df), np.nan)
test_pred_local = np.full(len(test_df), np.nan) if len(test_df) else None

# Pour les modèles locaux, on retire airport des features
local_features = [f for f in features if f != "airport"]
local_cat_features = [i for i, c in enumerate(local_features) if c in ["season"]]

print("\n================ LOCAL MODELS ================\n")

for airport in sorted(train_df["airport"].astype(str).unique()):
    tr_a = train_df[train_df["airport"].astype(str) == airport].copy()
    va_a = valid_df[valid_df["airport"].astype(str) == airport].copy()

    if len(tr_a) < MIN_LOCAL_TRAIN_ROWS or tr_a["target"].sum() < MIN_LOCAL_POSITIVES:
        print(f"[SKIP] {airport}: train_rows={len(tr_a)}, positives={tr_a['target'].sum()}")
        continue

    X_tr_a = tr_a[local_features].copy()
    y_tr_a = tr_a["target"].copy()

    X_va_a = va_a[local_features].copy()
    y_va_a = va_a["target"].copy()

    train_pool_a = Pool(X_tr_a, y_tr_a, cat_features=local_cat_features)
    valid_pool_a = Pool(X_va_a, y_va_a, cat_features=local_cat_features)

    model_a = make_catboost()
    print(f"\n----- LOCAL MODEL: {airport} -----")
    model_a.fit(
        train_pool_a,
        eval_set=valid_pool_a,
        use_best_model=True
    )

    local_models[airport] = model_a

    mask_val = (valid_df["airport"].astype(str) == airport).values
    valid_pred_local[mask_val] = model_a.predict_proba(valid_pool_a)[:, 1]

    if len(test_df):
        te_a = test_df[test_df["airport"].astype(str) == airport].copy()
        if len(te_a):
            X_te_a = te_a[local_features].copy()
            y_te_a = te_a["target"].copy()
            test_pool_a = Pool(X_te_a, y_te_a, cat_features=local_cat_features)

            mask_test = (test_df["airport"].astype(str) == airport).values
            test_pred_local[mask_test] = model_a.predict_proba(test_pool_a)[:, 1]

# ============================================================
# BLEND GLOBAL + LOCAL
# ============================================================
valid_pred_blend = valid_pred_global.copy()
mask_valid_local = ~np.isnan(valid_pred_local)
valid_pred_blend[mask_valid_local] = (
    GLOBAL_WEIGHT * valid_pred_global[mask_valid_local] +
    LOCAL_WEIGHT * valid_pred_local[mask_valid_local]
)

if len(test_df):
    test_pred_blend = test_pred_global.copy()
    mask_test_local = ~np.isnan(test_pred_local)
    test_pred_blend[mask_test_local] = (
        GLOBAL_WEIGHT * test_pred_global[mask_test_local] +
        LOCAL_WEIGHT * test_pred_local[mask_test_local]
    )

# ============================================================
# EVALUATION
# ============================================================
results = []

results.append(evaluate_split("VALID_GLOBAL", y_valid, valid_pred_global))
results.append(evaluate_split("VALID_BLEND", y_valid, valid_pred_blend))

if len(test_df):
    results.append(evaluate_split("TEST_GLOBAL", y_test, test_pred_global))
    results.append(evaluate_split("TEST_BLEND", y_test, test_pred_blend))

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

# ============================================================
# FEATURE IMPORTANCE GLOBAL
# ============================================================
fi = pd.DataFrame({
    "feature": features,
    "importance": global_model.get_feature_importance(train_pool_global)
}).sort_values("importance", ascending=False)

print("\nTop 30 global feature importances:")
print(fi.head(30))
fi.to_csv(OUTPUT_DIR / "feature_importance_global.csv", index=False)

# ============================================================
# EXPORT PREDICTIONS
# ============================================================
valid_out = valid_df[["lightning_id", "airport", "date", "airport_alert_id", "target"]].copy()
valid_out["pred_global"] = valid_pred_global
valid_out["pred_blend"] = valid_pred_blend
valid_out.to_csv(OUTPUT_DIR / "valid_predictions.csv", index=False)

if len(test_df):
    test_out = test_df[["lightning_id", "airport", "date", "airport_alert_id", "target"]].copy()
    test_out["pred_global"] = test_pred_global
    test_out["pred_blend"] = test_pred_blend
    test_out.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# ============================================================
# DIAGNOSTICS PAR AEROPORT / SAISON
# ============================================================
valid_diag = valid_out.merge(valid_df[["lightning_id", "season"]], on="lightning_id", how="left")

by_airport_global = valid_diag.rename(columns={"pred_global": "pred"}).groupby("airport").apply(slice_metrics).reset_index()
by_airport_blend = valid_diag.rename(columns={"pred_blend": "pred"}).groupby("airport").apply(slice_metrics).reset_index()

by_airport_global["model"] = "global"
by_airport_blend["model"] = "blend"
by_airport = pd.concat([by_airport_global, by_airport_blend], ignore_index=True)

by_season_global = valid_diag.rename(columns={"pred_global": "pred"}).groupby("season").apply(slice_metrics).reset_index()
by_season_blend = valid_diag.rename(columns={"pred_blend": "pred"}).groupby("season").apply(slice_metrics).reset_index()

by_season_global["model"] = "global"
by_season_blend["model"] = "blend"
by_season = pd.concat([by_season_global, by_season_blend], ignore_index=True)

print("\nValidation par aéroport:")
print(by_airport)

print("\nValidation par saison:")
print(by_season)

by_airport.to_csv(OUTPUT_DIR / "valid_metrics_by_airport.csv", index=False)
by_season.to_csv(OUTPUT_DIR / "valid_metrics_by_season.csv", index=False)

# ============================================================
# SAVE MODELS
# ============================================================
global_model.save_model(str(OUTPUT_DIR / "catboost_global_model.cbm"))

for airport, model_a in local_models.items():
    model_a.save_model(str(OUTPUT_DIR / f"catboost_local_{airport}.cbm"))

print("\nTerminé. Outputs dans :", OUTPUT_DIR.resolve())