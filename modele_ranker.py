import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostRanker, Pool

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "data/segment_alerts_all_airports_train.csv"
OUTPUT_DIR = Path("catboost_ranker_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

USE_PISE_2016 = False
RANDOM_SEED = 42

TRAIN_END_YEAR = 2020
VALID_YEAR = 2021
TEST_YEAR = 2022

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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def topk_alert_accuracy(df, pred_col="raw_score", k=1):
    correct = 0
    total = 0

    for _, g in df.groupby("alert_key"):
        if g["target"].sum() != 1:
            continue
        g = g.sort_values(pred_col, ascending=False).reset_index(drop=True)
        if (g.head(k)["target"] == 1).any():
            correct += 1
        total += 1

    return correct / total if total > 0 else np.nan

def mean_reciprocal_rank(df, pred_col="raw_score"):
    rr = []

    for _, g in df.groupby("alert_key"):
        if g["target"].sum() != 1:
            continue
        g = g.sort_values(pred_col, ascending=False).reset_index(drop=True)
        pos = np.where(g["target"].values == 1)[0]
        if len(pos) == 1:
            rr.append(1.0 / (pos[0] + 1))

    return np.mean(rr) if rr else np.nan

def evaluate_rank_split(name, df_eval, raw_scores):
    out = df_eval.copy()
    out["raw_score"] = raw_scores
    out["pred"] = sigmoid(raw_scores)

    top1 = topk_alert_accuracy(out, pred_col="raw_score", k=1)
    top3 = topk_alert_accuracy(out, pred_col="raw_score", k=3)
    mrr = mean_reciprocal_rank(out, pred_col="raw_score")

    print(f"\n{name}")
    print(f"Top1 alert accuracy : {top1:.6f}")
    print(f"Top3 alert accuracy : {top3:.6f}")
    print(f"MRR                : {mrr:.6f}")

    keep_cols = [
        "lightning_id", "airport", "date", "airport_alert_id",
        "alert_key", "season", "target", "raw_score", "pred"
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].copy()

    metrics = {
        "split": name,
        "top1_alert_accuracy": top1,
        "top3_alert_accuracy": top3,
        "mrr": mrr,
    }
    return out, metrics

def slice_metrics_rank(df_slice, pred_col="raw_score"):
    if len(df_slice) == 0:
        return pd.Series({
            "n": 0,
            "target_rate": np.nan,
            "top1": np.nan,
            "top3": np.nan,
            "mrr": np.nan
        })

    return pd.Series({
        "n": len(df_slice),
        "target_rate": df_slice["target"].mean(),
        "top1": topk_alert_accuracy(df_slice, pred_col=pred_col, k=1),
        "top3": topk_alert_accuracy(df_slice, pred_col=pred_col, k=3),
        "mrr": mean_reciprocal_rank(df_slice, pred_col=pred_col),
    })

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
# ROLLING FEATURES PAR AEROPORT
# ============================================================
parts = []

for airport, g in df.groupby("airport", observed=True):
    g = g.sort_values("date").copy()
    g = g.set_index("date")

    for win in ["5min", "10min", "30min"]:
        g[f"cnt_all_{win}"] = g["lightning_id"].rolling(win).count()
        g[f"cnt_lt20_{win}"] = g["is_lt20"].rolling(win).sum()
        g[f"cnt_20_30_{win}"] = g["is_20_30"].rolling(win).sum()
        g[f"cnt_cg_{win}"] = g["is_cg"].rolling(win).sum()

        g[f"min_dist_{win}"] = g["dist"].rolling(win).min()
        g[f"mean_dist_{win}"] = g["dist"].rolling(win).mean()

    g = g.reset_index()
    parts.append(g)

df_feat = pd.concat(parts, ignore_index=True)

# ============================================================
# GAPS
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
# DATASET SUPERVISE
# ============================================================
model_df = df_feat[df_feat["in_alert_zone"]].copy()
model_df = model_df.sort_values(["alert_key", "date", "lightning_airport_id"]).reset_index(drop=True)
model_df["target"] = (model_df["is_last_lightning_cloud_ground"] == True).astype(int)

# ============================================================
# FEATURES "FIN EXACTE"
# ============================================================
# décroissance
model_df["decay_lt20"] = model_df["cnt_lt20_5min"] / (model_df["cnt_lt20_30min"] + 1.0)
model_df["decay_all"] = model_df["cnt_all_5min"] / (model_df["cnt_all_30min"] + 1.0)
model_df["decay_cg"] = model_df["cnt_cg_5min"] / (model_df["cnt_cg_30min"] + 1.0)

# silence
model_df["silence_lt20"] = model_df["gap_since_last_lt20_sec"]
model_df["silence_cg"] = model_df["gap_since_last_cg_lt20_sec"]
model_df["silence_ratio"] = model_df["gap_since_last_lt20_sec"] / (model_df["cnt_lt20_30min"] + 1.0)

# activité nulle récente
model_df["no_activity_5min"] = (model_df["cnt_lt20_5min"] == 0).astype(int)
model_df["no_activity_10min"] = (model_df["cnt_lt20_10min"] == 0).astype(int)

# éclair isolé
model_df["isolated_event"] = (
    (model_df["cnt_lt20_5min"] <= 1) &
    (model_df["cnt_lt20_10min"] <= 2)
).astype(int)

# chute d'activité
model_df["drop_activity"] = model_df["cnt_lt20_10min"] - model_df["cnt_lt20_5min"]

# gap dominant
# ici on approxime avec le gap aéroport, car pas de gap intra-alerte dans cette version compacte
model_df["gap_dominance"] = (
    model_df["gap_since_last_lt20_sec"] / (model_df["gap_prev_sec_airport"] + 1.0)
)

# proche vs loin
model_df["ratio_far_vs_close_10min"] = model_df["cnt_20_30_10min"] / (model_df["cnt_lt20_10min"] + 1.0)
model_df["close_minus_far_10min"] = model_df["cnt_lt20_10min"] - model_df["cnt_20_30_10min"]
model_df["share_lt20_10min"] = model_df["cnt_lt20_10min"] / (model_df["cnt_all_10min"] + 1.0)
model_df["share_20_30_10min"] = model_df["cnt_20_30_10min"] / (model_df["cnt_all_10min"] + 1.0)

# éloignement
model_df["moving_away"] = model_df["mean_dist_10min"] - model_df["mean_dist_5min"]
model_df["trend_dist_10min_minus_30min"] = model_df["mean_dist_10min"] - model_df["mean_dist_30min"]
model_df["last_event_far"] = (model_df["min_dist_5min"] > 20).astype(int)

# logs
for c in [
    "cnt_lt20_5min", "cnt_lt20_10min", "cnt_lt20_30min",
    "cnt_20_30_10min", "cnt_all_10min", "cnt_all_30min",
    "gap_since_last_lt20_sec", "gap_since_last_cg_lt20_sec",
    "decay_lt20", "decay_all", "decay_cg",
    "silence_lt20", "silence_cg", "silence_ratio",
    "ratio_far_vs_close_10min", "gap_dominance"
]:
    model_df[f"log1p_{c}"] = np.log1p(
        model_df[c].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    )

# ============================================================
# FEATURES FINALES ULTRA CIBLEES
# ============================================================
features = [
    "airport", "season", "month", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "dist", "amplitude", "maxis",

    "cnt_lt20_5min", "cnt_lt20_10min", "cnt_lt20_30min",
    "cnt_20_30_10min", "cnt_all_10min", "cnt_all_30min",

    "log1p_cnt_lt20_5min", "log1p_cnt_lt20_10min", "log1p_cnt_lt20_30min",
    "log1p_cnt_20_30_10min", "log1p_cnt_all_10min", "log1p_cnt_all_30min",

    "decay_lt20", "decay_all", "decay_cg",
    "log1p_decay_lt20", "log1p_decay_all", "log1p_decay_cg",

    "silence_lt20", "silence_cg", "silence_ratio",
    "log1p_silence_lt20", "log1p_silence_cg", "log1p_silence_ratio",

    "no_activity_5min", "no_activity_10min",
    "isolated_event",
    "drop_activity",
    "gap_dominance", "log1p_gap_dominance",

    "ratio_far_vs_close_10min", "log1p_ratio_far_vs_close_10min",
    "close_minus_far_10min",
    "share_lt20_10min", "share_20_30_10min",

    "moving_away", "trend_dist_10min_minus_30min", "last_event_far",
]

features = [c for c in features if c in model_df.columns]

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

train_df["group_id"] = train_df["alert_key"].astype("category").cat.codes
valid_df["group_id"] = valid_df["alert_key"].astype("category").cat.codes
if len(test_df):
    test_df["group_id"] = test_df["alert_key"].astype("category").cat.codes

X_train = train_df[features].copy()
y_train = train_df["target"].copy()

X_valid = valid_df[features].copy()
y_valid = valid_df["target"].copy()

if len(test_df):
    X_test = test_df[features].copy()
    y_test = test_df["target"].copy()

cat_features = [i for i, c in enumerate(features) if c in ["airport", "season"]]

train_pool = Pool(X_train, y_train, group_id=train_df["group_id"], cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, group_id=valid_df["group_id"], cat_features=cat_features)

if len(test_df):
    test_pool = Pool(X_test, y_test, group_id=test_df["group_id"], cat_features=cat_features)

# ============================================================
# TRAIN
# ============================================================
model = CatBoostRanker(
    loss_function="YetiRankPairwise",
    eval_metric="NDCG",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=8,
    random_seed=RANDOM_SEED,
    early_stopping_rounds=200,
    verbose=100
)

model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

# ============================================================
# EVAL
# ============================================================
results = []

valid_raw = model.predict(valid_pool)
valid_out, valid_metrics = evaluate_rank_split("VALID", valid_df, valid_raw)
results.append(valid_metrics)

if len(test_df):
    test_raw = model.predict(test_pool)
    test_out, test_metrics = evaluate_rank_split("TEST", test_df, test_raw)
    results.append(test_metrics)

pd.DataFrame(results).to_csv(OUTPUT_DIR / "metrics.csv", index=False)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
fi = pd.DataFrame({
    "feature": features,
    "importance": model.get_feature_importance(train_pool)
}).sort_values("importance", ascending=False)

print("\nTop feature importances:")
print(fi)
fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

# ============================================================
# DIAGNOSTICS
# ============================================================
valid_diag = valid_out.copy()

by_airport = valid_diag.groupby("airport").apply(slice_metrics_rank, pred_col="raw_score").reset_index()
by_season = valid_diag.groupby("season").apply(slice_metrics_rank, pred_col="raw_score").reset_index()

print("\nValidation par aéroport:")
print(by_airport)

print("\nValidation par saison:")
print(by_season)

by_airport.to_csv(OUTPUT_DIR / "valid_metrics_by_airport.csv", index=False)
by_season.to_csv(OUTPUT_DIR / "valid_metrics_by_season.csv", index=False)

# ============================================================
# EXPORT
# ============================================================
valid_out.to_csv(OUTPUT_DIR / "valid_predictions.csv", index=False)
if len(test_df):
    test_out.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

model.save_model(str(OUTPUT_DIR / "catboost_ranker_sharp_model.cbm"))

print("\nTerminé. Outputs dans :", OUTPUT_DIR.resolve())