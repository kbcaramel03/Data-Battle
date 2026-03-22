import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "data/segment_alerts_all_airports_train.csv"
OUTPUT_DIR = Path("eda_outputs_databattle")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DROP_PISA_2016_FOR_ICLOUD_ANALYSIS = True
MAX_ROWS_SAMPLE_FOR_PAIRPLOTS = 15000

# ============================================================
# UTILS
# ============================================================
def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()

def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

def safe_mode(series):
    if series.dropna().empty:
        return np.nan
    return series.mode().iloc[0]

def q95(x):
    return x.quantile(0.95)

def q99(x):
    return x.quantile(0.99)

def ensure_datetime(df, col="date"):
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

# ============================================================
# LOAD
# ============================================================
print_section("1) CHARGEMENT")
df = pd.read_csv(CSV_PATH)
print("Shape brute:", df.shape)
print("Colonnes:", list(df.columns))

df = ensure_datetime(df, "date")

# Harmonisation noms si besoin
if "alert_airport_id" in df.columns and "airport_alert_id" not in df.columns:
    df = df.rename(columns={"alert_airport_id": "airport_alert_id"})

expected_cols = [
    "lightning_id", "lightning_airport_id", "date", "lon", "lat",
    "amplitude", "maxis", "icloud", "dist", "azimuth", "airport",
    "airport_alert_id", "is_last_lightning_cloud_ground"
]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes: {missing}")

# Typage
df["airport"] = df["airport"].astype("category")
df["icloud"] = df["icloud"].astype("boolean")
df["is_last_lightning_cloud_ground"] = df["is_last_lightning_cloud_ground"].astype("boolean")

# Tri chronologique
df = df.sort_values(["airport", "date", "lightning_airport_id"]).reset_index(drop=True)

# ============================================================
# FEATURES DE BASE
# ============================================================
print_section("2) FEATURES DE BASE")

# bool utile : ligne dans zone d'alerte 20km
df["in_alert_zone"] = df["airport_alert_id"].notna()

# année / mois / heure
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["hour"] = df["date"].dt.hour
df["minute"] = df["date"].dt.minute
df["dayofyear"] = df["date"].dt.dayofyear

# saison simple
def month_to_season(m):
    if m in [12, 1, 2]:
        return "winter"
    if m in [3, 4, 5]:
        return "spring"
    if m in [6, 7, 8]:
        return "summer"
    return "autumn"

df["season"] = df["month"].map(month_to_season).astype("category")

# nature éclair
df["is_cg"] = (~df["icloud"].fillna(False)).astype(int)  # cloud-ground
df["is_ic"] = (df["icloud"].fillna(False)).astype(int)

# bande distance
df["dist_band"] = pd.cut(
    df["dist"],
    bins=[0, 5, 10, 20, 30, np.inf],
    labels=["0_5", "5_10", "10_20", "20_30", "30_plus"],
    include_lowest=True
)

# coord polaires -> cartésiennes relatives à l'aéroport
az_rad = np.deg2rad(df["azimuth"])
df["x_rel"] = df["dist"] * np.sin(az_rad)
df["y_rel"] = df["dist"] * np.cos(az_rad)

print(df.dtypes)

# ============================================================
# QUALITE / MANQUANTS / DUPLICATS
# ============================================================
print_section("3) QUALITE DES DONNEES")

quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_missing": df.isna().sum(),
    "pct_missing": df.isna().mean().mul(100).round(3),
    "n_unique": df.nunique(dropna=True)
}).sort_values("pct_missing", ascending=False)

print(quality)
quality.to_csv(OUTPUT_DIR / "quality_report.csv")

dup_all = df.duplicated().sum()
dup_lightning = df["lightning_id"].duplicated().sum()
dup_airport_lightning = df.duplicated(subset=["airport", "lightning_airport_id"]).sum()

print("Duplicats lignes complètes:", dup_all)
print("Duplicats lightning_id:", dup_lightning)
print("Duplicats (airport, lightning_airport_id):", dup_airport_lightning)

# ============================================================
# VUE D'ENSEMBLE
# ============================================================
print_section("4) VUE D'ENSEMBLE GLOBALE")

print("Nombre total de lignes:", len(df))
print("Nombre d'aéroports:", df["airport"].nunique())
print("Part lignes dans zone d'alerte:", round(df["in_alert_zone"].mean(), 4))
print("Distribution aéroports:")
print(df["airport"].value_counts())

print("\nDistribution icloud:")
print(df["icloud"].value_counts(dropna=False))

print("\nDistribution cible sur lignes d'alerte:")
target_dist = df.loc[df["in_alert_zone"], "is_last_lightning_cloud_ground"].value_counts(dropna=False)
print(target_dist)

# Histogramme dist
plt.figure(figsize=(8, 5))
df["dist"].plot(kind="hist", bins=60)
plt.title("Distribution de dist")
save_fig("dist_hist.png")

# Histogramme amplitude
plt.figure(figsize=(8, 5))
df["amplitude"].clip(df["amplitude"].quantile(0.01), df["amplitude"].quantile(0.99)).plot(kind="hist", bins=60)
plt.title("Distribution amplitude (clippée 1%-99%)")
save_fig("amplitude_hist_clipped.png")

# Par aéroport
airport_counts = df["airport"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
airport_counts.plot(kind="bar")
plt.title("Nombre d'éclairs par aéroport")
save_fig("airport_counts.png")

# ============================================================
# IDS D'ALERTES
# ============================================================
print_section("5) STRUCTURE DES IDS D'ALERTE")

alerts_per_airport = df.groupby("airport")["airport_alert_id"].nunique(dropna=True).sort_values(ascending=False)
print("Nombre d'alertes uniques par aéroport:")
print(alerts_per_airport)
alerts_per_airport.to_csv(OUTPUT_DIR / "alerts_per_airport.csv")

# vérifie si un même airport_alert_id existe dans plusieurs aéroports
alert_cross_airport = (
    df.loc[df["in_alert_zone"], ["airport", "airport_alert_id"]]
    .drop_duplicates()
    .groupby("airport_alert_id")["airport"]
    .nunique()
)
n_cross = (alert_cross_airport > 1).sum()
print("Nombre d'IDs d'alerte présents dans plusieurs aéroports:", n_cross)

# clé d'alerte propre
df["alert_key"] = np.where(
    df["in_alert_zone"],
    df["airport"].astype(str) + "__" + df["airport_alert_id"].astype("Int64").astype(str),
    pd.NA
)

# ============================================================
# TABLE ALERTES
# ============================================================
print_section("6) CONSTRUCTION TABLE ALERTES")

alert_df = df.loc[df["in_alert_zone"]].copy()

# durée / taille / cible par alerte
alert_summary = (
    alert_df.groupby("alert_key")
    .agg(
        airport=("airport", "first"),
        airport_alert_id=("airport_alert_id", "first"),
        start=("date", "min"),
        end=("date", "max"),
        n_lightnings_alert=("lightning_id", "count"),
        n_cg_alert=("is_cg", "sum"),
        n_ic_alert=("is_ic", "sum"),
        min_dist=("dist", "min"),
        mean_dist=("dist", "mean"),
        max_dist=("dist", "max"),
        last_label_count=("is_last_lightning_cloud_ground", lambda s: (s.fillna(False) == True).sum()),
        last_label_exists=("is_last_lightning_cloud_ground", lambda s: (s.fillna(False) == True).any())
    )
    .reset_index()
)

alert_summary["duration_min"] = (alert_summary["end"] - alert_summary["start"]).dt.total_seconds() / 60.0
alert_summary["cg_ratio"] = alert_summary["n_cg_alert"] / alert_summary["n_lightnings_alert"]
alert_summary["ic_ratio"] = alert_summary["n_ic_alert"] / alert_summary["n_lightnings_alert"]

print(alert_summary.head())
alert_summary.to_csv(OUTPUT_DIR / "alert_summary.csv", index=False)

print("Stats taille alertes:")
print(alert_summary["n_lightnings_alert"].describe())
print("\nStats durée alertes (min):")
print(alert_summary["duration_min"].describe())

print("\nNombre d'alertes sans label True:")
print((~alert_summary["last_label_exists"]).sum())

print("\nDistribution du nombre de True par alerte:")
print(alert_summary["last_label_count"].value_counts().sort_index())

# Graph taille alertes
plt.figure(figsize=(8, 5))
alert_summary["n_lightnings_alert"].clip(upper=alert_summary["n_lightnings_alert"].quantile(0.99)).plot(kind="hist", bins=60)
plt.title("Taille des alertes (# éclairs dans 20km) - clippée au 99e pct")
save_fig("alert_size_hist.png")

# Graph durée alertes
plt.figure(figsize=(8, 5))
alert_summary["duration_min"].clip(upper=alert_summary["duration_min"].quantile(0.99)).plot(kind="hist", bins=60)
plt.title("Durée des alertes (minutes) - clippée au 99e pct")
save_fig("alert_duration_hist.png")

# ============================================================
# VALIDATION DE LA CIBLE
# ============================================================
print_section("7) VALIDATION DE LA CIBLE")

# vrai label unique par alerte ?
target_check = (
    alert_df.groupby("alert_key")["is_last_lightning_cloud_ground"]
    .apply(lambda s: (s.fillna(False) == True).sum())
)
print(target_check.describe())
print("Alertes avec exactement un True:", (target_check == 1).sum())
print("Alertes avec 0 True:", (target_check == 0).sum())
print("Alertes avec >1 True:", (target_check > 1).sum())

# vérifier si True correspond au dernier CG chronologique dans l'alerte
def check_last_cg_consistency(g):
    g = g.sort_values("date")
    cg = g[g["is_cg"] == 1]
    if cg.empty:
        return pd.Series({"has_cg": False, "consistent": pd.NA})
    true_rows = g[g["is_last_lightning_cloud_ground"] == True]
    if true_rows.empty:
        return pd.Series({"has_cg": True, "consistent": False})
    last_cg_idx = cg.index[-1]
    true_idx = true_rows.index[-1]
    return pd.Series({"has_cg": True, "consistent": last_cg_idx == true_idx})

consistency = alert_df.groupby("alert_key").apply(check_last_cg_consistency).reset_index()
print(consistency["consistent"].value_counts(dropna=False))
consistency.to_csv(OUTPUT_DIR / "target_consistency.csv", index=False)

# ============================================================
# TEMPS ENTRE ECLAIRS
# ============================================================
print_section("8) DYNAMIQUE TEMPORELLE GLOBALE")

# gaps entre éclairs par aéroport
df["gap_prev_sec_airport"] = df.groupby("airport")["date"].diff().dt.total_seconds()

# gaps entre éclairs en zone d'alerte par alerte
alert_df = alert_df.sort_values(["alert_key", "date"]).copy()
alert_df["gap_prev_sec_alert"] = alert_df.groupby("alert_key")["date"].diff().dt.total_seconds()

# gap depuis dernier CG par aéroport
df["date_cg_only"] = df["date"].where(df["is_cg"] == 1)
df["last_cg_date_airport"] = df.groupby("airport")["date_cg_only"].ffill()
df["gap_since_last_cg_sec_airport"] = (df["date"] - df["last_cg_date_airport"]).dt.total_seconds()

gap_stats = df["gap_prev_sec_airport"].describe()
print("Stats gap entre éclairs par aéroport:")
print(gap_stats)

plt.figure(figsize=(8, 5))
df["gap_prev_sec_airport"].clip(upper=df["gap_prev_sec_airport"].quantile(0.99)).plot(kind="hist", bins=80)
plt.title("Gap entre éclairs (sec) - global clippé 99e pct")
save_fig("gap_airport_hist.png")

# ============================================================
# ANALYSE DE FIN D'ALERTE
# ============================================================
print_section("9) ANALYSE DE LA FIN D'ALERTE")

# rang chronologique dans alerte
alert_df["rank_in_alert"] = alert_df.groupby("alert_key").cumcount() + 1
alert_df["rank_rev_in_alert"] = alert_df.groupby("alert_key").cumcount(ascending=False) + 1
alert_df["is_target"] = (alert_df["is_last_lightning_cloud_ground"] == True).astype(int)

# distance au dernier événement
alert_df["n_in_alert"] = alert_df.groupby("alert_key")["lightning_id"].transform("count")

# stats des dernières positions dans l'alerte
rank_rev_stats = (
    alert_df.groupby("rank_rev_in_alert")
    .agg(
        n=("lightning_id", "count"),
        mean_dist=("dist", "mean"),
        mean_amplitude=("amplitude", "mean"),
        cg_ratio=("is_cg", "mean"),
        target_rate=("is_target", "mean")
    )
    .reset_index()
    .sort_values("rank_rev_in_alert")
)

rank_rev_stats.to_csv(OUTPUT_DIR / "rank_rev_stats.csv", index=False)
print(rank_rev_stats.head(10))

top_ranks = rank_rev_stats[rank_rev_stats["rank_rev_in_alert"] <= 10]

plt.figure(figsize=(8, 5))
plt.plot(top_ranks["rank_rev_in_alert"], top_ranks["mean_dist"], marker="o")
plt.gca().invert_xaxis()
plt.title("Distance moyenne selon position avant fin d'alerte")
plt.xlabel("rang avant fin (1 = dernière ligne de l'alerte)")
save_fig("mean_dist_by_reverse_rank.png")

plt.figure(figsize=(8, 5))
plt.plot(top_ranks["rank_rev_in_alert"], top_ranks["cg_ratio"], marker="o")
plt.gca().invert_xaxis()
plt.title("Part de cloud-ground selon position avant fin d'alerte")
plt.xlabel("rang avant fin (1 = dernière ligne de l'alerte)")
save_fig("cg_ratio_by_reverse_rank.png")

# ============================================================
# FEATURES HISTORIQUES SIMPLEMENT UTILES
# ============================================================
print_section("10) CREATION FEATURES HISTORIQUES EDA")

# on calcule sur l'historique de chaque aéroport
df = df.sort_values(["airport", "date"]).copy()

# compteurs cumulés utiles
df["cum_count_airport"] = df.groupby("airport").cumcount() + 1
df["cum_cg_airport"] = df.groupby("airport")["is_cg"].cumsum()
df["cum_ic_airport"] = df.groupby("airport")["is_ic"].cumsum()

# rolling time windows
# technique: index temporel par aéroport
feat_list = []

for airport, g in df.groupby("airport", observed=True):
    g = g.sort_values("date").copy()
    g = g.set_index("date")

    # features sur tous les éclairs
    for win in ["5min", "10min", "20min", "30min", "60min"]:
        g[f"cnt_all_{win}"] = g["lightning_id"].rolling(win).count()
        g[f"cnt_cg_{win}"] = g["is_cg"].rolling(win).sum()
        g[f"cnt_ic_{win}"] = g["is_ic"].rolling(win).sum()
        g[f"mean_dist_{win}"] = g["dist"].rolling(win).mean()
        g[f"min_dist_{win}"] = g["dist"].rolling(win).min()
        g[f"max_dist_{win}"] = g["dist"].rolling(win).max()
        g[f"mean_abs_amp_{win}"] = g["amplitude"].abs().rolling(win).mean()
        g[f"mean_maxis_{win}"] = g["maxis"].rolling(win).mean()

        # séparation 20 km / 20-30 km
        g["_lt20"] = (g["dist"] <= 20).astype(int)
        g["_20_30"] = ((g["dist"] > 20) & (g["dist"] <= 30)).astype(int)

        g[f"cnt_lt20_{win}"] = g["_lt20"].rolling(win).sum()
        g[f"cnt_20_30_{win}"] = g["_20_30"].rolling(win).sum()

    g = g.reset_index()
    feat_list.append(g)

df_feat = pd.concat(feat_list, ignore_index=True)

# gap depuis dernier éclair/CG dans 20km
df_feat["date_lt20"] = df_feat["date"].where(df_feat["dist"] <= 20)
df_feat["last_lt20_date_airport"] = df_feat.groupby("airport")["date_lt20"].ffill()
df_feat["gap_since_last_lt20_sec"] = (df_feat["date"] - df_feat["last_lt20_date_airport"]).dt.total_seconds()

df_feat["date_cg_lt20"] = df_feat["date"].where((df_feat["dist"] <= 20) & (df_feat["is_cg"] == 1))
df_feat["last_cg_lt20_date_airport"] = df_feat.groupby("airport")["date_cg_lt20"].ffill()
df_feat["gap_since_last_cg_lt20_sec"] = (df_feat["date"] - df_feat["last_cg_lt20_date_airport"]).dt.total_seconds()

# ne garder que les lignes supervisées pour analyse label
model_df = df_feat[df_feat["in_alert_zone"]].copy()
model_df["target"] = (model_df["is_last_lightning_cloud_ground"] == True).astype(int)

print("Shape model_df:", model_df.shape)
print(model_df["target"].value_counts())

# ============================================================
# CORRELATION UNIVARIEE AVEC LA CIBLE
# ============================================================
print_section("11) FEATURES LES PLUS DISCRIMINANTES")

candidate_features = [
    "dist", "azimuth", "amplitude", "maxis",
    "gap_since_last_lt20_sec", "gap_since_last_cg_lt20_sec",
    "cnt_all_5min", "cnt_all_10min", "cnt_all_20min", "cnt_all_30min",
    "cnt_cg_5min", "cnt_cg_10min", "cnt_cg_20min", "cnt_cg_30min",
    "cnt_lt20_5min", "cnt_lt20_10min", "cnt_lt20_20min", "cnt_lt20_30min",
    "cnt_20_30_5min", "cnt_20_30_10min", "cnt_20_30_20min", "cnt_20_30_30min",
    "mean_dist_5min", "mean_dist_10min", "mean_dist_20min", "mean_dist_30min",
    "min_dist_5min", "min_dist_10min", "min_dist_20min", "min_dist_30min",
    "mean_abs_amp_5min", "mean_abs_amp_10min", "mean_abs_amp_20min", "mean_abs_amp_30min",
    "mean_maxis_5min", "mean_maxis_10min", "mean_maxis_20min", "mean_maxis_30min",
]

existing_features = [c for c in candidate_features if c in model_df.columns]

univariate_rows = []
for col in existing_features:
    s0 = model_df.loc[model_df["target"] == 0, col]
    s1 = model_df.loc[model_df["target"] == 1, col]
    univariate_rows.append({
        "feature": col,
        "mean_target0": s0.mean(),
        "mean_target1": s1.mean(),
        "median_target0": s0.median(),
        "median_target1": s1.median(),
        "diff_mean": s1.mean() - s0.mean(),
        "abs_diff_mean": abs(s1.mean() - s0.mean()),
    })

univariate_df = pd.DataFrame(univariate_rows).sort_values("abs_diff_mean", ascending=False)
print(univariate_df.head(20))
univariate_df.to_csv(OUTPUT_DIR / "univariate_feature_signal.csv", index=False)

# Boxplots top features
top_plot_features = univariate_df["feature"].head(8).tolist()
for col in top_plot_features:
    plt.figure(figsize=(6, 4))
    data0 = model_df.loc[model_df["target"] == 0, col].dropna()
    data1 = model_df.loc[model_df["target"] == 1, col].dropna()

    # clip pour lisibilité
    if not data0.empty and not data1.empty:
        upper = pd.concat([data0, data1]).quantile(0.99)
        data0 = data0.clip(upper=upper)
        data1 = data1.clip(upper=upper)

    plt.boxplot([data0, data1], labels=["target_0", "target_1"], showfliers=False)
    plt.title(f"Distribution de {col} selon la cible")
    save_fig(f"box_{col}.png")

# ============================================================
# SAISONNALITE / AEROPORTS
# ============================================================
print_section("12) SAISONNALITE ET HETEROGENEITE")

# Taux d'alertes et taille par saison
alert_summary["year"] = alert_summary["start"].dt.year
alert_summary["month"] = alert_summary["start"].dt.month
alert_summary["season"] = alert_summary["month"].map(month_to_season)

season_stats = (
    alert_summary.groupby("season")
    .agg(
        n_alerts=("alert_key", "count"),
        mean_size=("n_lightnings_alert", "mean"),
        median_size=("n_lightnings_alert", "median"),
        mean_duration=("duration_min", "mean"),
        median_duration=("duration_min", "median"),
    )
    .sort_values("n_alerts", ascending=False)
)
print(season_stats)
season_stats.to_csv(OUTPUT_DIR / "season_stats.csv")

airport_stats = (
    alert_summary.groupby("airport")
    .agg(
        n_alerts=("alert_key", "count"),
        mean_size=("n_lightnings_alert", "mean"),
        median_size=("n_lightnings_alert", "median"),
        mean_duration=("duration_min", "mean"),
        median_duration=("duration_min", "median"),
    )
    .sort_values("n_alerts", ascending=False)
)
print(airport_stats)
airport_stats.to_csv(OUTPUT_DIR / "airport_alert_stats.csv")

# heatmap-like pivot via table
airport_season = (
    alert_summary.pivot_table(
        index="airport",
        columns="season",
        values="alert_key",
        aggfunc="count",
        fill_value=0
    )
)
print(airport_season)
airport_season.to_csv(OUTPUT_DIR / "airport_season_alert_counts.csv")

# Alerts by year
alerts_by_year = alert_summary["year"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
alerts_by_year.plot(marker="o")
plt.title("Nombre d'alertes par année")
save_fig("alerts_by_year.png")

# ============================================================
# PISE 2016
# ============================================================
print_section("13) FOCUS PISE 2016")

pisa_2016 = df[(df["airport"] == "Pise") & (df["year"] == 2016)]
pisa_other = df[(df["airport"] == "Pise") & (df["year"] != 2016)]

if len(pisa_2016) > 0 and len(pisa_other) > 0:
    pisa_compare = pd.DataFrame({
        "pise_2016": {
            "n": len(pisa_2016),
            "icloud_rate": pisa_2016["is_ic"].mean(),
            "cg_rate": pisa_2016["is_cg"].mean(),
            "mean_dist": pisa_2016["dist"].mean(),
            "mean_abs_amp": pisa_2016["amplitude"].abs().mean(),
        },
        "pise_other_years": {
            "n": len(pisa_other),
            "icloud_rate": pisa_other["is_ic"].mean(),
            "cg_rate": pisa_other["is_cg"].mean(),
            "mean_dist": pisa_other["dist"].mean(),
            "mean_abs_amp": pisa_other["amplitude"].abs().mean(),
        }
    })
    print(pisa_compare)
    pisa_compare.to_csv(OUTPUT_DIR / "pise_2016_compare.csv")

# ============================================================
# BASELINES STRATEGIQUES
# ============================================================
print_section("14) INDICATEURS POUR LA STRATEGIE")

# fréquences cibles par contexte
context_rates = {
    "global_target_rate": model_df["target"].mean(),
    "target_rate_by_airport": model_df.groupby("airport")["target"].mean().to_dict(),
    "target_rate_by_season": model_df.groupby("season")["target"].mean().to_dict(),
}

print(context_rates)

# quelques tables de décision utiles
decile_candidates = [
    "gap_since_last_cg_lt20_sec",
    "cnt_cg_10min",
    "cnt_lt20_10min",
    "min_dist_10min",
    "mean_dist_10min"
]

for col in decile_candidates:
    if col in model_df.columns and model_df[col].notna().sum() > 0:
        tmp = model_df[[col, "target"]].dropna().copy()
        try:
            tmp["bin"] = pd.qcut(tmp[col], q=10, duplicates="drop")
            dec = tmp.groupby("bin", observed=False)["target"].agg(["count", "mean"]).reset_index()
            dec.to_csv(OUTPUT_DIR / f"deciles_{col}.csv", index=False)
            print(f"\nDéciles pour {col}:")
            print(dec)
        except Exception as e:
            print(f"Impossible de faire qcut pour {col}: {e}")

# ============================================================
# RAPPORT TEXTE FINAL
# ============================================================
print_section("15) RAPPORT SYNTHETIQUE")

report_lines = []

report_lines.append(f"Nombre total de lignes: {len(df)}")
report_lines.append(f"Nombre total d'aéroports: {df['airport'].nunique()}")
report_lines.append(f"Part de lignes avec alert_id (zone 20 km): {df['in_alert_zone'].mean():.4f}")
report_lines.append(f"Nombre total d'alertes: {alert_summary['alert_key'].nunique()}")
report_lines.append(f"Taille médiane des alertes: {alert_summary['n_lightnings_alert'].median():.2f}")
report_lines.append(f"Durée médiane des alertes (min): {alert_summary['duration_min'].median():.2f}")
report_lines.append(f"Alertes avec exactement 1 True: {(target_check == 1).sum()}")
report_lines.append(f"Alertes avec 0 True: {(target_check == 0).sum()}")
report_lines.append(f"Alertes avec >1 True: {(target_check > 1).sum()}")
report_lines.append("Top features univariées (différence de moyenne absolue) :")
for _, row in univariate_df.head(10).iterrows():
    report_lines.append(
        f"  - {row['feature']}: mean(target=0)={row['mean_target0']:.4f}, "
        f"mean(target=1)={row['mean_target1']:.4f}, diff={row['diff_mean']:.4f}"
    )

report_lines.append("\nRecommandations stratégiques provisoires :")
report_lines.append("1. Ne superviser que sur les lignes avec airport_alert_id non nul.")
report_lines.append("2. Utiliser tous les éclairs du même aéroport pour fabriquer les features historiques.")
report_lines.append("3. Séparer explicitement le signal 0-20 km et 20-30 km.")
report_lines.append("4. Faire la validation par années, jamais en aléatoire.")
report_lines.append("5. Tester un modèle global avec airport + saison + historique récent.")
report_lines.append("6. Tester la robustesse avec et sans Pise 2016 pour les features liées à icloud.")

report_path = OUTPUT_DIR / "rapport_eda.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n".join(report_lines))

print_section("TERMINE")
print(f"Tous les outputs sont dans: {OUTPUT_DIR.resolve()}")