"""
Génère tous les graphiques de la présentation en PNG.
Lance : python generate_charts.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = "catboost_survival_v6"
CHART_DIR = "charts_presentation"

import os
os.makedirs(CHART_DIR, exist_ok=True)

# Charger les données
ep = pd.read_csv(f"{OUTPUT_DIR}/eval_predictions_survival.csv")

# Style commun
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

NAVY = '#1E2761'
TEAL = '#0D9488'
GRAY = '#94A3B8'
BLUE = '#3B82F6'
RED = '#EF4444'
GREEN = '#22C55E'
AMBER = '#F59E0B'

# ============================================================
# 1. COURBE DE SURVIE MOYENNE (prédit vs réel)
# ============================================================
print("1. Courbe de survie moyenne...")

curve = ep.groupby("check_time_min").agg(
    pred=("pred", "mean"),
    real=("target_at_t", "mean")
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(curve["check_time_min"], curve["pred"], 'o-', color=TEAL, linewidth=2.5, markersize=7, label="Prédit par le modèle")
ax.plot(curve["check_time_min"], curve["real"], 'o-', color=GRAY, linewidth=2.5, markersize=7, label="Réel observé")
ax.fill_between(curve["check_time_min"], curve["pred"], curve["real"], alpha=0.1, color=TEAL)
ax.set_xlabel("Minutes de silence après le dernier éclair", fontsize=13)
ax.set_ylabel("P(orage terminé)", fontsize=13)
ax.set_title("Courbe de survie moyenne dans les données d'évaluation 2023/2025", fontsize=16, fontweight='bold', color=NAVY)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 31)
ax.legend(fontsize=12, loc='lower right')
ax.set_xticks(curve["check_time_min"])
for i, row in curve.iterrows():
    ax.annotate(f'{row["pred"]:.0%}', (row["check_time_min"], row["pred"]),
                textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9, color=TEAL)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_courbe_survie_moyenne.png", dpi=200)
plt.close()

# ============================================================
# 2. COURBES PAR SAISON
# ============================================================
print("2. Courbes par saison...")

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"winter": BLUE, "summer": RED, "spring": GREEN, "autumn": AMBER}
labels_fr = {"winter": "Hiver", "summer": "Été", "spring": "Printemps", "autumn": "Automne"}

for season in ["winter", "spring", "autumn", "summer"]:
    sub = ep[ep["season"] == season]
    curve_s = sub.groupby("check_time_min")["pred"].mean().reset_index()
    ax.plot(curve_s["check_time_min"], curve_s["pred"], 'o-',
            color=colors[season], linewidth=2.5, markersize=6, label=labels_fr[season])

ax.set_xlabel("Minutes de silence", fontsize=13)
ax.set_ylabel("P(orage terminé) prédit", fontsize=13)
ax.set_title("Courbes de survie par saison dans les données d'évaluation 2023/2025", fontsize=16, fontweight='bold', color=NAVY)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 31)
ax.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_courbes_par_saison.png", dpi=200)
plt.close()

# ============================================================
# 3. COURBES PAR SAISON — RÉEL (pour comparaison)
# ============================================================
print("3. Courbes par saison (réel)...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, season in enumerate(["winter", "spring", "summer", "autumn"]):
    ax = axes[idx // 2][idx % 2]
    sub = ep[ep["season"] == season]
    curve_s = sub.groupby("check_time_min").agg(pred=("pred", "mean"), real=("target_at_t", "mean")).reset_index()
    ax.plot(curve_s["check_time_min"], curve_s["pred"], 'o-', color=colors[season], linewidth=2, markersize=5, label="Prédit")
    ax.plot(curve_s["check_time_min"], curve_s["real"], 'o--', color=GRAY, linewidth=2, markersize=5, label="Réel")
    ax.set_title(labels_fr[season], fontsize=14, fontweight='bold', color=colors[season])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Min de silence")
    ax.set_ylabel("P(fini)")
    ax.legend(fontsize=10)

plt.suptitle("Prédit vs Réel par saison — Eval 2023-2025", fontsize=16, fontweight='bold', color=NAVY, y=1.01)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_predit_vs_reel_par_saison.png", dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# 4. BARPLOT AP PAR AÉROPORT
# ============================================================
print("4. AP par aéroport...")
from sklearn.metrics import average_precision_score, roc_auc_score

airports = []
for airport, grp in ep.groupby("airport"):
    yt, pt = grp["target_at_t"], grp["pred"]
    if yt.nunique() < 2:
        continue
    airports.append({
        "airport": airport,
        "auc": roc_auc_score(yt, pt),
        "ap": average_precision_score(yt, pt),
        "n": len(grp)
    })
adf = pd.DataFrame(airports).sort_values("ap", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(adf["airport"], adf["ap"], color=TEAL, height=0.5)
ax.set_xlabel("Average Precision", fontsize=13)
ax.set_title("AP par aéroport — Eval 2023-2025", fontsize=16, fontweight='bold', color=NAVY)
ax.set_xlim(0, 0.85)
for bar, val in zip(bars, adf["ap"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', fontsize=11, color=NAVY, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_ap_par_aeroport.png", dpi=200)
plt.close()

# ============================================================
# 5. BARPLOT AUC PAR AÉROPORT
# ============================================================
print("5. AUC par aéroport...")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(adf["airport"], adf["auc"], color=NAVY, height=0.5)
ax.set_xlabel("AUC", fontsize=13)
ax.set_title("AUC par aéroport — Eval 2023-2025", fontsize=16, fontweight='bold', color=NAVY)
ax.set_xlim(0, 0.9)
for bar, val in zip(bars, adf["auc"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', fontsize=11, color=NAVY, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_auc_par_aeroport.png", dpi=200)
plt.close()

# ============================================================
# 6. CONTEXTE : PRED À T=10 PAR SAISON ET TAILLE
# ============================================================
print("6. Contexte à T=10min...")

t10 = ep[ep["check_time_min"] == 10].copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Par saison
season_data = t10.groupby("season").agg(pred=("pred", "mean"), real=("target_at_t", "mean")).reindex(["winter", "spring", "autumn", "summer"])
x = np.arange(len(season_data))
w = 0.35
axes[0].bar(x - w/2, season_data["pred"], w, color=TEAL, label="Prédit")
axes[0].bar(x + w/2, season_data["real"], w, color=GRAY, label="Réel")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Hiver", "Printemps", "Automne", "Été"])
axes[0].set_ylabel("P(fini) à T=10min")
axes[0].set_title("Par saison", fontsize=14, fontweight='bold', color=NAVY)
axes[0].legend()
axes[0].set_ylim(0, 0.85)
for i, (p, r) in enumerate(zip(season_data["pred"], season_data["real"])):
    axes[0].text(i - w/2, p + 0.02, f'{p:.0%}', ha='center', fontsize=10, color=TEAL, fontweight='bold')
    axes[0].text(i + w/2, r + 0.02, f'{r:.0%}', ha='center', fontsize=10, color=GRAY, fontweight='bold')

# Par taille d'alerte
t10["size_bin"] = pd.cut(t10["rank_in_alert"], bins=[0, 3, 10, 50, 99999],
                         labels=["1-3", "4-10", "11-50", "50+"])
size_data = t10.groupby("size_bin", observed=True).agg(pred=("pred", "mean"), real=("target_at_t", "mean"))
x2 = np.arange(len(size_data))
axes[1].bar(x2 - w/2, size_data["pred"], w, color=TEAL, label="Prédit")
axes[1].bar(x2 + w/2, size_data["real"], w, color=GRAY, label="Réel")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(size_data.index)
axes[1].set_xlabel("Taille de l'alerte (nb éclairs)")
axes[1].set_ylabel("P(fini) à T=10min")
axes[1].set_title("Par taille d'alerte", fontsize=14, fontweight='bold', color=NAVY)
axes[1].legend()
axes[1].set_ylim(0, 0.85)
for i, (p, r) in enumerate(zip(size_data["pred"], size_data["real"])):
    axes[1].text(i - w/2, p + 0.02, f'{p:.0%}', ha='center', fontsize=10, color=TEAL, fontweight='bold')
    axes[1].text(i + w/2, r + 0.02, f'{r:.0%}', ha='center', fontsize=10, color=GRAY, fontweight='bold')

plt.suptitle("Adaptation au contexte — T=10 min de silence", fontsize=16, fontweight='bold', color=NAVY)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_contexte_t10.png", dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print("7. Feature importance...")

try:
    fi = pd.read_csv(f"{OUTPUT_DIR}/feature_importance.csv")
    fi = fi.sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    bar_colors = [AMBER if "check_time" in f or "log_check" in f
                  else TEAL if "gap" in f
                  else NAVY for f in fi["feature"]]
    bars = ax.barh(fi["feature"], fi["importance"], color=bar_colors, height=0.6)
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_title("Top 15 features", fontsize=16, fontweight='bold', color=NAVY)

    # Légende manuelle
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=AMBER, label='Temps de silence (check_time)'),
        Patch(facecolor=TEAL, label='Dynamique des gaps'),
        Patch(facecolor=NAVY, label='Contexte et propriétés'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/07_feature_importance.png", dpi=200)
    plt.close()
except Exception as e:
    print(f"  Erreur feature importance: {e}")

# ============================================================
# 8. CALIBRATION
# ============================================================
print("8. Calibration...")

ep_cal = ep.copy()
ep_cal["pred_bin"] = pd.qcut(ep_cal["pred"], 10, duplicates="drop")
cal = ep_cal.groupby("pred_bin", observed=True).agg(
    pred_mean=("pred", "mean"),
    real_mean=("target_at_t", "mean")
).reset_index()

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], '--', color=GRAY, linewidth=1, label="Calibration parfaite")
ax.plot(cal["pred_mean"], cal["real_mean"], 'o-', color=TEAL, linewidth=2.5, markersize=8, label="Notre modèle")
ax.set_xlabel("Probabilité prédite (moyenne par décile)", fontsize=13)
ax.set_ylabel("Taux réel observé", fontsize=13)
ax.set_title("Calibration du modèle", fontsize=16, fontweight='bold', color=NAVY)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=12)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_calibration.png", dpi=200)
plt.close()

# ============================================================
# 9. DISTRIBUTION DES ALERTES (EDA)
# ============================================================
print("9. Distribution des alertes...")

df_raw = pd.read_csv("data/segment_alerts_all_airports_train.csv")
df_raw["airport_alert_id"] = df_raw["airport_alert_id"] if "airport_alert_id" in df_raw.columns else df_raw["alert_airport_id"]
alert_sizes = df_raw[df_raw["airport_alert_id"].notna()].groupby(
    ["airport", "airport_alert_id"]).size().reset_index(name="size")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogramme tailles
axes[0].hist(alert_sizes["size"].clip(upper=50), bins=50, color=TEAL, edgecolor='white')
axes[0].set_xlabel("Nombre d'éclairs par alerte (tronqué à 50)")
axes[0].set_ylabel("Nombre d'alertes")
axes[0].set_title("Distribution des tailles d'alertes", fontsize=14, fontweight='bold', color=NAVY)
axes[0].axvline(alert_sizes["size"].median(), color=RED, linestyle='--', label=f'Médiane = {alert_sizes["size"].median():.0f}')
axes[0].legend()

# Par aéroport
airport_counts = alert_sizes.groupby("airport").size().sort_values(ascending=True)
axes[1].barh(airport_counts.index, airport_counts.values, color=NAVY, height=0.5)
axes[1].set_xlabel("Nombre d'alertes")
axes[1].set_title("Alertes par aéroport (train)", fontsize=14, fontweight='bold', color=NAVY)
for i, v in enumerate(airport_counts.values):
    axes[1].text(v + 5, i, str(v), va='center', fontsize=11, color=NAVY, fontweight='bold')

plt.suptitle("Exploration des données", fontsize=16, fontweight='bold', color=NAVY, y=1.01)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/09_eda_alertes.png", dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# 10. SAISONNALITÉ (EDA)
# ============================================================
print("10. Saisonnalité...")

df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
df_raw["month"] = df_raw["date"].dt.month

def to_season(m):
    if m in [12, 1, 2]: return "Hiver"
    if m in [3, 4, 5]: return "Printemps"
    if m in [6, 7, 8]: return "Été"
    return "Automne"

alert_data = df_raw[df_raw["airport_alert_id"].notna()].copy()
alert_data["season"] = alert_data["month"].map(to_season)
alert_data["alert_key"] = alert_data["airport"].astype(str) + "__" + alert_data["airport_alert_id"].astype(int).astype(str)

season_stats = alert_data.groupby("alert_key").agg(
    season=("season", "first"),
    size=("airport_alert_id", "size"),
).groupby("season").agg(
    n_alerts=("size", "count"),
    median_size=("size", "median"),
).reindex(["Hiver", "Printemps", "Été", "Automne"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
season_colors = [BLUE, GREEN, RED, AMBER]

axes[0].bar(season_stats.index, season_stats["n_alerts"], color=season_colors)
axes[0].set_ylabel("Nombre d'alertes")
axes[0].set_title("Alertes par saison", fontsize=14, fontweight='bold', color=NAVY)
for i, v in enumerate(season_stats["n_alerts"]):
    axes[0].text(i, v + 10, str(v), ha='center', fontsize=11, fontweight='bold')

axes[1].bar(season_stats.index, season_stats["median_size"], color=season_colors)
axes[1].set_ylabel("Taille médiane (éclairs)")
axes[1].set_title("Taille médiane par saison", fontsize=14, fontweight='bold', color=NAVY)
for i, v in enumerate(season_stats["median_size"]):
    axes[1].text(i, v + 0.1, f'{v:.0f}', ha='center', fontsize=11, fontweight='bold')

plt.suptitle("Saisonnalité des orages", fontsize=16, fontweight='bold', color=NAVY, y=1.01)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/10_saisonnalite.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"\nTerminé ! {len(os.listdir(CHART_DIR))} graphiques dans {CHART_DIR}/")
print("Fichiers :")
for f in sorted(os.listdir(CHART_DIR)):
    print(f"  {f}")