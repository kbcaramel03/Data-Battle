import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
 
OUTPUT_DIR = "catboost_survival_v6"
 
# Charger
ep = pd.read_csv(f"{OUTPUT_DIR}/eval_predictions_survival.csv")
print(f"Épisodes eval: {len(ep)}")
print(f"Target rate global: {ep['target_at_t'].mean():.4f}")
 
# ============================================================
# 1. MÉTRIQUES GLOBALES
# ============================================================
print("\n" + "=" * 60)
print("1. MÉTRIQUES GLOBALES (tous T confondus)")
print("=" * 60)
y, p = ep["target_at_t"], ep["pred"]
print(f"  AUC:     {roc_auc_score(y, p):.4f}")
print(f"  AP:      {average_precision_score(y, p):.4f}")
print(f"  LogLoss: {log_loss(y, p):.4f}")
print(f"  Brier:   {brier_score_loss(y, p):.4f}")
 
# ============================================================
# 2. PAR HORIZON DE SILENCE
# ============================================================
print("\n" + "=" * 60)
print("2. PAR HORIZON DE SILENCE")
print("=" * 60)
print(f"  {'T(min)':>6s} {'n':>7s} {'target%':>8s} {'pred_moy':>9s} {'AUC':>8s} {'AP':>8s}")
for t, grp in ep.groupby("check_time_min"):
    yt, pt = grp["target_at_t"], grp["pred"]
    if yt.nunique() < 2:
        continue
    print(f"  {t:6.0f} {len(grp):7d} {yt.mean():8.2%} {pt.mean():9.3f} "
          f"{roc_auc_score(yt, pt):8.4f} {average_precision_score(yt, pt):8.4f}")
 
# ============================================================
# 3. PAR AÉROPORT (tous T confondus)
# ============================================================
print("\n" + "=" * 60)
print("3. PAR AÉROPORT (tous T confondus)")
print("=" * 60)
print(f"  {'Airport':>10s} {'n':>7s} {'target%':>8s} {'AUC':>8s} {'AP':>8s}")
for airport, grp in ep.groupby("airport"):
    yt, pt = grp["target_at_t"], grp["pred"]
    if yt.nunique() < 2:
        continue
    print(f"  {airport:>10s} {len(grp):7d} {yt.mean():8.2%} "
          f"{roc_auc_score(yt, pt):8.4f} {average_precision_score(yt, pt):8.4f}")
 
# ============================================================
# 4. PAR AÉROPORT × HORIZON (T=10min)
# ============================================================
print("\n" + "=" * 60)
print("4. PAR AÉROPORT à T=10min")
print("=" * 60)
t10 = ep[ep["check_time_min"] == 10]
print(f"  {'Airport':>10s} {'n':>6s} {'target%':>8s} {'pred_moy':>9s} {'AUC':>8s} {'AP':>8s}")
for airport, grp in t10.groupby("airport"):
    yt, pt = grp["target_at_t"], grp["pred"]
    if yt.nunique() < 2:
        print(f"  {airport:>10s} {len(grp):6d} {yt.mean():8.2%} {pt.mean():9.3f}      -        -")
        continue
    print(f"  {airport:>10s} {len(grp):6d} {yt.mean():8.2%} {pt.mean():9.3f} "
          f"{roc_auc_score(yt, pt):8.4f} {average_precision_score(yt, pt):8.4f}")
 
# ============================================================
# 5. PAR SAISON (tous T confondus)
# ============================================================
print("\n" + "=" * 60)
print("5. PAR SAISON (tous T confondus)")
print("=" * 60)
print(f"  {'Saison':>10s} {'n':>7s} {'target%':>8s} {'AUC':>8s} {'AP':>8s}")
for season, grp in ep.groupby("season"):
    yt, pt = grp["target_at_t"], grp["pred"]
    if yt.nunique() < 2:
        continue
    print(f"  {season:>10s} {len(grp):7d} {yt.mean():8.2%} "
          f"{roc_auc_score(yt, pt):8.4f} {average_precision_score(yt, pt):8.4f}")
 
# ============================================================
# 6. PAR SAISON à T=10min
# ============================================================
print("\n" + "=" * 60)
print("6. PAR SAISON à T=10min")
print("=" * 60)
print(f"  {'Saison':>10s} {'n':>6s} {'target%':>8s} {'pred_moy':>9s} {'AUC':>8s}")
for season, grp in t10.groupby("season"):
    yt, pt = grp["target_at_t"], grp["pred"]
    if yt.nunique() < 2:
        print(f"  {season:>10s} {len(grp):6d} {yt.mean():8.2%} {pt.mean():9.3f}      -")
        continue
    print(f"  {season:>10s} {len(grp):6d} {yt.mean():8.2%} {pt.mean():9.3f} "
          f"{roc_auc_score(yt, pt):8.4f}")
 
# ============================================================
# 7. COURBE DE SURVIE MOYENNE
# ============================================================
print("\n" + "=" * 60)
print("7. COURBE DE SURVIE MOYENNE (pred et réel)")
print("=" * 60)
print(f"  {'T(min)':>6s} {'P(fini) pred':>13s} {'P(fini) réel':>13s} {'écart':>8s}")
for t, grp in ep.groupby("check_time_min"):
    pred_m = grp["pred"].mean()
    real_m = grp["target_at_t"].mean()
    print(f"  {t:6.0f} {pred_m:13.3f} {real_m:13.3f} {pred_m - real_m:8.3f}")
 
# ============================================================
# 8. COURBE PAR SAISON
# ============================================================
print("\n" + "=" * 60)
print("8. COURBES DE SURVIE PAR SAISON")
print("=" * 60)
for season in ["winter", "spring", "summer", "autumn"]:
    sub = ep[ep["season"] == season]
    if len(sub) == 0:
        continue
    print(f"\n  {season.upper()}:")
    print(f"    {'T':>4s} {'pred':>7s} {'réel':>7s}")
    for t, grp in sub.groupby("check_time_min"):
        print(f"    {t:4.0f} {grp['pred'].mean():7.3f} {grp['target_at_t'].mean():7.3f}")
 
# ============================================================
# 9. COURBE PAR TAILLE D'ALERTE
# ============================================================
print("\n" + "=" * 60)
print("9. COURBES PAR TAILLE D'ALERTE (à T=10min)")
print("=" * 60)
t10 = t10.copy()
t10["size_bin"] = pd.cut(t10["rank_in_alert"], bins=[0, 3, 10, 50, 99999],
                         labels=["1-3", "4-10", "11-50", "50+"])
print(f"  {'Taille':>8s} {'n':>6s} {'pred_moy':>9s} {'réel':>7s}")
for b, grp in t10.groupby("size_bin", observed=True):
    print(f"  {b:>8s} {len(grp):6d} {grp['pred'].mean():9.3f} {grp['target_at_t'].mean():7.3f}")
 
# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("10. FEATURE IMPORTANCE (top 20)")
print("=" * 60)
try:
    fi = pd.read_csv(f"{OUTPUT_DIR}/feature_importance.csv")
    fi = fi.sort_values("importance", ascending=False)
    print(fi.head(20).to_string(index=False))
except:
    print("  feature_importance.csv non trouvé")
 
# ============================================================
# 11. CALIBRATION
# ============================================================
print("\n" + "=" * 60)
print("11. CALIBRATION (pred vs réel par décile de proba)")
print("=" * 60)
ep["pred_bin"] = pd.qcut(ep["pred"], 10, duplicates="drop")
cal = ep.groupby("pred_bin", observed=True).agg(
    n=("target_at_t", "count"),
    pred_mean=("pred", "mean"),
    real_mean=("target_at_t", "mean")
).reset_index()
print(f"  {'Bin':>25s} {'n':>7s} {'pred':>7s} {'réel':>7s} {'écart':>7s}")
for _, row in cal.iterrows():
    print(f"  {str(row['pred_bin']):>25s} {row['n']:7.0f} {row['pred_mean']:7.3f} "
          f"{row['real_mean']:7.3f} {row['pred_mean'] - row['real_mean']:7.3f}")
 