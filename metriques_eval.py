"""
Calcul rapide des métriques sur l'eval externe.
Charge le modèle déjà entraîné + les prédictions.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

OUTPUT_DIR = "catboost_global_outputs_v4"

# Charger les prédictions eval
preds = pd.read_csv(f"{OUTPUT_DIR}/eval_predictions.csv")
print(f"Prédictions eval: {len(preds)} lignes")

# Charger la base eval pour récupérer les labels
df_eval = pd.read_csv("data/segment_alerts_all_airports_eval.csv")
df_eval["is_last_lightning_cloud_ground"] = df_eval["is_last_lightning_cloud_ground"].fillna(False)

# Merge pour associer labels et prédictions
merged = preds.merge(
    df_eval[["lightning_id", "is_last_lightning_cloud_ground"]].drop_duplicates(),
    on="lightning_id",
    how="left"
)

y_true = merged["is_last_lightning_cloud_ground"].astype(int)
y_pred = merged["pred"]

print(f"\nTarget rate: {y_true.mean():.4f}")
print(f"N positifs: {y_true.sum()}, N négatifs: {(1 - y_true).sum()}")

# Métriques globales
auc = roc_auc_score(y_true, y_pred)
ap = average_precision_score(y_true, y_pred)
ll = log_loss(y_true, y_pred, labels=[0, 1])
bs = brier_score_loss(y_true, y_pred)

print(f"\n{'='*50}")
print(f"EVAL EXTERNE (2023-2025)")
print(f"{'='*50}")
print(f"  AUC    : {auc:.4f}")
print(f"  AP     : {ap:.4f}")
print(f"  LogLoss: {ll:.4f}")
print(f"  Brier  : {bs:.4f}")

# Métriques par aéroport
print(f"\n{'='*50}")
print(f"PAR AÉROPORT")
print(f"{'='*50}")
for airport, grp in merged.groupby("airport"):
    y_a = grp["is_last_lightning_cloud_ground"].astype(int)
    if y_a.nunique() < 2:
        print(f"  {airport}: n={len(grp)}, target_rate={y_a.mean():.4f} — pas assez de classes")
        continue
    print(f"  {airport}: n={len(grp)}, target_rate={y_a.mean():.4f}, "
          f"AUC={roc_auc_score(y_a, grp['pred']):.4f}, "
          f"AP={average_precision_score(y_a, grp['pred']):.4f}")

# Analyse intra-alerte
print(f"\n{'='*50}")
print(f"RANKING INTRA-ALERTE")
print(f"{'='*50}")
merged["alert_key"] = merged["airport"] + "__" + merged["airport_alert_id"].astype(str)

def alert_analysis(g):
    y = g["is_last_lightning_cloud_ground"].astype(int)
    if y.sum() == 0:
        return None
    target_idx = g[y == 1].index[0]
    target_pred = g.loc[target_idx, "pred"]
    is_top1 = g["pred"].idxmax() == target_idx
    rank = (g["pred"] >= target_pred).sum()
    return {"is_top1": is_top1, "rank": rank, "size": len(g), "target_pred": target_pred}

results = []
for key, grp in merged.groupby("alert_key"):
    r = alert_analysis(grp)
    if r:
        results.append(r)

ra = pd.DataFrame(results)
print(f"  Alertes analysées: {len(ra)}")
print(f"  Target = top-1 pred: {ra['is_top1'].mean():.2%}")
print(f"  Rang moyen du target: {ra['rank'].mean():.1f}")
print(f"  Pred moyenne sur targets: {ra['target_pred'].mean():.4f}")

# Par taille d'alerte
ra["size_bin"] = pd.cut(ra["size"], bins=[0, 2, 5, 10, 30, 100, 10000],
                        labels=["1-2", "3-5", "6-10", "11-30", "31-100", "100+"])
print(f"\n  Par taille d'alerte:")
for bin_name, grp in ra.groupby("size_bin", observed=True):
    print(f"    {bin_name}: n={len(grp)}, top1={grp['is_top1'].mean():.0%}, "
          f"rang_moy={grp['rank'].mean():.1f}, pred_moy={grp['target_pred'].mean():.4f}")