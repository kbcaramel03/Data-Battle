"""
MODÈLE V6 — Courbe de survie : P(orage fini | T minutes de silence)
=====================================================================

REFORMULATION OPÉRATIONNELLE :
L'aéroport ne veut pas savoir "est-ce le dernier éclair ?" au moment où
il tombe. Il veut savoir : "ça fait 5/10/15/20 minutes qu'on n'a pas eu
d'éclair — est-ce qu'on peut reprendre ?"

APPROCHE :
Pour chaque alerte, on connaît le gap entre le dernier éclair et le
précédent (= durée de silence avant fin réelle). On connaît aussi les
gaps entre tous les autres éclairs (= silences temporaires).

On construit un dataset :
- Pour chaque gap dans une alerte, on a un "épisode de silence"
- Features = contexte au moment où le silence commence
- Target = 1 si c'est le silence final (après le dernier éclair), 0 sinon

Ensuite on peut prédire : étant donné le contexte actuel,
quelle est la proba que le silence en cours soit le silence final ?

AVANTAGE DÉCISIF :
On peut maintenant inclure la DURÉE DU SILENCE EN COURS comme feature !
C'est le signal qu'on n'avait pas avant. Plus le silence dure, plus
la proba que l'orage soit fini augmente — et le modèle apprend la
vitesse à laquelle cette proba monte en fonction du contexte.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    f1_score, precision_recall_curve
)

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "data/segment_alerts_all_airports_train.csv"
EVAL_CSV_PATH = "data/segment_alerts_all_airports_eval.csv"
OUTPUT_DIR = Path("catboost_survival_v6")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

USE_PISE_2016 = False
RANDOM_SEED = 42

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def load_and_prepare(csv_path, remove_pise_2016=False):
    """Charge et prépare les données brutes."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if "alert_airport_id" in df.columns and "airport_alert_id" not in df.columns:
        df = df.rename(columns={"alert_airport_id": "airport_alert_id"})
    
    df = df.sort_values(["airport", "date", "lightning_airport_id"]).reset_index(drop=True)
    df["airport"] = df["airport"].astype("category")
    df["icloud"] = df["icloud"].astype("boolean")
    if "is_last_lightning_cloud_ground" in df.columns:
        df["is_last_lightning_cloud_ground"] = df["is_last_lightning_cloud_ground"].astype("boolean")
    
    df["in_alert_zone"] = df["airport_alert_id"].notna()
    df["is_cg"] = (~df["icloud"].fillna(False)).astype(int)
    df["is_lt20"] = (df["dist"] <= 20).astype(int)
    df["is_20_30"] = ((df["dist"] > 20) & (df["dist"] <= 30)).astype(int)
    df["is_lt10"] = (df["dist"] <= 10).astype(int)
    
    df["alert_key"] = np.where(
        df["in_alert_zone"],
        df["airport"].astype(str) + "__" + df["airport_alert_id"].astype("Int64").astype(str),
        pd.NA
    )
    
    if remove_pise_2016:
        mask = (df["airport"].astype(str) == "Pise") & (df["date"].dt.year == 2016)
        print(f"Suppression Pise 2016: {mask.sum()} lignes")
        df = df.loc[~mask].copy()
    
    return df


def build_silence_episodes(df):
    """
    Construit le dataset d'épisodes de silence.
    
    Pour chaque alerte, on prend chaque paire d'éclairs consécutifs.
    Le "silence" commence après l'éclair N et dure jusqu'à l'éclair N+1.
    Le dernier silence (après le dernier éclair) est le target=1.
    
    Features = contexte au moment de l'éclair qui PRÉCÈDE le silence.
    """
    alert_df = df[df["in_alert_zone"]].copy()
    alert_df = alert_df.sort_values(["alert_key", "date"]).reset_index(drop=True)
    
    # Enrichir chaque éclair avec son contexte
    alert_df["year"] = alert_df["date"].dt.year
    alert_df["month"] = alert_df["date"].dt.month
    alert_df["hour"] = alert_df["date"].dt.hour
    alert_df["dayofyear"] = alert_df["date"].dt.dayofyear
    alert_df["hour_sin"] = np.sin(2 * np.pi * alert_df["hour"] / 24)
    alert_df["hour_cos"] = np.cos(2 * np.pi * alert_df["hour"] / 24)
    alert_df["doy_sin"] = np.sin(2 * np.pi * alert_df["dayofyear"] / 365.25)
    alert_df["doy_cos"] = np.cos(2 * np.pi * alert_df["dayofyear"] / 365.25)
    
    def month_to_season(m):
        if m in [12, 1, 2]: return "winter"
        if m in [3, 4, 5]: return "spring"
        if m in [6, 7, 8]: return "summer"
        return "autumn"
    
    alert_df["season"] = alert_df["month"].map(month_to_season).astype("category")
    
    # Rang dans l'alerte
    alert_df["rank_in_alert"] = alert_df.groupby("alert_key").cumcount() + 1
    alert_df["alert_start_time"] = alert_df.groupby("alert_key")["date"].transform("min")
    alert_df["elapsed_sec"] = (alert_df["date"] - alert_df["alert_start_time"]).dt.total_seconds()
    
    # Gap avec l'éclair précédent dans l'alerte
    alert_df["gap_prev"] = alert_df.groupby("alert_key")["date"].diff().dt.total_seconds()
    
    # Gap avec l'éclair suivant (= durée du silence qui SUIT cet éclair)
    alert_df["gap_next"] = alert_df.groupby("alert_key")["date"].diff(-1).dt.total_seconds().abs()
    
    # Target : est-ce le dernier éclair de l'alerte ?
    if "is_last_lightning_cloud_ground" in alert_df.columns:
        alert_df["target"] = (alert_df["is_last_lightning_cloud_ground"] == True).astype(int)
    else:
        # Fallback : le dernier éclair de chaque alerte
        alert_df["_is_last"] = alert_df.groupby("alert_key").cumcount(ascending=False) == 0
        alert_df["target"] = alert_df["_is_last"].astype(int)
        alert_df.drop(columns=["_is_last"], inplace=True)
    
    # Stats cumulatives des gaps (expanding, pas de leakage)
    gap_grp = alert_df.groupby("alert_key")["gap_prev"]
    alert_df["gap_max_sofar"] = gap_grp.transform(lambda x: x.expanding().max())
    alert_df["gap_mean_sofar"] = gap_grp.transform(lambda x: x.expanding().mean())
    alert_df["gap_median_sofar"] = gap_grp.transform(lambda x: x.expanding().median())
    
    # Ratios de gap
    alert_df["gap_vs_max"] = alert_df["gap_prev"] / (alert_df["gap_max_sofar"] + 1.0)
    alert_df["gap_vs_mean"] = alert_df["gap_prev"] / (alert_df["gap_mean_sofar"] + 1.0)
    alert_df["gap_is_new_max"] = (alert_df["gap_prev"] >= alert_df["gap_max_sofar"]).astype(int)
    
    # Tendance des gaps
    alert_df["gap_last3_mean"] = gap_grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
    alert_df["gap_trend"] = alert_df["gap_last3_mean"] / (alert_df["gap_mean_sofar"] + 1.0)
    
    # Distance et amplitude du dernier éclair
    alert_df["abs_amplitude"] = alert_df["amplitude"].abs()
    
    # Tendance distance (orage qui part ?)
    dist_grp = alert_df.groupby("alert_key")["dist"]
    alert_df["dist_last3_mean"] = dist_grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
    alert_df["dist_mean_sofar"] = dist_grp.transform(lambda x: x.expanding().mean())
    alert_df["dist_trend"] = alert_df["dist_last3_mean"] - alert_df["dist_mean_sofar"]
    
    # Nombre d'éclairs récents (dans la zone aéroport, tous types)
    # On utilise le rolling temporel sur l'aéroport complet (pas juste l'alerte)
    # Pour ça, on doit merger avec le df complet
    
    # Pour l'instant, comptages intra-alerte par fenêtre de rang
    alert_df["cg_in_last3"] = alert_df.groupby("alert_key")["is_cg"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    alert_df["cg_in_last5"] = alert_df.groupby("alert_key")["is_cg"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    alert_df["lt20_in_last5"] = alert_df.groupby("alert_key")["is_lt20"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    
    # Elapsed per rank (rythme moyen)
    alert_df["elapsed_per_rank"] = alert_df["elapsed_sec"] / alert_df["rank_in_alert"].clip(lower=1)
    
    # ================================================================
    # MAINTENANT : construire les épisodes de silence à différents T
    # Pour chaque éclair, on simule "et si on attendait T minutes ?"
    # ================================================================
    
    # Pour le TRAIN : on connaît gap_next, donc on sait si le silence
    # de durée T est "terminal" (gap_next > T) ou pas
    
    # Temps de vérification (en secondes)
    check_times = [60, 120, 180, 300, 420, 600, 900, 1200, 1500, 1800]
    
    episodes = []
    for t_sec in check_times:
        t_min = t_sec / 60
        ep = alert_df.copy()
        ep["check_time_sec"] = t_sec
        ep["check_time_min"] = t_min
        
        # Pour les non-derniers éclairs : silence terminé si gap_next > t_sec
        # Mais attention : on ne garde que les épisodes où on a effectivement
        # attendu t_sec minutes (i.e., gap_next >= t_sec OU c'est le dernier)
        
        # Un épisode est "observable" à temps T si :
        # - C'est le dernier éclair (on attend T minutes sans rien → observable)
        # - Ou le prochain éclair arrive après T minutes (gap_next >= T)
        #   Dans ce cas, à T minutes on n'a toujours rien vu → observable, target=0
        #   (car un éclair arrive plus tard)
        # - Si gap_next < T, à T minutes on a déjà vu le prochain éclair
        #   → cet épisode n'existe pas à T, on le skip
        
        ep["is_observable"] = (ep["gap_next"].isna()) | (ep["gap_next"] >= t_sec)
        
        # Target pour cet épisode :
        # - 1 si c'est le dernier éclair (gap_next est NaN) ET c'est vraiment le target
        # - 0 si gap_next >= t_sec (un éclair arrive plus tard)
        ep["target_at_t"] = (ep["target"] == 1).astype(int)
        
        # Ne garder que les épisodes observables
        ep = ep[ep["is_observable"]].copy()
        
        episodes.append(ep)
    
    all_episodes = pd.concat(episodes, ignore_index=True)
    
    return all_episodes, alert_df


# ============================================================
# FEATURES POUR LE MODÈLE
# ============================================================
FEATURE_COLS = [
    # Contexte temporel
    "airport", "season",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    # Position dans l'alerte
    "rank_in_alert", "elapsed_sec", "elapsed_per_rank",
    # Dernier éclair : propriétés
    "dist", "abs_amplitude", "is_cg", "is_lt20", "is_20_30", "azimuth",
    # Gap précédent et stats
    "gap_prev", "gap_max_sofar", "gap_mean_sofar", "gap_median_sofar",
    "gap_vs_max", "gap_vs_mean", "gap_is_new_max", "gap_trend",
    "gap_last3_mean",
    # Distance trend
    "dist_last3_mean", "dist_mean_sofar", "dist_trend",
    # Comptages récents
    "cg_in_last3", "cg_in_last5", "lt20_in_last5",
    # LE SIGNAL CLÉ : combien de temps de silence on est en train d'observer
    "check_time_sec", "check_time_min",
    # Logs
    "log_gap_prev", "log_elapsed", "log_rank",
    "log_check_time", "log_gap_max_sofar",
]

CAT_COLS = {"airport", "season"}


def prepare_features(df):
    """Prépare les features et logs."""
    df = df.copy()
    df["log_gap_prev"] = np.log1p(df["gap_prev"].fillna(0).clip(lower=0))
    df["log_elapsed"] = np.log1p(df["elapsed_sec"].fillna(0).clip(lower=0))
    df["log_rank"] = np.log1p(df["rank_in_alert"])
    df["log_check_time"] = np.log1p(df["check_time_sec"])
    df["log_gap_max_sofar"] = np.log1p(df["gap_max_sofar"].fillna(0).clip(lower=0))
    
    # Imputation
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
        if str(df[col].dtype) in ["category", "object"]:
            df[col] = df[col].astype(str).fillna("MISSING")
        else:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    return df


# ============================================================
# MAIN
# ============================================================
print("=" * 80)
print("V6 — MODÈLE DE SURVIE")
print("P(orage fini | T minutes de silence, contexte)")
print("=" * 80)

# --- Charger données train ---
df_train = load_and_prepare(CSV_PATH, remove_pise_2016=USE_PISE_2016 == False)
print(f"Train brut: {df_train.shape}")

episodes_train, alert_train = build_silence_episodes(df_train)
episodes_train = prepare_features(episodes_train)
episodes_train["_year"] = episodes_train["date"].dt.year

print(f"Épisodes de silence: {len(episodes_train)}")
print(f"Target rate: {episodes_train['target_at_t'].mean():.4f}")
print(f"Distribution check_time_min:")
print(episodes_train.groupby("check_time_min")["target_at_t"].agg(["count", "mean"]).to_string())

# Split temporel
train_ep = episodes_train[episodes_train["_year"] <= 2020].copy()
valid_ep = episodes_train[episodes_train["_year"] == 2021].copy()

print(f"\nTrain (≤2020): {len(train_ep)} épisodes, target rate: {train_ep['target_at_t'].mean():.4f}")
print(f"Valid (2021):  {len(valid_ep)} épisodes, target rate: {valid_ep['target_at_t'].mean():.4f}")

# Features
cat_feature_indices = [i for i, c in enumerate(FEATURE_COLS) if c in CAT_COLS]

X_train = train_ep[FEATURE_COLS]
y_train = train_ep["target_at_t"]
X_valid = valid_ep[FEATURE_COLS]
y_valid = valid_ep["target_at_t"]

train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_feature_indices)

# ============================================================
# ENTRAÎNEMENT (hyperparamètres V4)
# ============================================================
print("\nEntraînement...")
model_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 3000,
    "learning_rate": 0.07496028016,
    "depth": 8,
    "l2_leaf_reg": 1.179393083,
    "subsample": 0.7094997377,
    "colsample_bylevel": 0.500217169,
    "min_data_in_leaf": 20,
    "scale_pos_weight": 4.158602233,
    "random_seed": RANDOM_SEED,
    "early_stopping_rounds": 200,
    "verbose": 100,
}

model = CatBoostClassifier(**model_params)
model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

# Métriques valid
valid_pred = model.predict_proba(valid_pool)[:, 1]
print(f"\nVALID (2021):")
print(f"  AUC: {roc_auc_score(y_valid, valid_pred):.4f}")
print(f"  AP:  {average_precision_score(y_valid, valid_pred):.4f}")

# Feature importance
fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": model.get_feature_importance(train_pool)})
fi = fi.sort_values("importance", ascending=False)
print(f"\nTop 15 features:")
print(fi.head(15).to_string(index=False))
fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

# ============================================================
# RETRAIN ≤2022
# ============================================================
print("\nRetrain ≤2022...")
full_train_ep = episodes_train[episodes_train["_year"] <= 2022].copy()
X_full = full_train_ep[FEATURE_COLS]
y_full = full_train_ep["target_at_t"]
full_pool = Pool(X_full, y_full, cat_features=cat_feature_indices)
es_pool = Pool(X_valid, y_valid, cat_features=cat_feature_indices)

model_final = CatBoostClassifier(**model_params)
model_final.fit(full_pool, eval_set=es_pool, use_best_model=True)

# ============================================================
# EVAL EXTERNE
# ============================================================
print("\n" + "=" * 80)
print("EVAL EXTERNE (2023-2025)")
print("=" * 80)

try:
    df_eval = load_and_prepare(EVAL_CSV_PATH)
    episodes_eval, alert_eval = build_silence_episodes(df_eval)
    episodes_eval = prepare_features(episodes_eval)
    
    print(f"Épisodes eval: {len(episodes_eval)}")
    
    X_eval = episodes_eval[FEATURE_COLS]
    eval_pool = Pool(X_eval, cat_features=cat_feature_indices)
    eval_pred = model_final.predict_proba(eval_pool)[:, 1]
    
    episodes_eval["pred"] = eval_pred
    
    if "target_at_t" in episodes_eval.columns:
        y_eval = episodes_eval["target_at_t"]
        print(f"\n  Global:")
        print(f"    AUC: {roc_auc_score(y_eval, eval_pred):.4f}")
        print(f"    AP:  {average_precision_score(y_eval, eval_pred):.4f}")
        
        # Par temps de vérification
        print(f"\n  Par temps de silence (minutes):")
        print(f"  {'T':>5s} {'n':>7s} {'target_rate':>12s} {'AUC':>8s} {'AP':>8s}")
        for t, grp in episodes_eval.groupby("check_time_min"):
            y_t = grp["target_at_t"]
            if y_t.nunique() < 2:
                continue
            auc_t = roc_auc_score(y_t, grp["pred"])
            ap_t = average_precision_score(y_t, grp["pred"])
            print(f"  {t:5.0f} {len(grp):7d} {y_t.mean():12.4f} {auc_t:8.4f} {ap_t:8.4f}")
    
    # ============================================================
    # GÉNÉRER LES COURBES DE SURVIE PAR ALERTE
    # ============================================================
    print(f"\n  Courbes de survie par alerte:")
    
    # Pour chaque alerte, on a les prédictions à chaque T
    # Prendre le dernier éclair de chaque alerte et ses prédictions à chaque T
    last_per_alert = alert_eval.groupby("alert_key").tail(1)[["alert_key", "lightning_id", "airport", "date"]]
    
    survival_curves = episodes_eval.merge(
        last_per_alert[["lightning_id"]], on="lightning_id"
    )
    
    if len(survival_curves) > 0:
        # Pour le dernier éclair : pred à chaque T = P(orage fini après T min de silence)
        example_alerts = survival_curves.groupby("alert_key").apply(
            lambda g: g[["check_time_min", "pred", "target_at_t"]].sort_values("check_time_min")
        ).reset_index(drop=True)
        
        print(f"    Exemple (moyenné sur toutes les alertes):")
        avg_curve = survival_curves.groupby("check_time_min")["pred"].mean()
        for t, p in avg_curve.items():
            print(f"      T={t:5.0f} min → P(fini) = {p:.3f}")
    
    # Exporter
    episodes_eval.to_csv(OUTPUT_DIR / "eval_predictions_survival.csv", index=False)
    
    # Aussi exporter au format "par éclair" pour comparaison
    # On prend la prédiction à T=30min (le seuil actuel Meteorage)
    eval_at_30 = episodes_eval[episodes_eval["check_time_min"] == 30].copy()
    eval_at_30_out = eval_at_30[["lightning_id", "airport", "date", "airport_alert_id", "pred", "target_at_t"]].copy()
    eval_at_30_out.to_csv(OUTPUT_DIR / "eval_predictions_at_30min.csv", index=False)
    
    if "target_at_t" in eval_at_30.columns and eval_at_30["target_at_t"].nunique() == 2:
        print(f"\n  À T=30min (comparable au seuil Meteorage actuel):")
        print(f"    AUC: {roc_auc_score(eval_at_30['target_at_t'], eval_at_30['pred']):.4f}")
        print(f"    AP:  {average_precision_score(eval_at_30['target_at_t'], eval_at_30['pred']):.4f}")

except FileNotFoundError:
    print(f"Fichier eval non trouvé: {EVAL_CSV_PATH}")

# Save
model.save_model(str(OUTPUT_DIR / "catboost_survival_v6.cbm"))
model_final.save_model(str(OUTPUT_DIR / "catboost_survival_v6_final.cbm"))

print("\n" + "=" * 80)
print(f"TERMINÉ. Outputs dans: {OUTPUT_DIR.resolve()}")
print("=" * 80)