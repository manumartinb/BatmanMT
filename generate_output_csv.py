#!/usr/bin/env python3
"""
Genera el CSV de salida con target + features T+0 permitidas + derivadas top
"""

import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('combined_BATMAN_mediana_w_stats_w_vix_labeled_NOSPXCHG.csv')

TARGET = 'PnL_fwd_pts_05_mediana'

# Features excluidas
all_cols = df.columns.tolist()
cols_fwd = [c for c in all_cols if 'fwd' in c.lower() and c != TARGET]
cols_chg = [c for c in all_cols if 'chg' in c.lower()]

cols_suspicious = []
for c in all_cols:
    c_lower = c.lower()
    if 'label' in c_lower and any(x in c_lower for x in ['7', '21', '63', '252']):
        cols_suspicious.append(c)
    if 'score' in c_lower and any(x in c_lower for x in ['7', '21', '63', '252']):
        cols_suspicious.append(c)
    if c in ['WL_PRE', 'RANK_PRE']:
        cols_suspicious.append(c)

cols_non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

all_excluded = set(cols_fwd + cols_chg + cols_suspicious + cols_non_numeric)
features_t0 = [c for c in all_cols if c not in all_excluded and c != TARGET]

# Crear dataset de salida
df_out = df[[TARGET] + features_t0].copy()

# Añadir top 10 features derivadas
top10_base = ['net_credit_diff', 'SPX_Stoch_K', 'SPX_Williams_R', 'SPX_BB_Pct',
              'SPX_ZScore20', 'SPX_Stoch_D', 'SPX_ROC7', 'SPX_ZScore50',
              'SPX_RSI14', 'SPX_minus_SMA7']

# Crear derivadas top
for feat in top10_base[:5]:
    if feat in df.columns:
        x = df[feat]
        # Rank
        df_out[f"{feat}_rank"] = x.rank(pct=True)
        # Z-score robusto
        median = x.median()
        mad = (x - median).abs().median()
        if mad > 0:
            df_out[f"{feat}_zscore_robust"] = (x - median) / mad

# Interacciones top
if 'net_credit_diff' in df.columns and 'SPX_Stoch_K' in df.columns:
    df_out['net_credit_diff_mult_SPX_Stoch_K'] = df['net_credit_diff'] * df['SPX_Stoch_K']

if 'net_credit_diff' in df.columns and 'SPX_BB_Pct' in df.columns:
    df_out['net_credit_diff_mult_SPX_BB_Pct'] = df['net_credit_diff'] * df['SPX_BB_Pct']

# Guardar
df_out.to_csv('dataset_T0_features_top.csv', index=False)
print(f"✅ Dataset guardado: dataset_T0_features_top.csv")
print(f"   Dimensiones: {df_out.shape[0]} filas × {df_out.shape[1]} columnas")
print(f"   Columnas: {list(df_out.columns[:10])}... (y {len(df_out.columns)-10} más)")
