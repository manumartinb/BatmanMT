#!/usr/bin/env python3
"""
BÚSQUEDA FINAL EXHAUSTIVA
Combina todas las técnicas para encontrar la mejor fórmula posible
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

TARGET = 'PnL_fwd_pts_05_mediana'
EXCLUDE = ['net_credit_diff']

print("="*80)
print("BÚSQUEDA FINAL EXHAUSTIVA")
print("="*80)

# Cargar datos
df = pd.read_csv('combined_BATMAN_mediana_w_stats_w_vix_labeled_NOSPXCHG.csv')
df['dia'] = pd.to_datetime(df['dia'])
df = df.sort_values('dia').reset_index(drop=True)

# Filtrar columnas
all_cols = df.columns.tolist()
cols_fwd = [c for c in all_cols if 'fwd' in c.lower() and c != TARGET]
cols_chg = [c for c in all_cols if 'chg' in c.lower()]
cols_label = [c for c in all_cols if 'label' in c.lower()]
cols_score = [c for c in all_cols if 'score' in c.lower() and any(x in c for x in ['7','21','63','252'])]
cols_non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

all_excluded = set(cols_fwd + cols_chg + cols_label + cols_score +
                   cols_non_numeric + EXCLUDE + ['RANK_PRE', 'WL_PRE'])

df_valid = df.dropna(subset=[TARGET]).copy()
y = df_valid[TARGET].values
N = len(y)

tscv = TimeSeriesSplit(n_splits=5)

# Cargar variables clave
SPX_Stoch = df_valid['SPX_Stoch_K'].values.astype(float)
SPX_Williams = df_valid['SPX_Williams_R'].values.astype(float)
SPX_BB = df_valid['SPX_BB_Pct'].values.astype(float)
SPX_ZScore20 = df_valid['SPX_ZScore20'].values.astype(float)
SPX_Stoch_D = df_valid['SPX_Stoch_D'].values.astype(float)
SPX_RSI = df_valid['SPX_RSI14'].values.astype(float)
SPX_ROC7 = df_valid['SPX_ROC7'].values.astype(float)
SPX_ROC20 = df_valid['SPX_ROC20'].values.astype(float)
SPX_ATR = df_valid['SPX_ATR14'].values.astype(float)
SPX_SMA7 = df_valid['SPX_minus_SMA7'].values.astype(float)
SPX_SMA20 = df_valid['SPX_minus_SMA20'].values.astype(float)
SPX_SMA50 = df_valid['SPX_minus_SMA50'].values.astype(float)
SPX_ZScore50 = df_valid['SPX_ZScore50'].values.astype(float)
SPX_MACD = df_valid['SPX_MACD_Histogram'].values.astype(float)
SPX_HV20 = df_valid['SPX_HV20'].values.astype(float)
VIX = df_valid['VIX_Close'].values.astype(float)
theta_total = df_valid['theta_total'].values.astype(float)
delta_total = df_valid['delta_total'].values.astype(float)
net_credit = df_valid['net_credit'].values.astype(float)
net_credit_med = df_valid['net_credit_mediana'].values.astype(float)
PnLDV = df_valid['PnLDV'].values.astype(float)

def eval_oos(values, y):
    """Evalúa correlación OOS promedio"""
    values = np.nan_to_num(values, nan=0)
    rhos = []
    for train_idx, test_idx in tscv.split(values):
        rho, _ = spearmanr(values[test_idx], y[test_idx])
        if not np.isnan(rho):
            rhos.append(rho)
    return np.mean(rhos) if rhos else 0

def eval_is(values, y):
    """Evalúa correlación IS"""
    values = np.nan_to_num(values, nan=0)
    rho, _ = spearmanr(values, y)
    return rho if not np.isnan(rho) else 0

# =============================================================================
# GENERACIÓN MASIVA DE FÓRMULAS
# =============================================================================

print("\n📊 Generando miles de fórmulas...")

formulas = {}

# Normalización de indicadores base
stoch_norm = SPX_Stoch / 100
williams_norm = (100 + SPX_Williams) / 100
bb_norm = SPX_BB
rsi_norm = SPX_RSI / 100
roc7_norm = np.tanh(SPX_ROC7 / 5)
zscore20_norm = np.tanh(SPX_ZScore20)
zscore50_norm = np.tanh(SPX_ZScore50)
atr_inv = 1 / (SPX_ATR + 1)
vix_inv = 1 / (VIX + 10)
sma7_norm = np.tanh(SPX_SMA7 / 50)
sma20_norm = np.tanh(SPX_SMA20 / 100)

# Pesos a probar
weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 1. Combinaciones de 2 indicadores con diferentes pesos
print("  → Combinaciones de 2 indicadores...")
indicators = {
    'stoch': stoch_norm,
    'williams': williams_norm,
    'bb': bb_norm,
    'rsi': rsi_norm,
    'roc7': roc7_norm,
    'zscore20': zscore20_norm,
    'zscore50': zscore50_norm,
    'sma7': sma7_norm,
    'sma20': sma20_norm
}

for (n1, v1), (n2, v2) in combinations(indicators.items(), 2):
    for w in [0.3, 0.5, 0.7]:
        name = f"{w:.1f}*{n1}+{1-w:.1f}*{n2}"
        formulas[name] = w * v1 + (1-w) * v2

# 2. Productos y ratios de 2 indicadores
print("  → Productos y ratios...")
for (n1, v1), (n2, v2) in combinations(indicators.items(), 2):
    formulas[f"{n1}*{n2}"] = v1 * v2
    formulas[f"{n1}/{n2}"] = v1 / (np.abs(v2) + 0.1)

# 3. Combinaciones de 3-4 indicadores con diferentes pesos
print("  → Combinaciones de 3-4 indicadores...")
for combo in combinations(list(indicators.keys())[:6], 3):
    for w1, w2 in [(0.5, 0.3), (0.4, 0.4), (0.6, 0.2)]:
        w3 = 1 - w1 - w2
        name = f"combo3_{combo[0][:3]}_{combo[1][:3]}_{combo[2][:3]}_{w1}"
        vals = [indicators[combo[0]], indicators[combo[1]], indicators[combo[2]]]
        formulas[name] = w1 * vals[0] + w2 * vals[1] + w3 * vals[2]

# 4. Fórmulas con VIX como divisor
print("  → Fórmulas con VIX...")
for n, v in indicators.items():
    formulas[f"{n}/vix"] = v * vix_inv
    formulas[f"{n}*vix_inv"] = v / (VIX + 5)

# 5. Fórmulas con ATR como divisor
print("  → Fórmulas con ATR...")
for n, v in indicators.items():
    formulas[f"{n}/atr"] = v * atr_inv
    formulas[f"{n}*atr_inv"] = v / (SPX_ATR + 1)

# 6. Fórmulas cuadráticas
print("  → Fórmulas cuadráticas...")
for n, v in indicators.items():
    formulas[f"{n}^2"] = v ** 2
    formulas[f"sqrt_{n}"] = np.sqrt(np.abs(v))
    formulas[f"sign_{n}*{n}^2"] = np.sign(v) * (v ** 2)

# 7. Fórmulas con productos de 3
print("  → Productos triples...")
for combo in combinations(list(indicators.keys())[:5], 3):
    vals = [indicators[c] for c in combo]
    formulas[f"prod_{combo[0][:2]}_{combo[1][:2]}_{combo[2][:2]}"] = vals[0] * vals[1] * vals[2]

# 8. Fórmulas con condicionales
print("  → Fórmulas condicionales...")
for n, v in indicators.items():
    formulas[f"{n}_pos"] = np.where(v > 0.5, v, 0)
    formulas[f"{n}_neg"] = np.where(v < 0.5, v, 0)
    formulas[f"{n}_extreme"] = np.where((v > 0.8) | (v < 0.2), v, 0.5)

# 9. Combinaciones con momentum y volatilidad
print("  → Momentum/Volatilidad combos...")
momentum_indicators = ['stoch', 'williams', 'rsi', 'roc7']
vol_indicators = ['atr_inv', 'vix_inv', 'zscore20', 'zscore50']

for m in momentum_indicators:
    for v in vol_indicators:
        m_val = indicators.get(m, stoch_norm)
        if v == 'atr_inv':
            v_val = atr_inv
        elif v == 'vix_inv':
            v_val = vix_inv
        else:
            v_val = indicators.get(v, zscore20_norm)
        formulas[f"{m}_x_{v}"] = m_val * v_val
        formulas[f"{m}_div_{v}"] = m_val / (np.abs(v_val) + 0.1)

# 10. Fórmulas compuestas especiales
print("  → Fórmulas compuestas especiales...")

# Composite base (la mejor encontrada antes)
formulas['composite_4'] = (stoch_norm + williams_norm/2 + bb_norm + zscore20_norm) / 4

# Variantes del composite
for w in [0.3, 0.35, 0.4, 0.45, 0.5]:
    formulas[f'composite_stoch_{w}'] = w*stoch_norm + (1-w)*(williams_norm/2 + bb_norm + zscore20_norm)/3

# Momentum score variantes
for w1, w2 in [(0.4, 0.3), (0.5, 0.25), (0.6, 0.2)]:
    w3 = 1 - w1 - w2
    formulas[f'momentum_{w1}_{w2}'] = w1*stoch_norm + w2*williams_norm + w3*bb_norm

# Zscore combos
formulas['zscore_combo_1'] = zscore20_norm + zscore50_norm
formulas['zscore_combo_2'] = 0.7*zscore20_norm + 0.3*zscore50_norm
formulas['zscore_roc'] = zscore20_norm + roc7_norm
formulas['zscore_rsi'] = zscore20_norm * rsi_norm

# RSI-based
formulas['rsi_stoch_prod'] = rsi_norm * stoch_norm
formulas['rsi_williams_sum'] = rsi_norm + williams_norm
formulas['rsi_bb_prod'] = rsi_norm * bb_norm

# VIX-adjusted momentum
formulas['vix_adj_stoch'] = stoch_norm * (30 / (VIX + 10))
formulas['vix_adj_rsi'] = rsi_norm * (30 / (VIX + 10))
formulas['vix_adj_bb'] = bb_norm * (30 / (VIX + 10))

# ATR-adjusted
formulas['atr_adj_stoch'] = stoch_norm / (SPX_ATR / 20 + 0.5)
formulas['atr_adj_zscore'] = zscore20_norm / (SPX_ATR / 20 + 0.5)

# Geometric means
formulas['geom_stoch_bb'] = np.sign(stoch_norm - 0.5) * np.sqrt(np.abs(stoch_norm * bb_norm) + 0.01)
formulas['geom_rsi_williams'] = np.sign(williams_norm - 0.5) * np.sqrt(np.abs(rsi_norm * williams_norm) + 0.01)
formulas['geom_triple'] = np.cbrt(stoch_norm * williams_norm * bb_norm)

# Harmonic means
formulas['harm_stoch_bb'] = 2 * stoch_norm * bb_norm / (stoch_norm + bb_norm + 0.01)
formulas['harm_rsi_stoch'] = 2 * rsi_norm * stoch_norm / (rsi_norm + stoch_norm + 0.01)

# Ranked combinations
stoch_rank = pd.Series(SPX_Stoch).rank(pct=True).values
bb_rank = pd.Series(SPX_BB).rank(pct=True).values
rsi_rank = pd.Series(SPX_RSI).rank(pct=True).values
williams_rank = pd.Series(SPX_Williams + 100).rank(pct=True).values

formulas['rank_sum_4'] = (stoch_rank + bb_rank + rsi_rank + williams_rank) / 4
formulas['rank_prod_2'] = stoch_rank * bb_rank
formulas['rank_weighted'] = 0.4*stoch_rank + 0.3*bb_rank + 0.3*williams_rank

# Polynomial features
formulas['poly_stoch_bb'] = stoch_norm + bb_norm + stoch_norm*bb_norm
formulas['poly_triple'] = stoch_norm + williams_norm + bb_norm + stoch_norm*williams_norm + williams_norm*bb_norm
formulas['poly_quad'] = stoch_norm**2 + bb_norm**2 + stoch_norm*bb_norm

# Difference-based
formulas['stoch_minus_rsi'] = stoch_norm - rsi_norm
formulas['bb_minus_williams'] = bb_norm - williams_norm
formulas['zscore_diff'] = zscore20_norm - zscore50_norm

# Ratio-based composite
formulas['ratio_composite_1'] = (stoch_norm / (rsi_norm + 0.1)) * bb_norm
formulas['ratio_composite_2'] = stoch_norm * williams_norm / (zscore20_norm + 1)

# Extreme indicator combinations
formulas['extreme_momentum'] = np.where(stoch_norm > 0.8, 2*bb_norm, np.where(stoch_norm < 0.2, -bb_norm, bb_norm))
formulas['extreme_composite'] = np.where(SPX_Stoch > 90, 1.5*(stoch_norm + bb_norm),
                                         np.where(SPX_Stoch < 10, 0.5*(stoch_norm + bb_norm),
                                                  stoch_norm + bb_norm))

print(f"\n📊 Total fórmulas generadas: {len(formulas)}")

# =============================================================================
# EVALUACIÓN MASIVA
# =============================================================================

print("\n🔍 Evaluando todas las fórmulas...")

results = []
for name, values in formulas.items():
    rho_oos = eval_oos(values, y)
    rho_is = eval_is(values, y)
    results.append({
        'name': name,
        'rho_oos': rho_oos,
        'rho_is': rho_is,
        'abs_rho_oos': abs(rho_oos)
    })

results_df = pd.DataFrame(results).sort_values('abs_rho_oos', ascending=False)

# =============================================================================
# TOP RESULTADOS
# =============================================================================

print("\n" + "="*80)
print("TOP 30 FÓRMULAS POR ρ_OOS")
print("="*80)

for i, row in results_df.head(30).iterrows():
    print(f"  {row['name']:<50} | IS: {row['rho_is']:+.4f} | OOS: {row['rho_oos']:+.4f}")

# =============================================================================
# ANÁLISIS DE LA MEJOR FÓRMULA
# =============================================================================

best = results_df.iloc[0]
print("\n" + "="*80)
print(f"MEJOR FÓRMULA: {best['name']}")
print("="*80)
print(f"  ρ_IS:  {best['rho_is']:+.4f}")
print(f"  ρ_OOS: {best['rho_oos']:+.4f}")

# Análisis por fold
best_values = formulas[best['name']]
best_values = np.nan_to_num(best_values, nan=0)

print("\n  Correlación por fold temporal:")
for i, (train_idx, test_idx) in enumerate(tscv.split(best_values)):
    rho, _ = spearmanr(best_values[test_idx], y[test_idx])
    print(f"    Fold {i+1}: ρ = {rho:+.4f}")

# Análisis por deciles
print("\n  Análisis por deciles de la fórmula:")
deciles = pd.qcut(best_values, 10, labels=False, duplicates='drop')
print(f"  {'Decil':<8} | {'N':>5} | {'Mediana Target':>15} | {'Mean Target':>12}")
print("  " + "-"*50)
for d in range(deciles.max() + 1):
    mask = deciles == d
    y_d = y[mask]
    print(f"  D{d:<7} | {mask.sum():>5} | {np.median(y_d):>15.3f} | {np.mean(y_d):>12.3f}")

lift = np.median(y[deciles == deciles.max()]) - np.median(y[deciles == 0])
print(f"\n  📈 LIFT (Top vs Bottom decile): {lift:+.3f}")

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================

results_df.to_csv('final_formulas_ranking.csv', index=False)
print(f"\n✅ Ranking completo guardado en: final_formulas_ranking.csv")

# Guardar top 10 con sus valores
top10 = results_df.head(10)['name'].tolist()
top10_data = pd.DataFrame({name: formulas[name] for name in top10})
top10_data[TARGET] = y
top10_data.to_csv('top10_formulas_values.csv', index=False)
print(f"✅ Valores de top 10 fórmulas guardados en: top10_formulas_values.csv")

print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"\n🏆 MEJOR ρ_OOS ENCONTRADO: {best['rho_oos']:+.4f}")
print(f"   Fórmula: {best['name']}")
