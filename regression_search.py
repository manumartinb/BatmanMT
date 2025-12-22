#!/usr/bin/env python3
"""
Búsqueda de combinaciones lineales óptimas con regresión regularizada
+ Búsqueda de polinomios
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

TARGET = 'PnL_fwd_pts_05_mediana'
EXCLUDE = ['net_credit_diff']

print("="*80)
print("BÚSQUEDA CON REGRESIÓN REGULARIZADA Y POLINOMIOS")
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

features_base = [c for c in all_cols if c not in all_excluded and c != TARGET]

df_valid = df.dropna(subset=[TARGET]).copy()
y = df_valid[TARGET].values

# Seleccionar features numéricas con pocos NaNs
good_features = []
for f in features_base:
    x = df_valid[f].values
    nan_ratio = np.isnan(x.astype(float)).sum() / len(x)
    if nan_ratio < 0.1:
        good_features.append(f)

print(f"Features disponibles: {len(good_features)}")

# Crear matriz X
X = df_valid[good_features].values.astype(float)
X = np.nan_to_num(X, nan=0)

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tscv = TimeSeriesSplit(n_splits=5)

def cv_predict_ts(model, X, y, cv):
    """Predicción OOS con TimeSeriesSplit"""
    preds = []
    actuals = []
    for train_idx, test_idx in cv.split(X):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        preds.extend(pred)
        actuals.extend(y[test_idx])
    return np.array(preds), np.array(actuals)

# =============================================================================
# 1. REGRESIÓN RIDGE CON DIFERENTES ALPHAS
# =============================================================================

print("\n" + "="*60)
print("1. REGRESIÓN RIDGE - Buscando mejor combinación lineal")
print("="*60)

best_ridge = {'alpha': None, 'rho_oos': 0}
alphas = [0.01, 0.1, 1, 10, 100, 1000]

for alpha in alphas:
    model = Ridge(alpha=alpha)
    y_pred, y_actual = cv_predict_ts(model, X_scaled, y, tscv)
    rho, _ = spearmanr(y_pred, y_actual)

    print(f"  α={alpha:>6}: ρ = {rho:+.4f}")

    if abs(rho) > abs(best_ridge['rho_oos']):
        best_ridge = {'alpha': alpha, 'rho_oos': rho}

print(f"\n  Mejor Ridge: α={best_ridge['alpha']}, ρ_OOS={best_ridge['rho_oos']:+.4f}")

# Entrenar modelo final y ver coeficientes
model = Ridge(alpha=best_ridge['alpha'])
model.fit(X_scaled, y)

# Top features por coeficiente
coefs = pd.DataFrame({
    'feature': good_features,
    'coef': model.coef_,
    'abs_coef': np.abs(model.coef_)
}).sort_values('abs_coef', ascending=False)

print("\n  Top 15 features por coeficiente Ridge:")
for _, row in coefs.head(15).iterrows():
    print(f"    {row['feature'][:40]:<40} | coef = {row['coef']:+.4f}")

# =============================================================================
# 2. LASSO PARA SELECCIÓN SPARSE
# =============================================================================

print("\n" + "="*60)
print("2. LASSO - Selección sparse de features")
print("="*60)

best_lasso = {'alpha': None, 'rho_oos': 0, 'n_features': 0}
alphas_lasso = [0.001, 0.01, 0.1, 0.5, 1]

for alpha in alphas_lasso:
    model = Lasso(alpha=alpha, max_iter=5000)
    y_pred, y_actual = cv_predict_ts(model, X_scaled, y, tscv)
    rho, _ = spearmanr(y_pred, y_actual)

    # Contar features seleccionadas
    model.fit(X_scaled, y)
    n_selected = np.sum(model.coef_ != 0)

    print(f"  α={alpha:>5}: ρ = {rho:+.4f} | n_features = {n_selected}")

    if abs(rho) > abs(best_lasso['rho_oos']):
        best_lasso = {'alpha': alpha, 'rho_oos': rho, 'n_features': n_selected}

print(f"\n  Mejor Lasso: α={best_lasso['alpha']}, ρ_OOS={best_lasso['rho_oos']:+.4f}")

# Features seleccionadas
model = Lasso(alpha=best_lasso['alpha'], max_iter=5000)
model.fit(X_scaled, y)

selected = [(good_features[i], model.coef_[i]) for i in range(len(good_features)) if model.coef_[i] != 0]
selected = sorted(selected, key=lambda x: abs(x[1]), reverse=True)

print(f"\n  Features seleccionadas por Lasso ({len(selected)}):")
for name, coef in selected[:20]:
    print(f"    {name[:40]:<40} | coef = {coef:+.4f}")

# =============================================================================
# 3. POLINOMIOS GRADO 2 CON TOP FEATURES
# =============================================================================

print("\n" + "="*60)
print("3. POLINOMIOS GRADO 2 con Top Features")
print("="*60)

# Usar top 10 features por correlación individual
top_features = []
for f in good_features:
    x = df_valid[f].values.astype(float)
    mask = ~np.isnan(x)
    if mask.sum() > 50:
        rho, _ = spearmanr(x[mask], y[mask])
        if not np.isnan(rho):
            top_features.append((f, abs(rho)))

top_features = sorted(top_features, key=lambda x: x[1], reverse=True)[:10]
top_names = [f[0] for f in top_features]

print(f"  Top 10 features base: {[f[:15] for f in top_names]}")

X_top = df_valid[top_names].values.astype(float)
X_top = np.nan_to_num(X_top, nan=0)

# Crear polinomios grado 2
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_top)
feature_names_poly = poly.get_feature_names_out(top_names)

print(f"  Features polinomiales generadas: {X_poly.shape[1]}")

# Escalar
scaler_poly = StandardScaler()
X_poly_scaled = scaler_poly.fit_transform(X_poly)

# Ridge sobre polinomios
best_poly_ridge = {'alpha': None, 'rho_oos': 0}
for alpha in [0.1, 1, 10, 100, 1000]:
    model = Ridge(alpha=alpha)
    y_pred, y_actual = cv_predict_ts(model, X_poly_scaled, y, tscv)
    rho, _ = spearmanr(y_pred, y_actual)

    if abs(rho) > abs(best_poly_ridge['rho_oos']):
        best_poly_ridge = {'alpha': alpha, 'rho_oos': rho}

print(f"\n  Mejor Ridge Polinomial: α={best_poly_ridge['alpha']}, ρ_OOS={best_poly_ridge['rho_oos']:+.4f}")

# Top términos polinomiales
model = Ridge(alpha=best_poly_ridge['alpha'])
model.fit(X_poly_scaled, y)

poly_coefs = pd.DataFrame({
    'term': feature_names_poly,
    'coef': model.coef_,
    'abs_coef': np.abs(model.coef_)
}).sort_values('abs_coef', ascending=False)

print("\n  Top 15 términos polinomiales:")
for _, row in poly_coefs.head(15).iterrows():
    print(f"    {row['term'][:45]:<45} | coef = {row['coef']:+.4f}")

# =============================================================================
# 4. SUBCONJUNTOS ÓPTIMOS DE 3-5 FEATURES
# =============================================================================

print("\n" + "="*60)
print("4. BÚSQUEDA DE SUBCONJUNTOS ÓPTIMOS (3-5 features)")
print("="*60)

top20_names = [f[0] for f in sorted(top_features, key=lambda x: x[1], reverse=True)]

best_subset = {'features': None, 'rho_oos': 0, 'formula': '', 'n': 0}

# Subconjuntos de 3
print("\n  Probando subconjuntos de 3 features...")
count = 0
for combo in combinations(top20_names[:10], 3):
    X_sub = df_valid[list(combo)].values.astype(float)
    X_sub = np.nan_to_num(X_sub, nan=0)
    X_sub_scaled = StandardScaler().fit_transform(X_sub)

    model = Ridge(alpha=1)
    y_pred, y_actual = cv_predict_ts(model, X_sub_scaled, y, tscv)
    rho, _ = spearmanr(y_pred, y_actual)

    if abs(rho) > abs(best_subset['rho_oos']):
        model.fit(X_sub_scaled, y)
        coefs = model.coef_
        formula = " + ".join([f"{coefs[i]:+.3f}*{combo[i][:12]}" for i in range(3)])
        best_subset = {'features': combo, 'rho_oos': rho, 'formula': formula, 'n': 3}

    count += 1

print(f"    Probados {count} subconjuntos de 3")
print(f"    Mejor: ρ_OOS = {best_subset['rho_oos']:+.4f}")

# Subconjuntos de 4
print("\n  Probando subconjuntos de 4 features...")
count = 0
for combo in combinations(top20_names[:10], 4):
    X_sub = df_valid[list(combo)].values.astype(float)
    X_sub = np.nan_to_num(X_sub, nan=0)
    X_sub_scaled = StandardScaler().fit_transform(X_sub)

    model = Ridge(alpha=1)
    y_pred, y_actual = cv_predict_ts(model, X_sub_scaled, y, tscv)
    rho, _ = spearmanr(y_pred, y_actual)

    if abs(rho) > abs(best_subset['rho_oos']):
        model.fit(X_sub_scaled, y)
        coefs = model.coef_
        formula = " + ".join([f"{coefs[i]:+.3f}*{combo[i][:10]}" for i in range(4)])
        best_subset = {'features': combo, 'rho_oos': rho, 'formula': formula, 'n': 4}

    count += 1

print(f"    Probados {count} subconjuntos de 4")
print(f"\n  Mejor subconjunto de {best_subset['n']}: ρ_OOS = {best_subset['rho_oos']:+.4f}")
print(f"  Features: {best_subset['features']}")
print(f"  Fórmula: {best_subset['formula']}")

# =============================================================================
# 5. ELASTIC NET
# =============================================================================

print("\n" + "="*60)
print("5. ELASTIC NET")
print("="*60)

best_enet = {'alpha': None, 'l1_ratio': None, 'rho_oos': 0}

for alpha in [0.01, 0.1, 0.5]:
    for l1_ratio in [0.2, 0.5, 0.8]:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        y_pred, y_actual = cv_predict_ts(model, X_scaled, y, tscv)
        rho, _ = spearmanr(y_pred, y_actual)

        if abs(rho) > abs(best_enet['rho_oos']):
            best_enet = {'alpha': alpha, 'l1_ratio': l1_ratio, 'rho_oos': rho}

print(f"  Mejor ElasticNet: α={best_enet['alpha']}, l1={best_enet['l1_ratio']}, ρ_OOS={best_enet['rho_oos']:+.4f}")

# =============================================================================
# 6. FÓRMULAS ESPECÍFICAS
# =============================================================================

print("\n" + "="*60)
print("6. FÓRMULAS ESPECÍFICAS BASADAS EN ANÁLISIS")
print("="*60)

# Variables clave
SPX_Stoch = df_valid['SPX_Stoch_K'].values.astype(float)
SPX_Williams = df_valid['SPX_Williams_R'].values.astype(float)
SPX_BB = df_valid['SPX_BB_Pct'].values.astype(float)
SPX_ZScore20 = df_valid['SPX_ZScore20'].values.astype(float)
SPX_Stoch_D = df_valid['SPX_Stoch_D'].values.astype(float)
SPX_RSI = df_valid['SPX_RSI14'].values.astype(float)
SPX_ROC7 = df_valid['SPX_ROC7'].values.astype(float)
SPX_ATR = df_valid['SPX_ATR14'].values.astype(float)
SPX_SMA7 = df_valid['SPX_minus_SMA7'].values.astype(float)
SPX_SMA20 = df_valid['SPX_minus_SMA20'].values.astype(float)
VIX = df_valid['VIX_Close'].values.astype(float)

formulas = []

# Fórmula 1: Score de momentum simple
f1 = SPX_Stoch / 100 + (100 + SPX_Williams) / 100 + SPX_BB
formulas.append(('momentum_score', f1))

# Fórmula 2: Ratio momentum/volatilidad
f2 = (SPX_Stoch + SPX_Williams + 100) / (SPX_ATR + 1)
formulas.append(('momentum_vol_ratio', f2))

# Fórmula 3: Zscore compuesto
f3 = SPX_ZScore20 + np.tanh(SPX_ROC7 / 5)
formulas.append(('zscore_composite', f3))

# Fórmula 4: VIX adjustado
f4 = SPX_Stoch / (VIX + 10)
formulas.append(('stoch_vix_ratio', f4))

# Fórmula 5: Mean reversion score
f5 = SPX_BB * np.sign(SPX_Williams + 50)
formulas.append(('mean_reversion', f5))

# Fórmula 6: Combinación geométrica
f6 = np.sign(SPX_Stoch - 50) * np.sqrt(np.abs(SPX_Stoch - 50) * np.abs(SPX_Williams + 50) + 1)
formulas.append(('geom_momentum', f6))

# Fórmula 7: Score compuesto multi-indicador
f7 = (SPX_Stoch / 100 + (100 + SPX_Williams) / 200 + SPX_BB + np.tanh(SPX_ZScore20)) / 4
formulas.append(('composite_4', f7))

# Fórmula 8: Ratio con VIX
f8 = (SPX_Stoch * SPX_BB) / (VIX + 10)
formulas.append(('stoch_bb_vix', f8))

# Fórmula 9: Polynomio cuadrático
f9 = SPX_Stoch**2 / 10000 + SPX_Williams / 100 + SPX_BB
formulas.append(('poly_momentum', f9))

# Fórmula 10: Score de RSI+Stoch normalizado
f10 = (SPX_RSI / 100 + SPX_Stoch / 100) / 2
formulas.append(('rsi_stoch_avg', f10))

# Fórmula 11: SMA momentum composite
f11 = SPX_SMA7 / (np.abs(SPX_SMA20) + 1) * np.sign(SPX_Stoch - 50)
formulas.append(('sma_momentum', f11))

# Fórmula 12: Ratio técnico compuesto
f12 = (SPX_Stoch * SPX_RSI) / (100 * (VIX + 5))
formulas.append(('tech_ratio', f12))

# Fórmula 13: Momentum extremo
f13 = np.where(SPX_Stoch > 80, SPX_BB * 2, np.where(SPX_Stoch < 20, -SPX_BB, SPX_BB))
formulas.append(('momentum_extreme', f13))

# Fórmula 14: Triple momentum
f14 = SPX_Stoch/100 * (1 + SPX_BB) * (1 + np.tanh(SPX_ZScore20))
formulas.append(('triple_momentum', f14))

# Fórmula 15: Anti-VIX momentum
f15 = (SPX_Stoch - 50) / (VIX + 1) + SPX_BB
formulas.append(('anti_vix_momentum', f15))

print("  Evaluando fórmulas específicas:")
best_formula = {'name': '', 'rho_oos': 0}

for name, values in formulas:
    values = np.nan_to_num(values, nan=0)

    # OOS evaluation
    rhos = []
    for train_idx, test_idx in tscv.split(values):
        rho, _ = spearmanr(values[test_idx], y[test_idx])
        if not np.isnan(rho):
            rhos.append(rho)

    rho_oos = np.mean(rhos) if rhos else 0
    rho_is, _ = spearmanr(values, y)

    print(f"    {name:<25} | IS: {rho_is:+.4f} | OOS: {rho_oos:+.4f}")

    if abs(rho_oos) > abs(best_formula['rho_oos']):
        best_formula = {'name': name, 'rho_oos': rho_oos, 'rho_is': rho_is}

print(f"\n  Mejor fórmula: {best_formula['name']} (ρ_OOS = {best_formula['rho_oos']:+.4f})")

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

results_summary = [
    ('Ridge (lineal)', best_ridge['rho_oos']),
    ('Lasso (sparse)', best_lasso['rho_oos']),
    ('Ridge Polinomial', best_poly_ridge['rho_oos']),
    ('Mejor subconjunto', best_subset['rho_oos']),
    ('ElasticNet', best_enet['rho_oos']),
    (f'Fórmula: {best_formula["name"]}', best_formula['rho_oos'])
]

print("\nMETODO                          | ρ_OOS")
print("-"*50)
for name, rho in sorted(results_summary, key=lambda x: abs(x[1]), reverse=True):
    print(f"{name:<30} | {rho:+.4f}")

# Guardar mejores resultados
pd.DataFrame({
    'method': [r[0] for r in results_summary],
    'rho_oos': [r[1] for r in results_summary]
}).to_csv('regression_results.csv', index=False)

print("\n✅ Resultados guardados en regression_results.csv")
