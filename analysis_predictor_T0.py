#!/usr/bin/env python3
"""
Análisis Predictivo T+0 → PnL_fwd_pts_05_mediana
Objetivo: Descubrir variables predictivas en T+0 para el target futuro
SIN leakage / look-ahead
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# (A) CARGA Y VALIDACIONES INICIALES
# =============================================================================

print("="*80)
print("ANÁLISIS PREDICTIVO T+0 → PnL_fwd_pts_05_mediana")
print("="*80)

# Cargar datos
df = pd.read_csv('combined_BATMAN_mediana_w_stats_w_vix_labeled_NOSPXCHG.csv')

print(f"\n📊 DIMENSIONES: {df.shape[0]} filas × {df.shape[1]} columnas")

# Target
TARGET = 'PnL_fwd_pts_05_mediana'
print(f"\n🎯 TARGET: {TARGET}")

# Verificar que el target existe
if TARGET not in df.columns:
    raise ValueError(f"Target {TARGET} no encontrado en el dataset")

# =============================================================================
# AUDITORÍA ANTI-LEAKAGE
# =============================================================================

print("\n" + "="*80)
print("AUDITORÍA ANTI-LEAKAGE")
print("="*80)

all_cols = df.columns.tolist()

# Columnas con 'fwd' o 'chg' (case-insensitive)
cols_fwd = [c for c in all_cols if 'fwd' in c.lower()]
cols_chg = [c for c in all_cols if 'chg' in c.lower()]

# Excluir todas excepto el target
cols_excluded_fwd_chg = [c for c in (cols_fwd + cols_chg) if c != TARGET]
cols_excluded_fwd_chg = list(set(cols_excluded_fwd_chg))

print(f"\n🚫 Columnas excluidas por contener 'fwd': {len([c for c in cols_fwd if c != TARGET])}")
print(f"🚫 Columnas excluidas por contener 'chg': {len(cols_chg)}")

# Columnas sospechosas de leakage adicional
# - Variables que contienen 'label' o 'score' con números (pueden ser forward labels)
# - PnL calculados hacia adelante
cols_suspicious = []

for c in all_cols:
    c_lower = c.lower()
    # Labels que pueden ser forward-looking
    if 'label' in c_lower and any(x in c_lower for x in ['7', '21', '63', '252']):
        if c not in cols_excluded_fwd_chg:
            cols_suspicious.append(c)
    # Scores que pueden ser forward-looking
    if 'score' in c_lower and any(x in c_lower for x in ['7', '21', '63', '252']):
        if c not in cols_excluded_fwd_chg:
            cols_suspicious.append(c)
    # WL_PRE y RANK_PRE pueden ser forward
    if c in ['WL_PRE', 'RANK_PRE']:
        if c not in cols_excluded_fwd_chg:
            cols_suspicious.append(c)

print(f"🚫 Columnas sospechosas de leakage (labels/scores futuros): {len(cols_suspicious)}")
for c in cols_suspicious[:10]:
    print(f"   - {c}")
if len(cols_suspicious) > 10:
    print(f"   ... y {len(cols_suspicious)-10} más")

# Columnas no numéricas (excluir de análisis)
cols_non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\n📝 Columnas no numéricas (excluidas del análisis): {len(cols_non_numeric)}")
for c in cols_non_numeric:
    print(f"   - {c}")

# FEATURES PERMITIDAS (T+0)
all_excluded = set(cols_excluded_fwd_chg + cols_suspicious + cols_non_numeric + [TARGET])
features_t0 = [c for c in all_cols if c not in all_excluded]

print(f"\n✅ FEATURES T+0 PERMITIDAS: {len(features_t0)}")

# =============================================================================
# ESTADÍSTICAS BÁSICAS
# =============================================================================

print("\n" + "="*80)
print("ESTADÍSTICAS BÁSICAS")
print("="*80)

# N total
N = len(df)
print(f"\n📊 N total: {N}")

# NaNs en el target
target_nans = df[TARGET].isna().sum()
print(f"📊 NaNs en target: {target_nans} ({100*target_nans/N:.2f}%)")

# Distribución del target
target_valid = df[TARGET].dropna()
print(f"\n📈 Distribución del TARGET ({TARGET}):")
print(f"   Media:    {target_valid.mean():.4f}")
print(f"   Mediana:  {target_valid.median():.4f}")
print(f"   Std:      {target_valid.std():.4f}")
print(f"   Min:      {target_valid.min():.4f}")
print(f"   Max:      {target_valid.max():.4f}")
print(f"   Q25:      {target_valid.quantile(0.25):.4f}")
print(f"   Q75:      {target_valid.quantile(0.75):.4f}")

# Outliers (IQR)
Q1 = target_valid.quantile(0.25)
Q3 = target_valid.quantile(0.75)
IQR = Q3 - Q1
outliers_low = (target_valid < Q1 - 1.5*IQR).sum()
outliers_high = (target_valid > Q3 + 1.5*IQR).sum()
print(f"\n📊 Outliers en target (IQR ×1.5):")
print(f"   Bajos: {outliers_low} ({100*outliers_low/len(target_valid):.2f}%)")
print(f"   Altos: {outliers_high} ({100*outliers_high/len(target_valid):.2f}%)")

# Verificar columna de fecha
date_col = None
for c in ['dia', 'date', 'Date', 'DATE', 'fecha']:
    if c in df.columns:
        date_col = c
        break

if date_col:
    print(f"\n📅 Columna de fecha detectada: {date_col}")
    df[date_col] = pd.to_datetime(df[date_col])
    print(f"   Rango: {df[date_col].min()} → {df[date_col].max()}")
    validation_method = "TimeSeriesSplit"
else:
    validation_method = "KFold"

print(f"\n🔄 Método de validación: {validation_method}")

# =============================================================================
# (B) BASELINES - CORRELACIONES
# =============================================================================

print("\n" + "="*80)
print("(B) BASELINES - CORRELACIONES CON TARGET")
print("="*80)

def bootstrap_ci(x, y, func, n_boot=1000, ci=0.95):
    """Calcula IC bootstrap para correlación"""
    n = len(x)
    boot_vals = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            val = func(x.iloc[idx], y.iloc[idx])[0]
            if not np.isnan(val):
                boot_vals.append(val)
        except:
            pass
    if len(boot_vals) < 100:
        return np.nan, np.nan
    alpha = 1 - ci
    return np.percentile(boot_vals, 100*alpha/2), np.percentile(boot_vals, 100*(1-alpha/2))

# Calcular correlaciones para todas las features T+0
results = []

# Preparar datos (solo filas con target válido)
df_valid = df.dropna(subset=[TARGET]).copy()
y = df_valid[TARGET]

print(f"\nCalculando correlaciones para {len(features_t0)} features...")
print("(Esto puede tomar unos minutos...)\n")

for i, feat in enumerate(features_t0):
    if i % 20 == 0:
        print(f"  Procesando feature {i+1}/{len(features_t0)}...")

    x = df_valid[feat]

    # Eliminar NaNs pareados
    mask = ~(x.isna() | y.isna())
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 30:
        continue

    try:
        # Pearson
        pearson_r, pearson_p = pearsonr(x_clean, y_clean)

        # Spearman
        spearman_r, spearman_p = spearmanr(x_clean, y_clean)

        # Bootstrap IC para Spearman (más robusto)
        ci_low, ci_high = bootstrap_ci(x_clean.reset_index(drop=True),
                                        y_clean.reset_index(drop=True),
                                        spearmanr, n_boot=500)

        # Análisis por deciles (top vs bottom)
        try:
            deciles = pd.qcut(x_clean, 10, labels=False, duplicates='drop')
            bottom_decile = y_clean[deciles == 0]
            top_decile = y_clean[deciles == deciles.max()]

            if len(bottom_decile) >= 5 and len(top_decile) >= 5:
                lift = top_decile.median() - bottom_decile.median()
                # Mann-Whitney U test
                _, mw_p = mannwhitneyu(top_decile, bottom_decile, alternative='two-sided')
            else:
                lift = np.nan
                mw_p = np.nan
        except:
            lift = np.nan
            mw_p = np.nan

        results.append({
            'feature': feat,
            'n_valid': len(x_clean),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'spearman_ci_low': ci_low,
            'spearman_ci_high': ci_high,
            'lift_top_bottom': lift,
            'mw_pvalue': mw_p
        })

    except Exception as e:
        continue

# Crear DataFrame de resultados
df_results = pd.DataFrame(results)

# Ajuste FDR (Benjamini-Hochberg)
from scipy.stats import false_discovery_control
if len(df_results) > 0:
    # Para Spearman
    pvals = df_results['spearman_p'].values
    pvals_valid = ~np.isnan(pvals)
    df_results['spearman_p_adj'] = np.nan
    if pvals_valid.sum() > 0:
        # Usar método manual de BH
        pvals_clean = pvals[pvals_valid]
        n = len(pvals_clean)
        sorted_idx = np.argsort(pvals_clean)
        sorted_pvals = pvals_clean[sorted_idx]
        adjusted = np.zeros(n)
        adjusted[n-1] = sorted_pvals[n-1]
        for i in range(n-2, -1, -1):
            adjusted[i] = min(adjusted[i+1], sorted_pvals[i] * n / (i + 1))
        unsort_idx = np.argsort(sorted_idx)
        adjusted_unsorted = adjusted[unsort_idx]
        df_results.loc[pvals_valid, 'spearman_p_adj'] = adjusted_unsorted

# Ordenar por |spearman_r|
df_results['abs_spearman'] = df_results['spearman_r'].abs()
df_results = df_results.sort_values('abs_spearman', ascending=False)

# Mostrar top 30
print("\n" + "="*80)
print("TOP 30 FEATURES POR CORRELACIÓN SPEARMAN")
print("="*80)

top30 = df_results.head(30)
for idx, row in top30.iterrows():
    sig = "***" if row['spearman_p_adj'] < 0.001 else "**" if row['spearman_p_adj'] < 0.01 else "*" if row['spearman_p_adj'] < 0.05 else ""
    print(f"{row['feature'][:40]:<40} | ρ={row['spearman_r']:+.4f} | IC=[{row['spearman_ci_low']:+.3f},{row['spearman_ci_high']:+.3f}] | p_adj={row['spearman_p_adj']:.4f}{sig} | lift={row['lift_top_bottom']:+.3f} | n={row['n_valid']}")

# =============================================================================
# (C) ANÁLISIS POR DECILES PARA TOP FEATURES
# =============================================================================

print("\n" + "="*80)
print("(C) ANÁLISIS POR DECILES - TOP 10 FEATURES")
print("="*80)

top10_features = df_results.head(10)['feature'].tolist()

for feat in top10_features:
    print(f"\n{'='*60}")
    print(f"📊 {feat}")
    print('='*60)

    x = df_valid[feat]
    mask = ~x.isna()
    x_clean = x[mask]
    y_clean = y[mask]

    try:
        # Crear deciles
        deciles = pd.qcut(x_clean, 10, labels=False, duplicates='drop')
        n_deciles = deciles.max() + 1

        print(f"{'Decil':<8} | {'N':>6} | {'Mediana Target':>14} | {'Mean Target':>12} | {'Rango X':>20}")
        print("-"*70)

        for d in range(n_deciles):
            mask_d = deciles == d
            y_d = y_clean[mask_d]
            x_d = x_clean[mask_d]
            print(f"D{d:<7} | {len(y_d):>6} | {y_d.median():>14.3f} | {y_d.mean():>12.3f} | [{x_d.min():.2f}, {x_d.max():.2f}]")

        # Lift total
        bottom = y_clean[deciles == 0].median()
        top = y_clean[deciles == n_deciles-1].median()
        print(f"\n📈 LIFT (Top vs Bottom): {top - bottom:+.3f}")
        print(f"   Bottom decile median: {bottom:.3f}")
        print(f"   Top decile median: {top:.3f}")

    except Exception as e:
        print(f"   Error: {e}")

# =============================================================================
# (D) FEATURE ENGINEERING PARSIMONIOSO
# =============================================================================

print("\n" + "="*80)
print("(D) FEATURE ENGINEERING PARSIMONIOSO")
print("="*80)

# Usar top 10 features base para crear derivadas
top10_base = df_results.head(10)['feature'].tolist()

derived_features = {}

print(f"\nCreando features derivadas desde top 10 base...")

for feat in top10_base:
    x = df_valid[feat].copy()

    # 1. Rank (percentil)
    name = f"{feat}_rank"
    derived_features[name] = x.rank(pct=True)

    # 2. Abs
    name = f"{feat}_abs"
    derived_features[name] = x.abs()

    # 3. Log1p(abs)
    name = f"{feat}_log1p_abs"
    derived_features[name] = np.log1p(x.abs())

    # 4. Z-score robusto (mediana/MAD)
    median = x.median()
    mad = (x - median).abs().median()
    if mad > 0:
        name = f"{feat}_zscore_robust"
        derived_features[name] = (x - median) / mad

    # 5. Winsorizado (clip al 1-99 percentil)
    p01, p99 = x.quantile(0.01), x.quantile(0.99)
    name = f"{feat}_winsor"
    derived_features[name] = x.clip(p01, p99)

# Interacciones entre top 5
top5_base = top10_base[:5]
for i, f1 in enumerate(top5_base):
    for f2 in top5_base[i+1:]:
        x1 = df_valid[f1]
        x2 = df_valid[f2]

        # Ratio
        name = f"{f1[:15]}_div_{f2[:15]}"
        derived_features[name] = x1 / (x2.abs() + 1e-6)

        # Spread
        name = f"{f1[:15]}_minus_{f2[:15]}"
        derived_features[name] = x1 - x2

        # Producto
        name = f"{f1[:15]}_mult_{f2[:15]}"
        derived_features[name] = x1 * x2

print(f"✅ Features derivadas creadas: {len(derived_features)}")

# Evaluar features derivadas
print("\nEvaluando features derivadas...")

derived_results = []
for name, series in derived_features.items():
    mask = ~(series.isna() | y.isna())
    x_clean = series[mask]
    y_clean = y[mask]

    if len(x_clean) < 30:
        continue

    try:
        spearman_r, spearman_p = spearmanr(x_clean, y_clean)

        # Decile lift
        try:
            deciles = pd.qcut(x_clean, 10, labels=False, duplicates='drop')
            bottom = y_clean[deciles == 0].median()
            top = y_clean[deciles == deciles.max()].median()
            lift = top - bottom
        except:
            lift = np.nan

        derived_results.append({
            'feature': name,
            'type': 'derived',
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'lift_top_bottom': lift,
            'n_valid': len(x_clean)
        })
    except:
        continue

df_derived = pd.DataFrame(derived_results)
df_derived['abs_spearman'] = df_derived['spearman_r'].abs()
df_derived = df_derived.sort_values('abs_spearman', ascending=False)

print("\n📊 TOP 20 FEATURES DERIVADAS:")
print("-"*80)
for idx, row in df_derived.head(20).iterrows():
    print(f"{row['feature'][:50]:<50} | ρ={row['spearman_r']:+.4f} | lift={row['lift_top_bottom']:+.3f}")

# =============================================================================
# (E-F) VALIDACIÓN OOS - TimeSeriesSplit
# =============================================================================

print("\n" + "="*80)
print("(E-F) VALIDACIÓN OOS (TimeSeriesSplit)")
print("="*80)

from sklearn.model_selection import TimeSeriesSplit

# Ordenar por fecha si existe
if date_col:
    df_valid_sorted = df_valid.sort_values(date_col).reset_index(drop=True)
else:
    df_valid_sorted = df_valid.reset_index(drop=True)

# Combinar features base + derivadas top
top_base = df_results.head(15)['feature'].tolist()
top_derived = df_derived.head(15)['feature'].tolist()

# Crear DataFrame con derivadas
df_with_derived = df_valid_sorted.copy()
for name in top_derived:
    if name in derived_features:
        df_with_derived[name] = derived_features[name].values

all_candidates = top_base + top_derived

# TimeSeriesSplit
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

print(f"\nValidación con {n_splits} folds temporales")
print(f"Candidatos: {len(all_candidates)} features")

oos_results = {feat: {'spearman_oos': [], 'lift_oos': []} for feat in all_candidates}

y_sorted = df_with_derived[TARGET]

for fold, (train_idx, test_idx) in enumerate(tscv.split(df_with_derived)):
    print(f"\n  Fold {fold+1}: train={len(train_idx)}, test={len(test_idx)}")

    y_test = y_sorted.iloc[test_idx]

    for feat in all_candidates:
        if feat not in df_with_derived.columns:
            continue

        x_test = df_with_derived[feat].iloc[test_idx]

        mask = ~(x_test.isna() | y_test.isna())
        x_clean = x_test[mask]
        y_clean = y_test[mask]

        if len(x_clean) < 20:
            continue

        try:
            rho, _ = spearmanr(x_clean, y_clean)
            oos_results[feat]['spearman_oos'].append(rho)

            # Lift
            deciles = pd.qcut(x_clean, 5, labels=False, duplicates='drop')
            bottom = y_clean[deciles == 0].median()
            top = y_clean[deciles == deciles.max()].median()
            oos_results[feat]['lift_oos'].append(top - bottom)
        except:
            pass

# Calcular métricas OOS agregadas
oos_summary = []
for feat in all_candidates:
    if len(oos_results[feat]['spearman_oos']) >= 3:
        rhos = oos_results[feat]['spearman_oos']
        lifts = oos_results[feat]['lift_oos']

        oos_summary.append({
            'feature': feat,
            'spearman_oos_mean': np.mean(rhos),
            'spearman_oos_std': np.std(rhos),
            'spearman_oos_min': np.min(rhos),
            'lift_oos_mean': np.mean(lifts) if lifts else np.nan,
            'lift_oos_std': np.std(lifts) if lifts else np.nan,
            'n_folds': len(rhos),
            'stable': np.std(rhos) < 0.1 and np.min(rhos) > 0  # Criterio de estabilidad
        })

df_oos = pd.DataFrame(oos_summary)
df_oos = df_oos.sort_values('spearman_oos_mean', ascending=False)

print("\n" + "="*80)
print("📊 RANKING FINAL OOS - TOP 20")
print("="*80)

for idx, row in df_oos.head(20).iterrows():
    stable_mark = "✅" if row['stable'] else "⚠️"
    print(f"{stable_mark} {row['feature'][:45]:<45} | ρ_OOS={row['spearman_oos_mean']:+.4f}±{row['spearman_oos_std']:.3f} | lift_OOS={row['lift_oos_mean']:+.3f} | folds={row['n_folds']}")

# =============================================================================
# FEATURE REGISTRY
# =============================================================================

print("\n" + "="*80)
print("FEATURE REGISTRY")
print("="*80)

# Crear registro completo
registry = []

# Features base
for idx, row in df_results.head(20).iterrows():
    feat = row['feature']
    oos_row = df_oos[df_oos['feature'] == feat]

    registry.append({
        'feature': feat,
        'type': 'base',
        'formula': f"raw({feat})",
        'spearman_IS': row['spearman_r'],
        'spearman_IS_ci': f"[{row['spearman_ci_low']:.3f}, {row['spearman_ci_high']:.3f}]",
        'spearman_p_adj': row['spearman_p_adj'],
        'lift_IS': row['lift_top_bottom'],
        'spearman_OOS_mean': oos_row['spearman_oos_mean'].values[0] if len(oos_row) > 0 else np.nan,
        'spearman_OOS_std': oos_row['spearman_oos_std'].values[0] if len(oos_row) > 0 else np.nan,
        'lift_OOS_mean': oos_row['lift_oos_mean'].values[0] if len(oos_row) > 0 else np.nan,
        'stable': oos_row['stable'].values[0] if len(oos_row) > 0 else False,
        'n_valid': row['n_valid']
    })

# Features derivadas
for idx, row in df_derived.head(20).iterrows():
    feat = row['feature']
    oos_row = df_oos[df_oos['feature'] == feat]

    # Determinar fórmula
    if '_rank' in feat:
        formula = f"rank({feat.replace('_rank', '')})"
    elif '_abs' in feat and '_log' not in feat:
        formula = f"abs({feat.replace('_abs', '')})"
    elif '_log1p_abs' in feat:
        formula = f"log1p(abs({feat.replace('_log1p_abs', '')}))"
    elif '_zscore_robust' in feat:
        formula = f"(x - median) / MAD for {feat.replace('_zscore_robust', '')}"
    elif '_winsor' in feat:
        formula = f"winsorize({feat.replace('_winsor', '')}, 1-99%)"
    elif '_div_' in feat:
        formula = f"ratio: {feat}"
    elif '_minus_' in feat:
        formula = f"spread: {feat}"
    elif '_mult_' in feat:
        formula = f"product: {feat}"
    else:
        formula = feat

    registry.append({
        'feature': feat,
        'type': 'derived',
        'formula': formula,
        'spearman_IS': row['spearman_r'],
        'spearman_IS_ci': 'N/A',
        'spearman_p_adj': row['spearman_p'],
        'lift_IS': row['lift_top_bottom'],
        'spearman_OOS_mean': oos_row['spearman_oos_mean'].values[0] if len(oos_row) > 0 else np.nan,
        'spearman_OOS_std': oos_row['spearman_oos_std'].values[0] if len(oos_row) > 0 else np.nan,
        'lift_OOS_mean': oos_row['lift_oos_mean'].values[0] if len(oos_row) > 0 else np.nan,
        'stable': oos_row['stable'].values[0] if len(oos_row) > 0 else False,
        'n_valid': row['n_valid']
    })

df_registry = pd.DataFrame(registry)

# Guardar Feature Registry
df_registry.to_csv('feature_registry_T0_predictor.csv', index=False)
print(f"\n✅ Feature Registry guardado: feature_registry_T0_predictor.csv")

# =============================================================================
# CONCLUSIÓN FINAL
# =============================================================================

print("\n" + "="*80)
print("CONCLUSIÓN FINAL")
print("="*80)

# Identificar mejores features
best_oos = df_oos[df_oos['stable'] == True].head(5)

if len(best_oos) > 0:
    print("\n🏆 FEATURES CON SEÑAL ESTABLE OOS:")
    for idx, row in best_oos.iterrows():
        print(f"   • {row['feature']}")
        print(f"     Spearman OOS: {row['spearman_oos_mean']:+.4f} ± {row['spearman_oos_std']:.3f}")
        print(f"     Lift OOS: {row['lift_oos_mean']:+.3f}")
        print()
else:
    print("\n⚠️ NO SE ENCONTRARON FEATURES CON SEÑAL ESTABLE OOS")
    print("   Esto puede deberse a:")
    print("   - Ruido en los datos")
    print("   - No estacionariedad del mercado")
    print("   - Tamaño de efecto pequeño")
    print("   - Falta de potencia estadística")

# Mostrar también las mejores aunque no sean totalmente estables
print("\n📊 MEJORES FEATURES OOS (aunque no cumplan todos los criterios de estabilidad):")
for idx, row in df_oos.head(10).iterrows():
    sign_consistency = "+" if row['spearman_oos_min'] > 0 else "±" if row['spearman_oos_mean'] > 0 else "-"
    print(f"   {sign_consistency} {row['feature'][:50]:<50}")
    print(f"     ρ_OOS = {row['spearman_oos_mean']:+.4f} ± {row['spearman_oos_std']:.3f} (min: {row['spearman_oos_min']:+.3f})")

print("\n" + "="*80)
print("FIN DEL ANÁLISIS")
print("="*80)
