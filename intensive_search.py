#!/usr/bin/env python3
"""
Búsqueda INTENSIVA de Features Predictivas
Objetivo: Encontrar correlación muy alta (ρ ≥ 0.45)
Más operaciones matemáticas, más combinaciones
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

TARGET = 'PnL_fwd_pts_05_mediana'
EXCLUDE_EXPLICIT = ['net_credit_diff']
MIN_CORRELATION_TARGET = 0.50
MAX_ITERATIONS = 30

print("="*80)
print("BÚSQUEDA INTENSIVA - OPERACIONES AVANZADAS")
print(f"Objetivo: ρ ≥ {MIN_CORRELATION_TARGET}")
print("="*80)

# =============================================================================
# CARGA Y PREPARACIÓN
# =============================================================================

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
                   cols_non_numeric + EXCLUDE_EXPLICIT + ['RANK_PRE', 'WL_PRE'])

features_base = [c for c in all_cols if c not in all_excluded and c != TARGET]

df_valid = df.dropna(subset=[TARGET]).copy()
y = df_valid[TARGET].values
N = len(y)

print(f"Features base: {len(features_base)}, N={N}")

# =============================================================================
# FUNCIONES AVANZADAS
# =============================================================================

def eval_corr(x, y):
    """Evalúa correlación Spearman"""
    mask = ~(np.isnan(x) | np.isinf(x))
    if mask.sum() < 50:
        return np.nan
    try:
        rho, _ = spearmanr(x[mask], y[mask])
        return rho if not np.isnan(rho) else np.nan
    except:
        return np.nan

def eval_oos(x, y, n_splits=5):
    """Evalúa OOS"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rhos = []
    for _, test_idx in tscv.split(x):
        x_t, y_t = x[test_idx], y[test_idx]
        mask = ~(np.isnan(x_t) | np.isinf(x_t))
        if mask.sum() >= 20:
            try:
                rho, _ = spearmanr(x_t[mask], y_t[mask])
                if not np.isnan(rho):
                    rhos.append(rho)
            except:
                pass
    return (np.mean(rhos), np.std(rhos), np.min(rhos)) if len(rhos) >= 3 else (np.nan, np.nan, np.nan)

# =============================================================================
# OPERACIONES MATEMÁTICAS EXPANDIDAS
# =============================================================================

def apply_unary(x, name, ops_done):
    """Transformaciones unarias expandidas"""
    results = {}
    eps = 1e-10

    transforms = {
        'rank': lambda a: pd.Series(a).rank(pct=True).values,
        'abs': np.abs,
        'sign': np.sign,
        'log1p': lambda a: np.log1p(np.abs(a)),
        'sqrt': lambda a: np.sqrt(np.abs(a)),
        'sq': lambda a: a**2,
        'cube': lambda a: a**3,
        'inv': lambda a: 1/(a + eps),
        'neg': lambda a: -a,
        'exp_neg': lambda a: np.exp(-np.abs(a)/np.nanstd(a)),
        'tanh': lambda a: np.tanh(a/(np.nanstd(a) + eps)),
        'sigmoid': lambda a: 1/(1 + np.exp(-a/(np.nanstd(a) + eps))),
        'zscore': lambda a: (a - np.nanmean(a))/(np.nanstd(a) + eps),
        'zrob': lambda a: (a - np.nanmedian(a))/(np.nanmedian(np.abs(a - np.nanmedian(a))) + eps),
        'win5': lambda a: np.clip(a, *np.nanpercentile(a, [5, 95])),
        'win1': lambda a: np.clip(a, *np.nanpercentile(a, [1, 99])),
        'decile': lambda a: pd.qcut(pd.Series(a), 10, labels=False, duplicates='drop').values,
        'quintile': lambda a: pd.qcut(pd.Series(a), 5, labels=False, duplicates='drop').values,
        'pos': lambda a: np.where(a > 0, a, 0),
        'neg_only': lambda a: np.where(a < 0, a, 0),
        'log_sign': lambda a: np.sign(a) * np.log1p(np.abs(a)),
        'cbrt': lambda a: np.cbrt(a),  # raíz cúbica (preserva signo)
        'exp_clip': lambda a: np.clip(np.exp(a/(np.nanstd(a)+eps)), 0, 100),
    }

    for tname, tfunc in transforms.items():
        key = f"{name}_{tname}"
        if key not in ops_done:
            try:
                results[key] = tfunc(x)
            except:
                pass
    return results

def apply_binary(x1, x2, n1, n2, ops_done):
    """Operaciones binarias expandidas"""
    results = {}
    eps = 1e-10

    operations = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / (np.abs(b) + eps),
        'max': np.maximum,
        'min': np.minimum,
        'geom': lambda a, b: np.sign(a*b) * np.sqrt(np.abs(a*b)),
        'harm': lambda a, b: 2*a*b / (np.abs(a) + np.abs(b) + eps),
        'ndiff': lambda a, b: (a - b) / (np.abs(a) + np.abs(b) + eps),
        'hypot': lambda a, b: np.sqrt(a**2 + b**2),
        'atan2': lambda a, b: np.arctan2(a, b + eps),
        'pow': lambda a, b: np.sign(a) * np.power(np.abs(a) + eps, np.clip(b/10, -2, 2)),
        'wgt7030': lambda a, b: 0.7*a + 0.3*b,
        'wgt5050': lambda a, b: 0.5*a + 0.5*b,
        'absdiff': lambda a, b: np.abs(a - b),
        'sqsum': lambda a, b: np.sqrt(a**2 + b**2),
        'lograt': lambda a, b: np.log((np.abs(a) + eps) / (np.abs(b) + eps)),
        'modulo': lambda a, b: np.mod(a, np.abs(b) + eps),
        'clip_by': lambda a, b: np.clip(a, -np.abs(b), np.abs(b)),
        'cond_pos': lambda a, b: np.where(a > 0, b, -b),
        'cond_neg': lambda a, b: np.where(a < 0, b, -b),
        'rank_prod': lambda a, b: pd.Series(a).rank(pct=True).values * pd.Series(b).rank(pct=True).values,
        'rank_diff': lambda a, b: pd.Series(a).rank(pct=True).values - pd.Series(b).rank(pct=True).values,
    }

    for opname, opfunc in operations.items():
        key = f"{n1[:10]}_{opname}_{n2[:10]}"
        if key not in ops_done:
            try:
                results[key] = opfunc(x1, x2)
            except:
                pass
    return results

def apply_ternary(x1, x2, x3, n1, n2, n3, ops_done):
    """Operaciones ternarias"""
    results = {}
    eps = 1e-10

    operations = {
        'avg3': lambda a, b, c: (a + b + c) / 3,
        'wsum': lambda a, b, c: 0.5*a + 0.3*b + 0.2*c,
        'prod3': lambda a, b, c: a * b * c,
        'rat3': lambda a, b, c: a / (np.abs(b * c) + eps),
        'geom3': lambda a, b, c: np.sign(a*b*c) * np.power(np.abs(a*b*c) + eps, 1/3),
        'harm3': lambda a, b, c: 3 / (1/(a+eps) + 1/(b+eps) + 1/(c+eps)),
        'med3': lambda a, b, c: np.median([a, b, c], axis=0),
        'max3': lambda a, b, c: np.maximum(np.maximum(a, b), c),
        'min3': lambda a, b, c: np.minimum(np.minimum(a, b), c),
        'range3': lambda a, b, c: np.maximum(np.maximum(a, b), c) - np.minimum(np.minimum(a, b), c),
        'cond3': lambda a, b, c: np.where(a > 0, b, c),
        'blend3': lambda a, b, c: a * b + (1 - a) * c if np.all((a >= 0) & (a <= 1)) else (a + b + c) / 3,
    }

    for opname, opfunc in operations.items():
        key = f"{opname}({n1[:6]},{n2[:6]},{n3[:6]})"
        if key not in ops_done:
            try:
                results[key] = opfunc(x1, x2, x3)
            except:
                pass
    return results

def apply_quaternary(x1, x2, x3, x4, n1, n2, n3, n4, ops_done):
    """Operaciones de 4 variables"""
    results = {}
    eps = 1e-10

    operations = {
        'avg4': lambda a, b, c, d: (a + b + c + d) / 4,
        'prod4': lambda a, b, c, d: a * b * c * d,
        'rat4': lambda a, b, c, d: (a * b) / (np.abs(c * d) + eps),
        'diffrat': lambda a, b, c, d: (a - b) / (np.abs(c - d) + eps),
        'geom4': lambda a, b, c, d: np.sign(a*b*c*d) * np.power(np.abs(a*b*c*d) + eps, 0.25),
        'max4': lambda a, b, c, d: np.maximum(np.maximum(a, b), np.maximum(c, d)),
        'min4': lambda a, b, c, d: np.minimum(np.minimum(a, b), np.minimum(c, d)),
        'range4': lambda a, b, c, d: np.maximum(np.maximum(a, b), np.maximum(c, d)) - np.minimum(np.minimum(a, b), np.minimum(c, d)),
    }

    for opname, opfunc in operations.items():
        key = f"{opname}4({n1[:5]},{n2[:5]},{n3[:5]},{n4[:5]})"
        if key not in ops_done:
            try:
                results[key] = opfunc(x1, x2, x3, x4)
            except:
                pass
    return results

# =============================================================================
# REGISTRO Y DATOS
# =============================================================================

registry = {}  # nombre -> correlación
data = {}      # nombre -> array

# Inicializar con features base
print("\nEvaluando features base...")
for feat in features_base:
    x = df_valid[feat].values.astype(float)
    rho = eval_corr(x, y)
    if not np.isnan(rho):
        registry[feat] = rho
        data[feat] = x

print(f"Features base evaluadas: {len(registry)}")

# Ordenar por correlación absoluta
sorted_base = sorted(registry.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nTop 10 features base:")
for name, rho in sorted_base[:10]:
    print(f"  {name[:40]:<40} | ρ = {rho:+.4f}")

best = max(abs(v) for v in registry.values())
best_name = [k for k, v in registry.items() if abs(v) == best][0]
print(f"\nMejor base: {best_name} (ρ = {registry[best_name]:+.4f})")

# =============================================================================
# ITERACIONES
# =============================================================================

for iteration in range(1, MAX_ITERATIONS + 1):
    print(f"\n{'='*80}")
    print(f"ITERACIÓN {iteration}")
    print(f"{'='*80}")

    # Top features para combinar
    sorted_all = sorted(registry.items(), key=lambda x: abs(x[1]), reverse=True)
    top_n = min(25, len(sorted_all))
    top_features = [(name, data[name]) for name, _ in sorted_all[:top_n] if name in data]

    new_features = {}

    # 1. Transformaciones unarias (top 12)
    print("  → Unarias...")
    for name, x in top_features[:12]:
        new_features.update(apply_unary(x, name, registry))

    # 2. Operaciones binarias (top 15 x top 15)
    print("  → Binarias...")
    for i, (n1, x1) in enumerate(top_features[:15]):
        for n2, x2 in top_features[i+1:15]:
            new_features.update(apply_binary(x1, x2, n1, n2, registry))
            if len(new_features) > 800:
                break
        if len(new_features) > 800:
            break

    # 3. Operaciones ternarias (top 8)
    print("  → Ternarias...")
    top8 = top_features[:8]
    for combo in combinations(range(len(top8)), 3):
        n1, x1 = top8[combo[0]]
        n2, x2 = top8[combo[1]]
        n3, x3 = top8[combo[2]]
        new_features.update(apply_ternary(x1, x2, x3, n1, n2, n3, registry))
        if len(new_features) > 1200:
            break

    # 4. Operaciones cuaternarias (top 6)
    print("  → Cuaternarias...")
    top6 = top_features[:6]
    for combo in combinations(range(len(top6)), 4):
        n1, x1 = top6[combo[0]]
        n2, x2 = top6[combo[1]]
        n3, x3 = top6[combo[2]]
        n4, x4 = top6[combo[3]]
        new_features.update(apply_quaternary(x1, x2, x3, x4, n1, n2, n3, n4, registry))
        if len(new_features) > 1500:
            break

    # Evaluar nuevas features
    print(f"  → Evaluando {len(new_features)} nuevas features...")

    improvements = []
    for name, x in new_features.items():
        if name in registry:
            continue
        rho = eval_corr(x, y)
        if not np.isnan(rho):
            registry[name] = rho
            data[name] = x
            if abs(rho) > best:
                improvements.append((name, rho))

    # Actualizar mejor
    current_best = max(abs(v) for v in registry.values())
    if current_best > best:
        best = current_best
        best_name = [k for k, v in registry.items() if abs(v) == best][0]
        print(f"\n  🔥 NUEVO MEJOR: {best_name}")
        print(f"     ρ = {registry[best_name]:+.4f}")

    # Mostrar mejores de la ronda
    new_sorted = sorted([(n, registry[n]) for n in new_features.keys() if n in registry],
                        key=lambda x: abs(x[1]), reverse=True)[:10]
    print(f"\n  Top 10 nuevas (ronda {iteration}):")
    for name, rho in new_sorted:
        marker = "🔥" if abs(rho) >= best else "  "
        print(f"  {marker} {name[:50]:<50} | ρ = {rho:+.4f}")

    print(f"\n  📈 Mejor actual: {best:.4f} | Total features: {len(registry)}")

    # Verificar objetivo
    if best >= MIN_CORRELATION_TARGET:
        print(f"\n🎯 ¡OBJETIVO ALCANZADO! ρ = {best:.4f}")
        break

    # Verificar estancamiento
    if iteration > 5 and len(improvements) == 0:
        print("\n⚠️ Sin mejoras en esta ronda")

# =============================================================================
# VALIDACIÓN OOS FINAL
# =============================================================================

print("\n" + "="*80)
print("VALIDACIÓN OOS - TOP 50")
print("="*80)

sorted_final = sorted(registry.items(), key=lambda x: abs(x[1]), reverse=True)[:50]

oos_results = []
for name, rho_is in sorted_final:
    if name in data:
        mean_oos, std_oos, min_oos = eval_oos(data[name], y)
        if not np.isnan(mean_oos):
            oos_results.append({
                'name': name,
                'spearman_IS': rho_is,
                'spearman_OOS_mean': mean_oos,
                'spearman_OOS_std': std_oos,
                'spearman_OOS_min': min_oos,
                'stable': min_oos > 0
            })

oos_results = sorted(oos_results, key=lambda x: abs(x['spearman_OOS_mean']), reverse=True)

print("\nTop 20 por OOS:")
for r in oos_results[:20]:
    mark = "✅" if r['stable'] else "⚠️"
    print(f"{mark} {r['name'][:55]:<55} | IS={r['spearman_IS']:+.3f} | OOS={r['spearman_OOS_mean']:+.3f}±{r['spearman_OOS_std']:.3f}")

# =============================================================================
# RESUMEN
# =============================================================================

print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

print(f"\n📊 Total features evaluadas: {len(registry)}")
print(f"🏆 Mejor ρ IS: {best:.4f}")
print(f"   Feature: {best_name}")

if oos_results:
    best_oos = max(oos_results, key=lambda x: abs(x['spearman_OOS_mean']))
    print(f"\n🏆 Mejor ρ OOS: {best_oos['spearman_OOS_mean']:+.4f}")
    print(f"   Feature: {best_oos['name']}")

# Guardar resultados
pd.DataFrame([
    {'feature': name, 'spearman': rho, 'abs_spearman': abs(rho)}
    for name, rho in sorted(registry.items(), key=lambda x: abs(x[1]), reverse=True)[:200]
]).to_csv('intensive_search_results.csv', index=False)

pd.DataFrame(oos_results).to_csv('intensive_search_oos.csv', index=False)

print("\n✅ Resultados guardados")
