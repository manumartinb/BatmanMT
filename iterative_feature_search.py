#!/usr/bin/env python3
"""
Búsqueda Iterativa Exhaustiva de Features Predictivas
Target: PnL_fwd_pts_05_mediana
EXCLUYE: net_credit_diff (por indicación del usuario)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import TimeSeriesSplit
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

TARGET = 'PnL_fwd_pts_05_mediana'
EXCLUDE_EXPLICIT = ['net_credit_diff']  # Excluir por indicación del usuario
MIN_CORRELATION_TARGET = 0.40  # Objetivo mínimo de correlación
MAX_ITERATIONS = 20  # Máximo de rondas
MAX_FEATURES_PER_ROUND = 500  # Máximo features nuevas por ronda

print("="*80)
print("BÚSQUEDA ITERATIVA EXHAUSTIVA DE PREDICTORES")
print(f"Target: {TARGET}")
print(f"Objetivo: ρ ≥ {MIN_CORRELATION_TARGET}")
print("="*80)

# =============================================================================
# CARGA Y PREPARACIÓN
# =============================================================================

df = pd.read_csv('combined_BATMAN_mediana_w_stats_w_vix_labeled_NOSPXCHG.csv')
print(f"\n📊 Dataset: {df.shape[0]} filas × {df.shape[1]} columnas")

# Columna de fecha
df['dia'] = pd.to_datetime(df['dia'])
df = df.sort_values('dia').reset_index(drop=True)

# Filtrar columnas
all_cols = df.columns.tolist()

# Excluir por regla
cols_fwd = [c for c in all_cols if 'fwd' in c.lower() and c != TARGET]
cols_chg = [c for c in all_cols if 'chg' in c.lower()]
cols_label = [c for c in all_cols if 'label' in c.lower()]
cols_score = [c for c in all_cols if 'score' in c.lower() and any(x in c for x in ['7','21','63','252'])]
cols_non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

all_excluded = set(cols_fwd + cols_chg + cols_label + cols_score +
                   cols_non_numeric + EXCLUDE_EXPLICIT + ['RANK_PRE', 'WL_PRE'])

features_base = [c for c in all_cols if c not in all_excluded and c != TARGET]
print(f"✅ Features base permitidas: {len(features_base)}")

# Preparar datos
df_valid = df.dropna(subset=[TARGET]).copy()
y = df_valid[TARGET].values
N = len(y)
print(f"✅ N válido: {N}")

# =============================================================================
# FUNCIONES DE EVALUACIÓN
# =============================================================================

def evaluate_feature(x, y, name=""):
    """Evalúa una feature contra el target"""
    mask = ~(np.isnan(x) | np.isinf(x))
    if mask.sum() < 50:
        return None

    x_clean = x[mask]
    y_clean = y[mask]

    try:
        rho, p = spearmanr(x_clean, y_clean)
        if np.isnan(rho):
            return None
        return {
            'name': name,
            'spearman': rho,
            'abs_spearman': abs(rho),
            'p_value': p,
            'n_valid': len(x_clean)
        }
    except:
        return None

def evaluate_oos(x, y, n_splits=5):
    """Evalúa OOS con TimeSeriesSplit"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rhos = []

    for train_idx, test_idx in tscv.split(x):
        x_test = x[test_idx]
        y_test = y[test_idx]

        mask = ~(np.isnan(x_test) | np.isinf(x_test))
        if mask.sum() < 20:
            continue

        try:
            rho, _ = spearmanr(x_test[mask], y_test[mask])
            if not np.isnan(rho):
                rhos.append(rho)
        except:
            pass

    if len(rhos) >= 3:
        return {
            'mean': np.mean(rhos),
            'std': np.std(rhos),
            'min': np.min(rhos),
            'max': np.max(rhos),
            'n_folds': len(rhos)
        }
    return None

# =============================================================================
# GENERADORES DE FEATURES
# =============================================================================

def generate_unary_transforms(x, name, registry):
    """Genera transformaciones unarias"""
    transforms = {}

    # Rank percentil
    key = f"{name}_rank"
    if key not in registry:
        transforms[key] = pd.Series(x).rank(pct=True).values

    # Abs
    key = f"{name}_abs"
    if key not in registry:
        transforms[key] = np.abs(x)

    # Sign
    key = f"{name}_sign"
    if key not in registry:
        transforms[key] = np.sign(x)

    # Log1p(abs)
    key = f"{name}_log1p"
    if key not in registry:
        transforms[key] = np.log1p(np.abs(x))

    # Sqrt(abs)
    key = f"{name}_sqrt"
    if key not in registry:
        transforms[key] = np.sqrt(np.abs(x))

    # Square
    key = f"{name}_sq"
    if key not in registry:
        transforms[key] = x ** 2

    # Cube
    key = f"{name}_cube"
    if key not in registry:
        transforms[key] = x ** 3

    # Inverse
    key = f"{name}_inv"
    if key not in registry:
        with np.errstate(divide='ignore', invalid='ignore'):
            transforms[key] = 1.0 / (x + 1e-10)

    # Z-score robusto
    key = f"{name}_zrob"
    if key not in registry:
        median = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - median))
        if mad > 0:
            transforms[key] = (x - median) / mad

    # Winsorize 5-95
    key = f"{name}_win"
    if key not in registry:
        p5, p95 = np.nanpercentile(x, [5, 95])
        transforms[key] = np.clip(x, p5, p95)

    # Tanh (squash)
    key = f"{name}_tanh"
    if key not in registry:
        transforms[key] = np.tanh(x / (np.nanstd(x) + 1e-10))

    return transforms

def generate_binary_operations(x1, x2, name1, name2, registry):
    """Genera operaciones binarias entre dos features"""
    ops = {}

    # Suma
    key = f"{name1[:12]}+{name2[:12]}"
    if key not in registry:
        ops[key] = x1 + x2

    # Diferencia
    key = f"{name1[:12]}-{name2[:12]}"
    if key not in registry:
        ops[key] = x1 - x2

    # Producto
    key = f"{name1[:12]}*{name2[:12]}"
    if key not in registry:
        ops[key] = x1 * x2

    # Ratio
    key = f"{name1[:12]}/{name2[:12]}"
    if key not in registry:
        with np.errstate(divide='ignore', invalid='ignore'):
            ops[key] = x1 / (np.abs(x2) + 1e-10)

    # Ratio inverso
    key = f"{name2[:12]}/{name1[:12]}"
    if key not in registry:
        with np.errstate(divide='ignore', invalid='ignore'):
            ops[key] = x2 / (np.abs(x1) + 1e-10)

    # Media geométrica (con signo)
    key = f"geom({name1[:10]},{name2[:10]})"
    if key not in registry:
        sign = np.sign(x1) * np.sign(x2)
        ops[key] = sign * np.sqrt(np.abs(x1 * x2))

    # Media armónica
    key = f"harm({name1[:10]},{name2[:10]})"
    if key not in registry:
        with np.errstate(divide='ignore', invalid='ignore'):
            ops[key] = 2 * x1 * x2 / (np.abs(x1) + np.abs(x2) + 1e-10)

    # Diff normalizada
    key = f"ndiff({name1[:10]},{name2[:10]})"
    if key not in registry:
        with np.errstate(divide='ignore', invalid='ignore'):
            ops[key] = (x1 - x2) / (np.abs(x1) + np.abs(x2) + 1e-10)

    # Max
    key = f"max({name1[:10]},{name2[:10]})"
    if key not in registry:
        ops[key] = np.maximum(x1, x2)

    # Min
    key = f"min({name1[:10]},{name2[:10]})"
    if key not in registry:
        ops[key] = np.minimum(x1, x2)

    return ops

def generate_ternary_operations(x1, x2, x3, n1, n2, n3, registry):
    """Genera operaciones ternarias"""
    ops = {}

    # Media
    key = f"avg3({n1[:8]},{n2[:8]},{n3[:8]})"
    if key not in registry:
        ops[key] = (x1 + x2 + x3) / 3

    # Suma ponderada
    key = f"wsum({n1[:8]},{n2[:8]},{n3[:8]})"
    if key not in registry:
        ops[key] = 0.5*x1 + 0.3*x2 + 0.2*x3

    # Producto triple
    key = f"prod3({n1[:8]},{n2[:8]},{n3[:8]})"
    if key not in registry:
        ops[key] = x1 * x2 * x3

    # Ratio compuesto
    key = f"rat3({n1[:8]},{n2[:8]},{n3[:8]})"
    if key not in registry:
        with np.errstate(divide='ignore', invalid='ignore'):
            ops[key] = x1 / (np.abs(x2 * x3) + 1e-10)

    return ops

# =============================================================================
# BÚSQUEDA ITERATIVA
# =============================================================================

# Registro de todas las features evaluadas
feature_registry = {}
feature_data = {}  # Almacena los arrays

# Fase 0: Evaluar features base
print("\n" + "="*80)
print("FASE 0: Evaluación de features base")
print("="*80)

base_results = []
for feat in features_base:
    x = df_valid[feat].values.astype(float)
    result = evaluate_feature(x, y, feat)
    if result:
        base_results.append(result)
        feature_registry[feat] = result
        feature_data[feat] = x

base_results = sorted(base_results, key=lambda x: x['abs_spearman'], reverse=True)

print(f"\n📊 Top 20 features base:")
for r in base_results[:20]:
    print(f"   {r['name'][:40]:<40} | ρ = {r['spearman']:+.4f}")

best_base = base_results[0]['abs_spearman']
print(f"\n🏆 Mejor correlación base: {best_base:.4f}")

# Tracking del mejor
best_overall = {'name': base_results[0]['name'], 'spearman': base_results[0]['spearman'],
                'abs_spearman': best_base, 'round': 0}

# =============================================================================
# ITERACIONES
# =============================================================================

for iteration in range(1, MAX_ITERATIONS + 1):
    print("\n" + "="*80)
    print(f"ITERACIÓN {iteration}")
    print("="*80)

    # Seleccionar top features para combinar
    top_n = min(30, len(feature_registry))
    sorted_features = sorted(feature_registry.items(),
                            key=lambda x: x[1]['abs_spearman'],
                            reverse=True)[:top_n]
    top_names = [f[0] for f in sorted_features]

    print(f"Combinando top {len(top_names)} features...")

    new_features = {}

    # 1. Transformaciones unarias de las mejores
    print("  → Generando transformaciones unarias...")
    for name in top_names[:15]:
        if name in feature_data:
            x = feature_data[name]
            transforms = generate_unary_transforms(x, name, feature_registry)
            new_features.update(transforms)

    # 2. Operaciones binarias entre top features
    print("  → Generando operaciones binarias...")
    for i, name1 in enumerate(top_names[:20]):
        for name2 in top_names[i+1:20]:
            if name1 in feature_data and name2 in feature_data:
                x1 = feature_data[name1]
                x2 = feature_data[name2]
                ops = generate_binary_operations(x1, x2, name1, name2, feature_registry)
                new_features.update(ops)

                if len(new_features) > MAX_FEATURES_PER_ROUND:
                    break
        if len(new_features) > MAX_FEATURES_PER_ROUND:
            break

    # 3. Operaciones ternarias (solo top 10)
    print("  → Generando operaciones ternarias...")
    for combo in combinations(top_names[:10], 3):
        n1, n2, n3 = combo
        if n1 in feature_data and n2 in feature_data and n3 in feature_data:
            x1, x2, x3 = feature_data[n1], feature_data[n2], feature_data[n3]
            ops = generate_ternary_operations(x1, x2, x3, n1, n2, n3, feature_registry)
            new_features.update(ops)

            if len(new_features) > MAX_FEATURES_PER_ROUND:
                break

    print(f"  → Evaluando {len(new_features)} nuevas features...")

    # Evaluar nuevas features
    round_results = []
    for name, x in new_features.items():
        if name in feature_registry:
            continue
        result = evaluate_feature(x, y, name)
        if result:
            round_results.append(result)
            feature_registry[name] = result
            feature_data[name] = x

    if not round_results:
        print("  ❌ No se encontraron nuevas features válidas")
        continue

    round_results = sorted(round_results, key=lambda x: x['abs_spearman'], reverse=True)

    # Mostrar mejores de esta ronda
    print(f"\n  📊 Top 10 nuevas features (ronda {iteration}):")
    for r in round_results[:10]:
        improvement = r['abs_spearman'] - best_overall['abs_spearman']
        marker = "🔥" if improvement > 0 else "  "
        print(f"   {marker} {r['name'][:45]:<45} | ρ = {r['spearman']:+.4f}")

    # Actualizar mejor si hay mejora
    if round_results[0]['abs_spearman'] > best_overall['abs_spearman']:
        best_overall = {
            'name': round_results[0]['name'],
            'spearman': round_results[0]['spearman'],
            'abs_spearman': round_results[0]['abs_spearman'],
            'round': iteration
        }
        print(f"\n  🏆 NUEVO MEJOR: {best_overall['name']}")
        print(f"     ρ = {best_overall['spearman']:+.4f}")

    # Verificar objetivo
    if best_overall['abs_spearman'] >= MIN_CORRELATION_TARGET:
        print(f"\n🎯 ¡OBJETIVO ALCANZADO! ρ = {best_overall['abs_spearman']:.4f} ≥ {MIN_CORRELATION_TARGET}")
        break

    print(f"\n  📈 Mejor actual: {best_overall['abs_spearman']:.4f} (objetivo: {MIN_CORRELATION_TARGET})")
    print(f"  📦 Total features en registro: {len(feature_registry)}")

# =============================================================================
# VALIDACIÓN OOS DE LOS MEJORES
# =============================================================================

print("\n" + "="*80)
print("VALIDACIÓN OOS DE TOP 30 FEATURES")
print("="*80)

# Obtener top 30
top_30 = sorted(feature_registry.items(),
                key=lambda x: x[1]['abs_spearman'],
                reverse=True)[:30]

oos_results = []
for name, result in top_30:
    if name in feature_data:
        x = feature_data[name]
        oos = evaluate_oos(x, y)
        if oos:
            oos_results.append({
                'name': name,
                'spearman_IS': result['spearman'],
                'spearman_OOS_mean': oos['mean'],
                'spearman_OOS_std': oos['std'],
                'spearman_OOS_min': oos['min'],
                'n_folds': oos['n_folds']
            })

oos_results = sorted(oos_results, key=lambda x: x['spearman_OOS_mean'], reverse=True)

print("\n📊 Ranking por Spearman OOS:")
print("-"*90)
for r in oos_results[:20]:
    stable = "✅" if r['spearman_OOS_min'] > 0 else "⚠️"
    print(f"{stable} {r['name'][:50]:<50} | IS={r['spearman_IS']:+.3f} | OOS={r['spearman_OOS_mean']:+.3f}±{r['spearman_OOS_std']:.3f}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

print(f"\n📊 Total features evaluadas: {len(feature_registry)}")
print(f"🏆 Mejor correlación IS encontrada: {best_overall['abs_spearman']:.4f}")
print(f"   Feature: {best_overall['name']}")
print(f"   Encontrada en ronda: {best_overall['round']}")

if oos_results:
    best_oos = oos_results[0]
    print(f"\n🏆 Mejor correlación OOS:")
    print(f"   Feature: {best_oos['name']}")
    print(f"   ρ_OOS = {best_oos['spearman_OOS_mean']:+.4f} ± {best_oos['spearman_OOS_std']:.3f}")

# Guardar resultados
results_df = pd.DataFrame([
    {
        'rank': i+1,
        'feature': name,
        'spearman_IS': result['spearman'],
        'abs_spearman_IS': result['abs_spearman'],
        'p_value': result['p_value'],
        'n_valid': result['n_valid']
    }
    for i, (name, result) in enumerate(sorted(feature_registry.items(),
                                              key=lambda x: x[1]['abs_spearman'],
                                              reverse=True)[:100])
])

results_df.to_csv('feature_search_results.csv', index=False)
print(f"\n✅ Resultados guardados en: feature_search_results.csv")

# Guardar OOS
if oos_results:
    oos_df = pd.DataFrame(oos_results)
    oos_df.to_csv('feature_search_oos_results.csv', index=False)
    print(f"✅ Resultados OOS guardados en: feature_search_oos_results.csv")

print("\n" + "="*80)
print("FIN DE LA BÚSQUEDA")
print("="*80)
