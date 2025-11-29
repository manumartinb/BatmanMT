"""
Generador de Visualizaciones - Análisis de Correlaciones PnL vs Drivers
Dataset: MIDTERM_combined_mediana_labeled.csv
Genera 6 gráficos profesionales en PNG (300 DPI)
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# ================================
# CONFIGURACIÓN
# ================================

DATASET_NAME = "MIDTERM_combined_mediana_labeled"
CSV_FILE = f"{DATASET_NAME}.csv"

# Variables Driver (independientes)
DRIVERS = [
    'LABEL_GENERAL_SCORE',
    'BQI_ABS',
    'FF_ATM',
    'delta_total',
    'theta_total',
    'FF_BAT'
]

# Variables PnL (dependientes)
PNL_VARS = [
    'PnL_fwd_pts_01_mediana',
    'PnL_fwd_pts_05_mediana',
    'PnL_fwd_pts_25_mediana',
    'PnL_fwd_pts_50_mediana',
    'PnL_fwd_pts_90_mediana'
]

# Nombres cortos para ventanas
PNL_SHORT_NAMES = {
    'PnL_fwd_pts_01_mediana': '01d',
    'PnL_fwd_pts_05_mediana': '05d',
    'PnL_fwd_pts_25_mediana': '25d',
    'PnL_fwd_pts_50_mediana': '50d',
    'PnL_fwd_pts_90_mediana': '90d'
}

# ================================
# FUNCIONES AUXILIARES
# ================================

def clean_data(df, drivers, pnl_vars):
    """Limpia datos: reemplaza inf con NaN y elimina filas con NaN"""
    df_clean = df.copy()
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    cols_needed = drivers + pnl_vars
    cols_available = [col for col in cols_needed if col in df_clean.columns]
    df_clean = df_clean.dropna(subset=cols_available)
    return df_clean

def calc_correlation_with_pvalue(x, y, method='pearson'):
    """Calcula correlación y p-value"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 3:
        return np.nan, np.nan
    if method == 'pearson':
        corr, pval = pearsonr(x_clean, y_clean)
    else:
        from scipy.stats import spearmanr
        corr, pval = spearmanr(x_clean, y_clean)
    return corr, pval

# ================================
# CARGA Y PREPARACIÓN DE DATOS
# ================================

print("Cargando datos...")
df = pd.read_csv(CSV_FILE)
print(f"Total de filas: {len(df)}")

# Verificar columnas disponibles
drivers_available = [d for d in DRIVERS if d in df.columns]
pnl_available = [p for p in PNL_VARS if p in df.columns]

print(f"Drivers disponibles: {drivers_available}")
print(f"Variables PnL disponibles: {pnl_available}")

# Limpiar datos
df_clean = clean_data(df, drivers_available, pnl_available)
print(f"Filas válidas: {len(df_clean)}")

# Calcular correlaciones Pearson
corr_pearson = pd.DataFrame(index=drivers_available, columns=pnl_available)
for driver in drivers_available:
    for pnl in pnl_available:
        corr, _ = calc_correlation_with_pvalue(
            df_clean[driver].values,
            df_clean[pnl].values,
            method='pearson'
        )
        corr_pearson.loc[driver, pnl] = corr

corr_pearson = corr_pearson.astype(float)

# Ranking de drivers
ranking_data = []
for driver in drivers_available:
    corrs = [abs(corr_pearson.loc[driver, pnl]) for pnl in pnl_available
             if not np.isnan(corr_pearson.loc[driver, pnl])]
    if corrs:
        avg_corr = np.mean(corrs)
        ranking_data.append({'Driver': driver, 'Correlación_Promedio_Abs': avg_corr})

ranking_df = pd.DataFrame(ranking_data)
ranking_df = ranking_df.sort_values('Correlación_Promedio_Abs', ascending=False)
best_driver = ranking_df.iloc[0]['Driver']

print(f"\nMejor driver identificado: {best_driver}")
print()

# ================================
# GRÁFICO 1: HEATMAP DE CORRELACIONES
# ================================

print("Generando Gráfico 1: Heatmap de Correlaciones...")

fig, ax = plt.subplots(figsize=(10, 6))

# Preparar datos para heatmap
corr_display = corr_pearson.copy()
corr_display.columns = [PNL_SHORT_NAMES.get(c, c) for c in corr_display.columns]

# Crear heatmap
sns.heatmap(
    corr_display.astype(float),
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    center=0,
    vmin=-0.3,
    vmax=0.3,
    cbar_kws={'label': 'Correlación de Pearson'},
    linewidths=0.5,
    linecolor='gray',
    ax=ax
)

ax.set_title(f'Matriz de Correlación: Drivers vs PnL Forward Points\n{DATASET_NAME}',
             fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Ventanas PnL (días)', fontsize=11, fontweight='bold')
ax.set_ylabel('Variables Driver', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{DATASET_NAME}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Guardado: {DATASET_NAME}_correlation_heatmap.png")

# ================================
# GRÁFICO 2: PNL POR CUARTILES DEL MEJOR DRIVER
# ================================

print(f"Generando Gráfico 2: PnL por Cuartiles de {best_driver}...")

# Calcular cuartiles
quartiles = pd.qcut(df_clean[best_driver], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
df_clean['quartile'] = quartiles

# Preparar datos para gráfico
quartile_means = {}
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    quartile_means[q] = []
    subset = df_clean[df_clean['quartile'] == q]
    for pnl in pnl_available:
        quartile_means[q].append(subset[pnl].mean())

# Crear gráfico
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(pnl_available))
width = 0.2

colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # Q1=rojo, Q2=naranja, Q3=verde, Q4=azul

for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, quartile_means[q], width, label=q, color=colors[i], alpha=0.8)

    # Añadir valores sobre las barras
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Ventanas PnL', fontsize=11, fontweight='bold')
ax.set_ylabel('PnL Medio (pts)', fontsize=11, fontweight='bold')
ax.set_title(f'PnL por Cuartiles de {best_driver}\n{DATASET_NAME}',
             fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([PNL_SHORT_NAMES.get(p, p) for p in pnl_available])
ax.legend(title='Cuartil', loc='best')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig(f'{DATASET_NAME}_pnl_by_{best_driver.lower()}_quartiles.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Guardado: {DATASET_NAME}_pnl_by_{best_driver.lower()}_quartiles.png")

# ================================
# GRÁFICO 3: SCATTER PLOTS - MEJOR DRIVER VS PNL
# ================================

print(f"Generando Gráfico 3: Scatter Plots {best_driver} vs PnL...")

# Configurar subplots
n_pnl = len(pnl_available)
if n_pnl <= 4:
    nrows, ncols = 2, 2
else:
    nrows, ncols = 2, 3

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
axes = axes.flatten()

for i, pnl in enumerate(pnl_available):
    ax = axes[i]

    # Datos limpios
    x = df_clean[best_driver].values
    y = df_clean[pnl].values

    # Scatter plot
    ax.scatter(x, y, alpha=0.4, s=20, color='steelblue', edgecolors='none')

    # Línea de tendencia
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() > 2:
        slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
        x_line = np.array([x[mask].min(), x[mask].max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'y={slope:.2f}x+{intercept:.2f}')

        # Correlación en el título
        corr, _ = calc_correlation_with_pvalue(x, y, method='pearson')
        ax.set_title(f'{PNL_SHORT_NAMES.get(pnl, pnl)} (r={corr:.3f})',
                     fontsize=11, fontweight='bold')

    ax.set_xlabel(best_driver, fontsize=10)
    ax.set_ylabel('PnL (pts)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8)

# Ocultar subplots vacíos si los hay
for i in range(n_pnl, len(axes)):
    axes[i].axis('off')

fig.suptitle(f'Análisis de Correlación: {best_driver} vs PnL Forward Points\n{DATASET_NAME}',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig(f'{DATASET_NAME}_scatter_{best_driver.lower()}_vs_pnl.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Guardado: {DATASET_NAME}_scatter_{best_driver.lower()}_vs_pnl.png")

# ================================
# GRÁFICO 4: RANKING DE DRIVERS
# ================================

print("Generando Gráfico 4: Ranking de Drivers...")

fig, ax = plt.subplots(figsize=(10, 6))

# Ordenar de menor a mayor para gráfico horizontal
ranking_sorted = ranking_df.sort_values('Correlación_Promedio_Abs', ascending=True)

colors_rank = ['darkgreen' if d == best_driver else 'steelblue'
               for d in ranking_sorted['Driver']]

bars = ax.barh(ranking_sorted['Driver'], ranking_sorted['Correlación_Promedio_Abs'],
               color=colors_rank, alpha=0.8)

# Añadir valores al final de las barras
for i, (bar, val) in enumerate(zip(bars, ranking_sorted['Correlación_Promedio_Abs'])):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}',
            ha='left', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Correlación Promedio Absoluta', fontsize=11, fontweight='bold')
ax.set_ylabel('Variable Driver', fontsize=11, fontweight='bold')
ax.set_title(f'Ranking de Drivers por Poder Predictivo\n{DATASET_NAME}',
             fontsize=13, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{DATASET_NAME}_driver_rankings.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Guardado: {DATASET_NAME}_driver_rankings.png")

# ================================
# GRÁFICO 5: PNL POR RANGOS DEL MEJOR DRIVER
# ================================

print(f"Generando Gráfico 5: PnL por Rangos de {best_driver}...")

# Crear 5 bins personalizados
bins = df_clean[best_driver].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
bin_labels = ['Muy Bajo\n(0-20%)', 'Bajo\n(20-40%)', 'Medio\n(40-60%)',
              'Alto\n(60-80%)', 'Muy Alto\n(80-100%)']

df_clean['range_bin'] = pd.cut(df_clean[best_driver], bins=bins, labels=bin_labels, include_lowest=True)

# Calcular medias por bin
range_means = {}
for pnl in pnl_available:
    range_means[pnl] = []
    for bin_label in bin_labels:
        subset = df_clean[df_clean['range_bin'] == bin_label]
        range_means[pnl].append(subset[pnl].mean())

# Crear gráfico
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(bin_labels))
n_pnl = len(pnl_available)
width = 0.15

colors_pnl = plt.cm.viridis(np.linspace(0.2, 0.9, n_pnl))

for i, pnl in enumerate(pnl_available):
    offset = (i - (n_pnl-1)/2) * width
    bars = ax.bar(x + offset, range_means[pnl], width,
                   label=PNL_SHORT_NAMES.get(pnl, pnl),
                   color=colors_pnl[i], alpha=0.8)

    # Valores sobre barras
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 5:  # Solo mostrar si es significativo
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=7)

ax.set_xlabel(f'Rango de {best_driver}', fontsize=11, fontweight='bold')
ax.set_ylabel('PnL Medio (pts)', fontsize=11, fontweight='bold')
ax.set_title(f'PnL por Rangos de {best_driver}\n{DATASET_NAME}',
             fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(bin_labels)
ax.legend(title='Ventana PnL', loc='best', ncol=n_pnl)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig(f'{DATASET_NAME}_pnl_by_{best_driver.lower()}_ranges.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Guardado: {DATASET_NAME}_pnl_by_{best_driver.lower()}_ranges.png")

# ================================
# GRÁFICO 6: ANÁLISIS VARIABLE ESPECIAL
# ================================

print("Generando Gráfico 6: Análisis Variable Especial...")

if 'LABEL_GENERAL_SCORE' in drivers_available:
    # Análisis de LABEL_GENERAL_SCORE
    print("Analizando LABEL_GENERAL_SCORE...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Obtener categorías únicas ordenadas
    categories = sorted(df_clean['LABEL_GENERAL_SCORE'].unique())

    # Calcular PnL medio por categoría
    pnl_by_category = {}
    for pnl in pnl_available:
        pnl_by_category[pnl] = []
        for cat in categories:
            subset = df_clean[df_clean['LABEL_GENERAL_SCORE'] == cat]
            pnl_by_category[pnl].append(subset[pnl].mean())

    # Gráfico de líneas
    for pnl in pnl_available:
        ax.plot(categories, pnl_by_category[pnl],
                marker='o', linewidth=2, markersize=8,
                label=PNL_SHORT_NAMES.get(pnl, pnl))

    ax.set_xlabel('LABEL_GENERAL_SCORE', fontsize=11, fontweight='bold')
    ax.set_ylabel('PnL Medio (pts)', fontsize=11, fontweight='bold')
    ax.set_title(f'PnL por Categorías de LABEL_GENERAL_SCORE\n{DATASET_NAME}',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(title='Ventana PnL', loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Añadir advertencia si hay paradoja
    first_cat_pnl = pnl_by_category[pnl_available[0]][0]
    last_cat_pnl = pnl_by_category[pnl_available[0]][-1]

    if first_cat_pnl > last_cat_pnl:
        ax.text(0.5, 0.95, '⚠️ CORRELACIÓN INVERSA DETECTADA',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{DATASET_NAME}_pnl_by_label_general_score_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Guardado: {DATASET_NAME}_pnl_by_label_general_score_analysis.png")

else:
    # Análisis del segundo mejor driver
    if len(ranking_df) >= 2:
        second_best = ranking_df.iloc[1]['Driver']
        print(f"Analizando segundo mejor driver: {second_best}...")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Crear deciles
        deciles = pd.qcut(df_clean[second_best], q=10, labels=range(1, 11), duplicates='drop')
        df_clean['decile'] = deciles

        # Calcular PnL medio por decil
        decile_means = {}
        for pnl in pnl_available:
            decile_means[pnl] = []
            for dec in range(1, 11):
                subset = df_clean[df_clean['decile'] == dec]
                if len(subset) > 0:
                    decile_means[pnl].append(subset[pnl].mean())
                else:
                    decile_means[pnl].append(np.nan)

        # Gráfico de líneas
        x_deciles = list(range(1, 11))
        for pnl in pnl_available:
            ax.plot(x_deciles, decile_means[pnl],
                    marker='o', linewidth=2, markersize=6,
                    label=PNL_SHORT_NAMES.get(pnl, pnl))

        ax.set_xlabel(f'{second_best} (Deciles)', fontsize=11, fontweight='bold')
        ax.set_ylabel('PnL Medio (pts)', fontsize=11, fontweight='bold')
        ax.set_title(f'PnL por Deciles de {second_best}\n{DATASET_NAME}',
                     fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(x_deciles)
        ax.legend(title='Ventana PnL', loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        plt.tight_layout()
        plt.savefig(f'{DATASET_NAME}_pnl_by_{second_best.lower()}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Guardado: {DATASET_NAME}_pnl_by_{second_best.lower()}_analysis.png")

# ================================
# RESUMEN
# ================================

print()
print("="*80)
print("VISUALIZACIONES COMPLETADAS")
print("="*80)
print()
print("Archivos generados:")
print(f"  1. {DATASET_NAME}_correlation_heatmap.png")
print(f"  2. {DATASET_NAME}_pnl_by_{best_driver.lower()}_quartiles.png")
print(f"  3. {DATASET_NAME}_scatter_{best_driver.lower()}_vs_pnl.png")
print(f"  4. {DATASET_NAME}_driver_rankings.png")
print(f"  5. {DATASET_NAME}_pnl_by_{best_driver.lower()}_ranges.png")
print(f"  6. {DATASET_NAME}_pnl_by_*_analysis.png")
print()
print("✓ Todas las visualizaciones generadas exitosamente (300 DPI)")
