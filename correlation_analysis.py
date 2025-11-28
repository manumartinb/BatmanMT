#!/usr/bin/env python3
"""
Análisis de Correlación: PNL_FWD_PTS vs Variables Objetivo
============================================================
Estudio estadístico profesional de las correlaciones entre las ventas (PNL_FWD_PTS)
y las variables: LABEL_GENERAL_SCORE, BQI_ABS, FF_ATM, delta_total, theta_total, FF_BAT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================
print("="*80)
print("ANÁLISIS DE CORRELACIÓN: PNL_FWD_PTS vs VARIABLES OBJETIVO")
print("="*80)
print("\n[1/6] Cargando datos...")

# Cargar el archivo CSV
df = pd.read_csv('MIDTERM_combined_mediana_labeled.csv')
print(f"✓ Datos cargados: {len(df):,} registros")

# Definir las variables de interés
ventas_vars = [
    'PnL_fwd_pts_01_mediana',
    'PnL_fwd_pts_05_mediana',
    'PnL_fwd_pts_25_mediana',
    'PnL_fwd_pts_50_mediana',
    'PnL_fwd_pts_90_mediana'
]

objetivo_vars = [
    'LABEL_GENERAL_SCORE',
    'BQI_ABS',
    'FF_ATM',
    'delta_total',
    'theta_total',
    'FF_BAT'
]

# Verificar que todas las columnas existen
all_vars = ventas_vars + objetivo_vars
missing_cols = [col for col in all_vars if col not in df.columns]
if missing_cols:
    print(f"⚠ ADVERTENCIA: Columnas faltantes: {missing_cols}")
    all_vars = [col for col in all_vars if col in df.columns]

# Crear subset con las variables de interés
df_analysis = df[all_vars].copy()

# Eliminar filas con valores NaN
df_clean = df_analysis.dropna()
print(f"✓ Datos limpios (sin NaN): {len(df_clean):,} registros")
print(f"  Registros eliminados: {len(df) - len(df_clean):,} ({100*(len(df) - len(df_clean))/len(df):.2f}%)")

# ============================================================
# 2. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================
print("\n[2/6] Calculando estadísticas descriptivas...")

# Guardar estadísticas descriptivas
stats_desc = df_clean.describe().T
stats_desc['median'] = df_clean.median()
stats_desc['skewness'] = df_clean.skew()
stats_desc['kurtosis'] = df_clean.kurtosis()

# Guardar a CSV
stats_desc.to_csv('estadisticas_descriptivas.csv')
print("✓ Estadísticas descriptivas guardadas en: estadisticas_descriptivas.csv")

# ============================================================
# 3. ANÁLISIS DE CORRELACIÓN
# ============================================================
print("\n[3/6] Calculando correlaciones...")

# Matriz de correlación de Pearson
corr_pearson = df_clean.corr(method='pearson')
corr_pearson_subset = corr_pearson.loc[ventas_vars, objetivo_vars]

# Matriz de correlación de Spearman
corr_spearman = df_clean.corr(method='spearman')
corr_spearman_subset = corr_spearman.loc[ventas_vars, objetivo_vars]

# Guardar correlaciones
corr_pearson_subset.to_csv('correlaciones_pearson.csv')
corr_spearman_subset.to_csv('correlaciones_spearman.csv')
print("✓ Correlaciones de Pearson guardadas en: correlaciones_pearson.csv")
print("✓ Correlaciones de Spearman guardadas en: correlaciones_spearman.csv")

# ============================================================
# 4. PRUEBAS DE SIGNIFICANCIA ESTADÍSTICA
# ============================================================
print("\n[4/6] Realizando pruebas de significancia estadística...")

# Crear matrices para p-valores
pvalues_pearson = pd.DataFrame(index=ventas_vars, columns=objetivo_vars, dtype=float)
pvalues_spearman = pd.DataFrame(index=ventas_vars, columns=objetivo_vars, dtype=float)

# Calcular p-valores
for venta_var in ventas_vars:
    for obj_var in objetivo_vars:
        # Pearson
        r_pearson, p_pearson = pearsonr(df_clean[venta_var], df_clean[obj_var])
        pvalues_pearson.loc[venta_var, obj_var] = p_pearson

        # Spearman
        r_spearman, p_spearman = spearmanr(df_clean[venta_var], df_clean[obj_var])
        pvalues_spearman.loc[venta_var, obj_var] = p_spearman

# Guardar p-valores
pvalues_pearson.to_csv('pvalues_pearson.csv')
pvalues_spearman.to_csv('pvalues_spearman.csv')
print("✓ P-valores de Pearson guardados en: pvalues_pearson.csv")
print("✓ P-valores de Spearman guardados en: pvalues_spearman.csv")

# Crear matriz de significancia (marcadores para p < 0.05, p < 0.01, p < 0.001)
def get_significance_markers(pvalues):
    """Retorna marcadores de significancia estadística"""
    markers = pd.DataFrame(index=pvalues.index, columns=pvalues.columns, dtype=str)
    for i in pvalues.index:
        for j in pvalues.columns:
            p = pvalues.loc[i, j]
            if p < 0.001:
                markers.loc[i, j] = '***'
            elif p < 0.01:
                markers.loc[i, j] = '**'
            elif p < 0.05:
                markers.loc[i, j] = '*'
            else:
                markers.loc[i, j] = 'ns'
    return markers

sig_markers_pearson = get_significance_markers(pvalues_pearson)
sig_markers_spearman = get_significance_markers(pvalues_spearman)

# ============================================================
# 5. VISUALIZACIONES
# ============================================================
print("\n[5/6] Generando visualizaciones...")

# 5.1. Heatmap de Correlaciones de Pearson
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_pearson_subset, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Coeficiente de Correlación de Pearson'})
ax.set_title('Correlaciones de Pearson: Ventas (PNL_FWD_PTS) vs Variables Objetivo',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Variables Objetivo', fontsize=12, fontweight='bold')
ax.set_ylabel('Variables de Ventas (Mediana)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('heatmap_pearson.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Heatmap Pearson guardado en: heatmap_pearson.png")

# 5.2. Heatmap de Correlaciones de Spearman
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_spearman_subset, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Coeficiente de Correlación de Spearman'})
ax.set_title('Correlaciones de Spearman: Ventas (PNL_FWD_PTS) vs Variables Objetivo',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Variables Objetivo', fontsize=12, fontweight='bold')
ax.set_ylabel('Variables de Ventas (Mediana)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('heatmap_spearman.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Heatmap Spearman guardado en: heatmap_spearman.png")

# 5.3. Heatmap de Significancia Estadística (Pearson)
fig, ax = plt.subplots(figsize=(14, 10))
# Crear anotaciones con correlación + significancia
annot_pearson = pd.DataFrame(index=corr_pearson_subset.index,
                              columns=corr_pearson_subset.columns, dtype=str)
for i in corr_pearson_subset.index:
    for j in corr_pearson_subset.columns:
        corr_val = corr_pearson_subset.loc[i, j]
        sig_mark = sig_markers_pearson.loc[i, j]
        annot_pearson.loc[i, j] = f'{corr_val:.3f}\n{sig_mark}'

sns.heatmap(corr_pearson_subset, annot=annot_pearson, fmt='', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Coeficiente de Correlación'})
ax.set_title('Correlaciones de Pearson con Significancia Estadística\n(***p<0.001, **p<0.01, *p<0.05, ns=no significativo)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Variables Objetivo', fontsize=12, fontweight='bold')
ax.set_ylabel('Variables de Ventas (Mediana)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('heatmap_pearson_significancia.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Heatmap con significancia guardado en: heatmap_pearson_significancia.png")

# 5.4. Scatter plots para las correlaciones más fuertes
print("\n  Generando scatter plots para correlaciones más fuertes...")

# Encontrar las 12 correlaciones más fuertes (en valor absoluto)
corr_flat = []
for venta in ventas_vars:
    for obj in objetivo_vars:
        corr_flat.append({
            'venta': venta,
            'objetivo': obj,
            'pearson': abs(corr_pearson_subset.loc[venta, obj]),
            'spearman': abs(corr_spearman_subset.loc[venta, obj]),
            'pearson_raw': corr_pearson_subset.loc[venta, obj],
            'p_value': pvalues_pearson.loc[venta, obj]
        })

corr_df = pd.DataFrame(corr_flat)
corr_df = corr_df.sort_values('pearson', ascending=False)

# Top 12 correlaciones
top_corr = corr_df.head(12)

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
axes = axes.flatten()

for idx, (_, row) in enumerate(top_corr.iterrows()):
    venta = row['venta']
    obj = row['objetivo']
    ax = axes[idx]

    # Scatter plot
    ax.scatter(df_clean[obj], df_clean[venta], alpha=0.5, s=20, edgecolors='k', linewidths=0.5)

    # Línea de regresión
    z = np.polyfit(df_clean[obj], df_clean[venta], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean[obj].min(), df_clean[obj].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Regresión lineal')

    # Título con información de correlación
    pearson_val = row['pearson_raw']
    p_val = row['p_value']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

    ax.set_title(f'{venta.replace("PnL_fwd_pts_", "PNL_").replace("_mediana", "")}\nvs {obj}\n' +
                 f'r={pearson_val:.3f} ({sig})', fontsize=10, fontweight='bold')
    ax.set_xlabel(obj, fontsize=9)
    ax.set_ylabel(venta.replace("PnL_fwd_pts_", "PNL_").replace("_mediana", ""), fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Top 12 Correlaciones más Fuertes: Scatter Plots',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('scatter_plots_top_correlaciones.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Scatter plots guardados en: scatter_plots_top_correlaciones.png")

# 5.5. Distribuciones de las variables
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, var in enumerate(objetivo_vars):
    ax = axes[idx]
    df_clean[var].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribución: {var}', fontsize=12, fontweight='bold')
    ax.set_xlabel(var, fontsize=10)
    ax.set_ylabel('Frecuencia', fontsize=10)
    ax.axvline(df_clean[var].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df_clean[var].mean():.2f}')
    ax.axvline(df_clean[var].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df_clean[var].median():.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Distribuciones de Variables Objetivo', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('distribuciones_variables_objetivo.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Distribuciones guardadas en: distribuciones_variables_objetivo.png")

# 5.6. Matriz de correlación completa
fig, ax = plt.subplots(figsize=(16, 14))
corr_full = df_clean.corr(method='pearson')
mask = np.triu(np.ones_like(corr_full, dtype=bool))
sns.heatmap(corr_full, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Coeficiente de Correlación de Pearson'},
            annot_kws={'fontsize': 8})
ax.set_title('Matriz de Correlación Completa (Pearson)', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('matriz_correlacion_completa.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Matriz completa guardada en: matriz_correlacion_completa.png")

# ============================================================
# 6. GENERACIÓN DE INFORME
# ============================================================
print("\n[6/6] Generando informe estadístico...")

report_lines = []
report_lines.append("# INFORME ESTADÍSTICO: ANÁLISIS DE CORRELACIONES")
report_lines.append("## Correlación entre Ventas (PNL_FWD_PTS) y Variables Objetivo")
report_lines.append("")
report_lines.append("---")
report_lines.append("")

# Resumen ejecutivo
report_lines.append("## 1. RESUMEN EJECUTIVO")
report_lines.append("")
report_lines.append(f"- **Total de registros analizados:** {len(df_clean):,}")
report_lines.append(f"- **Registros eliminados (NaN):** {len(df) - len(df_clean):,} ({100*(len(df) - len(df_clean))/len(df):.2f}%)")
report_lines.append(f"- **Variables de ventas analizadas:** {len(ventas_vars)}")
report_lines.append(f"- **Variables objetivo analizadas:** {len(objetivo_vars)}")
report_lines.append("")

# Identificar correlaciones más fuertes
report_lines.append("### Correlaciones más Fuertes (Pearson |r| > 0.3)")
report_lines.append("")
strong_corr = corr_df[corr_df['pearson'] > 0.3]
if len(strong_corr) > 0:
    report_lines.append("| Venta | Variable Objetivo | Correlación (r) | p-valor | Significancia |")
    report_lines.append("|-------|-------------------|-----------------|---------|---------------|")
    for _, row in strong_corr.iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'
        report_lines.append(f"| {row['venta']} | {row['objetivo']} | {row['pearson_raw']:.4f} | {row['p_value']:.4e} | {sig} |")
else:
    report_lines.append("No se encontraron correlaciones fuertes (|r| > 0.3)")
report_lines.append("")

# Estadísticas descriptivas
report_lines.append("---")
report_lines.append("")
report_lines.append("## 2. ESTADÍSTICAS DESCRIPTIVAS")
report_lines.append("")
report_lines.append("### Variables de Ventas (PNL_FWD_PTS)")
report_lines.append("")
report_lines.append("| Variable | Media | Mediana | Desv. Est. | Min | Max |")
report_lines.append("|----------|-------|---------|------------|-----|-----|")
for var in ventas_vars:
    report_lines.append(f"| {var} | {stats_desc.loc[var, 'mean']:.2f} | {stats_desc.loc[var, 'median']:.2f} | " +
                       f"{stats_desc.loc[var, 'std']:.2f} | {stats_desc.loc[var, 'min']:.2f} | {stats_desc.loc[var, 'max']:.2f} |")
report_lines.append("")

report_lines.append("### Variables Objetivo")
report_lines.append("")
report_lines.append("| Variable | Media | Mediana | Desv. Est. | Min | Max |")
report_lines.append("|----------|-------|---------|------------|-----|-----|")
for var in objetivo_vars:
    report_lines.append(f"| {var} | {stats_desc.loc[var, 'mean']:.2f} | {stats_desc.loc[var, 'median']:.2f} | " +
                       f"{stats_desc.loc[var, 'std']:.2f} | {stats_desc.loc[var, 'min']:.2f} | {stats_desc.loc[var, 'max']:.2f} |")
report_lines.append("")

# Correlaciones por variable objetivo
report_lines.append("---")
report_lines.append("")
report_lines.append("## 3. ANÁLISIS DE CORRELACIÓN POR VARIABLE OBJETIVO")
report_lines.append("")

for obj_var in objetivo_vars:
    report_lines.append(f"### {obj_var}")
    report_lines.append("")
    report_lines.append("| PNL Variable | Pearson (r) | p-valor | Spearman (ρ) | p-valor | Interpretación |")
    report_lines.append("|--------------|-------------|---------|--------------|---------|----------------|")

    for venta_var in ventas_vars:
        r_p = corr_pearson_subset.loc[venta_var, obj_var]
        p_p = pvalues_pearson.loc[venta_var, obj_var]
        r_s = corr_spearman_subset.loc[venta_var, obj_var]
        p_s = pvalues_spearman.loc[venta_var, obj_var]

        # Interpretación
        if abs(r_p) < 0.1:
            interp = "Muy débil"
        elif abs(r_p) < 0.3:
            interp = "Débil"
        elif abs(r_p) < 0.5:
            interp = "Moderada"
        elif abs(r_p) < 0.7:
            interp = "Fuerte"
        else:
            interp = "Muy fuerte"

        sig_p = '***' if p_p < 0.001 else '**' if p_p < 0.01 else '*' if p_p < 0.05 else 'ns'
        sig_s = '***' if p_s < 0.001 else '**' if p_s < 0.01 else '*' if p_s < 0.05 else 'ns'

        report_lines.append(f"| {venta_var} | {r_p:.4f} {sig_p} | {p_p:.4e} | {r_s:.4f} {sig_s} | {p_s:.4e} | {interp} |")

    report_lines.append("")

# Metodología
report_lines.append("---")
report_lines.append("")
report_lines.append("## 4. METODOLOGÍA")
report_lines.append("")
report_lines.append("### Coeficientes de Correlación")
report_lines.append("")
report_lines.append("- **Pearson (r):** Mide la correlación lineal entre dos variables continuas. Sensible a outliers.")
report_lines.append("- **Spearman (ρ):** Mide la correlación monotónica entre dos variables. Robusto ante outliers y no asume linealidad.")
report_lines.append("")
report_lines.append("### Niveles de Significancia")
report_lines.append("")
report_lines.append("- `***` p < 0.001 (altamente significativo)")
report_lines.append("- `**` p < 0.01 (muy significativo)")
report_lines.append("- `*` p < 0.05 (significativo)")
report_lines.append("- `ns` p ≥ 0.05 (no significativo)")
report_lines.append("")
report_lines.append("### Interpretación de Correlaciones")
report_lines.append("")
report_lines.append("| Rango |r| | Interpretación |")
report_lines.append("|-----------|----------------|")
report_lines.append("| 0.00 - 0.10 | Muy débil |")
report_lines.append("| 0.10 - 0.30 | Débil |")
report_lines.append("| 0.30 - 0.50 | Moderada |")
report_lines.append("| 0.50 - 0.70 | Fuerte |")
report_lines.append("| 0.70 - 1.00 | Muy fuerte |")
report_lines.append("")

# Hallazgos clave
report_lines.append("---")
report_lines.append("")
report_lines.append("## 5. HALLAZGOS CLAVE")
report_lines.append("")

# Análisis automático de hallazgos
hallazgos = []

# 1. Variable objetivo con mayor correlación promedio
avg_corr_per_obj = {}
for obj_var in objetivo_vars:
    avg_corr = corr_pearson_subset[obj_var].abs().mean()
    avg_corr_per_obj[obj_var] = avg_corr

max_avg_var = max(avg_corr_per_obj, key=avg_corr_per_obj.get)
hallazgos.append(f"1. **Variable objetivo con mayor correlación promedio:** {max_avg_var} " +
                f"(|r| promedio = {avg_corr_per_obj[max_avg_var]:.3f})")

# 2. Horizonte temporal con correlaciones más fuertes
avg_corr_per_venta = {}
for venta_var in ventas_vars:
    avg_corr = corr_pearson_subset.loc[venta_var].abs().mean()
    avg_corr_per_venta[venta_var] = avg_corr

max_avg_venta = max(avg_corr_per_venta, key=avg_corr_per_venta.get)
hallazgos.append(f"2. **Horizonte temporal con correlaciones más fuertes:** {max_avg_venta} " +
                f"(|r| promedio = {avg_corr_per_venta[max_avg_venta]:.3f})")

# 3. Correlación más fuerte encontrada
max_corr_row = corr_df.iloc[0]
hallazgos.append(f"3. **Correlación más fuerte:** {max_corr_row['venta']} vs {max_corr_row['objetivo']} " +
                f"(r = {max_corr_row['pearson_raw']:.4f}, p = {max_corr_row['p_value']:.4e})")

# 4. Número de correlaciones significativas
sig_count = len(corr_df[corr_df['p_value'] < 0.05])
total_count = len(corr_df)
hallazgos.append(f"4. **Correlaciones estadísticamente significativas (p < 0.05):** {sig_count}/{total_count} " +
                f"({100*sig_count/total_count:.1f}%)")

# 5. Correlaciones moderadas o fuertes
moderate_strong = len(corr_df[corr_df['pearson'] >= 0.3])
hallazgos.append(f"5. **Correlaciones moderadas o fuertes (|r| ≥ 0.3):** {moderate_strong}/{total_count} " +
                f"({100*moderate_strong/total_count:.1f}%)")

for hallazgo in hallazgos:
    report_lines.append(hallazgo)
    report_lines.append("")

# Conclusiones
report_lines.append("---")
report_lines.append("")
report_lines.append("## 6. CONCLUSIONES")
report_lines.append("")
report_lines.append("Este análisis proporciona una evaluación completa de las correlaciones entre las variables de ventas (PNL_FWD_PTS) ")
report_lines.append("y las variables objetivo especificadas. Los resultados incluyen:")
report_lines.append("")
report_lines.append("- Correlaciones de Pearson y Spearman para evaluar relaciones lineales y monotónicas")
report_lines.append("- Pruebas de significancia estadística para validar la fiabilidad de las correlaciones")
report_lines.append("- Visualizaciones comprehensivas (heatmaps, scatter plots, distribuciones)")
report_lines.append("- Análisis detallado por variable y horizonte temporal")
report_lines.append("")
report_lines.append("Los archivos generados incluyen:")
report_lines.append("- `correlaciones_pearson.csv` / `correlaciones_spearman.csv`: Matrices de correlación")
report_lines.append("- `pvalues_pearson.csv` / `pvalues_spearman.csv`: Matrices de p-valores")
report_lines.append("- `estadisticas_descriptivas.csv`: Estadísticas descriptivas de todas las variables")
report_lines.append("- Múltiples gráficos en formato PNG de alta resolución (300 DPI)")
report_lines.append("")

# Guardar el informe
report_text = '\n'.join(report_lines)
with open('INFORME_CORRELACIONES.md', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("✓ Informe estadístico guardado en: INFORME_CORRELACIONES.md")

# ============================================================
# 7. RESUMEN EN CONSOLA
# ============================================================
print("\n" + "="*80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)
print("\nARCHIVOS GENERADOS:")
print("  📊 Datos:")
print("     - estadisticas_descriptivas.csv")
print("     - correlaciones_pearson.csv")
print("     - correlaciones_spearman.csv")
print("     - pvalues_pearson.csv")
print("     - pvalues_spearman.csv")
print("\n  📈 Gráficos:")
print("     - heatmap_pearson.png")
print("     - heatmap_spearman.png")
print("     - heatmap_pearson_significancia.png")
print("     - scatter_plots_top_correlaciones.png")
print("     - distribuciones_variables_objetivo.png")
print("     - matriz_correlacion_completa.png")
print("\n  📝 Informe:")
print("     - INFORME_CORRELACIONES.md")
print("\n" + "="*80)
print("\nTop 5 Correlaciones más Fuertes:")
print("="*80)
for idx, (_, row) in enumerate(corr_df.head(5).iterrows(), 1):
    sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'
    print(f"{idx}. {row['venta']:30s} vs {row['objetivo']:20s}")
    print(f"   Pearson r = {row['pearson_raw']:7.4f} ({sig}), p = {row['p_value']:.4e}")
    print()

print("="*80)
print("Análisis finalizado. Revisa los archivos generados para más detalles.")
print("="*80)
