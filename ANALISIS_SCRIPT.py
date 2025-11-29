"""
================================================================================
ANÁLISIS ESTADÍSTICO COMPLETO: CORRELACIONES PnL vs DRIVERS
================================================================================

Script unificado que realiza:
1. Análisis estadístico completo (9 secciones)
2. Generación de 6 visualizaciones profesionales (PNG 300 DPI)
3. Resumen ejecutivo en Markdown

Autor: Sistema BatmanMT
Fecha: 2025-11-29
Versión: 1.0
================================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURACIÓN - MODIFICAR ESTAS RUTAS SEGÚN NECESIDAD
# ================================================================================

# Archivo de entrada (CSV)
INPUT_CSV = "MIDTERM_combined_mediana_labeled.csv"

# Carpeta de salida para todos los archivos generados
OUTPUT_DIR = "output_analisis"

# Prefijo para nombres de archivos de salida (se usará el nombre del dataset por defecto)
OUTPUT_PREFIX = None  # Si es None, se usa el nombre del archivo CSV sin extensión

# Variables Driver (independientes) a analizar
DRIVERS = [
    'LABEL_GENERAL_SCORE',
    'BQI_ABS',
    'FF_ATM',
    'delta_total',
    'theta_total',
    'FF_BAT'
]

# Variables PnL (dependientes) a analizar
PNL_VARS = [
    'PnL_fwd_pts_01_mediana',
    'PnL_fwd_pts_05_mediana',
    'PnL_fwd_pts_25_mediana',
    'PnL_fwd_pts_50_mediana',
    'PnL_fwd_pts_90_mediana'
]

# Nombres cortos para ventanas PnL (para gráficos)
PNL_SHORT_NAMES = {
    'PnL_fwd_pts_01_mediana': '01d',
    'PnL_fwd_pts_05_mediana': '05d',
    'PnL_fwd_pts_25_mediana': '25d',
    'PnL_fwd_pts_50_mediana': '50d',
    'PnL_fwd_pts_90_mediana': '90d'
}

# Configuración de gráficos
PLOT_DPI = 300
PLOT_STYLE = "whitegrid"

# ================================================================================
# FUNCIONES AUXILIARES
# ================================================================================

def setup_output_directory(output_dir):
    """Crea el directorio de salida si no existe"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Directorio creado: {output_dir}")
    return output_dir

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
    elif method == 'spearman':
        corr, pval = spearmanr(x_clean, y_clean)
    else:
        return np.nan, np.nan

    return corr, pval

def format_pvalue(pval):
    """Formatea p-value con estrellas de significancia"""
    if np.isnan(pval):
        return "N/A"
    if pval < 0.001:
        return f"{pval:.4f}***"
    elif pval < 0.01:
        return f"{pval:.4f}**"
    elif pval < 0.05:
        return f"{pval:.4f}*"
    else:
        return f"{pval:.4f}"

# ================================================================================
# CLASE PRINCIPAL DE ANÁLISIS
# ================================================================================

class AnalisisPnLDrivers:

    def __init__(self, csv_path, output_dir, output_prefix=None, drivers=None, pnl_vars=None):
        self.csv_path = csv_path
        self.output_dir = setup_output_directory(output_dir)

        # Determinar prefijo de salida
        if output_prefix is None:
            self.output_prefix = os.path.splitext(os.path.basename(csv_path))[0]
        else:
            self.output_prefix = output_prefix

        self.drivers = drivers if drivers else DRIVERS
        self.pnl_vars = pnl_vars if pnl_vars else PNL_VARS

        # Datos
        self.df = None
        self.df_clean = None
        self.total_rows = 0
        self.valid_rows = 0
        self.valid_pct = 0

        # Resultados
        self.drivers_available = []
        self.pnl_available = []
        self.corr_pearson = None
        self.corr_spearman = None
        self.pval_pearson = None
        self.pval_spearman = None
        self.ranking_df = None
        self.best_driver = None

        # Reportes
        self.report_lines = []

    def log(self, message, to_report=True):
        """Imprime y opcionalmente guarda en reporte"""
        print(message)
        if to_report:
            self.report_lines.append(message)

    def cargar_datos(self):
        """Carga y limpia los datos"""
        self.log("="*80)
        self.log("ANÁLISIS ESTADÍSTICO COMPLETO: CORRELACIONES PnL vs DRIVERS")
        self.log(f"Dataset: {self.output_prefix}")
        self.log("="*80)
        self.log("")

        self.log(f"Cargando datos desde: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        self.total_rows = len(self.df)
        self.log(f"Total de filas cargadas: {self.total_rows}")

        # Verificar columnas disponibles
        self.drivers_available = [d for d in self.drivers if d in self.df.columns]
        self.pnl_available = [p for p in self.pnl_vars if p in self.df.columns]

        self.log(f"Drivers disponibles: {len(self.drivers_available)}/{len(self.drivers)}")
        self.log(f"Variables PnL disponibles: {len(self.pnl_available)}/{len(self.pnl_vars)}")

        if not self.drivers_available or not self.pnl_available:
            raise ValueError("ERROR: No hay suficientes variables para el análisis")

        # Limpiar datos
        self.log("\nLimpiando datos...")
        self.df_clean = clean_data(self.df, self.drivers_available, self.pnl_available)
        self.valid_rows = len(self.df_clean)
        self.valid_pct = (self.valid_rows / self.total_rows) * 100

        self.log(f"Filas válidas después de limpieza: {self.valid_rows}")
        self.log(f"Porcentaje válido: {self.valid_pct:.2f}%")
        self.log("")

    def seccion_1_estadisticas_descriptivas(self):
        """Sección 1: Estadísticas Descriptivas"""
        self.log("="*80)
        self.log("SECCIÓN 1: ESTADÍSTICAS DESCRIPTIVAS")
        self.log("="*80)
        self.log("")

        self.log("--- DRIVERS ---")
        desc_drivers = self.df_clean[self.drivers_available].describe()
        self.log(desc_drivers.to_string())
        self.log("")

        self.log("--- VARIABLES PnL ---")
        desc_pnl = self.df_clean[self.pnl_available].describe()
        self.log(desc_pnl.to_string())
        self.log("")

    def seccion_2_correlaciones(self):
        """Sección 2: Correlaciones con PnL"""
        self.log("="*80)
        self.log("SECCIÓN 2: CORRELACIONES CON PNL")
        self.log("="*80)
        self.log("")

        # Matriz de correlación Pearson
        self.log("--- CORRELACIÓN DE PEARSON ---")
        self.corr_pearson = pd.DataFrame(index=self.drivers_available, columns=self.pnl_available)
        self.pval_pearson = pd.DataFrame(index=self.drivers_available, columns=self.pnl_available)

        for driver in self.drivers_available:
            for pnl in self.pnl_available:
                corr, pval = calc_correlation_with_pvalue(
                    self.df_clean[driver].values,
                    self.df_clean[pnl].values,
                    method='pearson'
                )
                self.corr_pearson.loc[driver, pnl] = corr
                self.pval_pearson.loc[driver, pnl] = pval

        self.corr_pearson = self.corr_pearson.astype(float)
        self.pval_pearson = self.pval_pearson.astype(float)

        # Mostrar correlaciones
        corr_pearson_display = self.corr_pearson.copy()
        corr_pearson_display.columns = [PNL_SHORT_NAMES.get(c, c) for c in corr_pearson_display.columns]
        self.log(corr_pearson_display.to_string(float_format='%.4f'))
        self.log("")

        # P-values
        self.log("--- P-VALUES (Pearson) ---")
        self.log("(* p<0.05, ** p<0.01, *** p<0.001)")
        pval_display = self.pval_pearson.copy()
        pval_display.columns = [PNL_SHORT_NAMES.get(c, c) for c in pval_display.columns]
        for col in pval_display.columns:
            pval_display[col] = pval_display[col].apply(format_pvalue)
        self.log(pval_display.to_string())
        self.log("")

        # Matriz de correlación Spearman
        self.log("--- CORRELACIÓN DE SPEARMAN ---")
        self.corr_spearman = pd.DataFrame(index=self.drivers_available, columns=self.pnl_available)
        self.pval_spearman = pd.DataFrame(index=self.drivers_available, columns=self.pnl_available)

        for driver in self.drivers_available:
            for pnl in self.pnl_available:
                corr, pval = calc_correlation_with_pvalue(
                    self.df_clean[driver].values,
                    self.df_clean[pnl].values,
                    method='spearman'
                )
                self.corr_spearman.loc[driver, pnl] = corr
                self.pval_spearman.loc[driver, pnl] = pval

        self.corr_spearman = self.corr_spearman.astype(float)
        self.pval_spearman = self.pval_spearman.astype(float)

        corr_spearman_display = self.corr_spearman.copy()
        corr_spearman_display.columns = [PNL_SHORT_NAMES.get(c, c) for c in corr_spearman_display.columns]
        self.log(corr_spearman_display.to_string(float_format='%.4f'))
        self.log("")

        # Guardar correlaciones a CSV
        pearson_path = os.path.join(self.output_dir, f"{self.output_prefix}_correlations_pearson.csv")
        spearman_path = os.path.join(self.output_dir, f"{self.output_prefix}_correlations_spearman.csv")

        self.corr_pearson.to_csv(pearson_path)
        self.corr_spearman.to_csv(spearman_path)

        self.log(f"✓ Guardado: {pearson_path}")
        self.log(f"✓ Guardado: {spearman_path}")
        self.log("")

    def seccion_3_ranking_drivers(self):
        """Sección 3: Ranking de Drivers"""
        self.log("="*80)
        self.log("SECCIÓN 3: RANKING DE DRIVERS")
        self.log("="*80)
        self.log("")

        ranking_data = []
        for driver in self.drivers_available:
            corrs = [abs(self.corr_pearson.loc[driver, pnl]) for pnl in self.pnl_available
                     if not np.isnan(self.corr_pearson.loc[driver, pnl])]
            if corrs:
                avg_corr = np.mean(corrs)
                n_windows = len(corrs)
                ranking_data.append({
                    'Driver': driver,
                    'Correlación_Promedio_Abs': avg_corr,
                    'N_Ventanas': n_windows
                })

        self.ranking_df = pd.DataFrame(ranking_data)
        self.ranking_df = self.ranking_df.sort_values('Correlación_Promedio_Abs', ascending=False)
        self.ranking_df['Rank'] = range(1, len(self.ranking_df) + 1)
        self.ranking_df = self.ranking_df[['Rank', 'Driver', 'Correlación_Promedio_Abs', 'N_Ventanas']]

        self.log(self.ranking_df.to_string(index=False, float_format='%.4f'))
        self.log("")

        self.best_driver = self.ranking_df.iloc[0]['Driver']
        self.log(f"⭐ MEJOR DRIVER: {self.best_driver}")
        self.log("")

    def seccion_4_analisis_por_rangos(self):
        """Sección 4: Análisis por Rangos del Mejor Driver"""
        self.log("="*80)
        self.log(f"SECCIÓN 4: ANÁLISIS POR RANGOS - {self.best_driver}")
        self.log("="*80)
        self.log("")

        percentiles = [25, 50, 75, 90]

        for pct in percentiles:
            threshold = self.df_clean[self.best_driver].quantile(pct/100)

            above = self.df_clean[self.df_clean[self.best_driver] > threshold]
            below = self.df_clean[self.df_clean[self.best_driver] <= threshold]

            self.log(f"--- PERCENTIL {pct} ({self.best_driver} > {threshold:.4f}) ---")
            self.log(f"N encima: {len(above)}, N debajo: {len(below)}")
            self.log("")
            self.log(f"{'Ventana':<15} {'PnL_Encima':>12} {'PnL_Debajo':>12} {'Diferencial':>12} {'Ganador':>10}")
            self.log("-" * 65)

            for pnl in self.pnl_available:
                pnl_above = above[pnl].mean()
                pnl_below = below[pnl].mean()
                diff = pnl_above - pnl_below
                winner = "ENCIMA" if diff > 0 else "DEBAJO"

                pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
                self.log(f"{pnl_short:<15} {pnl_above:>12.2f} {pnl_below:>12.2f} {diff:>12.2f} {winner:>10}")

            self.log("")

    def seccion_5_analisis_por_cuartiles(self):
        """Sección 5: Análisis por Cuartiles"""
        self.log("="*80)
        self.log("SECCIÓN 5: ANÁLISIS POR CUARTILES")
        self.log("="*80)
        self.log("")

        for driver in self.drivers_available:
            self.log(f"--- {driver} ---")

            quartiles = pd.qcut(self.df_clean[driver], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            self.df_clean[f'{driver}_quartile'] = quartiles

            self.log(f"{'Cuartil':<10}", end="")
            for pnl in self.pnl_available:
                pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)[:10]
                self.log(f"{pnl_short:>12}", end="")
            self.log("")
            self.log("-" * (10 + 12 * len(self.pnl_available)))

            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                subset = self.df_clean[self.df_clean[f'{driver}_quartile'] == q]
                self.log(f"{q:<10}", end="")
                for pnl in self.pnl_available:
                    pnl_mean = subset[pnl].mean()
                    self.log(f"{pnl_mean:>12.2f}", end="")
                self.log(f"  (N={len(subset)})")

            self.log("")

    def seccion_6_top_bottom_10(self):
        """Sección 6: Top 10% vs Bottom 10%"""
        self.log("="*80)
        self.log("SECCIÓN 6: TOP 10% vs BOTTOM 10%")
        self.log("="*80)
        self.log("")

        self.log(f"{'Driver':<25} {'Ventana':<12} {'Top10%':>10} {'Bot10%':>10} {'Spread':>10} {'Tipo':>10}")
        self.log("-" * 80)

        for driver in self.drivers_available:
            top_threshold = self.df_clean[driver].quantile(0.90)
            bottom_threshold = self.df_clean[driver].quantile(0.10)

            top10 = self.df_clean[self.df_clean[driver] >= top_threshold]
            bottom10 = self.df_clean[self.df_clean[driver] <= bottom_threshold]

            for pnl in self.pnl_available:
                pnl_top = top10[pnl].mean()
                pnl_bottom = bottom10[pnl].mean()
                spread = pnl_top - pnl_bottom
                tipo = "POSITIVO" if spread > 0 else "INVERSO"

                pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
                self.log(f"{driver:<25} {pnl_short:<12} {pnl_top:>10.2f} {pnl_bottom:>10.2f} {spread:>10.2f} {tipo:>10}")

        self.log("")

    def seccion_7_escenarios_extremos(self):
        """Sección 7: Escenarios Extremos del Mejor Driver"""
        self.log("="*80)
        self.log(f"SECCIÓN 7: ESCENARIOS EXTREMOS - {self.best_driver}")
        self.log("="*80)
        self.log("")

        extreme_percentiles = [75, 85, 95]

        for pct in extreme_percentiles:
            threshold = self.df_clean[self.best_driver].quantile(pct/100)
            extreme = self.df_clean[self.df_clean[self.best_driver] >= threshold]

            self.log(f"--- PERCENTIL {pct} ({self.best_driver} >= {threshold:.4f}) ---")
            self.log(f"N trades: {len(extreme)}")
            self.log("")
            self.log(f"{'Ventana':<15} {'PnL_Medio':>12} {'Desv_Std':>12} {'Min':>10} {'Max':>10}")
            self.log("-" * 62)

            for pnl in self.pnl_available:
                pnl_mean = extreme[pnl].mean()
                pnl_std = extreme[pnl].std()
                pnl_min = extreme[pnl].min()
                pnl_max = extreme[pnl].max()

                pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
                self.log(f"{pnl_short:<15} {pnl_mean:>12.2f} {pnl_std:>12.2f} {pnl_min:>10.2f} {pnl_max:>10.2f}")

            self.log("")

    def seccion_8_recomendaciones_filtros(self):
        """Sección 8: Recomendaciones de Filtros"""
        self.log("="*80)
        self.log(f"SECCIÓN 8: RECOMENDACIONES DE FILTROS - {self.best_driver}")
        self.log("="*80)
        self.log("")

        # Conservador (P75)
        p75 = self.df_clean[self.best_driver].quantile(0.75)
        conservador = self.df_clean[self.df_clean[self.best_driver] >= p75]
        retention_p75 = (len(conservador) / len(self.df_clean)) * 100

        # Equilibrado (P90)
        p90 = self.df_clean[self.best_driver].quantile(0.90)
        equilibrado = self.df_clean[self.df_clean[self.best_driver] >= p90]
        retention_p90 = (len(equilibrado) / len(self.df_clean)) * 100

        # Agresivo (P95)
        p95 = self.df_clean[self.best_driver].quantile(0.95)
        agresivo = self.df_clean[self.df_clean[self.best_driver] >= p95]
        retention_p95 = (len(agresivo) / len(self.df_clean)) * 100

        self.log("--- FILTROS PROPUESTOS ---")
        self.log("")

        self.log(f"1. CONSERVADOR (P75): {self.best_driver} >= {p75:.4f}")
        self.log(f"   Retención: {retention_p75:.2f}% ({len(conservador)} trades)")
        self.log(f"   PnL esperado por ventana:")
        for pnl in self.pnl_available:
            pnl_mean = conservador[pnl].mean()
            pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
            self.log(f"      {pnl_short}: {pnl_mean:.2f} pts")
        self.log("")

        self.log(f"2. EQUILIBRADO (P90): {self.best_driver} >= {p90:.4f}")
        self.log(f"   Retención: {retention_p90:.2f}% ({len(equilibrado)} trades)")
        self.log(f"   PnL esperado por ventana:")
        for pnl in self.pnl_available:
            pnl_mean = equilibrado[pnl].mean()
            pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
            self.log(f"      {pnl_short}: {pnl_mean:.2f} pts")
        self.log("")

        self.log(f"3. AGRESIVO (P95): {self.best_driver} >= {p95:.4f}")
        self.log(f"   Retención: {retention_p95:.2f}% ({len(agresivo)} trades)")
        self.log(f"   PnL esperado por ventana:")
        for pnl in self.pnl_available:
            pnl_mean = agresivo[pnl].mean()
            pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
            self.log(f"      {pnl_short}: {pnl_mean:.2f} pts")
        self.log("")

        # Anti-filtros
        p25 = self.df_clean[self.best_driver].quantile(0.25)
        anti_filter = self.df_clean[self.df_clean[self.best_driver] <= p25]

        self.log("--- ANTI-FILTROS (A EVITAR) ---")
        self.log("")
        self.log(f"EVITAR: {self.best_driver} <= {p25:.4f}")
        self.log(f"Trades afectados: {len(anti_filter)} ({(len(anti_filter)/len(self.df_clean)*100):.2f}%)")
        self.log(f"PnL promedio por ventana:")
        for pnl in self.pnl_available:
            pnl_mean = anti_filter[pnl].mean()
            pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
            self.log(f"   {pnl_short}: {pnl_mean:.2f} pts")
        self.log("")

    def seccion_9_resumen_conclusiones(self):
        """Sección 9: Resumen y Conclusiones"""
        self.log("="*80)
        self.log("SECCIÓN 9: RESUMEN Y CONCLUSIONES")
        self.log("="*80)
        self.log("")

        self.log("--- RESUMEN EJECUTIVO ---")
        self.log("")
        self.log(f"Dataset: {self.output_prefix}")
        self.log(f"Total observaciones: {self.total_rows}")
        self.log(f"Observaciones válidas: {self.valid_rows} ({self.valid_pct:.2f}%)")
        self.log("")

        self.log("Top 3 Drivers (por correlación promedio):")
        for i in range(min(3, len(self.ranking_df))):
            row = self.ranking_df.iloc[i]
            self.log(f"  {i+1}. {row['Driver']}: {row['Correlación_Promedio_Abs']:.4f}")
        self.log("")

        self.log("--- HALLAZGOS PRINCIPALES ---")
        self.log("")
        self.log("Análisis de paradojas:")
        for driver in self.drivers_available[:3]:
            top_threshold = self.df_clean[driver].quantile(0.90)
            bottom_threshold = self.df_clean[driver].quantile(0.10)

            top10 = self.df_clean[self.df_clean[driver] >= top_threshold]
            bottom10 = self.df_clean[self.df_clean[driver] <= bottom_threshold]

            if self.pnl_available:
                first_pnl = self.pnl_available[0]
                pnl_top = top10[first_pnl].mean()
                pnl_bottom = bottom10[first_pnl].mean()

                if pnl_top < pnl_bottom:
                    self.log(f"⚠️  PARADOJA DETECTADA en {driver}:")
                    self.log(f"   Top 10% rinde PEOR que Bottom 10% ({pnl_top:.2f} vs {pnl_bottom:.2f} pts)")
                else:
                    self.log(f"✅ {driver}: Comportamiento esperado (Top > Bottom)")

        self.log("")

        self.log("--- RECOMENDACIONES FINALES ---")
        self.log("")

        p90 = self.df_clean[self.best_driver].quantile(0.90)
        p25 = self.df_clean[self.best_driver].quantile(0.25)

        self.log(f"✅ IMPLEMENTAR: Filtro basado en {self.best_driver}")
        self.log(f"   - Recomendado: Usar percentil 90 como umbral")
        self.log(f"   - Umbral sugerido: {self.best_driver} >= {p90:.4f}")
        self.log("")

        if len(self.ranking_df) >= 2:
            second_best = self.ranking_df.iloc[1]['Driver']
            self.log(f"⚠️  INVESTIGAR: Combinación de {self.best_driver} y {second_best}")
            self.log(f"   - Potencial para filtros multi-variable")
        self.log("")

        self.log(f"🚫 EVITAR: Trades con {self.best_driver} <= {p25:.4f}")
        self.log(f"   - Este rango muestra bajo rendimiento consistente")
        self.log("")

    def ejecutar_analisis_estadistico(self):
        """Ejecuta todas las secciones del análisis estadístico"""
        self.cargar_datos()
        self.seccion_1_estadisticas_descriptivas()
        self.seccion_2_correlaciones()
        self.seccion_3_ranking_drivers()
        self.seccion_4_analisis_por_rangos()
        self.seccion_5_analisis_por_cuartiles()
        self.seccion_6_top_bottom_10()
        self.seccion_7_escenarios_extremos()
        self.seccion_8_recomendaciones_filtros()
        self.seccion_9_resumen_conclusiones()

        self.log("="*80)
        self.log("ANÁLISIS ESTADÍSTICO COMPLETADO")
        self.log("="*80)
        self.log("")

        # Guardar reporte
        report_path = os.path.join(self.output_dir, f"{self.output_prefix}_analysis_results.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))

        self.log(f"✓ Reporte guardado: {report_path}", to_report=False)

    def generar_visualizaciones(self):
        """Genera las 6 visualizaciones profesionales"""

        # Configurar estilo
        sns.set_style(PLOT_STYLE)
        plt.rcParams['figure.dpi'] = PLOT_DPI
        plt.rcParams['savefig.dpi'] = PLOT_DPI
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12

        print("\n" + "="*80)
        print("GENERANDO VISUALIZACIONES")
        print("="*80)
        print()

        # Gráfico 1: Heatmap de Correlaciones
        self._generar_heatmap_correlaciones()

        # Gráfico 2: PnL por Cuartiles del Mejor Driver
        self._generar_pnl_por_cuartiles()

        # Gráfico 3: Scatter Plots
        self._generar_scatter_plots()

        # Gráfico 4: Ranking de Drivers
        self._generar_ranking_drivers()

        # Gráfico 5: PnL por Rangos
        self._generar_pnl_por_rangos()

        # Gráfico 6: Análisis Variable Especial
        self._generar_analisis_variable_especial()

        print("\n✓ Todas las visualizaciones generadas exitosamente")

    def _generar_heatmap_correlaciones(self):
        """Gráfico 1: Heatmap de Correlaciones"""
        print("Generando Gráfico 1: Heatmap de Correlaciones...")

        fig, ax = plt.subplots(figsize=(10, 6))

        corr_display = self.corr_pearson.copy()
        corr_display.columns = [PNL_SHORT_NAMES.get(c, c) for c in corr_display.columns]

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

        ax.set_title(f'Matriz de Correlación: Drivers vs PnL Forward Points\n{self.output_prefix}',
                     fontsize=13, fontweight='bold', pad=20)
        ax.set_xlabel('Ventanas PnL (días)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variables Driver', fontsize=11, fontweight='bold')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{self.output_prefix}_correlation_heatmap.png')
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        print(f"✓ Guardado: {output_path}")

    def _generar_pnl_por_cuartiles(self):
        """Gráfico 2: PnL por Cuartiles del Mejor Driver"""
        print(f"Generando Gráfico 2: PnL por Cuartiles de {self.best_driver}...")

        quartiles = pd.qcut(self.df_clean[self.best_driver], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        self.df_clean['quartile'] = quartiles

        quartile_means = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            quartile_means[q] = []
            subset = self.df_clean[self.df_clean['quartile'] == q]
            for pnl in self.pnl_available:
                quartile_means[q].append(subset[pnl].mean())

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(self.pnl_available))
        width = 0.2
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

        for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, quartile_means[q], width, label=q, color=colors[i], alpha=0.8)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Ventanas PnL', fontsize=11, fontweight='bold')
        ax.set_ylabel('PnL Medio (pts)', fontsize=11, fontweight='bold')
        ax.set_title(f'PnL por Cuartiles de {self.best_driver}\n{self.output_prefix}',
                     fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([PNL_SHORT_NAMES.get(p, p) for p in self.pnl_available])
        ax.legend(title='Cuartil', loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{self.output_prefix}_pnl_by_{self.best_driver.lower()}_quartiles.png')
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        print(f"✓ Guardado: {output_path}")

    def _generar_scatter_plots(self):
        """Gráfico 3: Scatter Plots - Mejor Driver vs PnL"""
        print(f"Generando Gráfico 3: Scatter Plots {self.best_driver} vs PnL...")

        n_pnl = len(self.pnl_available)
        if n_pnl <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
        axes = axes.flatten()

        for i, pnl in enumerate(self.pnl_available):
            ax = axes[i]

            x = self.df_clean[self.best_driver].values
            y = self.df_clean[pnl].values

            ax.scatter(x, y, alpha=0.4, s=20, color='steelblue', edgecolors='none')

            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
                x_line = np.array([x[mask].min(), x[mask].max()])
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'y={slope:.2f}x+{intercept:.2f}')

                corr, _ = calc_correlation_with_pvalue(x, y, method='pearson')
                ax.set_title(f'{PNL_SHORT_NAMES.get(pnl, pnl)} (r={corr:.3f})',
                             fontsize=11, fontweight='bold')

            ax.set_xlabel(self.best_driver, fontsize=10)
            ax.set_ylabel('PnL (pts)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.legend(fontsize=8)

        for i in range(n_pnl, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f'Análisis de Correlación: {self.best_driver} vs PnL Forward Points\n{self.output_prefix}',
                     fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{self.output_prefix}_scatter_{self.best_driver.lower()}_vs_pnl.png')
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        print(f"✓ Guardado: {output_path}")

    def _generar_ranking_drivers(self):
        """Gráfico 4: Ranking de Drivers"""
        print("Generando Gráfico 4: Ranking de Drivers...")

        fig, ax = plt.subplots(figsize=(10, 6))

        ranking_sorted = self.ranking_df.sort_values('Correlación_Promedio_Abs', ascending=True)

        colors_rank = ['darkgreen' if d == self.best_driver else 'steelblue'
                       for d in ranking_sorted['Driver']]

        bars = ax.barh(ranking_sorted['Driver'], ranking_sorted['Correlación_Promedio_Abs'],
                       color=colors_rank, alpha=0.8)

        for bar, val in zip(bars, ranking_sorted['Correlación_Promedio_Abs']):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}',
                    ha='left', va='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Correlación Promedio Absoluta', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variable Driver', fontsize=11, fontweight='bold')
        ax.set_title(f'Ranking de Drivers por Poder Predictivo\n{self.output_prefix}',
                     fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{self.output_prefix}_driver_rankings.png')
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        print(f"✓ Guardado: {output_path}")

    def _generar_pnl_por_rangos(self):
        """Gráfico 5: PnL por Rangos del Mejor Driver"""
        print(f"Generando Gráfico 5: PnL por Rangos de {self.best_driver}...")

        bins = self.df_clean[self.best_driver].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
        bin_labels = ['Muy Bajo\n(0-20%)', 'Bajo\n(20-40%)', 'Medio\n(40-60%)',
                      'Alto\n(60-80%)', 'Muy Alto\n(80-100%)']

        self.df_clean['range_bin'] = pd.cut(self.df_clean[self.best_driver], bins=bins, labels=bin_labels, include_lowest=True)

        range_means = {}
        for pnl in self.pnl_available:
            range_means[pnl] = []
            for bin_label in bin_labels:
                subset = self.df_clean[self.df_clean['range_bin'] == bin_label]
                range_means[pnl].append(subset[pnl].mean())

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(bin_labels))
        n_pnl = len(self.pnl_available)
        width = 0.15
        colors_pnl = plt.cm.viridis(np.linspace(0.2, 0.9, n_pnl))

        for i, pnl in enumerate(self.pnl_available):
            offset = (i - (n_pnl-1)/2) * width
            bars = ax.bar(x + offset, range_means[pnl], width,
                           label=PNL_SHORT_NAMES.get(pnl, pnl),
                           color=colors_pnl[i], alpha=0.8)

            for bar in bars:
                height = bar.get_height()
                if abs(height) > 5:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.0f}',
                            ha='center', va='bottom' if height > 0 else 'top',
                            fontsize=7)

        ax.set_xlabel(f'Rango de {self.best_driver}', fontsize=11, fontweight='bold')
        ax.set_ylabel('PnL Medio (pts)', fontsize=11, fontweight='bold')
        ax.set_title(f'PnL por Rangos de {self.best_driver}\n{self.output_prefix}',
                     fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.legend(title='Ventana PnL', loc='best', ncol=n_pnl)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{self.output_prefix}_pnl_by_{self.best_driver.lower()}_ranges.png')
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        print(f"✓ Guardado: {output_path}")

    def _generar_analisis_variable_especial(self):
        """Gráfico 6: Análisis Variable Especial"""
        print("Generando Gráfico 6: Análisis Variable Especial...")

        if 'LABEL_GENERAL_SCORE' in self.drivers_available:
            # Análisis de LABEL_GENERAL_SCORE
            fig, ax = plt.subplots(figsize=(12, 7))

            categories = sorted(self.df_clean['LABEL_GENERAL_SCORE'].unique())

            pnl_by_category = {}
            for pnl in self.pnl_available:
                pnl_by_category[pnl] = []
                for cat in categories:
                    subset = self.df_clean[self.df_clean['LABEL_GENERAL_SCORE'] == cat]
                    pnl_by_category[pnl].append(subset[pnl].mean())

            for pnl in self.pnl_available:
                ax.plot(categories, pnl_by_category[pnl],
                        marker='o', linewidth=2, markersize=8,
                        label=PNL_SHORT_NAMES.get(pnl, pnl))

            ax.set_xlabel('LABEL_GENERAL_SCORE', fontsize=11, fontweight='bold')
            ax.set_ylabel('PnL Medio (pts)', fontsize=11, fontweight='bold')
            ax.set_title(f'PnL por Categorías de LABEL_GENERAL_SCORE\n{self.output_prefix}',
                         fontsize=13, fontweight='bold', pad=20)
            ax.legend(title='Ventana PnL', loc='best')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

            first_cat_pnl = pnl_by_category[self.pnl_available[0]][0]
            last_cat_pnl = pnl_by_category[self.pnl_available[0]][-1]

            if first_cat_pnl > last_cat_pnl:
                ax.text(0.5, 0.95, '⚠️ CORRELACIÓN INVERSA DETECTADA',
                        transform=ax.transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        fontsize=11, fontweight='bold')

            plt.tight_layout()
            output_path = os.path.join(self.output_dir, f'{self.output_prefix}_pnl_by_label_general_score_analysis.png')
            plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()

            print(f"✓ Guardado: {output_path}")

        else:
            # Análisis del segundo mejor driver
            if len(self.ranking_df) >= 2:
                second_best = self.ranking_df.iloc[1]['Driver']

                fig, ax = plt.subplots(figsize=(12, 7))

                deciles = pd.qcut(self.df_clean[second_best], q=10, labels=range(1, 11), duplicates='drop')
                self.df_clean['decile'] = deciles

                decile_means = {}
                for pnl in self.pnl_available:
                    decile_means[pnl] = []
                    for dec in range(1, 11):
                        subset = self.df_clean[self.df_clean['decile'] == dec]
                        if len(subset) > 0:
                            decile_means[pnl].append(subset[pnl].mean())
                        else:
                            decile_means[pnl].append(np.nan)

                x_deciles = list(range(1, 11))
                for pnl in self.pnl_available:
                    ax.plot(x_deciles, decile_means[pnl],
                            marker='o', linewidth=2, markersize=6,
                            label=PNL_SHORT_NAMES.get(pnl, pnl))

                ax.set_xlabel(f'{second_best} (Deciles)', fontsize=11, fontweight='bold')
                ax.set_ylabel('PnL Medio (pts)', fontsize=11, fontweight='bold')
                ax.set_title(f'PnL por Deciles de {second_best}\n{self.output_prefix}',
                             fontsize=13, fontweight='bold', pad=20)
                ax.set_xticks(x_deciles)
                ax.legend(title='Ventana PnL', loc='best')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

                plt.tight_layout()
                output_path = os.path.join(self.output_dir, f'{self.output_prefix}_pnl_by_{second_best.lower()}_analysis.png')
                plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()

                print(f"✓ Guardado: {output_path}")

    def generar_resumen_ejecutivo_markdown(self):
        """Genera el resumen ejecutivo en formato Markdown"""

        print("\n" + "="*80)
        print("GENERANDO RESUMEN EJECUTIVO EN MARKDOWN")
        print("="*80)
        print()

        md_lines = []

        # Encabezado
        md_lines.append(f"# RESUMEN EJECUTIVO: Análisis de Correlaciones PnL vs Drivers")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        md_lines.append(f"**📅 Fecha:** {datetime.now().strftime('%Y-%m-%d')}")
        md_lines.append(f"**📊 Dataset:** {os.path.basename(self.csv_path)}")
        md_lines.append(f"**📈 Observaciones Válidas:** {self.valid_rows:,} de {self.total_rows:,} ({self.valid_pct:.2f}%)")
        md_lines.append("**🎯 Objetivo:** Identificar variables driver con mayor poder predictivo sobre PnL Forward Points")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # Hallazgos Principales
        md_lines.append("## 🏆 HALLAZGOS PRINCIPALES")
        md_lines.append("")
        md_lines.append(f"**Mejor Driver Identificado:** {self.best_driver}")
        md_lines.append(f"**Correlación Promedio Absoluta:** {self.ranking_df.iloc[0]['Correlación_Promedio_Abs']:.4f}")
        md_lines.append("")

        # Ranking de Drivers
        md_lines.append("### 📊 Ranking de Drivers")
        md_lines.append("")
        md_lines.append("| Rank | Driver | Correlación Promedio | Evaluación |")
        md_lines.append("|------|--------|----------------------|------------|")

        for idx, row in self.ranking_df.iterrows():
            rank = row['Rank']
            driver = row['Driver']
            corr = row['Correlación_Promedio_Abs']

            if rank == 1:
                eval_text = "⭐⭐⭐ EXCELENTE"
            elif rank <= 3:
                eval_text = "✅ BUENO"
            else:
                eval_text = "⚠️ MODERADO"

            md_lines.append(f"| {rank} | **{driver}** | {corr:.4f} | {eval_text} |")

        md_lines.append("")

        # Correlaciones del Mejor Driver
        md_lines.append(f"### 🎯 Correlaciones Detalladas: {self.best_driver}")
        md_lines.append("")
        md_lines.append("| Ventana PnL | Correlación Pearson | P-value | Significancia |")
        md_lines.append("|-------------|---------------------|---------|---------------|")

        for pnl in self.pnl_available:
            corr = self.corr_pearson.loc[self.best_driver, pnl]
            pval = self.pval_pearson.loc[self.best_driver, pnl]

            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
            else:
                sig = "ns"

            pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
            md_lines.append(f"| {pnl_short} | {corr:.4f} | {pval:.4f} | {sig} |")

        md_lines.append("")
        md_lines.append("> **Nota:** *** p<0.001, ** p<0.01, * p<0.05, ns = no significativo")
        md_lines.append("")

        # Recomendaciones de Filtros
        md_lines.append("## ✅ RECOMENDACIONES DE FILTROS")
        md_lines.append("")

        p75 = self.df_clean[self.best_driver].quantile(0.75)
        p90 = self.df_clean[self.best_driver].quantile(0.90)
        p95 = self.df_clean[self.best_driver].quantile(0.95)

        conservador = self.df_clean[self.df_clean[self.best_driver] >= p75]
        equilibrado = self.df_clean[self.df_clean[self.best_driver] >= p90]
        agresivo = self.df_clean[self.df_clean[self.best_driver] >= p95]

        md_lines.append("### 1️⃣ CONSERVADOR (Percentil 75)")
        md_lines.append("")
        md_lines.append(f"```")
        md_lines.append(f"CONDICIÓN: {self.best_driver} >= {p75:.4f}")
        md_lines.append(f"```")
        md_lines.append("")
        md_lines.append(f"- **Retención:** {(len(conservador)/len(self.df_clean)*100):.2f}% ({len(conservador)} trades)")
        md_lines.append("- **PnL Esperado:**")
        for pnl in self.pnl_available:
            pnl_mean = conservador[pnl].mean()
            pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
            md_lines.append(f"  - {pnl_short}: {pnl_mean:.2f} pts")
        md_lines.append("")

        md_lines.append("### 2️⃣ EQUILIBRADO (Percentil 90) ⭐ **RECOMENDADO**")
        md_lines.append("")
        md_lines.append(f"```")
        md_lines.append(f"CONDICIÓN: {self.best_driver} >= {p90:.4f}")
        md_lines.append(f"```")
        md_lines.append("")
        md_lines.append(f"- **Retención:** {(len(equilibrado)/len(self.df_clean)*100):.2f}% ({len(equilibrado)} trades)")
        md_lines.append("- **PnL Esperado:**")
        for pnl in self.pnl_available:
            pnl_mean = equilibrado[pnl].mean()
            pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
            md_lines.append(f"  - {pnl_short}: {pnl_mean:.2f} pts")
        md_lines.append("")

        md_lines.append("### 3️⃣ AGRESIVO (Percentil 95)")
        md_lines.append("")
        md_lines.append(f"```")
        md_lines.append(f"CONDICIÓN: {self.best_driver} >= {p95:.4f}")
        md_lines.append(f"```")
        md_lines.append("")
        md_lines.append(f"- **Retención:** {(len(agresivo)/len(self.df_clean)*100):.2f}% ({len(agresivo)} trades)")
        md_lines.append("- **PnL Esperado:**")
        for pnl in self.pnl_available:
            pnl_mean = agresivo[pnl].mean()
            pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
            md_lines.append(f"  - {pnl_short}: {pnl_mean:.2f} pts")
        md_lines.append("")

        # Anti-filtros
        p25 = self.df_clean[self.best_driver].quantile(0.25)
        anti_filter = self.df_clean[self.df_clean[self.best_driver] <= p25]

        md_lines.append("### 🚫 ANTI-FILTRO (A EVITAR)")
        md_lines.append("")
        md_lines.append(f"```")
        md_lines.append(f"EVITAR: {self.best_driver} <= {p25:.4f}")
        md_lines.append(f"```")
        md_lines.append("")
        md_lines.append(f"- **Trades Afectados:** {len(anti_filter)} ({(len(anti_filter)/len(self.df_clean)*100):.2f}%)")
        md_lines.append("- **PnL Promedio (bajo rendimiento):**")
        for pnl in self.pnl_available:
            pnl_mean = anti_filter[pnl].mean()
            pnl_short = PNL_SHORT_NAMES.get(pnl, pnl)
            md_lines.append(f"  - {pnl_short}: {pnl_mean:.2f} pts")
        md_lines.append("")

        # Archivos Generados
        md_lines.append("## 📁 ARCHIVOS GENERADOS")
        md_lines.append("")
        md_lines.append("### Scripts y Datos:")
        md_lines.append(f"- `{self.output_prefix}_analysis_results.txt` - Reporte completo")
        md_lines.append(f"- `{self.output_prefix}_correlations_pearson.csv` - Correlaciones Pearson")
        md_lines.append(f"- `{self.output_prefix}_correlations_spearman.csv` - Correlaciones Spearman")
        md_lines.append("")
        md_lines.append("### Visualizaciones (PNG 300 DPI):")
        md_lines.append(f"1. `{self.output_prefix}_correlation_heatmap.png`")
        md_lines.append(f"2. `{self.output_prefix}_pnl_by_{self.best_driver.lower()}_quartiles.png`")
        md_lines.append(f"3. `{self.output_prefix}_scatter_{self.best_driver.lower()}_vs_pnl.png`")
        md_lines.append(f"4. `{self.output_prefix}_driver_rankings.png`")
        md_lines.append(f"5. `{self.output_prefix}_pnl_by_{self.best_driver.lower()}_ranges.png`")
        md_lines.append(f"6. `{self.output_prefix}_pnl_by_*_analysis.png`")
        md_lines.append("")

        # Footer
        md_lines.append("---")
        md_lines.append("")
        md_lines.append(f"**Generado por:** ANALISIS_SCRIPT.py")
        md_lines.append(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("")

        # Guardar archivo
        md_path = os.path.join(self.output_dir, f"RESUMEN_EJECUTIVO_{self.output_prefix}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))

        print(f"✓ Resumen ejecutivo guardado: {md_path}")

    def ejecutar_completo(self):
        """Ejecuta el análisis completo: estadísticas + visualizaciones + resumen"""
        print("\n" + "="*80)
        print("INICIANDO ANÁLISIS COMPLETO")
        print("="*80)
        print()

        # Ejecutar análisis estadístico
        self.ejecutar_analisis_estadistico()

        # Generar visualizaciones
        self.generar_visualizaciones()

        # Generar resumen ejecutivo
        self.generar_resumen_ejecutivo_markdown()

        print("\n" + "="*80)
        print("✓ ANÁLISIS COMPLETO FINALIZADO")
        print("="*80)
        print()
        print(f"Todos los archivos generados en: {self.output_dir}/")
        print()

# ================================================================================
# EJECUCIÓN PRINCIPAL
# ================================================================================

if __name__ == "__main__":

    print("\n" + "="*80)
    print("ANÁLISIS ESTADÍSTICO COMPLETO: PnL vs DRIVERS")
    print("Sistema BatmanMT - Versión 1.0")
    print("="*80)
    print()

    # Crear instancia del analizador
    analizador = AnalisisPnLDrivers(
        csv_path=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        output_prefix=OUTPUT_PREFIX,
        drivers=DRIVERS,
        pnl_vars=PNL_VARS
    )

    # Ejecutar análisis completo
    try:
        analizador.ejecutar_completo()

        print("\n✓✓✓ ANÁLISIS COMPLETADO EXITOSAMENTE ✓✓✓")
        print()
        print(f"📂 Revisa los resultados en: {OUTPUT_DIR}/")
        print()

    except Exception as e:
        print(f"\n❌ ERROR durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()
