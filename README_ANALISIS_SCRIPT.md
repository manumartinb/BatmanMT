# 📊 ANALISIS_SCRIPT.py - Guía de Uso

Script unificado para análisis estadístico completo de correlaciones PnL vs Drivers en estrategias de trading de opciones Batman.

## 🚀 Inicio Rápido

### Instalación de Dependencias

```bash
pip install pandas numpy scipy matplotlib seaborn
```

### Ejecución Básica

```bash
python ANALISIS_SCRIPT.py
```

El script generará automáticamente:
- **3 archivos de datos** (TXT, 2x CSV)
- **6 visualizaciones profesionales** (PNG 300 DPI)
- **1 resumen ejecutivo** (Markdown)

Todos los archivos se guardarán en la carpeta `output_analisis/`

---

## ⚙️ Configuración Personalizada

Abre `ANALISIS_SCRIPT.py` y modifica las siguientes variables en la sección **CONFIGURACIÓN** (líneas 28-68):

### 1. Archivo de Entrada

```python
# Ruta a tu archivo CSV
INPUT_CSV = "MIDTERM_combined_mediana_labeled.csv"

# También puedes usar ruta absoluta:
# INPUT_CSV = "/ruta/completa/al/archivo/mi_dataset.csv"
```

### 2. Carpeta de Salida

```python
# Carpeta donde se guardarán todos los resultados
OUTPUT_DIR = "output_analisis"

# Ejemplo con subcarpetas por fecha:
# OUTPUT_DIR = "analisis_2025_11_29"
```

### 3. Prefijo de Archivos

```python
# Si es None, usa el nombre del CSV
OUTPUT_PREFIX = None

# Para personalizar el nombre de los archivos:
# OUTPUT_PREFIX = "MIDTERM_analisis"
```

### 4. Variables Driver a Analizar

```python
DRIVERS = [
    'LABEL_GENERAL_SCORE',  # Sistema de scoring
    'BQI_ABS',              # Body Quality Index
    'FF_ATM',               # Forward Factor ATM
    'delta_total',          # Delta total
    'theta_total',          # Theta total
    'FF_BAT'                # Forward Factor Batman
]

# Puedes añadir o quitar variables según tu dataset
```

### 5. Variables PnL a Analizar

```python
PNL_VARS = [
    'PnL_fwd_pts_01_mediana',  # Ventana 1 día
    'PnL_fwd_pts_05_mediana',  # Ventana 5 días
    'PnL_fwd_pts_25_mediana',  # Ventana 25 días
    'PnL_fwd_pts_50_mediana',  # Ventana 50 días
    'PnL_fwd_pts_90_mediana'   # Ventana 90 días
]

# Modifica según las columnas disponibles en tu CSV
```

### 6. Configuración de Gráficos

```python
PLOT_DPI = 300           # Resolución de gráficos (300 = alta calidad)
PLOT_STYLE = "whitegrid" # Estilo: whitegrid, darkgrid, white, dark, ticks
```

---

## 📁 Estructura de Archivos Generados

Después de ejecutar el script, encontrarás en `output_analisis/`:

### Archivos de Datos (3)

```
📄 MIDTERM_combined_mediana_labeled_analysis_results.txt
   └─ Reporte completo con las 9 secciones de análisis

📊 MIDTERM_combined_mediana_labeled_correlations_pearson.csv
   └─ Matriz de correlaciones de Pearson (drivers × ventanas PnL)

📊 MIDTERM_combined_mediana_labeled_correlations_spearman.csv
   └─ Matriz de correlaciones de Spearman (drivers × ventanas PnL)
```

### Visualizaciones PNG 300 DPI (6)

```
🖼️ MIDTERM_combined_mediana_labeled_correlation_heatmap.png
   └─ Heatmap de correlaciones con escala de color

🖼️ MIDTERM_combined_mediana_labeled_pnl_by_ff_atm_quartiles.png
   └─ PnL medio por cuartiles del mejor driver

🖼️ MIDTERM_combined_mediana_labeled_scatter_ff_atm_vs_pnl.png
   └─ 5 scatter plots con líneas de tendencia

🖼️ MIDTERM_combined_mediana_labeled_driver_rankings.png
   └─ Ranking visual de drivers por poder predictivo

🖼️ MIDTERM_combined_mediana_labeled_pnl_by_ff_atm_ranges.png
   └─ PnL por rangos del mejor driver (5 bins)

🖼️ MIDTERM_combined_mediana_labeled_pnl_by_label_general_score_analysis.png
   └─ Análisis de variable especial (LABEL o segundo mejor driver)
```

### Resumen Ejecutivo (1)

```
📝 RESUMEN_EJECUTIVO_MIDTERM_combined_mediana_labeled.md
   └─ Documento Markdown con hallazgos, recomendaciones y conclusiones
```

---

## 🔍 Análisis Realizados (9 Secciones)

El script ejecuta automáticamente:

### 1️⃣ Estadísticas Descriptivas
- Distribución de drivers (count, mean, std, min, 25%, 50%, 75%, max)
- Distribución de variables PnL

### 2️⃣ Correlaciones con PnL
- Matriz de Pearson (correlación lineal)
- Matriz de Spearman (correlación monotónica)
- P-values de significancia estadística

### 3️⃣ Ranking de Drivers
- Correlación promedio absoluta por driver
- Identificación del mejor predictor

### 4️⃣ Análisis por Rangos
- Percentiles 25, 50, 75, 90 del mejor driver
- PnL medio "por encima" vs "por debajo" del umbral

### 5️⃣ Análisis por Cuartiles
- División en Q1, Q2, Q3, Q4 para TODOS los drivers
- PnL medio por cuartil en cada ventana temporal

### 6️⃣ Top 10% vs Bottom 10%
- Comparación de extremos para cada driver
- Identificación de spreads positivos/inversos

### 7️⃣ Escenarios Extremos
- Percentiles 75, 85, 95 del mejor driver
- PnL medio ± desviación estándar

### 8️⃣ Recomendaciones de Filtros
- Filtro Conservador (P75)
- Filtro Equilibrado (P90) ⭐ Recomendado
- Filtro Agresivo (P95)
- Anti-filtros (umbrales a evitar)

### 9️⃣ Resumen y Conclusiones
- Top 3 drivers
- Detección de paradojas
- Recomendaciones finales

---

## 💻 Ejecución en VSCode

### Opción 1: Terminal Integrado

1. Abre VSCode
2. Abre la carpeta del proyecto (`/home/user/BatmanMT`)
3. Abre el terminal integrado (`` Ctrl+` `` o `View > Terminal`)
4. Ejecuta:

```bash
python ANALISIS_SCRIPT.py
```

### Opción 2: Ejecutar con F5 (Debug)

1. Abre `ANALISIS_SCRIPT.py` en VSCode
2. Presiona `F5` o `Run > Start Debugging`
3. Selecciona "Python File"

### Opción 3: Click Derecho

1. Abre `ANALISIS_SCRIPT.py` en VSCode
2. Click derecho en el editor
3. Selecciona "Run Python File in Terminal"

---

## 📊 Ejemplo de Uso con Otro Dataset

### Caso: Analizar dataset "LONGTERM_combined_mediana.csv"

1. Abre `ANALISIS_SCRIPT.py`

2. Modifica la configuración:

```python
# Cambiar archivo de entrada
INPUT_CSV = "LONGTERM_combined_mediana.csv"

# Cambiar carpeta de salida
OUTPUT_DIR = "output_longterm"

# (Opcional) Personalizar prefijo
OUTPUT_PREFIX = "LONGTERM"
```

3. Si tu dataset NO tiene la columna `LABEL_GENERAL_SCORE`, quítala:

```python
DRIVERS = [
    # 'LABEL_GENERAL_SCORE',  # <-- Comentar si no existe
    'BQI_ABS',
    'FF_ATM',
    'delta_total',
    'theta_total',
    'FF_BAT'
]
```

4. Ejecuta:

```bash
python ANALISIS_SCRIPT.py
```

5. Los resultados estarán en `output_longterm/`

---

## 🛠️ Solución de Problemas

### Error: "No module named 'pandas'"

```bash
pip install pandas numpy scipy matplotlib seaborn
```

### Error: "FileNotFoundError: [Errno 2] No such file or directory"

- Verifica que `INPUT_CSV` apunte al archivo correcto
- Usa ruta absoluta si el archivo está en otra ubicación

```python
INPUT_CSV = "/ruta/completa/al/archivo.csv"
```

### Error: "KeyError: 'LABEL_GENERAL_SCORE'"

- Tu dataset no tiene esa columna
- Edita `DRIVERS` y quita las variables que no existan en tu CSV

### Warning: "duplicates='drop' in pd.qcut"

- Es normal si hay muchos valores repetidos
- El script continúa automáticamente

### Los gráficos no se ven bien

- Modifica `PLOT_DPI`:

```python
PLOT_DPI = 150  # Para previsualización rápida
PLOT_DPI = 300  # Para calidad de publicación (default)
PLOT_DPI = 600  # Para impresión de alta calidad
```

---

## 🎯 Interpretación de Resultados

### Archivo Principal a Revisar

📝 **`RESUMEN_EJECUTIVO_*.md`** - Empieza por aquí

Este archivo contiene:
- ⭐ Mejor driver identificado
- 📊 Ranking completo de drivers
- 🎯 Correlaciones detalladas
- ✅ Filtros recomendados (Conservador, Equilibrado, Agresivo)
- 🚫 Anti-filtros (qué evitar)

### Métricas Clave

#### Correlación Promedio Absoluta
- **> 0.15**: Excelente predictor ⭐⭐⭐
- **0.10 - 0.15**: Buen predictor ⭐⭐
- **0.05 - 0.10**: Predictor moderado ⭐
- **< 0.05**: Predictor débil ⚠️

#### P-values de Significancia
- **p < 0.001**: Altamente significativo ***
- **p < 0.01**: Muy significativo **
- **p < 0.05**: Significativo *
- **p >= 0.05**: No significativo

#### Spread Top 10% vs Bottom 10%
- **Spread > 0**: Correlación positiva ✅ (esperado)
- **Spread < 0**: Correlación inversa ⚠️ (investigar)
- **|Spread| > 50 pts**: Muy fuerte diferenciación

---

## 📈 Ejemplo de Salida del Script

```
================================================================================
INICIANDO ANÁLISIS COMPLETO
================================================================================

Cargando datos desde: MIDTERM_combined_mediana_labeled.csv
Total de filas cargadas: 2609
Drivers disponibles: 6/6
Variables PnL disponibles: 5/5

Limpiando datos...
Filas válidas después de limpieza: 2214
Porcentaje válido: 84.86%

================================================================================
SECCIÓN 1: ESTADÍSTICAS DESCRIPTIVAS
================================================================================

--- DRIVERS ---
       LABEL_GENERAL_SCORE      BQI_ABS       FF_ATM  delta_total  theta_total
count           2214.00000  2214.000000  2214.000000  2214.000000  2214.000000
mean               0.01770    39.440764     0.124490     0.072667    -0.124836
...

⭐ MEJOR DRIVER: FF_ATM

...

✓ Reporte guardado: output_analisis/MIDTERM_combined_mediana_labeled_analysis_results.txt
✓ Guardado: output_analisis/MIDTERM_combined_mediana_labeled_correlations_pearson.csv
✓ Guardado: output_analisis/MIDTERM_combined_mediana_labeled_correlations_spearman.csv

================================================================================
GENERANDO VISUALIZACIONES
================================================================================

Generando Gráfico 1: Heatmap de Correlaciones...
✓ Guardado: output_analisis/MIDTERM_combined_mediana_labeled_correlation_heatmap.png

...

✓✓✓ ANÁLISIS COMPLETADO EXITOSAMENTE ✓✓✓

📂 Revisa los resultados en: output_analisis/
```

---

## 🔧 Personalización Avanzada

### Modificar Nombres Cortos de Ventanas

```python
PNL_SHORT_NAMES = {
    'PnL_fwd_pts_01_mediana': '1D',   # Más corto
    'PnL_fwd_pts_05_mediana': '1W',   # Semana
    'PnL_fwd_pts_25_mediana': '1M',   # Mes
    'PnL_fwd_pts_50_mediana': '2M',   # 2 Meses
    'PnL_fwd_pts_90_mediana': '3M'    # 3 Meses
}
```

### Cambiar Percentiles de Filtros

Edita el método `seccion_8_recomendaciones_filtros()`:

```python
# En lugar de P75, P90, P95, usar P80, P90, P97
p75 = self.df_clean[self.best_driver].quantile(0.80)  # Cambiar 0.75 a 0.80
p90 = self.df_clean[self.best_driver].quantile(0.90)  # Mantener
p95 = self.df_clean[self.best_driver].quantile(0.97)  # Cambiar 0.95 a 0.97
```

### Añadir Nuevas Variables Driver

Si tu CSV tiene columnas adicionales:

```python
DRIVERS = [
    'LABEL_GENERAL_SCORE',
    'BQI_ABS',
    'FF_ATM',
    'delta_total',
    'theta_total',
    'FF_BAT',
    'gamma_total',     # Nueva variable
    'vega_total',      # Nueva variable
    'IV_skew'          # Nueva variable
]
```

---

## 📚 Estructura del Código

```
ANALISIS_SCRIPT.py
│
├── CONFIGURACIÓN (líneas 28-68)
│   ├── INPUT_CSV
│   ├── OUTPUT_DIR
│   ├── DRIVERS
│   ├── PNL_VARS
│   └── PLOT_DPI
│
├── FUNCIONES AUXILIARES (líneas 74-128)
│   ├── setup_output_directory()
│   ├── clean_data()
│   ├── calc_correlation_with_pvalue()
│   └── format_pvalue()
│
├── CLASE AnalisisPnLDrivers (líneas 134-1120)
│   ├── __init__()
│   ├── cargar_datos()
│   ├── seccion_1_estadisticas_descriptivas()
│   ├── seccion_2_correlaciones()
│   ├── seccion_3_ranking_drivers()
│   ├── seccion_4_analisis_por_rangos()
│   ├── seccion_5_analisis_por_cuartiles()
│   ├── seccion_6_top_bottom_10()
│   ├── seccion_7_escenarios_extremos()
│   ├── seccion_8_recomendaciones_filtros()
│   ├── seccion_9_resumen_conclusiones()
│   ├── ejecutar_analisis_estadistico()
│   ├── generar_visualizaciones()
│   │   ├── _generar_heatmap_correlaciones()
│   │   ├── _generar_pnl_por_cuartiles()
│   │   ├── _generar_scatter_plots()
│   │   ├── _generar_ranking_drivers()
│   │   ├── _generar_pnl_por_rangos()
│   │   └── _generar_analisis_variable_especial()
│   ├── generar_resumen_ejecutivo_markdown()
│   └── ejecutar_completo()
│
└── EJECUCIÓN PRINCIPAL (líneas 1126-1154)
    └── if __name__ == "__main__":
```

---

## 🤝 Soporte

### Logs y Debugging

Si necesitas ver más detalles durante la ejecución:

```python
# Añadir al inicio del script, después de imports
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Verificar Versiones de Librerías

```bash
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
python -c "import seaborn; print('seaborn:', seaborn.__version__)"
python -c "import scipy; print('scipy:', scipy.__version__)"
```

### Versiones Recomendadas

```
pandas >= 1.3.0
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

---

## 📄 Licencia y Créditos

**Script:** ANALISIS_SCRIPT.py
**Sistema:** BatmanMT
**Versión:** 1.0
**Fecha:** 2025-11-29

---

## 🎉 ¡Listo para Usar!

Simplemente ejecuta:

```bash
python ANALISIS_SCRIPT.py
```

Y obtén un análisis estadístico completo y profesional en minutos. 🚀
