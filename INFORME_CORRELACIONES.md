# INFORME ESTADÍSTICO: ANÁLISIS DE CORRELACIONES
## Correlación entre Ventas (PNL_FWD_PTS) y Variables Objetivo

---

## 1. RESUMEN EJECUTIVO

- **Total de registros analizados:** 2,214
- **Registros eliminados (NaN):** 395 (15.14%)
- **Variables de ventas analizadas:** 5
- **Variables objetivo analizadas:** 6

### Correlaciones más Fuertes (Pearson |r| > 0.3)

No se encontraron correlaciones fuertes (|r| > 0.3)

---

## 2. ESTADÍSTICAS DESCRIPTIVAS

### Variables de Ventas (PNL_FWD_PTS)

| Variable | Media | Mediana | Desv. Est. | Min | Max |
|----------|-------|---------|------------|-----|-----|
| PnL_fwd_pts_01_mediana | 0.52 | -0.15 | 4.92 | -11.72 | 34.50 |
| PnL_fwd_pts_05_mediana | 1.36 | -0.10 | 9.32 | -59.42 | 48.60 |
| PnL_fwd_pts_25_mediana | 6.58 | 3.59 | 20.25 | -69.17 | 155.25 |
| PnL_fwd_pts_50_mediana | 15.76 | 13.43 | 30.73 | -52.08 | 187.20 |
| PnL_fwd_pts_90_mediana | 19.77 | 18.29 | 49.40 | -110.90 | 264.25 |

### Variables Objetivo

| Variable | Media | Mediana | Desv. Est. | Min | Max |
|----------|-------|---------|------------|-----|-----|
| LABEL_GENERAL_SCORE | 0.02 | 0.00 | 0.53 | -1.88 | 2.75 |
| BQI_ABS | 39.44 | 1.32 | 188.97 | 0.48 | 1009.50 |
| FF_ATM | 0.12 | 0.10 | 0.12 | -0.13 | 0.85 |
| delta_total | 0.07 | 0.08 | 0.03 | 0.00 | 0.13 |
| theta_total | -0.12 | -0.15 | 0.17 | -0.53 | 0.86 |
| FF_BAT | 0.98 | 0.60 | 2.00 | 0.09 | 42.13 |

---

## 3. ANÁLISIS DE CORRELACIÓN POR VARIABLE OBJETIVO

### LABEL_GENERAL_SCORE

| PNL Variable | Pearson (r) | p-valor | Spearman (ρ) | p-valor | Interpretación |
|--------------|-------------|---------|--------------|---------|----------------|
| PnL_fwd_pts_01_mediana | 0.0071 ns | 7.3929e-01 | 0.0190 ns | 3.7181e-01 | Muy débil |
| PnL_fwd_pts_05_mediana | 0.0233 ns | 2.7212e-01 | 0.0225 ns | 2.9099e-01 | Muy débil |
| PnL_fwd_pts_25_mediana | 0.0159 ns | 4.5537e-01 | 0.0277 ns | 1.9232e-01 | Muy débil |
| PnL_fwd_pts_50_mediana | 0.0702 *** | 9.5274e-04 | 0.0767 *** | 3.0511e-04 | Muy débil |
| PnL_fwd_pts_90_mediana | 0.1147 *** | 6.2123e-08 | 0.1086 *** | 3.0269e-07 | Débil |

### BQI_ABS

| PNL Variable | Pearson (r) | p-valor | Spearman (ρ) | p-valor | Interpretación |
|--------------|-------------|---------|--------------|---------|----------------|
| PnL_fwd_pts_01_mediana | 0.0325 ns | 1.2586e-01 | 0.0619 ** | 3.5579e-03 | Muy débil |
| PnL_fwd_pts_05_mediana | -0.0374 ns | 7.8154e-02 | 0.0192 ns | 3.6741e-01 | Muy débil |
| PnL_fwd_pts_25_mediana | 0.0356 ns | 9.3780e-02 | 0.0965 *** | 5.3980e-06 | Muy débil |
| PnL_fwd_pts_50_mediana | 0.0648 ** | 2.2922e-03 | 0.1201 *** | 1.4238e-08 | Muy débil |
| PnL_fwd_pts_90_mediana | 0.0591 ** | 5.3708e-03 | 0.2139 *** | 2.5622e-24 | Muy débil |

### FF_ATM

| PNL Variable | Pearson (r) | p-valor | Spearman (ρ) | p-valor | Interpretación |
|--------------|-------------|---------|--------------|---------|----------------|
| PnL_fwd_pts_01_mediana | 0.0979 *** | 3.8986e-06 | 0.0815 *** | 1.2361e-04 | Muy débil |
| PnL_fwd_pts_05_mediana | 0.0640 ** | 2.6029e-03 | 0.0895 *** | 2.4713e-05 | Muy débil |
| PnL_fwd_pts_25_mediana | 0.0825 *** | 1.0136e-04 | 0.0862 *** | 4.9155e-05 | Muy débil |
| PnL_fwd_pts_50_mediana | 0.1176 *** | 2.8720e-08 | 0.1115 *** | 1.4390e-07 | Débil |
| PnL_fwd_pts_90_mediana | 0.1024 *** | 1.3881e-06 | 0.0762 *** | 3.3435e-04 | Débil |

### delta_total

| PNL Variable | Pearson (r) | p-valor | Spearman (ρ) | p-valor | Interpretación |
|--------------|-------------|---------|--------------|---------|----------------|
| PnL_fwd_pts_01_mediana | -0.0043 ns | 8.4113e-01 | -0.0251 ns | 2.3823e-01 | Muy débil |
| PnL_fwd_pts_05_mediana | 0.0144 ns | 4.9878e-01 | -0.0409 ns | 5.4054e-02 | Muy débil |
| PnL_fwd_pts_25_mediana | 0.0089 ns | 6.7587e-01 | -0.0146 ns | 4.9192e-01 | Muy débil |
| PnL_fwd_pts_50_mediana | 0.0893 *** | 2.5965e-05 | 0.0605 ** | 4.4118e-03 | Muy débil |
| PnL_fwd_pts_90_mediana | 0.0151 ns | 4.7686e-01 | -0.0122 ns | 5.6511e-01 | Muy débil |

### theta_total

| PNL Variable | Pearson (r) | p-valor | Spearman (ρ) | p-valor | Interpretación |
|--------------|-------------|---------|--------------|---------|----------------|
| PnL_fwd_pts_01_mediana | 0.0465 * | 2.8643e-02 | 0.0612 ** | 3.9763e-03 | Muy débil |
| PnL_fwd_pts_05_mediana | -0.0383 ns | 7.1575e-02 | 0.0366 ns | 8.4742e-02 | Muy débil |
| PnL_fwd_pts_25_mediana | 0.0389 ns | 6.6939e-02 | 0.0766 *** | 3.0993e-04 | Muy débil |
| PnL_fwd_pts_50_mediana | 0.0306 ns | 1.4949e-01 | 0.0583 ** | 6.0957e-03 | Muy débil |
| PnL_fwd_pts_90_mediana | 0.1319 *** | 4.6129e-10 | 0.1418 *** | 2.0797e-11 | Débil |

### FF_BAT

| PNL Variable | Pearson (r) | p-valor | Spearman (ρ) | p-valor | Interpretación |
|--------------|-------------|---------|--------------|---------|----------------|
| PnL_fwd_pts_01_mediana | 0.0431 * | 4.2457e-02 | 0.0414 ns | 5.1471e-02 | Muy débil |
| PnL_fwd_pts_05_mediana | 0.0494 * | 2.0131e-02 | 0.0466 * | 2.8405e-02 | Muy débil |
| PnL_fwd_pts_25_mediana | 0.0308 ns | 1.4751e-01 | 0.0458 * | 3.0984e-02 | Muy débil |
| PnL_fwd_pts_50_mediana | 0.0492 * | 2.0619e-02 | 0.1383 *** | 6.3629e-11 | Muy débil |
| PnL_fwd_pts_90_mediana | 0.0233 ns | 2.7247e-01 | 0.0906 *** | 1.9744e-05 | Muy débil |

---

## 4. METODOLOGÍA

### Coeficientes de Correlación

- **Pearson (r):** Mide la correlación lineal entre dos variables continuas. Sensible a outliers.
- **Spearman (ρ):** Mide la correlación monotónica entre dos variables. Robusto ante outliers y no asume linealidad.

### Niveles de Significancia

- `***` p < 0.001 (altamente significativo)
- `**` p < 0.01 (muy significativo)
- `*` p < 0.05 (significativo)
- `ns` p ≥ 0.05 (no significativo)

### Interpretación de Correlaciones

| Rango |r| | Interpretación |
|-----------|----------------|
| 0.00 - 0.10 | Muy débil |
| 0.10 - 0.30 | Débil |
| 0.30 - 0.50 | Moderada |
| 0.50 - 0.70 | Fuerte |
| 0.70 - 1.00 | Muy fuerte |

---

## 5. HALLAZGOS CLAVE

1. **Variable objetivo con mayor correlación promedio:** FF_ATM (|r| promedio = 0.093)

2. **Horizonte temporal con correlaciones más fuertes:** PnL_fwd_pts_90_mediana (|r| promedio = 0.074)

3. **Correlación más fuerte:** PnL_fwd_pts_90_mediana vs theta_total (r = 0.1319, p = 4.6129e-10)

4. **Correlaciones estadísticamente significativas (p < 0.05):** 15/30 (50.0%)

5. **Correlaciones moderadas o fuertes (|r| ≥ 0.3):** 0/30 (0.0%)

---

## 6. CONCLUSIONES

Este análisis proporciona una evaluación completa de las correlaciones entre las variables de ventas (PNL_FWD_PTS) 
y las variables objetivo especificadas. Los resultados incluyen:

- Correlaciones de Pearson y Spearman para evaluar relaciones lineales y monotónicas
- Pruebas de significancia estadística para validar la fiabilidad de las correlaciones
- Visualizaciones comprehensivas (heatmaps, scatter plots, distribuciones)
- Análisis detallado por variable y horizonte temporal

Los archivos generados incluyen:
- `correlaciones_pearson.csv` / `correlaciones_spearman.csv`: Matrices de correlación
- `pvalues_pearson.csv` / `pvalues_spearman.csv`: Matrices de p-valores
- `estadisticas_descriptivas.csv`: Estadísticas descriptivas de todas las variables
- Múltiples gráficos en formato PNG de alta resolución (300 DPI)
