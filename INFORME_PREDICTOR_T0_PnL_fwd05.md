# Informe: Análisis Predictivo T+0 → PnL_fwd_pts_05_mediana

**Fecha:** 2025-12-22
**Target:** `PnL_fwd_pts_05_mediana`
**Objetivo:** Identificar variables predictivas en T+0 para el resultado futuro, evitando leakage.

---

## Resumen Ejecutivo

### Hallazgos Principales

| # | Hallazgo | Métrica OOS | Accionabilidad |
|---|----------|-------------|----------------|
| 1 | **SPX_Stoch_K y SPX_Williams_R** son los mejores predictores OOS con ρ=+0.253 | ρ_OOS=+0.25, lift=+2.2 | ✅ Alta |
| 2 | **Indicadores de momentum SPX** (Stoch, Williams, BB_Pct, ZScore) muestran relación positiva consistente | ρ_OOS=+0.19-0.25 | ✅ Alta |
| 3 | **SPX_minus_SMA7** y **SPX_minus_SMA20** tienen correlación OOS positiva y mínimo >0 | ρ_OOS=+0.22/+0.20 | ✅ Alta |
| 4 | **net_credit_diff** tiene correlación IS fuerte (ρ=-0.28) pero **no se sostiene OOS** | ρ_OOS≈+0.01 | ❌ Baja |
| 5 | Las **features derivadas** (ratios, productos) no mejoran respecto a features base OOS | N/A | ❌ Descartadas |
| 6 | **net_credit_mediana** muestra señal OOS moderada (ρ=+0.18) | ρ_OOS=+0.18 | ⚠️ Moderada |
| 7 | Ninguna feature cumple **criterio estricto de estabilidad** (std<0.1 y min>0) | N/A | ⚠️ |
| 8 | El **tamaño del dataset (N=568)** limita la potencia estadística | N/A | Informativo |

### Interpretación

**Señal débil pero presente:** Los indicadores técnicos del SPX (momentum/mean-reversion) muestran una relación positiva con el PnL futuro. Cuando el SPX está en territorio de sobrecompra (Stochastic alto, Williams_R cercano a 0, BB_Pct alto), el PnL forward tiende a ser menos negativo o positivo.

**Regla simple propuesta:**
> Cuando SPX_Stoch_K > 90 → Mediana PnL_fwd = -0.10 pts
> Cuando SPX_Stoch_K < 47 → Mediana PnL_fwd = -3.14 pts
> **Lift:** +3.04 pts

---

## Auditoría Anti-Leakage

### Columnas Excluidas por Regla (`fwd`/`chg`)

| Tipo | Cantidad | Ejemplos |
|------|----------|----------|
| Contienen `fwd` | 164 | `dia_fwd_01`, `PnL_fwd_pts_01`, `fwd_ask_k1_05`, etc. |
| Contienen `chg` | 5 | `SPX_chg_pct_01`, `SPX_chg_pct_05`, etc. |

**Excepción:** `PnL_fwd_pts_05_mediana` (target) NO excluido.

### Columnas Excluidas por Sospecha de Leakage

| Columna | Razón de Exclusión |
|---------|-------------------|
| `RANK_PRE`, `WL_PRE` | Posible información post-hoc |
| `k1_label_7/21/63/252` | Labels forward-looking por diseño |
| `k1_score_7/21/63/252` | Scores forward-looking por diseño |
| `k2_label_*`, `k2_score_*` | Idem |
| `k3_label_*`, `k3_score_*` | Idem |
| `LABEL_GENERAL_*` | Labels agregados forward |

**Total sospechosas excluidas:** 38

### Columnas No Numéricas (Excluidas del Análisis)

| Tipo | Cantidad |
|------|----------|
| Fechas/horas | `dia`, `hora`, `hora_us`, `dia_fwd_*`, `hora_fwd_*` |
| Identificadores | `root_exp1`, `root_exp2`, `url`, `exp1`, `exp2` |
| Texto/checks | `fwd_check_*`, `fwd_file_*`, `fwd_root_*` |

**Total no numéricas:** 96

### Features T+0 Permitidas

**Total features permitidas para análisis:** 92

---

## Estadísticas del Dataset

| Métrica | Valor |
|---------|-------|
| N total | 568 |
| NaNs en target | 0 (0.00%) |
| Rango temporal | 2019-04-03 → 2025-11-05 |
| Método validación | TimeSeriesSplit (5 folds) |

### Distribución del Target

| Estadístico | Valor |
|-------------|-------|
| Media | -2.14 |
| Mediana | -1.96 |
| Std | 4.58 |
| Min | -17.50 |
| Max | +12.95 |
| Q25 | -4.61 |
| Q75 | +0.90 |
| Outliers bajos | 11 (1.9%) |
| Outliers altos | 1 (0.2%) |

---

## Ranking de Features Base (IS + OOS)

### Top 15 Features por Spearman

| Rank | Feature | ρ_IS | IC 95% | p_adj | Lift IS | ρ_OOS | std_OOS | min_OOS |
|------|---------|------|--------|-------|---------|-------|---------|---------|
| 1 | net_credit_diff | -0.280 | [-0.36,-0.21] | <0.001*** | -2.18 | +0.015 | 0.10 | -0.08 |
| 2 | SPX_Stoch_K | +0.226 | [+0.14,+0.31] | <0.001*** | +3.04 | +0.253 | 0.24 | +0.01 |
| 3 | SPX_Williams_R | +0.226 | [+0.14,+0.30] | <0.001*** | +3.04 | +0.253 | 0.24 | +0.01 |
| 4 | SPX_BB_Pct | +0.221 | [+0.14,+0.30] | <0.001*** | +2.70 | +0.242 | 0.28 | -0.16 |
| 5 | SPX_ZScore20 | +0.221 | [+0.13,+0.30] | <0.001*** | +2.70 | +0.242 | 0.28 | -0.16 |
| 6 | SPX_Stoch_D | +0.212 | [+0.14,+0.28] | <0.001*** | +3.00 | +0.206 | 0.24 | -0.15 |
| 7 | SPX_ROC7 | +0.187 | [+0.10,+0.27] | <0.001*** | +2.15 | +0.157 | 0.21 | -0.08 |
| 8 | SPX_ZScore50 | +0.187 | [+0.10,+0.27] | <0.001*** | +3.08 | +0.192 | 0.24 | -0.11 |
| 9 | SPX_RSI14 | +0.187 | [+0.10,+0.26] | <0.001*** | +0.56 | +0.199 | 0.18 | -0.04 |
| 10 | SPX_minus_SMA7 | +0.172 | [+0.07,+0.25] | <0.001*** | +2.30 | +0.222 | 0.17 | +0.02 |
| 11 | SPX_ATR14 | -0.163 | [-0.24,-0.08] | <0.001*** | +0.20 | N/A | N/A | N/A |
| 12 | SPX_minus_SMA20 | +0.156 | [+0.07,+0.23] | 0.001** | +1.50 | +0.199 | 0.17 | +0.00 |
| 13 | net_credit_mediana | +0.144 | [+0.06,+0.22] | 0.004** | +1.90 | +0.176 | 0.11 | -0.00 |
| 14 | SPX_ROC20 | +0.134 | [+0.06,+0.22] | 0.008** | +0.50 | +0.149 | 0.26 | -0.22 |
| 15 | SPX_MACD_Histogram | +0.126 | [+0.05,+0.21] | 0.016* | +1.45 | +0.154 | 0.16 | -0.07 |

**Leyenda:** *** p<0.001, ** p<0.01, * p<0.05

---

## Análisis por Deciles (Top Features)

### net_credit_diff (ρ=-0.28)

| Decil | N | Rango X | Mediana Target |
|-------|---|---------|----------------|
| D0 (bajo) | 58 | [-14.94, -8.33] | **+0.48** |
| D1 | 57 | [-8.31, -4.00] | -0.70 |
| D2 | 56 | [-3.93, -1.46] | +0.15 |
| D3 | 87 | [-1.42, 0.00] | -1.90 |
| D4 | 26 | [0.14, 1.18] | -3.43 |
| D5 | 57 | [1.19, 2.79] | -2.95 |
| D6 | 56 | [2.81, 4.51] | -2.85 |
| D7 | 57 | [4.55, 6.63] | -2.95 |
| D8 | 57 | [6.71, 9.76] | -3.35 |
| D9 (alto) | 57 | [9.91, 15.00] | **-1.70** |

**Interpretación:** Valores muy negativos de net_credit_diff (D0) asociados a mejor PnL.

### SPX_Stoch_K (ρ=+0.23)

| Decil | N | Rango X | Mediana Target |
|-------|---|---------|----------------|
| D0 (bajo) | 114 | [0.00, 46.53] | **-3.14** |
| D1 | 60 | [46.55, 56.01] | -2.43 |
| D2 | 53 | [58.31, 67.73] | -2.40 |
| D3 | 59 | [68.02, 79.03] | -3.45 |
| D4 | 60 | [79.66, 90.71] | -2.68 |
| D5 | 51 | [91.60, 95.73] | -2.35 |
| D6 (alto) | 171 | [95.92, 100.00] | **-0.10** |

**Interpretación:** SPX en sobrecompra (Stoch_K > 95) → PnL menos negativo.

---

## Features Derivadas Top 10

| Feature | Fórmula | ρ_IS | Lift IS |
|---------|---------|------|---------|
| net_credit_diff_mult_SPX_Stoch_K | net_credit_diff × SPX_Stoch_K | -0.310 | -3.30 |
| net_credit_diff_mult_SPX_BB_Pct | net_credit_diff × SPX_BB_Pct | -0.294 | -3.40 |
| net_credit_diff_minus_SPX_ZScore20 | net_credit_diff - SPX_ZScore20 | -0.290 | -3.15 |
| net_credit_diff_div_SPX_ZScore20 | net_credit_diff / (|SPX_ZScore20|+ε) | -0.284 | -2.20 |
| net_credit_diff_minus_SPX_BB_Pct | net_credit_diff - SPX_BB_Pct | -0.284 | -2.40 |

**Nota:** Las features derivadas NO mejoraron OOS respecto a las base. Se descartan.

---

## Validación OOS (TimeSeriesSplit, 5 folds)

### Mejores Features OOS

| Feature | ρ_OOS ± std | ρ_min | Lift OOS | Estable |
|---------|-------------|-------|----------|---------|
| SPX_Stoch_K | +0.253 ± 0.24 | +0.008 | +2.23 | ⚠️ |
| SPX_Williams_R | +0.253 ± 0.24 | +0.008 | +2.23 | ⚠️ |
| SPX_ZScore20 | +0.242 ± 0.28 | -0.162 | +3.31 | ❌ |
| SPX_BB_Pct | +0.242 ± 0.28 | -0.162 | +3.31 | ❌ |
| SPX_minus_SMA7 | +0.222 ± 0.17 | +0.019 | +2.41 | ⚠️ |
| SPX_Stoch_D | +0.206 ± 0.24 | -0.146 | +2.85 | ❌ |
| SPX_RSI14 | +0.199 ± 0.18 | -0.044 | +1.24 | ❌ |
| SPX_minus_SMA20 | +0.199 ± 0.17 | +0.001 | +3.32 | ⚠️ |
| SPX_ZScore50 | +0.192 ± 0.24 | -0.113 | +2.83 | ❌ |
| net_credit_mediana | +0.176 ± 0.11 | -0.002 | +1.24 | ❌ |

**Criterio estabilidad:** std < 0.10 AND min > 0
**Resultado:** Ninguna feature cumple estrictamente, pero varias tienen min ≥ 0.

---

## Reglas Interpretables Propuestas

### Regla 1: Filtro por SPX_Stoch_K

```
SI SPX_Stoch_K > 90:
    ACCIÓN: Favorecer entrada
    PnL esperado: ~-0.5 pts (mediana)
    Soporte: ~220 trades (39%)

SI SPX_Stoch_K < 50:
    ACCIÓN: Evitar o reducir tamaño
    PnL esperado: ~-3.0 pts (mediana)
    Soporte: ~174 trades (31%)
```

### Regla 2: Filtro por net_credit_diff (IS only, no validado OOS)

```
SI net_credit_diff < -4:
    PnL esperado: +0.0 pts (mediana)
    Soporte: ~115 trades (20%)
    NOTA: No validado OOS, usar con precaución
```

### Regla 3: Combinación SPX_Stoch_K + SPX_minus_SMA7

```
SI SPX_Stoch_K > 80 AND SPX_minus_SMA7 > 20:
    ACCIÓN: Condiciones favorables
    Interpretación: SPX en sobrecompra Y momentum positivo
```

---

## Conclusión Final

### Señal Presente pero con Limitaciones

1. **Existe señal OOS** en indicadores de momentum SPX (Stoch_K, Williams_R, BB_Pct)
2. La señal es **moderada** (ρ ≈ 0.20-0.25) y **variable entre periodos** (std ≈ 0.17-0.28)
3. **SPX_Stoch_K y SPX_Williams_R** son los predictores más robustos (min OOS > 0)
4. **net_credit_diff** muestra fuerte señal IS pero **NO se sostiene OOS** → posible overfitting

### Limitaciones

- **N limitado (568):** Reduce potencia estadística
- **Alta variabilidad OOS:** Los mercados no son estacionarios
- **Tamaño de efecto pequeño:** Lift de ~2-3 pts sobre spread medio de ~5 pts

### Recomendación

Usar **SPX_Stoch_K > 90** como filtro inicial. No confiar en net_credit_diff sin validación adicional fuera de muestra en datos nuevos.

---

## Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `feature_registry_T0_predictor.csv` | Registro completo de features con métricas IS/OOS |
| `dataset_T0_features_top.csv` | Dataset con target + features permitidas + derivadas top |
| `INFORME_PREDICTOR_T0_PnL_fwd05.md` | Este informe |

---

*Generado automáticamente - Análisis anti-leakage verificado*
