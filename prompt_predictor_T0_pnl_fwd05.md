# Prompt de análisis (T+0 predictor vs PnL futuro)

> **Objetivo:** descubrir variables/factores **predictivos en T+0** (snapshot del CSV) que se relacionen de forma **estable** con el target futuro **`PnL_fwd_pts_05_mediana`**, evitando estrictamente **leakage / look-ahead**.

---

## Contexto y objetivo

Tienes un CSV con un “snapshot” de variables disponibles en **T+0** (momento de decisión/entrada).  
El dataset contiene también columnas con información futura (look-ahead), además del propio target.

**Quiero descubrir 1 o varias variables/factores (existentes o derivados) que sean predictoras en T+0** del resultado futuro medido por:

- **Target:** `PnL_fwd_pts_05_mediana`

---

## Archivo

En el directorio de trabajo existe exactamente este CSV y debes cargarlo:

- `/mnt/data/combined_BATMAN_mediana_w_stats_w_vix_labeled_NOSPXCHG.csv`

---

## Restricción crítica (anti-leakage) — OBLIGATORIA

1) **Excluir totalmente** de cualquier cálculo/entrenamiento/selección/feature engineering **todas** las columnas cuyo nombre contenga (case-insensitive):
- `fwd` o `chg`

2) **Excepción única:** **NO excluir** el target `PnL_fwd_pts_05_mediana` aunque contenga `fwd`.

3) Además, **detectar y excluir** columnas con look-ahead **aunque no contengan** esas cadenas, por ejemplo:
- variables construidas usando resultados posteriores,
- labels futuros,
- agregados/rolling que incluyan datos posteriores a T+0,
- cualquier columna derivada explícitamente del PnL futuro o de cambios posteriores.

> Si hay duda, **excluir** (mejor falsos positivos que leakage).

4) Documentar en un bloque **AUDIT**:
- columnas excluidas por regla (`fwd/chg`),
- columnas excluidas por sospecha de leakage (con explicación),
- columnas finalmente permitidas (features T+0).

---

## Qué significa “éxito” (criterio práctico)

Busco **edge estadístico** usable para ordenar/filtrar trades en T+0.  
El entregable final debe ser:

- una **regla interpretable** (umbral/rango/cuantiles) con soporte suficiente, **o**
- un **score parsimonioso** (2–5 inputs) que rankee trades,
- y que muestre **estabilidad OOS**.

**Métricas de éxito (preferencia):**
- **Lift Top decil vs Bottom decil** en mediana del target,
- **Spearman OOS** entre score y target,
- IC por bootstrap,
- control de múltiples tests (FDR).

---

# Metodología requerida (ejecutar en este orden)

## (A) Validaciones iniciales

1) Identifica columna de fecha/tiempo si existe (por nombre o tipo).  
   - Si hay fecha: validación **walk-forward / TimeSeriesSplit**.  
   - Si no: KFold (idealmente con control de distribución del target).

2) Reporta:
- N total,
- NaNs, infinitos, duplicados,
- distribución del target,
- outliers.

3) Define mínimos para conclusiones:
- Si **N < 100**, no saques conclusiones predictivas (solo descriptivo).
- Para bins/cuantiles: mínimo **≥ 30** muestras por bin (o ≥ 5% de N, el mayor).

---

## (B) Baselines (señal directa, robusta)

Para cada feature permitida:
- Pearson y Spearman vs target (con **IC bootstrap**).
- Tamaño de efecto por cuantiles (top/bottom decil; Cliff’s delta si aplica).
- p-values por permutación del target (si es viable) + ajuste **FDR**.

**Salida:** ranking de features “raw” (existentes) por señal y estabilidad.

---

## (C) Reglas interpretables por rangos/umbrales

Para cada feature top:
- Búsqueda de umbral **simple**: split `X < t` vs `X ≥ t` optimizando `Δ mediana(target)` con restricciones de soporte.
- Análisis por quintiles/deciles: tabla y boxplots.
- Reportar: `Δ mediana`, `IC95%`, soporte, `p_adj`, y estabilidad por folds.

---

## (D) Feature engineering parsimonioso y seguro

Genera features derivadas **solo** desde columnas permitidas y con transformaciones seguras (sin mirar al futuro):

- Transformaciones: `rank`, `abs`, `log1p(abs(x))`, `winsor`, `zscore_robusto` (mediana/MAD).
- Ratios: `x/(abs(y)+ε)` (ε pequeño).
- Spreads: `x - y`.
- Interacciones limitadas: solo entre **top 10** features base.
- Combinaciones lineales **muy simples** (2–5 variables) con regularización (Ridge/Lasso) y validación OOS.

**Prohibido:** explosión combinatoria sin control.

---

## (E) Iteración generativa “en bucle” pero con control

Implementa un ciclo de descubrimiento con **memoria de features**:

- Mantén un **Feature Registry** con:
  - nombre, fórmula, inputs usados, transformaciones,
  - interpretación (qué mide y por qué podría tener sentido),
  - métricas IS/OOS,
  - si pasa filtros o es descartada.

**Rondas:**
1) Ronda 0: solo features existentes (permitidas).
2) Ronda 1: derivadas desde top features (máx **200** nuevas).
3) Ronda 2: solo si hay señal OOS: refina desde el top 5 (máx **100** nuevas).

**Criterios de parada:**
- si durante 2 rondas no mejora el mejor score OOS (p.ej. lift top/bottom decil) más de un umbral mínimo,
- o si las señales son inestables entre folds,
- o si el soporte cae por debajo del mínimo.

---

## (F) Validación anti-overfit (obligatoria)

- OOS por folds (time-split si hay fecha).
- Reporta para cada candidato final:
  - Spearman OOS,
  - Lift top decil vs bottom decil (`Δ mediana target`),
  - estabilidad: media y dispersión por fold,
  - IC bootstrap,
  - FDR cuando aplique.

Si un hallazgo funciona solo in-sample: **descartarlo**.

---

# Entregables obligatorios

1) **Informe Markdown** con:
- Resumen ejecutivo: 5–10 hallazgos accionables (o “no hay relación clara” con razones).
- Auditoría anti-leakage (columnas excluidas y por qué).
- Ranking top features base y derivadas con métricas completas (IS/OOS, IC, p_adj, soporte).
- Reglas por umbral/rangos con soporte y estabilidad.
- Gráficos mínimos: scatter con tendencia, bins por deciles, boxplot por decil del score.

2) **CSV de salida**:
- dataset con el target + **solo** features permitidas + las **top N** derivadas finales (N pequeño, p.ej. 20).
- archivo adicional con el **Feature Registry** (tabla con fórmulas y métricas).

3) **Conclusión final honesta**:
- si no existe señal estable OOS, dilo explícitamente y explica si es por ruido, no estacionariedad, tamaño de efecto pequeño, o falta de potencia estadística.
