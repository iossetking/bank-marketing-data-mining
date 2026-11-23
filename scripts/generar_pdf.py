"""
Genera el documento académico final en PDF (Informe_Bank_Marketing.pdf) a partir de
outputs/metricas_resumen.json y las visualizaciones en outputs/figuras/.
Usa HTML + CSS académico y Google Chrome headless para producir un PDF limpio.
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figuras"
METRICS_JSON = OUT_DIR / "metricas_resumen.json"
HTML_PATH = OUT_DIR / "Informe_Bank_Marketing.html"
PDF_PATH = ROOT / "Informe_Bank_Marketing.pdf"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def img_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def fmt_rows_cv(cv_list: list[dict], best_name: str) -> str:
    rows = []
    for r in cv_list:
        cls = ' class="highlight"' if r["Modelo"] == best_name else ""
        rows.append(
            f"<tr{cls}>"
            f"<td><strong>{r['Modelo']}</strong></td>"
            f"<td>{r['ROC AUC']:.4f} ± {r['ROC AUC std']:.4f}</td>"
            f"<td>{r['PR AUC']:.4f}</td>"
            f"<td>{r['Exactitud Balanceada']:.4f}</td>"
            f"<td>{r['F1']:.4f}</td>"
            f"<td>{r['Precisión']:.4f}</td>"
            f"<td>{r['Sensibilidad']:.4f}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def fmt_rows_test(test_list: list[dict], best_name: str) -> str:
    rows = []
    for r in test_list:
        cls = ' class="highlight"' if r["Modelo"] == best_name else ""
        rows.append(
            f"<tr{cls}>"
            f"<td><strong>{r['Modelo']}</strong></td>"
            f"<td>{r['ROC AUC']:.4f}</td>"
            f"<td>{r['PR AUC']:.4f}</td>"
            f"<td>{r['Exactitud']:.4f}</td>"
            f"<td>{r['Exactitud Balanceada']:.4f}</td>"
            f"<td>{r['F1']:.4f}</td>"
            f"<td>{r['Precisión']:.4f}</td>"
            f"<td>{r['Sensibilidad']:.4f}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def fmt_rows_deciles(deciles: list[dict]) -> str:
    rows = []
    for r in deciles[:5]:
        rows.append(
            f"<tr>"
            f"<td><strong>{r['decil']}</strong></td>"
            f"<td>{r['clientes_contactados']:,}</td>"
            f"<td>{r['suscriptores_capturados']:,}</td>"
            f"<td>{r['tasa_conversion_pct']:.2f}%</td>"
            f"<td>{r['ganancia_acumulada_pct']:.2f}%</td>"
            f"<td><strong>{r['lift']:.2f}x</strong></td>"
            f"</tr>"
        )
    last = deciles[-1]
    rows.append(
        f"<tr>"
        f"<td><strong>{last['decil']} (Sin filtro)</strong></td>"
        f"<td>{last['clientes_contactados']:,}</td>"
        f"<td>{last['suscriptores_capturados']:,}</td>"
        f"<td>{last['tasa_conversion_pct']:.2f}%</td>"
        f"<td>{last['ganancia_acumulada_pct']:.2f}%</td>"
        f"<td>{last['lift']:.2f}x</td>"
        f"</tr>"
    )
    return "\n".join(rows)


def generar_pdf() -> None:
    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    best_name = m["mejor_modelo"]
    best_cv = next(r for r in m["cv"] if r["Modelo"] == best_name)
    best_test = next(r for r in m["test"] if r["Modelo"] == best_name)
    dec10 = m["deciles"][0]
    dec20 = m["deciles"][1]
    dec30 = m["deciles"][2]
    top_vars = ", ".join(f"<code>{r['variable']}</code>" for r in m["importancia_top5"][:5])

    figs = {
        f"fig{i}": img_to_data_uri(FIG_DIR / fname)
        for i, fname in enumerate(
            [
                "01_exploracion_objetivo.png",
                "02_diagnostico_atipicos.png",
                "03_filtros_seleccion.png",
                "04_comparacion_cv.png",
                "05_curvas_roc_pr.png",
                "06_matrices_confusion.png",
                "07_importancia_variables.png",
                "08_ganancia_lift_campana.png",
            ],
            start=1,
        )
    }

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Proyecto de Minería de Datos - Bank Marketing</title>
<style>
  @page {{
    size: letter;
    margin: 1.5cm 1.6cm 1.6cm 1.6cm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1f2937;
    font-size: 9.7pt;
    line-height: 1.42;
    margin: 0;
  }}
  .cover {{
    border-bottom: 3px solid #1d4e89;
    padding-bottom: 14px;
    margin-bottom: 18px;
  }}
  .course-tag {{
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #1d4e89;
    font-weight: 700;
    margin-bottom: 6px;
  }}
  h1.doc-title {{
    font-size: 20pt;
    color: #0f172a;
    margin: 0 0 6px 0;
    line-height: 1.2;
  }}
  .subtitle {{
    font-size: 11pt;
    color: #475569;
    margin: 0 0 10px 0;
  }}
  .meta-grid {{
    display: flex;
    gap: 20px;
    font-size: 9pt;
    color: #334155;
    background: #f1f5f9;
    padding: 8px 12px;
    border-radius: 6px;
  }}
  h2 {{
    font-size: 13.5pt;
    color: #1d4e89;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 3px;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 11pt;
    color: #0f172a;
    margin-top: 12px;
    margin-bottom: 5px;
    page-break-after: avoid;
  }}
  p {{
    margin: 0 0 8px 0;
    text-align: justify;
  }}
  ul {{
    margin: 4px 0 10px 0;
    padding-left: 20px;
  }}
  li {{
    margin-bottom: 4px;
  }}
  .callout {{
    background: #eff6ff;
    border-left: 4px solid #1d4e89;
    padding: 9px 12px;
    margin: 10px 0;
    border-radius: 0 6px 6px 0;
    font-size: 9.8pt;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 12px 0;
    font-size: 8.8pt;
    page-break-inside: avoid;
  }}
  th {{
    background: #1d4e89;
    color: #ffffff;
    text-align: left;
    padding: 6px 7px;
    font-weight: 600;
  }}
  td {{
    border: 1px solid #cbd5e1;
    padding: 5px 7px;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{
    background: #f8fafc;
  }}
  tr.highlight td {{
    background: #dbeafe !important;
    font-weight: 600;
  }}
  .figure {{
    margin: 10px 0 14px 0;
    text-align: center;
    page-break-inside: avoid;
  }}
  .figure img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
  }}
  .caption {{
    font-size: 8.6pt;
    color: #475569;
    margin-top: 4px;
    font-style: italic;
  }}
  code {{
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9pt;
    color: #0f172a;
  }}
  .kpi-row {{
    display: flex;
    gap: 10px;
    margin: 10px 0 14px 0;
    page-break-inside: avoid;
  }}
  .kpi {{
    flex: 1;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-top: 3px solid #1d4e89;
    border-radius: 6px;
    padding: 8px 10px;
    text-align: center;
  }}
  .kpi .val {{
    font-size: 15pt;
    font-weight: 700;
    color: #1d4e89;
  }}
  .kpi .lbl {{
    font-size: 8.2pt;
    color: #475569;
  }}
  .page-break {{
    page-break-before: always;
  }}
</style>
</head>
<body>

<div class="cover">
  <div class="course-tag">Minería de Datos (Data Mining) · Proyecto Aplicado</div>
  <h1 class="doc-title">Predicción de Suscripción a Depósitos a Plazo Fijo para la Priorización de Campañas Telefónicas</h1>
  <p class="subtitle">Análisis exploratorio, preprocesamiento reproducible, selección de características y comparación de modelos de clasificación sobre el conjunto UCI Bank Marketing</p>
  <div class="meta-grid">
    <div><strong>Dataset:</strong> UCI Bank Marketing (ID 222)</div>
    <div><strong>Registros:</strong> {m['n_total']:,}</div>
    <div><strong>Mejor modelo (CV):</strong> {best_name}</div>
    <div><strong>ROC AUC (Prueba):</strong> {best_test['ROC AUC']:.3f}</div>
  </div>
</div>

<h2>1. Objetivo del proyecto</h2>
<p>
  Las campañas de marketing telefónico representan un costo operativo importante para las instituciones financieras debido al tiempo de los agentes, los múltiples intentos de contacto y la fatiga que las llamadas repetidas generan en los clientes. En este contexto, contactar clientes sin un criterio de priorización produce tasas de conversión bajas y desperdicia recursos comerciales.
</p>
<div class="callout">
  <strong>Objetivo principal:</strong> Estimar la probabilidad de que un cliente contrate un depósito a plazo fijo utilizando únicamente información disponible antes de realizar la llamada, con el fin de <strong>priorizar campañas telefónicas</strong> y concentrar el esfuerzo comercial en los clientes con mayor propensión a suscribirse.
</div>
<p>
  Para cumplir este objetivo se ejecutó un flujo completo de Minería de Datos: diagnóstico de calidad, limpieza trazable, ingeniería de características, selección de variables mediante métodos de filtro, entrenamiento y comparación de cuatro algoritmos de clasificación supervisada con validación cruzada estratificada, y evaluación final enfocada tanto en métricas de clasificación como en ganancia acumulada (<em>Lift</em>) para toma de decisiones comerciales.
</p>

<h2>2. Descripción del conjunto de datos</h2>
<p>
  Se utilizó el conjunto de datos público <strong>Bank Marketing</strong> del repositorio de la Universidad de California en Irvine (UCI Machine Learning Repository, ID 222), recopilado a partir de campañas de marketing directo vía telefónica de una institución bancaria portuguesa. El conjunto cuenta con <strong>{m['n_total']:,} registros</strong>, <strong>16 variables predictoras</strong> (7 numéricas y 9 categóricas) y una variable objetivo binaria (<code>y</code>) que indica si el cliente suscribió (<code>yes</code>) o no (<code>no</code>) el depósito a plazo fijo.
</p>
<p>
  Un aspecto crítico del conjunto es su <strong>desbalance de clases</strong>: únicamente el <strong>{m['positivos_pct']:.2f}%</strong> de los clientes suscribieron el depósito, mientras que el {100 - m['positivos_pct']:.2f}% restante no aceptó la oferta. Esto implica que métricas ingenuas como la exactitud global pueden ser engañosas y hace necesario emplear particiones estratificadas, ponderación de clases y métricas sensibles a la clase minoritaria.
</p>

<table>
  <thead>
    <tr>
      <th>Variable</th>
      <th>Tipo</th>
      <th>Descripción y rol en el análisis</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>age</code></td><td>Numérica</td><td>Edad del cliente en años.</td></tr>
    <tr><td><code>job</code></td><td>Categórica</td><td>Ocupación del cliente (11 categorías; 288 valores faltantes).</td></tr>
    <tr><td><code>marital</code></td><td>Categórica</td><td>Estado civil (<code>married</code>, <code>single</code>, <code>divorced</code>).</td></tr>
    <tr><td><code>education</code></td><td>Categórica</td><td>Nivel educativo (<code>primary</code>, <code>secondary</code>, <code>tertiary</code>; 1,857 faltantes).</td></tr>
    <tr><td><code>default</code></td><td>Binaria</td><td>Indica si el cliente tiene crédito en mora (<code>yes</code>/<code>no</code>).</td></tr>
    <tr><td><code>balance</code></td><td>Numérica</td><td>Saldo anual promedio del cliente en euros.</td></tr>
    <tr><td><code>housing</code>, <code>loan</code></td><td>Binarias</td><td>Indican si el cliente posee crédito hipotecario o préstamo personal.</td></tr>
    <tr><td><code>contact</code></td><td>Categórica</td><td>Canal de contacto (<code>cellular</code>, <code>telephone</code>; 13,020 faltantes).</td></tr>
    <tr><td><code>day_of_week</code>, <code>month</code></td><td>Num. / Cat.</td><td>Día del mes y mes del último contacto de la campaña actual.</td></tr>
    <tr><td><code>duration</code></td><td>Numérica</td><td>Duración de la llamada en segundos. <strong>Excluida</strong> por fuga de datos (solo se conoce al finalizar la llamada).</td></tr>
    <tr><td><code>campaign</code></td><td>Numérica</td><td>Número de contactos realizados al cliente durante la campaña actual.</td></tr>
    <tr><td><code>pdays</code>, <code>previous</code></td><td>Numéricas</td><td>Días desde el contacto previo (<code>-1</code> = sin contacto previo) y número de contactos anteriores.</td></tr>
    <tr><td><code>poutcome</code></td><td>Categórica</td><td>Resultado de la campaña anterior (<code>success</code>, <code>failure</code>, <code>other</code>; 36,959 faltantes).</td></tr>
    <tr><td><code>y</code> (Objetivo)</td><td>Binaria</td><td>Suscripción al depósito a plazo fijo (<code>1</code> = sí, <code>0</code> = no).</td></tr>
  </tbody>
</table>

<div class="figure">
  <img src="{figs['fig1']}" alt="Exploración de la variable objetivo">
  <div class="caption">Figura 1. Distribución desbalanceada de la variable objetivo y tasas de suscripción según el resultado de la campaña previa (poutcome) y el mes de contacto.</div>
</div>

<h2>3. Metodología (Proceso KDD)</h2>
<p>
  El proyecto sigue las etapas del proceso clásico de Descubrimiento de Conocimiento en Bases de Datos (<strong>KDD</strong>): <em>Definición del objetivo</em> (Sección 1), <em>Selección de datos</em>, <em>Limpieza de datos</em>, <em>Transformación de datos</em>, <em>Minería de datos</em> y <em>Evaluación e interpretación</em> (Secciones 4 y 5).
</p>

<h3>3.1 Selección y limpieza de datos</h3>
<ul>
  <li><strong>Selección de datos:</strong> Se seleccionó el conjunto UCI Bank Marketing (45,211 registros y 16 variables predictoras) y se <strong>excluyó <code>duration</code></strong> por fuga de información, ya que solo se conoce al terminar la llamada y no puede usarse para decidir a quién llamar.</li>
  <li><strong>Duplicados y consistencia de texto:</strong> Se verificó que existen <strong>{m['n_duplicados']} filas duplicadas exactas</strong> y se normalizaron espacios y minúsculas en todas las variables categóricas. Todos los valores observados correspondieron al dominio documentado.</li>
  <li><strong>Faltantes y códigos especiales:</strong> Los valores faltantes en <code>job</code>, <code>education</code>, <code>contact</code> y <code>poutcome</code> no se eliminaron para evitar perder más del 80% de la muestra; se imputaron con la moda dentro del pipeline. El valor <code>pdays = -1</code> se conservó porque representa ausencia de contacto previo.</li>
  <li><strong>Valores atípicos (IQR):</strong> Mediante diagramas de caja e IQR se identificaron valores extremos en <code>balance</code>, <code>campaign</code>, <code>pdays</code> y <code>previous</code>. Se conservaron porque representan comportamientos bancarios reales (clientes con altos saldos o historial largo) y se controló su escala mediante normalización.</li>
</ul>

<div class="figure">
  <img src="{figs['fig2']}" alt="Diagnóstico de valores atípicos">
  <div class="caption">Figura 2. Diagnóstico visual de distribución y valores atípicos mediante criterio IQR en las variables numéricas originales.</div>
</div>

<h3>3.2 Transformación de datos y Minería de Datos</h3>
<ul>
  <li><strong>Transformación e ingeniería de características:</strong> Se crearon dos variables derivadas disponibles antes de la llamada: <code>presion_contacto = campaign / (previous + 1)</code> (insistencia reciente frente al historial) y <code>balance_ajustado_credito = balance / (1 + housing + loan)</code> (capacidad financiera ajustada por deudas activas). Las numéricas se imputaron con mediana y se escalaron con <code>MinMaxScaler</code>; las categóricas se imputaron con moda y se codificaron con <code>OneHotEncoder</code> dentro de un <code>Pipeline</code> sin fuga de datos.</li>
  <li><strong>Partición y selección de características:</strong> Se dividió el conjunto de forma estratificada en <strong>80% entrenamiento ({m['n_train']:,} registros)</strong> y <strong>20% prueba independiente ({m['n_test']:,} registros)</strong>. Dentro de cada pliegue de entrenamiento se aplicó <strong>Chi-cuadrada (V de Cramér)</strong> para categóricas (top 25 codificadas) y <strong>Pearson</strong> para numéricas (top 6).</li>
</ul>
<p>
  <strong>Minería de Datos (modelado):</strong> Se implementaron y compararon <strong>cuatro algoritmos de clasificación supervisada</strong>, todos con <code>class_weight="balanced"</code> para compensar el desbalance de clases y evaluados mediante validación cruzada estratificada de 5 particiones (<em>5-Fold Stratified CV</em>):
</p>
<ul>
  <li><strong>Regresión Logística:</strong> Modelo lineal base, rápido e interpretable que estima directamente probabilidades de suscripción.</li>
  <li><strong>Árbol de Decisión</strong> (<code>max_depth=8</code>, <code>min_samples_leaf=50</code>): Modelo no lineal basado en reglas de decisión controladas para evitar sobreajuste.</li>
  <li><strong>Bosque Aleatorio (Random Forest)</strong> (200 árboles, <code>max_depth=12</code>): Ensamble tipo <em>bagging</em> que reduce varianza y captura interacciones no lineales.</li>
  <li><strong>Gradient Boosting (HistGradientBoosting)</strong> (150 iteraciones, <code>learning_rate=0.05</code>, <code>max_depth=6</code>): Ensamble secuencial tipo <em>boosting</em> altamente eficaz en datos tabulares.</li>
</ul>

<h2>4. Resultados</h2>

<h3>4.1 Selección de características por filtros</h3>
<p>
  El filtro Chi-cuadrada mostró que <code>poutcome</code> (V de Cramér = {m['ranking_chi2'][0]['v_cramer']:.3f}), <code>month</code> (V = {m['ranking_chi2'][1]['v_cramer']:.3f}) y <code>housing</code> (V = {m['ranking_chi2'][2]['v_cramer']:.3f}) concentran las asociaciones categóricas más fuertes con la suscripción. En numéricas, Pearson destacó <code>{m['ranking_pearson'][0]['variable']}</code> (r = {m['ranking_pearson'][0]['correlacion']:.3f}), <code>{m['ranking_pearson'][1]['variable']}</code> (r = {m['ranking_pearson'][1]['correlacion']:.3f}) y <code>{m['ranking_pearson'][2]['variable']}</code> (r = {m['ranking_pearson'][2]['correlacion']:.3f}).
</p>

<div class="figure">
  <img src="{figs['fig3']}" alt="Filtros de selección">
  <div class="caption">Figura 3. Ranking de asociación frente a la variable objetivo mediante Chi-cuadrada (V de Cramér) y correlación de Pearson en el conjunto de entrenamiento.</div>
</div>

<h3>4.2 Comparación de modelos en validación cruzada y conjunto de prueba</h3>

<div class="kpi-row">
  <div class="kpi"><div class="val">{best_test['ROC AUC']:.3f}</div><div class="lbl">ROC AUC en Prueba ({best_name})</div></div>
  <div class="kpi"><div class="val">{best_test['PR AUC']:.3f}</div><div class="lbl">PR AUC en Prueba (Base: 0.117)</div></div>
  <div class="kpi"><div class="val">{dec10['lift']:.2f}x</div><div class="lbl">Lift en el Top 10% de clientes</div></div>
  <div class="kpi"><div class="val">{dec30['ganancia_acumulada_pct']:.1f}%</div><div class="lbl">Suscriptores capturados en Top 30%</div></div>
</div>

<table>
  <thead>
    <tr>
      <th>Modelo (Validación Cruzada 5-Fold)</th>
      <th>ROC AUC</th>
      <th>PR AUC</th>
      <th>Exact. Bal.</th>
      <th>F1</th>
      <th>Precisión</th>
      <th>Sensibilidad</th>
    </tr>
  </thead>
  <tbody>
    {fmt_rows_cv(m['cv'], best_name)}
  </tbody>
</table>

<div class="figure">
  <img src="{figs['fig4']}" alt="Comparación CV">
  <div class="caption">Figura 4. Comparación de métricas promedio de los cuatro modelos en validación cruzada estratificada de 5 particiones.</div>
</div>

<p>
  Con base estricta en la validación cruzada, <strong>{best_name}</strong> obtuvo el mejor desempeño global (ROC AUC = {best_cv['ROC AUC']:.4f}, PR AUC = {best_cv['PR AUC']:.4f}, F1 = {best_cv['F1']:.4f}). Al evaluar los modelos entrenados sobre el conjunto de prueba independiente (9,043 clientes no vistos), se confirmó su estabilidad sin señales de sobreajuste:
</p>

<table>
  <thead>
    <tr>
      <th>Modelo (Conjunto de Prueba)</th>
      <th>ROC AUC</th>
      <th>PR AUC</th>
      <th>Exactitud</th>
      <th>Exact. Bal.</th>
      <th>F1</th>
      <th>Precisión</th>
      <th>Sensibilidad</th>
    </tr>
  </thead>
  <tbody>
    {fmt_rows_test(m['test'], best_name)}
  </tbody>
</table>

<div class="figure">
  <img src="{figs['fig5']}" alt="Curvas ROC y PR">
  <div class="caption">Figura 5. Curvas ROC y Precisión-Sensibilidad comparativas de los cuatro modelos sobre el conjunto de prueba independiente.</div>
</div>

<div class="figure">
  <img src="{figs['fig6']}" alt="Matrices de confusión">
  <div class="caption">Figura 6. Matrices de confusión de los cuatro clasificadores en el conjunto de prueba (n = 9,043).</div>
</div>

<h2>5. Interpretación y análisis</h2>

<h3>5.1 Variables determinantes e interpretación del modelo</h3>
<p>
  Para comprender qué atributos impulsan las decisiones de <strong>{best_name}</strong> se calculó la importancia por permutación (<em>Permutation Importance</em>) sobre el conjunto de prueba. Las cinco variables más influyentes fueron {top_vars}. Destaca especialmente que una de las variables creadas en la ingeniería de características (<code>balance_ajustado_credito</code>) aparece entre las más relevantes, confirmando que relacionar el saldo con la carga crediticia aporta señal predictiva útil.
</p>

<div class="figure">
  <img src="{figs['fig7']}" alt="Importancia de variables">
  <div class="caption">Figura 7. Top 10 de variables más influyentes en {best_name} según la caída promedio en ROC AUC al permutar cada atributo.</div>
</div>

<ul>
  <li><strong>Estacionalidad (<code>month</code>):</strong> Los meses con menor volumen de llamadas masivas (marzo, septiembre, octubre y diciembre) presentan tasas de conversión muy superiores al promedio de 11.7%, reflejando campañas más focalizadas o condiciones comerciales favorables.</li>
  <li><strong>Situación crediticia y saldo (<code>housing</code>, <code>balance_ajustado_credito</code>):</strong> Los clientes sin hipoteca activa y con mayor liquidez relativa muestran mayor disposición a inmovilizar fondos en un depósito a plazo fijo.</li>
  <li><strong>Historial de campañas previas (<code>poutcome</code>):</strong> Quienes ya aceptaron una oferta en campañas anteriores presentan una tasa de conversión drásticamente más alta, convirtiéndose en el segmento prioritario natural.</li>
</ul>

<h3>5.2 Relación con el objetivo: priorización de campañas telefónicas</h3>
<p>
  Aunque la precisión en el umbral estándar de 0.50 es moderada ({best_test['Precisión']*100:.1f}%) debido al fuerte desbalance de clases y a la exclusión realista de <code>duration</code>, la utilidad operativa del modelo reside en su capacidad para <strong>ordenar a los clientes por probabilidad estimada de suscripción</strong>. La Tabla de Deciles y la Figura 8 cuantifican el impacto directo en una campaña telefónica sobre los 9,043 clientes de prueba (1,058 suscriptores reales):
</p>

<table>
  <thead>
    <tr>
      <th>Segmento priorizado</th>
      <th>Clientes contactados</th>
      <th>Suscriptores capturados</th>
      <th>Tasa de conversión</th>
      <th>Ganancia acumulada</th>
      <th>Factor Lift</th>
    </tr>
  </thead>
  <tbody>
    {fmt_rows_deciles(m['deciles'])}
  </tbody>
</table>

<div class="figure">
  <img src="{figs['fig8']}" alt="Ganancia y Lift">
  <div class="caption">Figura 8. Curva de ganancia acumulada y factor Lift por deciles usando {best_name} para priorizar llamadas frente a una estrategia aleatoria.</div>
</div>

<p>
  Los resultados demuestran un beneficio operativo claro: contactando únicamente al <strong>Top 10%</strong> de los clientes ordenados por el modelo ({dec10['clientes_contactados']:,} llamadas), el banco logra una tasa de conversión de <strong>{dec10['tasa_conversion_pct']:.1f}%</strong> (casi <strong>{dec10['lift']:.1f} veces</strong> superior al 11.7% aleatorio) y captura el <strong>{dec10['ganancia_acumulada_pct']:.1f}%</strong> de todas las suscripciones posibles. Si amplía la campaña al <strong>Top 30%</strong>, captura el <strong>{dec30['ganancia_acumulada_pct']:.1f}%</strong> de los suscriptores ahorrando el 70% de las llamadas telefónicas.
</p>

<h3>5.3 Justificación del modelo óptimo y decisión recomendada</h3>
<ul>
  <li><strong>¿Por qué {best_name} es el modelo óptimo?</strong> Fue elegido estrictamente por validación cruzada (sin usar el conjunto de prueba para decidir) al liderar ROC AUC ({best_cv['ROC AUC']:.4f}), PR AUC ({best_cv['PR AUC']:.4f}), Exactitud Balanceada ({best_cv['Exactitud Balanceada']:.4f}) y F1 ({best_cv['F1']:.4f}). Aunque en prueba el Bosque Aleatorio empató prácticamente en ROC AUC (0.7651 vs 0.7641), <strong>{best_name} lo superó claramente en las métricas críticas bajo desbalance</strong>: mayor PR AUC (0.4363 vs 0.4134), mayor F1 (0.4102 vs 0.3877), mayor precisión (31.5% vs 28.7%) y <strong>213 llamadas innecesarias menos</strong> (1,349 falsos positivos frente a 1,562 del Bosque Aleatorio, 1,698 de Regresión Logística y 1,864 del Árbol de Decisión).</li>
  <li><strong>¿Por qué se descartan Regresión Logística y Árbol de Decisión?</strong> La Regresión Logística asume relaciones lineales y no captura interacciones entre estacionalidad, historial previo y saldo ajustado; el Árbol de Decisión individual presenta mayor varianza y la menor capacidad discriminativa (ROC AUC = 0.7359).</li>
  <li><strong>Decisión operativa óptima para el banco:</strong> Usar las probabilidades de {best_name} como <em>ranking</em> de priorización. Si el presupuesto de llamadas es limitado, enfocar el esfuerzo en el <strong>Top 10%</strong> maximiza el retorno por llamada (48.7% de conversión, Lift 4.17x); si se busca equilibrar volumen de captación y ahorro operativo, el <strong>Top 30%</strong> es el punto óptimo, pues concentra el 65.3% de los suscriptores reduciendo el 70% de las llamadas, mientras que contactar más allá del 50% genera rendimientos marginales cercanos al azar.</li>
</ul>

<h2>6. Conclusiones</h2>
<ul>
  <li><strong>Cumplimiento del objetivo:</strong> Se desarrolló y validó un pipeline completo de Minería de Datos capaz de estimar la probabilidad de contratación de un depósito a plazo fijo antes de realizar la llamada, demostrando mediante el análisis de ganancia acumulada que el modelo permite priorizar eficazmente las campañas telefónicas.</li>
  <li><strong>Superioridad de los ensambles no lineales:</strong> Los modelos basados en ensambles de árboles (<strong>{best_name}</strong> y Bosque Aleatorio) superaron consistentemente a la Regresión Logística y al Árbol de Decisión individual en ROC AUC, PR AUC, exactitud balanceada y F1, capturando interacciones entre estacionalidad, historial previo y capacidad financiera.</li>
  <li><strong>Rigor metodológico y ausencia de fuga de datos:</strong> Excluir <code>duration</code>, mantener los valores extremos reales, encapsular la imputación y los filtros dentro de cada partición y validar con 5-Fold estratificado garantizó métricas honestas y consistentes entre validación cruzada ({best_cv['ROC AUC']:.3f}) y prueba ({best_test['ROC AUC']:.3f}).</li>
  <li><strong>Recomendaciones futuras:</strong> En un entorno productivo se recomienda calibrar el umbral de decisión en función del costo unitario por llamada y el margen financiero de cada depósito, así como incorporar variables macroeconómicas y una validación temporal por fechas exactas para monitorear cambios de comportamiento en el tiempo.</li>
</ul>

</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")

    cmd = [
        CHROME_BIN,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH.as_uri(),
    ]
    subprocess.run(cmd, check=True)
    print(f"PDF generado exitosamente en: {PDF_PATH}")


if __name__ == "__main__":
    generar_pdf()
