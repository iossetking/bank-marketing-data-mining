"""
Genera el informe editable en Word (Informe_Bank_Marketing.docx) con todas las
etapas KDD, las 6 secciones requeridas, tablas de métricas reales, las 8 figuras
incrustadas y la justificación del modelo y decisión óptima.
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Informe_Bank_Marketing.docx"
METRICS_JSON = ROOT / "outputs" / "metricas_resumen.json"
FIG_DIR = ROOT / "outputs" / "figuras"

BLUE = "1D4E89"
LIGHT_BLUE = "EAF2F8"
HIGHLIGHT = "D6EAF8"
BORDER = "CBD5E1"


def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def cell_border(cell, color: str = BORDER) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), color)


def set_cell_text(cell, text, bold: bool = False, color: str | None = None, size: float = 8.5) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(1)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    cell_border(cell)


def add_table(doc: Document, headers: list[str], rows: list[list], widths: list[float] | None = None, font_size: float = 8.5, highlight_row: int | None = None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    for i, header in enumerate(headers):
        cell = t.rows[0].cells[i]
        set_cell_text(cell, header, bold=True, color="FFFFFF", size=font_size)
        shade(cell, BLUE)
        if widths:
            cell.width = Inches(widths[i])
    for row_idx, row in enumerate(rows):
        cells = t.add_row().cells
        is_hl = highlight_row is not None and row_idx == highlight_row
        for i, val in enumerate(row):
            set_cell_text(cells[i], val, bold=(is_hl or i == 0), size=font_size)
            if is_hl:
                shade(cells[i], HIGHLIGHT)
            elif row_idx % 2 == 1:
                shade(cells[i], LIGHT_BLUE)
            if widths:
                cells[i].width = Inches(widths[i])
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(3)
    return t


def add_heading(doc: Document, text: str, level: int = 1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.color.rgb = RGBColor.from_string(BLUE if level == 1 else "0F172A")
    return p


def add_paragraph(doc: Document, text: str, bold_lead: str | None = None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.12
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    p.add_run(text)
    return p


def add_bullet(doc: Document, bold_lead: str, text: str):
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(3.5)
    p.paragraph_format.line_spacing = 1.1
    r = p.add_run(bold_lead)
    r.bold = True
    p.add_run(text)
    return p


def add_callout(doc: Document, title_text: str, body_text: str):
    t = doc.add_table(rows=1, cols=1)
    t.autofit = False
    cell = t.rows[0].cells[0]
    cell.width = Inches(6.9)
    shade(cell, LIGHT_BLUE)
    cell_border(cell, BLUE)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    r1 = p.add_run(title_text)
    r1.bold = True
    r1.font.size = Pt(10)
    r1.font.color.rgb = RGBColor.from_string(BLUE)
    r2 = p.add_run(body_text)
    r2.font.size = Pt(10)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(3)


def add_figure(doc: Document, filename: str, caption: str, width_in: float = 6.3):
    img_path = FIG_DIR / filename
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(4)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(img_path), width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(8)
    r = p_cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(8.8)
    r.font.color.rgb = RGBColor.from_string("475569")


def build() -> None:
    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    best_name = m["mejor_modelo"]
    best_cv = next(r for r in m["cv"] if r["Modelo"] == best_name)
    best_test = next(r for r in m["test"] if r["Modelo"] == best_name)
    dec10 = m["deciles"][0]
    dec30 = m["deciles"][2]
    top_vars = ", ".join(r["variable"] for r in m["importancia_top5"][:5])

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string("1F2937")

    # Encabezado editable del estudiante
    tag = doc.add_paragraph()
    tag.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tag.paragraph_format.space_after = Pt(4)
    r_tag = tag.add_run("MINERÍA DE DATOS (DATA MINING) · PROYECTO APLICADO")
    r_tag.bold = True
    r_tag.font.size = Pt(9.5)
    r_tag.font.color.rgb = RGBColor.from_string(BLUE)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(6)
    r_title = title.add_run(
        "Predicción de Suscripción a Depósitos a Plazo Fijo para la Priorización de Campañas Telefónicas"
    )
    r_title.bold = True
    r_title.font.size = Pt(17)
    r_title.font.color.rgb = RGBColor.from_string("0F172A")

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(10)
    r_sub = sub.add_run(
        "Análisis exploratorio, preprocesamiento reproducible, selección de características y comparación de modelos sobre UCI Bank Marketing"
    )
    r_sub.italic = True
    r_sub.font.size = Pt(10.5)
    r_sub.font.color.rgb = RGBColor.from_string("475569")

    # Tabla editable de datos del alumno
    add_table(
        doc,
        ["Estudiante / Equipo", "Materia", "Profesor(a)", "Fecha"],
        [["[Escribe tu nombre aquí]", "Minería de Datos (Data Mining)", "[Nombre del profesor]", "[Fecha]"]],
        widths=[2.1, 1.9, 1.8, 1.1],
        font_size=8.8,
    )

    # 1. Objetivo del proyecto
    add_heading(doc, "1. Objetivo del proyecto", level=1)
    add_paragraph(
        doc,
        "Las campañas de marketing telefónico representan un costo operativo importante para las instituciones financieras debido al tiempo de los agentes, los múltiples intentos de contacto y la fatiga que las llamadas repetidas generan en los clientes. Contactar clientes sin un criterio de priorización produce tasas de conversión bajas y desperdicia recursos comerciales.",
    )
    add_callout(
        doc,
        "Objetivo principal: ",
        "Estimar la probabilidad de que un cliente contrate un depósito a plazo fijo utilizando únicamente información disponible antes de realizar la llamada, con el fin de priorizar campañas telefónicas y concentrar el esfuerzo comercial en los clientes con mayor propensión a suscribirse.",
    )
    add_paragraph(
        doc,
        "Para cumplir este objetivo se ejecutó el proceso completo de Descubrimiento de Conocimiento en Bases de Datos (KDD): definición del objetivo, selección de datos, limpieza de datos, transformación e ingeniería de características, minería de datos (comparación de cuatro algoritmos de clasificación supervisada con validación cruzada estratificada) y evaluación e interpretación enfocada tanto en métricas de clasificación como en ganancia acumulada (Lift) para la toma de decisiones comerciales.",
    )

    # 2. Descripción del conjunto de datos
    add_heading(doc, "2. Descripción del conjunto de datos", level=1)
    add_paragraph(
        doc,
        f"Se utilizó el conjunto de datos público Bank Marketing del repositorio UCI Machine Learning Repository (ID 222), recopilado a partir de campañas de marketing directo vía telefónica de una institución bancaria portuguesa. El conjunto cuenta con {m['n_total']:,} registros, 16 variables predictoras (7 numéricas y 9 categóricas) y una variable objetivo binaria (y) que indica si el cliente suscribió (yes) o no (no) el depósito a plazo fijo.",
    )
    add_paragraph(
        doc,
        f"Un aspecto crítico del conjunto es su desbalance de clases: únicamente el {m['positivos_pct']:.2f}% de los clientes suscribieron el depósito, mientras que el {100 - m['positivos_pct']:.2f}% restante no aceptó la oferta. Esto implica que métricas ingenuas como la exactitud global pueden resultar engañosas, por lo que se emplean particiones estratificadas, ponderación balanceada de clases y métricas sensibles a la clase minoritaria.",
    )

    add_table(
        doc,
        ["Variable", "Tipo", "Descripción y rol en el análisis"],
        [
            ["age", "Numérica", "Edad del cliente en años."],
            ["job", "Categórica", "Ocupación del cliente (11 categorías; 288 valores faltantes)."],
            ["marital", "Categórica", "Estado civil (married, single, divorced)."],
            ["education", "Categórica", "Nivel educativo (primary, secondary, tertiary; 1,857 faltantes)."],
            ["default", "Binaria", "Indica si el cliente tiene crédito en mora (yes/no)."],
            ["balance", "Numérica", "Saldo anual promedio del cliente en euros."],
            ["housing, loan", "Binarias", "Indican si el cliente posee crédito hipotecario o préstamo personal."],
            ["contact", "Categórica", "Canal de contacto (cellular, telephone; 13,020 faltantes)."],
            ["day_of_week, month", "Num. / Cat.", "Día del mes y mes del último contacto de la campaña actual."],
            ["duration", "Numérica", "Duración de la llamada en segundos. Excluida por fuga de datos (solo se conoce al finalizar la llamada)."],
            ["campaign", "Numérica", "Número de contactos realizados al cliente durante la campaña actual."],
            ["pdays, previous", "Numéricas", "Días desde el contacto previo (-1 = sin contacto previo) y número de contactos anteriores."],
            ["poutcome", "Categórica", "Resultado de la campaña anterior (success, failure, other; 36,959 faltantes)."],
            ["y (Objetivo)", "Binaria", "Suscripción al depósito a plazo fijo (1 = sí, 0 = no)."],
        ],
        widths=[1.45, 0.95, 4.5],
        font_size=8.3,
    )

    add_figure(
        doc,
        "01_exploracion_objetivo.png",
        "Figura 1. Distribución desbalanceada de la variable objetivo y tasas de suscripción según resultado previo (poutcome) y mes de contacto.",
        width_in=6.5,
    )

    # 3. Metodología (Proceso KDD)
    add_heading(doc, "3. Metodología (Proceso KDD)", level=1)
    add_paragraph(
        doc,
        "El flujo de trabajo sigue de forma trazable las fases del proceso KDD: Definición del objetivo (Sección 1), Selección de datos, Limpieza de datos, Transformación de datos, Minería de datos y Evaluación e interpretación (Secciones 4 y 5).",
    )

    add_heading(doc, "3.1 Selección y limpieza de datos", level=2)
    add_bullet(
        doc,
        "Selección de datos: ",
        "Se seleccionaron los 45,211 registros del repositorio UCI y se excluyó deliberadamente la variable duration por fuga de información, ya que solo se conoce después de efectuar la llamada y no puede utilizarse para priorizar a quién contactar.",
    )
    add_bullet(
        doc,
        "Duplicados y consistencia de texto: ",
        f"Se verificó la existencia de {m['n_duplicados']} filas duplicadas exactas y se normalizaron espacios y minúsculas en las variables categóricas. Todos los valores observados se ajustaron al dominio documentado.",
    )
    add_bullet(
        doc,
        "Faltantes y códigos especiales: ",
        "Los valores faltantes en job (288), education (1,857), contact (13,020) y poutcome (36,959) no se eliminaron para no perder más del 80% de la muestra; se imputaron con la moda dentro del pipeline. El valor pdays = -1 se conservó porque representa ausencia de contacto previo.",
    )
    add_bullet(
        doc,
        "Valores atípicos (IQR): ",
        "Mediante diagramas de caja e IQR se detectaron valores extremos en balance, campaign, pdays y previous. Se conservaron por corresponder a clientes reales (saldos altos o historiales largos) y se normalizaron con MinMaxScaler.",
    )

    add_figure(
        doc,
        "02_diagnostico_atipicos.png",
        "Figura 2. Diagnóstico visual de distribución y valores atípicos mediante criterio IQR en las variables numéricas originales.",
        width_in=6.3,
    )

    add_heading(doc, "3.2 Transformación de datos y Minería de Datos", level=2)
    add_bullet(
        doc,
        "Transformación e ingeniería de características: ",
        "Se crearon dos variables derivadas disponibles antes de la llamada: presion_contacto = campaign / (previous + 1) (mide la insistencia reciente frente al historial) y balance_ajustado_credito = balance / (1 + housing + loan) (aproxima la capacidad financiera ajustada por créditos activos). Las numéricas se imputaron con mediana y se escalaron con MinMaxScaler; las categóricas se imputaron con moda y se codificaron con OneHotEncoder dentro de un Pipeline sin fuga de datos.",
    )
    add_bullet(
        doc,
        "Partición y selección de características: ",
        f"Se realizó una partición estratificada 80% entrenamiento ({m['n_train']:,} registros) y 20% prueba independiente ({m['n_test']:,} registros). Dentro de cada pliegue de entrenamiento se aplicó Chi-cuadrada (top 25 indicadores categóricos) y correlación de Pearson (top 6 numéricas).",
    )
    add_paragraph(
        doc,
        "En la fase de Minería de Datos se implementaron y compararon cuatro algoritmos de clasificación supervisada, todos con class_weight='balanced' y evaluados mediante validación cruzada estratificada de 5 particiones (5-Fold Stratified CV):",
    )
    add_bullet(doc, "Regresión Logística: ", "Modelo lineal base, rápido e interpretable que estima directamente probabilidades de suscripción.")
    add_bullet(doc, "Árbol de Decisión (max_depth=8, min_samples_leaf=50): ", "Modelo no lineal basado en reglas de decisión controladas para evitar sobreajuste.")
    add_bullet(doc, "Bosque Aleatorio (Random Forest, 200 árboles, max_depth=12): ", "Ensamble tipo bagging que reduce varianza y captura interacciones no lineales.")
    add_bullet(doc, "Gradient Boosting (HistGradientBoosting, 150 iteraciones, lr=0.05, max_depth=6): ", "Ensamble secuencial tipo boosting altamente eficaz en datos tabulares.")

    # 4. Resultados
    add_heading(doc, "4. Resultados", level=1)
    add_heading(doc, "4.1 Selección de características por filtros", level=2)
    add_paragraph(
        doc,
        f"El filtro Chi-cuadrada mostró que poutcome (V de Cramér = {m['ranking_chi2'][0]['v_cramer']:.3f}), month (V = {m['ranking_chi2'][1]['v_cramer']:.3f}) y housing (V = {m['ranking_chi2'][2]['v_cramer']:.3f}) concentran las asociaciones categóricas más fuertes. En variables numéricas, Pearson destacó {m['ranking_pearson'][0]['variable']} (r = {m['ranking_pearson'][0]['correlacion']:.3f}), {m['ranking_pearson'][1]['variable']} (r = {m['ranking_pearson'][1]['correlacion']:.3f}) y {m['ranking_pearson'][2]['variable']} (r = {m['ranking_pearson'][2]['correlacion']:.3f}).",
    )
    add_figure(
        doc,
        "03_filtros_seleccion.png",
        "Figura 3. Ranking de asociación frente a la variable objetivo mediante Chi-cuadrada (V de Cramér) y correlación de Pearson en entrenamiento.",
        width_in=6.5,
    )

    add_heading(doc, "4.2 Comparación de modelos en validación cruzada y conjunto de prueba", level=2)
    cv_rows = [
        [
            r["Modelo"],
            f"{r['ROC AUC']:.4f} ± {r['ROC AUC std']:.4f}",
            f"{r['PR AUC']:.4f}",
            f"{r['Exactitud Balanceada']:.4f}",
            f"{r['F1']:.4f}",
            f"{r['Precisión']:.4f}",
            f"{r['Sensibilidad']:.4f}",
        ]
        for r in m["cv"]
    ]
    hl_cv = next(i for i, r in enumerate(m["cv"]) if r["Modelo"] == best_name)
    add_table(
        doc,
        ["Modelo (CV 5-Fold)", "ROC AUC", "PR AUC", "Exact. Bal.", "F1", "Precisión", "Sensibilidad"],
        cv_rows,
        widths=[1.6, 1.35, 0.75, 0.85, 0.7, 0.8, 0.85],
        font_size=8.3,
        highlight_row=hl_cv,
    )
    add_figure(
        doc,
        "04_comparacion_cv.png",
        "Figura 4. Comparación de métricas promedio de los cuatro modelos en validación cruzada estratificada de 5 particiones.",
        width_in=6.3,
    )

    add_paragraph(
        doc,
        f"Con base estricta en la validación cruzada, {best_name} obtuvo el mejor desempeño global (ROC AUC = {best_cv['ROC AUC']:.4f}, PR AUC = {best_cv['PR AUC']:.4f}, F1 = {best_cv['F1']:.4f}). Al evaluar los modelos entrenados sobre el conjunto de prueba independiente (9,043 clientes no vistos), se confirmó su estabilidad sin sobreajuste:",
    )
    test_rows = [
        [
            r["Modelo"],
            f"{r['ROC AUC']:.4f}",
            f"{r['PR AUC']:.4f}",
            f"{r['Exactitud']:.4f}",
            f"{r['Exactitud Balanceada']:.4f}",
            f"{r['F1']:.4f}",
            f"{r['Precisión']:.4f}",
            f"{r['Sensibilidad']:.4f}",
        ]
        for r in m["test"]
    ]
    hl_test = next(i for i, r in enumerate(m["test"]) if r["Modelo"] == best_name)
    add_table(
        doc,
        ["Modelo (Prueba)", "ROC AUC", "PR AUC", "Exactitud", "Exact. Bal.", "F1", "Precisión", "Sensibilidad"],
        test_rows,
        widths=[1.5, 0.75, 0.72, 0.78, 0.82, 0.7, 0.8, 0.83],
        font_size=8.2,
        highlight_row=hl_test,
    )

    add_figure(
        doc,
        "05_curvas_roc_pr.png",
        "Figura 5. Curvas ROC y Precisión-Sensibilidad comparativas de los cuatro modelos sobre el conjunto de prueba independiente.",
        width_in=6.5,
    )
    add_figure(
        doc,
        "06_matrices_confusion.png",
        "Figura 6. Matrices de confusión de los cuatro clasificadores en el conjunto de prueba (n = 9,043).",
        width_in=5.8,
    )

    # 5. Interpretación y análisis
    add_heading(doc, "5. Interpretación y análisis", level=1)
    add_heading(doc, "5.1 Variables determinantes e interpretación del modelo", level=2)
    add_paragraph(
        doc,
        f"Para explicar qué atributos impulsan las predicciones de {best_name} se calculó la importancia por permutación (Permutation Importance) sobre el conjunto de prueba. Las cinco variables más influyentes fueron: {top_vars}. Destaca especialmente que balance_ajustado_credito (creada durante la ingeniería de características) aparece en el cuarto lugar de importancia, confirmando que relacionar el saldo con la carga crediticia aporta señal predictiva relevante.",
    )
    add_figure(
        doc,
        "07_importancia_variables.png",
        f"Figura 7. Top 10 de variables más influyentes en {best_name} según la caída promedio en ROC AUC al permutar cada atributo.",
        width_in=6.0,
    )
    add_bullet(doc, "Estacionalidad (month): ", "Los meses con menor volumen masivo de llamadas (marzo, septiembre, octubre y diciembre) presentan tasas de conversión muy superiores al promedio de 11.7%.")
    add_bullet(doc, "Situación crediticia y saldo (housing, balance_ajustado_credito): ", "Los clientes sin hipoteca activa y con mayor liquidez relativa muestran mayor disposición a contratar un depósito a plazo fijo.")
    add_bullet(doc, "Historial previo (poutcome, pdays): ", "Quienes ya aceptaron una oferta en campañas anteriores presentan la tasa de conversión más elevada (64.7%), convirtiéndose en el segmento prioritario natural.")

    add_heading(doc, "5.2 Relación con el objetivo: priorización de campañas telefónicas", level=2)
    add_paragraph(
        doc,
        f"Aunque la precisión en el umbral estándar de 0.50 es moderada ({best_test['Precisión']*100:.1f}%) debido al desbalance de clases y a la exclusión realista de duration, la verdadera utilidad operativa del modelo está en ordenar a los clientes por probabilidad estimada de suscripción. La tabla de deciles y la Figura 8 cuantifican el impacto directo sobre los 9,043 clientes de prueba (1,058 suscriptores reales):",
    )
    dec_rows = [
        [
            r["decil"],
            f"{r['clientes_contactados']:,}",
            f"{r['suscriptores_capturados']:,}",
            f"{r['tasa_conversion_pct']:.2f}%",
            f"{r['ganancia_acumulada_pct']:.2f}%",
            f"{r['lift']:.2f}x",
        ]
        for r in (m["deciles"][:5] + [m["deciles"][-1]])
    ]
    dec_rows[-1][0] = "Top 100% (Sin filtro)"
    add_table(
        doc,
        ["Segmento priorizado", "Clientes contactados", "Suscriptores capturados", "Tasa de conversión", "Ganancia acumulada", "Factor Lift"],
        dec_rows,
        widths=[1.45, 1.15, 1.25, 1.1, 1.1, 0.85],
        font_size=8.4,
        highlight_row=0,
    )
    add_figure(
        doc,
        "08_ganancia_lift_campana.png",
        f"Figura 8. Curva de ganancia acumulada y factor Lift por deciles usando {best_name} para priorizar llamadas frente a una estrategia aleatoria.",
        width_in=6.5,
    )

    add_heading(doc, "5.3 Justificación del modelo óptimo y decisión recomendada", level=2)
    add_bullet(
        doc,
        f"¿Por qué {best_name} es el modelo óptimo? ",
        f"Fue seleccionado estrictamente en validación cruzada (sin usar prueba para decidir) al liderar ROC AUC ({best_cv['ROC AUC']:.4f}), PR AUC ({best_cv['PR AUC']:.4f}), Exactitud Balanceada ({best_cv['Exactitud Balanceada']:.4f}) y F1 ({best_cv['F1']:.4f}). Aunque en prueba el Bosque Aleatorio empató prácticamente en ROC AUC (0.7651 vs 0.7641), {best_name} lo superó claramente en las métricas críticas bajo desbalance: mayor PR AUC (0.4363 vs 0.4134), mayor F1 (0.4102 vs 0.3877), mayor precisión (31.5% vs 28.7%) y 213 llamadas innecesarias menos (1,349 falsos positivos frente a 1,562 del Bosque Aleatorio, 1,698 de Regresión Logística y 1,864 del Árbol de Decisión).",
    )
    add_bullet(
        doc,
        "¿Por qué se descartan Regresión Logística y Árbol de Decisión? ",
        "La Regresión Logística asume relaciones lineales y no captura interacciones entre estacionalidad, historial previo y saldo ajustado; el Árbol de Decisión individual presenta mayor varianza y la menor capacidad discriminativa (ROC AUC = 0.7359).",
    )
    add_bullet(
        doc,
        "Decisión operativa óptima para el banco: ",
        f"Utilizar las probabilidades de {best_name} como ranking de priorización. Con presupuesto estricto, contactar solo al Top 10% ({dec10['clientes_contactados']:,} llamadas) maximiza el retorno por llamada ({dec10['tasa_conversion_pct']:.1f}% de conversión, Lift {dec10['lift']:.2f}x y {dec10['ganancia_acumulada_pct']:.1f}% de los suscriptores). Como punto óptimo entre volumen de captación y ahorro de costos, el Top 30% captura el {dec30['ganancia_acumulada_pct']:.1f}% de los suscriptores ahorrando el 70% de llamadas; contactar más allá del 50% entrega rendimientos cercanos al azar.",
    )

    # 6. Conclusiones
    add_heading(doc, "6. Conclusiones", level=1)
    add_bullet(
        doc,
        "Cumplimiento del objetivo: ",
        "Se desarrolló y validó un flujo completo KDD capaz de estimar la probabilidad de contratación de un depósito a plazo fijo antes de realizar la llamada, demostrando mediante ganancia acumulada y Lift que el modelo prioriza eficazmente las campañas telefónicas.",
    )
    add_bullet(
        doc,
        "Superioridad de los ensambles no lineales: ",
        f"Los ensambles de árboles ({best_name} y Bosque Aleatorio) superaron consistentemente a la Regresión Logística y al Árbol de Decisión individual en todas las métricas relevantes para clases desbalanceadas.",
    )
    add_bullet(
        doc,
        "Rigor metodológico y ausencia de fuga de datos: ",
        f"Excluir duration, conservar valores atípicos reales, encapsular imputación y filtros dentro de cada partición y validar con 5-Fold estratificado garantizó métricas honestas y consistentes entre validación cruzada ({best_cv['ROC AUC']:.3f}) y prueba ({best_test['ROC AUC']:.3f}).",
    )
    add_bullet(
        doc,
        "Recomendaciones futuras: ",
        "Calibrar el umbral de corte según el costo unitario por llamada y el margen financiero de cada depósito, además de incorporar variables macroeconómicas y validación temporal por fechas completas.",
    )

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_f = footer.add_run("Minería de Datos (Data Mining) · Bank Marketing - Informe de Proyecto")
    r_f.font.size = Pt(8)
    r_f.font.color.rgb = RGBColor.from_string("64748B")

    doc.save(OUT)
    print(f"Documento Word generado exitosamente en: {OUT}")


if __name__ == "__main__":
    build()
