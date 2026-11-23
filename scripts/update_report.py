"""
Genera el documento Word (Informe_Bank_Marketing.docx) con redacción clara y
natural estilo proyecto universitario de Minería de Datos, cubriendo las 6
etapas KDD, los 6 apartados solicitados y las 6 gráficas principales.
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

HEADER_BG = "2B547E"
ROW_ALT = "F2F6FA"
BORDER = "CCCCCC"


def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_borders(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        el = borders.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), BORDER)


def add_table(doc: Document, headers: list[str], rows: list[list], widths: list[float] | None = None, size: float = 9):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(size)
        r.font.color.rgb = RGBColor(255, 255, 255)
        shade(c, HEADER_BG)
        set_borders(c)
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        if widths:
            c.width = Inches(widths[i])
    for idx, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            c = cells[i]
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(str(val))
            r.font.size = Pt(size)
            if idx % 2 == 1:
                shade(c, ROW_ALT)
            set_borders(c)
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if widths:
                c.width = Inches(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_title_h1(doc: Document, text: str):
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.color.rgb = RGBColor.from_string(HEADER_BG)


def add_title_h2(doc: Document, text: str):
    p = doc.add_paragraph(style="Heading 2")
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.font.color.rgb = RGBColor(30, 30, 30)


def add_p(doc: Document, text: str, bold_start: str | None = None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    if bold_start:
        rb = p.add_run(bold_start)
        rb.bold = True
    p.add_run(text)


def add_bullet(doc: Document, bold_start: str, text: str):
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.12
    rb = p.add_run(bold_start)
    rb.bold = True
    p.add_run(text)


def add_fig(doc: Document, fname: str, caption: str, width: float = 6.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG_DIR / fname), width=Inches(width))

    pc = doc.add_paragraph()
    pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pc.paragraph_format.space_after = Pt(8)
    rc = pc.add_run(caption)
    rc.italic = True
    rc.font.size = Pt(9)
    rc.font.color.rgb = RGBColor(90, 90, 90)


def build() -> None:
    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    mejor = m["mejor_modelo"]
    cv_mejor = next(r for r in m["cv"] if r["Modelo"] == mejor)
    test_mejor = next(r for r in m["test"] if r["Modelo"] == mejor)
    p10 = m["priorizacion"][0]
    p20 = m["priorizacion"][1]
    p30 = m["priorizacion"][2]

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.85)
    sec.bottom_margin = Inches(0.85)
    sec.left_margin = Inches(0.9)
    sec.right_margin = Inches(0.9)

    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10.5)

    # Portada sencilla editable
    p_tit = doc.add_paragraph()
    p_tit.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tit.paragraph_format.space_after = Pt(4)
    r_tit = p_tit.add_run("Proyecto de Minería de Datos: Bank Marketing")
    r_tit.bold = True
    r_tit.font.size = Pt(16)
    r_tit.font.color.rgb = RGBColor.from_string(HEADER_BG)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(10)
    r_sub = p_sub.add_run("Predicción de contratación de depósitos a plazo fijo para priorizar campañas telefónicas")
    r_sub.italic = True
    r_sub.font.size = Pt(11)

    add_table(
        doc,
        ["Alumno(s)", "Materia", "Profesor", "Fecha"],
        [["[Tu nombre aquí]", "Minería de Datos", "[Nombre del profesor]", "[Fecha]"]],
        widths=[2.0, 1.6, 1.9, 1.2],
        size=9,
    )

    # 1. Objetivo del proyecto
    add_title_h1(doc, "1. Objetivo del proyecto")
    add_p(
        doc,
        "En las campañas de marketing telefónico de un banco, llamar a todos los clientes cuesta tiempo y dinero, y muchas veces la mayoría de las personas rechazan la oferta. Si se logra saber antes de marcar qué clientes tienen más probabilidad de aceptar un depósito a plazo fijo, el equipo de ventas puede llamar primero a esas personas y aprovechar mejor sus recursos.",
    )
    add_p(
        doc,
        "Estimar si un cliente contratará un depósito a plazo fijo utilizando los datos disponibles antes de realizar la llamada, para así priorizar las campañas telefónicas.",
        bold_start="Objetivo: ",
    )
    add_p(
        doc,
        "Para lograrlo se siguieron las etapas del proceso de Minería de Datos (KDD): definición del objetivo, selección de datos, limpieza, transformación, aplicación de modelos de clasificación, y evaluación e interpretación de resultados.",
    )

    # 2. Descripción del conjunto de datos
    add_title_h1(doc, "2. Descripción del conjunto de datos")
    add_p(
        doc,
        f"Se trabajó con el dataset Bank Marketing obtenido del repositorio UCI Machine Learning (ID 222), el cual recopila llamadas telefónicas de campañas de marketing de un banco en Portugal. Tiene {m['n_total']:,} registros y 16 variables predictoras (7 numéricas y 9 categóricas), además de la variable objetivo y, que indica si el cliente contrató el depósito (yes) o no (no).",
    )
    add_p(
        doc,
        f"Un detalle importante que se observó desde el inicio es que los datos están desbalanceados: solo el {m['positivos_pct']:.1f}% de los clientes dijeron que sí, mientras que el {100 - m['positivos_pct']:.1f}% dijeron que no. Por eso no basta con mirar la exactitud (accuracy) normal, sino que hay que usar particiones estratificadas y métricas que tomen en cuenta la clase minoritaria.",
    )

    add_table(
        doc,
        ["Variable", "Tipo", "Qué representa"],
        [
            ["age", "Numérica", "Edad del cliente en años."],
            ["job", "Categórica", "Trabajo u ocupación (288 datos faltantes)."],
            ["marital", "Categórica", "Estado civil (married, single, divorced)."],
            ["education", "Categórica", "Nivel de estudios (1,857 datos faltantes)."],
            ["default", "Categórica", "Si el cliente tiene crédito en mora (yes/no)."],
            ["balance", "Numérica", "Saldo anual promedio en euros."],
            ["housing / loan", "Categóricas", "Si tiene crédito hipotecario o préstamo personal."],
            ["contact", "Categórica", "Medio de contacto: celular o teléfono fijo (13,020 faltantes)."],
            ["day_of_week / month", "Num. / Cat.", "Día del mes y mes del último contacto."],
            ["duration", "Numérica", "Duración de la llamada en segundos (se excluyó del modelo)."],
            ["campaign", "Numérica", "Número de llamadas hechas al cliente en esta campaña."],
            ["pdays / previous", "Numéricas", "Días desde el contacto anterior (-1 si nunca se le llamó) y llamadas previas."],
            ["poutcome", "Categórica", "Resultado de la campaña anterior (36,959 faltantes)."],
            ["y", "Objetivo", "Si contrató el depósito a plazo fijo (yes = 1, no = 0)."],
        ],
        widths=[1.5, 1.0, 4.2],
        size=8.5,
    )

    add_fig(
        doc,
        "01_distribucion_y.png",
        "Figura 1. Distribución de la variable objetivo (y) y tasa de suscripción según el resultado de la campaña previa.",
        width=5.9,
    )

    # 3. Metodología
    add_title_h1(doc, "3. Metodología")
    add_title_h2(doc, "3.1 Selección y limpieza de datos")
    add_bullet(
        doc,
        "Exclusión de duration (Selección): ",
        "La duración de la llamada solo se conoce hasta que la llamada ya terminó. Si la dejáramos en el modelo, los resultados saldrían artificialmente altos pero el modelo no serviría en la práctica para decidir a quién llamar antes de marcar.",
    )
    add_bullet(
        doc,
        "Duplicados y texto (Limpieza): ",
        f"Se revisó si había filas repetidas exactas y el resultado fue {m['duplicados']}. También se quitaron espacios extra y se pasaron los textos a minúsculas para evitar categorías duplicadas.",
    )
    add_bullet(
        doc,
        "Valores faltantes y códigos especiales: ",
        "En lugar de borrar las filas con datos faltantes en job, education, contact y poutcome (lo cual eliminaría más del 80% de los datos), se imputaron con la moda dentro del pipeline. El valor -1 en pdays se dejó tal cual porque significa que el cliente no había sido contactado antes.",
    )
    add_bullet(
        doc,
        "Valores atípicos: ",
        "Al revisar los diagramas de caja (Figura 2) se vieron valores extremos en balance, campaign, pdays y previous. No se eliminaron porque en un banco sí existen clientes con mucho saldo o varias llamadas; mejor se conservan y se escalan.",
    )

    add_fig(
        doc,
        "02_boxplots.png",
        "Figura 2. Diagramas de caja para revisar la distribución y valores extremos de las variables numéricas.",
        width=5.9,
    )

    add_title_h2(doc, "3.2 Transformación de datos y selección de características")
    add_p(
        doc,
        "Antes de entrenar los modelos se crearon dos variables nuevas a partir de las que ya existían (ambas conocidas antes de llamar): presion_contacto = campaign / (previous + 1), para ver qué tanto se ha insistido en esta campaña comparado con el historial, y balance_ajustado_credito = balance / (1 + housing + loan), para aproximar el saldo disponible tomando en cuenta si el cliente tiene créditos activos.",
    )
    add_p(
        doc,
        f"Después se dividió el conjunto de forma estratificada en 80% entrenamiento ({m['n_train']:,} filas) y 20% prueba ({m['n_test']:,} filas). Dentro de un Pipeline de scikit-learn se imputaron numéricas con la mediana y se escalaron con MinMaxScaler, y las categóricas se imputaron con la moda y se codificaron con OneHotEncoder. También se aplicaron filtros de selección dentro del pipeline: Chi-cuadrada para variables categóricas y correlación de Pearson para numéricas.",
    )

    add_title_h2(doc, "3.3 Minería de Datos (métodos implementados)")
    add_p(
        doc,
        "Para no quedarnos con un solo algoritmo, se probaron y compararon tres modelos clásicos de clasificación supervisada vistos en clase, usando class_weight='balanced' en todos para compensar el desbalance de la variable objetivo y validación cruzada estratificada de 5 particiones:",
    )
    add_bullet(doc, "Regresión Logística: ", "Es un modelo lineal sencillo y fácil de interpretar, útil como punto de partida.")
    add_bullet(doc, "Árbol de Decisión: ", "Divide los datos con reglas de decisión (se limitó la profundidad máxima a 6 para evitar sobreajuste).")
    add_bullet(doc, "Random Forest (Bosque Aleatorio): ", "Combina 150 árboles de decisión entrenados con distintas muestras para mejorar la estabilidad y captar relaciones no lineales.")

    # 4. Resultados
    add_title_h1(doc, "4. Resultados")
    add_p(
        doc,
        f"En los filtros de selección sobre entrenamiento (Figura 3), las variables categóricas con mayor relación con la suscripción fueron poutcome (V de Cramér = {m['ranking_chi2'][0]['v_cramer']:.3f}), month ({m['ranking_chi2'][1]['v_cramer']:.3f}) y housing ({m['ranking_chi2'][2]['v_cramer']:.3f}). Entre las numéricas destacaron {m['ranking_pearson'][0]['variable']} (r = {m['ranking_pearson'][0]['correlacion']:.3f}), {m['ranking_pearson'][1]['variable']} (r = {m['ranking_pearson'][1]['correlacion']:.3f}) y {m['ranking_pearson'][2]['variable']} (r = {m['ranking_pearson'][2]['correlacion']:.3f}).",
    )
    add_fig(
        doc,
        "03_filtros.png",
        "Figura 3. Resultados de los filtros de selección Chi-cuadrada (categóricas) y Pearson (numéricas).",
        width=6.0,
    )

    add_p(doc, "En la siguiente tabla y en la Figura 4 se muestran los resultados promedio de la validación cruzada de 5 particiones para los tres modelos:")
    add_table(
        doc,
        ["Modelo (Validación cruzada)", "ROC AUC", "Exact. Balanceada", "F1", "Precisión", "Sensibilidad"],
        [
            [r["Modelo"], f"{r['ROC AUC']:.3f} (±{r['ROC AUC std']:.3f})", f"{r['Exactitud Balanceada']:.3f}", f"{r['F1']:.3f}", f"{r['Precisión']:.3f}", f"{r['Sensibilidad']:.3f}"]
            for r in m["cv"]
        ],
        widths=[1.75, 1.25, 1.15, 0.8, 0.85, 0.9],
        size=8.8,
    )
    add_fig(
        doc,
        "04_comparacion_modelos.png",
        "Figura 4. Comparación del desempeño de los tres modelos en validación cruzada.",
        width=5.8,
    )

    add_p(doc, "Al evaluar los modelos entrenados en el conjunto de prueba independiente (9,043 clientes que los modelos nunca vieron), se obtuvieron las siguientes métricas:")
    add_table(
        doc,
        ["Modelo (Conjunto de prueba)", "ROC AUC", "Exactitud", "Exact. Balanceada", "F1", "Precisión", "Sensibilidad"],
        [
            [r["Modelo"], f"{r['ROC AUC']:.3f}", f"{r['Exactitud']:.3f}", f"{r['Exactitud Balanceada']:.3f}", f"{r['F1']:.3f}", f"{r['Precisión']:.3f}", f"{r['Sensibilidad']:.3f}"]
            for r in m["test"]
        ],
        widths=[1.65, 0.8, 0.8, 1.05, 0.75, 0.8, 0.85],
        size=8.6,
    )
    add_fig(
        doc,
        "05_roc_y_confusion.png",
        f"Figura 5. Curvas ROC comparativas en prueba y matriz de confusión del mejor modelo ({mejor}).",
        width=6.1,
    )

    # 5. Interpretación y análisis
    add_title_h1(doc, "5. Interpretación y análisis")
    add_title_h2(doc, "5.1 ¿Por qué Random Forest fue el mejor modelo?")
    add_p(
        doc,
        f"De los tres algoritmos evaluados, {mejor} obtuvo los mejores resultados tanto en validación cruzada (ROC AUC = {cv_mejor['ROC AUC']:.3f}, F1 = {cv_mejor['F1']:.3f}) como en el conjunto de prueba (ROC AUC = {test_mejor['ROC AUC']:.3f}, Exactitud balanceada = {test_mejor['Exactitud Balanceada']:.3f}, F1 = {test_mejor['F1']:.3f}).",
    )
    arbol_test = next(r for r in m["test"] if r["Modelo"] == "Árbol de Decisión")
    add_p(
        doc,
        f"La razón es que la Regresión Logística solo busca relaciones lineales entre las variables y el resultado, por lo que le cuesta captar combinaciones de factores; mientras que un solo Árbol de Decisión es más inestable y dio el ROC AUC más bajo ({arbol_test['ROC AUC']:.3f}) y menor sensibilidad ({arbol_test['Sensibilidad']:.3f}). En cambio, Random Forest promedia muchos árboles distintos, lo que reduce errores y logra separar mejor a los clientes que sí contratan sin sobreajustar (sus métricas en validación cruzada y prueba son muy parecidas).",
    )

    add_title_h2(doc, "5.2 Variables importantes y aplicación a la priorización de llamadas")
    add_p(
        doc,
        "Al revisar la importancia de variables del Random Forest (Figura 6, izquierda), se observa que los factores que más influyen en la predicción son el resultado de campañas anteriores (poutcome_success), el historial de contacto (pdays, presion_contacto), la situación financiera (housing, balance y la variable creada balance_ajustado_credito) y el mes de la llamada.",
    )
    add_p(
        doc,
        "Para comprobar si el modelo realmente cumple con el objetivo de priorizar campañas telefónicas, ordenamos a los 9,043 clientes de prueba de mayor a menor según la probabilidad que calculó Random Forest y revisamos cuántos clientes contratan si solo llamamos a los mejores porcentajes:",
    )

    add_table(
        doc,
        ["Grupo de clientes contactados", "Llamadas", "Suscriptores logrados", "Tasa de éxito", "% del total capturado", "Mejora vs azar"],
        [
            [r["grupo"], f"{r['llamadas']:,}", f"{r['suscriptores']:,}", f"{r['tasa_exito_pct']:.1f}%", f"{r['captura_total_pct']:.1f}%", f"{r['mejora_vs_azar']:.2f}x"]
            for r in m["priorizacion"]
        ],
        widths=[1.75, 0.85, 1.2, 0.95, 1.1, 0.85],
        size=8.6,
    )
    add_fig(
        doc,
        "06_importancia_y_priorizacion.png",
        "Figura 6. Variables más importantes del Random Forest y tasa de suscripción obtenida al priorizar llamadas.",
        width=6.1,
    )
    add_p(
        doc,
        f"Esto demuestra cuál es la decisión más práctica para el banco: sin modelo, la tasa de éxito al llamar al azar es de 11.7%. Si se usa Random Forest para llamar primero al Top 10% ({p10['llamadas']:,} clientes), la tasa de éxito sube a {p10['tasa_exito_pct']:.1f}% (casi {p10['mejora_vs_azar']:.1f} veces mejor que al azar) y se consigue el {p10['captura_total_pct']:.1f}% de todos los clientes que iban a contratar. Si el banco tiene capacidad para contactar al Top 30%, logra capturar el {p30['captura_total_pct']:.1f}% de los suscriptores ahorrándose el 70% de las llamadas.",
    )

    # 6. Conclusiones
    add_title_h1(doc, "6. Conclusiones")
    add_bullet(
        doc,
        "Cumplimiento del objetivo: ",
        f"Se logró construir un modelo capaz de estimar la probabilidad de que un cliente contrate un depósito a plazo fijo antes de llamarlo. Al ordenar a los clientes con {mejor}, el banco puede concentrar sus llamadas en el Top 10%–30% y obtener tasas de conversión entre {p30['tasa_exito_pct']:.1f}% y {p10['tasa_exito_pct']:.1f}% en lugar del 11.7% promedio.",
    )
    add_bullet(
        doc,
        "Importancia del preprocesamiento: ",
        "Excluir la variable duration evitó resultados engañosos, y meter la imputación, el escalamiento y los filtros dentro de un Pipeline evitó fuga de información entre entrenamiento y prueba. Además, las dos variables creadas (presion_contacto y balance_ajustado_credito) resultaron útiles para el modelo.",
    )
    add_bullet(
        doc,
        "Selección del mejor modelo: ",
        f"Comparar tres algoritmos permitió comprobar que {mejor} supera a la Regresión Logística y al Árbol de Decisión individual en ROC AUC, exactitud balanceada y F1, manteniendo un comportamiento estable en prueba.",
    )
    add_bullet(
        doc,
        "Trabajo futuro: ",
        "Como mejora se podría ajustar el umbral de probabilidad según cuánto le cuesta al banco cada llamada y cuánto gana por cada depósito contratado, para definir el corte exacto de llamadas en cada campaña.",
    )

    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rf = footer.add_run("Proyecto de Minería de Datos - Bank Marketing")
    rf.font.size = Pt(8.5)
    rf.font.color.rgb = RGBColor(120, 120, 120)

    doc.save(OUT)
    print(f"Word generado correctamente en: {OUT}")


if __name__ == "__main__":
    build()
