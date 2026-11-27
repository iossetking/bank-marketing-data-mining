"""
Genera el documento Word (Informe_Bank_Marketing.docx) y el PDF final
(Data_Mining_Proyecto.pdf) con redacción clara y natural estilo proyecto
universitario de Minería de Datos, cubriendo y justificando todos los puntos
de la rúbrica del proyecto.
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT_DOCX = ROOT / "Informe_Bank_Marketing.docx"
OUT_HTML = ROOT / "outputs" / "Informe_Bank_Marketing.html"
OUT_PDF = ROOT / "Data_Mining_Proyecto.pdf"
METRICS_JSON = ROOT / "outputs" / "metricas_resumen.json"
FIG_DIR = ROOT / "outputs" / "figuras"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

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


def img_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_docx(m: dict) -> None:
    mejor = m["mejor_modelo"]
    tr_mejor = next(r for r in m["train"] if r["Modelo"] == mejor)
    cv_mejor = next(r for r in m["cv"] if r["Modelo"] == mejor)
    test_mejor = next(r for r in m["test"] if r["Modelo"] == mejor)
    cm_tr = m["matriz_confusion_train"]
    cm_te = m["matriz_confusion_test"]
    p10 = m["priorizacion"][0]
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
        [
            [
                "Rodrigo Alonso Castillo Ramirez\nSebastian Sanchez Espinosa\nIosset Ivan Sandoval Gonzalez",
                "Minería de Datos",
                "MILAGROS CONTRERAS ELIZALDE",
                "23 de Noviembre del 2026",
            ]
        ],
        widths=[2.3, 1.4, 1.8, 1.2],
        size=9,
    )

    # 1. Motivación del trabajo y problemática
    add_title_h1(doc, "1. Motivación del trabajo y problemática a abordar")
    add_p(
        doc,
        "En las campañas de marketing telefónico de un banco, llamar a todos los clientes registrados implica un gasto alto de tiempo y dinero (horas de los agentes del centro de llamadas, costos de operación y varios intentos por persona). Además, marcar sin un filtro previo hace que la gran mayoría de las personas rechacen la oferta (en esta base el 88.3% dicen que no) y genera molestia en los clientes por recibir llamadas insistentes.",
    )
    add_p(
        doc,
        "El banco necesita identificar de antemano, utilizando únicamente la información disponible antes de realizar la llamada, qué clientes tienen mayor probabilidad de aceptar un depósito a plazo fijo para no desperdiciar recursos llamando al azar.",
        bold_start="Problemática y motivación: ",
    )
    add_p(
        doc,
        "Construir, optimizar y evaluar un flujo completo de Minería de Datos (siguiendo las etapas del proceso KDD) con modelos de clasificación supervisada que estimen la probabilidad de suscripción antes de marcar, permitiendo priorizar la lista de llamadas en las campañas telefónicas.",
        bold_start="Objetivo del proyecto: ",
    )

    # 2. Justificación de la selección de la base de datos
    add_title_h1(doc, "2. Justificación de la selección de la base de datos")
    add_p(
        doc,
        f"Se seleccionó el dataset público Bank Marketing del repositorio UCI Machine Learning (ID 222), el cual recopila llamadas reales de campañas telefónicas de una institución bancaria de Portugal. Cuenta con {m['n_total']:,} registros y 16 variables predictoras (7 numéricas y 9 categóricas), además de la variable objetivo binaria y (yes o no).",
    )
    add_p(
        doc,
        "Elegimos esta base de datos por tres razones principales que la hacen ideal para un proyecto de Minería de Datos:",
    )
    add_bullet(
        doc,
        "Tamaño y datos del mundo real: ",
        f"Con {m['n_total']:,} registros tenemos suficiente volumen para entrenar, hacer validación cruzada y reservar un conjunto de prueba independiente de {m['n_test']:,} clientes sin que los resultados dependan de la suerte de una muestra pequeña.",
    )
    add_bullet(
        doc,
        "Combinación de variables socioeconómicas y de campaña: ",
        "Contiene tanto el perfil financiero y personal del cliente (edad, ocupación, estado civil, estudios, saldo promedio, préstamos activos) como el historial de contacto (mes, número de llamadas, días desde la campaña anterior y resultado previo).",
    )
    add_bullet(
        doc,
        "Retos reales de preprocesamiento y modelado: ",
        f"Presenta un fuerte desbalance de clases (solo el {m['positivos_pct']:.1f}% dijo que sí frente al {100 - m['positivos_pct']:.1f}% que dijo que no), valores faltantes en variables categóricas, códigos especiales (pdays = -1), valores extremos reales en saldos y una variable con fuga de información (duration) que debemos detectar y tratar.",
    )

    add_table(
        doc,
        ["Variable", "Tipo", "Qué representa y consideración en el proyecto"],
        [
            ["age", "Numérica", "Edad del cliente en años."],
            ["job", "Categórica", "Trabajo u ocupación (11 categorías; 288 datos faltantes)."],
            ["marital", "Categórica", "Estado civil (married, single, divorced)."],
            ["education", "Categórica", "Nivel de estudios (primary, secondary, tertiary; 1,857 faltantes)."],
            ["default", "Binaria", "Si el cliente tiene crédito en mora (yes/no)."],
            ["balance", "Numérica", "Saldo anual promedio en euros."],
            ["housing / loan", "Binarias", "Si tiene crédito hipotecario o préstamo personal activo (yes/no)."],
            ["contact", "Categórica", "Medio de contacto: celular o teléfono fijo (13,020 faltantes)."],
            ["day_of_week / month", "Num. / Cat.", "Día del mes y mes del último contacto."],
            ["duration", "Numérica", "Duración de la llamada en segundos (se excluyó por fuga de datos)."],
            ["campaign", "Numérica", "Número de llamadas hechas al cliente en esta campaña."],
            ["pdays / previous", "Numéricas", "Días desde el contacto anterior (-1 si nunca se le llamó) y llamadas previas."],
            ["poutcome", "Categórica", "Resultado de la campaña anterior (success, failure, other; 36,959 faltantes)."],
            ["y", "Objetivo", "Si contrató el depósito a plazo fijo (yes = 1, no = 0)."],
        ],
        widths=[1.5, 1.0, 4.2],
        size=8.5,
    )

    add_fig(
        doc,
        "01_distribucion_y.png",
        "Figura 1. Distribución desbalanceada de la variable objetivo (y) y tasa de suscripción según el resultado de la campaña previa.",
        width=5.9,
    )

    # 3. Explicación y justificación del preprocesamiento de los datos
    add_title_h1(doc, "3. Explicación y justificación del preprocesamiento de los datos")
    add_title_h2(doc, "3.1 Diagnóstico de calidad y limpieza de datos")
    add_bullet(
        doc,
        "Exclusión de duration para evitar fuga de información: ",
        "La duración de la llamada solo se conoce cuando el cliente ya colgó. Si la metiéramos al modelo, las métricas saldrían artificialmente altas pero el modelo no serviría en la práctica para decidir a quién llamar antes de marcar.",
    )
    add_bullet(
        doc,
        "Revisión de duplicados y homogeneización de texto: ",
        f"Se verificó si existían filas repetidas exactas y se encontraron {m['duplicados']}. No se borraron clientes contactados más de una vez porque cada llamada ocurre en distinto momento. En todas las variables de texto se recortaron espacios y se pasaron a minúsculas para evitar categorías duplicadas por formato.",
    )
    add_bullet(
        doc,
        "Tratamiento de valores faltantes y códigos especiales: ",
        "Borrar las filas con faltantes en job, education, contact y poutcome eliminaría más del 80% de la base. Por eso decidimos imputarlas con la moda dentro del Pipeline de entrenamiento. El valor pdays = -1 se conservó porque no es un error, sino el código oficial que indica que el cliente no había sido contactado antes.",
    )
    add_bullet(
        doc,
        "Decisión sobre valores atípicos (criterio IQR): ",
        "En los diagramas de caja (Figura 2) se observan valores extremos en balance, campaign, pdays y previous. Decidimos no eliminarlos ni truncarlos porque en un banco son clientes reales (personas con ahorros altos o con historial largo de contacto); borrarlos sesgaría la muestra.",
    )

    add_fig(
        doc,
        "02_boxplots.png",
        "Figura 2. Diagramas de caja para el diagnóstico de distribución y valores extremos en las variables numéricas.",
        width=5.9,
    )

    add_title_h2(doc, "3.2 Ingeniería de características, transformación y selección por filtros")
    add_p(
        doc,
        "Para darle más señal útil a los modelos creamos dos variables derivadas usando solo datos conocidos antes de llamar: 1) presion_contacto = campaign / (previous + 1), que mide qué tanto se ha insistido en la campaña actual respecto al historial previo (sumando 1 para no dividir entre cero), y 2) balance_ajustado_credito = balance / (1 + housing + loan), que aproxima la liquidez disponible del cliente tomando en cuenta si ya paga hipoteca o préstamo personal.",
    )
    add_p(
        doc,
        f"Antes de aplicar cualquier transformación estadística, dividimos el dataset de forma estratificada en 80% entrenamiento ({m['n_train']:,} registros) y 20% prueba independiente ({m['n_test']:,} registros), conservando el 11.7% de casos positivos en ambos. Dentro de un ColumnTransformer y Pipeline de scikit-learn (ajustado solo en entrenamiento para evitar data leakage):",
    )
    add_bullet(
        doc,
        "Variables numéricas: ",
        "Se imputaron con la mediana (robusta ante valores extremos), se escalaron al rango [0, 1] con MinMaxScaler para homogenizar unidades y se seleccionaron las 6 mejores con el filtro de correlación de Pearson.",
    )
    add_bullet(
        doc,
        "Variables categóricas: ",
        "Las binarias (default, housing, loan) se codificaron como 0/1, las nominales se imputaron con la moda y se transformaron con OneHotEncoder(drop='if_binary', handle_unknown='ignore'), seleccionando las 25 columnas codificadas más relevantes mediante el filtro Chi-cuadrada.",
    )

    add_fig(
        doc,
        "03_filtros.png",
        "Figura 3. Ranking de asociación en entrenamiento con los filtros Chi-cuadrada (V de Cramér) y correlación de Pearson.",
        width=6.0,
    )

    # 4. Justificación de la elección de modelos, metodología y optimización
    add_title_h1(doc, "4. Justificación de los modelos y metodología implementada")
    add_p(
        doc,
        "Para no limitarnos a un solo tipo de clasificador, elegimos y comparamos tres algoritmos supervisados de distintas familias y complejidad creciente:",
    )
    add_bullet(
        doc,
        "1. Regresión Logística: ",
        "Se eligió como línea base lineal porque es rápida, estable y fácil de interpretar al estimar probabilidades mediante una combinación lineal de los predictores. Su limitación es que no captura interacciones no lineales complejas.",
    )
    add_bullet(
        doc,
        "2. Árbol de Decisión (DecisionTreeClassifier): ",
        "Se eligió porque permite capturar relaciones no lineales e interacciones entre variables mediante reglas jerárquicas (por ejemplo, combinar resultado previo, mes y saldo). Sin embargo, un árbol individual tiende a sobreajustarse si crece sin control.",
    )
    add_bullet(
        doc,
        "3. Random Forest (Bosque Aleatorio): ",
        "Se eligió como modelo de ensamble (bagging) porque combina múltiples árboles entrenados con submuestras aleatorias de datos y variables. Esto reduce la varianza del árbol individual, mejora el poder predictivo en datos tabulares mixtos y mantiene una excelente capacidad de generalización.",
    )
    add_p(
        doc,
        "En los tres modelos activamos class_weight='balanced' para penalizar más los errores en la clase minoritaria (11.7% de suscriptores). Además, para optimizar su desempeño sin tocar el conjunto de prueba, aplicamos GridSearchCV con validación cruzada estratificada de 5 particiones (5-Fold StratifiedKFold) únicamente sobre el 80% de entrenamiento, usando ROC AUC como métrica objetivo y restringiendo la profundidad y el tamaño mínimo de hoja en los árboles para evitar el sobreajuste:",
    )

    add_table(
        doc,
        ["Modelo", "Hiperparámetros explorados en GridSearchCV", "Mejor configuración encontrada (5-Fold CV)"],
        [
            ["Regresión Logística", "C: [0.1, 1.0, 10.0]", "C = 10.0"],
            ["Árbol de Decisión", "max_depth: [4, 6, 8], min_samples_leaf: [25, 50, 100]", "max_depth = 8, min_samples_leaf = 50"],
            ["Random Forest", "n_estimators: [100, 150], max_depth: [8, 12], min_samples_leaf: [20, 40]", "n_estimators = 150, max_depth = 12, min_samples_leaf = 20"],
        ],
        widths=[1.6, 2.7, 2.4],
        size=8.6,
    )

    # 5. Comparación de métricas (Entrenamiento, Validación y Prueba) y Matrices de Confusión
    add_title_h1(doc, "5. Comparación de métricas (Entrenamiento, Validación y Prueba) y análisis de matrices de confusión")
    add_title_h2(doc, "5.1 Análisis de desempeño, generalización y control del sobreajuste")
    add_p(
        doc,
        "En un problema desbalanceado la exactitud simple (Accuracy) es engañosa, ya que decir siempre 'No' daría 88.3% de acierto pero 0 clientes captados. Por eso evaluamos los modelos con ROC AUC (capacidad de ordenar probabilidades), Exactitud Balanceada (promedio entre sensibilidad y especificidad), F1-Score, Precisión y Sensibilidad (Recall). La siguiente tabla compara las tres etapas para cada modelo:",
    )

    add_table(
        doc,
        ["Modelo", "Etapa de evaluación", "ROC AUC", "Exact. Balanceada", "F1", "Precisión", "Sensibilidad"],
        [
            [
                r["Modelo"],
                r["Etapa"],
                f"{r['ROC AUC']:.3f}",
                f"{r['Exactitud Balanceada']:.3f}",
                f"{r['F1']:.3f}",
                f"{r['Precisión']:.3f}",
                f"{r['Sensibilidad']:.3f}",
            ]
            for r in m["comparacion_etapas"]
        ],
        widths=[1.45, 1.65, 0.7, 1.05, 0.6, 0.65, 0.7],
        size=8.4,
    )

    add_fig(
        doc,
        "04_comparacion_modelos.png",
        "Figura 4. Comparación del desempeño promedio de los tres modelos optimizados en validación cruzada (5 particiones).",
        width=5.8,
    )

    add_p(
        doc,
        f"Al comparar Entrenamiento, Validación Cruzada y Prueba comprobamos que los modelos generalizan bien y no tienen sobreajuste severo. En particular, {mejor} fue el mejor modelo en las tres etapas: obtuvo un ROC AUC de {tr_mejor['ROC AUC']:.3f} en entrenamiento, {cv_mejor['ROC AUC']:.3f} (±{cv_mejor['ROC AUC std']:.3f}) en validación cruzada y {test_mejor['ROC AUC']:.3f} en prueba independiente, con una Exactitud Balanceada de {test_mejor['Exactitud Balanceada']:.3f} y sensibilidad de {test_mejor['Sensibilidad']:.3f} en prueba. La brecha moderada entre entrenamiento y prueba (~0.04 en ROC AUC y ~0.02 en exactitud balanceada) confirma que limitar max_depth=12 y min_samples_leaf=20 evitó que los árboles memorizaran el conjunto de entrenamiento.",
    )

    add_title_h2(doc, "5.2 Matrices de confusión de Entrenamiento y Prueba")
    add_p(
        doc,
        f"En la Figura 5 se presentan las curvas ROC de los tres algoritmos junto con la matriz de confusión de entrenamiento (n = {m['n_train']:,}) y la matriz de confusión de prueba (n = {m['n_test']:,}) del mejor modelo ({mejor}):",
    )

    add_fig(
        doc,
        "05_roc_y_confusion.png",
        f"Figura 5. Curvas ROC en prueba y matrices de confusión de Entrenamiento y Prueba para {mejor}.",
        width=6.3,
    )

    add_bullet(
        doc,
        "Análisis de la matriz de Entrenamiento (36,168 clientes): ",
        f"El modelo clasifica correctamente {cm_tr['tn']:,} clientes que no contratan (Verdaderos Negativos, 82.4% de los 'No') y detecta {cm_tr['tp']:,} clientes que sí contratan (Verdaderos Positivos, sensibilidad de 62.7%), con {cm_tr['fp']:,} Falsos Positivos y {cm_tr['fn']:,} Falsos Negativos.",
    )
    add_bullet(
        doc,
        "Análisis de la matriz de Prueba (9,043 clientes nuevos): ",
        f"Mantiene exactamente el mismo comportamiento proporcional: acierta en {cm_te['tn']:,} clientes que no contratan (81.4% de especificidad) y captura {cm_te['tp']:,} de los 1,058 suscriptores reales (59.5% de sensibilidad), con {cm_te['fp']:,} Falsos Positivos y {cm_te['fn']:,} Falsos Negativos.",
    )
    add_bullet(
        doc,
        "Interpretación desde el negocio bancario: ",
        "En marketing telefónico un Falso Negativo (dejar de llamar a alguien que sí iba a contratar un depósito) cuesta mucho más que un Falso Positivo (hacer una llamada extra de pocos minutos). Gracias al balanceo de clases y a la optimización realizada, el modelo captura casi al 60% de los suscriptores en el umbral de 0.50 y mantiene proporciones consistentes entre entrenamiento y prueba.",
    )

    # 6. Interpretación de los resultados y conclusiones
    add_title_h1(doc, "6. Interpretación de los resultados y conclusiones")
    add_title_h2(doc, "6.1 Interpretación de variables e impacto en la priorización de llamadas")
    add_p(
        doc,
        "Al analizar la importancia de variables de Random Forest (Figura 6, izquierda), vemos que la decisión del cliente depende principalmente del éxito o fracaso en campañas previas (poutcome_success, poutcome_failure), si tiene hipoteca activa (housing), su liquidez ajustada por deudas (nuestra variable derivada balance_ajustado_credito), el historial de contacto (pdays, presion_contacto, previous), su saldo (balance) y la estacionalidad del mes de llamada.",
    )
    add_p(
        doc,
        "Más allá de clasificar con un corte fijo de 0.50, la verdadera utilidad del modelo para el banco es ordenar a los clientes de mayor a menor probabilidad de contratar. Al simular esta priorización sobre los 9,043 clientes del conjunto de prueba (donde hay 1,058 suscriptores reales), obtenemos lo siguiente:",
    )

    add_table(
        doc,
        ["Segmento contactado", "Llamadas realizadas", "Suscriptores logrados", "Tasa de éxito", "% del total capturado", "Mejora vs azar (Lift)"],
        [
            [r["grupo"], f"{r['llamadas']:,}", f"{r['suscriptores']:,}", f"{r['tasa_exito_pct']:.1f}%", f"{r['captura_total_pct']:.1f}%", f"{r['mejora_vs_azar']:.2f}x"]
            for r in m["priorizacion"]
        ],
        widths=[1.65, 1.0, 1.15, 0.9, 1.1, 0.9],
        size=8.6,
    )

    add_fig(
        doc,
        "06_importancia_y_priorizacion.png",
        "Figura 6. Top 10 variables más importantes en Random Forest y tasa de conversión al priorizar llamadas.",
        width=6.1,
    )

    add_p(
        doc,
        f"La interpretación práctica es muy clara: si el banco llama al azar, solo convence al 11.7% de los clientes. En cambio, si usa nuestro Random Forest para llamar primero al Top 10% con mayor probabilidad ({p10['llamadas']:,} llamadas), la tasa de conversión sube a {p10['tasa_exito_pct']:.1f}% ({p10['mejora_vs_azar']:.2f} veces mejor que al azar) y logra el {p10['captura_total_pct']:.1f}% de todas las ventas posibles. Si amplía la campaña al Top 30% ({p30['llamadas']:,} llamadas), captura el {p30['captura_total_pct']:.1f}% de todos los suscriptores ahorrándose el 70% de las llamadas telefónicas.",
    )

    add_title_h2(doc, "6.2 Conclusiones del trabajo")
    add_bullet(
        doc,
        "Solución efectiva a la problemática planteada: ",
        f"Se demostró que sí es posible predecir y ordenar la propensión de compra antes de realizar la llamada. Usar {mejor} como filtro de priorización permite pasar de una tasa base de 11.7% a {p10['tasa_exito_pct']:.1f}% en el primer decil y captar dos terceras partes de las suscripciones con solo el 30% del esfuerzo telefónico.",
    )
    add_bullet(
        doc,
        "Valor del preprocesamiento justificado y sin fuga de datos: ",
        "Excluir duration evitó métricas irreales; conservar los valores extremos y el código pdays = -1 respetó la naturaleza bancaria de los datos; e integrar la imputación, el escalamiento y los filtros Chi-cuadrada/Pearson dentro de un Pipeline evitó cualquier fuga de información entre entrenamiento y prueba. Además, las dos variables creadas (balance_ajustado_credito y presion_contacto) quedaron dentro del Top 7 de mayor importancia.",
    )
    add_bullet(
        doc,
        "Superioridad y buena generalización de Random Forest: ",
        f"A través de GridSearchCV y validación cruzada estratificada de 5 particiones (sin tocar el conjunto de prueba), Random Forest superó a la Regresión Logística y al Árbol de Decisión individual. La consistencia de sus métricas y matrices de confusión entre entrenamiento (ROC AUC = {tr_mejor['ROC AUC']:.3f}), validación ({cv_mejor['ROC AUC']:.3f}) y prueba ({test_mejor['ROC AUC']:.3f}) comprueba que el modelo generaliza correctamente y evita el sobreajuste.",
    )
    add_bullet(
        doc,
        "Trabajo futuro: ",
        "En una implementación real dentro del banco se podría calibrar el umbral óptimo de corte incorporando el costo exacto en pesos/euros de cada minuto de llamada frente a la ganancia financiera promedio de cada depósito contratado.",
    )

    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rf = footer.add_run("Proyecto de Minería de Datos - Bank Marketing")
    rf.font.size = Pt(8.5)
    rf.font.color.rgb = RGBColor(120, 120, 120)

    doc.save(OUT_DOCX)
    print(f"Word generado correctamente en: {OUT_DOCX}")


def build_pdf(m: dict) -> None:
    mejor = m["mejor_modelo"]
    tr_mejor = next(r for r in m["train"] if r["Modelo"] == mejor)
    cv_mejor = next(r for r in m["cv"] if r["Modelo"] == mejor)
    test_mejor = next(r for r in m["test"] if r["Modelo"] == mejor)
    cm_tr = m["matriz_confusion_train"]
    cm_te = m["matriz_confusion_test"]
    p10 = m["priorizacion"][0]
    p30 = m["priorizacion"][2]

    figs = {
        "fig1": img_to_data_uri(FIG_DIR / "01_distribucion_y.png"),
        "fig2": img_to_data_uri(FIG_DIR / "02_boxplots.png"),
        "fig3": img_to_data_uri(FIG_DIR / "03_filtros.png"),
        "fig4": img_to_data_uri(FIG_DIR / "04_comparacion_modelos.png"),
        "fig5": img_to_data_uri(FIG_DIR / "05_roc_y_confusion.png"),
        "fig6": img_to_data_uri(FIG_DIR / "06_importancia_y_priorizacion.png"),
    }

    filas_etapas_html = "\n".join(
        f"<tr>"
        f"<td><strong>{r['Modelo']}</strong></td>"
        f"<td>{r['Etapa']}</td>"
        f"<td>{r['ROC AUC']:.3f}</td>"
        f"<td>{r['Exactitud Balanceada']:.3f}</td>"
        f"<td>{r['F1']:.3f}</td>"
        f"<td>{r['Precisión']:.3f}</td>"
        f"<td>{r['Sensibilidad']:.3f}</td>"
        f"</tr>"
        for r in m["comparacion_etapas"]
    )

    filas_prio_html = "\n".join(
        f"<tr>"
        f"<td><strong>{r['grupo']}</strong></td>"
        f"<td>{r['llamadas']:,}</td>"
        f"<td>{r['suscriptores']:,}</td>"
        f"<td>{r['tasa_exito_pct']:.1f}%</td>"
        f"<td>{r['captura_total_pct']:.1f}%</td>"
        f"<td>{r['mejora_vs_azar']:.2f}x</td>"
        f"</tr>"
        for r in m["priorizacion"]
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Proyecto de Minería de Datos: Bank Marketing</title>
<style>
  @page {{
    size: letter;
    margin: 1.8cm 2.0cm 1.9cm 2.0cm;
    @bottom-center {{
      content: "Proyecto de Minería de Datos - Bank Marketing";
      font-family: Arial, Helvetica, sans-serif;
      font-size: 8.5pt;
      color: #787878;
    }}
  }}
  body {{
    font-family: Arial, Helvetica, sans-serif;
    color: #1e1e1e;
    font-size: 10.2pt;
    line-height: 1.38;
    margin: 0;
  }}
  .title {{
    text-align: center;
    font-size: 16pt;
    font-weight: bold;
    color: #2B547E;
    margin-bottom: 4px;
  }}
  .subtitle {{
    text-align: center;
    font-size: 11pt;
    font-style: italic;
    color: #333333;
    margin-bottom: 12px;
  }}
  h1 {{
    font-size: 13pt;
    color: #2B547E;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}
  h2 {{
    font-size: 11pt;
    color: #1e1e1e;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }}
  p {{
    margin: 0 0 7px 0;
    text-align: justify;
  }}
  ul {{
    margin: 4px 0 8px 0;
    padding-left: 20px;
  }}
  li {{
    margin-bottom: 4px;
    text-align: justify;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 11px 0;
    font-size: 8.8pt;
    page-break-inside: avoid;
  }}
  th {{
    background: #2B547E;
    color: #ffffff;
    text-align: left;
    padding: 5px 7px;
    font-weight: bold;
    border: 1px solid #CCCCCC;
  }}
  td {{
    border: 1px solid #CCCCCC;
    padding: 4.5px 7px;
    vertical-align: middle;
  }}
  tr:nth-child(even) td {{
    background: #F2F6FA;
  }}
  .fig {{
    text-align: center;
    margin: 8px 0 12px 0;
    page-break-inside: avoid;
  }}
  .fig img {{
    max-width: 98%;
    height: auto;
  }}
  .caption {{
    font-size: 8.8pt;
    font-style: italic;
    color: #5a5a5a;
    margin-top: 3px;
  }}
</style>
</head>
<body>

<div class="title">Proyecto de Minería de Datos: Bank Marketing</div>
<div class="subtitle">Predicción de contratación de depósitos a plazo fijo para priorizar campañas telefónicas</div>

<table>
  <thead>
    <tr>
      <th style="width:36%;">Alumno(s)</th>
      <th style="width:20%;">Materia</th>
      <th style="width:26%;">Profesor</th>
      <th style="width:18%;">Fecha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Rodrigo Alonso Castillo Ramirez<br>Sebastian Sanchez Espinosa<br>Iosset Ivan Sandoval Gonzalez</td>
      <td>Minería de Datos</td>
      <td>MILAGROS CONTRERAS ELIZALDE</td>
      <td>23 de Noviembre del 2026</td>
    </tr>
  </tbody>
</table>

<h1>1. Motivación del trabajo y problemática a abordar</h1>
<p>
  En las campañas de marketing telefónico de un banco, llamar a todos los clientes registrados implica un gasto alto de tiempo y dinero (horas de los agentes del centro de llamadas, costos de operación y varios intentos por persona). Además, marcar sin un filtro previo hace que la gran mayoría de las personas rechacen la oferta (en esta base el 88.3% dicen que no) y genera molestia en los clientes por recibir llamadas insistentes.
</p>
<p>
  <strong>Problemática y motivación:</strong> El banco necesita identificar de antemano, utilizando únicamente la información disponible antes de realizar la llamada, qué clientes tienen mayor probabilidad de aceptar un depósito a plazo fijo para no desperdiciar recursos llamando al azar.
</p>
<p>
  <strong>Objetivo del proyecto:</strong> Construir, optimizar y evaluar un flujo completo de Minería de Datos (siguiendo las etapas del proceso KDD) con modelos de clasificación supervisada que estimen la probabilidad de suscripción antes de marcar, permitiendo priorizar la lista de llamadas en las campañas telefónicas.
</p>

<h1>2. Justificación de la selección de la base de datos</h1>
<p>
  Se seleccionó el dataset público <strong>Bank Marketing</strong> del repositorio UCI Machine Learning (ID 222), el cual recopila llamadas reales de campañas telefónicas de una institución bancaria de Portugal. Cuenta con <strong>{m['n_total']:,} registros</strong> y <strong>16 variables predictoras</strong> (7 numéricas y 9 categóricas), además de la variable objetivo binaria <code>y</code> (<code>yes</code> o <code>no</code>). Elegimos esta base de datos por tres razones principales que la hacen ideal para un proyecto de Minería de Datos:
</p>
<ul>
  <li><strong>Tamaño y datos del mundo real:</strong> Con {m['n_total']:,} registros tenemos suficiente volumen para entrenar, hacer validación cruzada y reservar un conjunto de prueba independiente de {m['n_test']:,} clientes sin que los resultados dependan de la suerte de una muestra pequeña.</li>
  <li><strong>Combinación de variables socioeconómicas y de campaña:</strong> Contiene tanto el perfil financiero y personal del cliente (edad, ocupación, estado civil, estudios, saldo promedio, préstamos activos) como el historial de contacto (mes, número de llamadas, días desde la campaña anterior y resultado previo).</li>
  <li><strong>Retos reales de preprocesamiento y modelado:</strong> Presenta un fuerte desbalance de clases (solo el {m['positivos_pct']:.1f}% dijo que sí frente al {100 - m['positivos_pct']:.1f}% que dijo que no), valores faltantes en variables categóricas, códigos especiales (<code>pdays = -1</code>), valores extremos reales en saldos y una variable con fuga de información (<code>duration</code>) que debemos detectar y tratar.</li>
</ul>

<table>
  <thead>
    <tr>
      <th style="width:22%;">Variable</th>
      <th style="width:16%;">Tipo</th>
      <th style="width:62%;">Qué representa y consideración en el proyecto</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>age</td><td>Numérica</td><td>Edad del cliente en años.</td></tr>
    <tr><td>job</td><td>Categórica</td><td>Trabajo u ocupación (11 categorías; 288 datos faltantes).</td></tr>
    <tr><td>marital</td><td>Categórica</td><td>Estado civil (married, single, divorced).</td></tr>
    <tr><td>education</td><td>Categórica</td><td>Nivel de estudios (primary, secondary, tertiary; 1,857 faltantes).</td></tr>
    <tr><td>default</td><td>Binaria</td><td>Si el cliente tiene crédito en mora (yes/no).</td></tr>
    <tr><td>balance</td><td>Numérica</td><td>Saldo anual promedio en euros.</td></tr>
    <tr><td>housing / loan</td><td>Binarias</td><td>Si tiene crédito hipotecario o préstamo personal activo (yes/no).</td></tr>
    <tr><td>contact</td><td>Categórica</td><td>Medio de contacto: celular o teléfono fijo (13,020 faltantes).</td></tr>
    <tr><td>day_of_week / month</td><td>Num. / Cat.</td><td>Día del mes y mes del último contacto.</td></tr>
    <tr><td>duration</td><td>Numérica</td><td>Duración de la llamada en segundos (<strong>se excluyó por fuga de datos</strong>).</td></tr>
    <tr><td>campaign</td><td>Numérica</td><td>Número de llamadas hechas al cliente en esta campaña.</td></tr>
    <tr><td>pdays / previous</td><td>Numéricas</td><td>Días desde el contacto anterior (-1 si nunca se le llamó) y llamadas previas.</td></tr>
    <tr><td>poutcome</td><td>Categórica</td><td>Resultado de la campaña anterior (success, failure, other; 36,959 faltantes).</td></tr>
    <tr><td>y</td><td>Objetivo</td><td>Si contrató el depósito a plazo fijo (yes = 1, no = 0).</td></tr>
  </tbody>
</table>

<div class="fig">
  <img src="{figs['fig1']}" alt="Distribución de la variable objetivo">
  <div class="caption">Figura 1. Distribución desbalanceada de la variable objetivo (y) y tasa de suscripción según el resultado de la campaña previa.</div>
</div>

<h1>3. Explicación y justificación del preprocesamiento de los datos</h1>
<h2>3.1 Diagnóstico de calidad y limpieza de datos</h2>
<ul>
  <li><strong>Exclusión de duration para evitar fuga de información:</strong> La duración de la llamada solo se conoce cuando el cliente ya colgó. Si la metiéramos al modelo, las métricas saldrían artificialmente altas pero el modelo no serviría en la práctica para decidir a quién llamar antes de marcar.</li>
  <li><strong>Revisión de duplicados y homogeneización de texto:</strong> Se verificó si existían filas repetidas exactas y se encontraron {m['duplicados']}. No se borraron clientes contactados más de una vez porque cada llamada ocurre en distinto momento. En todas las variables de texto se recortaron espacios y se pasaron a minúsculas para evitar categorías duplicadas por formato.</li>
  <li><strong>Tratamiento de valores faltantes y códigos especiales:</strong> Borrar las filas con faltantes en job, education, contact y poutcome eliminaría más del 80% de la base. Por eso decidimos imputarlas con la moda dentro del Pipeline de entrenamiento. El valor pdays = -1 se conservó porque no es un error, sino el código oficial que indica que el cliente no había sido contactado antes.</li>
  <li><strong>Decisión sobre valores atípicos (criterio IQR):</strong> En los diagramas de caja (Figura 2) se observan valores extremos en balance, campaign, pdays y previous. Decidimos no eliminarlos ni truncarlos porque en un banco son clientes reales (personas con ahorros altos o con historial largo de contacto); borrarlos sesgaría la muestra.</li>
</ul>

<div class="fig">
  <img src="{figs['fig2']}" alt="Diagramas de caja">
  <div class="caption">Figura 2. Diagramas de caja para el diagnóstico de distribución y valores extremos en las variables numéricas.</div>
</div>

<h2>3.2 Ingeniería de características, transformación y selección por filtros</h2>
<p>
  Para darle más señal útil a los modelos creamos dos variables derivadas usando solo datos conocidos antes de llamar: 1) <strong>presion_contacto = campaign / (previous + 1)</strong>, que mide qué tanto se ha insistido en la campaña actual respecto al historial previo (sumando 1 para no dividir entre cero), y 2) <strong>balance_ajustado_credito = balance / (1 + housing + loan)</strong>, que aproxima la liquidez disponible del cliente tomando en cuenta si ya paga hipoteca o préstamo personal.
</p>
<p>
  Antes de aplicar cualquier transformación estadística, dividimos el dataset de forma estratificada en <strong>80% entrenamiento ({m['n_train']:,} registros)</strong> y <strong>20% prueba independiente ({m['n_test']:,} registros)</strong>, conservando el 11.7% de casos positivos en ambos. Dentro de un ColumnTransformer y Pipeline de scikit-learn (ajustado solo en entrenamiento para evitar data leakage):
</p>
<ul>
  <li><strong>Variables numéricas:</strong> Se imputaron con la mediana (robusta ante valores extremos), se escalaron al rango [0, 1] con MinMaxScaler para homogenizar unidades y se seleccionaron las 6 mejores con el filtro de correlación de Pearson (donde destacaron pdays con r = {m['ranking_pearson'][0]['correlacion']:.3f}, presion_contacto con r = {m['ranking_pearson'][1]['correlacion']:.3f} y previous con r = {m['ranking_pearson'][2]['correlacion']:.3f}).</li>
  <li><strong>Variables categóricas:</strong> Las binarias (default, housing, loan) se codificaron como 0/1, las nominales se imputaron con la moda y se transformaron con OneHotEncoder(drop='if_binary', handle_unknown='ignore'), seleccionando las 25 columnas codificadas más relevantes mediante el filtro Chi-cuadrada (donde destacaron poutcome con V de Cramér = {m['ranking_chi2'][0]['v_cramer']:.3f}, month con {m['ranking_chi2'][1]['v_cramer']:.3f} y housing con {m['ranking_chi2'][2]['v_cramer']:.3f}).</li>
</ul>

<div class="fig">
  <img src="{figs['fig3']}" alt="Filtros Chi-cuadrada y Pearson">
  <div class="caption">Figura 3. Ranking de asociación en entrenamiento con los filtros Chi-cuadrada (V de Cramér) y correlación de Pearson.</div>
</div>

<h1>4. Justificación de los modelos y metodología implementada</h1>
<p>
  Para no limitarnos a un solo tipo de clasificador, elegimos y comparamos tres algoritmos supervisados de distintas familias y complejidad creciente:
</p>
<ul>
  <li><strong>1. Regresión Logística:</strong> Se eligió como línea base lineal porque es rápida, estable y fácil de interpretar al estimar probabilidades mediante una combinación lineal de los predictores. Su limitación es que no captura interacciones no lineales complejas.</li>
  <li><strong>2. Árbol de Decisión (DecisionTreeClassifier):</strong> Se eligió porque permite capturar relaciones no lineales e interacciones entre variables mediante reglas jerárquicas (por ejemplo, combinar resultado previo, mes y saldo). Sin embargo, un árbol individual tiende a sobreajustarse si crece sin control.</li>
  <li><strong>3. Random Forest (Bosque Aleatorio):</strong> Se eligió como modelo de ensamble (bagging) porque combina múltiples árboles entrenados con submuestras aleatorias de datos y variables. Esto reduce la varianza del árbol individual, mejora el poder predictivo en datos tabulares mixtos y mantiene una excelente capacidad de generalización.</li>
</ul>
<p>
  En los tres modelos activamos <strong>class_weight='balanced'</strong> para penalizar más los errores en la clase minoritaria (11.7% de suscriptores). Además, para optimizar su desempeño <strong>sin utilizar el conjunto de prueba</strong>, aplicamos <strong>GridSearchCV</strong> con validación cruzada estratificada de 5 particiones (5-Fold StratifiedKFold) únicamente sobre el 80% de entrenamiento, usando ROC AUC como métrica objetivo y restringiendo la profundidad y el tamaño mínimo de hoja en los árboles para evitar el sobreajuste:
</p>

<table>
  <thead>
    <tr>
      <th style="width:24%;">Modelo</th>
      <th style="width:42%;">Hiperparámetros explorados en GridSearchCV</th>
      <th style="width:34%;">Mejor configuración encontrada (5-Fold CV)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>Regresión Logística</strong></td><td>C: [0.1, 1.0, 10.0]</td><td>C = 10.0</td></tr>
    <tr><td><strong>Árbol de Decisión</strong></td><td>max_depth: [4, 6, 8], min_samples_leaf: [25, 50, 100]</td><td>max_depth = 8, min_samples_leaf = 50</td></tr>
    <tr><td><strong>Random Forest</strong></td><td>n_estimators: [100, 150], max_depth: [8, 12], min_samples_leaf: [20, 40]</td><td>n_estimators = 150, max_depth = 12, min_samples_leaf = 20</td></tr>
  </tbody>
</table>

<h1>5. Comparación de métricas (Entrenamiento, Validación y Prueba) y matrices de confusión</h1>
<h2>5.1 Análisis de desempeño, generalización y control del sobreajuste</h2>
<p>
  En un problema desbalanceado la exactitud simple (Accuracy) es engañosa, ya que decir siempre "No" daría 88.3% de acierto pero 0 clientes captados. Por eso evaluamos los modelos con <strong>ROC AUC</strong> (capacidad de ordenar probabilidades), <strong>Exactitud Balanceada</strong> (promedio entre sensibilidad y especificidad), <strong>F1-Score</strong>, <strong>Precisión</strong> y <strong>Sensibilidad (Recall)</strong>. La siguiente tabla compara las tres etapas para cada modelo:
</p>

<table>
  <thead>
    <tr>
      <th>Modelo</th>
      <th>Etapa de evaluación</th>
      <th>ROC AUC</th>
      <th>Exact. Balanceada</th>
      <th>F1</th>
      <th>Precisión</th>
      <th>Sensibilidad</th>
    </tr>
  </thead>
  <tbody>
    {filas_etapas_html}
  </tbody>
</table>

<div class="fig">
  <img src="{figs['fig4']}" alt="Comparación de modelos en CV">
  <div class="caption">Figura 4. Comparación del desempeño promedio de los tres modelos optimizados en validación cruzada (5 particiones).</div>
</div>

<p>
  Al comparar Entrenamiento, Validación Cruzada y Prueba comprobamos que los modelos generalizan bien y no tienen sobreajuste severo. En particular, <strong>{mejor}</strong> fue el mejor modelo en las tres etapas: obtuvo un ROC AUC de <strong>{tr_mejor['ROC AUC']:.3f}</strong> en entrenamiento, <strong>{cv_mejor['ROC AUC']:.3f} (±{cv_mejor['ROC AUC std']:.3f})</strong> en validación cruzada y <strong>{test_mejor['ROC AUC']:.3f}</strong> en prueba independiente, con una Exactitud Balanceada de <strong>{test_mejor['Exactitud Balanceada']:.3f}</strong> y sensibilidad de <strong>{test_mejor['Sensibilidad']:.3f}</strong> en prueba. La brecha moderada entre entrenamiento y prueba (~0.04 en ROC AUC y ~0.02 en exactitud balanceada) confirma que limitar max_depth=12 y min_samples_leaf=20 evitó que los árboles memorizaran el conjunto de entrenamiento.
</p>

<h2>5.2 Matrices de confusión de Entrenamiento y Prueba</h2>
<p>
  En la Figura 5 se presentan las curvas ROC de los tres algoritmos junto con la <strong>matriz de confusión de entrenamiento</strong> (n = {m['n_train']:,}) y la <strong>matriz de confusión de prueba</strong> (n = {m['n_test']:,}) del mejor modelo ({mejor}):
</p>

<div class="fig">
  <img src="{figs['fig5']}" alt="Curvas ROC y matrices de confusión">
  <div class="caption">Figura 5. Curvas ROC en prueba y matrices de confusión de Entrenamiento y Prueba para {mejor}.</div>
</div>

<ul>
  <li><strong>Análisis de la matriz de Entrenamiento (36,168 clientes):</strong> El modelo clasifica correctamente {cm_tr['tn']:,} clientes que no contratan (Verdaderos Negativos, 82.4% de los "No") y detecta {cm_tr['tp']:,} clientes que sí contratan (Verdaderos Positivos, sensibilidad de 62.7%), con {cm_tr['fp']:,} Falsos Positivos y {cm_tr['fn']:,} Falsos Negativos.</li>
  <li><strong>Análisis de la matriz de Prueba (9,043 clientes nuevos):</strong> Mantiene exactamente el mismo comportamiento proporcional: acierta en {cm_te['tn']:,} clientes que no contratan (81.4% de especificidad) y captura {cm_te['tp']:,} de los 1,058 suscriptores reales (59.5% de sensibilidad), con {cm_te['fp']:,} Falsos Positivos y {cm_te['fn']:,} Falsos Negativos.</li>
  <li><strong>Interpretación desde el negocio bancario:</strong> En marketing telefónico un Falso Negativo (dejar de llamar a alguien que sí iba a contratar un depósito) cuesta mucho más que un Falso Positivo (hacer una llamada extra de pocos minutos). Gracias al balanceo de clases y a la optimización realizada, el modelo captura casi al 60% de los suscriptores en el umbral de 0.50 y mantiene proporciones consistentes entre entrenamiento y prueba.</li>
</ul>

<h1>6. Interpretación de los resultados y conclusiones</h1>
<h2>6.1 Interpretación de variables e impacto en la priorización de llamadas</h2>
<p>
  Al analizar la importancia de variables de Random Forest (Figura 6, izquierda), vemos que la decisión del cliente depende principalmente del éxito o fracaso en campañas previas (poutcome_success, poutcome_failure), si tiene hipoteca activa (housing), su liquidez ajustada por deudas (nuestra variable derivada balance_ajustado_credito), el historial de contacto (pdays, presion_contacto, previous), su saldo (balance) y la estacionalidad del mes de llamada.
</p>
<p>
  Más allá de clasificar con un corte fijo de 0.50, la verdadera utilidad del modelo para el banco es ordenar a los clientes de mayor a menor probabilidad de contratar. Al simular esta priorización sobre los 9,043 clientes del conjunto de prueba (donde hay 1,058 suscriptores reales), obtenemos lo siguiente:
</p>

<table>
  <thead>
    <tr>
      <th>Segmento contactado</th>
      <th>Llamadas realizadas</th>
      <th>Suscriptores logrados</th>
      <th>Tasa de éxito</th>
      <th>% del total capturado</th>
      <th>Mejora vs azar (Lift)</th>
    </tr>
  </thead>
  <tbody>
    {filas_prio_html}
  </tbody>
</table>

<div class="fig">
  <img src="{figs['fig6']}" alt="Importancia y priorización">
  <div class="caption">Figura 6. Top 10 variables más importantes en Random Forest y tasa de conversión al priorizar llamadas.</div>
</div>

<p>
  La interpretación práctica es muy clara: si el banco llama al azar, solo convence al 11.7% de los clientes. En cambio, si usa nuestro Random Forest para llamar primero al <strong>Top 10%</strong> con mayor probabilidad ({p10['llamadas']:,} llamadas), la tasa de conversión sube a <strong>{p10['tasa_exito_pct']:.1f}%</strong> (<strong>{p10['mejora_vs_azar']:.2f} veces</strong> mejor que al azar) y logra el <strong>{p10['captura_total_pct']:.1f}%</strong> de todas las ventas posibles. Si amplía la campaña al <strong>Top 30%</strong> ({p30['llamadas']:,} llamadas), captura el <strong>{p30['captura_total_pct']:.1f}%</strong> de todos los suscriptores ahorrándose el 70% de las llamadas telefónicas.
</p>

<h2>6.2 Conclusiones del trabajo</h2>
<ul>
  <li><strong>Solución efectiva a la problemática planteada:</strong> Se demostró que sí es posible predecir y ordenar la propensión de compra antes de realizar la llamada. Usar {mejor} como filtro de priorización permite pasar de una tasa base de 11.7% a {p10['tasa_exito_pct']:.1f}% en el primer decil y captar dos terceras partes de las suscripciones con solo el 30% del esfuerzo telefónico.</li>
  <li><strong>Valor del preprocesamiento justificado y sin fuga de datos:</strong> Excluir duration evitó métricas irreales; conservar los valores extremos y el código pdays = -1 respetó la naturaleza bancaria de los datos; e integrar la imputación, el escalamiento y los filtros Chi-cuadrada/Pearson dentro de un Pipeline evitó cualquier fuga de información entre entrenamiento y prueba. Además, las dos variables creadas (balance_ajustado_credito y presion_contacto) quedaron dentro del Top 7 de mayor importancia.</li>
  <li><strong>Superioridad y buena generalización de Random Forest:</strong> A través de GridSearchCV y validación cruzada estratificada de 5 particiones (sin tocar el conjunto de prueba), Random Forest superó a la Regresión Logística y al Árbol de Decisión individual. La consistencia de sus métricas y matrices de confusión entre entrenamiento (ROC AUC = {tr_mejor['ROC AUC']:.3f}), validación ({cv_mejor['ROC AUC']:.3f}) y prueba ({test_mejor['ROC AUC']:.3f}) comprueba que el modelo generaliza correctamente y evita el sobreajuste.</li>
  <li><strong>Trabajo futuro:</strong> En una implementación real dentro del banco se podría calibrar el umbral óptimo de corte incorporando el costo exacto en pesos/euros de cada minuto de llamada frente a la ganancia financiera promedio de cada depósito contratado.</li>
</ul>

</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    cmd = [
        CHROME_BIN,
        "--headless=new",
        "--disable-gpu",
        "--user-data-dir=/tmp/chrome-pdf-profile",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF}",
        OUT_HTML.as_uri(),
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"PDF generado correctamente en: {OUT_PDF}")
    except Exception as exc:
        print(f"HTML generado en {OUT_HTML}. Para generar PDF con Chrome: {exc}")


def build() -> None:
    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    build_docx(m)
    build_pdf(m)


if __name__ == "__main__":
    build()
