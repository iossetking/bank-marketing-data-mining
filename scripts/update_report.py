from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = "Informe_Bank_Marketing.docx"
BLUE = "17365D"
LIGHT_BLUE = "EAF2F8"
LIGHT_GRAY = "F2F2F2"
BORDER = "D9D9D9"


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def cell_border(cell, color=BORDER):
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


def set_cell_text(cell, text, bold=False, color=None, size=9):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    cell_border(cell)


def table(doc, headers, rows, widths=None, font_size=9):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    for i, header in enumerate(headers):
        cell = t.rows[0].cells[i]
        set_cell_text(cell, header, bold=True, color="FFFFFF", size=font_size)
        shade(cell, BLUE)
        if widths:
            cell.width = Inches(widths[i])
    for row_index, row in enumerate(rows):
        cells = t.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value, size=font_size)
            if row_index % 2:
                shade(cells[i], LIGHT_BLUE)
            if widths:
                cells[i].width = Inches(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def paragraph(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.08
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    p.add_run(text)
    return p


def clear_paragraph_border(paragraph_obj):
    p_pr = paragraph_obj._p.get_or_add_pPr()
    border = p_pr.find(qn("w:pBdr"))
    if border is not None:
        p_pr.remove(border)


def build():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(10.5)
    for style_name in ("Title", "Heading 1", "Heading 2"):
        style = doc.styles[style_name]
        style.font.name = "Aptos Display" if style_name == "Title" else "Aptos"
        style.font.color.rgb = RGBColor(0, 0, 0)

    title = doc.add_paragraph(style="Title")
    clear_paragraph_border(title)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Análisis del conjunto Bank Marketing")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(18)
    run = subtitle.add_run("Informe técnico actualizado de preparación, filtros y evaluación")
    run.italic = True
    run.font.size = Pt(12)

    heading(doc, "Resumen ejecutivo")
    paragraph(
        doc,
        "Se actualizó el flujo de análisis para documentar la calidad de datos, tratar faltantes de forma reproducible y separar los filtros según el tipo de variable. El modelo final obtuvo ROC AUC de 0.752 en prueba y 0.745 en validación cruzada. Esta consistencia indica una capacidad moderada para ordenar clientes por probabilidad de suscripción.",
    )
    paragraph(
        doc,
        "La decisión metodológica más importante fue excluir duration. Esa variable solo se conoce cuando la llamada terminó; usarla para decidir a quién llamar introduciría fuga de información y produciría una estimación poco útil para la operación.",
    )

    heading(doc, "Objetivo y alcance")
    paragraph(
        doc,
        "El objetivo es estimar si un cliente contratará un depósito a plazo fijo para priorizar campañas telefónicas. El conjunto UCI Bank Marketing contiene 45,211 registros, 16 variables descriptivas y la respuesta y. La clase positiva representa 11.7% de los registros, por lo que se emplean particiones estratificadas y métricas que no dependen solo de la exactitud.",
    )

    heading(doc, "Diagnóstico de calidad")
    paragraph(
        doc,
        "La revisión cubrió tipos de dato, faltantes, duplicados, dominios categóricos y observaciones atípicas mediante el criterio IQR y diagramas de caja. Las marcas IQR sirven para revisar casos; no son evidencia suficiente para eliminar clientes o campañas reales.",
    )
    table(
        doc,
        ["Aspecto", "Resultado", "Interpretación y tratamiento"],
        [
            ["Duplicados completos", "0", "No se eliminaron filas por duplicación."],
            ["Faltantes en job", "288 (0.64%)", "Imputación por moda dentro del pipeline."],
            ["Faltantes en education", "1,857 (4.11%)", "Imputación por moda dentro del pipeline."],
            ["Faltantes en contact", "13,020 (28.80%)", "Imputación por moda; se conserva el registro."],
            ["Faltantes en poutcome", "36,959 (81.75%)", "Imputación por moda; se evita eliminar gran parte de la muestra."],
            ["pdays = -1", "36,954", "Se conserva: representa ausencia de contacto previo, no un error."],
            ["Clase positiva", "11.7%", "Se usa estratificación y ponderación balanceada."],
        ],
        widths=[1.5, 1.25, 3.95],
    )
    paragraph(
        doc,
        "Las categorías observadas fueron coherentes con el dominio de la fuente tras retirar espacios y normalizar mayúsculas. No se detectaron valores categóricos inesperados. Las observaciones atípicas IQR se concentraron en pdays y previous (8,257 cada una), balance (4,729), duration (3,235), campaign (3,064) y age (487). Se conservaron porque pueden describir comportamientos válidos; el escalamiento MinMax reduce el efecto de las diferencias de escala.",
    )

    heading(doc, "Decisiones de limpieza")
    table(
        doc,
        ["Problema", "Variables", "Decisión", "Justificación"],
        [
            ["Duplicados exactos", "Todas", "Eliminar solo si existen", "Evita sobreponderar una misma observación sin borrar contactos distintos."],
            ["Formato de texto", "Categóricas y y", "Recortar espacios y pasar a minúsculas", "Evita categorías distintas por captura."],
            ["Faltantes categóricos", "job, education, contact, poutcome", "Moda dentro del pipeline", "Conserva registros y evita fuga de información."],
            ["Extremos IQR", "Numéricas", "Conservar y escalar", "No hay evidencia de error de captura."],
            ["Código especial", "pdays", "Conservar -1", "El valor tiene significado de negocio documentado."],
        ],
        widths=[1.3, 1.55, 1.65, 2.2],
        font_size=8.5,
    )

    heading(doc, "Transformación e ingeniería de características")
    paragraph(doc, "La propuesta se aplicó después de la limpieza. El escalamiento y la imputación se ajustan únicamente con entrenamiento. Las categorías se convierten con one hot encoding, porque no existe una distancia numérica natural entre sus valores.")
    table(
        doc,
        ["Grupo de variables", "Tipo", "Transformación", "Justificación"],
        [
            ["age, balance, day_of_week, campaign, pdays, previous", "Numéricas", "Mediana y MinMax", "Escala comparable y valores no negativos."],
            ["job, marital, education, default, housing, loan, contact, month, poutcome", "Categóricas", "Moda y one hot", "No impone un orden inexistente y maneja faltantes."],
            ["presion_contacto", "Numérica derivada", "campaign / (previous + 1)", "Resume la presión actual frente al historial sin dividir entre cero."],
            ["balance_ajustado_credito", "Numérica derivada", "balance / (1 + housing + loan)", "Aproxima el saldo considerando créditos reportados."],
            ["duration", "Numérica", "Excluir", "No se conoce antes de realizar la llamada."],
            ["y", "Objetivo", "yes=1, no=0", "Permite clasificación binaria y correlación con numéricas."],
        ],
        widths=[2.0, 1.1, 1.85, 2.0],
        font_size=8.5,
    )

    heading(doc, "Filtros de selección de características")
    paragraph(doc, "Los filtros se estimaron solo con las 36,168 filas de entrenamiento. Chi-cuadrada analiza variables categóricas frente a y y se resume con V de Cramér; Pearson analiza variables numéricas frente a y codificada 0/1 y se ordena por el valor absoluto de r. Sus puntajes no se comparan entre sí, porque están en escalas diferentes.")
    table(
        doc,
        ["Filtro", "Variables", "Resultado principal", "Interpretación"],
        [
            ["Chi-cuadrada", "Categóricas", "poutcome: V=0.474; month: V=0.257; housing: V=0.139", "El resultado previo y la estacionalidad son las asociaciones categóricas más fuertes."],
            ["Pearson", "Numéricas", "pdays: r=0.101; presion_contacto: r=-0.092; previous: r=0.089", "Las relaciones lineales son débiles pero informativas; conviene combinarlas con otras variables."],
        ],
        widths=[1.3, 1.2, 2.55, 2.0],
        font_size=8.5,
    )
    paragraph(doc, "El notebook incorpora un ranking de barras para cada filtro, tablas de contingencia para las cuatro categóricas con mayor asociación y una matriz de Pearson de las numéricas junto al objetivo. Dentro de la validación cruzada, ambos filtros se vuelven a ajustar en cada pliegue para preservar la independencia de validación.")

    heading(doc, "Entrenamiento y evaluación")
    paragraph(doc, "Se reservó 20% de los registros para prueba (9,043 filas) y se usó validación cruzada estratificada de cinco particiones sobre el resto. La regresión logística usa class_weight balanced. Además de ROC AUC se reportan exactitud balanceada, F1, precisión y sensibilidad para no ocultar el efecto del desbalance.")
    table(
        doc,
        ["Métrica", "Validación cruzada", "Prueba independiente", "Lectura"],
        [
            ["ROC AUC", "0.745", "0.752", "Discriminación moderada y estable."],
            ["Exactitud", "No es criterio principal", "0.764", "Puede ser alta aun sin detectar suficientes positivos."],
            ["Exactitud balanceada", "0.688", "0.689", "Rendimiento similar entre ambas clases."],
            ["F1", "0.368", "0.369", "Equilibrio limitado entre precisión y sensibilidad."],
            ["Precisión", "0.268", "0.269", "Cerca de 27 de cada 100 marcados sí se suscriben."],
            ["Sensibilidad", "0.588", "0.590", "Se detectan cerca de 59 de cada 100 suscriptores."],
        ],
        widths=[1.55, 1.65, 1.55, 2.3],
    )

    heading(doc, "Conclusión")
    paragraph(doc, "El flujo actualizado mejora la trazabilidad de cada decisión: evidencia los faltantes, justifica conservar observaciones atípicas, define transformaciones por tipo de variable y compara filtros apropiados para categóricas y numéricas. El desempeño en prueba es coherente con la validación cruzada, por lo que el modelo es una base razonable para ordenar clientes.")
    paragraph(doc, "El umbral de clasificación debe decidirse según el costo de llamada y el valor de una suscripción. Como siguiente paso se recomienda ajustar ese umbral con información de negocio y probar una validación temporal si se dispone de fechas completas de campañas.")

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Bank Marketing - Informe técnico actualizado")
    footer.runs[0].font.size = Pt(8)

    doc.save(OUT)


if __name__ == "__main__":
    build()
