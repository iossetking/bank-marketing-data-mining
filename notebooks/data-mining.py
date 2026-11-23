import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bank Marketing Data Mining
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bank Marketing Dataset

    Dataset sobre campañas de marketing de un banco Portugues. Las campañas de marketing se basan en llamadas telefonicas, cada registro representa un cliente e informacion sobre la campaña de marketing (se puede repetir el cliente debido a que en algunos casos se contactaron varias veces al mismo cliente).

    UCI Bank Marketing (https://archive.ics.uci.edu/dataset/222/bank+marketing)

    - Numero de registros: 45,211
    - Numero de caracteristicas: 16
    - Variable ojetivo: `y` (Representa si un cliente se suscribio/contrato o no a un deposito a plazo fijo).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Nombre Variable | Descripcion |
    | --- | --- |
    | `age` | Edad del cliente, en años. |
    | `job` | Tipo de trabajo u ocupación: admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed o unknown. |
    | `marital` | Estado civil: divorced, married, single o unknown. `divorced` también incluye a personas viudas. |
    | `education` | Nivel educativo: primary, secondary o tertiary; algunos registros no lo reportan. |
    | `default` | Indica si el cliente tiene crédito en mora (yes/no). |
    | `balance` | Saldo anual promedio del cliente, en euros. |
    | `housing` | Indica si el cliente tiene un préstamo hipotecario (yes/no). |
    | `loan` | Indica si el cliente tiene un préstamo personal (yes/no). |
    | `contact` | Tipo de medio de contacto: telephone o cellular; algunos registros no lo reportan. |
    | `day_of_week` | Día del mes del último contacto de esta campaña (nombre conservado de la fuente). |
    | `month` | Mes del último contacto de esta campaña (jan–dec). |
    | `duration` | Duración del último contacto, en segundos. |
    | `campaign` | Número de contactos realizados durante esta campaña para este cliente, incluido el último contacto. |
    | `pdays` | Días desde el último contacto del cliente en una campaña anterior; `-1` indica que no había sido contactado anteriormente. |
    | `previous` | Número de contactos realizados para este cliente antes de esta campaña. |
    | `poutcome` | Resultado de la campaña anterior: other, failure o success; los faltantes indican que no se registró resultado previo. |
    | `y` | Variable objetivo: indica si el cliente suscribió un depósito a plazo fijo (yes/no). |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Planeamiento del Problema

    El banco realiza campañas de marketing telefónico para ofrecer depósitos a plazo fijo. Como contactar a los clientes requiere recursos y, en algunos casos, varios intentos, es útil identificar de antemano cuáles tienen mayor probabilidad de aceptar la oferta. El problema consiste en utilizar la información de los clientes y de las campañas anteriores para predecir si un cliente suscribirá el depósito, resultado representado por la variable y (yes o no).

    **Objetivo**: Desarrollar y evaluar un modelo de clasificación que, utilizando las variables disponibles del conjunto de datos Bank Marketing, prediga si un cliente suscribirá un depósito a plazo fijo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Importacion de Librerias
    """)
    return


@app.cell
def _():
    # Se importan las librerías para manipular datos, graficar y entrenar los modelos.
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier
    from scipy.stats import chi2_contingency, pearsonr
    from ucimlrepo import fetch_ucirepo

    return (
        ColumnTransformer,
        ConfusionMatrixDisplay,
        DecisionTreeClassifier,
        LogisticRegression,
        MinMaxScaler,
        OneHotEncoder,
        Path,
        Pipeline,
        RandomForestClassifier,
        SelectKBest,
        SimpleImputer,
        StratifiedKFold,
        accuracy_score,
        balanced_accuracy_score,
        chi2,
        chi2_contingency,
        classification_report,
        cross_validate,
        f1_score,
        fetch_ucirepo,
        np,
        pd,
        pearsonr,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Carga de Dataset
    """)
    return


@app.cell
def _(Path, fetch_ucirepo, pd):
    # Se carga la versión original del repositorio UCI (con respaldo local si no hay conexión)
    # para conservar los nombres y valores reportados por la fuente.
    ruta_local = Path("data/bank_marketing_uci.csv")
    if not ruta_local.exists():
        ruta_local = Path("../data/bank_marketing_uci.csv")

    if ruta_local.exists():
        df_uci = pd.read_csv(ruta_local)
        X = df_uci.drop(columns=["y"])
        y = df_uci[["y"]]
    else:
        bank_marketing = fetch_ucirepo(id=222)
        X = bank_marketing.data.features
        y = bank_marketing.data.targets

    print(f"Registros cargados: {X.shape[0]:,}; variables predictoras: {X.shape[1]}")
    return X, y


@app.cell
def _(X):
    # Se revisa la forma y las primeras filas para confirmar una carga correcta.
    print(f'X Shape: {X.shape}')
    X.head()
    return


@app.cell
def _(y):
    # Se comprueba que la variable objetivo tenga una sola columna y valores esperados.
    print(f'y Shape: {y.shape}')
    y.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Diagnostico de Calidad de los Datos

    Identificar valores faltantes, duplicados, inconsistencias y valores atípicos.
    """)
    return


@app.cell
def _(X, pd, y):
    # Se integra la respuesta con los predictores para diagnosticar el conjunto completo.
    datos_brutos = pd.concat([X.copy(), y.copy()], axis=1)
    datos_brutos.columns = [str(col).strip() for col in datos_brutos.columns]
    return (datos_brutos,)


@app.cell
def _(datos_brutos, pd):
    # Se resumen tipos, faltantes, duplicados y cardinalidad de cada variable.
    resumen_calidad = pd.DataFrame(
        {
            "tipo": datos_brutos.dtypes.astype(str),
            "faltantes": datos_brutos.isna().sum(),
            "porcentaje_faltante": (datos_brutos.isna().mean() * 100).round(2),
            "valores_unicos": datos_brutos.nunique(),
        }
    )
    duplicados_brutos = int(datos_brutos.duplicated().sum())
    print(f"Filas duplicadas completas: {duplicados_brutos:,}")
    print(
        "Registros con pdays = -1 (sin contacto previo): "
        f"{(datos_brutos['pdays'] == -1).sum():,}"
    )
    resumen_calidad
    return (duplicados_brutos,)


@app.cell
def _(datos_brutos, pd):
    # Se listan los valores únicos de cada variable numérica para detectar códigos especiales
    # (por ejemplo pdays = -1) y confirmar los rangos antes de graficar los atípicos.
    # Las variables continuas con cientos de valores se resumen con su rango para no saturar la salida.
    _maximo_valores_listados = 100
    _filas_valores = []
    for _columna in datos_brutos.select_dtypes(include="number").columns:
        _valores = sorted(datos_brutos[_columna].dropna().unique())
        if len(_valores) <= _maximo_valores_listados:
            _detalle = ", ".join(f"{_valor:g}" for _valor in _valores)
        else:
            _detalle = (
                f"{len(_valores):,} valores continuos "
                f"entre {_valores[0]:g} y {_valores[-1]:g}"
            )
        _filas_valores.append(
            {
                "variable": _columna,
                "n_unicos": len(_valores),
                "minimo": _valores[0],
                "maximo": _valores[-1],
                "valores_unicos": _detalle,
            }
        )
    valores_unicos_numericos = pd.DataFrame(_filas_valores).set_index("variable")
    print("Valores únicos de las variables numéricas:")
    valores_unicos_numericos
    return


@app.cell
def _(datos_brutos, np, pd, plt, sns):
    # IQR se usa como señal de revisión, no como regla automática de eliminación.
    _columnas_numericas_diagnostico = datos_brutos.select_dtypes(include="number").columns.tolist()
    _cuantiles = datos_brutos[_columnas_numericas_diagnostico].quantile([0.25, 0.75]).T
    _cuantiles.columns = ["q1", "q3"]
    _cuantiles["iqr"] = _cuantiles["q3"] - _cuantiles["q1"]
    _cuantiles["limite_inferior"] = _cuantiles["q1"] - 1.5 * _cuantiles["iqr"]
    _cuantiles["limite_superior"] = _cuantiles["q3"] + 1.5 * _cuantiles["iqr"]
    _conteo_atipicos = pd.DataFrame(
        {
            "valores_atipicos_iqr": [
                int(
                    (
                        (datos_brutos[_col] < _cuantiles.loc[_col, "limite_inferior"])
                        | (datos_brutos[_col] > _cuantiles.loc[_col, "limite_superior"])
                    ).sum()
                )
                for _col in _columnas_numericas_diagnostico
            ]
        },
        index=_columnas_numericas_diagnostico,
    )
    diagnostico_atipicos = _cuantiles.join(_conteo_atipicos).round(2)

    _n_columnas = 3
    _n_filas = int(np.ceil(len(_columnas_numericas_diagnostico) / _n_columnas))
    _fig_atipicos, _ejes_atipicos = plt.subplots(_n_filas, _n_columnas, figsize=(14, 3.6 * _n_filas))
    for _eje, _columna in zip(_ejes_atipicos.flat, _columnas_numericas_diagnostico):
        sns.boxplot(x=datos_brutos[_columna], ax=_eje, color="#8ecae6")
        _eje.set_title(f"{_columna}: distribución y atípicos IQR")
        _eje.set_xlabel(_columna)
    for _eje in _ejes_atipicos.flat[len(_columnas_numericas_diagnostico):]:
        _eje.set_visible(False)
    _fig_atipicos.suptitle("Diagnóstico visual de valores atípicos", y=1.02, fontsize=14)
    _fig_atipicos.tight_layout()
    plt.show()
    diagnostico_atipicos
    return


@app.cell
def _(datos_brutos, pd):
    # Se comparan las categorías observadas con los dominios documentados después de normalizar texto.
    _dominios_esperados = {
        "job": {"admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed"},
        "marital": {"divorced", "married", "single"},
        "education": {"primary", "secondary", "tertiary"},
        "default": {"no", "yes"},
        "housing": {"no", "yes"},
        "loan": {"no", "yes"},
        "contact": {"cellular", "telephone"},
        "month": {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"},
        "poutcome": {"failure", "other", "success"},
        "y": {"no", "yes"},
    }
    _filas_consistencia = []
    for _columna, _dominio in _dominios_esperados.items():
        _valores = datos_brutos[_columna].astype("string").str.strip().str.lower()
        _observadas = set(_valores.dropna().unique())
        _inesperadas = sorted(_observadas.difference(_dominio))
        _filas_consistencia.append(
            {
                "variable": _columna,
                "categorias_observadas": ", ".join(sorted(_observadas)),
                "faltantes": int(_valores.isna().sum()),
                "blancos": int(_valores.eq("").sum()),
                "categorias_inesperadas": ", ".join(_inesperadas) if _inesperadas else "ninguna",
                "resultado": "dominio consistente" if not _inesperadas and not _valores.eq("").any() else "requiere revisión",
            }
        )
    consistencia_categoricas = pd.DataFrame(_filas_consistencia)
    consistencia_categoricas
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretación del diagnóstico.** Las marcas IQR y los puntos extremos de los diagramas de caja
    identifican observaciones que merecen revisión, pero no son por sí mismos errores. En marketing
    bancario, saldos elevados, muchos contactos o intervalos largos entre campañas pueden describir
    clientes reales; eliminarlos sin evidencia cambiaría artificialmente la población objetivo. Por
    ello se conservarán los valores numéricos y se aplicará escalamiento robusto a la escala de cada
    variable. La tabla de consistencia permite confirmar que, una vez retirados espacios, las
    categorías están dentro de los dominios documentados. Los faltantes de las categóricas no se
    eliminan: se imputarán con la moda dentro del pipeline para no perder registros. `pdays = -1`
    se mantiene porque expresa ausencia de contacto previo y puede ser predictivo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Limpieza de Datos

    La limpieza conserva la información de negocio y corrige solamente problemas verificables de formato
    o duplicación. La siguiente tabla deja trazabilidad de cada decisión antes de ejecutarla.

    | Problema identificado | Variables | Decisión | Justificación |
    | --- | --- | --- | --- |
    | Filas duplicadas exactas | Todas | Eliminar solo registros idénticos en todas las columnas | Evita sobreponderar una misma observación sin eliminar llamadas distintas de un cliente. |
    | Espacios o diferencias de formato | Categóricas y objetivo | Recortar espacios y homogeneizar a minúsculas al evaluar dominios | Previene categorías duplicadas por captura; no altera su significado. |
    | Faltantes categóricos | `job`, `education`, `contact`, `poutcome` | Imputar con la moda dentro del pipeline | Conserva registros y evita que estadísticos del conjunto de prueba influyan en entrenamiento. |
    | `pdays = -1` | `pdays` | Conservar el valor | La documentación lo define como ausencia de contacto previo, no como dato faltante. |
    | Valores atípicos IQR | Numéricas | Conservar y escalar; no truncar | No hay evidencia de error y los extremos pueden ser clientes o campañas reales. |
    | Faltantes futuros | Numéricas y categóricas | Imputar dentro del pipeline de entrenamiento | Mantiene el flujo reproducible y evita fuga de información. |
    """)
    return


@app.cell
def _(datos_brutos, duplicados_brutos):
    # Se eliminan solo duplicados exactos. No se eliminan clientes repetidos porque cada llamada
    # puede corresponder a un intento distinto y aporta información de campaña.
    datos_limpios = datos_brutos.drop_duplicates().copy()

    # Se normalizan espacios en variables de texto para evitar categorías duplicadas por formato.
    columnas_texto_limpias = datos_limpios.select_dtypes(include=["object", "string"]).columns
    for _columna in columnas_texto_limpias:
        datos_limpios[_columna] = datos_limpios[_columna].str.strip().str.lower()

    # La imputación posterior queda dentro para no filtrar información del conjunto de prueba.
    print(f"Registros eliminados por duplicado: {duplicados_brutos:,}")
    print(f"Registros disponibles después de limpiar: {len(datos_limpios):,}")
    return (datos_limpios,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Transformación de Variables y 5. Ingeniería de características

    Las transformaciones se definen antes de ejecutarse para distinguir los datos disponibles al
    momento de priorizar una llamada de los datos que solo se conocen después. La imputación, el
    escalamiento y la codificación se ajustan únicamente dentro del entrenamiento.

    | Variable | Tipo de variable | Transformación propuesta | Justificación |
    | --- | --- | --- | --- |
    | `age` | Numérica | Mediana y MinMax | Atenúa el efecto de una escala distinta y mantiene valores no negativos. |
    | `job` | Categórica nominal | Moda y one-hot | No existe un orden natural entre ocupaciones. |
    | `marital` | Categórica nominal | Moda y one-hot | Representa estados sin jerarquía cuantitativa. |
    | `education` | Categórica ordinal aproximada | Moda y one-hot | Evita imponer distancias arbitrarias entre niveles educativos. |
    | `default` | Categórica binaria | `yes` a 1 y `no` a 0 | Una sola columna 0/1 conserva el indicador sin duplicarlo en dos columnas colineales. |
    | `balance` | Numérica | Mediana y MinMax | Conserva la señal del saldo en una escala comparable. |
    | `housing` | Categórica binaria | `yes` a 1 y `no` a 0 | Una sola columna 0/1 conserva el indicador de crédito hipotecario. |
    | `loan` | Categórica binaria | `yes` a 1 y `no` a 0 | Una sola columna 0/1 conserva el indicador de crédito personal. |
    | `contact` | Categórica nominal | Moda y one-hot | Los canales no tienen orden intrínseco. |
    | `day_of_week` | Numérica discreta | Mediana y MinMax | Mantiene el momento mensual sin crear una jerarquía categórica. |
    | `month` | Categórica temporal | Moda y one-hot | Permite patrones estacionales sin asumir distancia lineal entre meses. |
    | `duration` | Numérica | Excluir | Solo se conoce al terminar la llamada; incluirla causaría fuga de información. |
    | `campaign` | Numérica discreta | Mediana y MinMax y derivada | Resume la presión actual de contacto. |
    | `pdays` | Numérica con código especial | Mediana y MinMax; conservar `-1` | El código identifica explícitamente falta de contacto previo. |
    | `previous` | Numérica discreta | Mediana y MinMax y derivada | Aporta historial de campañas anteriores. |
    | `poutcome` | Categórica nominal | Moda y one-hot | El resultado previo puede tener asociación con la suscripción. |
    | `presion_contacto` | Numérica derivada | `campaign / (previous + 1)` y MinMax | Contrasta insistencia actual con historial previo, sin dividir entre cero. |
    | `balance_ajustado_credito` | Numérica derivada | `balance / (1 + housing + loan)` y MinMax | Aproxima el saldo disponible considerando créditos reportados. |
    | `y` | Objetivo binario | `yes` a 1 y `no` a 0 | Permite estimar una probabilidad de suscripción. |
    """)
    return


@app.cell
def _(datos_limpios):
    # Se crean exactamente dos variables derivadas mediante operaciones entre columnas existentes.
    # Ambas están disponibles antes de hacer una llamada y no usan la variable objetivo.
    datos_modelo = datos_limpios.copy()

    # Las tres categóricas binarias se codifican a 0/1 (yes = 1, no = 0). Se conservan en el
    # grupo categórico para que la chi-cuadrada las analice y no se traten como numéricas.
    for _columna_binaria in ["default", "housing", "loan"]:
        datos_modelo[_columna_binaria] = (
            datos_modelo[_columna_binaria].eq("yes").astype(int)
        )

    # Relaciona los contactos de la campaña actual con los de campañas previas.
    # Un valor alto señala insistencia reciente; un valor bajo indica más historial que presión actual.
    datos_modelo["presion_contacto"] = datos_modelo["campaign"] / (
        datos_modelo["previous"] + 1
    )

    # Ajusta el saldo por la cantidad de créditos activos reportados (vivienda y personal).
    # Sirve como aproximación de la capacidad financiera disponible para aceptar un depósito.
    datos_modelo["balance_ajustado_credito"] = datos_modelo["balance"] / (
        1 + datos_modelo["housing"] + datos_modelo["loan"]
    )

    # Se muestra un ejemplo para comprobar que las dos columnas sí fueron creadas.
    print("Ejemplo de variables derivadas:")
    print(
        datos_modelo[
            [
                "campaign",
                "previous",
                "presion_contacto",
                "balance",
                "housing",
                "loan",
                "balance_ajustado_credito",
            ]
        ].head()
    )

    # duration se excluye: se conoce al finalizar la llamada y no estaría disponible
    # cuando el banco tenga que decidir a quién contactar. Incluirla inflaría el desempeño.
    X_modelo = datos_modelo.drop(columns=["y", "duration"])
    y_modelo = (datos_modelo["y"].str.lower() == "yes").astype(int)

    print("Distribución de la respuesta (0 = no, 1 = sí):")
    print(y_modelo.value_counts(normalize=True).rename("proporción").round(4))
    return X_modelo, y_modelo


@app.cell
def _(X_modelo):
    # Se identifican columnas por tipo para aplicar una preparación adecuada a cada grupo.
    # default, housing y loan ya están codificadas 0/1, pero siguen siendo categóricas binarias:
    # se declaran explícitamente para que la chi-cuadrada las analice en lugar de Pearson.
    columnas_binarias = ["default", "housing", "loan"]
    columnas_numericas = [
        columna
        for columna in X_modelo.select_dtypes(include="number").columns
        if columna not in columnas_binarias
    ]
    columnas_categoricas = [
        *X_modelo.select_dtypes(exclude="number").columns,
        *columnas_binarias,
    ]
    print(f"Variables numéricas: {len(columnas_numericas)}")
    print(f"Variables categóricas: {len(columnas_categoricas)}")
    return columnas_categoricas, columnas_numericas


@app.cell
def _(
    ColumnTransformer,
    MinMaxScaler,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    columnas_categoricas,
    columnas_numericas,
):
    # Las numéricas se imputan con mediana y se escalan a [0,1], condición necesaria para chi-cuadrada.
    # Las categóricas se imputan con la moda y se convierten a indicadores mediante one-hot encoding.
    # drop="if_binary" deja una sola columna 0/1 en las binarias (default, housing, loan, contact)
    # en lugar de dos columnas colineales que duplicarían la misma información.
    transformador_numerico = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("escala", MinMaxScaler()),
        ]
    )
    transformador_categorico = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="most_frequent")),
            (
                "codificacion",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", transformador_numerico, columnas_numericas),
            ("categoricas", transformador_categorico, columnas_categoricas),
        ],
        remainder="drop",
    )
    return transformador_categorico, transformador_numerico


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Partición del conjunto de datos en entrenamiento y prueba

    Se reserva una muestra independiente antes de cualquier filtro. La estratificación mantiene en
    entrenamiento y prueba la proporción de suscripciones, que es minoritaria, y permite que la
    evaluación final represente la población original.
    """)
    return


@app.cell
def _(X_modelo, train_test_split, y_modelo):
    # Se reserva 20% para una evaluación final independiente. La estratificación conserva
    # la proporción de suscripciones en ambos subconjuntos, importante por el desbalance.
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X_modelo,
        y_modelo,
        test_size=0.20,
        random_state=42,
        stratify=y_modelo,
    )
    print(f"Entrenamiento: {X_entrenamiento.shape[0]:,} filas")
    print(f"Prueba: {X_prueba.shape[0]:,} filas")
    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Aplicación de métodos de filtro para la selección de características

    Los filtros se calculan solo sobre entrenamiento. Cada variable descriptiva se contrasta con
    `y`: chi-cuadrada para categóricas y Pearson para numéricas. Los puntajes miden asociaciones,
    no causalidad, y sirven para priorizar variables que deben revisarse junto con el contexto de
    negocio.

    | Criterio | Chi-cuadrada | Pearson |
    | --- | --- | --- |
    | Variables analizadas | Categóricas frente al objetivo binario | Numéricas frente al objetivo binario codificado 0/1 |
    | Evidencia reportada | Estadístico, p-valor y V de Cramér | Correlación `r`, p-valor y valor absoluto de `r` |
    | Lectura del ranking | Mayor V de Cramér indica asociación categórica más fuerte | Mayor `abs(r)` indica asociación lineal más fuerte |
    | Comparación entre filtros | No se comparan sus magnitudes directamente: usan escalas distintas. Se comparan las variables relevantes dentro de su propio tipo. | No se comparan sus magnitudes directamente: usan escalas distintas. Se comparan las variables relevantes dentro de su propio tipo. |
    """)
    return


@app.cell
def _(
    X_entrenamiento,
    chi2_contingency,
    columnas_categoricas,
    pd,
    plt,
    sns,
    y_entrenamiento,
):
    # Método 1: una tabla de contingencia por variable categórica frente a la respuesta.
    # Se usa la lista explícita para incluir las binarias ya codificadas 0/1.
    _columnas_categoricas_filtro = columnas_categoricas
    _resultados_chi = []
    _tablas_contingencia = {}
    for _columna in _columnas_categoricas_filtro:
        _tabla = pd.crosstab(X_entrenamiento[_columna], y_entrenamiento)
        _estadistico, _p_valor, _grados_libertad, _ = chi2_contingency(_tabla)
        _n = _tabla.to_numpy().sum()
        # V de Cramér permite comparar asociaciones con distinta cantidad de categorías.
        _minimo_dimension = min(_tabla.shape) - 1
        _v_cramer = (_estadistico / (_n * _minimo_dimension)) ** 0.5 if _minimo_dimension else 0.0
        _resultados_chi.append(
            {
                "variable": _columna,
                "chi_cuadrada": _estadistico,
                "p_valor": _p_valor,
                "grados_libertad": _grados_libertad,
                "v_cramer": _v_cramer,
            }
        )
        _tablas_contingencia[_columna] = _tabla
    ranking_chi2 = (
        pd.DataFrame(_resultados_chi)
        .sort_values(["v_cramer", "chi_cuadrada"], ascending=False)
        .reset_index(drop=True)
    )

    _fig_ranking_chi, _eje_ranking_chi = plt.subplots(figsize=(9, 5))
    sns.barplot(data=ranking_chi2, x="v_cramer", y="variable", hue="variable", legend=False, palette="Blues_r", ax=_eje_ranking_chi)
    _eje_ranking_chi.set_title("Ranking chi-cuadrada de categóricas frente a suscripción")
    _eje_ranking_chi.set_xlabel("V de Cramér (fuerza de asociación)")
    _eje_ranking_chi.set_ylabel("Variable descriptiva")
    _fig_ranking_chi.tight_layout()

    _variables_chi_grafica = ranking_chi2.head(4)["variable"].tolist()
    _fig_tablas_chi, _ejes_tablas_chi = plt.subplots(2, 2, figsize=(12, 8))
    for _eje, _columna in zip(_ejes_tablas_chi.flat, _variables_chi_grafica):
        sns.heatmap(_tablas_contingencia[_columna], annot=True, fmt="d", cmap="Blues", cbar=False, ax=_eje)
        _eje.set_title(f"{_columna} vs y")
        _eje.set_xlabel("Suscripción (0=no, 1=sí)")
        _eje.set_ylabel(_columna)
    _fig_tablas_chi.suptitle("Tablas de contingencia de las categóricas con mayor asociación", y=1.02, fontsize=14)
    _fig_tablas_chi.tight_layout()
    plt.show()
    ranking_chi2.round(4)
    return (ranking_chi2,)


@app.cell(hide_code=True)
def _(mo, ranking_chi2):
    principal_chi = ranking_chi2.iloc[0]
    mo.md(
        f"""
        **Interpretación de chi-cuadrada.** `{principal_chi['variable']}` presenta la mayor asociación
        categórica con la suscripción (V de Cramér = **{principal_chi['v_cramer']:.3f}**; p-valor =
        **{principal_chi['p_valor']:.3g}**). Las gráficas de contingencia muestran cómo se distribuyen
        los sí y no dentro de cada categoría. Las variables con p-valor bajo son candidatas a conservar,
        pero el tamaño de efecto evita confundir un resultado significativo por el gran tamaño muestral
        con una asociación sustantiva.
        """
    )
    return


@app.cell
def _(
    X_entrenamiento,
    columnas_numericas,
    pd,
    pearsonr,
    plt,
    sns,
    y_entrenamiento,
):
    # Método 2: Pearson cuantifica la asociación lineal de cada numérica con y codificada 0/1.
    # La lista explícita excluye las binarias 0/1, que se evalúan con chi-cuadrada.
    _columnas_numericas_filtro = columnas_numericas
    _resultados_pearson = []
    for _columna in _columnas_numericas_filtro:
        _correlacion, _p_valor = pearsonr(X_entrenamiento[_columna], y_entrenamiento)
        _resultados_pearson.append(
            {
                "variable": _columna,
                "correlacion_pearson": _correlacion,
                "abs_correlacion": abs(_correlacion),
                "p_valor": _p_valor,
            }
        )
    ranking_pearson = (
        pd.DataFrame(_resultados_pearson)
        .sort_values("abs_correlacion", ascending=False)
        .reset_index(drop=True)
    )

    _matriz_pearson = X_entrenamiento[_columnas_numericas_filtro].copy()
    _matriz_pearson["y"] = y_entrenamiento.to_numpy()
    _fig_matriz_pearson, _eje_matriz_pearson = plt.subplots(figsize=(11, 8))
    sns.heatmap(_matriz_pearson.corr(method="pearson"), cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, ax=_eje_matriz_pearson)
    _eje_matriz_pearson.set_title("Matriz de correlación de Pearson: numéricas y objetivo")
    _fig_matriz_pearson.tight_layout()

    _fig_ranking_pearson, _eje_ranking_pearson = plt.subplots(figsize=(9, 5))
    sns.barplot(data=ranking_pearson, x="correlacion_pearson", y="variable", hue="variable", legend=False, palette="vlag", ax=_eje_ranking_pearson)
    _eje_ranking_pearson.axvline(0, color="black", linewidth=0.8)
    _eje_ranking_pearson.set_title("Ranking Pearson de numéricas frente a suscripción")
    _eje_ranking_pearson.set_xlabel("Correlación de Pearson con y")
    _eje_ranking_pearson.set_ylabel("Variable descriptiva")
    _fig_ranking_pearson.tight_layout()
    plt.show()
    ranking_pearson.round(4)
    return (ranking_pearson,)


@app.cell(hide_code=True)
def _(mo, ranking_pearson):
    principal_pearson = ranking_pearson.iloc[0]
    direccion = "positiva" if principal_pearson["correlacion_pearson"] > 0 else "negativa"
    mo.md(
        f"""
        **Interpretación de Pearson.** `{principal_pearson['variable']}` tiene la mayor asociación lineal
        absoluta entre las numéricas (r = **{principal_pearson['correlacion_pearson']:.3f}**, asociación
        {direccion}; p-valor = **{principal_pearson['p_valor']:.3g}**). En la matriz, los tonos cercanos
        a cero indican relaciones lineales débiles; por ello una variable puede ser útil en un modelo aun
        si no domina este ranking, especialmente si su efecto depende de otras variables.
        """
    )
    return


@app.cell
def _(pd, ranking_chi2, ranking_pearson):
    # Comparación consolidada: cada ranking se conserva en su propia escala de asociación.
    resumen_filtros = pd.concat(
        [
            ranking_chi2.assign(metodo="Chi-cuadrada", medida="V de Cramér", puntaje=ranking_chi2["v_cramer"])[["metodo", "variable", "medida", "puntaje", "p_valor"]],
            ranking_pearson.assign(metodo="Pearson", medida="abs(r)", puntaje=ranking_pearson["abs_correlacion"])[["metodo", "variable", "medida", "puntaje", "p_valor"]],
        ],
        ignore_index=True,
    ).sort_values(["metodo", "puntaje"], ascending=[True, False])
    print("Comparación de variables descriptivas frente al objetivo, ordenada dentro de cada filtro:")
    resumen_filtros.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Comparación de modelos con validación cruzada (5 particiones)

    Para comparar distintos algoritmos bajo las mismas condiciones, probamos tres modelos clásicos
    de clasificación supervisada dentro del mismo pipeline (con imputación, escalamiento y filtros
    Chi-cuadrada y Pearson en cada partición para evitar fuga de datos):

    1. **Regresión Logística** (modelo lineal base).
    2. **Árbol de Decisión** (`max_depth=6`, `min_samples_leaf=50`).
    3. **Random Forest** (150 árboles, `max_depth=12`, `min_samples_leaf=20`).

    Los tres usan `class_weight="balanced"` porque la clase positiva (sí contrata) es minoritaria (11.7%).
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    DecisionTreeClassifier,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    SelectKBest,
    chi2,
    columnas_categoricas,
    columnas_numericas,
    np,
    pearsonr,
    transformador_categorico,
    transformador_numerico,
):
    # Función de Pearson para usarla dentro de SelectKBest.
    def puntaje_pearson(X, y):
        puntajes, p_valores = [], []
        for indice in range(X.shape[1]):
            r, p_valor = pearsonr(X[:, indice], y)
            puntajes.append(0.0 if np.isnan(r) else abs(r))
            p_valores.append(1.0 if np.isnan(p_valor) else p_valor)
        return np.asarray(puntajes), np.asarray(p_valores)

    def crear_preprocesador():
        return ColumnTransformer(
            transformers=[
                (
                    "numericas",
                    Pipeline(
                        steps=[
                            ("preparacion", transformador_numerico),
                            ("seleccion_pearson", SelectKBest(score_func=puntaje_pearson, k=6)),
                        ]
                    ),
                    columnas_numericas,
                ),
                (
                    "categoricas",
                    Pipeline(
                        steps=[
                            ("preparacion", transformador_categorico),
                            ("seleccion_chi2", SelectKBest(score_func=chi2, k=25)),
                        ]
                    ),
                    columnas_categoricas,
                ),
            ],
            remainder="drop",
        )

    modelos = {
        "Regresión Logística": Pipeline(
            steps=[
                ("prep", crear_preprocesador()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
        "Árbol de Decisión": Pipeline(
            steps=[
                ("prep", crear_preprocesador()),
                ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("prep", crear_preprocesador()),
                ("clf", RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1)),
            ]
        ),
    }
    return (modelos,)


@app.cell
def _(
    StratifiedKFold,
    X_entrenamiento,
    cross_validate,
    modelos,
    pd,
    plt,
    sns,
    y_entrenamiento,
):
    # Validación cruzada estratificada de 5 particiones sobre los tres modelos.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    _filas_cv = []
    for _nombre, _pipe in modelos.items():
        _res = cross_validate(_pipe, X_entrenamiento, y_entrenamiento, cv=cv, scoring=scoring, n_jobs=-1)
        _filas_cv.append(
            {
                "Modelo": _nombre,
                "ROC AUC": _res["test_roc_auc"].mean(),
                "ROC AUC std": _res["test_roc_auc"].std(),
                "Exactitud balanceada": _res["test_balanced_accuracy"].mean(),
                "F1": _res["test_f1"].mean(),
                "Precisión": _res["test_precision"].mean(),
                "Sensibilidad": _res["test_recall"].mean(),
            }
        )

    resultados_validacion = pd.DataFrame(_filas_cv).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    mejor_modelo_nombre = resultados_validacion.iloc[0]["Modelo"]

    _df_graf = resultados_validacion.melt(
        id_vars="Modelo",
        value_vars=["ROC AUC", "Exactitud balanceada", "F1", "Precisión", "Sensibilidad"],
        var_name="Métrica",
        value_name="Valor",
    )
    _fig, _ax = plt.subplots(figsize=(9.5, 4.5))
    sns.barplot(data=_df_graf, x="Métrica", y="Valor", hue="Modelo", palette="Blues_d", ax=_ax)
    _ax.set_title("Comparación de modelos en validación cruzada (5 particiones)")
    _ax.set_ylim(0, 1.0)
    for _c in _ax.containers:
        _ax.bar_label(_c, fmt="%.2f", fontsize=8, padding=2)
    _fig.tight_layout()
    plt.show()

    print(f"Mejor modelo en validación cruzada: {mejor_modelo_nombre}")
    resultados_validacion.round(3)
    return mejor_modelo_nombre, resultados_validacion


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Evaluación en prueba y priorización de llamadas

    Ahora entrenamos los tres modelos con todo el conjunto de entrenamiento (80%) y los evaluamos
    en el conjunto de prueba (20%) para ver sus curvas ROC, la matriz de confusión del mejor modelo,
    las variables más importantes y qué tanto mejora la tasa de suscripción al priorizar llamadas.
    """)
    return


@app.cell
def _(
    X_entrenamiento,
    X_prueba,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mejor_modelo_nombre,
    modelos,
    pd,
    precision_score,
    recall_score,
    roc_auc_score,
    y_entrenamiento,
    y_prueba,
):
    modelos_ajustados = {}
    probabilidades_prueba = {}
    predicciones_prueba = {}
    _filas_prueba = []

    for _nombre, _pipe in modelos.items():
        _pipe.fit(X_entrenamiento, y_entrenamiento)
        _pred = _pipe.predict(X_prueba)
        _prob = _pipe.predict_proba(X_prueba)[:, 1]
        modelos_ajustados[_nombre] = _pipe
        predicciones_prueba[_nombre] = _pred
        probabilidades_prueba[_nombre] = _prob
        _filas_prueba.append(
            {
                "Modelo": _nombre,
                "ROC AUC": roc_auc_score(y_prueba, _prob),
                "Exactitud": accuracy_score(y_prueba, _pred),
                "Exactitud balanceada": balanced_accuracy_score(y_prueba, _pred),
                "F1": f1_score(y_prueba, _pred),
                "Precisión": precision_score(y_prueba, _pred),
                "Sensibilidad": recall_score(y_prueba, _pred),
            }
        )

    resultados_prueba = pd.DataFrame(_filas_prueba).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    print(f"Reporte de clasificación ({mejor_modelo_nombre}):\n")
    print(classification_report(y_prueba, predicciones_prueba[mejor_modelo_nombre], target_names=["no", "sí"]))
    resultados_prueba.round(3)
    return modelos_ajustados, probabilidades_prueba, resultados_prueba


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_prueba,
    mejor_modelo_nombre,
    modelos_ajustados,
    plt,
    probabilidades_prueba,
    roc_auc_score,
    roc_curve,
    y_prueba,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for _nombre, _prob in probabilidades_prueba.items():
        _fpr, _tpr, _ = roc_curve(y_prueba, _prob)
        _auc = roc_auc_score(y_prueba, _prob)
        _axes[0].plot(_fpr, _tpr, label=f"{_nombre} (AUC = {_auc:.3f})", linewidth=2)
    _axes[0].plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.500)")
    _axes[0].set_title("Curvas ROC en el conjunto de prueba")
    _axes[0].set_xlabel("Tasa de Falsos Positivos")
    _axes[0].set_ylabel("Sensibilidad (Verdaderos Positivos)")
    _axes[0].legend(loc="lower right", fontsize=8.5)

    ConfusionMatrixDisplay.from_estimator(
        modelos_ajustados[mejor_modelo_nombre],
        X_prueba,
        y_prueba,
        display_labels=["No", "Sí"],
        cmap="Blues",
        colorbar=False,
        ax=_axes[1],
    )
    _axes[1].grid(False)
    _axes[1].set_title(f"Matriz de confusión - {mejor_modelo_nombre}")
    _axes[1].set_xlabel("Predicción")
    _axes[1].set_ylabel("Valor real")
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(
    columnas_categoricas,
    columnas_numericas,
    mejor_modelo_nombre,
    modelos_ajustados,
    np,
    pd,
    plt,
    probabilidades_prueba,
    y_prueba,
):
    # Importancia de variables del mejor modelo (Random Forest)
    _mejor_pipe = modelos_ajustados[mejor_modelo_nombre]
    _prep_fit = _mejor_pipe.named_steps["prep"]
    _clf_fit = _mejor_pipe.named_steps["clf"]

    _nombres_num = _prep_fit.named_transformers_["numericas"].named_steps["seleccion_pearson"].get_feature_names_out(columnas_numericas)
    _onehot_names = _prep_fit.named_transformers_["categoricas"].named_steps["preparacion"].named_steps["codificacion"].get_feature_names_out(columnas_categoricas)
    _nombres_cat = _prep_fit.named_transformers_["categoricas"].named_steps["seleccion_chi2"].get_feature_names_out(_onehot_names)
    _nombres_todos = list(_nombres_num) + list(_nombres_cat)

    importancia = (
        pd.DataFrame({"variable": _nombres_todos, "importancia": _clf_fit.feature_importances_})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    # Priorización de llamadas en prueba (Top 10%, 20%, 30%, 50%, 100%)
    _df_ord = pd.DataFrame(
        {"y": y_prueba.to_numpy(), "prob": probabilidades_prueba[mejor_modelo_nombre]}
    ).sort_values("prob", ascending=False).reset_index(drop=True)
    _total_si = int(_df_ord["y"].sum())
    _tasa_base = _total_si / len(_df_ord)

    _filas_prio = []
    for _pct in [10, 20, 30, 50, 100]:
        _corte = int(np.ceil(len(_df_ord) * _pct / 100))
        _sub = _df_ord.iloc[:_corte]
        _si = int(_sub["y"].sum())
        _tasa = _si / len(_sub)
        _filas_prio.append(
            {
                "Grupo": f"Top {_pct}%" if _pct < 100 else "Todos (100%)",
                "Llamadas": len(_sub),
                "Suscriptores": _si,
                "Tasa de éxito (%)": round(_tasa * 100, 1),
                "% del total capturado": round(_si / _total_si * 100, 1),
                "Mejora vs azar": round(_tasa / _tasa_base, 2),
            }
        )
    tabla_priorizacion = pd.DataFrame(_filas_prio)

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _top10 = importancia.head(10)
    _axes[0].barh(_top10["variable"][::-1], _top10["importancia"][::-1], color="#219ebc")
    _axes[0].set_title(f"Top 10 variables más importantes ({mejor_modelo_nombre})")
    _axes[0].set_xlabel("Importancia")

    _grupos = tabla_priorizacion.iloc[:-1]
    _axes[1].bar(_grupos["Grupo"], _grupos["Tasa de éxito (%)"], color="#8ecae6", label=mejor_modelo_nombre)
    _axes[1].axhline(_tasa_base * 100, color="red", linestyle="--", label=f"Sin modelo ({_tasa_base*100:.1f}%)")
    _axes[1].set_title("Tasa de suscripción al priorizar llamadas")
    _axes[1].set_ylabel("Clientes que contratan (%)")
    for _i, _v in enumerate(_grupos["Tasa de éxito (%)"]):
        _axes[1].text(_i, _v + 1.2, f"{_v:.1f}%", ha="center", fontsize=9)
    _axes[1].set_ylim(0, max(_grupos["Tasa de éxito (%)"]) * 1.2)
    _axes[1].legend(fontsize=8.5)
    _fig.tight_layout()
    plt.show()

    tabla_priorizacion
    return (tabla_priorizacion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Conclusiones
    """)
    return


@app.cell(hide_code=True)
def _(
    mejor_modelo_nombre,
    mo,
    ranking_chi2,
    ranking_pearson,
    resultados_prueba,
    resultados_validacion,
    tabla_priorizacion,
):
    variables_chi = ", ".join(ranking_chi2.head(3)["variable"])
    variables_pearson = ", ".join(ranking_pearson.head(3)["variable"])
    fila_cv = resultados_validacion.loc[resultados_validacion["Modelo"] == mejor_modelo_nombre].iloc[0]
    fila_test = resultados_prueba.loc[resultados_prueba["Modelo"] == mejor_modelo_nombre].iloc[0]
    p10 = tabla_priorizacion.iloc[0]
    p30 = tabla_priorizacion.iloc[2]

    mo.md(
        f"""
        ### Conclusión del proyecto

        - **Preprocesamiento:** Se limpiaron formatos de texto, se verificó que no había duplicados
          exactos, se excluyó `duration` para evitar fuga de información y se crearon dos variables nuevas
          (`presion_contacto` y `balance_ajustado_credito`). Los filtros destacaron **{variables_chi}**
          en categóricas y **{variables_pearson}** en numéricas.
        - **Mejor modelo:** Entre los tres modelos probados, **{mejor_modelo_nombre}** dio el mejor
          resultado tanto en validación cruzada (ROC AUC = **{fila_cv['ROC AUC']:.3f}**, F1 = **{fila_cv['F1']:.3f}**)
          como en prueba (ROC AUC = **{fila_test['ROC AUC']:.3f}**, Exactitud balanceada = **{fila_test['Exactitud balanceada']:.3f}**,
          F1 = **{fila_test['F1']:.3f}**), superando a la Regresión Logística y al Árbol de Decisión.
        - **Aplicación a campañas telefónicas:** Al ordenar a los clientes por la probabilidad estimada,
          llamar solo al **Top 10%** sube la tasa de éxito de 11.7% a **{p10['Tasa de éxito (%)']:.1f}%**
          (**{p10['Mejora vs azar']:.2f}x** mejor que al azar) y captura el **{p10['% del total capturado']:.1f}%**
          de los suscriptores. Si se contacta al **Top 30%**, se captura el **{p30['% del total capturado']:.1f}%**
          ahorrando el 70% de llamadas.
        """
    )
    return


if __name__ == "__main__":
    app.run()
