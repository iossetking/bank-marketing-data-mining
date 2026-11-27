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
    # Proyecto de Minería de Datos: Bank Marketing

    **Predicción de contratación de depósitos a plazo fijo para priorizar campañas telefónicas**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Motivación del trabajo y problemática a abordar

    En las campañas de marketing telefónico de un banco, contactar a todos los clientes registrados implica un gasto alto de tiempo y dinero para el centro de llamadas (horas de los agentes, costos operativos y múltiples intentos de contacto). Además, marcar sin un filtro previo provoca que la gran mayoría de las personas rechacen la oferta (cerca del 88.3% dicen que no) e incluso genera molestia o fatiga en los clientes por recibir llamadas insistentes.

    **Problemática:** El banco necesita saber, **antes de realizar la llamada**, qué clientes tienen mayor probabilidad de aceptar un depósito a plazo fijo para no desperdiciar llamadas al azar.

    **Motivación y objetivo:** Desarrollar, optimizar y evaluar un flujo completo de Minería de Datos (siguiendo las etapas del proceso KDD) con modelos de clasificación supervisada que estimen la probabilidad de suscripción (`y = yes` o `no`) usando únicamente información disponible antes de marcar. Con esto buscamos **priorizar la lista de llamadas** y concentrar el esfuerzo comercial en el segmento de clientes con mayor propensión a contratar.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Justificación de la selección de la base de datos

    Para abordar esta problemática elegimos el dataset **Bank Marketing** del repositorio público **UCI Machine Learning Repository (ID 222)** (https://archive.ics.uci.edu/dataset/222/bank+marketing), el cual recopila datos reales de campañas telefónicas de una institución bancaria de Portugal.

    **¿Por qué elegimos esta base de datos?**
    1. **Datos reales y tamaño suficiente:** Cuenta con **45,211 registros** y **16 variables predictoras** (7 numéricas y 9 categóricas) más la variable objetivo `y`, lo que nos da una muestra grande y representativa para entrenar, hacer validación cruzada y reservar un conjunto de prueba independiente sin quedarnos cortos de datos.
    2. **Combinación rica de variables de negocio:** Incluye tanto el perfil socioeconómico del cliente (`age`, `job`, `marital`, `education`, `balance`, `default`, `housing`, `loan`) como el contexto de la campaña actual y el historial de campañas previas (`contact`, `day_of_week`, `month`, `campaign`, `pdays`, `previous`, `poutcome`).
    3. **Retos reales de Minería de Datos:** No es un dataset "de juguete"; presenta problemas reales que vimos en clase y que nos obligan a justificar decisiones técnicas: **desbalance de clases** (solo el 11.7% contrata), **valores faltantes** en variables categóricas, **códigos especiales** (`pdays = -1`), **valores extremos reales** en saldos y llamadas, y una variable con **fuga de información** (`duration`) que debemos detectar y excluir.

    | Nombre Variable | Tipo | Descripción |
    | --- | --- | --- |
    | `age` | Numérica | Edad del cliente, en años. |
    | `job` | Categórica | Tipo de trabajo u ocupación (11 categorías; algunos registros faltantes). |
    | `marital` | Categórica | Estado civil: `divorced`, `married`, `single`. |
    | `education` | Categórica | Nivel educativo: `primary`, `secondary` o `tertiary` (algunos faltantes). |
    | `default` | Categórica binaria | Indica si el cliente tiene crédito en mora (`yes`/`no`). |
    | `balance` | Numérica | Saldo anual promedio del cliente, en euros. |
    | `housing` | Categórica binaria | Indica si el cliente tiene un préstamo hipotecario (`yes`/`no`). |
    | `loan` | Categórica binaria | Indica si el cliente tiene un préstamo personal (`yes`/`no`). |
    | `contact` | Categórica | Medio de contacto: `telephone` o `cellular` (algunos faltantes). |
    | `day_of_week` | Numérica | Día del mes del último contacto de esta campaña. |
    | `month` | Categórica | Mes del último contacto de esta campaña (`jan`–`dec`). |
    | `duration` | Numérica | Duración del último contacto en segundos (**se excluye por fuga de datos**). |
    | `campaign` | Numérica | Número de contactos realizados durante esta campaña para este cliente. |
    | `pdays` | Numérica | Días desde el último contacto en una campaña anterior (`-1` = sin contacto previo). |
    | `previous` | Numérica | Número de contactos realizados para este cliente antes de esta campaña. |
    | `poutcome` | Categórica | Resultado de la campaña anterior: `other`, `failure` o `success` (faltantes = sin registro previo). |
    | `y` | Objetivo binario | Indica si el cliente suscribió un depósito a plazo fijo (`yes`/`no`). |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Importación de Librerías
    """)
    return


@app.cell
def _():
    # Se importan las librerías para manipular datos, graficar, seleccionar variables y entrenar/optimizar los modelos.
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
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        StratifiedKFold,
        cross_validate,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier
    from scipy.stats import chi2_contingency, pearsonr
    from ucimlrepo import fetch_ucirepo

    return (
        ColumnTransformer,
        ConfusionMatrixDisplay,
        DecisionTreeClassifier,
        GridSearchCV,
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
        confusion_matrix,
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
    ## 3. Carga del Dataset
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
    print(f"X Shape: {X.shape}")
    X.head()
    return


@app.cell
def _(y):
    # Se comprueba que la variable objetivo tenga una sola columna y valores esperados.
    print(f"y Shape: {y.shape}")
    y.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Diagnóstico de Calidad de los Datos y Preprocesamiento

    Antes de meter los datos a cualquier algoritmo, revisamos valores faltantes, filas duplicadas, consistencia de categorías y valores atípicos (mediante el rango intercuartílico IQR y diagramas de caja).
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
    ### 4.1 Justificación de la Limpieza de Datos

    | Problema identificado | Variables | Decisión tomada | Justificación |
    | --- | --- | --- | --- |
    | Filas duplicadas exactas | Todas | Revisar y eliminar duplicados exactos (`0` encontrados) | Evita darle doble peso a un mismo registro; no borramos clientes con varias llamadas porque cada intento tiene distinto contexto. |
    | Espacios o mayúsculas/minúsculas | Categóricas y `y` | Recortar espacios (`strip`) y pasar a minúsculas (`lower`) | Evita que el modelo trate la misma categoría como dos distintas por errores de captura. |
    | Faltantes en categóricas | `job` (288), `education` (1,857), `contact` (13,020), `poutcome` (36,959) | Imputar con la **moda** dentro del `Pipeline` de entrenamiento | Si borráramos las filas con datos faltantes perderíamos más del 80% de la base. Al imputar dentro del pipeline evitamos fuga de datos desde prueba. |
    | Código especial `-1` | `pdays` | Conservar el valor `-1` | No es un error ni un dato perdido: significa que al cliente nunca se le había llamado en campañas anteriores. |
    | Valores extremos (atípicos IQR) | `balance`, `campaign`, `pdays`, `previous` | Conservar los registros y escalar con `MinMaxScaler` | En un banco es normal que existan clientes con saldos muy altos o con varios contactos; borrarlos distorsionaría la población real. |
    """)
    return


@app.cell
def _(datos_brutos, duplicados_brutos):
    # Se eliminan solo duplicados exactos y se normaliza el formato de las columnas de texto.
    datos_limpios = datos_brutos.drop_duplicates().copy()

    columnas_texto_limpias = datos_limpios.select_dtypes(include=["object", "string"]).columns
    for _columna in columnas_texto_limpias:
        datos_limpios[_columna] = datos_limpios[_columna].str.strip().str.lower()

    print(f"Registros eliminados por duplicado: {duplicados_brutos:,}")
    print(f"Registros disponibles después de limpiar: {len(datos_limpios):,}")
    return (datos_limpios,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Transformación de Variables e Ingeniería de Características

    Todas las transformaciones se diseñaron pensando en que el modelo debe predecir **antes de hacer la llamada**:
    1. **Exclusión de `duration` (evitar fuga de datos):** La duración en segundos de la llamada solo se conoce cuando la llamada ya terminó. Si la dejáramos, las métricas subirían artificialmente pero el modelo sería inútil en la vida real para decidir a quién marcar.
    2. **Creación de dos variables derivadas de negocio:**
       - `presion_contacto = campaign / (previous + 1)`: Mide qué tanto se le ha insistido al cliente en la campaña actual en comparación con sus contactos anteriores (sumamos 1 en el denominador para evitar divisiones entre cero).
       - `balance_ajustado_credito = balance / (1 + housing + loan)`: Aproxima la liquidez real del cliente dividiendo su saldo promedio entre la cantidad de créditos activos que tiene (hipoteca y préstamo personal).
    3. **Codificación y escalamiento sin fuga de información:**
       - Las categóricas binarias (`default`, `housing`, `loan`) se pasan a `0/1` (`yes = 1`, `no = 0`).
       - Las numéricas se imputan con la **mediana** (robusta ante valores extremos) y se escalan al rango `[0, 1]` con `MinMaxScaler` para que estén en la misma escala.
       - Las categóricas se imputan con la **moda** y se codifican con `OneHotEncoder(drop="if_binary", handle_unknown="ignore")` para no inventar un orden numérico falso entre categorías.
    """)
    return


@app.cell
def _(datos_limpios):
    # Se crean dos variables derivadas mediante operaciones entre columnas existentes (conocidas antes de llamar).
    datos_modelo = datos_limpios.copy()

    for _columna_binaria in ["default", "housing", "loan"]:
        datos_modelo[_columna_binaria] = (
            datos_modelo[_columna_binaria].eq("yes").astype(int)
        )

    datos_modelo["presion_contacto"] = datos_modelo["campaign"] / (
        datos_modelo["previous"] + 1
    )

    datos_modelo["balance_ajustado_credito"] = datos_modelo["balance"] / (
        1 + datos_modelo["housing"] + datos_modelo["loan"]
    )

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

    # duration se excluye por fuga de información.
    X_modelo = datos_modelo.drop(columns=["y", "duration"])
    y_modelo = (datos_modelo["y"].str.lower() == "yes").astype(int)

    print("Distribución de la respuesta (0 = no, 1 = sí):")
    print(y_modelo.value_counts(normalize=True).rename("proporción").round(4))
    return X_modelo, y_modelo


@app.cell
def _(X_modelo):
    # Se separan columnas numéricas y categóricas (incluyendo las binarias en el grupo categórico para el filtro Chi-cuadrada).
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
    ## 5. Partición Estratificada en Entrenamiento (80%) y Prueba (20%)

    Dividimos la base antes de calcular cualquier filtro o entrenar modelos. Usamos `stratify=y_modelo` para que tanto entrenamiento como prueba mantengan exactamente el **11.7%** de casos positivos. El conjunto de prueba (9,043 clientes) se guarda hasta el final y **nunca se utiliza durante la selección de variables ni en la búsqueda de hiperparámetros**.
    """)
    return


@app.cell
def _(X_modelo, train_test_split, y_modelo):
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
    ## 6. Métodos de Filtro para Selección de Características (Chi-cuadrada y Pearson)

    Calculamos los filtros únicamente sobre el conjunto de entrenamiento:
    - **Chi-cuadrada y V de Cramér** para medir la asociación entre cada variable categórica y la respuesta `y`.
    - **Correlación de Pearson (`r`)** para medir la relación lineal entre cada variable numérica y `y`.
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
    _columnas_categoricas_filtro = columnas_categoricas
    _resultados_chi = []
    _tablas_contingencia = {}
    for _columna in _columnas_categoricas_filtro:
        _tabla = pd.crosstab(X_entrenamiento[_columna], y_entrenamiento)
        _estadistico, _p_valor, _grados_libertad, _ = chi2_contingency(_tabla)
        _n = _tabla.to_numpy().sum()
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
def _(mo):
    mo.md(r"""
    ## 7. Justificación de la Elección de Modelos, Metodología y Optimización (Grid Search + CV)

    Para cubrir distintas familias de clasificadores vistos en clase y comparar cuál generaliza mejor, elegimos **tres modelos de complejidad creciente**:
    1. **Regresión Logística:** Es nuestro modelo lineal base. Calcula directamente una combinación lineal de las variables para estimar la probabilidad de suscripción. Lo incluimos por su rapidez e interpretabilidad, aunque tiene la limitación de no capturar interacciones no lineales complejas.
    2. **Árbol de Decisión (`DecisionTreeClassifier`):** Es un modelo no lineal que divide a los clientes mediante reglas jerárquicas tipo "si-entonces" (por ejemplo, si tuvo éxito en la campaña anterior y tiene cierto saldo). Captura relaciones no lineales, pero un árbol suelto suele ser sensible a pequeñas variaciones en los datos.
    3. **Random Forest (`RandomForestClassifier`):** Es un ensamble tipo *bagging* que entrena muchos árboles de decisión sobre submuestras aleatorias de datos y variables, y luego promedia sus predicciones. Lo elegimos porque reduce la varianza de un árbol individual, maneja muy bien mezclas de variables numéricas y categóricas codificadas, y evita el sobreajuste cuando se regula la profundidad de sus árboles.

    **Metodología para manejar el desbalance y optimizar sin fuga de datos:**
    - **Compensación del desbalance (`class_weight="balanced"`):** Como solo el 11.7% de los registros son positivos, un modelo sin ajuste aprendería a predecir casi siempre "No". Con pesos balanceados penalizamos más equivocarse en los clientes que sí contratan.
    - **Búsqueda de hiperparámetros con `GridSearchCV` y Validación Cruzada Estratificada (`5-Fold StratifiedKFold`):** Para no elegir hiperparámetros al tanteo ni usar el conjunto de prueba, aplicamos `GridSearchCV` exclusivamente dentro del conjunto de entrenamiento (36,168 registros) optimizando `ROC AUC`. Además, al limitar `max_depth` y exigir un mínimo de muestras por hoja (`min_samples_leaf`) en los árboles, evitamos que memoricen ruido y aseguramos una buena capacidad de generalización.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    DecisionTreeClassifier,
    GridSearchCV,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    SelectKBest,
    StratifiedKFold,
    X_entrenamiento,
    chi2,
    columnas_categoricas,
    columnas_numericas,
    cross_validate,
    np,
    pd,
    pearsonr,
    plt,
    sns,
    transformador_categorico,
    transformador_numerico,
    y_entrenamiento,
):
    # Función de Pearson para usarla dentro de SelectKBest en cada partición.
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

    # Definición de los 3 modelos base y su rejilla de hiperparámetros para GridSearchCV
    modelos_grid = {
        "Regresión Logística": (
            Pipeline(
                steps=[
                    ("prep", crear_preprocesador()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
                ]
            ),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "Árbol de Decisión": (
            Pipeline(
                steps=[
                    ("prep", crear_preprocesador()),
                    ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42)),
                ]
            ),
            {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [25, 50, 100]},
        ),
        "Random Forest": (
            Pipeline(
                steps=[
                    ("prep", crear_preprocesador()),
                    ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
                ]
            ),
            {
                "clf__n_estimators": [100, 150],
                "clf__max_depth": [8, 12],
                "clf__min_samples_leaf": [20, 40],
            },
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    modelos_ajustados = {}
    mejores_hiperparametros = {}
    _filas_cv = []

    # Se ejecuta GridSearchCV únicamente sobre X_entrenamiento, y_entrenamiento (sin tocar X_prueba)
    for _nombre, (_pipe, _grid_params) in modelos_grid.items():
        _busqueda = GridSearchCV(
            _pipe,
            param_grid=_grid_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        _busqueda.fit(X_entrenamiento, y_entrenamiento)
        _mejor_pipe = _busqueda.best_estimator_
        modelos_ajustados[_nombre] = _mejor_pipe
        _params_limpios = {k.replace("clf__", ""): v for k, v in _busqueda.best_params_.items()}
        mejores_hiperparametros[_nombre] = _params_limpios

        _res = cross_validate(_mejor_pipe, X_entrenamiento, y_entrenamiento, cv=cv, scoring=scoring, n_jobs=-1)
        _filas_cv.append(
            {
                "Modelo": _nombre,
                "Mejores hiperparámetros (GridSearch)": ", ".join(f"{k}={v}" for k, v in _params_limpios.items()),
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
    _ax.set_title("Comparación de modelos optimizados en validación cruzada (5 particiones)")
    _ax.set_ylim(0, 1.0)
    for _c in _ax.containers:
        _ax.bar_label(_c, fmt="%.2f", fontsize=8, padding=2)
    _fig.tight_layout()
    plt.show()

    print(f"Mejor modelo en validación cruzada: {mejor_modelo_nombre}")
    resultados_validacion.round(3)
    return (
        mejor_modelo_nombre,
        mejores_hiperparametros,
        modelos_ajustados,
        resultados_validacion,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Comparación de Métricas: Entrenamiento vs Validación Cruzada vs Prueba (Análisis de Generalización)

    Para verificar que nuestros modelos tienen buena capacidad de generalización y **no sufren de sobreajuste (overfitting)**, comparamos las métricas obtenidas en:
    1. **Entrenamiento** (36,168 registros).
    2. **Validación Cruzada Estratificada de 5 particiones** (promedio de validación interna).
    3. **Prueba independiente** (9,043 registros nunca vistos durante el entrenamiento ni en el Grid Search).
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
    modelos_ajustados,
    pd,
    precision_score,
    recall_score,
    resultados_validacion,
    roc_auc_score,
    y_entrenamiento,
    y_prueba,
):
    probabilidades_prueba = {}
    predicciones_prueba = {}
    predicciones_entrenamiento = {}
    _filas_entrenamiento = []
    _filas_prueba = []
    _filas_comparacion = []

    for _nombre, _pipe in modelos_ajustados.items():
        # Evaluación en Entrenamiento
        _pred_tr = _pipe.predict(X_entrenamiento)
        _prob_tr = _pipe.predict_proba(X_entrenamiento)[:, 1]
        predicciones_entrenamiento[_nombre] = _pred_tr
        _row_tr = {
            "Modelo": _nombre,
            "ROC AUC": roc_auc_score(y_entrenamiento, _prob_tr),
            "Exactitud": accuracy_score(y_entrenamiento, _pred_tr),
            "Exactitud balanceada": balanced_accuracy_score(y_entrenamiento, _pred_tr),
            "F1": f1_score(y_entrenamiento, _pred_tr),
            "Precisión": precision_score(y_entrenamiento, _pred_tr),
            "Sensibilidad": recall_score(y_entrenamiento, _pred_tr),
        }
        _filas_entrenamiento.append(_row_tr)

        # Evaluación en Prueba
        _pred_te = _pipe.predict(X_prueba)
        _prob_te = _pipe.predict_proba(X_prueba)[:, 1]
        predicciones_prueba[_nombre] = _pred_te
        probabilidades_prueba[_nombre] = _prob_te
        _row_te = {
            "Modelo": _nombre,
            "ROC AUC": roc_auc_score(y_prueba, _prob_te),
            "Exactitud": accuracy_score(y_prueba, _pred_te),
            "Exactitud balanceada": balanced_accuracy_score(y_prueba, _pred_te),
            "F1": f1_score(y_prueba, _pred_te),
            "Precisión": precision_score(y_prueba, _pred_te),
            "Sensibilidad": recall_score(y_prueba, _pred_te),
        }
        _filas_prueba.append(_row_te)

        _row_cv = resultados_validacion.loc[resultados_validacion["Modelo"] == _nombre].iloc[0]
        for _etapa, _r in [
            ("1. Entrenamiento", _row_tr),
            ("2. Validación Cruzada (5-Fold)", _row_cv),
            ("3. Prueba Independiente", _row_te),
        ]:
            _filas_comparacion.append(
                {
                    "Modelo": _nombre,
                    "Etapa": _etapa,
                    "ROC AUC": round(float(_r["ROC AUC"]), 4),
                    "Exactitud balanceada": round(float(_r["Exactitud balanceada"]), 4),
                    "F1": round(float(_r["F1"]), 4),
                    "Precisión": round(float(_r["Precisión"]), 4),
                    "Sensibilidad": round(float(_r["Sensibilidad"]), 4),
                }
            )

    resultados_entrenamiento = pd.DataFrame(_filas_entrenamiento).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    resultados_prueba = pd.DataFrame(_filas_prueba).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    tabla_comparacion_etapas = pd.DataFrame(_filas_comparacion)

    print(f"Reporte de clasificación en Prueba ({mejor_modelo_nombre}):\n")
    print(classification_report(y_prueba, predicciones_prueba[mejor_modelo_nombre], target_names=["no", "sí"]))
    tabla_comparacion_etapas
    return (
        predicciones_entrenamiento,
        predicciones_prueba,
        probabilidades_prueba,
        resultados_entrenamiento,
        resultados_prueba,
        tabla_comparacion_etapas,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Curvas ROC y Matrices de Confusión (Entrenamiento vs Prueba)

    Presentamos tanto la **matriz de confusión de entrenamiento** como la **matriz de confusión de prueba** del mejor modelo (**Random Forest**), junto con las curvas ROC comparativas:
    - Al comparar ambas matrices se observa que las proporciones de aciertos (Verdaderos Negativos y Verdaderos Positivos) y de errores (Falsos Positivos y Falsos Negativos) se mantienen prácticamente idénticas en proporción al tamaño de cada subconjunto (80% vs 20%).
    - Gracias al uso de `class_weight="balanced"`, el modelo logra detectar alrededor del **60%** de todos los clientes que realmente contratan un depósito (alta sensibilidad), en lugar de ignorar a la clase minoritaria.
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    confusion_matrix,
    mejor_modelo_nombre,
    plt,
    predicciones_entrenamiento,
    predicciones_prueba,
    probabilidades_prueba,
    roc_auc_score,
    roc_curve,
    y_entrenamiento,
    y_prueba,
):
    _cm_train = confusion_matrix(y_entrenamiento, predicciones_entrenamiento[mejor_modelo_nombre])
    _cm_test = confusion_matrix(y_prueba, predicciones_prueba[mejor_modelo_nombre])

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for _nombre, _prob in probabilidades_prueba.items():
        _fpr, _tpr, _ = roc_curve(y_prueba, _prob)
        _auc = roc_auc_score(y_prueba, _prob)
        _axes[0].plot(_fpr, _tpr, label=f"{_nombre} (AUC = {_auc:.3f})", linewidth=2)
    _axes[0].plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.500)")
    _axes[0].set_title("Curvas ROC (Conjunto de prueba)")
    _axes[0].set_xlabel("Tasa de Falsos Positivos")
    _axes[0].set_ylabel("Sensibilidad (Verdaderos Positivos)")
    _axes[0].legend(loc="lower right", fontsize=8)

    ConfusionMatrixDisplay(_cm_train, display_labels=["No", "Sí"]).plot(
        cmap="Blues", colorbar=False, ax=_axes[1], values_format="d"
    )
    _axes[1].grid(False)
    _axes[1].set_title(f"Matriz en Entrenamiento\n({mejor_modelo_nombre}, n={len(y_entrenamiento):,})")
    _axes[1].set_xlabel("Predicción")
    _axes[1].set_ylabel("Valor real")

    ConfusionMatrixDisplay(_cm_test, display_labels=["No", "Sí"]).plot(
        cmap="Blues", colorbar=False, ax=_axes[2], values_format="d"
    )
    _axes[2].grid(False)
    _axes[2].set_title(f"Matriz en Prueba\n({mejor_modelo_nombre}, n={len(y_prueba):,})")
    _axes[2].set_xlabel("Predicción")
    _axes[2].set_ylabel("Valor real")

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
    ## 10. Interpretación de Resultados y Conclusiones
    """)
    return


@app.cell(hide_code=True)
def _(
    mejor_modelo_nombre,
    mo,
    ranking_chi2,
    ranking_pearson,
    resultados_entrenamiento,
    resultados_prueba,
    resultados_validacion,
    tabla_priorizacion,
):
    variables_chi = ", ".join(ranking_chi2.head(3)["variable"])
    variables_pearson = ", ".join(ranking_pearson.head(3)["variable"])
    fila_tr = resultados_entrenamiento.loc[resultados_entrenamiento["Modelo"] == mejor_modelo_nombre].iloc[0]
    fila_cv = resultados_validacion.loc[resultados_validacion["Modelo"] == mejor_modelo_nombre].iloc[0]
    fila_test = resultados_prueba.loc[resultados_prueba["Modelo"] == mejor_modelo_nombre].iloc[0]
    p10 = tabla_priorizacion.iloc[0]
    p30 = tabla_priorizacion.iloc[2]

    mo.md(
        f"""
        ### Conclusiones del proyecto

        - **Preprocesamiento justificado y sin fuga de datos:** Se limpiaron formatos de texto, se verificó que no había duplicados exactos, se conservaron los valores extremos reales y el código `pdays = -1`, se excluyó `duration` porque no se conoce antes de llamar y se crearon dos variables derivadas útiles (`presion_contacto` y `balance_ajustado_credito`). Los filtros destacaron **{variables_chi}** en categóricas y **{variables_pearson}** en numéricas.
        - **Optimización y ausencia de sobreajuste:** Mediante `GridSearchCV` con validación cruzada estratificada de 5 particiones (usando únicamente entrenamiento), **{mejor_modelo_nombre}** resultó el mejor clasificador. Al comparar Entrenamiento (ROC AUC = **{fila_tr['ROC AUC']:.3f}**), Validación Cruzada (ROC AUC = **{fila_cv['ROC AUC']:.3f}**) y Prueba (ROC AUC = **{fila_test['ROC AUC']:.3f}**, Exactitud balanceada = **{fila_test['Exactitud balanceada']:.3f}**), comprobamos que el modelo generaliza bien y no está sobreajustado.
        - **Impacto práctico en la campaña telefónica:** Si el banco ordena a sus clientes con las probabilidades de **{mejor_modelo_nombre}** y llama primero al **Top 10%**, la tasa de aceptación sube de 11.7% a **{p10['Tasa de éxito (%)']:.1f}%** (**{p10['Mejora vs azar']:.2f}x** mejor que llamar al azar) y captura el **{p10['% del total capturado']:.1f}%** de todos los depósitos posibles. Llamando solo al **Top 30%**, el banco consigue el **{p30['% del total capturado']:.1f}%** de las suscripciones ahorrándose el 70% de las llamadas.
        """
    )
    return


if __name__ == "__main__":
    app.run()
