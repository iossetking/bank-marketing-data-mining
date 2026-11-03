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
    | `education` | Nivel educativo: basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree o unknown. |
    | `default` | Indica si el cliente tiene crédito en mora (yes/no). |
    | `balance` | Saldo anual promedio del cliente, en euros. |
    | `housing` | Indica si el cliente tiene un préstamo hipotecario (yes/no). |
    | `loan` | Indica si el cliente tiene un préstamo personal (yes/no). |
    | `contact` | Tipo de medio de contacto: unknown, telephone o cellular. |
    | `day` | Día del mes del último contacto de esta campaña. |
    | `month` | Mes del último contacto de esta campaña (jan–dec). |
    | `duration` | Duración del último contacto, en segundos. |
    | `campaign` | Número de contactos realizados durante esta campaña para este cliente, incluido el último contacto. |
    | `pdays` | Días desde el último contacto del cliente en una campaña anterior; `-1` indica que no había sido contactado anteriormente. |
    | `previous` | Número de contactos realizados para este cliente antes de esta campaña. |
    | `poutcome` | Resultado de la campaña de marketing anterior: unknown, other, failure o success. |
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
    # Se importan las librerías para manipular datos, graficar y entrenar el modelo.
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
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
    )
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from ucimlrepo import fetch_ucirepo

    return (
        ColumnTransformer,
        ConfusionMatrixDisplay,
        LogisticRegression,
        MinMaxScaler,
        OneHotEncoder,
        Pipeline,
        SelectKBest,
        SimpleImputer,
        StratifiedKFold,
        accuracy_score,
        balanced_accuracy_score,
        chi2,
        classification_report,
        cross_validate,
        f1_score,
        fetch_ucirepo,
        mutual_info_classif,
        pd,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Carga de Dataset
    """)
    return


@app.cell
def _(fetch_ucirepo):
    # Se descarga la versión original del repositorio UCI para conservar
    # los nombres y valores reportados por la fuente.
    bank_marketing = fetch_ucirepo(id=222) 

    # Se separan las variables explicativas y la respuesta por claridad.
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
    resumen_calidad
    return (duplicados_brutos,)


@app.cell
def _(datos_brutos, pd):
    # Se inspeccionan categorías especiales; en este conjunto unknown no se asume faltante.
    columnas_texto = datos_brutos.select_dtypes(include=["object", "string"]).columns
    conteo_unknown = pd.DataFrame(
        {
            "unknown": [
                (datos_brutos[col].astype(str).str.lower() == "unknown").sum()
                for col in columnas_texto
            ]
        },
        index=columnas_texto,
    )
    print(
        "Registros con pdays = -1 (sin contacto previo): "
        f"{(datos_brutos['pdays'] == -1).sum():,}"
    )
    conteo_unknown
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Limpieza de Datos

    Aplicar estrategias necesarias para tratar los problemas identificados y justificando las decisiones tomadas.
    """)
    return


@app.cell
def _(datos_brutos, duplicados_brutos):
    # Se eliminan solo duplicados exactos. No se eliminan clientes repetidos porque cada llamada
    # puede corresponder a un intento distinto y aporta información de campaña.
    datos_limpios = datos_brutos.drop_duplicates().copy()

    # Se normalizan espacios en variables de texto para evitar categorías duplicadas por formato.
    columnas_texto_limpias = datos_limpios.select_dtypes(include=["object", "string"]).columns
    for columna in columnas_texto_limpias:
        datos_limpios[columna] = datos_limpios[columna].str.strip()

    # unknown se conserva como categoría: expresa que la información no fue registrada.


    # La imputación posterior queda dentro para no filtrar información del conjunto de prueba.
    print(f"Registros eliminados por duplicado: {duplicados_brutos:,}")
    print(f"Registros disponibles después de limpiar: {len(datos_limpios):,}")
    return (datos_limpios,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Transformación de Variables

    Incluir codificación de variables categóricas y el escalamiento de variables numéricas, según corresponda.
    """)
    return


@app.cell
def _(datos_limpios):
    # Se crean exactamente dos variables derivadas mediante operaciones entre columnas existentes.
    # Ambas están disponibles antes de hacer una llamada y no usan la variable objetivo.
    datos_modelo = datos_limpios.copy()

    # Relaciona los contactos de la campaña actual con los de campañas previas.
    # Un valor alto señala insistencia reciente; un valor bajo indica más historial que presión actual.
    datos_modelo["presion_contacto"] = datos_modelo["campaign"] / (
        datos_modelo["previous"] + 1
    )

    # Ajusta el saldo por la cantidad de créditos activos reportados (vivienda y personal).
    # Sirve como aproximación de la capacidad financiera disponible para aceptar un depósito.
    datos_modelo["balance_ajustado_credito"] = datos_modelo["balance"] / (
        1
        + datos_modelo["housing"].eq("yes").astype(int)
        + datos_modelo["loan"].eq("yes").astype(int)
    )

    # duration se excluye: se conoce al finalizar la llamada y no estaría disponible
    # cuando el banco tenga que decidir a quién contactar. Incluirla inflaría el desempeño.
    X_modelo = datos_modelo.drop(columns=["y", "duration"])
    y_modelo = (datos_modelo["y"].str.lower() == "yes").astype(int)

    print("Distribución de la respuesta (0 = no, 1 = sí):")
    print(y_modelo.value_counts(normalize=True).rename("proporción").round(4))
    return X_modelo, y_modelo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Ingeniería de características

    Mediante la creación de al menos dos variables derivadas, explicando su utilidad.
    """)
    return


@app.cell
def _(X_modelo):
    # Se identifican columnas por tipo para aplicar una preparación adecuada a cada grupo.
    columnas_numericas = X_modelo.select_dtypes(include="number").columns.tolist()
    columnas_categoricas = X_modelo.select_dtypes(exclude="number").columns.tolist()
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
    transformador_numerico = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("escala", MinMaxScaler()),
        ]
    )
    transformador_categorico = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="most_frequent")),
            ("codificacion", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", transformador_numerico, columnas_numericas),
            ("categoricas", transformador_categorico, columnas_categoricas),
        ],
        remainder="drop",
    )
    return (preprocesador,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Aplicación de dos métodos de filtro para la selección de características

    Comparar e interpretar los resultados obtenidos.
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
    ## 7. Partición del conjunto de datos en entrenamiento y prueba

    Justificar la estrategia utilizada.
    """)
    return


@app.cell
def _(
    X_entrenamiento,
    chi2,
    mutual_info_classif,
    pd,
    preprocesador,
    y_entrenamiento,
):
    # Ambos filtros se calculan solo con entrenamiento. Así el conjunto de prueba se mantiene intacto.
    matriz_entrenamiento = preprocesador.fit_transform(X_entrenamiento)
    nombres_transformados = preprocesador.get_feature_names_out()

    # Filtro 1: chi-cuadrada mide asociación entre cada predictor no negativo y la clase.
    estadistico_chi2, p_valores_chi2 = chi2(matriz_entrenamiento, y_entrenamiento)
    ranking_chi2 = (
        pd.DataFrame(
            {
                "caracteristica": nombres_transformados,
                "estadistico_chi2": estadistico_chi2,
                "p_valor": p_valores_chi2,
            }
        )
        .sort_values("estadistico_chi2", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    # Filtro 2: información mutua detecta asociaciones lineales y no lineales.
    puntajes_mi = mutual_info_classif(
        matriz_entrenamiento, y_entrenamiento, random_state=42
    )
    ranking_mi = (
        pd.DataFrame(
            {"caracteristica": nombres_transformados, "informacion_mutua": puntajes_mi}
        )
        .sort_values("informacion_mutua", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    print("Las tablas muestran las 15 variables transformadas con mayor relevancia según cada filtro.")
    return ranking_chi2, ranking_mi


@app.cell
def _(ranking_chi2):
    # Se muestra el primer filtro para comparar cuáles indicadores se asocian más con la respuesta.
    ranking_chi2
    return


@app.cell
def _(ranking_mi):
    # Se muestra el segundo filtro; coincidencias con chi-cuadrada dan mayor confianza.
    ranking_mi
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Implementación de validación cruzada de 5 particiones

    Considerar la naturaleza del problema.
    """)
    return


@app.cell
def _(LogisticRegression, Pipeline, SelectKBest, chi2, preprocesador):
    # El pipeline encapsula preparación, selección y clasificación. De este modo, cada
    # partición de validación cruza ajusta sus propios pasos sin conocer sus datos de validación.
    modelo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("seleccion", SelectKBest(score_func=chi2, k=25)),
            (
                "clasificador",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    return (modelo,)


@app.cell
def _(
    StratifiedKFold,
    X_entrenamiento,
    cross_validate,
    modelo,
    pd,
    y_entrenamiento,
):
    # Se usa validación cruzada estratificada de cinco particiones porque la clase positiva
    # es minoritaria. Se reportan métricas que no dependen solo de la clase mayoritaria.
    cv_estratificada = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    puntajes_cv = cross_validate(
        modelo,
        X_entrenamiento,
        y_entrenamiento,
        cv=cv_estratificada,
        scoring={
            "roc_auc": "roc_auc",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
        },
        n_jobs=-1,
        return_train_score=False,
    )
    resultados_validacion = pd.DataFrame(
        {
            "métrica": ["ROC AUC", "Exactitud balanceada", "F1", "Precisión", "Sensibilidad"],
            "promedio": [
                puntajes_cv["test_roc_auc"].mean(),
                puntajes_cv["test_balanced_accuracy"].mean(),
                puntajes_cv["test_f1"].mean(),
                puntajes_cv["test_precision"].mean(),
                puntajes_cv["test_recall"].mean(),
            ],
            "desviación estándar": [
                puntajes_cv["test_roc_auc"].std(),
                puntajes_cv["test_balanced_accuracy"].std(),
                puntajes_cv["test_f1"].std(),
                puntajes_cv["test_precision"].std(),
                puntajes_cv["test_recall"].std(),
            ],
        }
    ).round(3)
    resultados_validacion
    return (resultados_validacion,)


@app.cell
def _(
    X_entrenamiento,
    X_prueba,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    modelo,
    pd,
    precision_score,
    recall_score,
    roc_auc_score,
    y_entrenamiento,
    y_prueba,
):
    # Se ajusta el modelo final solo con entrenamiento y se evalúa una única vez en prueba.
    modelo.fit(X_entrenamiento, y_entrenamiento)
    prediccion_prueba = modelo.predict(X_prueba)
    probabilidad_prueba = modelo.predict_proba(X_prueba)[:, 1]
    metricas_prueba = {
        "ROC AUC": roc_auc_score(y_prueba, probabilidad_prueba),
        "Exactitud": accuracy_score(y_prueba, prediccion_prueba),
        "Exactitud balanceada": balanced_accuracy_score(y_prueba, prediccion_prueba),
        "F1": f1_score(y_prueba, prediccion_prueba),
        "Precisión": precision_score(y_prueba, prediccion_prueba),
        "Sensibilidad": recall_score(y_prueba, prediccion_prueba),
    }
    print(pd.Series(metricas_prueba).round(3))
    print("\nReporte de clasificación:\n")
    print(classification_report(y_prueba, prediccion_prueba, target_names=["no", "sí"]))
    return (metricas_prueba,)


@app.cell
def _(ConfusionMatrixDisplay, X_prueba, modelo, plt, y_prueba):
    # La matriz permite observar los aciertos y errores para ambas clases en prueba.
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        modelo,
        X_prueba,
        y_prueba,
        display_labels=["No", "Sí"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Matriz de confusión en el conjunto de prueba")
    fig.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Conclusiones

    Resultados obtenidos durante la preparación de datos y las características seleccionadas.
    """)
    return


@app.cell(hide_code=True)
def _(metricas_prueba, mo, ranking_chi2, ranking_mi, resultados_validacion):
    # Se construye una conclusión a partir de los valores calculados en la ejecución.
    coincidencias = set(ranking_chi2["caracteristica"]).intersection(
        set(ranking_mi["caracteristica"])
    )
    variables_coincidentes = ", ".join(sorted(coincidencias)[:5]) or "sin coincidencias directas"
    mo.md(
        f"""
        ### Conclusión

        Se preparó el conjunto eliminando duplicados exactos, conservando `unknown` como una
        categoría informativa y creando dos variables derivadas: presión de contacto y balance
        ajustado por crédito. La variable `duration` se excluyó deliberadamente: usarla daría
        un resultado poco realista porque solo se conoce cuando la llamada ya terminó.

        Los filtros de chi-cuadrada e información mutua coincidieron, entre otras, en: **{variables_coincidentes}**.
        Esta coincidencia sugiere que dichas señales son candidatas estables, aunque la selección
        definitiva se repite dentro de cada partición del modelo para evitar fuga de información.

        En validación cruzada, el ROC AUC promedio fue de **{resultados_validacion.loc[0, 'promedio']:.3f}**
        y la exactitud balanceada fue de **{resultados_validacion.loc[1, 'promedio']:.3f}**. En el conjunto de
        prueba independiente, el ROC AUC fue **{metricas_prueba['ROC AUC']:.3f}**, la precisión para la clase
        positiva fue **{metricas_prueba['Precisión']:.3f}** y la sensibilidad fue
        **{metricas_prueba['Sensibilidad']:.3f}**. Por ello, el modelo puede servir para ordenar clientes
        por probabilidad de suscripción y priorizar contactos, pero el umbral de decisión debe ajustarse
        según el costo de llamar a un cliente y el valor esperado de una suscripción.
        """
    )
    return


if __name__ == "__main__":
    app.run()
