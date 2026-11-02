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
    from ucimlrepo import fetch_ucirepo 

    import pandas as pd

    return (fetch_ucirepo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Carga de Dataset
    """)
    return


@app.cell
def _(fetch_ucirepo):
  
    # fetch dataset 
    bank_marketing = fetch_ucirepo(id=222) 
  
    # data (as pandas dataframes) 
    X = bank_marketing.data.features 
    y = bank_marketing.data.targets 
  
    # metadata 
    print(bank_marketing.metadata) 
  
    # variable information 
    print(bank_marketing.variables) 
    return X, y


@app.cell
def _(X):
    print(f'X Shape: {X.shape}')
    X.head()
    return


@app.cell
def _(y):
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Limpieza de Datos

    Aplicar estrategias necesarias para tratar los problemas identificados y justificando las decisiones tomadas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Transformación de Variables

    Incluir codificación de variables categóricas y el escalamiento de variables numéricas, según corresponda.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Ingeniería de características

    Mediante la creación de al menos dos variables derivadas, explicando su utilidad.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Aplicación de dos métodos de filtro para la selección de características

    Comparar e interpretar los resultados obtenidos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Partición del conjunto de datos en entrenamiento y prueba

    Justificar la estrategia utilizada.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Implementación de validación cruzada de 5 particiones

    Considerar la naturaleza del problema.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Conclusiones

    Resultados obtenidos durante la preparación de datos y las características seleccionadas.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
