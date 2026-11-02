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

    - Link: UCI Bank Marketing (https://archive.ics.uci.edu/dataset/222/bank+marketing)

    - Numero de registros: 45,211

    - Numero de caracteristicas: 16

    - Variable ojetivo: `y` (Representa si un cliente se suscribio/contrato o no a un deposito a plazo fijo).
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
