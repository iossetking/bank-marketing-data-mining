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
    # Cafe Sales Data Mining
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cafe Sales Dataset

    Dataset de transacciones de cafes hechas en una cafeteria. Contiene informacion sobre cada pedido/transaccion como el tipo de cafe, el tipo de pago, y la fecha, etc.

    - link: https://huggingface.co/datasets/adityasuyal/cafe-sales-cleaned/blob/main/dirty_cafe_sales.csv

    - Numero de registros: 10,000

    - Numero de caracteristicas: 8

    - Variable ojetivo: ``
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
    import pandas as pd

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Carga de Dataset
    """)
    return


@app.cell
def _(pd):
    path = ('./datasets/raw/dirty_cafe_sales.csv')
    df = pd.read_csv(path)
    return (df,)


@app.cell
def _(df):
    print(f'Shape: {df.shape}')
    df.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
