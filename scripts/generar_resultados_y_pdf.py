"""
Script de ejecución completa del proyecto Bank Marketing (Data Mining).
Ejecuta el preprocesamiento, selección de características, comparación de modelos,
evaluación en prueba, análisis de priorización de campañas (Ganancia/Lift),
genera todas las visualizaciones en outputs/figuras/ y construye Informe_Bank_Marketing.pdf.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "bank_marketing_uci.csv"
FIG_DIR = ROOT / "outputs" / "figuras"
OUT_DIR = ROOT / "outputs"
HTML_PATH = OUT_DIR / "Informe_Bank_Marketing.html"
PDF_PATH = ROOT / "Informe_Bank_Marketing.pdf"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = ["#1D4E89", "#00B2CA", "#F79256", "#7DCFB6"]


def cargar_datos() -> tuple[pd.DataFrame, pd.Series]:
    """Carga el conjunto UCI Bank Marketing desde caché local o repositorio UCI."""
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        X = df.drop(columns=["y"])
        y = df[["y"]]
    else:
        from ucimlrepo import fetch_ucirepo

        bm = fetch_ucirepo(id=222)
        X = bm.data.features
        y = bm.data.targets
        DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([X, y], axis=1).to_csv(DATA_CSV, index=False)
    return X, y


def puntaje_pearson(X_arr: np.ndarray, y_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Función de puntaje compatible con SelectKBest basada en |r| de Pearson."""
    puntajes, p_valores = [], []
    for idx in range(X_arr.shape[1]):
        r, p_val = pearsonr(X_arr[:, idx], y_arr)
        puntajes.append(0.0 if np.isnan(r) else abs(r))
        p_valores.append(1.0 if np.isnan(p_val) else p_val)
    return np.asarray(puntajes), np.asarray(p_valores)


def calcular_tabla_deciles(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Construye tabla de ganancia acumulada y lift por deciles para priorización."""
    df_eval = pd.DataFrame({"y": y_true, "prob": y_prob}).sort_values(
        "prob", ascending=False
    ).reset_index(drop=True)
    n_total = len(df_eval)
    total_positivos = int(df_eval["y"].sum())
    tasa_base = total_positivos / n_total

    filas = []
    for decil in range(1, 11):
        corte = int(np.ceil(n_total * decil / 10))
        sub = df_eval.iloc[:corte]
        positivos_capturados = int(sub["y"].sum())
        tasa_respuesta = positivos_capturados / len(sub)
        ganancia_acum = positivos_capturados / total_positivos
        lift = tasa_respuesta / tasa_base
        filas.append(
            {
                "decil": f"Top {decil * 10}%",
                "clientes_contactados": len(sub),
                "suscriptores_capturados": positivos_capturados,
                "tasa_conversion_pct": round(tasa_respuesta * 100, 2),
                "ganancia_acumulada_pct": round(ganancia_acum * 100, 2),
                "lift": round(lift, 2),
            }
        )
    return pd.DataFrame(filas)


def ejecutar_analisis() -> dict:
    # 1. Carga y diagnóstico
    X_raw, y_raw = cargar_datos()
    datos_brutos = pd.concat([X_raw.copy(), y_raw.copy()], axis=1)
    datos_brutos.columns = [str(col).strip() for col in datos_brutos.columns]

    duplicados_brutos = int(datos_brutos.duplicated().sum())
    faltantes_resumen = (
        datos_brutos.isna().sum()[lambda s: s > 0].to_dict()
    )

    # Figura 1: Distribución de la variable objetivo y tasa por poutcome/month
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    conteo_y = datos_brutos["y"].str.strip().str.lower().value_counts()
    axes[0].bar(["No", "Sí"], [conteo_y["no"], conteo_y["yes"]], color=["#8ECAE6", "#1D4E89"])
    for i, val in enumerate([conteo_y["no"], conteo_y["yes"]]):
        pct = val / len(datos_brutos) * 100
        axes[0].text(i, val + 600, f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_title("Distribución de la variable objetivo (y)")
    axes[0].set_ylabel("Número de clientes")
    axes[0].set_ylim(0, conteo_y.max() * 1.18)

    tasa_pout = (
        datos_brutos.assign(
            poutcome_clean=datos_brutos["poutcome"].fillna("sin_registro"),
            y_bin=(datos_brutos["y"].str.strip().str.lower() == "yes").astype(int),
        )
        .groupby("poutcome_clean")["y_bin"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )
    sns.barplot(x=tasa_pout.values, y=tasa_pout.index, hue=tasa_pout.index, legend=False, palette="Blues_r", ax=axes[1])
    axes[1].set_title("Tasa de suscripción por resultado previo")
    axes[1].set_xlabel("Tasa de suscripción (%)")
    axes[1].set_ylabel("poutcome")
    for idx, val in enumerate(tasa_pout.values):
        axes[1].text(val + 1, idx, f"{val:.1f}%", va="center", fontsize=8.5)
    axes[1].set_xlim(0, max(tasa_pout.values) * 1.2)

    orden_meses = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    tasa_mes = (
        datos_brutos.assign(y_bin=(datos_brutos["y"].str.strip().str.lower() == "yes").astype(int))
        .groupby("month")["y_bin"]
        .mean()
        .reindex(orden_meses)
        * 100
    )
    axes[2].bar(tasa_mes.index, tasa_mes.values, color="#1D4E89")
    axes[2].axhline(datos_brutos["y"].eq("yes").mean() * 100, color="#E76F51", linestyle="--", linewidth=1.2, label="Promedio (11.7%)")
    axes[2].set_title("Tasa de suscripción por mes")
    axes[2].set_ylabel("Tasa de suscripción (%)")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_exploracion_objetivo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figura 2: Diagnóstico visual de valores atípicos IQR
    cols_num_diag = datos_brutos.select_dtypes(include="number").columns.tolist()
    n_cols = 3
    n_rows = int(np.ceil(len(cols_num_diag) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.0 * n_rows))
    for ax, col in zip(axes.flat, cols_num_diag):
        sns.boxplot(x=datos_brutos[col], ax=ax, color="#8ECAE6")
        ax.set_title(f"{col}: distribución y atípicos IQR", fontsize=10)
        ax.set_xlabel(col)
    for ax in axes.flat[len(cols_num_diag):]:
        ax.set_visible(False)
    fig.suptitle("Diagnóstico visual de valores atípicos en variables numéricas", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_diagnostico_atipicos.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. Limpieza e ingeniería de características
    datos_limpios = datos_brutos.drop_duplicates().copy()
    for col in datos_limpios.select_dtypes(include=["object", "string"]).columns:
        datos_limpios[col] = datos_limpios[col].str.strip().str.lower()

    datos_modelo = datos_limpios.copy()
    datos_modelo["presion_contacto"] = datos_modelo["campaign"] / (datos_modelo["previous"] + 1)
    datos_modelo["balance_ajustado_credito"] = datos_modelo["balance"] / (
        1
        + datos_modelo["housing"].eq("yes").astype(int)
        + datos_modelo["loan"].eq("yes").astype(int)
    )

    X_modelo = datos_modelo.drop(columns=["y", "duration"])
    y_modelo = (datos_modelo["y"] == "yes").astype(int)

    columnas_numericas = X_modelo.select_dtypes(include="number").columns.tolist()
    columnas_categoricas = X_modelo.select_dtypes(exclude="number").columns.tolist()

    # 3. Partición estratificada 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_modelo, y_modelo, test_size=0.20, random_state=42, stratify=y_modelo
    )

    # 4. Filtros de selección de características (solo en entrenamiento)
    res_chi = []
    for col in columnas_categoricas:
        tabla = pd.crosstab(X_train[col], y_train)
        est, p_val, gl, _ = chi2_contingency(tabla)
        n = tabla.to_numpy().sum()
        min_dim = min(tabla.shape) - 1
        v_cramer = (est / (n * min_dim)) ** 0.5 if min_dim else 0.0
        res_chi.append(
            {"variable": col, "chi2": est, "p_valor": p_val, "gl": gl, "v_cramer": v_cramer}
        )
    ranking_chi2 = pd.DataFrame(res_chi).sort_values(["v_cramer", "chi2"], ascending=False).reset_index(drop=True)

    res_pearson = []
    for col in columnas_numericas:
        r, p_val = pearsonr(X_train[col], y_train)
        res_pearson.append(
            {"variable": col, "correlacion": r, "abs_correlacion": abs(r), "p_valor": p_val}
        )
    ranking_pearson = pd.DataFrame(res_pearson).sort_values("abs_correlacion", ascending=False).reset_index(drop=True)

    # Figura 3: Filtros Chi-cuadrada y Pearson lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    sns.barplot(data=ranking_chi2, x="v_cramer", y="variable", hue="variable", legend=False, palette="Blues_r", ax=axes[0])
    axes[0].set_title("Filtro Chi-cuadrada (categóricas vs y)")
    axes[0].set_xlabel("V de Cramér (fuerza de asociación)")
    axes[0].set_ylabel("Variable categórica")
    for idx, val in enumerate(ranking_chi2["v_cramer"]):
        axes[0].text(val + 0.008, idx, f"{val:.3f}", va="center", fontsize=8.5)
    axes[0].set_xlim(0, ranking_chi2["v_cramer"].max() * 1.18)

    sns.barplot(data=ranking_pearson, x="correlacion", y="variable", hue="variable", legend=False, palette="vlag", ax=axes[1])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Filtro Pearson (numéricas vs y)")
    axes[1].set_xlabel("Correlación de Pearson (r)")
    axes[1].set_ylabel("Variable numérica")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_filtros_seleccion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 5. Pipelines de preprocesamiento + selección + 4 modelos de clasificación
    def crear_preprocesador_seleccion() -> ColumnTransformer:
        tf_num = Pipeline(
            steps=[
                ("imputacion", SimpleImputer(strategy="median")),
                ("escala", MinMaxScaler()),
                ("seleccion_pearson", SelectKBest(score_func=puntaje_pearson, k=6)),
            ]
        )
        tf_cat = Pipeline(
            steps=[
                ("imputacion", SimpleImputer(strategy="most_frequent")),
                ("codificacion", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("seleccion_chi2", SelectKBest(score_func=chi2, k=25)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("numericas", tf_num, columnas_numericas),
                ("categoricas", tf_cat, columnas_categoricas),
            ],
            remainder="drop",
        )

    modelos = {
        "Regresión Logística": Pipeline(
            steps=[
                ("prep", crear_preprocesador_seleccion()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
        "Árbol de Decisión": Pipeline(
            steps=[
                ("prep", crear_preprocesador_seleccion()),
                ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=42)),
            ]
        ),
        "Bosque Aleatorio": Pipeline(
            steps=[
                ("prep", crear_preprocesador_seleccion()),
                ("clf", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("prep", crear_preprocesador_seleccion()),
                ("clf", HistGradientBoostingClassifier(max_depth=6, max_iter=150, learning_rate=0.05, class_weight="balanced", random_state=42)),
            ]
        ),
    }

    # 6. Validación cruzada estratificada 5-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    filas_cv = []
    for nombre, pipe in modelos.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        filas_cv.append(
            {
                "Modelo": nombre,
                "ROC AUC": round(float(res["test_roc_auc"].mean()), 4),
                "ROC AUC std": round(float(res["test_roc_auc"].std()), 4),
                "PR AUC": round(float(res["test_pr_auc"].mean()), 4),
                "Exactitud Balanceada": round(float(res["test_balanced_accuracy"].mean()), 4),
                "F1": round(float(res["test_f1"].mean()), 4),
                "Precisión": round(float(res["test_precision"].mean()), 4),
                "Sensibilidad": round(float(res["test_recall"].mean()), 4),
            }
        )
    df_cv = pd.DataFrame(filas_cv).sort_values("ROC AUC", ascending=False).reset_index(drop=True)

    # Figura 4: Comparativa de validación cruzada
    df_cv_melt = df_cv.melt(
        id_vars="Modelo",
        value_vars=["ROC AUC", "Exactitud Balanceada", "F1", "Precisión", "Sensibilidad"],
        var_name="Métrica",
        value_name="Valor",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=df_cv_melt, x="Métrica", y="Valor", hue="Modelo", palette=PALETTE, ax=ax)
    ax.set_title("Comparación de modelos en validación cruzada estratificada (5 particiones)")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Modelo", loc="upper right", frameon=True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7.5, padding=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_comparacion_cv.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 7. Entrenamiento final y evaluación en conjunto de prueba
    filas_test = []
    probabilidades_test = {}
    predicciones_test = {}
    modelos_ajustados = {}

    for nombre, pipe in modelos.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        modelos_ajustados[nombre] = pipe
        probabilidades_test[nombre] = y_prob
        predicciones_test[nombre] = y_pred
        filas_test.append(
            {
                "Modelo": nombre,
                "ROC AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
                "PR AUC": round(float(average_precision_score(y_test, y_prob)), 4),
                "Exactitud": round(float(accuracy_score(y_test, y_pred)), 4),
                "Exactitud Balanceada": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
                "F1": round(float(f1_score(y_test, y_pred)), 4),
                "Precisión": round(float(precision_score(y_test, y_pred)), 4),
                "Sensibilidad": round(float(recall_score(y_test, y_pred)), 4),
            }
        )
    df_test = pd.DataFrame(filas_test).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    # Selección del mejor modelo basada estrictamente en validación cruzada (sin fuga de prueba)
    mejor_modelo_nombre = df_cv.iloc[0]["Modelo"]
    mejor_modelo = modelos_ajustados[mejor_modelo_nombre]

    # Figura 5: Curvas ROC y Precisión-Sensibilidad comparativas en prueba
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for (nombre, y_prob), color in zip(probabilidades_test.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{nombre} (AUC = {auc_val:.3f})", color=color, linewidth=2)

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap_val = average_precision_score(y_test, y_prob)
        axes[1].plot(rec, prec, label=f"{nombre} (AP = {ap_val:.3f})", color=color, linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar (AUC = 0.500)")
    axes[0].set_title("Curvas ROC en conjunto de prueba")
    axes[0].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[0].set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    axes[0].legend(loc="lower right", fontsize=8.5)
    tasa_pos = float(y_test.mean())
    axes[1].axhline(tasa_pos, color="black", linestyle="--", linewidth=1, label=f"Base aleatoria ({tasa_pos:.3f})")
    axes[1].set_title("Curvas Precisión-Sensibilidad en prueba")
    axes[1].set_xlabel("Sensibilidad (Recall)")
    axes[1].set_ylabel("Precisión (Precision)")
    axes[1].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_curvas_roc_pr.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figura 6: Matrices de confusión comparativas (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, (nombre, pipe) in zip(axes.flat, modelos_ajustados.items()):
        ConfusionMatrixDisplay.from_estimator(
            pipe, X_test, y_test, display_labels=["No", "Sí"], cmap="Blues", colorbar=False, ax=ax
        )
        ax.grid(False)
        auc_m = df_test.loc[df_test["Modelo"] == nombre, "ROC AUC"].iloc[0]
        ax.set_title(f"{nombre} (ROC AUC = {auc_m:.3f})", fontsize=10.5)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
    fig.suptitle("Matrices de confusión en el conjunto de prueba (n = 9,043)", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_matrices_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figura 7: Importancia de variables por permutación del mejor modelo
    perm = permutation_importance(
        mejor_modelo,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    df_imp = (
        pd.DataFrame(
            {
                "variable": X_test.columns,
                "importancia_media": perm.importances_mean,
                "importancia_std": perm.importances_std,
            }
        )
        .sort_values("importancia_media", ascending=False)
        .reset_index(drop=True)
    )
    top_imp = df_imp.head(10)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.barh(
        top_imp["variable"][::-1],
        top_imp["importancia_media"][::-1],
        xerr=top_imp["importancia_std"][::-1],
        color="#1D4E89",
        alpha=0.9,
        capsize=3,
    )
    ax.set_title(f"Top 10 variables más influyentes ({mejor_modelo_nombre} - Permutation Importance)")
    ax.set_xlabel("Caída promedio en ROC AUC al permutar la variable")
    ax.set_ylabel("Variable")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_importancia_variables.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figura 8: Curva de Ganancia Acumulada y Lift para priorización telefónica
    tabla_deciles = calcular_tabla_deciles(
        y_test.to_numpy(), probabilidades_test[mejor_modelo_nombre]
    )
    pct_clientes = np.arange(0, 101, 10)
    ganancias = [0.0] + tabla_deciles["ganancia_acumulada_pct"].tolist()
    lifts = tabla_deciles["lift"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(pct_clientes, ganancias, marker="o", linewidth=2.2, color="#1D4E89", label=mejor_modelo_nombre)
    axes[0].plot([0, 100], [0, 100], "k--", linewidth=1.2, label="Llamadas al azar")
    axes[0].fill_between(pct_clientes, pct_clientes, ganancias, color="#8ECAE6", alpha=0.25)
    for x_pt in [10, 20, 30]:
        y_pt = ganancias[x_pt // 10]
        axes[0].annotate(f"{y_pt:.1f}%", xy=(x_pt, y_pt), xytext=(x_pt + 3, y_pt - 5), fontsize=8.5, fontweight="bold", color="#1D4E89")
    axes[0].set_title("Curva de Ganancia Acumulada (Priorización de llamadas)")
    axes[0].set_xlabel("% de clientes contactados (ordenados por probabilidad)")
    axes[0].set_ylabel("% de suscriptores reales capturados")
    axes[0].legend(loc="lower right")

    axes[1].bar(tabla_deciles["decil"], lifts, color="#00B2CA")
    axes[1].axhline(1.0, color="#E76F51", linestyle="--", linewidth=1.2, label="Azar (Lift = 1.0)")
    for idx, val in enumerate(lifts):
        axes[1].text(idx, val + 0.08, f"{val:.2f}x", ha="center", va="bottom", fontsize=8)
    axes[1].set_title(f"Factor Lift acumulado por decil ({mejor_modelo_nombre})")
    axes[1].set_ylabel("Lift respecto a llamadas aleatorias")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylim(0, max(lifts) * 1.18)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_ganancia_lift_campana.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    resumen = {
        "n_total": int(len(datos_brutos)),
        "n_duplicados": duplicados_brutos,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positivos_pct": round(float(y_modelo.mean() * 100), 2),
        "faltantes": {k: int(v) for k, v in faltantes_resumen.items()},
        "ranking_chi2": ranking_chi2.head(5).to_dict(orient="records"),
        "ranking_pearson": ranking_pearson.head(5).to_dict(orient="records"),
        "cv": df_cv.to_dict(orient="records"),
        "test": df_test.to_dict(orient="records"),
        "mejor_modelo": mejor_modelo_nombre,
        "importancia_top5": df_imp.head(5).to_dict(orient="records"),
        "deciles": tabla_deciles.to_dict(orient="records"),
        "reporte_mejor_modelo": classification_report(
            y_test, predicciones_test[mejor_modelo_nombre], target_names=["No", "Sí"]
        ),
    }

    (OUT_DIR / "metricas_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return resumen


if __name__ == "__main__":
    res = ejecutar_analisis()
    print("=== VALIDACIÓN CRUZADA (5-FOLD) ===")
    print(pd.DataFrame(res["cv"]).to_string(index=False))
    print("\n=== CONJUNTO DE PRUEBA ===")
    print(pd.DataFrame(res["test"]).to_string(index=False))
    print(f"\nMejor modelo: {res['mejor_modelo']}")
    print("\n=== TOP 5 IMPORTANCIA (PERMUTATION) ===")
    print(pd.DataFrame(res["importancia_top5"]).to_string(index=False))
    print("\n=== TABLA DE DECILES (PRIORIZACIÓN) ===")
    print(pd.DataFrame(res["deciles"]).to_string(index=False))
