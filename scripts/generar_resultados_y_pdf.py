"""
Ejecuta el flujo completo del proyecto Bank Marketing (Data Mining) con 3 modelos
clásicos (Regresión Logística, Árbol de Decisión y Random Forest), guarda las
gráficas en outputs/figuras/ y las métricas en outputs/metricas_resumen.json.
"""

from __future__ import annotations

import json
import os
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

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "bank_marketing_uci.csv"
FIG_DIR = ROOT / "outputs" / "figuras"
OUT_DIR = ROOT / "outputs"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cargar_datos():
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        return df.drop(columns=["y"]), df[["y"]]
    from ucimlrepo import fetch_ucirepo

    bm = fetch_ucirepo(id=222)
    return bm.data.features, bm.data.targets


def puntaje_pearson(X_arr, y_arr):
    puntajes, p_valores = [], []
    for i in range(X_arr.shape[1]):
        r, p = pearsonr(X_arr[:, i], y_arr)
        puntajes.append(0.0 if np.isnan(r) else abs(r))
        p_valores.append(1.0 if np.isnan(p) else p)
    return np.asarray(puntajes), np.asarray(p_valores)


def ejecutar():
    X_raw, y_raw = cargar_datos()
    datos = pd.concat([X_raw.copy(), y_raw.copy()], axis=1)
    datos.columns = [str(c).strip() for c in datos.columns]

    duplicados = int(datos.duplicated().sum())
    datos_limpios = datos.drop_duplicates().copy()
    for col in datos_limpios.select_dtypes(include=["object", "string"]).columns:
        datos_limpios[col] = datos_limpios[col].str.strip().str.lower()

    # Gráfica 1: Distribución de la variable objetivo y tasa por resultado previo
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    conteo = datos_limpios["y"].value_counts()
    axes[0].bar(["No", "Sí"], [conteo["no"], conteo["yes"]], color=["#8ecae6", "#219ebc"])
    axes[0].set_title("Distribución de la variable objetivo (y)")
    axes[0].set_ylabel("Cantidad de clientes")
    for i, v in enumerate([conteo["no"], conteo["yes"]]):
        axes[0].text(i, v + 500, f"{v:,} ({v/len(datos_limpios)*100:.1f}%)", ha="center", fontsize=9)
    axes[0].set_ylim(0, conteo.max() * 1.15)

    tasa_pout = (
        datos_limpios.assign(
            pout=datos_limpios["poutcome"].fillna("sin registro"),
            y_bin=(datos_limpios["y"] == "yes").astype(int),
        )
        .groupby("pout")["y_bin"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )
    axes[1].bar(tasa_pout.index, tasa_pout.values, color="#219ebc")
    axes[1].set_title("Porcentaje de suscripción según campaña anterior")
    axes[1].set_ylabel("Suscripción (%)")
    for i, v in enumerate(tasa_pout.values):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    axes[1].set_ylim(0, max(tasa_pout.values) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_distribucion_y.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Gráfica 2: Boxplots de valores atípicos
    cols_num_diag = datos_limpios.select_dtypes(include="number").columns.tolist()
    n_cols = 3
    n_rows = int(np.ceil(len(cols_num_diag) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.8 * n_rows))
    for ax, col in zip(axes.flat, cols_num_diag):
        sns.boxplot(x=datos_limpios[col], ax=ax, color="#8ecae6")
        ax.set_title(f"{col}")
    for ax in axes.flat[len(cols_num_diag):]:
        ax.set_visible(False)
    fig.suptitle("Diagramas de caja de las variables numéricas", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Ingeniería de características (variables derivadas) y exclusión de duration
    datos_modelo = datos_limpios.copy()
    datos_modelo["presion_contacto"] = datos_modelo["campaign"] / (datos_modelo["previous"] + 1)
    datos_modelo["balance_ajustado_credito"] = datos_modelo["balance"] / (
        1
        + datos_modelo["housing"].eq("yes").astype(int)
        + datos_modelo["loan"].eq("yes").astype(int)
    )

    X = datos_modelo.drop(columns=["y", "duration"])
    y = (datos_modelo["y"] == "yes").astype(int)

    cols_num = X.select_dtypes(include="number").columns.tolist()
    cols_cat = X.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Filtros Chi-cuadrada y Pearson (solo en entrenamiento)
    res_chi = []
    for col in cols_cat:
        tabla = pd.crosstab(X_train[col], y_train)
        est, p, _, _ = chi2_contingency(tabla)
        n = tabla.to_numpy().sum()
        min_dim = min(tabla.shape) - 1
        v_cramer = (est / (n * min_dim)) ** 0.5 if min_dim else 0.0
        res_chi.append({"variable": col, "v_cramer": v_cramer, "p_valor": p})
    ranking_chi2 = pd.DataFrame(res_chi).sort_values("v_cramer", ascending=False).reset_index(drop=True)

    res_pearson = []
    for col in cols_num:
        r, p = pearsonr(X_train[col], y_train)
        res_pearson.append({"variable": col, "correlacion": r, "abs_r": abs(r), "p_valor": p})
    ranking_pearson = pd.DataFrame(res_pearson).sort_values("abs_r", ascending=False).reset_index(drop=True)

    # Gráfica 3: Ranking de filtros
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    sns.barplot(data=ranking_chi2, x="v_cramer", y="variable", color="#219ebc", ax=axes[0])
    axes[0].set_title("Variables categóricas (Chi-cuadrada / V de Cramér)")
    axes[0].set_xlabel("V de Cramér")
    axes[0].set_ylabel("")

    sns.barplot(data=ranking_pearson, x="correlacion", y="variable", color="#8ecae6", ax=axes[1])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Variables numéricas (Correlación de Pearson)")
    axes[1].set_xlabel("Correlación r con y")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_filtros.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Preprocesador con selección dentro del pipeline
    def crear_prep():
        tf_num = Pipeline(
            [
                ("imputacion", SimpleImputer(strategy="median")),
                ("escala", MinMaxScaler()),
                ("filtro_pearson", SelectKBest(score_func=puntaje_pearson, k=6)),
            ]
        )
        tf_cat = Pipeline(
            [
                ("imputacion", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("filtro_chi2", SelectKBest(score_func=chi2, k=25)),
            ]
        )
        return ColumnTransformer(
            [("num", tf_num, cols_num), ("cat", tf_cat, cols_cat)],
            remainder="drop",
        )

    # 3 modelos clásicos vistos en clase de Minería de Datos
    modelos = {
        "Regresión Logística": Pipeline(
            [
                ("prep", crear_prep()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
        "Árbol de Decisión": Pipeline(
            [
                ("prep", crear_prep()),
                ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", crear_prep()),
                ("clf", RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1)),
            ]
        ),
    }

    # Validación cruzada estratificada (5 particiones)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
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
                "Exactitud Balanceada": round(float(res["test_balanced_accuracy"].mean()), 4),
                "F1": round(float(res["test_f1"].mean()), 4),
                "Precisión": round(float(res["test_precision"].mean()), 4),
                "Sensibilidad": round(float(res["test_recall"].mean()), 4),
            }
        )
    df_cv = pd.DataFrame(filas_cv).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    mejor_nombre = df_cv.iloc[0]["Modelo"]

    # Gráfica 4: Comparación de modelos en validación cruzada
    df_melt = df_cv.melt(
        id_vars="Modelo",
        value_vars=["ROC AUC", "Exactitud Balanceada", "F1", "Precisión", "Sensibilidad"],
        var_name="Métrica",
        value_name="Valor",
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    sns.barplot(data=df_melt, x="Métrica", y="Valor", hue="Modelo", palette="Blues_d", ax=ax)
    ax.set_title("Comparación de modelos en validación cruzada (5 particiones)")
    ax.set_ylim(0, 1.0)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_comparacion_modelos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Evaluación en conjunto de prueba
    filas_test = []
    probs_test = {}
    preds_test = {}
    ajustados = {}
    for nombre, pipe in modelos.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        ajustados[nombre] = pipe
        preds_test[nombre] = pred
        probs_test[nombre] = prob
        filas_test.append(
            {
                "Modelo": nombre,
                "ROC AUC": round(float(roc_auc_score(y_test, prob)), 4),
                "Exactitud": round(float(accuracy_score(y_test, pred)), 4),
                "Exactitud Balanceada": round(float(balanced_accuracy_score(y_test, pred)), 4),
                "F1": round(float(f1_score(y_test, pred)), 4),
                "Precisión": round(float(precision_score(y_test, pred)), 4),
                "Sensibilidad": round(float(recall_score(y_test, pred)), 4),
            }
        )
    df_test = pd.DataFrame(filas_test).sort_values("ROC AUC", ascending=False).reset_index(drop=True)

    # Gráfica 5: Curvas ROC comparativas y matriz de confusión del mejor modelo
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for nombre, prob in probs_test.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        axes[0].plot(fpr, tpr, label=f"{nombre} (AUC = {auc:.3f})", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.500)")
    axes[0].set_title("Curvas ROC en el conjunto de prueba")
    axes[0].set_xlabel("Tasa de Falsos Positivos")
    axes[0].set_ylabel("Sensibilidad (Verdaderos Positivos)")
    axes[0].legend(loc="lower right", fontsize=8.5)

    ConfusionMatrixDisplay.from_estimator(
        ajustados[mejor_nombre],
        X_test,
        y_test,
        display_labels=["No", "Sí"],
        cmap="Blues",
        colorbar=False,
        ax=axes[1],
    )
    axes[1].grid(False)
    axes[1].set_title(f"Matriz de confusión - {mejor_nombre}")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Valor real")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_roc_y_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Importancia de características (feature_importances_ del Random Forest)
    mejor_pipe = ajustados[mejor_nombre]
    prep_fit = mejor_pipe.named_steps["prep"]
    clf_fit = mejor_pipe.named_steps["clf"]
    nombres_num = prep_fit.named_transformers_["num"].named_steps["filtro_pearson"].get_feature_names_out(cols_num)
    onehot_names = prep_fit.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cols_cat)
    nombres_cat = prep_fit.named_transformers_["cat"].named_steps["filtro_chi2"].get_feature_names_out(onehot_names)
    nombres_todos = list(nombres_num) + list(nombres_cat)

    df_imp = (
        pd.DataFrame({"caracteristica": nombres_todos, "importancia": clf_fit.feature_importances_})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    # Priorización de llamadas (Top 10%, Top 20%, Top 30%, Top 50%, 100%)
    df_ord = pd.DataFrame({"y": y_test.to_numpy(), "prob": probs_test[mejor_nombre]}).sort_values(
        "prob", ascending=False
    ).reset_index(drop=True)
    total_pos = int(df_ord["y"].sum())
    tasa_base = total_pos / len(df_ord)

    priorizacion = []
    for pct in [10, 20, 30, 50, 100]:
        corte = int(np.ceil(len(df_ord) * pct / 100))
        sub = df_ord.iloc[:corte]
        pos = int(sub["y"].sum())
        tasa = pos / len(sub)
        priorizacion.append(
            {
                "grupo": f"Top {pct}%" if pct < 100 else "Todos (100%)",
                "llamadas": len(sub),
                "suscriptores": pos,
                "tasa_exito_pct": round(tasa * 100, 1),
                "captura_total_pct": round(pos / total_pos * 100, 1),
                "mejora_vs_azar": round(tasa / tasa_base, 2),
            }
        )
    df_prio = pd.DataFrame(priorizacion)

    # Gráfica 6: Importancia de variables y priorización de llamadas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    top10 = df_imp.head(10)
    axes[0].barh(top10["caracteristica"][::-1], top10["importancia"][::-1], color="#219ebc")
    axes[0].set_title(f"Top 10 variables más importantes ({mejor_nombre})")
    axes[0].set_xlabel("Importancia")

    grupos_plot = df_prio.iloc[:-1]
    axes[1].bar(grupos_plot["grupo"], grupos_plot["tasa_exito_pct"], color="#8ecae6", label="Modelo Random Forest")
    axes[1].axhline(tasa_base * 100, color="red", linestyle="--", label=f"Sin modelo ({tasa_base*100:.1f}%)")
    axes[1].set_title("Tasa de suscripción al priorizar llamadas")
    axes[1].set_ylabel("Clientes que contratan (%)")
    for i, v in enumerate(grupos_plot["tasa_exito_pct"]):
        axes[1].text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9)
    axes[1].set_ylim(0, max(grupos_plot["tasa_exito_pct"]) * 1.2)
    axes[1].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_importancia_y_priorizacion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    resumen = {
        "n_total": int(len(datos)),
        "duplicados": duplicados,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positivos_pct": round(float(y.mean() * 100), 2),
        "ranking_chi2": ranking_chi2.head(4).to_dict(orient="records"),
        "ranking_pearson": ranking_pearson.head(4).to_dict(orient="records"),
        "cv": df_cv.to_dict(orient="records"),
        "test": df_test.to_dict(orient="records"),
        "mejor_modelo": mejor_nombre,
        "importancia_top8": df_imp.head(8).to_dict(orient="records"),
        "priorizacion": df_prio.to_dict(orient="records"),
        "reporte_clasificacion": classification_report(
            y_test, preds_test[mejor_nombre], target_names=["No", "Sí"]
        ),
    }
    (OUT_DIR / "metricas_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return resumen


if __name__ == "__main__":
    r = ejecutar()
    print("=== VALIDACIÓN CRUZADA ===")
    print(pd.DataFrame(r["cv"]).to_string(index=False))
    print("\n=== PRUEBA ===")
    print(pd.DataFrame(r["test"]).to_string(index=False))
    print(f"\nMejor modelo: {r['mejor_modelo']}")
    print("\n=== PRIORIZACIÓN ===")
    print(pd.DataFrame(r["priorizacion"]).to_string(index=False))
