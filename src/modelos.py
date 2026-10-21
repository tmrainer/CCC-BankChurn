"""
Definición de los modelos candidatos y validación cruzada.

Tres familias, de menor a mayor complejidad:
1. Regresión logística: lineal, interpretable, buen baseline.
2. Random Forest: muchos árboles independientes promediados (bagging).
3. HistGradientBoosting: árboles secuenciales, cada uno corrige errores del
   anterior (boosting). Suele ser el más preciso en datos tabulares.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocesamiento import construir_preprocesador

SEMILLA = 42

# Métricas que se calculan en cada fold. Con clases desbalanceadas, accuracy
# engaña (un modelo que diga "nadie abandona" acierta 84%), por eso se incluye
# pero no se usa para decidir.
METRICAS_CV = {
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
}


def construir_modelos(balancear_clases: bool = True) -> dict[str, Pipeline]:
    """
    Devuelve un diccionario {nombre: Pipeline(preprocesador + modelo)}.

    Parámetros
    ----------
    balancear_clases : bool
        Si True, cada modelo pondera más los errores sobre la clase minoritaria
        (abandono). Equivale a decirle al modelo "equivocarte con un cliente que
        se va cuesta ~5 veces más que equivocarte con uno que se queda"
        (5 ≈ 84/16). Es la forma más simple y honesta de atacar el desbalance:
        no inventa datos sintéticos (SMOTE) ni descarta datos reales (undersampling).
    """
    peso = "balanced" if balancear_clases else None

    logistica = LogisticRegression(
        max_iter=5000,
        class_weight=peso,
        random_state=SEMILLA,
        # C=1.0 por defecto: regularización L2 moderada. Con 13 numéricas + ~20
        # dummies y 8k filas no hay riesgo real de sobreajuste en un modelo lineal.
    )

    bosque = RandomForestClassifier(
        n_estimators=500,  # más árboles = menos varianza; 500 es suficiente, no mejora después
        min_samples_leaf=3,  # hojas con mínimo 3 clientes evita árboles que memoricen casos únicos
        max_features="sqrt",  # subconjunto de variables por partición: árboles más distintos entre sí
        class_weight="balanced_subsample" if balancear_clases else None,
        n_jobs=-1,
        random_state=SEMILLA,
    )

    boosting = HistGradientBoostingClassifier(
        learning_rate=0.05,  # paso pequeño: aprende despacio pero generaliza mejor
        max_iter=500,  # tope de árboles; el early stopping suele parar antes
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,  # separa 10% del train interno y para si no mejora
        validation_fraction=0.1,
        n_iter_no_change=30,
        class_weight=peso,
        random_state=SEMILLA,
    )

    return {
        "Regresión logística": Pipeline(
            [("prep", construir_preprocesador(escalar_numericas=True)), ("modelo", logistica)]
        ),
        "Random Forest": Pipeline(
            [("prep", construir_preprocesador(escalar_numericas=False)), ("modelo", bosque)]
        ),
        "HistGradientBoosting": Pipeline(
            [("prep", construir_preprocesador(escalar_numericas=False)), ("modelo", boosting)]
        ),
    }


def crear_cv(n_folds: int = 5) -> StratifiedKFold:
    """
    Validación cruzada estratificada.

    Cada fold conserva el ~16% de abandono. Con 5 folds, cada modelo se entrena
    5 veces con 80% del train y se evalúa en el 20% restante. El promedio de las
    5 métricas es una estimación más estable que una única partición.
    """
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)


def validar_modelos(
    modelos: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, n_folds: int = 5
) -> pd.DataFrame:
    """
    Ejecuta validación cruzada sobre cada modelo y devuelve una tabla
    con media y desviación estándar por métrica.

    Las métricas de clasificación (recall, precision, f1) usan umbral 0.5.
    ROC-AUC y PR-AUC no dependen del umbral: miden la calidad del ranking.
    """
    cv = crear_cv(n_folds)
    filas = []
    for nombre, pipeline in modelos.items():
        resultados = cross_validate(
            pipeline, X, y, cv=cv, scoring=METRICAS_CV, n_jobs=-1, return_train_score=False
        )
        fila = {"modelo": nombre}
        for metrica in METRICAS_CV:
            valores = resultados[f"test_{metrica}"]
            fila[f"{metrica}_media"] = valores.mean()
            fila[f"{metrica}_std"] = valores.std()
        filas.append(fila)
    return pd.DataFrame(filas).set_index("modelo")


def probabilidades_cv(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_folds: int = 5
) -> np.ndarray:
    """
    Probabilidad de abandono "fuera de muestra" para cada fila del train.

    Cada cliente recibe la predicción del modelo entrenado en los otros 4 folds,
    así que nunca se predice a sí mismo. Estas probabilidades sirven para elegir
    el umbral de decisión sin tocar el test.
    """
    cv = crear_cv(n_folds)
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
