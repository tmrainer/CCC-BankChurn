"""
Interpretación de modelos: qué variables importan y cómo explicarlo a negocio.

Dos herramientas:
- Coeficientes de la regresión logística, convertidos a odds ratios.
- Importancia por permutación: se baraja una variable y se mide cuánto cae la
  métrica. Si el modelo no la usaba, no cae nada. Funciona con cualquier modelo
  y mide importancia REAL sobre datos no vistos, a diferencia de la importancia
  interna de los árboles (que favorece variables con muchos valores distintos).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# Traducción de cada variable a lenguaje de negocio. Se usa en gráficos,
# en las conclusiones y en el CSV para Tableau.
DICCIONARIO_VARIABLES = {
    "Customer_Age": "Edad del cliente",
    "Gender": "Género",
    "Dependent_count": "Nº de dependientes",
    "Education_Level": "Nivel educativo",
    "Marital_Status": "Estado civil",
    "Income_Category": "Rango de ingresos",
    "Card_Category": "Tipo de tarjeta",
    "Months_on_book": "Meses como cliente",
    "Total_Relationship_Count": "Nº de productos contratados",
    "Months_Inactive_12_mon": "Meses inactivo (últimos 12)",
    "Contacts_Count_12_mon": "Contactos con el banco (últimos 12)",
    "Credit_Limit": "Límite de crédito",
    "Total_Revolving_Bal": "Saldo revolvente",
    "Total_Amt_Chng_Q4_Q1": "Cambio en monto gastado (T4 vs T1)",
    "Total_Trans_Amt": "Monto total de transacciones (12 m)",
    "Total_Trans_Ct": "Nº total de transacciones (12 m)",
    "Total_Ct_Chng_Q4_Q1": "Cambio en nº de transacciones (T4 vs T1)",
    "Avg_Utilization_Ratio": "Utilización promedio del crédito",
    "Ticket_Promedio": "Ticket promedio (monto / nº transacciones)",
}


def traducir(nombre_variable: str) -> str:
    """Traduce una columna (o dummy 'Columna_Categoria') a su nombre de negocio."""
    if nombre_variable in DICCIONARIO_VARIABLES:
        return DICCIONARIO_VARIABLES[nombre_variable]
    for original, traduccion in DICCIONARIO_VARIABLES.items():
        if nombre_variable.startswith(original + "_"):
            categoria = nombre_variable[len(original) + 1 :]
            return f"{traduccion} = {categoria}"
    return nombre_variable


def coeficientes_logistica(pipeline: Pipeline) -> pd.DataFrame:
    """
    Coeficientes de la regresión logística ya entrenada.

    odds_ratio = exp(coef). Lectura para variables numéricas estandarizadas:
    "subir la variable una desviación estándar multiplica las chances de
    abandono por odds_ratio". Para dummies: "pertenecer a esta categoría
    multiplica las chances por odds_ratio respecto a la categoría de referencia".
    """
    nombres = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["modelo"].coef_.ravel()
    tabla = pd.DataFrame(
        {
            "variable": nombres,
            "variable_negocio": [traducir(n) for n in nombres],
            "coeficiente": coefs,
            "odds_ratio": np.exp(coefs),
        }
    )
    tabla["abs_coef"] = tabla["coeficiente"].abs()
    return tabla.sort_values("abs_coef", ascending=False).drop(columns="abs_coef").reset_index(drop=True)


def importancia_permutacion(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    metrica: str = "average_precision",
    n_repeticiones: int = 10,
    semilla: int = 42,
) -> pd.DataFrame:
    """
    Importancia por permutación sobre las columnas ORIGINALES (antes del one-hot).

    Se permuta cada columna original completa, así una categórica se evalúa como
    un bloque y no dummy por dummy. Se mide la caída en PR-AUC: métrica alineada
    con el problema desbalanceado.
    """
    resultado = permutation_importance(
        pipeline, X, y, scoring=metrica, n_repeats=n_repeticiones, random_state=semilla, n_jobs=-1
    )
    tabla = pd.DataFrame(
        {
            "variable": X.columns,
            "variable_negocio": [traducir(c) for c in X.columns],
            "importancia_media": resultado.importances_mean,
            "importancia_std": resultado.importances_std,
        }
    )
    return tabla.sort_values("importancia_media", ascending=False).reset_index(drop=True)


def graficar_importancias(tabla: pd.DataFrame, top: int = 12, titulo: str = "", ruta_guardado=None):
    """Barras horizontales de las `top` variables más importantes."""
    datos = tabla.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * top + 1))
    ax.barh(
        datos["variable_negocio"],
        datos["importancia_media"],
        xerr=datos.get("importancia_std"),
        color="#2a6f97",
    )
    ax.set(xlabel="Caída en PR-AUC al barajar la variable", title=titulo)
    fig.tight_layout()
    if ruta_guardado:
        fig.savefig(ruta_guardado, dpi=120, bbox_inches="tight")
    return fig


def segmentar_riesgo(proba: np.ndarray, corte_alto: float, corte_medio: float) -> pd.Series:
    """
    Clasifica en tres segmentos para el dashboard.

    - Alto: proba >= corte_alto (umbral operativo elegido con recall objetivo).
    - Medio: entre corte_medio y corte_alto (vigilar, no llamar todavía).
    - Bajo: resto.
    """
    return pd.Series(
        np.select(
            [proba >= corte_alto, proba >= corte_medio],
            ["Alto", "Medio"],
            default="Bajo",
        ),
        name="segmento_riesgo",
    )
