"""
Preprocesamiento de variables para los modelos.

Se construye un `ColumnTransformer` que se inserta dentro del Pipeline de cada
modelo. Así el escalado y la codificación se "aprenden" únicamente con los datos
de entrenamiento de cada fold de la validación cruzada, nunca con los de
validación. Hacerlo fuera del pipeline contaminaría las métricas.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Variables categóricas (texto). Se codifican con one-hot.
COLUMNAS_CATEGORICAS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

# Variables numéricas. Se escalan solo para la regresión logística.
COLUMNAS_NUMERICAS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Ticket_Promedio",  # variable derivada creada en src/datos.py
]


def construir_preprocesador(escalar_numericas: bool) -> ColumnTransformer:
    """
    Crea el transformador de columnas.

    Parámetros
    ----------
    escalar_numericas : bool
        True para regresión logística (necesita variables en escalas comparables
        para que la regularización y el optimizador funcionen bien).
        False para Random Forest y boosting: los árboles parten por umbrales,
        son invariantes a la escala, y escalar solo dificulta la lectura.

    Decisiones:
    - `handle_unknown="ignore"`: si en producción apareciera una categoría nueva,
      el modelo no falla, la trata como "ninguna de las conocidas".
    - `drop="if_binary"`: para variables de 2 categorías (Gender) deja una sola
      columna. Evita la columna redundante que duplicaría información.
    - `sparse_output=False`: salida densa, más simple para inspeccionar y para
      HistGradientBoosting, que no acepta matrices dispersas.
    """
    transformador_categorico = OneHotEncoder(
        handle_unknown="ignore", drop="if_binary", sparse_output=False
    )
    transformador_numerico = StandardScaler() if escalar_numericas else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", transformador_numerico, COLUMNAS_NUMERICAS),
            ("cat", transformador_categorico, COLUMNAS_CATEGORICAS),
        ],
        verbose_feature_names_out=False,
    )


def nombres_de_variables_transformadas(preprocesador: ColumnTransformer) -> list[str]:
    """Nombres de columnas tras el preprocesamiento (ya ajustado)."""
    return list(preprocesador.get_feature_names_out())


def resumen_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla rápida: tipo, nº de valores únicos y ejemplo por columna."""
    return pd.DataFrame(
        {
            "tipo": df.dtypes.astype(str),
            "n_unicos": df.nunique(),
            "ejemplo": df.iloc[0],
        }
    )
