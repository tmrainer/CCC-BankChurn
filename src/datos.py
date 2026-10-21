"""
Carga y limpieza del dataset BankChurners (Kaggle: sakshigoyal7/credit-card-customers).

Responsabilidades de este módulo:
- Leer el CSV crudo.
- Eliminar columnas con fuga de información (leakage) y columnas redundantes.
- Construir la variable objetivo binaria `abandono`.
- Separar predictores / objetivo y hacer la partición train/test estratificada.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Ruta por defecto del CSV crudo, relativa a la raíz del proyecto.
RUTA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "BankChurners.csv"

# Columna identificadora del cliente. No es predictora, pero se conserva para
# unir las probabilidades con cada cliente en el CSV final para Tableau.
COLUMNA_ID = "CLIENTNUM"

# Columna original con la etiqueta de abandono (texto).
COLUMNA_OBJETIVO_ORIGINAL = "Attrition_Flag"

# Nombre de la variable objetivo binaria que creamos: 1 = abandonó, 0 = sigue.
COLUMNA_OBJETIVO = "abandono"

# Las dos últimas columnas del CSV son las probabilidades de un clasificador
# Naive Bayes que alguien entrenó usando la propia etiqueta de abandono.
# Usarlas sería hacer trampa: el modelo "vería" la respuesta a través de ellas.
# Esto se llama fuga de información (leakage) y hay que eliminarlas siempre.
COLUMNAS_LEAKAGE = [
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_"
    "Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_"
    "Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

# Avg_Open_To_Buy = Credit_Limit - Total_Revolving_Bal exactamente.
# Es una combinación lineal de otras dos columnas: no aporta información nueva
# y genera multicolinealidad perfecta, que vuelve inestables los coeficientes
# de la regresión logística. Se elimina.
COLUMNAS_REDUNDANTES = ["Avg_Open_To_Buy"]

# Variable derivada: monto promedio por transacción. El EDA y la dependencia
# parcial muestran que, a igual número de transacciones, el cliente que hace
# pocas compras grandes abandona más que el que hace muchas compras pequeñas.
# Esta variable captura ese patrón directamente, sin depender de que el modelo
# combine dos columnas colineales (monto y número de transacciones).
COLUMNA_TICKET = "Ticket_Promedio"


def cargar_datos(ruta: Path | str = RUTA_RAW) -> pd.DataFrame:
    """Lee el CSV crudo tal cual está en Kaggle, sin modificarlo."""
    return pd.read_csv(ruta)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la limpieza mínima necesaria para modelar.

    Pasos:
    1. Elimina columnas con leakage y redundantes.
    2. Crea la variable objetivo binaria `abandono`.
    3. Elimina la etiqueta original en texto.
    4. Crea `Ticket_Promedio` = monto total / nº de transacciones.

    No imputa nulos porque el dataset no tiene. Las categorías "Unknown" de
    educación, estado civil e ingreso se conservan como una categoría más:
    el hecho de que el banco no sepa el dato puede ser informativo por sí mismo.
    """
    df = df.copy()
    columnas_a_eliminar = [c for c in COLUMNAS_LEAKAGE + COLUMNAS_REDUNDANTES if c in df.columns]
    df = df.drop(columns=columnas_a_eliminar)

    df[COLUMNA_OBJETIVO] = (df[COLUMNA_OBJETIVO_ORIGINAL] == "Attrited Customer").astype(int)
    df = df.drop(columns=[COLUMNA_OBJETIVO_ORIGINAL])

    # Total_Trans_Ct es >= 10 en todo el dataset, no hay división por cero.
    df[COLUMNA_TICKET] = (df["Total_Trans_Amt"] / df["Total_Trans_Ct"]).round(2)
    return df


def separar_predictores_objetivo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y). X no incluye ni el ID ni el objetivo."""
    X = df.drop(columns=[COLUMNA_ID, COLUMNA_OBJETIVO])
    y = df[COLUMNA_OBJETIVO]
    return X, y


def dividir_train_test(
    df: pd.DataFrame, proporcion_test: float = 0.2, semilla: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partición estratificada train/test a nivel de DataFrame completo (incluye ID).

    Estratificar garantiza que ambos conjuntos tengan el mismo ~16% de abandono.
    Con clases desbalanceadas, una partición aleatoria simple podría dejar
    un test con muy pocos positivos y métricas poco fiables.

    El conjunto de test se guarda y NO se toca hasta la evaluación final:
    todas las decisiones (elección de modelo, umbral, hiperparámetros) se toman
    solo con validación cruzada sobre train.
    """
    train, test = train_test_split(
        df,
        test_size=proporcion_test,
        stratify=df[COLUMNA_OBJETIVO],
        random_state=semilla,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def cargar_y_preparar(ruta: Path | str = RUTA_RAW) -> pd.DataFrame:
    """Atajo: carga + limpieza en una sola llamada."""
    return limpiar_datos(cargar_datos(ruta))
