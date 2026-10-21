"""
Métricas y gráficos para clasificación con clases desbalanceadas.

Vocabulario (positivo = cliente que abandona):
- Recall (sensibilidad): de todos los que SÍ abandonan, ¿qué fracción detectamos?
- Precisión: de los que marcamos como riesgo, ¿qué fracción realmente abandona?
- ROC-AUC: capacidad de ordenar bien (un abandono al azar recibe mayor
  probabilidad que un no-abandono al azar). 0.5 = azar, 1 = perfecto.
- PR-AUC (average precision): como ROC-AUC pero centrada en la clase positiva.
  Con 16% de positivos es más exigente y más informativa que ROC-AUC.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calcular_metricas(y_real, proba, umbral: float = 0.5) -> dict[str, float]:
    """Todas las métricas relevantes para un vector de probabilidades y un umbral."""
    pred = (proba >= umbral).astype(int)
    return {
        "umbral": umbral,
        "recall": recall_score(y_real, pred),
        "precision": precision_score(y_real, pred, zero_division=0),
        "f1": f1_score(y_real, pred),
        "roc_auc": roc_auc_score(y_real, proba),
        "pr_auc": average_precision_score(y_real, proba),
        "n_marcados": int(pred.sum()),
    }


def tabla_umbrales(y_real, proba, umbrales=None) -> pd.DataFrame:
    """
    Métricas para varios umbrales. Muestra el trade-off recall vs precisión:
    bajar el umbral captura más abandonos (sube recall) pero marca más clientes
    que no se iban (baja precisión, sube el costo de la campaña).
    """
    if umbrales is None:
        umbrales = np.round(np.arange(0.1, 0.91, 0.1), 2)
    return pd.DataFrame([calcular_metricas(y_real, proba, u) for u in umbrales]).set_index("umbral")


def elegir_umbral_por_recall(y_real, proba, recall_minimo: float = 0.80) -> float:
    """
    Devuelve el umbral MÁS ALTO que todavía alcanza el recall mínimo exigido.

    Lógica de negocio: el banco fija primero cuántos abandonos quiere capturar
    (p. ej. 80%) y luego, dentro de ese compromiso, quiere molestar al menor
    número posible de clientes fieles. Umbral más alto = menos clientes marcados
    = mayor precisión, manteniendo el recall pactado.
    """
    precisiones, recalls, umbrales = precision_recall_curve(y_real, proba)
    # precision_recall_curve devuelve n+1 puntos; el último no tiene umbral.
    recalls, umbrales = recalls[:-1], umbrales
    candidatos = umbrales[recalls >= recall_minimo]
    if len(candidatos) == 0:
        return 0.5
    return float(candidatos.max())


def graficar_curvas(probas_por_modelo: dict[str, np.ndarray], y_real, ruta_guardado=None):
    """Curvas ROC y Precision-Recall de varios modelos en una misma figura."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    tasa_base = float(np.mean(y_real))

    for nombre, proba in probas_por_modelo.items():
        fpr, tpr, _ = roc_curve(y_real, proba)
        prec, rec, _ = precision_recall_curve(y_real, proba)
        ax_roc.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_real, proba):.3f})")
        ax_pr.plot(rec, prec, label=f"{nombre} (AP={average_precision_score(y_real, proba):.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Azar")
    ax_roc.set(xlabel="Tasa de falsos positivos", ylabel="Recall (tasa de verdaderos positivos)", title="Curva ROC")
    ax_roc.legend(loc="lower right")

    ax_pr.axhline(tasa_base, color="k", ls="--", lw=1, label=f"Azar (tasa base={tasa_base:.2f})")
    ax_pr.set(xlabel="Recall", ylabel="Precisión", title="Curva Precisión-Recall")
    ax_pr.legend(loc="upper right")

    fig.tight_layout()
    if ruta_guardado:
        fig.savefig(ruta_guardado, dpi=120, bbox_inches="tight")
    return fig


def graficar_matriz_confusion(y_real, proba, umbral: float, titulo: str = "", ruta_guardado=None):
    """Matriz de confusión con etiquetas en español."""
    pred = (proba >= umbral).astype(int)
    matriz = confusion_matrix(y_real, pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(matriz, display_labels=["Se queda", "Abandona"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set(xlabel="Predicción", ylabel="Realidad", title=titulo or f"Umbral = {umbral:.2f}")
    fig.tight_layout()
    if ruta_guardado:
        fig.savefig(ruta_guardado, dpi=120, bbox_inches="tight")
    return fig


def resumen_confusion(y_real, proba, umbral: float) -> dict[str, int]:
    """Conteos legibles de la matriz de confusión."""
    pred = (proba >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_real, pred).ravel()
    return {
        "verdaderos_positivos (abandonos detectados)": int(vp),
        "falsos_negativos (abandonos NO detectados)": int(fn),
        "falsos_positivos (fieles marcados por error)": int(fp),
        "verdaderos_negativos (fieles bien clasificados)": int(vn),
    }


def costo_esperado(y_real, proba, umbral: float, costo_abandono_no_detectado: float, costo_contacto: float) -> float:
    """
    Costo total de operar con un umbral dado, bajo supuestos de negocio explícitos.

    - Cada abandono NO detectado (falso negativo) cuesta `costo_abandono_no_detectado`
      (valor del cliente que se pierde).
    - Cada cliente marcado como riesgo (verdadero o falso positivo) cuesta
      `costo_contacto` (la llamada, el descuento, la oferta de retención).

    Sirve para ver que el umbral "correcto" no es una decisión estadística sino
    de negocio: depende de cuánto vale un cliente frente a cuánto cuesta retenerlo.
    """
    pred = (proba >= umbral).astype(int)
    _, fp, fn, vp = confusion_matrix(y_real, pred).ravel()
    return float(fn * costo_abandono_no_detectado + (fp + vp) * costo_contacto)


def tabla_costos(y_real, proba, costo_abandono_no_detectado: float, costo_contacto: float, umbrales=None) -> pd.DataFrame:
    """Costo esperado para una rejilla de umbrales, junto con recall y clientes marcados."""
    if umbrales is None:
        umbrales = np.round(np.arange(0.05, 0.96, 0.05), 2)
    filas = []
    for u in umbrales:
        m = calcular_metricas(y_real, proba, u)
        filas.append(
            {
                "umbral": u,
                "recall": m["recall"],
                "precision": m["precision"],
                "n_marcados": m["n_marcados"],
                "costo_total": costo_esperado(y_real, proba, u, costo_abandono_no_detectado, costo_contacto),
            }
        )
    return pd.DataFrame(filas).set_index("umbral")
