"""
Módulo Adicional: Regresión Lineal Simple y Múltiple en Python
Proyecto: Modelado Estadístico y Predicción del Comportamiento Digital en Universitarios de la CDMX
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


def ejecutar_regresion_lineal(ruta_csv="telemetria_limpia.csv"):
    print("--- [MÓDULO EXTRA] Regresión Lineal Simple y Múltiple ---")
    df = pd.read_csv(ruta_csv)
    y = df["promedio_academico"]

    # ==========================================================================
    # 1. REGRESIÓN LINEAL SIMPLE (Horas en Redes Sociales -> Promedio Académico)
    # ==========================================================================
    X_simple = df[["horas_redes_sociales"]]
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_simple, y, test_size=0.25, random_state=42
    )

    modelo_simple = LinearRegression()
    modelo_simple.fit(X_tr_s, y_tr_s)
    pred_simple = modelo_simple.predict(X_te_s)

    r2_s = r2_score(y_te_s, pred_simple)
    rmse_s = np.sqrt(mean_squared_error(y_te_s, pred_simple))
    mae_s = mean_absolute_error(y_te_s, pred_simple)

    print("\n1) Regresión Lineal Simple (Redes Sociales -> Promedio):")
    print(f"   Ecuación: Promedio = {modelo_simple.intercept_:.3f} + ({modelo_simple.coef_[0]:.3f} * horas_redes_sociales)")
    print(f"   R² (Varianza explicada): {r2_s:.4f} ({r2_s*100:.1f}%)")
    print(f"   RMSE (Error cuadrático medio): {rmse_s:.4f}")
    print(f"   MAE (Error absoluto medio): {mae_s:.4f}")

    # ==========================================================================
    # 2. REGRESIÓN LINEAL MÚLTIPLE (Todas las métricas -> Promedio Académico)
    # ==========================================================================
    features = [
        "horas_redes_sociales",
        "horas_productividad",
        "horas_ocio_streaming",
        "horas_uso_nocturno",
        "desbloqueos_dia",
    ]
    X_mult = df[features]
    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
        X_mult, y, test_size=0.25, random_state=42
    )

    modelo_mult = LinearRegression()
    modelo_mult.fit(X_tr_m, y_tr_m)
    pred_mult = modelo_mult.predict(X_te_m)

    r2_m = r2_score(y_te_m, pred_mult)
    rmse_m = np.sqrt(mean_squared_error(y_te_m, pred_mult))
    mae_m = mean_absolute_error(y_te_m, pred_mult)

    print("\n2) Regresión Lineal Múltiple (5 variables -> Promedio):")
    print(f"   Intercepto (b0): {modelo_mult.intercept_:.4f}")
    for col, coef in zip(features, modelo_mult.coef_):
        print(f"   Coeficiente {col}: {coef:+.4f}")
    print(f"   R² (Varianza explicada): {r2_m:.4f} ({r2_m*100:.1f}%)")
    print(f"   RMSE: {rmse_m:.4f} | MAE: {mae_m:.4f}\n")

    # ==========================================================================
    # 3. VISUALIZACIÓN DE AMBOS MODELOS
    # ==========================================================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Gráfica izquierda: Regresión Lineal Simple
    sns.regplot(
        data=df,
        x="horas_redes_sociales",
        y="promedio_academico",
        ax=axes[0],
        scatter_kws={"alpha": 0.6, "color": "#2b5c8f"},
        line_kws={"color": "#d9534f", "linewidth": 2.2, "label": f"Ajuste Lineal (R² = {r2_s:.2f})"},
    )
    axes[0].set_title("Regresión Lineal Simple: Redes Sociales vs. Promedio")
    axes[0].set_xlabel("Horas en Redes Sociales al día")
    axes[0].set_ylabel("Promedio Académico")
    axes[0].legend()

    # Gráfica derecha: Regresión Lineal Múltiple (Real vs Predicho)
    axes[1].scatter(y_te_m, pred_mult, alpha=0.7, color="#2e8b57", edgecolors="k", s=55)
    lim_min = min(y_te_m.min(), pred_mult.min()) - 0.2
    lim_max = max(y_te_m.max(), pred_mult.max()) + 0.2
    axes[1].plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=2, label="Predicción Ideal (y = x)")
    axes[1].set_title(f"Regresión Lineal Múltiple: Real vs. Predicho (R² = {r2_m:.2f})")
    axes[1].set_xlabel("Promedio Académico Real")
    axes[1].set_ylabel("Promedio Académico Predicho")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("grafica_5_regresion_lineal.png", dpi=300)
    print("Gráfica guardada como 'grafica_5_regresion_lineal.png' (300 DPI).")
    plt.show()


if __name__ == "__main__":
    ejecutar_regresion_lineal("telemetria_limpia.csv")
