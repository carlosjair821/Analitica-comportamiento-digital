# %%
"""
Proyecto: Modelado Estadístico y Predicción del Comportamiento Digital en Universitarios de la CDMX
Metodología: CRISP-DM
Archivo organizado en 5 módulos (1 por cada integrante del equipo).

Requisitos de librerías:
pip install pandas numpy scipy scikit-learn matplotlib seaborn
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# %%
# ==============================================================================
# PERSONA 1: INGENIERÍA DE DATOS Y LIMPIEZA (EDA)
# ==============================================================================
def modulo_1_limpieza_datos(ruta_csv="telemetria_universitarios_cdmx.csv"):
    print("--- [MÓDULO 1] Limpieza y Preparación de Datos ---")
    df = pd.read_csv(ruta_csv)
    print(f"Registros iniciales: {len(df)} | Valores nulos totales: {df.isnull().sum().sum()}")

    # 1. Eliminar valores fuera de rango lógico (errores de sensor/telemetría)
    df = df[
        (df["horas_redes_sociales"].isna() | (df["horas_redes_sociales"] <= 18)) &
        (df["horas_productividad"].isna() | (df["horas_productividad"] <= 18)) &
        (df["promedio_academico"].isna() | (df["promedio_academico"] <= 10.0))
    ].copy()

    # 2. Imputación de valores faltantes usando la mediana (robusta ante sesgos)
    columnas_numericas = [
        "horas_redes_sociales",
        "horas_productividad",
        "horas_ocio_streaming",
        "horas_uso_nocturno",
        "desbloqueos_dia",
        "promedio_academico",
    ]
    for col in columnas_numericas:
        df[col] = df[col].fillna(df[col].median())

    # 3. Detección y filtrado de outliers mediante Rango Intercuartílico (IQR)
    for col in ["horas_redes_sociales", "desbloqueos_dia"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR
        df = df[(df[col] >= limite_inf) & (df[col] <= limite_sup)]

    # Recalcular riesgo académico tras limpieza
    df["riesgo_academico"] = (df["promedio_academico"] < 7.5).astype(int)

    # 4. Normalización con StandardScaler para algoritmos de ML
    scaler = StandardScaler()
    cols_escalar = [
        "horas_redes_sociales",
        "horas_productividad",
        "horas_ocio_streaming",
        "horas_uso_nocturno",
        "desbloqueos_dia",
    ]
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[cols_escalar]),
        columns=[f"{c}_norm" for c in cols_escalar],
        index=df.index,
    )
    df_limpio = pd.concat([df, df_scaled], axis=1)
    df_limpio.to_csv("telemetria_limpia.csv", index=False, encoding="utf-8")
    print(f"Registros limpios guardados en 'telemetria_limpia.csv': {len(df_limpio)}\n")
    return df_limpio


# %%
# ==============================================================================
# PERSONA 2: ESTADÍSTICA DESCRIPTIVA Y CORRELACIÓN DE PEARSON
# ==============================================================================
def modulo_2_correlacion(df):
    print("--- [MÓDULO 2] Estadística Descriptiva y Correlación de Pearson ---")
    resumen = df[["horas_redes_sociales", "horas_productividad", "horas_uso_nocturno", "promedio_academico"]].describe()
    print("Estadísticas descriptivas:\n", resumen.round(2), "\n")

    variables_analizar = [
        "horas_redes_sociales",
        "horas_productividad",
        "horas_ocio_streaming",
        "horas_uso_nocturno",
        "desbloqueos_dia",
    ]
    resultados_pearson = []
    for var in variables_analizar:
        r, p_val = pearsonr(df[var], df["promedio_academico"])
        significancia = "Significativa (p < 0.05)" if p_val < 0.05 else "No significativa"
        resultados_pearson.append({
            "Variable": var,
            "Coeficiente_r": round(r, 4),
            "p_value": round(p_val, 6),
            "Interpretacion": significancia
        })
        print(f"Correlación {var} vs. Promedio: r = {r:.3f} (p = {p_val:.4e}) -> {significancia}")

    print()
    return pd.DataFrame(resultados_pearson)


# %%
# ==============================================================================
# PERSONA 3: SEGMENTACIÓN DE PERFILES (CLUSTERING: K-MEANS Y DBSCAN + PCA)
# ==============================================================================
def modulo_3_clustering(df):
    print("--- [MÓDULO 3] Segmentación Conductual (Clustering) ---")
    cols_norm = [
        "horas_redes_sociales_norm",
        "horas_productividad_norm",
        "horas_ocio_streaming_norm",
        "horas_uso_nocturno_norm",
        "desbloqueos_dia_norm",
    ]
    X = df[cols_norm]

    # 1. K-Means con k=3 perfiles (Productivos, Equilibrados, Procrastinadores)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster_kmeans"] = kmeans.fit_predict(X)
    silueta = silhouette_score(X, df["cluster_kmeans"])
    print(f"Puntaje de Silueta (K-Means, k=3): {silueta:.3f}")

    # Etiquetar automáticamente según el promedio de horas en redes sociales de cada clúster
    orden_clusters = df.groupby("cluster_kmeans")["horas_redes_sociales"].mean().sort_values().index
    mapa_nombres = {
        orden_clusters[0]: "Productivos",
        orden_clusters[1]: "Consumidores de Ocio",
        orden_clusters[2]: "Procrastinadores",
    }
    df["perfil_conductual"] = df["cluster_kmeans"].map(mapa_nombres)
    print("Distribución de perfiles encontrados:")
    print(df["perfil_conductual"].value_counts(), "\n")

    # 2. DBSCAN para detección de comportamiento atípico
    dbscan = DBSCAN(eps=1.3, min_samples=6)
    df["cluster_dbscan"] = dbscan.fit_predict(X)
    outliers_dbscan = (df["cluster_dbscan"] == -1).sum()
    print(f"Casos atípicos aislados por DBSCAN: {outliers_dbscan}")

    # 3. Reducción a 2 dimensiones con PCA para visualización
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["pca_1"] = coords[:, 0]
    df["pca_2"] = coords[:, 1]
    print(f"Varianza explicada por PCA (2D): {pca.explained_variance_ratio_.sum()*100:.1f}%\n")
    return df


# %%
# ==============================================================================
# PERSONA 4: MODELADO PREDICTIVO DE RIESGO ACADÉMICO
# ==============================================================================
def modulo_4_modelos_predictivos(df):
    print("--- [MÓDULO 4] Modelado Predictivo de Riesgo Académico ---")
    features = [
        "horas_redes_sociales",
        "horas_productividad",
        "horas_ocio_streaming",
        "horas_uso_nocturno",
        "desbloqueos_dia",
    ]
    X = df[features]
    y = df["riesgo_academico"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Modelo 1: Regresión Logística
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    pred_lr = log_reg.predict(X_test)
    auc_lr = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
    print(f"Regresión Logística -> ROC-AUC: {auc_lr:.3f}")
    print(classification_report(y_test, pred_lr, target_names=["Estable", "Riesgo"]))

    # Modelo 2: Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"Random Forest -> ROC-AUC: {auc_rf:.3f}")
    print(classification_report(y_test, pred_rf, target_names=["Estable", "Riesgo"]))

    importancias = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("Importancia de variables (Random Forest):")
    print(importancias.round(3), "\n")
    return rf, importancias, confusion_matrix(y_test, pred_rf)


# %%
# ==============================================================================
# PERSONA 5: VISUALIZACIÓN EN VIVO Y EXPORTACIÓN DE GRÁFICAS
# ==============================================================================
def modulo_5_visualizaciones(df, importancias, matriz_conf):
    print("--- [MÓDULO 5] Generando y Mostrando Gráficas en Pantalla ---")
    sns.set_theme(style="whitegrid")

    # 1. Heatmap de Correlación de Pearson
    plt.figure("Gráfica 1 - Correlación de Pearson", figsize=(8, 6))
    cols_corr = [
        "horas_redes_sociales",
        "horas_productividad",
        "horas_ocio_streaming",
        "horas_uso_nocturno",
        "desbloqueos_dia",
        "promedio_academico",
    ]
    sns.heatmap(df[cols_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Matriz de Correlación de Pearson: Hábitos Digitales vs. Promedio")
    plt.tight_layout()
    plt.savefig("grafica_1_correlacion.png", dpi=300)

    # 2. Gráfica de Clústeres (PCA 2D)
    plt.figure("Gráfica 2 - Clústeres PCA", figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="pca_1",
        y="pca_2",
        hue="perfil_conductual",
        palette="viridis",
        s=70,
        alpha=0.85,
    )
    plt.title("Segmentación de Universitarios CDMX (K-Means + PCA)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.tight_layout()
    plt.savefig("grafica_2_clusters_pca.png", dpi=300)

    # 3. Boxplot de Promedio Académico por Perfil Conductual
    plt.figure("Gráfica 3 - Promedio por Perfil", figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="perfil_conductual",
        y="promedio_academico",
        order=["Productivos", "Consumidores de Ocio", "Procrastinadores"],
        hue="perfil_conductual",
        palette="Set2",
        legend=False,
    )
    plt.title("Distribución del Promedio Académico por Perfil Conductual")
    plt.xlabel("Perfil Conductual")
    plt.ylabel("Promedio Académico")
    plt.tight_layout()
    plt.savefig("grafica_3_promedio_por_perfil.png", dpi=300)

    # 4. Importancia de Variables (Random Forest)
    plt.figure("Gráfica 4 - Importancia de Variables", figsize=(8, 5))
    sns.barplot(
        x=importancias.values,
        y=importancias.index,
        hue=importancias.index,
        palette="magma",
        legend=False,
    )
    plt.title("Factores Digitales más Determinantes en el Riesgo Académico")
    plt.xlabel("Importancia Relativa (Random Forest)")
    plt.tight_layout()
    plt.savefig("grafica_4_importancia_variables.png", dpi=300)

    print("Gráficas guardadas en PNG y abiertas en ventanas interactivas.")
    plt.show()


# %%
if __name__ == "__main__":
    df_datos = modulo_1_limpieza_datos("telemetria_universitarios_cdmx.csv")
    modulo_2_correlacion(df_datos)
    df_segmentado = modulo_3_clustering(df_datos)
    modelo_rf, imp_vars, cm = modulo_4_modelos_predictivos(df_segmentado)
    modulo_5_visualizaciones(df_segmentado, imp_vars, cm)
