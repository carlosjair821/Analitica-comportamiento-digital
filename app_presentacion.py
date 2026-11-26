import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Comportamiento Digital CDMX - Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid")


@st.cache_data
def cargar_y_procesar_datos(ruta_csv="telemetria_universitarios_cdmx.csv"):
    df_raw = pd.read_csv(ruta_csv)
    nulos_iniciales = int(df_raw.isnull().sum().sum())

    # 1. Filtrar errores de telemetría fuera de rango lógico
    df = df_raw[
        (df_raw["horas_redes_sociales"].isna() | (df_raw["horas_redes_sociales"] <= 18)) &
        (df_raw["horas_productividad"].isna() | (df_raw["horas_productividad"] <= 18)) &
        (df_raw["promedio_academico"].isna() | (df_raw["promedio_academico"] <= 10.0))
    ].copy()

    # 2. Imputación por mediana
    cols_num = [
        "horas_redes_sociales",
        "horas_productividad",
        "horas_ocio_streaming",
        "horas_uso_nocturno",
        "desbloqueos_dia",
        "promedio_academico",
    ]
    for col in cols_num:
        df[col] = df[col].fillna(df[col].median())

    # 3. Filtrado IQR
    for col in ["horas_redes_sociales", "desbloqueos_dia"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    df["riesgo_academico"] = (df["promedio_academico"] < 7.5).astype(int)

    # 4. Normalización
    cols_escalar = [
        "horas_redes_sociales",
        "horas_productividad",
        "horas_ocio_streaming",
        "horas_uso_nocturno",
        "desbloqueos_dia",
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols_escalar])
    df_scaled = pd.DataFrame(X_scaled, columns=[f"{c}_norm" for c in cols_escalar], index=df.index)
    df = pd.concat([df, df_scaled], axis=1)

    # 5. Clustering K-Means + PCA
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)
    silueta = silhouette_score(X_scaled, df["cluster_kmeans"])

    orden = df.groupby("cluster_kmeans")["horas_redes_sociales"].mean().sort_values().index
    mapa_nombres = {
        orden[0]: "Productivos",
        orden[1]: "Consumidores de Ocio",
        orden[2]: "Procrastinadores",
    }
    df["perfil_conductual"] = df["cluster_kmeans"].map(mapa_nombres)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df["pca_1"] = coords[:, 0]
    df["pca_2"] = coords[:, 1]
    var_pca = pca.explained_variance_ratio_.sum() * 100

    return df_raw, df, nulos_iniciales, silueta, var_pca, scaler, kmeans, mapa_nombres


@st.cache_resource
def entrenar_modelos(df):
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

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    importancias = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return rf, acc_lr, auc_lr, acc_rf, auc_rf, importancias


df_raw, df, nulos_ini, silueta, var_pca, scaler, kmeans, mapa_nombres = cargar_y_procesar_datos()
rf, acc_lr, auc_lr, acc_rf, auc_rf, importancias = entrenar_modelos(df)

# Encabezado principal
st.title("Modelado Estadístico y Predicción del Comportamiento Digital en Universitarios de la CDMX")
st.markdown("**Metodología:** CRISP-DM | **Entorno:** Python (`pandas`, `scipy`, `scikit-learn`, `seaborn`)")

# Métricas generales arriba
col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros Limpios", f"{len(df)} / {len(df_raw)}", f"-{len(df_raw)-len(df)} outliers")
col2.metric("Promedio General", f"{df['promedio_academico'].mean():.2f}")
col3.metric("Uso Redes (Media)", f"{df['horas_redes_sociales'].mean():.2f} hrs/día")
col4.metric("Precisión Random Forest", f"{acc_rf*100:.1f}%", f"AUC: {auc_rf:.3f}")

st.divider()

# Pestañas organizadas por módulo del equipo
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Datos y Limpieza (EDA)",
    "2. Correlación de Pearson",
    "3. Perfiles (Clustering)",
    "4. Predicción y Simulador en Vivo"
])

with tab1:
    st.subheader("Preparación de Microdatos y Estadística Descriptiva")
    st.write(f"Se detectaron **{nulos_ini} valores nulos** (imputados con mediana) y **{len(df_raw)-len(df)} registros atípicos** filtrados por rango lógico e IQR.")
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("**Muestra del Dataset Procesado**")
        st.dataframe(
            df[["id_estudiante", "alcaldia", "horas_redes_sociales", "horas_productividad", "horas_uso_nocturno", "promedio_academico", "perfil_conductual"]].head(15),
            use_container_width=True
        )
    with c2:
        st.markdown("**Resumen Estadístico**")
        st.dataframe(
            df[["horas_redes_sociales", "horas_productividad", "horas_ocio_streaming", "horas_uso_nocturno", "desbloqueos_dia", "promedio_academico"]].describe().round(2),
            use_container_width=True
        )

with tab2:
    st.subheader("Análisis de Correlación Matemática (Coeficiente de Pearson)")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        cols_corr = [
            "horas_redes_sociales",
            "horas_productividad",
            "horas_ocio_streaming",
            "horas_uso_nocturno",
            "desbloqueos_dia",
            "promedio_academico",
        ]
        sns.heatmap(df[cols_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax1)
        ax1.set_title("Matriz de Correlación de Pearson")
        st.pyplot(fig1)

    with c2:
        st.markdown("**Significancia Estadística frente al Promedio Académico**")
        filas_p = []
        for var in ["horas_redes_sociales", "horas_productividad", "horas_ocio_streaming", "horas_uso_nocturno", "desbloqueos_dia"]:
            r, p_val = pearsonr(df[var], df["promedio_academico"])
            filas_p.append({
                "Variable": var,
                "Coeficiente r": round(r, 3),
                "p-value": f"{p_val:.2e}",
                "Resultado": "Significativa (p < 0.05)" if p_val < 0.05 else "No significativa"
            })
        st.dataframe(pd.DataFrame(filas_p), use_container_width=True, hide_index=True)
        st.info("Un coeficiente negativo cercano a -1 (como redes sociales: r = -0.88) confirma que a mayor tiempo de uso, menor promedio académico.")

with tab3:
    st.subheader(f"Segmentación Conductual con K-Means y PCA (Silueta: {silueta:.3f} | Varianza PCA: {var_pca:.1f}%)")
    c1, c2 = st.columns(2)
    with c1:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            data=df, x="pca_1", y="pca_2", hue="perfil_conductual",
            palette="viridis", s=70, alpha=0.85, ax=ax2
        )
        ax2.set_title("Clústeres de Estudiantes (Proyección PCA 2D)")
        ax2.set_xlabel("Componente Principal 1")
        ax2.set_ylabel("Componente Principal 2")
        st.pyplot(fig2)

    with c2:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.boxplot(
            data=df, x="perfil_conductual", y="promedio_academico",
            order=["Productivos", "Consumidores de Ocio", "Procrastinadores"],
            hue="perfil_conductual", palette="Set2", legend=False, ax=ax3
        )
        ax3.set_title("Promedio Académico por Perfil Conductual")
        ax3.set_xlabel("Perfil Conductual")
        ax3.set_ylabel("Promedio Académico")
        st.pyplot(fig3)

with tab4:
    st.subheader("Modelado Predictivo y Simulador Interactivo")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig4, ax4 = plt.subplots(figsize=(7, 4.5))
        sns.barplot(
            x=importancias.values, y=importancias.index,
            hue=importancias.index, palette="magma", legend=False, ax=ax4
        )
        ax4.set_title("Importancia de Variables (Random Forest)")
        ax4.set_xlabel("Importancia Relativa")
        st.pyplot(fig4)
        st.caption(f"Comparativa: Random Forest (Exactitud: {acc_rf*100:.1f}%, AUC: {auc_rf:.3f}) vs. Regresión Logística (Exactitud: {acc_lr*100:.1f}%, AUC: {auc_lr:.3f}).")

    with c2:
        st.markdown("### Simulador en Vivo para Evaluación")
        st.write("Ajusta los hábitos diarios de un estudiante para predecir su perfil y riesgo académico:")
        s_redes = st.slider("Horas en Redes Sociales al día", 0.0, 9.0, 3.5, 0.1)
        s_prod = st.slider("Horas de Productividad / Estudio al día", 0.0, 7.0, 2.5, 0.1)
        s_ocio = st.slider("Horas de Ocio / Streaming al día", 0.0, 7.0, 2.5, 0.1)
        s_noct = st.slider("Horas de Uso Nocturno (madrugada)", 0.0, 5.0, 1.2, 0.1)
        s_desb = st.slider("Desbloqueos del teléfono al día", 20, 220, 90, 5)

        entrada = pd.DataFrame([[s_redes, s_prod, s_ocio, s_noct, s_desb]], columns=[
            "horas_redes_sociales", "horas_productividad", "horas_ocio_streaming",
            "horas_uso_nocturno", "desbloqueos_dia"
        ])
        prob_riesgo = rf.predict_proba(entrada)[0][1] * 100
        cluster_pred = kmeans.predict(scaler.transform(entrada))[0]
        perfil_pred = mapa_nombres[cluster_pred]

        st.success(f"**Perfil Conductual Asignado:** {perfil_pred}")
        if prob_riesgo >= 50:
            st.error(f"**Probabilidad de Riesgo Académico (< 7.5):** {prob_riesgo:.1f}% (Riesgo Alto)")
        else:
            st.info(f"**Probabilidad de Riesgo Académico (< 7.5):** {prob_riesgo:.1f}% (Rendimiento Estable)")
