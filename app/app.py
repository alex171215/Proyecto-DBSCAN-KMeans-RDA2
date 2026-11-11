"""
Aplicación Web de Segmentación de Estudiantes (Streamlit)
Responsable: Persona 3 (Paquete C)

Modelo final: K-Means (K=4), elegido en notebooks/03_final.ipynb e informe/decision_modelo.md.

- Recibe las 16 variables en escala original (sin normalizar).
- Asigna el segmento con models/pipeline_kmeans.joblib (imputer + scaler + kmeans).
- Proyecta al estudiante en 2D con models/pca.joblib, sólo para visualizar.
- No entrena ningún modelo: sólo carga artefactos ya exportados.

Ejecutar desde la raíz del proyecto:
    streamlit run app/app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "student_burnout_dropout_dataset_2_link 1.csv"

# Etiqueta legible, unidad y paso del control para cada variable del contrato.
CAMPOS = {
    "Age": ("Edad", "años", 1.0),
    "Year_of_Study": ("Año de estudio", "año", 1.0),
    "Attendance_Percent": ("Asistencia", "%", 0.5),
    "Study_Hours_Per_Day": ("Horas de estudio al día", "h", 0.1),
    "Previous_GPA": ("Promedio anterior (GPA)", "puntos", 0.1),
    "Backlogs": ("Materias pendientes", "materias", 1.0),
    "Sleep_Hours": ("Horas de sueño", "h", 0.1),
    "Screen_Time_Hours": ("Tiempo de pantalla", "h/día", 0.1),
    "Exercise_Freq_Per_Week": ("Ejercicio por semana", "veces", 1.0),
    "Social_Activity_Score": ("Actividad social", "puntaje", 0.1),
    "Financial_Stress_Score": ("Estrés financiero", "puntaje", 0.1),
    "Family_Support_Score": ("Apoyo familiar", "puntaje", 0.1),
    "Stress_Level": ("Nivel de estrés", "puntaje", 0.1),
    "Anxiety_Score": ("Ansiedad", "puntaje", 0.1),
    "Motivation_Score": ("Motivación", "puntaje", 0.1),
    "Peer_Pressure_Score": ("Presión de pares", "puntaje", 0.1),
}

RECOMENDACION = {
    "Rezago y presión financiera":
        "Becas o ayuda económica, tutorías para materias pendientes y plan de recuperación académica.",
    "Participación y apoyo familiar":
        "Mantener el acompañamiento; invitarlo como mentor par de estudiantes de otros grupos.",
    "Baja asistencia y alta pantalla":
        "Alertas tempranas de asistencia y talleres sobre uso de pantallas y hábitos de sueño.",
    "Buen GPA con seguimiento de bienestar":
        "Reconocimiento académico y ofrecer consejería preventiva por estrés financiero.",
}

COLORES = ["#2f6db5", "#e07b39", "#3a9e6f", "#9b59b6"]


@st.cache_resource
def cargar_modelos():
    pipeline = joblib.load(ROOT / "models" / "pipeline_kmeans.joblib")
    pca = joblib.load(ROOT / "models" / "pca.joblib")
    return pipeline, pca


@st.cache_data
def cargar_datos():
    features = json.loads((ROOT / "data" / "feature_names.json").read_text())
    segmentos = json.loads((ROOT / "segments" / "segments_kmeans.json").read_text())
    pca_2d = pd.read_csv(ROOT / "data" / "pca_2d.csv")
    labels = pd.read_csv(ROOT / "segments" / "labels_kmeans.csv")["cluster"]
    perfiles = pd.read_csv(ROOT / "segments" / "perfiles_kmeans.csv")
    crudo = pd.read_csv(DATA_PATH)[features]
    rangos = pd.DataFrame({"min": crudo.min(), "max": crudo.max(), "mediana": crudo.median()})
    return features, segmentos, pca_2d, labels, perfiles, rangos


def proyectar_2d(pipeline, pca, registro):
    """16D crudo → imputer + scaler del pipeline → PCA base; devuelve (PC1, PC2)."""
    escalado = pipeline[:-1].transform(registro)
    return pca.transform(escalado)[0, :2]


def mapa_2d(pca_2d, labels, segmentos, punto, cluster_nuevo):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for k in sorted(labels.unique()):
        m = labels == k
        ax.scatter(pca_2d.loc[m, "PC1"], pca_2d.loc[m, "PC2"], s=14, alpha=0.45,
                   color=COLORES[k], label=f"{k}: {segmentos[str(k)]}")
    ax.scatter(*punto, s=320, marker="*", color=COLORES[cluster_nuevo],
               edgecolor="black", linewidth=1.6, zorder=5, label="Estudiante evaluado")
    ax.set_xlabel("PC1 — apoyo y rendimiento ↔ pantalla y sueño")
    ax.set_ylabel("PC2 — actividad y asistencia ↔ presión económica")
    ax.set_title("Ubicación del estudiante en el mapa PCA 2D (15 % de la varianza)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Segmentación de estudiantes", page_icon="🎓", layout="wide")
    pipeline, pca = cargar_modelos()
    features, segmentos, pca_2d, labels, perfiles, rangos = cargar_datos()

    st.title("Segmentación de estudiantes con K-Means")
    st.caption("Proyecto 2 · Clustering con Machine Learning · Modelo final: K-Means (K = 4)")

    with st.sidebar:
        st.header("Cómo funciona")
        st.markdown(
            "1. Ingresa los 16 datos del estudiante **en su escala original**.\n"
            "2. El pipeline exportado imputa, escala y asigna el segmento más cercano.\n"
            "3. El mapa 2D usa PCA sólo para ubicar al estudiante; la asignación se hace en 16D.\n\n"
            "K-Means asigna siempre un segmento, así que no hay casos de ruido (−1). "
            "DBSCAN se descartó porque dejaba al 88 % de estudiantes como atípicos."
        )
        st.markdown(
            "**Importante:** los segmentos son tendencias para priorizar apoyo "
            "(Silhouette = 0.047), no un diagnóstico individual."
        )

    st.subheader("Datos del estudiante")
    valores = {}
    columnas = st.columns(4)
    for i, f in enumerate(features):
        etiqueta, unidad, paso = CAMPOS[f]
        r = rangos.loc[f]
        with columnas[i % 4]:
            valores[f] = st.number_input(
                f"{etiqueta} ({unidad})",
                min_value=float(r["min"]), max_value=float(r["max"]),
                value=float(round(r["mediana"] / paso) * paso), step=paso,
                help=f"Variable `{f}`. Rango observado: {r['min']:g} – {r['max']:g}.",
                key=f,
            )

    if st.button("Asignar segmento", type="primary"):
        registro = pd.DataFrame([valores], columns=features)
        cluster = int(pipeline.predict(registro)[0])
        nombre = segmentos[str(cluster)]
        st.session_state["resultado"] = (registro, cluster, nombre)

    if "resultado" in st.session_state:
        registro, cluster, nombre = st.session_state["resultado"]
        st.divider()
        izq, der = st.columns([1, 1.4])
        with izq:
            st.subheader("Resultado")
            st.success(f"**Segmento {cluster}: {nombre}**")
            st.markdown(f"**Recomendación:** {RECOMENDACION[nombre]}")
            fila = perfiles.set_index("cluster").loc[cluster]
            comp = pd.DataFrame({
                "Estudiante": registro.iloc[0][features].round(2),
                "Prom. segmento": fila[features].round(2),
            })
            comp.index = [CAMPOS[f][0] for f in features]
            st.dataframe(comp, width="stretch")
        with der:
            punto = proyectar_2d(pipeline, pca, registro)
            st.pyplot(mapa_2d(pca_2d, labels, segmentos, punto, cluster))

    with st.expander("Ver los 4 perfiles (promedios en escala original)"):
        tabla = perfiles.copy()
        tabla.insert(1, "segmento", tabla["cluster"].astype(str).map(segmentos))
        tabla["recomendación"] = tabla["segmento"].map(RECOMENDACION)
        st.dataframe(tabla.round(2), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
