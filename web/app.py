"""
Orienta · Página web para consejeros y tutores (Flask)

Usa el modelo final K-Means (K=4) ya exportado en models/pipeline_kmeans.joblib.
No entrena nada: sólo carga artefactos.

- 8 preguntas obligatorias (las variables que más separan los segmentos).
- 8 preguntas opcionales: si se dejan vacías, el SimpleImputer del pipeline
  las reemplaza por la mediana del dataset.
- El resultado dice a qué servicio derivar al estudiante y con qué confianza.

Ejecutar desde la raíz del proyecto:
    python web/app.py
y abrir http://127.0.0.1:5000
"""
import base64
import io
import json
import math
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, redirect, render_template, request, url_for  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__)

# ---------------------------------------------------------------- artefactos
PIPELINE = joblib.load(ROOT / "models" / "pipeline_kmeans.joblib")
PCA = joblib.load(ROOT / "models" / "pca.joblib")
FEATURES = json.loads((ROOT / "data" / "feature_names.json").read_text(encoding="utf-8"))
NOMBRES = json.loads((ROOT / "segments" / "segments_kmeans.json").read_text(encoding="utf-8"))
PERFILES = pd.read_csv(ROOT / "segments" / "perfiles_kmeans.csv").set_index("cluster")
PCA_2D = pd.read_csv(ROOT / "data" / "pca_2d.csv")
LABELS = pd.read_csv(ROOT / "segments" / "labels_kmeans.csv")["cluster"]
COMPARACION = pd.read_csv(ROOT / "tables" / "comparacion.csv")
_crudo = pd.read_csv(ROOT / "data" / "student_burnout_dropout_dataset_2_link 1.csv")[FEATURES]
MEDIANA = _crudo.median()
MINIMO, MAXIMO = _crudo.min(), _crudo.max()

# ---------------------------------------------------------------- preguntas
# clave: (pregunta, tipo, unidad, paso, obligatoria, etiqueta_baja, etiqueta_alta)
PREGUNTAS = {
    "Year_of_Study": ("¿En qué año de la carrera está?", "numero", "año", 1, True, "", ""),
    "Attendance_Percent": ("Porcentaje de asistencia a clases", "numero", "%", 0.5, True, "", ""),
    "Previous_GPA": ("Promedio del periodo anterior (0 a 10)", "numero", "/10", 0.1, True, "", ""),
    "Backlogs": ("Materias pendientes o reprobadas", "numero", "materias", 1, True, "", ""),
    "Study_Hours_Per_Day": ("Horas de estudio al día", "numero", "h", 0.5, False, "", ""),
    "Age": ("Edad", "numero", "años", 1, False, "", ""),
    "Screen_Time_Hours": ("Horas de pantalla al día (ocio)", "numero", "h", 0.5, True, "", ""),
    "Sleep_Hours": ("Horas de sueño por noche", "numero", "h", 0.5, False, "", ""),
    "Exercise_Freq_Per_Week": ("Días de ejercicio por semana", "numero", "días", 1, False, "", ""),
    "Social_Activity_Score": ("Actividad social", "escala", "", 0.5, False, "Aislado", "Muy activo"),
    "Family_Support_Score": ("Apoyo de la familia", "escala", "", 0.5, True, "Nulo", "Muy fuerte"),
    "Peer_Pressure_Score": ("Presión de compañeros", "escala", "", 0.5, True, "Ninguna", "Muy alta"),
    "Financial_Stress_Score": ("Preocupación por dinero", "escala", "", 0.5, True, "Ninguna", "Muy alta"),
    "Stress_Level": ("Nivel de estrés", "escala", "", 0.5, False, "Bajo", "Muy alto"),
    "Anxiety_Score": ("Ansiedad", "escala", "", 0.5, False, "Baja", "Muy alta"),
    "Motivation_Score": ("Motivación por estudiar", "escala", "", 0.5, False, "Baja", "Muy alta"),
}
PASOS = [
    ("Rendimiento académico", "Datos del registro académico del estudiante.",
     ["Year_of_Study", "Attendance_Percent", "Previous_GPA", "Backlogs", "Study_Hours_Per_Day", "Age"]),
    ("Hábitos", "Lo que el estudiante cuenta sobre su día a día.",
     ["Screen_Time_Hours", "Sleep_Hours", "Exercise_Freq_Per_Week", "Social_Activity_Score"]),
    ("Entorno y bienestar", "Escala de 0 a 10 según la entrevista con el estudiante.",
     ["Family_Support_Score", "Peer_Pressure_Score", "Financial_Stress_Score",
      "Stress_Level", "Anxiety_Score", "Motivation_Score"]),
]
CORTO = {
    "Year_of_Study": "Año de carrera", "Attendance_Percent": "Asistencia", "Previous_GPA": "Promedio",
    "Backlogs": "Materias pendientes", "Study_Hours_Per_Day": "Horas de estudio", "Age": "Edad",
    "Screen_Time_Hours": "Tiempo de pantalla", "Sleep_Hours": "Horas de sueño",
    "Exercise_Freq_Per_Week": "Ejercicio", "Social_Activity_Score": "Actividad social",
    "Family_Support_Score": "Apoyo familiar", "Peer_Pressure_Score": "Presión de pares",
    "Financial_Stress_Score": "Estrés financiero", "Stress_Level": "Estrés",
    "Anxiety_Score": "Ansiedad", "Motivation_Score": "Motivación",
}

# ---------------------------------------------------------------- segmentos
SEGMENTOS = {
    0: {
        "color": "#E5533D", "suave": "#FDECE9", "icono": "📚", "prioridad": "Alta",
        "resumen": "Tiene más materias pendientes y más preocupación económica que el resto.",
        "derivar": ["Bienestar estudiantil: becas y ayuda económica", "Tutorías académicas"],
        "acciones": [
            "Revisar si cumple requisitos para beca o ayuda económica.",
            "Asignar tutorías para las materias pendientes.",
            "Acordar un plan de recuperación con fechas y un seguimiento mensual.",
        ],
    },
    1: {
        "color": "#4CAF4F", "suave": "#E8F5E9", "icono": "🤝", "prioridad": "Baja",
        "resumen": "Tiene buen apoyo familiar, motivación y vida social. Va bien.",
        "derivar": ["Programa de mentoría entre pares (como mentor)"],
        "acciones": [
            "Mantener el seguimiento habitual del periodo.",
            "Invitarlo a ser mentor de compañeros de otros grupos.",
            "Reconocer su participación en actividades de la carrera.",
        ],
    },
    2: {
        "color": "#F2A33A", "suave": "#FEF4E6", "icono": "📱", "prioridad": "Media-alta",
        "resumen": "Falta más a clases y pasa más horas frente a la pantalla.",
        "derivar": ["Coordinación académica: alerta de asistencia",
                    "Taller de hábitos digitales y sueño"],
        "acciones": [
            "Activar la alerta temprana de asistencia con sus docentes.",
            "Inscribirlo en el taller de uso de pantallas y hábitos de sueño.",
            "Volver a revisar la asistencia en cuatro semanas.",
        ],
    },
    3: {
        "color": "#4A6CF7", "suave": "#ECF0FE", "icono": "🎯", "prioridad": "Media",
        "resumen": "Tiene el mejor promedio, pero duerme menos y siente presión económica.",
        "derivar": ["Consejería psicológica preventiva", "Orientación financiera"],
        "acciones": [
            "Reconocer su rendimiento académico.",
            "Ofrecer una cita de consejería preventiva por estrés.",
            "Informarle sobre orientación financiera y ayudas disponibles.",
        ],
    },
}
for k, s in SEGMENTOS.items():
    s["id"] = k
    s["nombre"] = NOMBRES[str(k)]
    s["n"] = int(PERFILES.loc[k, "n"])
    s["pct"] = round(100 * s["n"] / len(LABELS))


def leer_formulario(form):
    """Convierte el formulario en un registro de 16 columnas; vacío -> NaN (lo imputa el pipeline)."""
    valores, faltan = {}, []
    for f in FEATURES:
        crudo = form.get(f, "").strip().replace(",", ".")
        if crudo == "":
            if PREGUNTAS[f][4]:
                faltan.append(f)
            valores[f] = np.nan
            continue
        try:
            v = float(crudo)
        except ValueError:
            faltan.append(f)
            valores[f] = np.nan
            continue
        valores[f] = float(min(max(v, MINIMO[f]), MAXIMO[f]))
    return pd.DataFrame([valores], columns=FEATURES), faltan


def confianza(dist):
    """Razón entre el centroide más cercano y el segundo (1 = justo en la frontera).
    Umbrales = cuartiles 25 % y 75 % de esa razón en los 800 estudiantes."""
    d = np.sort(dist)
    r = d[0] / d[1]
    if r < 0.863:
        return "Alta", "El perfil del estudiante coincide claramente con este grupo.", 3
    if r < 0.957:
        return "Media", "Coincide con este grupo, aunque comparte rasgos con otro.", 2
    return "Baja", "Está en la frontera entre dos grupos: confirme en la entrevista.", 1


def con_unidad(f, v):
    unidad = PREGUNTAS[f][2]
    if not unidad:
        return f"{v:g}/10"
    if unidad == "%":
        return f"{v:g} %"
    if unidad.startswith("/"):
        return f"{v:g}{unidad}"
    if unidad == "materias" and v == 1:
        unidad = "materia"
    return f"{v:g} {unidad}"


def senales(registro):
    """Las 3 respuestas que más se alejan de lo típico (en desviaciones estándar)."""
    z = PIPELINE[:-1].transform(registro)[0]
    salida = []
    for i in np.argsort(-np.abs(z)):
        f = FEATURES[i]
        if math.isnan(registro.iloc[0][f]) or abs(z[i]) < 0.5:
            continue
        salida.append({
            "nombre": CORTO[f],
            "valor": con_unidad(f, registro.iloc[0][f]),
            "tipico": con_unidad(f, MEDIANA[f]),
            "sentido": "más alto" if z[i] > 0 else "más bajo",
        })
        if len(salida) == 3:
            break
    return salida


def mapa(registro, cluster):
    escalado = PIPELINE[:-1].transform(registro)
    punto = PCA.transform(escalado)[0, :2]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
    for k in sorted(SEGMENTOS):
        m = LABELS == k
        ax.scatter(PCA_2D.loc[m, "PC1"], PCA_2D.loc[m, "PC2"], s=10, alpha=0.35,
                   color=SEGMENTOS[k]["color"], label=SEGMENTOS[k]["nombre"])
    ax.scatter(*punto, s=360, marker="*", color=SEGMENTOS[cluster]["color"],
               edgecolor="#263238", linewidth=1.5, zorder=5, label="Este estudiante")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.set_xlabel("Componente 1", color="#717171")
    ax.set_ylabel("Componente 2", color="#717171")
    ax.tick_params(colors="#717171", labelsize=8)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- rutas
@app.context_processor
def comunes():
    return {"segmentos": SEGMENTOS}


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/evaluar", methods=["GET", "POST"])
def evaluar():
    if request.method == "GET":
        return render_template("evaluar.html", pasos=PASOS, preguntas=PREGUNTAS,
                               mediana=MEDIANA, minimo=MINIMO, maximo=MAXIMO, faltan=[], previo={})
    registro, faltan = leer_formulario(request.form)
    if faltan:
        return render_template("evaluar.html", pasos=PASOS, preguntas=PREGUNTAS,
                               mediana=MEDIANA, minimo=MINIMO, maximo=MAXIMO,
                               faltan=faltan, previo=request.form), 400
    cluster = int(PIPELINE.predict(registro)[0])
    dist = PIPELINE[-1].transform(PIPELINE[:-1].transform(registro))[0]
    nivel, texto, barras = confianza(dist)
    respondidas = int(registro.notna().sum(axis=1).iloc[0])
    tabla = [{
        "nombre": CORTO[f],
        "estudiante": "—" if math.isnan(registro.iloc[0][f]) else f"{registro.iloc[0][f]:g}",
        "grupo": f"{PERFILES.loc[cluster, f]:.1f}",
        "general": f"{MEDIANA[f]:g}",
    } for f in FEATURES]
    return render_template(
        "resultado.html", seg=SEGMENTOS[cluster], nivel=nivel, texto_conf=texto, barras=barras,
        senales=senales(registro), tabla=tabla, mapa=mapa(registro, cluster),
        respondidas=respondidas, alias=request.form.get("alias", "").strip()[:60],
    )


@app.route("/tecnico")
def tecnico():
    return render_template("tecnico.html", comparacion=COMPARACION.to_dict("records"))


@app.route("/figs/<nombre>")
def figura(nombre):
    from flask import send_from_directory
    return send_from_directory(ROOT / "figs", nombre)


@app.errorhandler(404)
def no_encontrado(_):
    return redirect(url_for("inicio"))


if __name__ == "__main__":
    # Local: http://127.0.0.1:5000 · En Render el puerto llega en la variable PORT
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
