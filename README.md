# Proyecto 2: Clustering con Machine Learning y Aplicación Web

**Asignatura:** Inteligencia Artificial — Sexto Semestre, Ingeniería en Sistemas de Información  
**Problema:** Segmentación de estudiantes basada en perfil académico, socioemocional y socioeconómico para prevención de deserción y atención al burnout.

---

## 📌 Fuente Única de Verdad
Antes de iniciar cualquier desarrollo, consulta obligatoriamente el archivo:
👉 [documento_maestro_proyecto_2.md](documento_maestro_proyecto_2.md)

Allí se detallan el contrato técnico estricto, la selección de variables, las restricciones de no supervisión y el formato exacto de todos los artefactos.

---

## 🚀 Paso Inicial Obligatorio: Ejecutar Base Fundacional
Todos los integrantes deben contar con las dependencias instaladas y ejecutar el script base inmutable:

```bash
python 00_foundation.py
```

Este script preprocesa `data/student_burnout_dropout_dataset_2_link 1.csv`, imputa, escala con `StandardScaler`, entrena el `PCA` base y genera los archivos base compartidos en `data/` y `models/`.

---

## 👥 División de Paquetes de Trabajo

| Paquete | Responsable | Rol Principal | Notebook / Código Principal |
| :--- | :--- | :--- | :--- |
| **Paquete A** | Persona 1 | Modelado K-Means, selección de $K$ (Codo y Silhouette), loadings PCA y perfilamiento | `notebooks/01_kmeans.ipynb` |
| **Paquete B** | Persona 2 | Modelado DBSCAN, selección de $\text{MinPts}$ y $\varepsilon$ (k-dist), reporte de ruido y pipeline con NN | `notebooks/02_dbscan.ipynb` |
| **Paquete C** | Persona 3 | Aplicación Web interactiva (Streamlit), comparación técnica de modelos, notebook consolidado e informe | `app/app.py`<br>`notebooks/03_final.ipynb`<br>`informe/` |

---

## 📁 Estructura del Proyecto

```text
├── 00_foundation.py           # Script base inmutable (ejecutar primero)
├── documento_maestro_proyecto_2.md # Guía maestra y contrato técnico
├── README.md
├── .gitignore
├── data/
│   └── student_burnout_dropout_dataset_2_link 1.csv # Dataset crudo
├── notebooks/
│   ├── 01_kmeans.ipynb        # Paquete A
│   ├── 02_dbscan.ipynb        # Paquete B
│   └── 03_final.ipynb         # Paquete C (Consolidado)
├── models/                    # Modelos y pipelines serializados (.joblib)
├── figs/                      # Gráficas generadas (codo, k-dist, mapas PCA, comparaciones)
├── tables/                    # Métricas y tablas comparativas (.csv)
├── segments/                  # Perfiles, etiquetas y diccionarios de segmentos (.csv / .json)
├── app/
│   ├── app.py                 # Aplicación Streamlit para inferencia y visualización 2D
│   └── requirements.txt       # Dependencias exactas
├── web/                       # Página pública "Orienta" para consejeros (Flask)
│   ├── app.py                 # Rutas: inicio, evaluar, resultado, detalle técnico
│   ├── templates/  static/    # HTML, CSS y JS
│   └── requirements.txt
└── informe/
    ├── informe.md             # Borrador del informe técnico (≤ 5 páginas)
    └── decision_modelo.md     # Justificación del modelo ganador
```

## 🌐 Página web para consejeros (Orienta)

Versión para el usuario final (tutor o consejero): 8 preguntas obligatorias, resultado con el servicio al que derivar, confianza y PDF. Usa el mismo `models/pipeline_kmeans.joblib`; no entrena nada.

```bash
pip install -r web/requirements.txt
python web/app.py
# abrir http://127.0.0.1:5000
```

La versión técnica en Streamlit sigue disponible con `python -m streamlit run app/app.py`.
