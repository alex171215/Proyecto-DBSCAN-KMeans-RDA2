# Paquete de Entregables Oficiales — Proyecto 2: Clustering con Machine Learning
## Inteligencia Artificial · Sexto Semestre · Ingeniería en Sistemas de Información

Este directorio reúne de forma ordenada, autocontenida y reproducible todos los entregables exigidos en el documento **`Proyecto 2 - Clustering con Machine Learning  App Web.pdf`**.

---

## 📁 Estructura del Directorio de Entrega

```text
Entrega_Proyecto_2/
├── 1_Notebook/
│   ├── Proyecto2_Clustering_Master.ipynb    # [ENTREGABLE 1] Notebook completo reproducible
│   └── data/
│       └── student_burnout_dropout_dataset_2_link 1.csv # Dataset oficial de Kaggle
│
├── 2_Informe/
│   ├── informe.pdf                         # [ENTREGABLE 2] Informe técnico formal (≤ 5 páginas)
│   ├── informe.docx                        # Formato editable en Word
│   ├── informe.md                          # Fuente en Markdown
│   └── capturas/                           # Capturas de la app web y diagramas
│
├── 3_Aplicacion_Web/
│   ├── app.py                              # [ENTREGABLE 3] Código fuente de la App (Streamlit)
│   ├── requirements.txt                    # Dependencias exactas
│   ├── models/                             # Pipelines y modelos exportados con joblib
│   │   ├── pipeline_kmeans.joblib          # Pipeline del modelo ganador (imputer + scaler + kmeans)
│   │   ├── pipeline_dbscan.joblib          # Paquete DBSCAN con NearestNeighbors
│   │   ├── pca.joblib
│   │   ├── scaler.joblib
│   │   └── imputer.joblib
│   ├── data/                               # Proyecciones y nombres de features
│   │   ├── pca_2d.csv
│   │   └── feature_names.json
│   └── segments/                           # Perfiles, diccionarios y etiquetas
│       ├── perfiles_kmeans.csv
│       ├── segments_kmeans.json
│       └── labels_kmeans.csv
│
├── 4_Guia_Sustentacion/
│   └── Guia_Defensa_Oral_Proyecto_2.md     # [SUSTENTACIÓN] Guion, preguntas trampa y demo en vivo
│
└── LEEME_ENTREGABLES.md                    # Este documento guía
```

---

## 🚀 1. Cómo Ejecutar el Notebook Maestro (`1_Notebook/`)

El notebook **`Proyecto2_Clustering_Master.ipynb`** contiene de principio a fin todo el proceso exigido en el "Alcance Técnico" de la rúbrica oficial (Puntos 1 al 8):
1. **Contexto y pregunta de negocio** (clustering vs clasificación).
2. **EDA completo** con 3 visualizaciones (distribución de variables, boxplots de Tukey con análisis de impacto en centroides, y matriz de correlación de Pearson).
3. **Preparación de datos** (exclusión de IDs/etiquetas, imputación mediana y escalado con `StandardScaler`).
4. **Reducción de dimensionalidad con PCA** (varianza explicada, loadings de PC1/PC2, proyección 2D y justificación de entrenar en 16D).
5. **Modelado dual obligatorio**:
   - **K-Means:** Método del codo y Silhouette probados de $K=2$ a $9$, justificando la elección de $K=4$.
   - **DBSCAN:** Heurística $\text{MinPts}=32$ ($2D$), gráfico $k$-dist, barrido de $\varepsilon$ y reporte de métricas ($88\%$ ruido, Silhouette sin ruido $0.1338$, explicando la maldición de la dimensionalidad).
6. **Evaluación y comparación**: Tabla formal de métricas, mapa 2D comparativo, gráfico de barras Silhouette vs Ruido y regla de penalización por ruido (>20%) para seleccionar K-Means.
7. **Interpretación de segmentos**: Perfil promedio en escala original (`groupby().mean()`), proporciones binarias, nombres descriptivos de los 4 segmentos y recomendaciones de retención.
8. **Exportación y verificación de inferencia**: Pipelines exportados y simulación de nuevos estudiantes con datos crudos.

### Pasos para ejecutar:
```bash
# 1. Navegar a la carpeta del notebook
cd "Entrega_Proyecto_2/1_Notebook"

# 2. Abrir en Jupyter Notebook o VS Code
jupyter notebook Proyecto2_Clustering_Master.ipynb
# Seleccionar el Kernel de Python y pulsar: Kernel -> Restart & Run All
```
*Todas las celdas se ejecutan sin errores y de forma reproducible.*

---

## 🌐 2. Cómo Ejecutar la Aplicación Web (`3_Aplicacion_Web/`)

La aplicación web interactiva está construida en **Streamlit** y corre completamente en local:
- Permite ingresar las 16 variables del estudiante en su **escala original** mediante controles interactivos.
- Al pulsar *"Asignar Segmento"*, procesa el registro en tiempo real con `pipeline_kmeans.joblib` (aplica el mismo preprocesamiento del entrenamiento sin escalar a mano).
- Muestra el número y nombre del segmento, el diagnóstico, la recomendación institucional y la **proyección PCA 2D** donde se resalta al nuevo estudiante con una estrella distintiva junto a la cohorte universitaria.

### Pasos para ejecutar:
```bash
# 1. Navegar a la carpeta de la aplicación
cd "Entrega_Proyecto_2/3_Aplicacion_Web"

# 2. Instalar dependencias (si no están instaladas)
pip install -r requirements.txt

# 3. Lanzar la aplicación Streamlit
streamlit run app.py
```
*Se abrirá automáticamente en el navegador en `http://localhost:8501`.*

---

## 📄 3. Informe de la Aplicación (`2_Informe/`)

- Archivo: **`informe.pdf`** (máximo ~5 páginas).
- Describe:
  1. Los segmentos encontrados y su interpretación de negocio.
  2. La arquitectura de la solución (entrenamiento e inferencia).
  3. Cómo se integra el modelo y el preprocesamiento mediante Pipelines de Scikit-Learn.
  4. Capturas de pantalla reales del funcionamiento de la app.
  5. Instrucciones de instalación y ejecución local.
  6. Limitaciones técnicas conocidas (dataset transversal sintético, supuestos de distancia euclidiana en 16D, etc.).

---

## 🎤 4. Guía para la Sustentación Oral (`4_Guia_Sustentacion/`)

- Archivo: **`Guia_Defensa_Oral_Proyecto_2.md`**.
- Contiene:
  - Reparto cronometrado de los 5 a 10 minutos de sustentación para los 3 integrantes.
  - Las 10 preguntas más difíciles y trampas que suele hacer el docente y la respuesta exacta para responder con seguridad.
  - Los datos exactos de los 2 registros de prueba para la demo en vivo frente al evaluador (un estudiante en riesgo financiero que cae en Cluster 0 y un estudiante de alto rendimiento que cae en Cluster 2).
