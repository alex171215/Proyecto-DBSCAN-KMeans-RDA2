# DOCUMENTO MAESTRO: PROYECTO 2 - CLUSTERING ML

Este documento es la fuente única de verdad para el desarrollo del Proyecto 2 de Aprendizaje No Supervisado. Define de forma estricta los parámetros técnicos, la división del trabajo y las reglas de integración para un equipo de tres personas.

### 1. Contexto y objetivo del proyecto
El objetivo del proyecto es construir un sistema de aprendizaje automático no supervisado capaz de segmentar estudiantes en distintos grupos basándose en su perfil académico, socioemocional y socioeconómico. Esta segmentación tiene un fin de gestión académica: asignar a cada grupo resultante una recomendación concreta orientada a prevenir la deserción estudiantil, brindar apoyo psicológico o reconocer el desempeño sobresaliente.

Se debe tener absoluta claridad técnica en que este es un problema de **clustering (no supervisado)**. Bajo ningún concepto se resolverá como un problema de clasificación. No se intentará predecir de forma supervisada si un estudiante va a desertar o si tiene burnout; el algoritmo debe descubrir agrupaciones latentes puramente basándose en la distancia de sus variables predictoras. 

### 2. Dataset seleccionado
- **Nombre:** `student_burnout_dropout_dataset_2_link 1.csv`
- **Origen:** Kaggle, validar con el docente.
- **Tamaño:** 800 filas y 25 columnas.
- **Justificación:** Posee variables multidimensionales (académicas, psicométricas y de hábitos) idóneas para aplicar algoritmos de agrupamiento espaciales (DBSCAN) y particionales (K-Means), simulando un escenario real de retención universitaria. 
- **Dataset descartado:** Se descartó el dataset `data (1).csv` (UCI academic success) debido a que su estructura prioriza variables categóricas curriculares de una institución específica, mientras que el dataset seleccionado ofrece un balance superior de variables numéricas continuas psicométricas, ajustándose mejor al requisito de utilizar al menos 5 variables numéricas y al enfoque de bienestar integral (burnout).

### 3. Contrato técnico compartido (obligatorio)

#### Variables Numéricas (Matriz X)
Tanto K-Means como DBSCAN se entrenarán utilizando única y exclusivamente las siguientes 16 variables, estrictamente en este orden:
1. `Age`
2. `Year_of_Study`
3. `Attendance_Percent`
4. `Study_Hours_Per_Day`
5. `Previous_GPA`
6. `Backlogs`
7. `Sleep_Hours`
8. `Screen_Time_Hours`
9. `Exercise_Freq_Per_Week`
10. `Social_Activity_Score`
11. `Financial_Stress_Score`
12. `Family_Support_Score`
13. `Stress_Level`
14. `Anxiety_Score`
15. `Motivation_Score`
16. `Peer_Pressure_Score`

#### Variables Excluidas
| Variable | Motivo de Exclusión | Uso Permitido Posteriori |
| :--- | :--- | :--- |
| `Student_ID` | Identificador único | No se usa en ningún paso del flujo. Solo referencia en el CSV original. |
| `Gender`, `Department`, `Residence_Type`, `Family_Income_Bracket` | Categóricas de texto | Solo descripción demográfica de perfiles. |
| `Part_Time_Job`, `Counseling_Access` | Binarias (0/1) | Interpretación de perfiles (calcular proporciones por cluster). |
| `Burnout_Level`, `Dropout_Risk` | **ETIQUETAS DE CLASE** | Totalmente prohibidas en entrenamiento, PCA y agrupamiento. |

#### Decisiones de Preprocesamiento
- **Imputación:** `SimpleImputer(strategy="median")`
- **Escalado:** `StandardScaler()`
- **PCA:** `PCA(n_components=None, random_state=42)` (se calculan todas las componentes para la varianza explicada, pero se guardan y usan solo las 2 primeras para la proyección 2D visual).
- **Entrenamiento:** Ambos modelos se optimizan y ajustan sobre el espacio de dimensionalidad completa (16D) ya escalado.
- **Semilla:** `RANDOM_STATE = 42` para todo elemento estocástico.

#### Estructura Exacta de Carpetas
```text
/
├── data/
│   ├── student_burnout_dropout_dataset_2_link 1.csv
│   ├── X_scaled.csv
│   ├── pca_2d.csv
│   ├── pca_varianza.csv
│   └── feature_names.json
├── notebooks/
│   ├── 01_kmeans.ipynb
│   ├── 02_dbscan.ipynb
│   └── 03_final.ipynb
├── models/
│   ├── imputer.joblib
│   ├── scaler.joblib
│   ├── pca.joblib
│   ├── pipeline_kmeans.joblib
│   └── pipeline_dbscan.joblib
├── figs/
│   ├── codo_kmeans.png
│   ├── silhouette_kmeans.png
│   ├── pca_kmeans.png
│   ├── kdist_dbscan.png
│   ├── pca_dbscan.png
│   └── comparacion.png
├── tables/
│   ├── metrics_kmeans.csv
│   ├── metrics_dbscan.csv
│   └── comparacion.csv
├── segments/
│   ├── perfiles_kmeans.csv
│   ├── labels_kmeans.csv
│   ├── segments_kmeans.json
│   ├── perfiles_dbscan.csv
│   ├── labels_dbscan.csv
│   └── segments_dbscan.json
├── app/
│   ├── app.py
│   └── requirements.txt
└── informe/
    ├── informe.pdf
    ├── informe.md
    └── decision_modelo.md
```

#### 3.7. Formato de los artefactos

**`tables/metrics_kmeans.csv` y `tables/metrics_dbscan.csv`** — una fila con estas columnas exactas:

| columna | tipo | ejemplo K-Means | ejemplo DBSCAN |
|---|---|---|---|
| `model` | str | `kmeans` | `dbscan` |
| `n_clusters` | int | `4` | `3` |
| `silhouette` | float | `0.42` | `0.38` |
| `noise_pct` | float | `0.0` | `0.22` |
| `inertia` | float | `12345.0` | `NaN` |
| `eps` | float | `NaN` | `0.85` |
| `min_samples` | int | `NaN` | `32` |

**`segments/segments_kmeans.json` y `segments/segments_dbscan.json`** — diccionario `etiqueta_str → nombre descriptivo`. Las claves son strings (JSON no admite enteros como claves). Ejemplo:

```json
{"0": "Alto riesgo", "1": "Riesgo moderado", "2": "Perfil estable", "3": "Destacado", "-1": "Atípico"}
```

**`segments/perfiles_kmeans.csv` y `segments/perfiles_dbscan.csv`** — una fila por cluster (incluir -1 en DBSCAN). Columnas exactas:
- `cluster` (int)
- `n` (int, número de estudiantes en el cluster)
- Las 16 columnas de variables originales con sus nombres exactos (`Age`, `Year_of_Study`, ..., `Peer_Pressure_Score`) con el promedio en escala original
- `prop_part_time_job` (float, proporción de 1 en `Part_Time_Job`)
- `prop_counseling_access` (float, proporción de 1 en `Counseling_Access`)

**`segments/labels_kmeans.csv` y `segments/labels_dbscan.csv`** — una sola columna llamada `cluster` con la etiqueta asignada a cada fila, en el mismo orden en que aparecen en el CSV original (para alinearlas con `pca_2d.csv`).

**`tables/comparacion.csv`** — una fila por modelo (2 filas en total). Columnas exactas:
- `model`
- `n_clusters`
- `silhouette`
- `noise_pct`
- `cobertura_estudiantes` (float, % de estudiantes asignados a algún cluster no-ruido)
- `utilidad_negocio` (str, evaluación cualitativa breve)
- `seleccionado` (str, "si" o "no")

**`models/pipeline_kmeans.joblib`** — el Pipeline de sklearn directamente (imputer + scaler + kmeans).

**`models/pipeline_dbscan.joblib`** — diccionario con las llaves:
- `pipeline` — el Pipeline de sklearn (imputer + scaler + dbscan).
- `nn_model` — NearestNeighbors ajustado sobre los core points.
- `core_labels` — array con la etiqueta de cada core point.
- `eps` — float, valor de eps usado.

#### Código Base: `00_foundation.py`
Todo participante debe ejecutar este script localmente para generar la base inmutable.

```python
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RANDOM_STATE = 42
DATA_PATH = "data/student_burnout_dropout_dataset_2_link 1.csv"

FEATURES = [
    "Age", "Year_of_Study", "Attendance_Percent", "Study_Hours_Per_Day",
    "Previous_GPA", "Backlogs", "Sleep_Hours", "Screen_Time_Hours",
    "Exercise_Freq_Per_Week", "Social_Activity_Score", "Financial_Stress_Score",
    "Family_Support_Score", "Stress_Level", "Anxiety_Score",
    "Motivation_Score", "Peer_Pressure_Score"
]

def run_foundation():
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X_raw = df[FEATURES].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=None, random_state=RANDOM_STATE)
    pca_full = pca.fit_transform(X_scaled)
    pca_2d = pca_full[:, :2]

    joblib.dump(imputer, "models/imputer.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(pca, "models/pca.joblib")

    pd.DataFrame(X_scaled, columns=FEATURES).to_csv("data/X_scaled.csv", index=False)
    pd.DataFrame(pca_2d, columns=["PC1", "PC2"]).to_csv("data/pca_2d.csv", index=False)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_variance)
    components = [f"PC{i+1}" for i in range(len(explained_variance))]
    
    df_var = pd.DataFrame({
        "component": components,
        "explained_variance": explained_variance,
        "cumulative": cumulative
    })
    df_var.to_csv("data/pca_varianza.csv", index=False)
    
    with open("data/feature_names.json", "w") as f:
        json.dump(FEATURES, f)

if __name__ == "__main__":
    run_foundation()
```

#### Esqueleto Pipeline K-Means
```python
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

pipeline_kmeans = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=3, random_state=42, n_init=10))
])

# Cargar X_raw desde el CSV original
import pandas as pd
df = pd.read_csv("data/student_burnout_dropout_dataset_2_link 1.csv")
X_raw = df[[
    "Age", "Year_of_Study", "Attendance_Percent", "Study_Hours_Per_Day",
    "Previous_GPA", "Backlogs", "Sleep_Hours", "Screen_Time_Hours",
    "Exercise_Freq_Per_Week", "Social_Activity_Score", 
    "Financial_Stress_Score", "Family_Support_Score", "Stress_Level",
    "Anxiety_Score", "Motivation_Score", "Peer_Pressure_Score"
]].copy()
pipeline_kmeans.fit(X_raw)
joblib.dump(pipeline_kmeans, 'models/pipeline_kmeans.joblib')
```

#### Esqueleto Pipeline DBSCAN + Asignación Completa
```python
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# El pipeline solo preprocesa y entrena DBSCAN
pipeline_dbscan = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('dbscan', DBSCAN(eps=0.5, min_samples=32))
])

# Cargar X_raw desde el CSV original
import pandas as pd
df = pd.read_csv("data/student_burnout_dropout_dataset_2_link 1.csv")
X_raw = df[[
    "Age", "Year_of_Study", "Attendance_Percent", "Study_Hours_Per_Day",
    "Previous_GPA", "Backlogs", "Sleep_Hours", "Screen_Time_Hours",
    "Exercise_Freq_Per_Week", "Social_Activity_Score", 
    "Financial_Stress_Score", "Family_Support_Score", "Stress_Level",
    "Anxiety_Score", "Motivation_Score", "Peer_Pressure_Score"
]].copy()
pipeline_dbscan.fit(X_raw)

# Extraer core points y sus etiquetas
X_core_scaled = pipeline_dbscan.named_steps['scaler'].transform(
    pipeline_dbscan.named_steps['imputer'].transform(X_raw)
)[pipeline_dbscan.named_steps['dbscan'].core_sample_indices_]

core_labels = pipeline_dbscan.named_steps['dbscan'].labels_[
    pipeline_dbscan.named_steps['dbscan'].core_sample_indices_
]

nn_model = NearestNeighbors(n_neighbors=1).fit(X_core_scaled)

paquete_dbscan = {
    'pipeline': pipeline_dbscan,
    'nn_model': nn_model,
    'core_labels': core_labels,
    'eps': 0.5
}
joblib.dump(paquete_dbscan, 'models/pipeline_dbscan.joblib')

# Algoritmo de asignación para nuevos registros (uso en App Web)
def asignar_cluster_dbscan(X_new_raw, paquete):
    X_new_scaled = paquete['pipeline'].named_steps['scaler'].transform(
        paquete['pipeline'].named_steps['imputer'].transform(X_new_raw)
    )
    distancias, indices = paquete['nn_model'].kneighbors(X_new_scaled)
    
    predicciones = []
    for dist, idx in zip(distancias, indices):
        if dist[0] <= paquete['eps']:
            predicciones.append(paquete['core_labels'][idx[0]])
        else:
            predicciones.append(-1) # Ruido
    return predicciones
```

### 4. División del trabajo en 3 paquetes

#### Paquete A — Persona 1 (K-Means)
- **Objetivo:** Desarrollar todo el modelado particional usando K-Means, seleccionar el hiperparámetro $K$ y generar el perfilamiento final.
- **Checklist obligatoria de EDA:**
  - [ ] Análisis de correlaciones entre las 16 variables numéricas.
  - [ ] Detección de outliers y evaluación de su impacto en el cálculo de centroides.
  - [ ] Mínimo 3 visualizaciones relevantes (distribuciones, boxplots, etc.).
  - [ ] Reporte de la varianza explicada por componente y varianza acumulada (desde `pca_varianza.csv`).
  - [ ] Análisis de loadings de las dos primeras componentes: identificar qué variables pesan más en PC1 y PC2 y qué significa en términos de negocio (ej. "PC1 captura riesgo académico-emocional").
- **Alcance detallado:** El notebook debe incluir carga de datos, demostración del método del codo y Silhouette sobre 16D, justificación de $K$, entrenamiento final, visualización 2D (usando el PCA base), extracción de perfiles promediados de variables originales y serialización del Pipeline.
- **Artefactos:**
  | Archivo | Descripción |
  | :--- | :--- |
  | `notebooks/01_kmeans.ipynb` | Código comentado y ejecutable. |
  | `models/pipeline_kmeans.joblib` | Pipeline completo entrenado (100% datos). |
  | `segments/perfiles_kmeans.csv` | Promedios numéricos y descriptores por cluster. |
  | `segments/labels_kmeans.csv` | Etiquetas asignadas a cada registro original. |
  | `segments/segments_kmeans.json` | Diccionario con reglas o perfiles interpretados. |
  | `tables/metrics_kmeans.csv` | Métricas finales (Inercia, Silhouette). |
  | `figs/codo_kmeans.png` | Gráfico del método del codo. |
  | `figs/silhouette_kmeans.png` | Curva de coeficientes Silhouette. |
  | `figs/pca_kmeans.png` | Dispersión 2D de clusters K-Means. |
- **Instrucciones IA:** No integrar variables objetivo. Ejecutar iteraciones paramétricas con `random_state=42`. Confiar en las salidas de `00_foundation.py`.

#### Paquete B — Persona 2 (DBSCAN)
- **Objetivo:** Desarrollar el agrupamiento basado en densidad descubriendo ruido e inconsistencias espaciales en los datos de estudiantes.
- **Checklist obligatoria de EDA:**
  - [ ] Análisis de correlaciones entre las 16 variables numéricas.
  - [ ] Detección de outliers (identificación de densidad).
  - [ ] Mínimo 3 visualizaciones relevantes (distribuciones, boxplots, etc.).
  - [ ] Reporte de la varianza explicada por componente y varianza acumulada (desde `pca_varianza.csv`).
  - [ ] Análisis de loadings de las dos primeras componentes: identificar qué variables pesan más en PC1 y PC2 y qué significa en términos de negocio (ej. "PC1 captura riesgo académico-emocional").
- **Alcance detallado:** El notebook implementará la regla $\text{MinPts} \ge 2D$, trazará el *k-dist graph* para encontrar $\varepsilon$, reportará % de ruido y Silhouette (sin ruido). Renderizará resultados sobre las componentes 2D, documentará medias de core points y exportará el paquete híbrido de modelo + `NearestNeighbors`.
- **Artefactos:**
  | Archivo | Descripción |
  | :--- | :--- |
  | `notebooks/02_dbscan.ipynb` | Código comentado y ejecutable. |
  | `models/pipeline_dbscan.joblib` | Diccionario con Pipeline, NN, etiquetas y `eps`. |
  | `segments/perfiles_dbscan.csv` | Promedios numéricos (excluyendo ruido). |
  | `segments/labels_dbscan.csv` | Etiquetas y ruido (-1) de cada registro original. |
  | `segments/segments_dbscan.json` | Diccionario con descripción de densidades. |
  | `tables/metrics_dbscan.csv` | Métricas finales (% ruido, Silhouette sin ruido). |
  | `figs/kdist_dbscan.png` | Gráfico de distancias al k-ésimo vecino. |
  | `figs/pca_dbscan.png` | Dispersión 2D de clusters DBSCAN y ruido. |
- **Instrucciones IA:** Calcular Silhouette excluyendo `labels_ == -1`. Extraer centroides basándose únicamente en puntos que no son ruido para no sesgar las interpretaciones.

#### Paquete C — Persona 3 (App + Informe)
- **Objetivo:** Orquestar el despliegue del mejor algoritmo encontrado, consolidar la narrativa técnica institucional y unificar los notebooks en un entregable definitivo.
- **Alcance detallado:** 
  - Construir aplicación web en Streamlit que reciba los 16 inputs originales crudos, aplique el modelo exportado y grafique la asignación en el espacio 2D interactivo.
  - **Explicación de la proyección 2D en la app:** Para el gráfico 2D, la app debe cargar `models/pca.joblib` por separado y proyectar tanto el dataset de entrenamiento (leyendo `data/pca_2d.csv`) como el registro nuevo (16D escalado → PCA). El pipeline entrenado NO incluye PCA; el PCA se aplica solo para visualización.
  - **Manejo explícito de ruido (-1) en la app:** Si el modelo final es DBSCAN, la app debe mostrar explícitamente cuando el registro se clasifica como ruido / atípico (-1), con:
    - Mensaje diferenciado: "Registro clasificado como atípico; no encaja en ningún perfil conocido."
    - Recomendación específica: "Requiere revisión manual por consejería."
    - En el gráfico 2D, el ruido se dibuja en gris y el nuevo registro atípico se resalta con un borde distinto.
  - **Comparación y decisión del modelo final:**
    - Leer `tables/metrics_kmeans.csv` y `tables/metrics_dbscan.csv`.
    - Construir `tables/comparacion.csv` con las columnas exactas definidas en la sección 3.7.
    - Evaluar con al menos 2 criterios: Silhouette (mayor es mejor), Cobertura de estudiantes (% asignado a clusters no-ruido; DBSCAN pierde cobertura por el ruido) y Utilidad de negocio (¿los clusters se traducen en acciones concretas?).
    - Regla explícita: si DBSCAN marca más del 20 % como ruido, se penaliza su selección porque deja sin acción a un quinto de los estudiantes.
    - Documentar la decisión en `informe/decision_modelo.md` con: métricas comparadas, criterio de elección, modelo elegido y justificación en prosa (mínimo 3 párrafos).
    - Generar `figs/comparacion.png` con dos subgráficos de barras: Silhouette vs modelo y % de ruido vs modelo.
- **Artefactos:**
  | Archivo | Descripción |
  | :--- | :--- |
  | `app/app.py` | Código fuente web local (Streamlit). |
  | `app/requirements.txt` | Dependencias del entorno. Persona 3 debe fijar las versiones realmente instaladas en su entorno (ej. `streamlit`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `matplotlib`), no versiones de ejemplo. |
  | `informe/informe.pdf` | Documento de ≤ 5 páginas con la síntesis del proyecto. |
  | `informe/informe.md` | Fuente Markdown del informe (el PDF se genera desde aquí). |
  | `informe/decision_modelo.md` | Justificación escrita de la elección del modelo final. |
  | `figs/comparacion.png` | Gráfico comparativo de Silhouette y % de ruido. |
  | `notebooks/03_final.ipynb` | Notebook consolidado que ejecuta de punta a punta: contexto, EDA, preparación, PCA, K-Means, DBSCAN, comparación, decisión del modelo final, interpretación de segmentos y exportación. |
  | `tables/comparacion.csv` | Cuadro resumen de hiperparámetros y Silhouette, con las columnas exactas definidas en la sección 3.7. |
- **Instrucciones IA:** Emplear variables estáticas ("mock") al principio para la interfaz, para luego integrarse nativamente con los `.joblib` cuando A y B finalicen. No entrenar modelos dentro de `app.py`.

### 5. Entregables por persona

| Persona | Paquete | Archivos Principales de Entrega | Depende de |
| :--- | :--- | :--- | :--- |
| Persona 1 | K-Means | `01_kmeans.ipynb`, artefactos en `/models`, `/figs`, `/tables` y `/segments` | `00_foundation.py` |
| Persona 2 | DBSCAN | `02_dbscan.ipynb`, artefactos en `/models`, `/figs`, `/tables` y `/segments` | `00_foundation.py` |
| Persona 3 | Producción | `app.py`, `app/requirements.txt`, `informe.pdf`, `comparacion.csv`, `03_final.ipynb`, `informe/informe.md`, `informe/decision_modelo.md`, `figs/comparacion.png` | Paquetes A y B finalizados |

### 6. Mapeo con la rúbrica

| Criterio | Puntos | Responsable Principal | Verificación Concreta |
| :--- | :--- | :--- | :--- |
| 1. EDA, prep, PCA | 2.0 | Personas 1 y 2 | Imputación, escalado y PCA con varianza explicada por componente (16) y análisis de loadings de PC1 y PC2. |
| 2. Selección de Hiperparámetros | 2.5 | Personas 1 y 2 | Codo + Silhouette probados en K-Means; $k$-dist y $\text{MinPts}$ en DBSCAN. |
| 3. Evaluación e Interpretación | 2.5 | Todos | % de ruido, perfiles legibles, recomendación por cluster (ej. "beca", "ayuda"). |
| 4. App funcional | 2.5 | Persona 3 | `app.py` predice con datos originales pasando por el mismo Scaler, muestra mapa 2D. |
| 5. Documentación | 1.5 | Personas 1, 2 y 3 | Notebooks narrados (`01_kmeans.ipynb`, `02_dbscan.ipynb`, `03_final.ipynb`), código sin fallos al ejecutarse de inicio a fin. |
| 6. Informe / Sustentación | 1.5 | Persona 3 / Todos | PDF estructurado y dominio conceptual por cada integrante (sin delegación). |

### 7. Reglas para agentes de IA
Todo agente artificial asignado a la ejecución de uno de estos paquetes se someterá estrictamente a las siguientes normativas:
1. **Inmutabilidad Fundacional:** No modificar `00_foundation.py` salvo instrucción expresa del usuario.
2. **Prohibición de Etiquetas:** Bajo ninguna circunstancia introducir las variables `Burnout_Level` ni `Dropout_Risk` en el preprocesamiento, escalado o entrenamiento espacial.
3. **Reproducibilidad:** Respetar la semilla paramétrica `RANDOM_STATE = 42`.
4. **Validación Autónoma:** Los notebooks (`.ipynb`) deben diseñarse para ser ejecutados de la primera a la última celda sin requerir intervención de consola.
5. **Decisiones Técnicas:** Documentar en texto plano (Markdown) cualquier decisión técnica no cubierta explícitamente, pero sin violar las normas actuales.
6. **Veracidad:** No inventar ni alucinar valores de métricas (Silhouette, inercia) sin haber ejecutado computacionalmente el cálculo.
7. **Consistencia de Artefactos:** Verificar que la exportación hacia los directorios `/models`, `/figs`, `/tables` y `/segments` coincida gramaticalmente con las rutas estipuladas.
8. **Estándar de Narrativa Pedagógica y Argumentario de Defensa (Estilo PhishGuard - Obligatorio):**
   Los notebooks (`01_kmeans.ipynb`, `02_dbscan.ipynb`, `03_final.ipynb`) no son simples cuadernos de código; son el material de estudio y el guion con el que los estudiantes defenderán individualmente el proyecto en la sustentación oral (5-10 min). Todo agente de IA que genere o edite código debe aplicar un estándar de narrativa pedagógica celda por celda:
   - **Antes de cada celda de código relevante (Propósito y Fundamento):** Una celda de Markdown explicando qué se va a hacer, cuál es el fundamento matemático o algorítmico (e.g. distancia euclidiana, cálculo de inercia intra-cluster, radio de vecindad $\varepsilon$ y densidad $\text{MinPts}$, varianza explicada en PCA) y por qué ese paso es necesario.
   - **Después de cada celda de código con salida numérica o gráfica (Interpretación y Argumentario de Defensa):** Una celda de Markdown con el bloque **`Interpretación y Argumento para Defensa:`** que responda claramente:
     1. *¿Qué significa este resultado en el dominio del negocio/estudiantes?* (e.g., qué revela la correlación, qué indica el codo o el salto en $k$-dist, cuáles son las características predominantes del cluster).
     2. *¿Qué decisión técnica se toma a partir de aquí?* (e.g., por qué se elige $K=3$ o $K=4$, por qué se descarta un $\varepsilon$ con >20% de ruido).
     3. *¿Cómo defenderlo ante el docente? (Pregunta típica de examen):* Una explicación breve y directa pensada para que el estudiante la lea y sepa contestar con seguridad si el profesor pregunta "¿Por qué usó StandardScaler y no MinMaxScaler?", "¿Por qué DBSCAN marcó esos puntos como ruido?", "¿Qué representan PC1 y PC2 en los loadings?".

### 8. Cómo se garantiza la independencia
La estructura de este documento elimina los cuellos de botella mediante tres directrices:
- La persona 1 y la persona 2 no comparten celdas de código. Ambos actúan sobre el output consolidado del script `00_foundation.py`, asegurando que compiten bajo idénticas transformaciones PCA y escalados.
- La persona 3 puede construir el cascarón de `app.py` de inmediato inicializando un modelo K-Means *mock* e inyectando luego los `.joblib` reales en la semana 3.
- El notebook `03_final.ipynb` es responsabilidad de la Persona 3 en la semana 4, con revisión de los otros dos integrantes, unificando de punta a punta los códigos y visualizaciones.

### 9. Timeline sugerido (Calculado desde 29 Oct. 2026)

| Fase | Semana (Finaliza) | Hitos Obligatorios |
| :--- | :--- | :--- |
| **Semana 1** | 01 Noviembre | EDA paralelo (correlaciones, outliers, loadings), ejecución de `00_foundation.py`, validación de X_scaled. |
| **Semana 2** | 06 Noviembre | Iteración de hiperparámetros (Codo/K-dist). Selección de modelos, generación de `.joblib` y métricas. |
| **Semana 3** | 11 Noviembre | Extracción de centroides promediados. `app.py` integra el algoritmo de asignación completo. |
| **Semana 4** | 15 Noviembre (Entrega) | Redacción de `informe.pdf`, consolidación en `03_final.ipynb`, ensayo de demo interactiva. |

### 10. Checklist final de entrega

**Para Notebooks (K-Means y DBSCAN):**
- [ ] Checklist obligatoria de EDA cumplida (3 visualizaciones, varianza PCA, loadings, correlaciones, outliers).
- [ ] **Estándar pedagógico PhishGuard cumplido:** celdas de Markdown antes y después de cada bloque clave con propósito teórico, interpretación de métricas/gráficos y argumentos de defensa oral.
- [ ] Ejecución de `Restart & Run All` exitosa.
- [ ] Pipeline exportado utilizando `joblib` junto a sus etiquetas.
- [ ] Nombres descriptivos de segmentos generados usando `groupby().mean()`.
- [ ] Gráfico PCA 2D renderizado con los colores de los clusters.
- [ ] Artefactos correctos en `figs/`, `tables/`, y `segments/`.

**Para Notebook consolidado (`03_final.ipynb`):**
- [ ] Ejecuta de punta a punta sin errores en una sesión limpia.
- [ ] Contiene: contexto, EDA, preparación, PCA, K-Means, DBSCAN, comparación, decisión del modelo final, interpretación de segmentos y exportación.
- [ ] Narrativa fluida y pedagógica que sirve como resumen ejecutivo y guía integral de estudio para la defensa grupal.
- [ ] Incluye la tabla comparativa y la justificación del modelo elegido.

**Para Aplicación Web y Entorno:**
- [ ] El script `app.py` no realiza sub-entrenamientos internos.
- [ ] Permite cargar variables sin normalizar en el formulario.
- [ ] `requirements.txt` documenta versiones exactas empleadas.
- [ ] Si el modelo final es DBSCAN: lógica de mapeo de `NearestNeighbors` configurada correctamente y manejo explícito del caso `-1` (ruido). Si es K-Means: predicción directa con el pipeline.

**Para Informe y Sustentación:**
- [ ] Extensión máxima de ~5 páginas respetada.
- [ ] Limitaciones técnicas explícitamente redactadas.
- [ ] Preparada la demo de 5–10 min asignando al menos 2 registros a segmentos distintos y explicando las recomendaciones de cada uno.
- [ ] Todos los integrantes pueden explicar cualquier parte del proyecto (no solo la suya), porque la sustentación puede preguntar a cualquiera.