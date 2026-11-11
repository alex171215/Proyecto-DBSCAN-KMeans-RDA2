# Justificación Técnica: Elección del Modelo Final

**Responsable:** Persona 3 (Paquete C)
**Fuente de los números:** `notebooks/03_final.ipynb` (reproduce ambos modelos y verifica que coinciden con los artefactos de los Paquetes A y B).

---

### 1. Métricas Comparadas

Resumen de `tables/comparacion.csv`:

| Modelo | Hiperparámetros | Clusters | Silhouette | % ruido | Cobertura | Seleccionado |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| K-Means | K = 4, `n_init=10`, `random_state=42` | 4 | 0.0475 | 0 % | 100 % | **sí** |
| DBSCAN | ε = 3.2, MinPts = 32 | 2 | 0.1338 (sin ruido) | 88 % | 12 % | no |

Inercia de K-Means: 11 290.70. El Silhouette de DBSCAN se calcula sólo sobre los 96 estudiantes que no son ruido, como exige el contrato técnico.

![Comparación de modelos](../figs/comparacion.png)

### 2. Criterios de Selección

1. **Silhouette (cohesión y separación):** mayor es mejor.
2. **Cobertura de estudiantes:** % asignado a un cluster que no es ruido. Regla acordada en el documento maestro antes de ver resultados: si DBSCAN marca más del 20 % como ruido, se penaliza su selección.
3. **Utilidad para la toma de decisiones:** si cada grupo se traduce en una acción institucional concreta.

Entre los modelos no penalizados se elige el de mayor Silhouette.

### 3. Modelo Seleccionado

**K-Means con K = 4** (`models/pipeline_kmeans.joblib`).

### 4. Justificación Técnica y de Negocio

DBSCAN obtiene un Silhouette mayor (0.134 frente a 0.047), pero esa cifra se mide sobre el 12 % de la población que logró formar grupos densos. Los otros 704 estudiantes (88 %) quedan como ruido. La regla del documento maestro penaliza a DBSCAN cuando el ruido supera el 20 %, porque deja sin acción a uno de cada cinco estudiantes; aquí deja sin acción a casi nueve de cada diez. Esto no es un problema de mala calibración. El barrido del Paquete B muestra que con ε ≤ 3.1 todo es ruido y con ε ≥ 3.3 todos los estudiantes densos se fusionan en un único cluster. ε = 3.2 es el único valor del rango probado que produce más de un grupo. Es el efecto de la maldición de la dimensionalidad: en 16 dimensiones las distancias entre estudiantes se parecen demasiado para que aparezcan regiones de densidad claramente distintas.

K-Means asigna a los 800 estudiantes a cuatro perfiles de tamaño parecido (171, 220, 206 y 203), y cada uno se traduce en una acción distinta: ayuda económica y tutorías para el grupo con más materias pendientes y estrés, mentoría entre pares para el grupo con más apoyo familiar y motivación, alertas de asistencia y hábitos para el grupo con más tiempo de pantalla, y reconocimiento con consejería preventiva para el grupo de mejor promedio pero mayor estrés financiero. Además, K-Means tiene `predict`, de modo que la aplicación web asigna un estudiante nuevo directamente con el pipeline exportado. DBSCAN requiere la aproximación con `NearestNeighbors` sobre los puntos core y, en la prueba de inferencia del notebook 03, dos de tres registros reales cayeron como ruido.

La decisión tiene una limitación que se declara sin rodeos: el Silhouette de K-Means es 0.047, cercano a cero. Los grupos se solapan, y muchos estudiantes están cerca de la frontera entre dos perfiles. Esto es coherente con lo que mostró el EDA: ninguna correlación entre variables supera |r| = 0.09, y PC1 + PC2 explican apenas el 15 % de la varianza. Por eso los segmentos se presentan como **tendencias para priorizar recursos de apoyo**, no como diagnósticos individuales. La recomendación para la institución es usar el segmento como primer filtro y confirmar cada caso con una entrevista de consejería. DBSCAN conserva un uso complementario: los registros que marca como atípicos pueden servir como lista de revisión manual.
