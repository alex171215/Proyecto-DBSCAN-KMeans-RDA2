# Guía Maestra para la Sustentación Oral y Defensa del Proyecto 2
## Clustering con Machine Learning y Aplicación Web
**Duración:** 5 a 10 minutos por grupo  
**Objetivo:** Demostrar dominio conceptual del ciclo completo de aprendizaje no supervisado, justificación de hiperparámetros, interpretación de negocio y demo interactiva en vivo.

---

## 1. Estructura Sugerida de la Presentación (5–10 Minutos)

| Minuto | Integrante Sugerido | Tema a Exponer | Apoyo Visual |
| :---: | :---: | :--- | :--- |
| **0:00 - 2:00** | **Integrante 1 (Datos y EDA)** | • Pregunta de negocio: por qué es clustering no supervisado (no clasificación).<br>• Dataset (800 estudiantes, 16 variables psicométricas y académicas).<br>• Justificación del escalado con `StandardScaler` e imputación mediana.<br>• Análisis de PCA: por qué entrenar en 16D y no en 2D (varianza acumulada de solo 15%). | Diapositiva / Notebook: Gráficas de EDA, correlación y varianza PCA. |
| **2:00 - 4:30** | **Integrante 2 (Modelado DBSCAN y K-Means)** | • Selección de $K=4$ en K-Means (codo de inercia y Silhouette).<br>• DBSCAN: heurística $\text{MinPts}=32$ ($2D$) y gráfico $k$-dist para $\varepsilon=3.20$.<br>• Explicación de la **maldición de la dimensionalidad** y por qué DBSCAN arrojó 88% de ruido.<br>• Regla de penalización (>20% de ruido) y por qué K-Means ganó la selección final. | Notebook: Gráficos de Codo, Silhouette, $k$-dist y barras comparativas. |
| **4:30 - 7:00** | **Integrante 3 (Interpretación y Demo Web)** | • Explicación de los 4 perfiles en escala original y acciones de retención.<br>• **Demo en vivo de la App:** Ingreso de Registro 1 (Estudiante vulnerable $\rightarrow$ Cluster 0) e ingreso de Registro 2 (Estudiante destacado $\rightarrow$ Cluster 2).<br>• Explicación de cómo predice la app (Pipeline K-Means con datos crudos). | Pantalla compartida con la App Web interactiva (`streamlit run app.py`). |
| **7:00 - 10:00** | **Todo el Equipo** | • Ronda de preguntas del docente y conclusiones. | Notebook / Código abierto ante cualquier consulta técnica. |

---

## 2. Las 10 Preguntas Clave del Docente y Respuestas de Alto Impacto

### 1. ¿Por qué este proyecto es de Clustering No Supervisado y no de Clasificación?
> **Respuesta:**  
> "Porque el objetivo no es predecir una etiqueta preexistente. Las columnas `Burnout_Level` y `Dropout_Risk` se excluyeron por completo desde el inicio para evitar fuga de información (*data leakage*). El algoritmo descubre de manera autónoma las agrupaciones naturales en el hiperespacio de 16 dimensiones basándose únicamente en distancias métricas entre estudiantes."

### 2. ¿Por qué usaron `StandardScaler` y no `MinMaxScaler`?
> **Respuesta:**  
> "`MinMaxScaler` acota artificialmente los datos entre 0 y 1 y es sumamente vulnerable a valores atípicos, comprimiendo a la mayoría de los estudiantes en rangos muy estrechos. `StandardScaler` estandariza a media $\mu=0$ y varianza $\sigma^2=1$. Esto es indispensable porque variables con rangos de 0 a 100 (como `Attendance_Percent`) dominarían las distancias euclidianas sobre variables de 0 a 10 (como `Previous_GPA` o `Stress_Level`). Con `StandardScaler`, cada variable aporta equitativamente a la distancia hiperespacial."

### 3. Si calcularon PCA, ¿por qué no entrenaron K-Means y DBSCAN sobre las componentes principales?
> **Respuesta:**  
> "Porque las dos primeras componentes (PC1 y PC2) solo explican el **15.03% de la varianza total** (PC1: 7.62% y PC2: 7.41%). Entrenar sobre PC1 y PC2 habría significado desechar casi el **85% de la información multidimensional** de los estudiantes. Por tanto, entrenamos sobre las 16 dimensiones completas para máxima fidelidad matemática y usamos PCA exclusivamente como herramienta de proyección bidimensional e interpretabilidad."

### 4. ¿Qué representan las dos primeras componentes principales (PC1 y PC2)?
> **Respuesta:**  
> "A través del análisis de *loadings* (cargas factoriales):
> - **PC1 representa el eje de 'Soporte Socioemocional y Rendimiento':** Cargas fuertemente positivas en `Family_Support_Score` (+0.48), `Previous_GPA` (+0.39) y `Motivation_Score` (+0.30), frente a cargas negativas en `Screen_Time_Hours` (-0.41) y `Sleep_Hours` (-0.34).
> - **PC2 representa el eje de 'Hábitos de Vida y Disciplina':** Dominada por `Exercise_Freq_Per_Week` (+0.56), `Attendance_Percent` (+0.36) y `Age` (+0.35)."

### 5. ¿Cómo eligieron el número de clusters $K=4$ en K-Means?
> **Respuesta:**  
> "Evaluamos $K$ de 2 a 9 analizando simultáneamente la Inercia (suma de distancias al cuadrado intra-cluster) y el Coeficiente de Silhouette. Observamos que entre $K=2$ y $K=4$ la inercia disminuye abruptamente de 12,155 a 10,919, formándose el 'codo' de desaceleración. Aunque $K=2$ tiene una silueta ligeramente más alta, agrupa a los estudiantes en solo dos mitades demasiado genéricas. $K=4$ mantiene una silueta de 0.063 y produce cuatro segmentos perfectamente balanceados de ~200 estudiantes cada uno, permitiendo a la universidad diseñar planes de retención específicos."

### 6. ¿Por qué DBSCAN arrojó 88% de puntos clasificados como ruido (-1)? ¿Falló el algoritmo?
> **Respuesta:**  
> "No, el algoritmo no falló; es una consecuencia matemática de la **maldición de la dimensionalidad**. En 16 dimensiones continuas, el hipervolumen del espacio es gigantesco y los puntos tienden a estar dispersos. Con $\text{MinPts}=32$ ($2D$), si $\varepsilon < 3.20$ el 100% es ruido; si $\varepsilon \ge 3.30$ todos los puntos se conectan en un solo megagrupo. El valor $\varepsilon=3.20$ (obtenido del gráfico $k$-dist) fue el único que aisló dos densidades puras (65 y 31 estudiantes), dejando a los 704 estudiantes restantes en la periferia como ruido (-1)."

### 7. ¿Por qué se seleccionó K-Means sobre DBSCAN como modelo final?
> **Respuesta:**  
> "Por el criterio de **utilidad institucional y cobertura**. La regla del proyecto estipula que un modelo con más del 20% de ruido se penaliza. DBSCAN dejó al 88% de la población como ruido atípico, lo cual dejaría a 704 de 800 estudiantes sin una estrategia definida. K-Means garantiza el **100% de cobertura**, segmentando a toda la cohorte en 4 grupos con acciones preventivas viables."

### 8. Si DBSCAN hubiera sido el modelo final, ¿cómo clasificaría registros nuevos si no tiene método `.predict()`?
> **Respuesta:**  
> "DBSCAN es un algoritmo transductivo sin método predictivo nativo. Para producción, implementamos la regla vista en clase: ajustamos un estimador `NearestNeighbors(n_neighbors=1)` sobre los 96 puntos *core* del entrenamiento. Cuando ingresa un nuevo estudiante, se estandariza y se calcula su distancia al core point más cercano: si la distancia es $\le \varepsilon$ (3.20), adopta el cluster de ese core point; si es mayor, se etiqueta como Ruido (-1) y la app emite una recomendación de revisión manual por bienestar estudiantil."

### 9. ¿Cómo se comunican el formulario de la App Web y el modelo exportado?
> **Respuesta:**  
> "El usuario ingresa las 16 variables en su **escala original cruda** (por ejemplo, GPA de 0 a 10, Asistencia de 0 a 100%). La app carga `pipeline_kmeans.joblib`, el cual ejecuta automáticamente el `SimpleImputer` y el `StandardScaler` con los mismos parámetros del entrenamiento, entregando la predicción en tiempo real sin requerir transformaciones manuales."

### 10. ¿Cuáles son los 4 segmentos encontrados y qué acciones concretas se recomiendan?
> **Respuesta:**  
> 1. **Cluster 0 — Rezago y Presión Financiera (171 est., 21.4%):** Mayor número de materias reprobadas (1.79 backlogs), alto estrés financiero (5.65) y baja motivación (4.45), aunque mantienen asistencia (85.3%). $\rightarrow$ *Acción:* Becas de auxilio económico, tutorías intensivas de nivelación y plan de recuperación académica.
> 2. **Cluster 1 — Participación y Apoyo Familiar (220 est., 27.5%):** Máximo soporte familiar (7.33), alta motivación (6.06), actividad social (5.98) y ejercicio (2.72). $\rightarrow$ *Acción:* Mantener el acompañamiento y convocarlos como mentores pares para estudiantes vulnerables.
> 3. **Cluster 2 — Baja Asistencia y Alta Pantalla (206 est., 25.8%):** Semestres más avanzados (año 3.04), asistencia más baja (69.1%) y el mayor tiempo en pantalla (6.72 h/día). $\rightarrow$ *Acción:* Alertas tempranas de inasistencia, talleres sobre gestión del tiempo y uso responsable de pantallas.
> 4. **Cluster 3 — Buen GPA con Seguimiento de Bienestar (203 est., 25.4%):** Estudiantes de primeros ciclos (año 1.66) con el GPA previo más alto (7.64), pero con el menor tiempo de sueño (5.62 h) y alto estrés financiero (5.93). $\rightarrow$ *Acción:* Reconocimiento académico, becas de manutención y apoyo psicológico preventivo para evitar el colapso por sobreexigencia.

---

## 3. Guion para la Demo en Vivo de la App (2 Registros Contrastantes)

Para la sustentación en vivo frente al docente, ingresen los siguientes dos perfiles en el formulario de la aplicación web (`streamlit run app.py`):

### Registro 1: Estudiante en Riesgo por Rezago y Presión Financiera $\rightarrow$ Asigna a Cluster 0
- **Año de Estudio:** 2
- **Materias Pendientes (Backlogs):** 3 materias
- **Estrés Financiero:** 8.0 puntaje
- **Nivel de Estrés:** 8.5 puntaje
- **Motivación:** 3.0 puntaje
- **Asistencia:** 85.0 %
- *Resultado en la App:* Asigna **Segmento 0: Rezago y presión financiera**. La app resalta el punto en el mapa PCA 2D y recomienda: *"Becas o ayuda económica, tutorías para materias pendientes y plan de recuperación académica"*.

### Registro 2: Estudiante con Desconexión y Alta Pantalla $\rightarrow$ Asigna a Cluster 2
- **Año de Estudio:** 3
- **Asistencia:** 60.0 %
- **Tiempo de Pantalla:** 8.0 h/día
- **Materias Pendientes (Backlogs):** 1 materia
- **Apoyo Familiar:** 4.5 puntaje
- *Resultado en la App:* Asigna **Segmento 2: Baja asistencia y alta pantalla**. La app resalta la estrella en el cuadrante de pantalla alta y recomienda: *"Alertas tempranas de asistencia y talleres sobre uso de pantallas y hábitos de sueño"*.

