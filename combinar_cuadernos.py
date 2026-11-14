import json
from pathlib import Path

# Paths to the 3 original notebooks
nb1_path = Path("Proyecto/notebooks/01_kmeans.ipynb")
nb2_path = Path("Proyecto/notebooks/02_dbscan.ipynb")
nb3_path = Path("Proyecto/notebooks/03_final.ipynb")

nb1 = json.loads(nb1_path.read_text(encoding="utf-8"))
nb2 = json.loads(nb2_path.read_text(encoding="utf-8"))
nb3 = json.loads(nb3_path.read_text(encoding="utf-8"))

combined_cells = []

# Title cell
title_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Proyecto 2: Clustering con Machine Learning — Cuaderno Maestro Integrado\n",
        "## Integración Completa de los Tres Cuadernos: K-Means (01), DBSCAN (02) y Consolidación Final (03)\n",
        "**Asignatura:** Inteligencia Artificial — 6to Semestre, Ingeniería en Sistemas de Información  \n",
        "**Integrantes:** Alejandro Flores, Paúl Rosero, Gloria Chushig  \n",
        "**Fecha:** Noviembre 2026  \n",
        "**Repositorio:** `https://github.com/alex171215/Proyecto-DBSCAN-KMeans-RDA2.git`\n",
        "\n",
        "---\n",
        "### Estructura del Cuaderno Unificado:\n",
        "- **PARTE I: Modelado con K-Means (Cuaderno 01 — Paquete A)**: Carga controlada, EDA de variables, PCA, selección de $K$ (codo e inercia), ajuste final, perfiles y exportación.\n",
        "- **PARTE II: Agrupamiento con DBSCAN (Cuaderno 02 — Paquete B)**: Fundamento de densidad, detección de atípicos, selección de $\\text{MinPts} = 32$ y $\\varepsilon = 3.20$ ($k$-dist), barrido experimental, perfilado, inferencia con `NearestNeighbors` y exportación.\n",
        "- **PARTE III: Pipeline Consolidado, Comparación y Decisión Final (Cuaderno 03 — Paquete C)**: Reproducción cruzada, tabla comparativa oficial, regla del 20% de ruido, selección del modelo ganador (K-Means), perfilamiento institucional en escala original y pruebas de inferencia para producción.\n",
        "---\n"
    ]
}
combined_cells.append(title_cell)

# Part I separator
part1_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# =============================================================================\n",
        "# PARTE I: MODELADO CON K-MEANS (Cuaderno 01 — Persona 1 / Paquete A)\n",
        "# =============================================================================\n",
        "> **Objetivo de la Parte I:** Explorar la cohorte estudiantil, analizar correlaciones y valores extremos, proyectar varianza con PCA, determinar el número óptimo de clusters $K$ mediante el método del codo y Silhouette, entrenar el Pipeline de K-Means, caracterizar sus 4 segmentos y exportar sus artefactos oficiales.\n"
    ]
}
combined_cells.append(part1_header)

for i, cell in enumerate(nb1["cells"]):
    c = json.loads(json.dumps(cell))
    if i == 0 and c["cell_type"] == "markdown":
        c["source"] = ["## Parte I — " + line for line in c["source"]]
    combined_cells.append(c)

# Part II separator
part2_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# =============================================================================\n",
        "# PARTE II: AGRUPAMIENTO CON DBSCAN (Cuaderno 02 — Persona 2 / Paquete B)\n",
        "# =============================================================================\n",
        "> **Objetivo de la Parte II:** Modelar la cohorte estudiantil mediante densidad con DBSCAN. Determinar $\\text{MinPts} = 2 \\cdot D = 32$, graficar las distancias al $k$-ésimo vecino ($k$-dist) para calibrar $\\varepsilon = 3.20$, evaluar la maldición de la dimensionalidad, analizar el 88% de ruido, perfilar los núcleos densos e implementar la regla de inferencia con `NearestNeighbors`.\n"
    ]
}
combined_cells.append(part2_header)

for i, cell in enumerate(nb2["cells"]):
    c = json.loads(json.dumps(cell))
    if i == 0 and c["cell_type"] == "markdown":
        c["source"] = ["## Parte II — " + line for line in c["source"]]
    combined_cells.append(c)

# Part III separator
part3_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# =============================================================================\n",
        "# PARTE III: PIPELINE CONSOLIDADO, COMPARACIÓN Y DECISIÓN FINAL (Cuaderno 03 — Persona 3 / Paquete C)\n",
        "# =============================================================================\n",
        "> **Objetivo de la Parte III:** Contrastar formalmente K-Means y DBSCAN. Re-ejecutar ambos modelos para verificar reproducibilidad exacta estudiante por estudiante contra los artefactos exportados, evaluar la regla de penalización por ruido (>20%), construir la tabla comparativa oficial, declarar el modelo ganador (K-Means), perfilar sus 4 segmentos en escala original y verificar las pruebas de inferencia para la aplicación web.\n"
    ]
}
combined_cells.append(part3_header)

for i, cell in enumerate(nb3["cells"]):
    c = json.loads(json.dumps(cell))
    if i == 0 and c["cell_type"] == "markdown":
        c["source"] = ["## Parte III — " + line for line in c["source"]]
    combined_cells.append(c)

out_nb = {
    "cells": combined_cells,
    "metadata": nb3.get("metadata", {}),
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = Path("Proyecto/Entrega_Proyecto_2/1_Notebook/Proyecto2_Clustering_Master.ipynb")
out_path.write_text(json.dumps(out_nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Total cells combined: {len(combined_cells)}")
print(f"Successfully saved to {out_path}")

# Also copy to Entrega folder if present
entrega_copy = Path("Proyecto/Entrega_Proyecto_2/Entrega/Proyecto2_Clustering_Master.ipynb")
if entrega_copy.parent.exists():
    entrega_copy.write_text(json.dumps(out_nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Also updated {entrega_copy}")
