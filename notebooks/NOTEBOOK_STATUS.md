# Estado del notebook — Film Junky Sentiment Analysis

## Estado de publicación

El notebook `film_junky_baseline.ipynb` se encuentra publicado en el repositorio y fue ejecutado completamente en Visual Studio Code utilizando el dataset original `imdb_reviews.tsv`.

La ejecución finalizó correctamente, conservando los resultados de los experimentos, las métricas de evaluación y el análisis de errores.

## Validaciones completadas

- Notebook incorporado a la carpeta `notebooks/`.
- Ejecución completa del notebook sin errores.
- Conservación de la partición original de entrenamiento y prueba.
- Evaluación mediante Accuracy, F1-score, ROC-AUC y Average Precision.
- Comparación de diferentes configuraciones TF-IDF.
- Ajuste de hiperparámetros mediante GridSearchCV.
- Análisis de falsos positivos y falsos negativos.
- Interpretación de coeficientes del modelo.
- Documentación de conclusiones y limitaciones.
- Dataset original excluido del repositorio público.

## Resultados del modelo final

| Métrica | Resultado |
|---|---:|
| F1-score | 0.8963 |
| Accuracy | 0.8967 |
| ROC-AUC | 0.9618 |
| Average Precision | 0.9597 |

**Modelo seleccionado:** TF-IDF con unigramas y bigramas (20,000 características) combinado con Regresión Logística (`C=4`).

## Consideraciones

Las métricas del conjunto de prueba deben interpretarse como resultados exploratorios, debido a que dicho conjunto se utilizó durante varias comparaciones experimentales.

El dataset `imdb_reviews.tsv` no se distribuye públicamente dentro del repositorio.

**Estado actual:** Notebook ejecutado y documentado, listo para consulta y revisión.
