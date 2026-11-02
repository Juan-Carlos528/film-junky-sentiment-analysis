# Film Junky Union — Análisis de sentimientos de reseñas de películas

**Tipo de proyecto:** Clasificación binaria de texto · NLP · Machine Learning supervisado  
**Programa:** Bootcamp de Data Scientist de TripleTen  
**Estado:** Starter Pack preparado; el baseline depurado está incluido, pendiente de ejecución limpia con el dataset autorizado.

## Problema de negocio

Film Junky Union busca identificar automáticamente el sentimiento de reseñas de películas y detectar reseñas negativas. El proyecto utiliza reseñas de IMDb etiquetadas para entrenar y evaluar modelos de clasificación binaria.

El objetivo establecido en el proyecto es alcanzar **F1 ≥ 0.85**.

## Datos

El notebook registra **47,331 reseñas** y 17 columnas. La variable objetivo es `pos`, mientras que `ds_part` identifica la partición predefinida de entrenamiento y prueba.

- Entrenamiento: **23,796 filas**
- Prueba: **23,535 filas**
- Distribución de la variable objetivo: aproximadamente **50.10% clase 0** y **49.90% clase 1**
- En la revisión inicial se registraron 2 valores faltantes en `average_rating` y 2 en `votes`; `review` no presenta valores faltantes.
- No se encontraron filas completamente duplicadas mediante `df_reviews.duplicated().sum()`.

El archivo de datos (`imdb_reviews.tsv`) no se incluye en este repositorio. Consulta [`data/README.md`](data/README.md).

## Metodología documentada

1. Análisis exploratorio y validación de calidad de datos.
2. Normalización del texto: conversión a minúsculas, eliminación de caracteres distintos de letras inglesas/apóstrofos y normalización de espacios.
3. Uso de la partición de entrenamiento/prueba incluida en el conjunto.
4. Vectorización TF-IDF con stopwords en inglés y `max_features=10000`.
5. Regresión Logística (`random_state=12345`, `max_iter=1000`).
6. Evaluación con F1, Accuracy, Average Precision Score (APS) y ROC AUC.

Se utiliza `DummyClassifier(strategy="most_frequent")` como línea base.

## Resultados históricos del notebook académico original

| Modelo | Partición | Accuracy | F1 | APS | ROC AUC |
|---|---|---:|---:|---:|---:|
| DummyClassifier (clase más frecuente) | Prueba | 0.50 | 0.00 | 0.50 | 0.50 |
| TF-IDF + Regresión Logística | Entrenamiento | 0.93 | 0.93 | 0.98 | 0.98 |
| TF-IDF + Regresión Logística | Prueba | **0.88** | **0.88** | **0.95** | **0.95** |

El resultado registrado de TF-IDF + Regresión Logística supera el objetivo de F1 ≥ 0.85 en la evaluación de prueba. Estos valores aparecen en las salidas guardadas del notebook académico original. **No se consideran reproducidos todavía** en el notebook depurado incluido en este repositorio. Ejecuta el nuevo notebook y actualiza esta sección únicamente con los resultados verificados.

## Tecnologías

- Python
- pandas y NumPy
- Matplotlib y Seaborn
- scikit-learn
- NLTK
- Jupyter Notebook

El notebook también contiene secciones exploratorias o incompletas con spaCy, LightGBM y BERT (Transformers/PyTorch). **No se atribuyen resultados finales a esas secciones**, porque el archivo entregado contiene celdas incompletas y marcadores de código pendiente.

## Estructura del repositorio

```text
film-junky-sentiment-analysis/
├── README.md
├── README_ES.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── notebooks/
    └── NOTEBOOK_STATUS.md
```

## Cómo reproducir el baseline documentado

1. Crear y activar un entorno virtual de Python.
2. Instalar las dependencias de `requirements.txt`.
3. Obtener el dataset por el canal autorizado del curso o fuente y colocar `imdb_reviews.tsv` en la raíz del proyecto (o ajustar la ruta del notebook).
4. Abrir `notebooks/film_junky_baseline.ipynb` y ejecutar todas las celdas desde un kernel limpio.
5. Verificar las métricas de prueba y completar las conclusiones; actualizar este README solo con resultados reproducidos.

Ejemplo de configuración:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

## Limitaciones y siguientes pasos

- Ejecutar el notebook depurado y confirmar que el pipeline se reproduce desde cero.
- Completar la revisión manual de falsos positivos y falsos negativos.
- Documentar el origen de los datos, interpretación de etiquetas y limitaciones.
- Agregar un resumen visual de los resultados.

## Uso responsable

La clasificación de sentimientos puede fallar con sarcasmo, ambigüedad, negaciones o lenguaje dependiente del contexto. Las predicciones son estimaciones del modelo, no juicios definitivos sobre las opiniones de los usuarios.

---

**Autor:** Juan Carlos Cardona  
**Enfoque del portafolio:** Ciencia de Datos · NLP · Machine Learning
