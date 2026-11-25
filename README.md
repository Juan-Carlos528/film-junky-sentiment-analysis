# Film Junky — Movie Review Sentiment Analysis
### NLP | Machine Learning | Python | Scikit-learn

**Languages:** English | Español

---

## 🇺🇸 English

### 1. Project Overview

This project focuses on developing a machine learning model to automatically detect the sentiment of movie reviews.

The objective is to classify reviews as either positive or negative using Natural Language Processing (NLP) techniques and supervised machine learning algorithms.

The project explores different text vectorization configurations, evaluates model performance, analyzes classification errors, and tunes hyperparameters to improve generalization.

### 2. Business Objective

Build a sentiment classification model that can help identify customer opinions about movies by analyzing written reviews.

Automated sentiment analysis can support:
- Understanding audience feedback.
- Identifying positive and negative opinions.
- Analyzing large volumes of textual information.
- Supporting data-driven decisions based on customer feedback.

### 3. Dataset

The project uses the `imdb_reviews.tsv` dataset.

**Dataset characteristics:**
- 47,331 movie reviews.
- 17 columns.
- Training set: 23,796 records.
- Test set: 23,535 records.
- Target variable: `pos` (0 = negative, 1 = positive).

The dataset contains review text, ratings, movie information, and sentiment labels.

**Note:** The original dataset is not included in this public repository.

### 4. Methodology

The project follows a structured machine learning workflow:

1. **Data loading and exploration**
   - Dataset inspection.
   - Data types and missing values.
   - Class distribution.
   - Duplicate review analysis.

2. **Data preparation**
   - Separation of training and test datasets.
   - Definition of text features and target variable.
   - Preservation of the original dataset split.

3. **Baseline model**
   - DummyClassifier using the most frequent class.
   - Establishment of a reference performance level.

4. **Text vectorization and model development**
   - TF-IDF with unigrams.
   - TF-IDF with unigrams and bigrams.
   - Comparison of 10,000 and 20,000 features.
   - Logistic Regression classification.

5. **Model evaluation**
   - Accuracy.
   - F1-score.
   - ROC-AUC.
   - Average Precision.
   - Confusion matrix.
   - Classification report.

6. **Error analysis**
   - False positive and false negative inspection.
   - Comparison between predicted sentiment and original ratings.
   - Identification of cases where review wording and assigned ratings differ.

7. **Cross-validation and hyperparameter tuning**
   - Stratified 5-fold cross-validation.
   - GridSearchCV.
   - Evaluation of different regularization values (`C`).

8. **Model interpretation**
   - Analysis of positive and negative model coefficients.
   - Identification of words and expressions associated with each sentiment class.

### 5. Models and Results

The experiments evaluated different TF-IDF configurations combined with Logistic Regression.

| Model | Accuracy | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|
| Dummy Baseline | 0.5015 | 0.0000 | — | — |
| TF-IDF Unigrams (10K) | 0.8828 | 0.8823 | 0.9529 | 0.9506 |
| TF-IDF Bigrams (10K) | 0.8932 | 0.8933 | 0.9584 | 0.9564 |
| TF-IDF Bigrams (20K) | 0.8935 | 0.8935 | 0.9594 | 0.9576 |
| Final Logistic Regression (C=4) | 0.8967 | 0.8963 | 0.9618 | 0.9597 |

**Final model:** TF-IDF with unigrams and bigrams (20,000 features) combined with Logistic Regression using `C=4`.

The final model achieved an F1-score of approximately **0.8963** on the test dataset.

The selected regularization parameter was chosen considering cross-validation performance, stability, and the training-validation performance gap. Although `C=6` obtained a marginally higher cross-validation F1-score, `C=4` provided a more stable result.

### 6. Key Findings

- Adding bigrams improved classification performance compared with using unigrams alone.
- Increasing the TF-IDF vocabulary from 10,000 to 20,000 features produced additional improvements.
- Hyperparameter tuning improved the model's cross-validation performance.
- Error analysis showed that some reviews contain mixed opinions or wording that does not fully match their assigned rating.
- Model coefficients helped identify words and expressions statistically associated with positive and negative sentiment.

### 7. Limitations

- The model relies on patterns learned from the training data and may not correctly interpret sarcasm, irony, or complex context.
- TF-IDF represents text through numerical word and phrase frequencies rather than deep semantic understanding.
- The sentiment labels are derived from the dataset's rating-based classification.
- The test dataset was used repeatedly during experimentation, so the final test metrics should be interpreted as exploratory rather than as a completely independent final evaluation.

### 8. Technologies Used

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- Jupyter Notebook
- Natural Language Processing (NLP)
- TF-IDF
- Logistic Regression
- GridSearchCV
- Stratified Cross-Validation

### 9. Repository Structure

```text
film-junky-sentiment-analysis/
│
├── data/
│   └── (Dataset excluded from the public repository)
│
├── notebooks/
│   └── film_junky_baseline.ipynb
│
├── .gitignore
└── README.md
```

### 10. How to Run the Project

**Requirements:** Python 3.12 or a compatible Python environment.

1. Clone the repository:

```bash
git clone https://github.com/Juan-Carlos528/film-junky-sentiment-analysis.git
```

2. Navigate to the project directory:

```bash
cd film-junky-sentiment-analysis
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel nltk
```

5. Place the `imdb_reviews.tsv` dataset in the `data/` directory.

6. Open the notebook located at:

```text
notebooks/film_junky_baseline.ipynb
```

7. Run the notebook cells sequentially.

### 11. Conclusion

This project demonstrates an end-to-end Natural Language Processing workflow, from data exploration and preparation to model development, evaluation, hyperparameter tuning, and interpretation.

The final Logistic Regression model achieved an F1-score of 0.8963, demonstrating the effectiveness of TF-IDF features and supervised learning for movie review sentiment classification.

---

## 🇲🇽 Español

### 1. Descripción del proyecto

Este proyecto se enfoca en desarrollar un modelo de Machine Learning capaz de detectar automáticamente el sentimiento de reseñas cinematográficas.

El objetivo es clasificar las reseñas como positivas o negativas mediante técnicas de Procesamiento de Lenguaje Natural (NLP) y algoritmos de aprendizaje supervisado.

Se experimenta con diferentes configuraciones de vectorización de texto, evaluación del rendimiento, análisis de errores y ajuste de hiperparámetros para mejorar la capacidad de generalización del modelo.

### 2. Objetivo de negocio

Construir un modelo de clasificación de sentimientos que permita identificar opiniones de los usuarios sobre películas mediante el análisis de reseñas escritas.

El análisis automatizado de sentimientos puede ayudar a:
- Comprender la opinión de las audiencias.
- Identificar comentarios positivos y negativos.
- Analizar grandes volúmenes de información textual.
- Apoyar la toma de decisiones basada en la retroalimentación de los usuarios.

### 3. Conjunto de datos

El proyecto utiliza el conjunto de datos `imdb_reviews.tsv`.

**Características del dataset:**
- 47,331 reseñas cinematográficas.
- 17 columnas.
- Conjunto de entrenamiento: 23,796 registros.
- Conjunto de prueba: 23,535 registros.
- Variable objetivo: `pos` (0 = negativo, 1 = positivo).

El conjunto contiene texto de reseñas, calificaciones, información de películas y etiquetas de sentimiento.

**Nota:** El dataset original no está incluido en este repositorio público.

### 4. Metodología

El proyecto sigue un flujo de trabajo estructurado de Machine Learning:

1. **Carga y exploración de datos**
   - Inspección del dataset.
   - Tipos de datos y valores faltantes.
   - Distribución de clases.
   - Análisis de reseñas duplicadas.

2. **Preparación de datos**
   - Separación de los conjuntos de entrenamiento y prueba.
   - Definición de variables predictoras y variable objetivo.
   - Conservación de la división original del dataset.

3. **Modelo baseline**
   - DummyClassifier utilizando la clase más frecuente.
   - Establecimiento de un rendimiento de referencia.

4. **Vectorización de texto y desarrollo de modelos**
   - TF-IDF con unigramas.
   - TF-IDF con unigramas y bigramas.
   - Comparación de 10,000 y 20,000 características.
   - Clasificación mediante Regresión Logística.

5. **Evaluación de modelos**
   - Accuracy.
   - F1-score.
   - ROC-AUC.
   - Average Precision.
   - Matriz de confusión.
   - Reporte de clasificación.

6. **Análisis de errores**
   - Inspección de falsos positivos y falsos negativos.
   - Comparación entre el sentimiento predicho y las calificaciones originales.
   - Identificación de casos donde la redacción de una reseña no coincide completamente con su calificación asignada.

7. **Validación cruzada y ajuste de hiperparámetros**
   - Validación cruzada estratificada de 5 particiones.
   - GridSearchCV.
   - Evaluación de diferentes valores de regularización (`C`).

8. **Interpretabilidad del modelo**
   - Análisis de coeficientes positivos y negativos.
   - Identificación de palabras y expresiones estadísticamente asociadas con cada clase de sentimiento.

### 5. Modelos y resultados

Se evaluaron diferentes configuraciones de TF-IDF combinadas con Regresión Logística.

| Modelo | Accuracy | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|
| Dummy Baseline | 0.5015 | 0.0000 | — | — |
| TF-IDF Unigramas (10K) | 0.8828 | 0.8823 | 0.9529 | 0.9506 |
| TF-IDF Bigramas (10K) | 0.8932 | 0.8933 | 0.9584 | 0.9564 |
| TF-IDF Bigramas (20K) | 0.8935 | 0.8935 | 0.9594 | 0.9576 |
| Regresión Logística Final (C=4) | 0.8967 | 0.8963 | 0.9618 | 0.9597 |

**Modelo final:** TF-IDF con unigramas y bigramas (20,000 características), combinado con Regresión Logística utilizando `C=4`.

El modelo final alcanzó un F1-score aproximado de **0.8963** sobre el conjunto de prueba.

El parámetro de regularización seleccionado se eligió considerando el rendimiento en validación cruzada, la estabilidad y la diferencia entre el rendimiento de entrenamiento y validación. Aunque `C=6` obtuvo un F1-score ligeramente superior en validación cruzada, `C=4` presentó un resultado más estable.

### 6. Principales hallazgos

- La incorporación de bigramas mejoró el rendimiento de clasificación frente al uso exclusivo de unigramas.
- Incrementar el vocabulario TF-IDF de 10,000 a 20,000 características generó mejoras adicionales.
- El ajuste de hiperparámetros mejoró el rendimiento del modelo en validación cruzada.
- El análisis de errores mostró que algunas reseñas contienen opiniones mixtas o expresiones que no coinciden completamente con la calificación asignada.
- Los coeficientes del modelo permitieron identificar palabras y expresiones estadísticamente asociadas con sentimientos positivos y negativos.

### 7. Limitaciones

- El modelo depende de los patrones aprendidos durante el entrenamiento y puede presentar dificultades para interpretar sarcasmo, ironía o contextos complejos.
- TF-IDF representa el texto mediante frecuencias numéricas de palabras y expresiones, en lugar de comprender profundamente su significado.
- Las etiquetas de sentimiento provienen de la clasificación basada en las calificaciones del dataset.
- El conjunto de prueba se utilizó repetidamente durante la experimentación, por lo que las métricas finales deben interpretarse como exploratorias y no como una evaluación final completamente independiente.

### 8. Tecnologías utilizadas

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- Jupyter Notebook
- Procesamiento de Lenguaje Natural (NLP)
- TF-IDF
- Regresión Logística
- GridSearchCV
- Validación cruzada estratificada

### 9. Estructura del repositorio

```text
film-junky-sentiment-analysis/
│
├── data/
│   └── (Dataset excluido del repositorio público)
│
├── notebooks/
│   └── film_junky_baseline.ipynb
│
├── .gitignore
└── README.md
```

### 10. Cómo ejecutar el proyecto

**Requisitos:** Python 3.12 o un entorno compatible de Python.

1. Clonar el repositorio:

```bash
git clone https://github.com/Juan-Carlos528/film-junky-sentiment-analysis.git
```

2. Entrar al directorio del proyecto:

```bash
cd film-junky-sentiment-analysis
```

3. Crear y activar un entorno virtual:

```bash
python -m venv .venv
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Instalar las bibliotecas necesarias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel nltk
```

5. Colocar el dataset `imdb_reviews.tsv` dentro del directorio `data/`.

6. Abrir el notebook ubicado en:

```text
notebooks/film_junky_baseline.ipynb
```

7. Ejecutar las celdas secuencialmente.

### 11. Conclusión

Este proyecto demuestra un flujo de trabajo integral de Procesamiento de Lenguaje Natural, desde la exploración y preparación de datos hasta el desarrollo, evaluación, ajuste e interpretación de modelos.

El modelo final de Regresión Logística alcanzó un F1-score de 0.8963, demostrando la efectividad de las características TF-IDF y el aprendizaje supervisado para clasificar sentimientos en reseñas cinematográficas.

---

**Author:** Juan Carlos Cardona  
**GitHub:** [Juan-Carlos528](https://github.com/Juan-Carlos528)  
**Project Repository:** [Film Junky — Sentiment Analysis](https://github.com/Juan-Carlos528/film-junky-sentiment-analysis)
