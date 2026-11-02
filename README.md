# Film Junky Union — Movie Review Sentiment Analysis

**Project type:** Binary text classification · Natural Language Processing (NLP) · Supervised Machine Learning  
**Program:** TripleTen Data Scientist Bootcamp  
**Status:** Documentation started; notebook cleanup and end-to-end reproducibility review pending.

## Business problem

Film Junky Union wants to automatically identify the sentiment of movie reviews and detect negative reviews. The project uses labeled IMDb review data to train and evaluate binary classification models.

The project brief sets a target of **F1 ≥ 0.85**.

## Dataset

The notebook records **47,331 reviews** and 17 columns. The target column is `pos`, and `ds_part` identifies the predefined train/test split.

- Training partition: **23,796 rows**
- Test partition: **23,535 rows**
- Target distribution: approximately **50.10% class 0** and **49.90% class 1**
- Missing values recorded in the initial review: 2 in `average_rating` and 2 in `votes`; `review` has no missing values.
- No fully duplicated rows were found by `df_reviews.duplicated().sum()`.

The dataset file (`imdb_reviews.tsv`) is not included in this repository. See [`data/README.md`](data/README.md) for setup notes.

## Method

The documented baseline pipeline includes:

1. Exploratory data analysis and data-quality checks.
2. Text normalization: lowercase conversion, removal of characters outside English letters/apostrophes, and whitespace normalization.
3. Reuse of the notebook's predefined training/test split.
4. TF-IDF text vectorization with English stopwords and `max_features=10000`.
5. Logistic Regression (`random_state=12345`, `max_iter=1000`).
6. Evaluation using F1, Accuracy, Average Precision Score (APS), and ROC AUC.

A `DummyClassifier(strategy="most_frequent")` is used as a baseline for comparison.

## Results recorded in the notebook

| Model | Split | Accuracy | F1 | APS | ROC AUC |
|---|---|---:|---:|---:|---:|
| DummyClassifier (most frequent) | Test | 0.50 | 0.00 | 0.50 | 0.50 |
| TF-IDF + Logistic Regression | Train | 0.93 | 0.93 | 0.98 | 0.98 |
| TF-IDF + Logistic Regression | Test | **0.88** | **0.88** | **0.95** | **0.95** |

The TF-IDF + Logistic Regression result exceeds the project target of F1 ≥ 0.85 on the recorded test evaluation. These are values stored in the submitted notebook; a clean end-to-end rerun is still required before claiming full reproducibility.

## Tools and libraries

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- NLTK
- Jupyter Notebook

The notebook also contains exploratory/unfinished sections for spaCy, LightGBM, and BERT (Transformers/PyTorch). **No final performance claims are made for those sections**, because the submitted notebook has incomplete cells/placeholders and they need to be reviewed and completed before publication.

## Repository structure

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

## Reproduce the documented baseline

1. Create and activate a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Obtain the project dataset through the authorized course/source channel and place `imdb_reviews.tsv` in the project root (or update the notebook path).
4. Review and clean the original notebook before running it from top to bottom.
5. Verify that the test metrics are reproduced; update this README if results differ.

Example setup:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

## Limitations and next steps

- Clean the original notebook: remove local machine paths and stale outputs, complete or remove unfinished model sections, and ensure all cells run in order.
- Confirm the exact preprocessing/model pipeline and reproduce the saved test metrics.
- Add a confusion matrix and error analysis, including examples of false positives and false negatives.
- Document model limitations, class-label interpretation, and dataset provenance.
- Add a concise visual summary of the model results.

## Responsible use

Sentiment classification can make mistakes, especially with sarcasm, ambiguity, negation, or context-dependent language. Predictions should be interpreted as model estimates, not as definitive judgments about a reviewer.

---

**Author:** Juan Carlos Cardona  
**Portfolio focus:** Data Science · NLP · Machine Learning
