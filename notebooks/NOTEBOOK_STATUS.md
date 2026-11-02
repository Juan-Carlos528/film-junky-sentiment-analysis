# Estado del notebook para publicación

Se agregó `film_junky_baseline.ipynb`, una versión depurada que conserva el baseline TF-IDF + Regresión Logística y añade matriz de confusión y revisión de errores. Excluye los experimentos incompletos de spaCy, LightGBM y BERT del notebook académico original.

## Antes de publicar resultados
- [ ] Obtener `imdb_reviews.tsv` por un canal autorizado y verificar sus condiciones de uso.
- [ ] Ejecutar `film_junky_baseline.ipynb` completo desde un kernel limpio.
- [ ] Confirmar que las métricas se reproducen; actualizar README solo con resultados de esta ejecución.
- [ ] Completar conclusiones y revisión manual de falsos positivos/negativos.
- [ ] Revisar que no haya rutas locales, datos restringidos, credenciales ni salidas sensibles.

El notebook académico original `Spring_17(5).ipynb` se conserva fuera del repositorio público. Las métricas F1 0.88, Accuracy 0.88 y ROC-AUC 0.95 estaban presentes en sus salidas guardadas, pero no se consideran reproducidas hasta ejecutar esta versión limpia.
