# Predicción de ventas de videojuegos con Machine Learning

Proyecto final de Machine Learning (Talento Tech): modelos de regresión que estiman las ventas globales de un videojuego a partir de su plataforma, año, género y publisher.

[Abrir en Google Colab](https://colab.research.google.com/drive/18qMCMSdXWveelYTFJgeaArK1dLOeupsF?usp=sharing) · [Ver notebook completo](https://nbviewer.org/github/jeqm0302/prediccion-ventas-videojuegos-ml/blob/main/prediccion_ventas.ipynb)

## Pregunta

¿Se pueden predecir las ventas globales de un videojuego a partir de sus características iniciales?

## Datos

[Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) (Kaggle, basado en VGChartz): 16.598 videojuegos con ventas por región, en millones de unidades.

## Proceso

1. **EDA:** nulos, estadística descriptiva, outliers (IQR) y análisis por plataforma, género, publisher y año.
2. **Feature engineering:** `Decada`, `Generacion_Consola` y `Publisher_Agrupado` (top 20 publishers, que concentran el 85,7 % de las ventas, más "Otros").
3. **Preparación:** one-hot encoding, split 80/20 estratificado por cuantiles del target y escalado de variables temporales.
4. **Modelos:** Regresión Ridge (baseline) y Random Forest Regressor, entrenados sobre `log1p(Global_Sales)`.
5. **Evaluación:** MAE, RMSE y R² en test, validación cruzada K-Fold (k = 5), análisis de residuos, importancia de variables y peores errores.

## Resultados

**Test** (escala original, millones de unidades):

| Métrica | Ridge | Random Forest |
|---|---|---|
| MAE | 0.444 | 0.441 |
| RMSE | 1.464 | 1.474 |
| R² | 0.106 | 0.094 |

**Validación cruzada, 5 folds** (escala logarítmica):

| Métrica | Ridge | Random Forest |
|---|---|---|
| R² | 0.257 ± 0.019 | 0.284 ± 0.026 |

## Conclusiones

- **Ambos modelos rinden casi igual:** la señal está limitada por las variables disponibles, no por la complejidad del modelo.
- **Poder predictivo bajo:** plataforma, género, año y publisher explican una parte chica de la variación de ventas. El error típico (0.44 M) es más del doble de la mediana (0.17 M).
- **Sesgo en los extremos:** sobreestima la mayoría de los juegos de ventas bajas y subestima fuertemente los blockbusters. Los 10 peores errores son subestimaciones (Mario Kart Wii: 35.8 M reales contra 1.1 M predichos).
- **Variables más influyentes:** año, publisher "Otros" y Nintendo. Las 5 primeras acumulan el 60 % de la importancia.
- **Rendimiento estable** entre folds (desvío del R² ≤ 0.026).

## Decisiones técnicas

- Se eliminaron las ventas regionales para evitar **fuga de datos** (`Global_Sales` es su suma).
- Se conservaron los outliers porque son lanzamientos reales, no errores de carga.
- Se transformó el target con `log1p` por su fuerte asimetría (mediana 0.17 M, máximo 82.7 M).

## Próximos pasos

- Probar XGBoost o LightGBM y ajustar hiperparámetros con `GridSearchCV`.
- Sumar variables de calidad (puntajes de Metacritic) y de marketing.
- Probar un modelo en dos etapas: clasificar si un juego será blockbuster y después estimar sus ventas.

## Cómo ejecutarlo

```bash
git clone https://github.com/jeqm0302/prediccion-ventas-videojuegos-ml.git
cd prediccion-ventas-videojuegos-ml
pip install -r requirements.txt
jupyter notebook prediccion_ventas.ipynb
```

## Estructura

```
├── data/
│   └── vgsales.csv
├── prediccion_ventas.ipynb
├── requirements.txt
└── README.md
```

## Herramientas

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Google Colab

## Autor

**Jesús Enrique Quijada Martínez** · [LinkedIn](https://linkedin.com/in/jesúsenrique-quijada-martínez-57b70922b) · [GitHub](https://github.com/jeqm0302)