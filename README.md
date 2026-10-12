# Predicción de ventas de videojuegos con Machine Learning

Proyecto final de Machine Learning (Talento Tech): modelo de regresión que estima las ventas globales de un videojuego a partir de su plataforma, año, género y publisher.

[Abrir en Google Colab](https://colab.research.google.com/drive/18qMCMSdXWveelYTFJgeaArK1dLOeupsF?usp=sharing) · [Ver notebook completo](https://nbviewer.org/github/jeqm0302/prediccion-ventas-videojuegos-ml/blob/main/prediccion_ventas.ipynb)

## Pregunta

¿Se pueden predecir las ventas globales de un videojuego a partir de sus características iniciales (plataforma, año, género y publisher)?

## Datos

[Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) (Kaggle): ~16.600 videojuegos con ventas por región, en millones de unidades.

## Proceso

1. **EDA:** carga, tratamiento de nulos, estadística descriptiva y visualizaciones.
2. **Feature engineering:** `Decada`, `Generacion_Consola` y `Publisher_Agrupado`.
3. **Modelos:** Regresión Ridge (baseline lineal e interpretable) y Random Forest Regressor (captura interacciones no lineales).
4. **Validación:** K-Fold con k = 5.

## Resultados

| Modelo | R² test | R² CV (media ± desvío) |
|---|---|---|
| Ridge | COMPLETAR | COMPLETAR |
| Random Forest | COMPLETAR | COMPLETAR |

- La validación cruzada muestra un rendimiento estable entre folds, así que las métricas no dependen de un split favorable.
- El modelo predice bien la masa central de juegos (ventas entre 0,05 M y 2 M).
- **Limitación:** subestima sistemáticamente los blockbusters (Wii Sports Resort, GTA V, Call of Duty). Plataforma y género son necesarios pero no suficientes para explicar un fenómeno cultural; factores como el marketing o la calidad no están en los datos.

## Utilidad

Sirve como estimador conservador para nuevos lanzamientos y como detector de casos atípicos: si un juego vende mucho más de lo predicho, ese desvío merece análisis cualitativo.

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