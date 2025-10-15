# 🚦 Traffic Flow Time Series Analysis & Forecasting

## 🧠 Описание
Проект реализует **полный анализ временного ряда** для данных из **Traffic Prediction Dataset**, включая:
- первичный статистический анализ;
- декомпозицию и визуализацию;
- генерацию более **200 временных признаков**;
- **отбор признаков (feature selection)** с использованием Forward и Backward Selection;
- обучение и сравнение **трёх моделей**: Linear Regression (Ridge), Gradient Boosting (XGBoost / LightGBM) и **LSTM**;
- расчёт метрик качества с учётом временной структуры данных.

Проект оформлен в виде Jupyter Notebook и полностью воспроизводим.

---

## 🚀 Ключевые особенности
- 📈 **Полный цикл анализа временного ряда** — от описательной статистики до финальных метрик.  
- 🧩 **200+ временных признаков**, включая лаговые, скользящие, частотные и статистические.  
- 🧠 **Feature Selection** с реализацией *Forward* и *Backward* методов.  
- 🤖 **Сравнение трёх подходов** — линейных, бустинговых и рекуррентных нейросетей.  
- 📊 **Автоматическая валидация временных данных** через `TimeSeriesSplit`.  
- 🧾 **Расчёт ключевых метрик**: MAE, RMSE, MAPE, SMAPE, R².  

---

## ⚙️ Рекомендуемые требования

Для ускорения обучения рекомендуется GPU с поддержкой CUDA.

```bash
# Проверка наличия GPU
nvidia-smi
```



## 💾 Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/<твой_профиль>/<имя_репозитория>.git
cd <имя_репозитория>
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

Если файл отсутствует — можно установить вручную:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels tqdm xgboost lightgbm torch torchvision torchaudio tensorflow keras jupyter notebook tsfresh
```

---

## 🧩 Использование

### Запуск проекта
```bash
jupyter notebook notebook9ba1ec2b83(1).ipynb
```

### Этапы анализа
1. **Первичный анализ данных:**
   - описательная статистика (среднее, медиана, σ, квартили);
   - распределения, выбросы, QQ-plot, boxplot.
2. **Визуализация временного ряда:**
   - исходный график, тренд, сезонность, остатки;
   - ACF и PACF графики.
3. **Статистические тесты:**
   - Дики–Фуллера (стационарность);
   - Шапиро–Уилка (нормальность остатков);
   - Льюнга–Бокса (автокорреляция).
4. **Генерация признаков (200+):**
   - лаговые, скользящие, экспоненциальные, технические, частотные и временные;
   - энтропийные, фрактальные и нелинейные признаки.
5. **Feature Selection:**
   - Forward Selection и Backward Elimination;
   - оценка по кросс-валидации.
6. **Обучение моделей:**
   - Ridge Regression (через GridSearchCV);
   - XGBoost / LightGBM (boosting);
   - LSTM (PyTorch / Keras).
7. **Валидация и метрики:**
   - TimeSeriesSplit (5 фолдов);
   - MAE, RMSE, MAPE, SMAPE, R².

---

## 📊 Метрики качества

| Метрика | Формула | Интерпретация |
|----------|----------|----------------|
| **MAE** | ![MAE](https://latex.codecogs.com/svg.image?MAE=\frac{1}{n}\sum|y_i-\hat{y}_i|) | Средняя абсолютная ошибка |
| **RMSE** | ![RMSE](https://latex.codecogs.com/svg.image?RMSE=\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2}) | Корень из MSE |
| **MAPE** | ![MAPE](https://latex.codecogs.com/svg.image?MAPE=\frac{100%}{n}\sum\frac{|y_i-\hat{y}_i|}{|y_i|}) | Ошибка в процентах |
| **SMAPE** | ![SMAPE](https://latex.codecogs.com/svg.image?SMAPE=\frac{100%}{n}\sum\frac{|y_i-\hat{y}_i|}{(|y_i|+|\hat{y}_i|)/2}) | Симметричная ошибка |
| **R²** | ![R²](https://latex.codecogs.com/svg.image?R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}) | Коэффициент детерминации |

---

## 📁 Структура проекта
```
.
├── notebook9ba1ec2b83(1).ipynb   # Основной ноутбук с анализом
├── requirements.txt               # Зависимости проекта
└── README.md                      # Описание проекта
```

---

## 🧠 Используемые технологии
- **Python 3.10+**
- **NumPy**, **Pandas** — обработка данных  
- **Matplotlib**, **Seaborn** — визуализация  
- **SciPy**, **Statsmodels** — статистические тесты  
- **Scikit-learn** — отбор признаков, метрики  
- **XGBoost / LightGBM** — бустинговые модели  
- **PyTorch / TensorFlow** — реализация LSTM  
- **tsfresh** — автоматическая генерация временных признаков  

---

