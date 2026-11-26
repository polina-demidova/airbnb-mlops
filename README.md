# Airbnb Price Prediction

Учебный проект ML/DLOps, НИУ ВШЭ. Задача — регрессия цены аренды по характеристикам объявления Inside Airbnb.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Заполните AWS_ACCESS_KEY_ID и AWS_SECRET_ACCESS_KEY в `.env` выданными ключами. Остальные параметры уже заданы: Yandex Object Storage, bucket `mlops26-course-bucket`, key `prepared/listings.csv`. Не публикуйте `.env` и `secretkeys.py`.

```bash
jupyter lab
```

- `01_read_data.ipynb` — чтение S3 и проверка локального fallback. Локальная копия создаётся автоматически после успешного чтения.
- `01_training.ipynb` — самостоятельное чтение S3, подготовка признаков, разбиение 80/20, baseline и Random Forest, MAE, RMSE, R².
- `requirements.txt` — зависимости.

Запускайте ячейки сверху вниз. Ручное скачивание данных не требуется. Метрики вычисляются при запуске на реальных данных; заранее не заявляются.

## Обучение модели

Ноутбук: `01_training.ipynb`.

Задача: прогнозирование стоимости аренды жилья на основе характеристик объявлений Inside Airbnb.

Этапы работы:

- Загрузка данных из S3 с возможностью использования локальной копии.
- Исследовательский анализ данных (EDA).
- Удаление дубликатов и проверка целевой переменной.
- Логарифмическое преобразование цены.
- Разделение данных по listing_id с помощью GroupShuffleSplit.
- Обработка пропусков и кодирование категориальных признаков.
- Обучение Baseline и Random Forest.
- Оценка качества и анализ важности признаков.

### Результаты на тестовой выборке

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 149.546 | 426.094 | -0.030 |
| Random Forest | 94.824 | 379.137 | 0.185 |

Random Forest снижает MAE на 36,6% и RMSE на 11,0% относительно Baseline.


## 3. Model Pipeline and Serialization

В ноутбуке `02_pipeline.ipynb` реализован единый sklearn Pipeline,
объединяющий предобработку данных и модель машинного обучения.

### Pipeline

- SimpleImputer — обработка пропущенных значений.
- OneHotEncoder — кодирование категориальных признаков.
- ColumnTransformer — объединение преобразований признаков.
- RandomForestRegressor — модель из 200 деревьев.
- TransformedTargetRegressor — логарифмирование целевой переменной
  и обратное преобразование предсказаний.

### Model evaluation

| Metric | Random Forest Pipeline |
|--------|------------------------|
| MAE    | 96.738                 |
| RMSE   | 382.317                |
| R²     | 0.171                  |

### Model serialization

Обученный Pipeline сохранён в формате pickle (`model.pkl`).
Размер модели составляет 198.96 МБ.

Проверено совпадение предсказаний до и после сериализации,
а также выполнение предсказания для нового объявления.

Модель загружена в S3:

`s3://mlops26-course-bucket/models/model.pkl`

Файл модели исключён из Git с помощью `.gitignore`.
