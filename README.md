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
