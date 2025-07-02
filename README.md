# Fraud Detection Service

Сервис детекции мошенничества в реальном времени с использованием Kafka и CatBoost, разработанный на основе данных соревнования [Kaggle Teta ML 2025](https://www.kaggle.com/competitions/teta-ml-1-2025).

## 📋 Описание

Система обрабатывает транзакции через Kafka, применяет [ML-модель](https://github.com/s1lver29/teta_mlops_2025/tree/homework_1) для детекции фрода и предоставляет веб-интерфейс для мониторинга результатов.

## 🏗️ Архитектура

```
📊 Web Interface (Streamlit) → 📡 Kafka → 🤖 ML Service → 💾 PostgreSQL
```
## 🚀 Быстрый старт

### 1. Предварительные требования
- Docker & Docker Compose
- Файл `train.csv` в папке `./fraud_service/train/`

### 2. Запуск системы
```bash
# Клонирование и переход в директорию
git clone clone https://github.com/s1lver29/teta_mlops_2025.git -b homework_2
cd teta_mlops_2025
mv .env.example .env

docker-compose up --build
```

### 3. Проверка готовности
Дождитесь сообщений о готовности всех сервисов:
- ✅ Kafka topics created
- ✅ Fraud service started  
- ✅ Scoring consumer connected to DB
- ✅ Streamlit interface ready


### 4. Доступ к интерфейсам

| Сервис | URL (по умолчанию в конфигурации) | Описание |
|--------|-----|----------|
| 🌐 **Web Interface** | http://localhost:8501 | Отправка транзакций и просмотр результатов |
| 📊 **Kafka UI** | http://localhost:8080 | Мониторинг Kafka топиков и сообщений |

## 📁 Структура проекта

```
teta_mlops_2025/
├── .env                     # Переменные окружения
├── docker-compose.yaml      # Оркестрация всех сервисов
├── README.md               # Документация
│
├── fraud_service/          # ML сервис детекции мошенничества
│   ├── app.py             # Kafka consumer + CatBoost модель
│   ├── Dockerfile         # Контейнер ML сервиса
│   ├── config/            # Конфигурационные файлы
│   ├── src/               # Исходный код (model.py, preprocessing.py)
│   ├── models/            # Обученная модель CatBoost
│   └── train/             # Тренировочные данные
│
├── interface/              # Web интерфейс (Streamlit)
│   ├── app.py             # Streamlit приложение
│   └── Dockerfile         # Контейнер веб-интерфейса
│
└── scoring_consumer/       # Сервис сохранения результатов
    ├── app.py             # Kafka consumer для записи в БД
    ├── database.py        # Подключение к PostgreSQL
    ├── models.py          # SQLAlchemy модели
    └── Dockerfile         # Контейнер consumer'а
```

## 🔧 Использование

### Отправка транзакций
1. Откройте http://localhost:8501
2. Загрузите CSV файл с транзакциями
3. Нажмите **Отправить**
4. Наблюдать процесс обработки в *логах* или K*afka UI*

### Просмотр результатов
- На той же странице нажмите "Посмотреть результаты"
- Для обновления результатов нажмите снове "Посмотреть результаты"

### Формат входных данных
CSV файл должен содержать столбцы из соревнование (например `test.csv`).

## 🐛 Устранение неполадок

### Проблемы с запуском
```bash
# Перезапуск с пересборкой
docker-compose down
docker-compose up --build

# Просмотр логов
docker-compose logs fraud_service
docker-compose logs interface

```

### Проверка топиков и сообщений

Для просмотра сообщений и их наличия можно через Kafka UI (по умолчанию находится `http://localhost:8080`)

### Проверка данных в БД
```bash
# Подключение к PostgreSQL
docker exec -it db_scoring_model_result psql -U fraud_user -d fraud_detection

# Просмотр результатов
\dt
SELECT COUNT(*) FROM scoring_results;
SELECT * FROM scoring_results ORDER BY created_at DESC LIMIT 5;
```

## 📊 Мониторинг

- **Логи ML сервиса**: `docker logs fraud_model`
- **Kafka UI**: `http://localhost:8080`
- **Статус БД**: `docker logs db_scoring_model_result`


