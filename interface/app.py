import json
import os
import time
import uuid

import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st
from kafka import KafkaProducer

KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BROKERS", "kafka:9092"),
    "topic": os.getenv("KAFKA_TOPIC", "transactions"),
}

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "fraud_detection"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres")
}


def load_file(uploaded_file):
    """Загрузка CSV файла в DataFrame"""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None


def send_to_kafka(df, topic, bootstrap_servers):
    """Отправка данных в Kafka с уникальным ID транзакции"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            security_protocol="PLAINTEXT",
        )

        df["transaction_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        progress_bar = st.progress(0)
        total_rows = len(df)

        for idx, row in df.iterrows():
            # Отправляем данные вместе с ID
            producer.send(
                topic,
                value={
                    "transaction_id": row["transaction_id"],
                    "data": row.drop("transaction_id").to_dict(),
                },
            )
            progress_bar.progress((idx + 1) / total_rows)
            time.sleep(0.01)

        producer.flush()

        return True
    except Exception as e:
        st.error(f"Ошибка отправки данных: {str(e)}")
        return False


def get_db_connection():
    """Создание соединения с PostgreSQL"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            port=DB_CONFIG["port"],
        )
        return conn
    except Exception as e:
        st.error(f"Ошибка подключения к базе данных: {str(e)}")
        return None


def get_fraud_transactions():
    """Получение 10 последних транзакций с флагом fraud_flag == 1"""
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        query = """
        SELECT * 
        FROM scoring_results 
        WHERE fraud_flag is True 
        ORDER BY created_at DESC 
        LIMIT 10
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Ошибка получения данных о мошеннических транзакциях: {str(e)}")
        if conn:
            conn.close()
        return None


def get_score_distribution():
    """Получение последних 100 транзакций для построения гистограммы скоров"""
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        query = """
        SELECT score 
        FROM scoring_results 
        WHERE score IS NOT NULL 
        ORDER BY created_at DESC 
        LIMIT 100
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Ошибка получения данных о скорах: {str(e)}")
        if conn:
            conn.close()
        return None


if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

st.title("📤 Отправка данных в Kafka")

uploaded_file = st.file_uploader("Загрузите CSV файл с транзакциями", type=["csv"])

if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
    # Добавляем файл в состояние
    st.session_state.uploaded_files[uploaded_file.name] = {
        "status": "Загружен",
        "df": load_file(uploaded_file),
    }
    st.success(f"Файл {uploaded_file.name} успешно загружен!")

if st.session_state.uploaded_files:
    st.subheader("🗂 Список загруженных файлов")

    for file_name, file_data in st.session_state.uploaded_files.items():
        cols = st.columns([4, 2, 2])

        with cols[0]:
            st.markdown(f"**Файл:** `{file_name}`")
            st.markdown(f"**Статус:** `{file_data['status']}`")

        with cols[2]:
            if st.button(f"Отправить {file_name}", key=f"send_{file_name}"):
                if file_data["df"] is not None:
                    with st.spinner("Отправка..."):
                        success = send_to_kafka(
                            file_data["df"],
                            KAFKA_CONFIG["topic"],
                            KAFKA_CONFIG["bootstrap_servers"],
                        )
                        if success:
                            st.session_state.uploaded_files[file_name]["status"] = (
                                "Отправлен"
                            )
                            st.rerun()
                else:
                    st.error("Файл не содержит данных")

st.markdown("---")
st.subheader("📊 Просмотр результатов")

if st.button("Посмотреть результаты", type="primary"):
    st.markdown("### 🚨 Последние мошеннические транзакции")

    with st.spinner("Загрузка данных о мошеннических транзакциях..."):
        fraud_df = get_fraud_transactions()

    if fraud_df is not None and not fraud_df.empty:
        st.dataframe(fraud_df, use_container_width=True, hide_index=True)
        st.success(f"Найдено {len(fraud_df)} мошеннических транзакций")
    elif fraud_df is not None and fraud_df.empty:
        st.info("🎉 Мошеннических транзакций не найдено!")
    else:
        st.warning("Не удалось получить данные о мошеннических транзакциях")

    st.markdown("### 📈 Распределение скоров последних транзакций")

    with st.spinner("Загрузка данных о скорах..."):
        scores_df = get_score_distribution()

    if scores_df is not None and not scores_df.empty:
        fig = px.histogram(
            scores_df,
            x="score",
            title=f"Распределение скоров мошенничества (последние {len(scores_df)} транзакций)",
            labels={
                "score": "Скор мошенничества",
                "count": "Количество транзакций",
            },
            range_x=[0, 1]
        )

        fig.update_traces(
            marker_color="lightblue", marker_line_color="darkblue", marker_line_width=1
        )

        fig.update_layout(
            xaxis_title="Скор мошенничества",
            yaxis_title="Количество транзакций",
            showlegend=False,
            xaxis=dict(
                range=[0, 1],
                tick0=0,     
                dtick=0.1    
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего транзакций", len(scores_df))
        with col2:
            st.metric("Средний скор", f"{scores_df['score'].mean():.3f}")
        with col3:
            st.metric("Максимальный скор", f"{scores_df['score'].max():.3f}")
        with col4:
            st.metric("Минимальный скор", f"{scores_df['score'].min():.3f}")

    elif scores_df is not None and scores_df.empty:
        st.info("📊 Данных о скорах пока нет")
    else:
        st.warning("Не удалось получить данные о скорах")
