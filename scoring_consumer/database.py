from models import Base


def init_db(engine):
    """Создание таблиц в базе данных"""
    Base.metadata.create_all(engine)