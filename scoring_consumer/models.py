from datetime import datetime, UTC

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Text, Float, Boolean, Integer, TIMESTAMP


class Base(DeclarativeBase):
    pass


class ScoringResult(Base):
    __tablename__ = "scoring_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    fraud_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.now(UTC), index=True
    )

    def to_dict(self):
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "score": self.score,
            "fraud_flag": self.fraud_flag,
            "created_at": self.created_at.isoformat(),
        }
