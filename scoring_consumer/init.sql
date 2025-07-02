CREATE TABLE IF NOT EXISTS scoring_results (
    id SERIAL PRIMARY KEY,
    transaction_id TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    fraud_flag BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transaction_id ON scoring_results(transaction_id);
CREATE INDEX IF NOT EXISTS idx_fraud_flag ON scoring_results(fraud_flag);
CREATE INDEX IF NOT EXISTS idx_created_at ON scoring_results(created_at);
