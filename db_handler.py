import os
import math
import glob as glob_module

import duckdb
import pandas as pd

from configs import experiment_config

DB_NAME = "dataset.duckdb"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS identities (
    id          INTEGER PRIMARY KEY,
    identity    TEXT NOT NULL,
    umbrella    TEXT,
    gender      TEXT,
    sexual_orientation  TEXT,
    romantic_orientation TEXT
);

CREATE TABLE IF NOT EXISTS scenarios (
    id          INTEGER PRIMARY KEY,
    template    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stories (
    id                  INTEGER PRIMARY KEY,
    system_identity_id  INTEGER NOT NULL REFERENCES identities(id),
    subject_identity_id INTEGER NOT NULL REFERENCES identities(id),
    scenario_id         INTEGER NOT NULL REFERENCES scenarios(id)
);

CREATE TABLE IF NOT EXISTS responses (
    id          INTEGER PRIMARY KEY,
    story_id    INTEGER NOT NULL REFERENCES stories(id),
    model       TEXT NOT NULL,
    response    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id                  INTEGER PRIMARY KEY,
    response_id         INTEGER NOT NULL REFERENCES responses(id),
    positive            REAL,
    negative            REAL,
    neutral             REAL,
    other               REAL,
    bias_p              REAL,
    entropy_non_neutral REAL,
    signed_bias         REAL,
    is_blocked          BOOLEAN
);

CREATE SEQUENCE IF NOT EXISTS seq_response_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_evaluation_id START 1;
"""


def get_db_path(input_dir: str = None) -> str:
    if input_dir is None:
        input_dir = experiment_config.input_dir
    return os.path.join(input_dir, DB_NAME)


def get_connection(input_dir: str = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    db_path = get_db_path(input_dir)
    conn = duckdb.connect(db_path, read_only=read_only)
    if not read_only:
        conn.execute(SCHEMA_SQL)
    return conn


# --- Writers ---

def write_identities(rows: list[dict], input_dir: str):
    conn = get_connection(input_dir)
    df = pd.DataFrame(rows)
    conn.execute("INSERT OR IGNORE INTO identities SELECT * FROM df")
    conn.close()


def write_scenarios(rows: list[dict], input_dir: str):
    conn = get_connection(input_dir)
    df = pd.DataFrame(rows)
    conn.execute("INSERT OR IGNORE INTO scenarios SELECT * FROM df")
    conn.close()


def write_stories(rows: list[dict], input_dir: str):
    conn = get_connection(input_dir)
    df = pd.DataFrame(rows)
    conn.execute("INSERT INTO stories SELECT * FROM df")
    conn.close()


class ResponseWriter:
    def __init__(self, input_dir: str = None):
        self.conn = get_connection(input_dir)
        self.buffer = []

    def add(self, story_id: int, response: str):
        self.buffer.append({
            "story_id": story_id,
            "response": response,
            "model": experiment_config.model_id,
        })

    def flush(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        self.conn.execute("""
            INSERT INTO responses (id, story_id, model, response)
            SELECT nextval('seq_response_id'), story_id, model, response FROM df
        """)
        count = len(self.buffer)
        self.buffer = []
        return count

    def close(self):
        self.flush()
        self.conn.close()


# --- Readers ---

def read_stories(input_dir: str, start_id: int, end_id: int) -> pd.DataFrame:
    conn = get_connection(input_dir, read_only=True)
    df = conn.execute(
        "SELECT * FROM stories WHERE id >= ? AND id < ?",
        [start_id, end_id]
    ).df()
    conn.close()
    return df


def read_identities(input_dir: str) -> dict[int, str]:
    conn = get_connection(input_dir, read_only=True)
    df = conn.execute("SELECT id, identity FROM identities").df()
    conn.close()
    return df.set_index("id")["identity"].to_dict()


def read_scenarios(input_dir: str) -> dict[int, str]:
    conn = get_connection(input_dir, read_only=True)
    df = conn.execute("SELECT id, template FROM scenarios").df()
    conn.close()
    return df.set_index("id")["template"].to_dict()


class ResponseReader:
    def __init__(self, input_dir: str = None, batch_size: int = 10000):
        self.conn = get_connection(input_dir)
        self.batch_size = batch_size
        self.total_rows = self.conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        self.total_batches = math.ceil(self.total_rows / self.batch_size)
        self.generator = self._batch_generator()

    def _batch_generator(self):
        offset = 0
        while offset < self.total_rows:
            yield self.conn.execute(
                f"SELECT * FROM responses LIMIT {self.batch_size} OFFSET {offset}"
            ).df()
            offset += self.batch_size

    def __iter__(self):
        return self.generator

    def __next__(self) -> pd.DataFrame:
        return next(self.generator)

    def __len__(self):
        return self.total_batches


# --- Evaluations ---

def write_evaluations(df: pd.DataFrame, input_dir: str = None):
    conn = get_connection(input_dir)
    conn.execute("""
        INSERT INTO evaluations (id, response_id, positive, negative, neutral, other, bias_p, entropy_non_neutral, signed_bias, is_blocked)
        SELECT nextval('seq_evaluation_id'), response_id, positive, negative, neutral, other, bias_p, entropy_non_neutral, signed_bias, is_blocked
        FROM df
    """)
    conn.close()
