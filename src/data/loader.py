import pandas as pd


def load_data_to_db(conn, df: pd.DataFrame) -> None:
    """
    Bulk inserts DataFrame records into the PostgreSQL 'films' table.

    Converts DataFrame rows to tuples and streams them using the PostgreSQL 
    COPY FROM STDIN protocol via psycopg for high-performance ingestion.

    Args:
        conn: An active psycopg database connection object.
        df (pd.DataFrame): The cleaned DataFrame containing film metadata. 
            Must match the schema order: film_id, title, genres, keywords, 
            overview, release_date, vote_average.

    Returns:
        None
    """
    # create a list of tuples because psycopg does not understand Pandas DataFrames
    # pandas internally stores missing values as float('nan'), not None —
    # even after fillna(None), numpy will expose the underlying nan again via to_numpy()
    # so we re-check at the boundary and convert to Python None for PostgreSQL NULL compatibility
    records = [
        tuple(None if (pd.isna(v)) else v for v in row)
        for row in df.to_numpy()
    ]

    # insert data to "films" table from standard input, not the file 
    copy_query = """
        COPY films (film_id, title, genres, keywords, overview, release_date, vote_average) 
        FROM STDIN
        """

    with conn.cursor() as cursor:
        with cursor.copy(copy_query) as copy:
            for record in records:
                copy.write_row(record)
    
    conn.commit()


def init_db(conn, schema_path: str = "src/db/schema.sql") -> None:
    """Reads and executes the SQL schema to create necessary database tables."""
    with open(file=schema_path, mode="r", encoding="utf-8") as f:
        schema_sql = f.read()

    with conn.cursor() as cursor:
        cursor.execute(schema_sql)
    
    conn.commit()
