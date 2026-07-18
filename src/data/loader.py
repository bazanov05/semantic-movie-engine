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


def fetch_films_overviews(conn) -> list[tuple[int, str]]:
    """
    Fetches all non-null film overviews along with their unique identifiers.

    Args:
        conn: An active psycopg database connection object.

    Returns:
        A list of tuples, where each tuple contains (film_id, overview_text).
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT film_id, overview "
            "FROM films "
            "WHERE overview IS NOT NULL;"
        ) 
        return cursor.fetchall()


def update_film_embeddings(conn, embedding_data):
    """
    Bulk updates the database with generated vector embeddings for each film.

    Utilizes executemany to efficiently update multiple rows in a single 
    transaction and automatically commits the changes to the database.

    Args:
        conn: An active psycopg database connection object.
        embedding_data: A list of tuples formatted as (embedding_vector, film_id).
    """
    with conn.cursor() as cursor:
        cursor.executemany(
            "UPDATE films " 
            "SET embedding = %s "
            "WHERE film_id = %s;", embedding_data
        )

    conn.commit()