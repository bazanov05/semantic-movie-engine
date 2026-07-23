import pandas as pd
from psycopg.rows import dict_row


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


def create_index(conn, num_of_probes: int = 10) -> None:
    """
    Creates an IVFFlat index on the embedding column and configures query-time probes.

    IVFFlat groups all vectors into clusters using k-means at build time.
    At search time Postgres compares the query vector against cluster centers first,
    then only searches inside the closest clusters instead of scanning every row.

    Args:
        conn: An active psycopg database connection object.
        num_of_probes: Number of clusters to search at query time. Higher values
                       increase accuracy at the cost of speed. Defaults to 10,
                       which follows the common rule of sqrt(lists) for 100 clusters.
                       Cannot be less than 1.
    """
    if num_of_probes <= 0:
            num_of_probes = 10

    with conn.cursor() as cursor:
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS films_embedding_idx "
            "ON films USING ivfflat (embedding vector_cosine_ops) "
            "WITH (lists = 100);"
        )
        
        cursor.execute(
            "SELECT set_config('ivfflat.probes', %s, false);",
            (str(num_of_probes),)
        )
    
    conn.commit()


def group_by_genres(conn) -> dict[str, list[int]]:
    """
    Fetches all films grouped by genre from the database.

    Unpacks the JSONB genres array for each film, extracts the genre name,
    and aggregates film IDs under each genre. Genres are ordered by number
    of films descending.

    Args:
        conn: An active psycopg database connection object.

    Returns:
        A dictionary mapping each genre name to a list of film IDs belonging to it.
        Example: {"Action": [1, 5, 23, ...], "Comedy": [2, 8, 14, ...]}
    """
    with conn.cursor() as cursor:
        cursor.row_factory = dict_row

        cursor.execute(
            "WITH expanded_json AS(" 
            "   SELECT film_id, genre FROM films " 
            "   CROSS JOIN jsonb_array_elements(genres) AS genre" 
            "), " 
            "   expanded_genre AS(" 
            "   SELECT " 
            "       film_id, " 
            "       genre ->> 'name' AS genre_name " 
            "   FROM expanded_json" 
            ") " 
            "SELECT " 
            "   genre_name, " 
            "   ARRAY_AGG(film_id ORDER BY film_id) AS movies " 
            "FROM expanded_genre " 
            "GROUP BY genre_name " 
            "ORDER BY COUNT(film_id) DESC;"
        )

        results = cursor.fetchall()
        return {row["genre_name"]: row["movies"] for row in results}
    