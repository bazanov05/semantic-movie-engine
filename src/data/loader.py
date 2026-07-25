import pandas as pd
from psycopg.rows import dict_row
import numpy as np


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


def update_film_embeddings(conn, embedding_data, column: str = "embedding") -> None:
    """
    Bulk updates the database with generated vector embeddings for each film.

    Utilizes executemany to efficiently update multiple rows in a single 
    transaction and automatically commits the changes to the database.

    Args:
        conn: An active psycopg database connection object.
        embedding_data: A list of tuples formatted as (embedding_vector, film_id).
        column: A name of embedding column, which should be updated.
    """
    with conn.cursor() as cursor:
        cursor.executemany(
            "UPDATE films " 
            f"SET {column} = %s "
            "WHERE film_id = %s;", embedding_data
        )

    conn.commit()


def create_index(conn, num_of_probes: int = 10, column: str = "embedding") -> None:
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
        column: A name of embedding column, for which the IVFFlat index should be created.
    """
    if num_of_probes <= 0:
            num_of_probes = 10

    if column == "embedding":
        index_name = "pretrained_embedding_idx"
    else:
        index_name = "finetuned_embedding_idx"

    with conn.cursor() as cursor:
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON films USING ivfflat ({column} vector_cosine_ops) "
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


def build_distance_matrix(conn, column: str = "embedding") -> dict[int, dict[int, float]]:
    """
    Builds a pairwise cosine distance matrix for all films with embeddings.

    Fetches all film embeddings from the database, normalizes them to unit length
    so cosine similarity reduces to a dot product, then computes all pairwise
    distances in one matrix multiplication. Used by FilmPairDataset to efficiently
    sample semi-hard negatives during training without querying the database
    on every batch.

    Args:
        conn: An active psycopg database connection object.
        column: name of embedding column.

    Returns:
        A nested dictionary mapping each film_id to a dict of all other film_ids
        and their cosine distances. Example: {1: {2: 0.3, 3: 0.7, ...}, ...}
        Distance of 0 means identical, 2 means opposite.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT " 
            "   film_id, " 
            f"   {column} " 
            "FROM films " 
            "ORDER BY film_id ASC;"
        )

        result = cursor.fetchall()  # result is list of tuples(film_id, embedding)

        # unzip result
        ids, embeddings = map(np.array, zip(*result))

        # for each vector calculate the norm
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        # normalize every vector, so the length of every vector is 1
        # in this case cosine similarity = dot product because lenghtes of vectors are 1
        normalized_embeddings = embeddings / norms

        # get the similarities between vectors (1000, 384) @ (384, 1000) = (1000, 1000)
        similarities = normalized_embeddings @ normalized_embeddings.T
        cosine_distances = np.subtract(1, similarities) # cosine_distace = 1 - cosine_similarity

        distance_matrix = {}

        # build a result dict - {id1: {id2: distance}}
        for i, id1 in enumerate(ids):
            distance_matrix[id1] = {}

            for j, id2 in enumerate(ids):
                distance_matrix[id1][id2] = cosine_distances[i][j]

        return distance_matrix


def fetch_films_embeddings(conn, column: str = "embedding") -> dict[int, list[float]]:
    """
    Fetch embeddings for all films from the database.

    Args:
        conn: An active PostgreSQL database connection.

    Returns:
        A dictionary mapping each film ID to its embedding vector.
    """
    with conn.cursor() as cursor:
        cursor.row_factory = dict_row

        cursor.execute(
            f"SELECT film_id, {column} FROM films "
            "ORDER BY film_id;"
        )

        return {row["film_id"]: row[f"{column}"] for row in cursor.fetchall()}


def group_by_keywords(conn, max_frequency: int = 200) -> dict[str, list[int]]:
    """
    Fetches films grouped by keyword from the database, filtering out overly common keywords.

    Unpacks the JSONB keywords array for each film, extracts the keyword name, and
    aggregates film IDs. Keywords that appear in `max_frequency` or more films are
    filtered out to remove generic noise (acting as an IDF/stop-word filter) and keep
    only high-signal, specific keywords.

    Args:
        conn: An active psycopg database connection object.
        max_frequency (int, optional): The maximum number of films a keyword can belong
                                       to before it is dropped as too generic.
                                       Defaults to 200.

    Returns:
        A dictionary mapping each filtered keyword name to a list of film IDs.
        Example: {"alien planet": [1, 23, 104], "space war": [1, 58]}
    """
    with conn.cursor() as cursor:
        cursor.row_factory = dict_row

        cursor.execute(
            "WITH expanded_json AS(" 
            "   SELECT film_id, keyword FROM films " 
            "   CROSS JOIN jsonb_array_elements(keywords) AS keyword" 
            "), " 
            "   expanded_keyword AS(" 
            "   SELECT " 
            "       film_id, " 
            "       keyword ->> 'name' AS keyword_name " 
            "   FROM expanded_json" 
            ") " 
            "SELECT " 
            "   keyword_name, " 
            "   ARRAY_AGG(film_id ORDER BY film_id) AS movies " 
            "FROM expanded_keyword " 
            "GROUP BY keyword_name " 
            "ORDER BY COUNT(film_id) DESC;"
                )

        results = cursor.fetchall()

        # we want to save only keywords which appear in less than 200 films
        # cause they are not noisy and carry semantic sense 
        filtered_results = {
            row["keyword_name"]: row["movies"] 
            for row in results 
            if len(row["movies"]) < max_frequency
        }

        return filtered_results


def fetch_films_semantic_meaning(conn) -> list[tuple[int, str, str, str]]:
    """
    Fetches film metadata needed to generate rich semantic embeddings.

    Unpacks the JSONB genres and keywords arrays for each film, aggregates them
    into space-separated strings, and joins with the overview. The resulting
    text fields can be concatenated to form a rich input for the embedding model,
    providing more signal than overview alone.

    Only films with both genres and keywords are returned due to INNER JOIN —
    films missing either field are excluded.

    Args:
        conn: An active psycopg database connection object.

    Returns:
        A list of tuples (film_id, overview, genres, keywords) ordered by film_id.
        Example: [(1, "A story about...", "Action Sci-Fi", "alien space battle"), ...]
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "WITH unpacked_json_genres AS ("
            "    SELECT film_id, jsonb_array_elements(genres) AS genre "
            "    FROM films"
            "), "
            "unpacked_genres AS ("
            "    SELECT film_id, genre ->> 'name' AS genre_name "
            "    FROM unpacked_json_genres"
            "), "
            "grouped_genres AS ("
            "    SELECT film_id, string_agg(genre_name, ' ') AS genres "
            "    FROM unpacked_genres "
            "    GROUP BY film_id"
            "), "
            "unpacked_json_keywords AS ("
            "    SELECT film_id, jsonb_array_elements(keywords) AS keyword "
            "    FROM films"
            "), "
            "unpacked_keywords AS ("
            "    SELECT film_id, keyword ->> 'name' AS keyword_name "
            "    FROM unpacked_json_keywords"
            "), "
            "grouped_keywords AS ("
            "    SELECT film_id, string_agg(keyword_name, ' ') AS keywords "
            "    FROM unpacked_keywords "
            "    GROUP BY film_id"
            ") "
            "SELECT "
            "    f.film_id, "
            "    f.overview, "
            "    g.genres, "
            "    k.keywords "
            "FROM films AS f "
            "INNER JOIN grouped_genres AS g "
            "    ON f.film_id = g.film_id "
            "INNER JOIN grouped_keywords AS k "
            "    ON f.film_id = k.film_id "
            "ORDER BY f.film_id;"
        )

        return cursor.fetchall()
