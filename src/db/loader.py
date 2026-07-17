import pandas as pd


def clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads, filters, and sanitizes movie dataset CSV file for database ingestion.

    Extracts essential metadata columns, renames keys to align with the SQL schema, 
    and handles missing values (e.g., converting missing dates to None for native 
    SQL NULL compatibility).

    Args:
        file_path (str): The path to the target .csv file.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing the sanitized movie records.

    Raises:
        ValueError: If the provided file path does not end with '.csv'.
    """
    if not file_path.endswith(".csv"):
        raise ValueError(".csv file should be loaded to Pandas DataFrame")
    
    df = pd.read_csv(filepath_or_buffer=file_path)  # read data from .csv file 
    # get rid of unnecessary columns
    df = df[
        [
        "id",
        "title",
        "genres",
        "keywords",
        "overview",
        "release_date",
        "vote_average"
        ]
    ]

    # rename "id" column so it matches SQL schema
    df = df.rename(columns={
        "id": "film_id"
    })

    # clean data - fill empty or destroyed data
    df["overview"] = df["overview"].fillna("")
    df["release_date"] = df["release_date"].where(df["release_date"].notna(), None)
    df["vote_average"] = df["vote_average"].fillna(0.0)
    df["keywords"] = df["keywords"].fillna("[]")
    df["genres"] = df["genres"].fillna("[]")
    
    return df


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
    # create a list of tuples cause db does not understand Pandas df
    # do not include pandas rows indices and add tuples without names
    records = list(df.itertuples(index=False, name=None))

    # insert data to "films" table from standard input, not the file 
    copy_query = """
        COPY films (film_id, title, genres, keywords, overview, release_date, vote_average) 
        FROM STDIN
        """

    with conn.cursor() as cursor:
        with cursor.copy(copy_query) as copy:
            copy.write_many(records)
    
    conn.commit()


def init_db(conn, schema_path: str = "src/db/schema.sql") -> None:
    """Reads and executes the SQL schema to create necessary database tables."""
    with open(file=schema_path, mode="r", encoding="utf-8") as f:
        schema_sql = f.read()

    with conn.cursor() as cursor:
        cursor.execute(schema_sql)
    
    conn.commit()