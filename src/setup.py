from src.data.cleaner import clean_data
from src.data.loader import (
    init_db, 
    load_data_to_db
)
from src.db import connection


PATH_TO_CSV_DATA = "./src/data/tmdb_5000_movies.csv"
PATH_TO_SQL_SCHEMA = "./src/db/schema.sql"


def setup():
    """
    Initializes the database schema and loads cleaned dataset records.

    Establishes a connection pool, processes raw CSV movie data through the 
    cleaning pipeline, executes the SQL schema script to construct database 
    tables, and performs bulk insertion of film records into PostgreSQL.
    """
    connection.init_pool()

    # clean data in .csv file
    clean_df = clean_data(file_path=PATH_TO_CSV_DATA)

    with connection.pool.connection() as conn:
        # initialize db based on schema and load cleaned data to it
        init_db(conn=conn, schema_path=PATH_TO_SQL_SCHEMA)
        load_data_to_db(conn=conn, df=clean_df)

    connection.close_pool()


if __name__ == "__main__":
    setup()
