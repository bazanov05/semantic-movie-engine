from src.data.cleaner import clean_data
from src.data.loader import init_db, load_data_to_db
from src.db import connection


PATH_TO_CSV_DATA = "./src/data/tmdb_5000_movies.csv"
PATH_TO_SQL_SCHEMA = "./src/db/schema.sql"


def main():
    connection.init_pool()

    clean_df = clean_data(file_path=PATH_TO_CSV_DATA)

    with connection.pool.connection() as conn:
        init_db(conn=conn, schema_path=PATH_TO_SQL_SCHEMA)
        load_data_to_db(conn=conn, df=clean_df)

    connection.close_pool()


if __name__ == "__main__":
    main()
