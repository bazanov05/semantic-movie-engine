from src.data.cleaner import clean_data
from src.data.loader import (
    init_db, 
    load_data_to_db,
    fetch_films_overviews,
    update_film_embeddings,
    create_index
)
from src.db import connection
from src.model.embedder import MovieEmbedder


PATH_TO_CSV_DATA = "./src/data/tmdb_5000_movies.csv"
PATH_TO_SQL_SCHEMA = "./src/db/schema.sql"


def main():
    connection.init_pool()

    clean_df = clean_data(file_path=PATH_TO_CSV_DATA)
    model = MovieEmbedder()

    with connection.pool.connection() as conn:
        init_db(conn=conn, schema_path=PATH_TO_SQL_SCHEMA)
        load_data_to_db(conn=conn, df=clean_df)

        # fetch film_ids and overviews from films table 
        data = fetch_films_overviews(conn=conn)

        # unzip data
        ids = [item[0] for item in data]
        overviews = [item[1] for item in data]

        # generate vector representation for each overview
        vectors = model.generate_embeddings(texts=overviews)
        updates = list(zip(vectors, ids))   # zip data 

        # insert vectors into the "embedding" cols based on provided ids 
        update_film_embeddings(conn=conn, embedding_data=updates)

        # create IVFFlat indices for embedding vectors to speed up the search
        # use default num_of_probes = 10
        create_index(conn=conn)

    connection.close_pool()


if __name__ == "__main__":
    main()
