from argparse import ArgumentParser
import sys
import os

from normalizer_python import TextNormalizer
from src.data.loader import (
    update_film_embeddings,
    create_index,
    fetch_films_semantic_meaning,
)
from src.db import connection
from src.model.embedder import MovieEmbedder
from src.model.film_encoder import FilmEncoder


PATH_TO_STOP_WORDS = "./normalizer/data/stopwords.txt"
PATH_TO_MODEL_WEIGHTS = "./src/model/data/weights.pt"


def fetch_stop_words(path: str = PATH_TO_STOP_WORDS) -> set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            stop_words = set()

            for line in f:
                cleaned_word = line.strip()

                if cleaned_word:
                    stop_words.add(cleaned_word)

            return stop_words
    except FileNotFoundError:
        raise FileNotFoundError(f"Path {path} does not exist")
    except IsADirectoryError:
        raise IsADirectoryError("You are trying to open the dir, not file")
    except PermissionError:
        raise PermissionError("You do not have permission to open this file")


def main(args):
    """
    Generates and stores film embeddings into the database.

    Parses CLI arguments to select either a pretrained or fine-tuned model. 
    Retrieves semantic film metadata, cleans rich overviews via C++ TextNormalizer, 
    generates vector representations, bulk updates PostgreSQL, and builds 
    an IVFFlat vector index for fast similarity search.

    Args:
        args: Command-line argument list (typically sys.argv).
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        default="pretrained"
    )

    arguments = parser.parse_args(args=args[1:])

    # init pool to get connection
    connection.init_pool()

    # if --model argument was not given - use pretrained model from HuggingFace
    # otherwise used finetuned model with extra Projection Layer
    if arguments.model == "pretrained":
        model = MovieEmbedder()
        column = "embedding"
    else:
        # try to load trained model's weights
        if os.path.exists(path=PATH_TO_MODEL_WEIGHTS):
            model = FilmEncoder.load(path=PATH_TO_MODEL_WEIGHTS)
        else:
            # otherwise raise NotImplementedError 
            # so the model with random weights in Projection Layer will not be created
            raise FileNotFoundError("Model was not trained - run 'train.py'")
    
        column = "embedding_finetuned"

    # fetch stop words and create text normalizer
    stop_words = fetch_stop_words()
    text_normalizer = TextNormalizer(stop_words)

    with connection.pool.connection() as conn:
        films_info = fetch_films_semantic_meaning(conn=conn)

        ids, overviews, genres, keywords = zip(*films_info)

        # enrich overviews with genres and keywords
        rich_overviews = [
            f"overview {overviews[i]} "
            f"genre {genres[i]} "
            f"keywords {keywords[i]} "
            for i in range(len(ids))
        ]

        # delete stop words from overviews with text normalizer
        rich_overviews = [text_normalizer.clean(rich_overview) for rich_overview in rich_overviews]

        # generate embeddings for films and store them in db
        vectors = model.generate_embeddings(texts=rich_overviews)
        updates = list(zip(vectors, ids))
        update_film_embeddings(conn=conn, embedding_data=updates, column=column)

        # create IVFFlat indices for newly created vectors 
        # with default probes = 10
        create_index(conn=conn, column=column)


    connection.close_pool()


if __name__ == "__main__":
    main(sys.argv)
