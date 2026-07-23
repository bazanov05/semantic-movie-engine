from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np


class FilmPairDataset(Dataset):
    """
    A PyTorch Dataset that generates triplets of film embeddings for triplet loss training.

    Each sample consists of an anchor film, a positive film (sharing 2+ genres with anchor),
    and a negative film (sharing fewer than 2 genres). Sampling follows a priority hierarchy:

    1. Semi-hard negative — positive is closer than negative to anchor, but not yet by the margin.
       Provides the strongest learning signal without destabilizing early training.
    2. Hard negative — negative is closer to anchor than positive. Used as fallback when
       no semi-hard candidate exists.
    3. Easy negative — random positive and negative pair. Weakest signal, used as last resort.

    All embeddings and distances are loaded into memory at init time from precomputed
    data structures, avoiding any database queries during training.

    Args:
        matrix_distances: Pairwise cosine distance matrix {film_id: {film_id: distance}}.
        genres: Genre to film ID mapping {genre_name: [film_ids]}.
        embeddings: Film ID to embedding vector mapping {film_id: [384 floats]}.
        margin: Minimum required distance gap between positive and negative pairs.
                Defaults to 0.2, consistent with the triplet loss margin.
    """
    def __init__(
            self, 
            matrix_distances: dict[int, dict[int, float]], 
            genres: dict[str, list[int]],
            embeddings: dict[int, list[float]],
            margin: float = 0.2
        ):
        super().__init__()

        self._matrix_distances = matrix_distances
        self._genres = genres
        self._embeddings = embeddings

        # distance between pos and anchor should be smaller than distance between neg and anchor
        # at least by margin
        self._margin = margin

        # DataLoader will give indices in range [0, 4802], cause we have 4803 films
        # but film_ids are not in this range - there are gaps
        # so we need to map those indices to real film_ids
        self._films_ids = list(embeddings.keys())

    def __len__(self):
        """Returns the total number of films in the dataset."""
        return len(self._embeddings)

    def __getitem__(self, index) -> tuple[list[float], list[float] | None, list[float] | None]:
        """
        Returns a triplet of embeddings for the film at the given index.

        Maps DataLoader's sequential index to a real film_id, then samples
        a positive and negative candidate using semi-hard → hard → easy fallback.
        Returns None for positive and negative if no valid candidates exist,
        which should be handled by a custom collate_fn in the DataLoader.

        Args:
            index: DataLoader's sequential index in range [0, len(dataset) - 1].

        Returns:
            A tuple of (anchor, positive, negative) embedding vectors.
            Positive and negative may be None if no valid candidates were found.
        """
        film_id = self._films_ids[index] # map DataLoader's index to film_id

        anchor = self._embeddings[film_id]  # get embedding vector for anchor

        # get positive and negative candidates for anchor
        positives, negatives = self._find_positives_and_negatives(film_id=film_id)

        # if some group is empty - skip the backward step
        if not positives or not negatives:
            return anchor, None, None

        # shuffle lists so we do not get the same movies every time 
        np.random.shuffle(positives)
        np.random.shuffle(negatives)

        # in case the semi-hard logic is not found use hard logic
        # condition : d_neg < d_pos
        best_hard_positive = None
        best_hard_negative = None

        # semi-hard logic: if the d_neg is bigger than d_pos but not by margin - return this case
        for positive_id in positives:
            d_pos = self._matrix_distances[film_id][positive_id]
            positive = self._embeddings[positive_id]

            for negative_id in negatives:
                negative = self._embeddings[negative_id]
                d_neg = self._matrix_distances[film_id][negative_id]

                # check if the semi-hard condition passed 
                if d_pos < d_neg and d_neg < d_pos + self._margin:
                    return anchor, positive, negative

                # hard logic, check witihin semi-hard logic not to run nested loop twice
                if d_neg < d_pos:
                    best_hard_negative = negative
                    best_hard_positive = positive

        if best_hard_positive is not None and best_hard_negative is not None:
            return anchor, best_hard_positive, best_hard_negative
        
        # easy logic: return random pos and random neg vectors
        # the weakest approach, becuase majority of negatives are already further than positives
        random_positive_id = np.random.choice(positives)
        random_negative_id = np.random.choice(negatives)

        random_positive = self._embeddings[random_positive_id]
        random_negative = self._embeddings[random_negative_id]

        return anchor, random_positive, random_negative


    def _find_positives_and_negatives(self, film_id: int) -> tuple[list[int], list[int]]:
        """
        Finds positive and negative candidate film IDs for a given anchor film.

        A film is considered a positive candidate if it shares 2 or more genres
        with the anchor — a stricter threshold than single genre overlap to ensure
        meaningful similarity signal. All other films are treated as negative candidates.

        Args:
            film_id: The anchor film ID to find candidates for.

        Returns:
            A tuple of (positives, negatives) where each is a list of film IDs.
            The anchor film itself is excluded from both lists.
        """
        positive_genres = set()

        # find which genres has input film_id 
        for genre, ids in self._genres.items():
            if film_id in ids:
                positive_genres.add(genre)

        num_of_genres_in_common = defaultdict(int)

        # create a dict of similiraties - increase if the curr film has genre in common with the input's one
        for genre, ids in self._genres.items():
            if genre in positive_genres:
                for id in ids:
                    num_of_genres_in_common[id] += 1

        positives = []
        negatives = []

        for id in self._films_ids:
            if id != film_id:
                # if film has 2 or more genres in common - it is positive
                if num_of_genres_in_common[id] >= 2:
                    positives.append(id)
                # otherwise - negative
                else:
                    negatives.append(id)

        return positives, negatives
