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
            keywords: dict[str, list[int]],
            embeddings: dict[int, list[float]],
            margin: float = 0.2
        ):
        super().__init__()

        self._matrix_distances = matrix_distances
        self._genres = genres
        self._keywords = keywords
        self._embeddings = embeddings

        # distance between pos and anchor should be smaller than distance between neg and anchor
        # at least by margin
        self._margin = margin

        # DataLoader will give indices in range [0, 4802], cause we have 4803 films
        # but film_ids are not in this range - there are gaps
        # so we need to map those indices to real film_ids
        self._films_ids = list(embeddings.keys())

        similarity_scores = self._compute_similarity_scores()
        self._positives, self._negatives = self._build_candidate_lists(scores=similarity_scores)

    def _compute_similarity_scores(self) -> dict[int, dict[int, int]]:
        """
        Precomputes weighted similarity scores for every pair of films.

        Builds inverted indexes mapping each film to its genre and keyword sets,
        then computes a weighted score for every pair using set intersection.
        Genres are weighted higher than keywords as they are a stronger categorical signal.

        Score formula: (2 * shared_genres) + (1 * shared_keywords)

        Returns:
            A nested dictionary mapping each film_id to a dict of all other film_ids
            and their similarity scores. Example: {1: {2: 5, 3: 1, ...}, ...}
        """
        # for each film_id find it's genres, build a dict - film_id: set of genres
        film_genres: dict[int, set[str]] = defaultdict(set)

        for genre, ids in self._genres.items():
            for film_id in ids:
                film_genres[film_id].add(genre)

        # for each film find it's keywords
        film_keywords: dict[int, set[str]] = defaultdict(set)

        for keyword, ids in self._keywords.items():
            for film_id in ids:
                film_keywords[film_id].add(keyword)

        # create dict of scores for each pair of ids
        scores: dict[int, dict[int, float]] = {}

        for id1 in self._films_ids:
            scores[id1] = {}
            for id2 in self._films_ids:
                if id1 != id2:
                    # find how many genres and keywords those films have in common
                    genre_similariy = len(film_genres[id1] & film_genres[id2])
                    keyword_similarity = len(film_keywords[id1] & film_keywords[id2])

                    # calculate the weighted score, genre has weight 2, keyword 1
                    score = genre_similariy * 2 + keyword_similarity
                    scores[id1][id2] = score

        return scores

    def _build_candidate_lists(
            self, 
            scores: dict[int, dict[int, int]]
        ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """
        Splits all film pairs into positive and negative candidate lists based on similarity score.

        A film is a positive candidate if its weighted similarity score with the anchor
        is 4 or above — equivalent to sharing 2 genres, or 1 genre and 2 keywords.
        All other films are treated as negative candidates.

        Args:
            scores: Precomputed similarity scores from _compute_similarity_scores.

        Returns:
            A tuple of (positives, negatives), each a dict mapping film_id to a list
            of candidate film_ids. Example: {1: [5, 23, 104], ...}
        """
        # build positive and negative candidates
        positives = defaultdict(list)
        negatives = defaultdict(list)

        for id1 in self._films_ids:
            for id2 in self._films_ids:
                if id1 != id2:
                    score = scores[id1][id2]
                    # if score >= 4 - this candidate is considered positive
                    if score >= 4:
                        positives[id1].append(id2)
                    # otherwise - negative 
                    else:
                        negatives[id1].append(id2)

        return positives, negatives

    def _sample_triplet(self, index: int) -> tuple[list[float], list[float] | None, list[float] | None]:
        """
        Samples a triplet of embeddings for the film at the given index.

        Applies a sampling hierarchy to find the most informative triplet:
        1. Semi-hard: d_pos < d_neg < d_pos + margin. Best learning signal.
        2. Hard: d_neg < d_pos. Strong signal, used as fallback.
        3. Easy: random positive and negative. Weakest signal, last resort.

        Args:
            index: DataLoader's sequential index mapped to a real film_id.

        Returns:
            A tuple of (anchor, positive, negative) embedding vectors.
            Positive and negative are None if no valid candidates exist.
        """
        film_id = self._films_ids[index]    # map index to film_id 
        anchor = self._embeddings[film_id]

        # get positive and negative candidates for anchor
        positives = self._positives[film_id]
        negatives = self._negatives[film_id]

        # if anchor is lack of some candidates - return Nones
        if not positives or not negatives:
            return anchor, None, None

        # shuffle ids
        np.random.shuffle(positives)
        np.random.shuffle(negatives)

        # track best case for hard logic in case semi-hard is not found
        best_hard_pos = None
        best_hard_neg = None

        for positive_id in positives:
            positive = self._embeddings[positive_id]
            for negative_id in negatives:
                negative = self._embeddings[negative_id]

                d_pos = self._matrix_distances[film_id][positive_id]
                d_neg = self._matrix_distances[film_id][negative_id]

                # semi-hard logic: d_pos is smaller than d_neg but not by margin
                if d_pos < d_neg and d_neg < d_pos + self._margin:
                    return anchor, positive, negative

                # hard logic: d_pos > d_neg
                if d_neg < d_pos:
                    best_hard_neg = negative
                    best_hard_pos = positive

        # return hard vectors in case semi-hard was not found and hard was found
        if best_hard_pos is not None and best_hard_neg is not None:
            return anchor, best_hard_pos, best_hard_neg

        # otherwise, use easy logic - return random positive and random negative
        random_pos_id = np.random.choice(positives)
        random_neg_id = np.random.choice(negatives)

        positive = self._embeddings[random_pos_id]
        negative = self._embeddings[random_neg_id]

        return anchor, positive, negative

    def __len__(self):
        """Returns the total number of films in the dataset."""
        return len(self._embeddings)

    def __getitem__(self, index) -> tuple[list[float], list[float] | None, list[float] | None]:
        """
        Returns a sampled triplet for the given index.

        Delegates to _sample_triplet. Called by DataLoader on every batch iteration.

        Args:
            index: Sequential index provided by DataLoader in range [0, len(dataset) - 1].

        Returns:
            A tuple of (anchor, positive, negative) embedding vectors.
        """
        return self._sample_triplet(index=index)

    