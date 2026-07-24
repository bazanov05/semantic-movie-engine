import pytest
import numpy as np
from src.data.dataset import FilmPairDataset 


@pytest.fixture
def mock_dataset_data():
    """
    Provides a controlled set of data to test all triplet mining conditions.
    Film 1 is our anchor for most tests. We test the new scoring logic:
    Score = (2 * shared_genres) + (1 * shared_keywords).
    
    - Film 1 & 2: share 'Action', 'Comedy' (2 genres) + 'space' (1 keyword). Score = 5 (Positive)
    - Film 1 & 3: share 'Action' (1 genre) + 'alien' (1 keyword). Score = 3 (Negative)
    - Film 1 & 4: share 'alien' (1 keyword). Score = 1 (Negative)
    """
    genres = {
        "Action": [1, 2, 3],
        "Comedy": [1, 2],
        "Drama": [3, 4]
    }
    
    keywords = {
        "space": [1, 2],
        "alien": [1, 3, 4]
    }
    
    embeddings = {
        1: [1.0], 
        2: [2.0], 
        3: [3.0], 
        4: [4.0]
    }
    
    # Default uniform distances to be overwritten in specific tests
    matrix_distances = {
        1: {2: 0.5, 3: 0.5, 4: 0.5},
        2: {1: 0.5, 3: 0.5, 4: 0.5},
        3: {1: 0.5, 2: 0.5, 4: 0.5},
        4: {1: 0.5, 2: 0.5, 3: 0.5},
    }
    
    return {
        "matrix_distances": matrix_distances,
        "genres": genres,
        "keywords": keywords,
        "embeddings": embeddings,
        "margin": 0.2
    }


def test_dataset_len(mock_dataset_data):
    """Tests if the dataset returns the correct total number of films."""
    dataset = FilmPairDataset(**mock_dataset_data)
    assert len(dataset) == 4


def test_precomputed_candidates(mock_dataset_data):
    """Tests if the similarity scoring and candidate pre-computation correctly divide films."""
    dataset = FilmPairDataset(**mock_dataset_data)
    
    # Check Film 1's precomputed lists
    positives = dataset._positives[1]
    negatives = dataset._negatives[1]
    
    # Film 2 has a score of 5 (>= 4), so it is Positive.
    # Films 3 (score 3) and 4 (score 1) are Negative.
    assert positives == [2]
    assert sorted(negatives) == [3, 4]


def test_getitem_semi_hard_logic(mock_dataset_data):
    """Tests the optimal tier: finding a negative that is further than positive, but within margin"""
    # Setup Semi-Hard: d_pos < d_neg < d_pos + margin
    mock_dataset_data["matrix_distances"][1][2] = 0.3  # Positive distance
    mock_dataset_data["matrix_distances"][1][3] = 0.4  # Negative distance (Semi-hard)
    mock_dataset_data["matrix_distances"][1][4] = 0.9  # Negative distance (Easy)
    
    dataset = FilmPairDataset(**mock_dataset_data)
    
    # Index 0 maps to film_id 1
    anchor, pos, neg = dataset[0]
    
    assert anchor == [1.0]
    assert pos == [2.0]
    # Should select Film 3 because it satisfies the semi-hard condition
    assert neg == [3.0]


def test_getitem_hard_fallback(mock_dataset_data):
    """Tests the secondary tier: falling back to a hard triplet if no semi-hard exists."""
    # Setup Hard: d_neg < d_pos
    mock_dataset_data["matrix_distances"][1][2] = 0.6  # Positive distance
    mock_dataset_data["matrix_distances"][1][3] = 0.4  # Negative distance (Hard)
    mock_dataset_data["matrix_distances"][1][4] = 0.9  # Negative distance (Easy)
    
    dataset = FilmPairDataset(**mock_dataset_data)
    anchor, pos, neg = dataset[0]
    
    assert anchor == [1.0]
    assert pos == [2.0]
    # Should select Film 3 because the semi-hard loop failed and the hard condition passed
    assert neg == [3.0]


def test_getitem_easy_random_fallback(mock_dataset_data):
    """Tests the last resort tier: falling back to random samples when no hard/semi-hard exists."""
    # Setup Easy: d_neg > d_pos + margin
    mock_dataset_data["matrix_distances"][1][2] = 0.1  # Positive distance
    mock_dataset_data["matrix_distances"][1][3] = 0.8  # Negative distance (Easy)
    mock_dataset_data["matrix_distances"][1][4] = 0.9  # Negative distance (Easy)
    
    dataset = FilmPairDataset(**mock_dataset_data)
    
    # Fix the random seed so we know exactly which easy negative it randomly picks
    np.random.seed(42)
    anchor, pos, neg = dataset[0]
    
    assert anchor == [1.0]
    assert pos == [2.0]
    # Must be one of the negatives since it relies on random.choice
    assert neg in ([3.0], [4.0])


def test_getitem_edge_case_empty_lists(mock_dataset_data):
    """Tests the edge case where a film has no valid positive matches."""
    # Modify genres and keywords so Film 1 shares a maximum of 1 genre and 0 keywords with anyone
    mock_dataset_data["genres"] = {
        "Action": [1, 2],
        "Comedy": [3, 4],
    }
    mock_dataset_data["keywords"] = {
        "space": [2, 3],
        "alien": [4]
    }
    
    dataset = FilmPairDataset(**mock_dataset_data)
    
    # Film 1 and Film 2 share only 1 genre and 0 keywords (Score = 2). 
    # Because Film 1 has no positives (score >= 4), the early return should trigger.
    anchor, pos, neg = dataset[0]
    
    assert anchor == [1.0]
    assert pos is None
    assert neg is None
