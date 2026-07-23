import pytest
import numpy as np
from src.data.dataset import FilmPairDataset 


@pytest.fixture
def mock_dataset_data():
    """
    Provides a controlled set of data to test all triplet mining conditions.
    Film 1 is our anchor for most tests:
    - Shares 'Action' and 'Comedy' with Film 2 (2 genres -> Positive).
    - Shares only 'Action' with Film 3 (1 genre -> Negative).
    - Shares only 'Comedy' with Film 4 (1 genre -> Negative).
    """
    genres = {
        "Action": [1, 2, 3],
        "Comedy": [1, 2, 4],
        "Drama": [3, 4]
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
        "embeddings": embeddings,
        "margin": 0.2
    }


def test_dataset_len(mock_dataset_data):
    """Tests if the dataset returns the correct total number of films."""
    dataset = FilmPairDataset(**mock_dataset_data)
    assert len(dataset) == 4


def test_find_positives_and_negatives(mock_dataset_data):
    """Tests if the positive and negative grouping logic strictly enforces the >= 2 genres rule."""
    dataset = FilmPairDataset(**mock_dataset_data)
    
    pos, neg = dataset._find_positives_and_negatives(film_id=1)
    
    # Film 2 shares >= 2 genres (Positive), Films 3 and 4 share < 2 genres (Negative)
    assert pos == [2]
    assert sorted(neg) == [3, 4]
    
    # The anchor film itself must be excluded from both lists
    assert 1 not in pos
    assert 1 not in neg


def test_getitem_semi_hard_logic(mock_dataset_data):
    """Tests the optimal tier: finding a negative that is further than positive, but within margin."""
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
    """Tests the edge case where a film has no valid positive matches[cite: 8]."""
    # Modify genres so Film 1 shares NO multiple genres with anyone
    mock_dataset_data["genres"] = {
        "Action": [1, 2],
        "Comedy": [1, 3],
        "Drama": [2, 3]
    }
    
    dataset = FilmPairDataset(**mock_dataset_data)
    
    # Because Film 1 has no positives, the early return should trigger returning None for candidates
    anchor, pos, neg = dataset[0]
    
    assert anchor == [1.0]
    assert pos is None
    assert neg is None
