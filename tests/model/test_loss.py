from src.model.loss import TripletLoss
import pytest
import torch


@pytest.fixture
def triplet_loss():
    """Provides a TripletLoss instance with the default margin of 0.2."""
    return TripletLoss(margin=0.2)


def test_perfect_triplet_zero_loss(triplet_loss):
    """
    Base Case: The positive is perfectly identical to the anchor, 
    and the negative is completely opposite.
    d_pos = 0.0, d_neg = 2.0. Loss should be max(0 - 2 + 0.2, 0) = 0.0.
    """
    anchors = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[1.0, 0.0]])   # Identical to anchor
    negatives = torch.tensor([[-1.0, 0.0]])  # Opposite to anchor
    
    loss = triplet_loss(anchors, positives, negatives)
    
    assert torch.allclose(loss, torch.tensor(0.0))


def test_worst_triplet_max_loss(triplet_loss):
    """
    Base Case: The positive is completely opposite, 
    and the negative is identical to the anchor.
    d_pos = 2.0, d_neg = 0.0. Loss should be max(2 - 0 + 0.2, 0) = 2.2.
    """
    anchors = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[-1.0, 0.0]])  # Opposite to anchor
    negatives = torch.tensor([[1.0, 0.0]])   # Identical to anchor
    
    loss = triplet_loss(anchors, positives, negatives)
    
    assert torch.allclose(loss, torch.tensor(2.2))


def test_equal_distance_margin_loss(triplet_loss):
    """
    Base Case: Both positive and negative are equally distant from the anchor.
    d_pos = 1.0, d_neg = 1.0. Difference is 0. Loss should be exactly the margin (0.2).
    """
    anchors = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[0.0, 1.0]])   # Orthogonal (d_pos = 1.0)
    negatives = torch.tensor([[0.0, 1.0]])   # Orthogonal (d_neg = 1.0)
    
    loss = triplet_loss(anchors, positives, negatives)
    
    assert torch.allclose(loss, torch.tensor(0.2))


def test_batch_processing_correctness(triplet_loss):
    """
    Edge Case: Ensures the loss correctly averages across a batch of multiple triplets.
    Batch 1: Perfect triplet (Loss = 0.0)
    Batch 2: Equal distance (Loss = 0.2)
    Mean Loss should be 0.1.
    """
    anchors = torch.tensor([
        [1.0, 0.0], 
        [1.0, 0.0]
    ])
    positives = torch.tensor([
        [1.0, 0.0],  # Perfect
        [0.0, 1.0]   # Equal
    ])
    negatives = torch.tensor([
        [-1.0, 0.0], # Perfect
        [0.0, 1.0]   # Equal
    ])
    
    loss = triplet_loss(anchors, positives, negatives)
    
    assert torch.allclose(loss, torch.tensor(0.1))


def test_device_agnostic(triplet_loss):
    """
    Edge Case: Tests if the loss function properly initializes internal tensors 
    on the same device as the input tensors (e.g., GPU/CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    anchors = torch.tensor([[1.0, 0.0]], device=device)
    positives = torch.tensor([[1.0, 0.0]], device=device)
    negatives = torch.tensor([[-1.0, 0.0]], device=device)
    
    # Should not throw a device mismatch runtime error
    loss = triplet_loss(anchors, positives, negatives)
    
    assert loss.device == device


def test_custom_margin():
    """
    Base Case: Ensures the loss respects custom margin configurations.
    """
    custom_loss = TripletLoss(margin=0.5)
    
    anchors = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[0.0, 1.0]])   # d_pos = 1.0
    negatives = torch.tensor([[0.0, 1.0]])   # d_neg = 1.0
    
    # Loss should be exactly the custom margin (0.5)
    loss = custom_loss(anchors, positives, negatives)
    
    assert torch.allclose(loss, torch.tensor(0.5))
