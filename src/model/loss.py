import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """
    Triplet loss function for training film embedding similarity.

    Pushes anchor embeddings closer to positive (similar) films and further
    from negative (dissimilar) films in vector space. Uses cosine distance
    as the distance metric, exploiting the fact that embeddings are normalized
    to unit length so cosine similarity reduces to a dot product.

    Loss formula: mean(max(d(anchor, positive) - d(anchor, negative) + margin, 0))

    If the positive is already closer than the negative by more than the margin,
    loss is 0 and no update occurs. Otherwise backprop adjusts weights to
    increase the gap.

    Args:
        margin: Minimum required distance gap between positive and negative pairs.
                Prevents the model from being lazy by requiring a meaningful separation.
                Defaults to 0.2, which is 10% of the full cosine distance range (0-2).
    """
    def __init__(self, margin: float = 0.2):
        """
        Initializes TripletLoss with the given margin.

        Args:
            margin: Minimum required distance gap between positive and negative pairs.
        """
        super().__init__()
        self._margin = margin

    def forward(
            self, 
            anchors: torch.Tensor, 
            positives: torch.Tensor, 
            negatives: torch.Tensor
            ) -> torch.Tensor:
        """
        Computes mean triplet loss over a batch of embedding triplets.

        Assumes all input embeddings are L2-normalized to unit length so that
        cosine similarity equals the dot product. Cosine distance is then
        derived as 1 - cosine_similarity.

        Args:
            anchors: Batch of anchor film embeddings of shape (batch_size, embedding_dim).
            positives: Batch of positive film embeddings of shape (batch_size, embedding_dim).
            negatives: Batch of negative film embeddings of shape (batch_size, embedding_dim).

        Returns:
            A scalar tensor representing the mean triplet loss across the batch.
        """
        batch, _ = anchors.shape
        device = anchors.device

        zero = torch.zeros(size=(batch,), device=device)  # for max()

        # compute cosine similarity via dot product
        # it works cause vectors are normalized, their lens = 1 
        pos_cosine_similarity = anchors * positives
        neg_cosine_similarity = anchors * negatives

        pos_cosine_similarity = pos_cosine_similarity.sum(dim=-1)
        neg_cosine_similarity = neg_cosine_similarity.sum(dim=-1)

        # cosine distance = 1 - cosine similarity
        positive_cosine_distances = 1 - pos_cosine_similarity
        negative_cosine_distances = 1 - neg_cosine_similarity

        # loss = max(d_pos - d_neg + margin, 0)
        # we want positive vector be closer to anchor at least by margin 
        loss = torch.max(
            positive_cosine_distances - negative_cosine_distances + self._margin,
            zero
        )

        # return mean across batch
        return loss.mean()
