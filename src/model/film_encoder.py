import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class FilmEncoder(nn.Module):
    """
    A fine-tunable film embedding model with a learned projection head.

    Wraps a pretrained SentenceTransformer to generate 384-dimensional text embeddings,
    then projects them to a specialized 128-dimensional space trained with triplet loss.
    The projection head discards general language structure and retains only the
    semantic dimensions relevant to film similarity.

    During training, forward() receives pre-computed 384-dim vectors from the database
    and passes them through the projection head only — the sentence transformer is bypassed
    for efficiency. After training, generate_embeddings() runs the full pipeline from
    raw text to normalized 128-dim vectors for storage in Postgres.

    Args:
        model_name: HuggingFace model identifier for the base sentence transformer.
                    Defaults to all-MiniLM-L6-v2.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()

        self._device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        # load pretrained model
        self._model = SentenceTransformer(model_name_or_path=model_name, device=self._device)
        # add Layer for projection to get rid of general language meaning
        # and save only films' semantic meaning
        self._projection = nn.Linear(in_features=384, out_features=128)
        self._dropout = nn.Dropout(p=0.25)  # kill 25% of 384 dim vector input
        self.to(device=self._device)    # put whole model to GPU cause projection defaults to CPU

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Projects pre-computed embeddings to 128-dimensional film similarity space.

        Expects L2-normalized 384-dim vectors from the DataLoader. Applies the
        linear projection and re-normalizes the output so cosine similarity
        equals the dot product, which is required by TripletLoss.

        Args:
            embeddings: Batch of 384-dim embedding tensors of shape (batch_size, 384).

        Returns:
            L2-normalized 128-dim tensors of shape (batch_size, 128).
        """
        embeddings = self._dropout(embeddings)
        embeddings = self._projection(embeddings)   # project vectors from 384 dim space to 128 one
        # normalize vectors so cosine similarity = dot product 
        return nn.functional.normalize(embeddings, p=2, dim=1)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generates 128-dimensional film embeddings from raw text descriptions.

        Runs the full pipeline: encodes text with the pretrained sentence transformer,
        projects to 128 dimensions, and L2-normalizes the result. Used after training
        to re-embed all films and store results in the embedding_finetuned Postgres column.
        Runs under torch.no_grad() since no gradient tracking is needed for inference.

        Args:
            texts: List of raw film description strings to embed.

        Returns:
            List of L2-normalized 128-dimensional float vectors, one per input text.
        """
        with torch.no_grad():
            # get 384 dim vectors
            vectors_before_projection = self._model.encode(texts, batch_size=32, show_progress_bar=True)
            vectors_before_projection = torch.tensor(data=vectors_before_projection, device=self._device)

            # project them to 128 dim space and normalize them 
            vectors_after_projection = self._projection(vectors_before_projection)
            vectors_after_projection = nn.functional.normalize(vectors_after_projection, p=2, dim=1)

        return vectors_after_projection.tolist()

    def save(self, path: str) -> None:
        """
        Saves the projection head weights to disk.

        Only the projection layer state is saved — the pretrained sentence transformer
        weights are not included since they are always loadable from HuggingFace.

        Args:
            path: File path where the checkpoint will be saved, e.g. './models/weights.pt'.
        """
        torch.save({
        "projection_state_dict": self._projection.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str) -> "FilmEncoder":
        """
        Loads a FilmEncoder from a saved projection head checkpoint.

        Constructs a fresh FilmEncoder instance, loads the pretrained sentence
        transformer, then restores the trained projection weights from disk.

        Args:
            path: File path to the saved checkpoint produced by save().

        Returns:
            A fully initialized FilmEncoder with trained projection weights loaded.
        """
        new_instance = FilmEncoder()
        checkpoint = torch.load(
        path,
        map_location=new_instance._device,
        weights_only=True
        )

        new_instance._projection.load_state_dict(
            checkpoint["projection_state_dict"]
        )

        return new_instance
