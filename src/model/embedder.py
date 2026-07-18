from sentence_transformers import SentenceTransformer
import torch


class MovieEmbedder:
    """
    A wrapper around a SentenceTransformer model for generating film embeddings.

    Downloads and caches the model on first use, then keeps it in memory
    for the lifetime of the instance. Automatically uses GPU (CUDA or MPS)
    if available, otherwise falls back to CPU.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder and load the model onto the best available device.

        Args:
            model_name: HuggingFace model identifier. Defaults to all-MiniLM-L6-v2
                        which produces 384-dimensional embeddings.
        """
        self._device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name_or_path=model_name, device=self._device)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text strings into 384-dimensional float vectors.

        Processes the input texts in memory-efficient batches of 32 to prevent
        high RAM usage, displaying a progress bar during generation. Returns
        plain Python lists so psycopg can seamlessly convert and insert them 
        into the Postgres vector column.

        Args:
            texts: List of strings to embed.

        Returns:
            List of 384-dimensional float arrays, one per input text.
        """
        # encode() processes chunks sequentially under the hood but returns
        # one unified numpy array, which we cast to a list for psycopg compatibility.
        embeddings_ndarray = self._model.encode(texts, batch_size=32, show_progress_bar=True)
        return embeddings_ndarray.tolist()
    