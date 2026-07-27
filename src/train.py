import torch
from torch.optim import AdamW

from torch.utils.data import DataLoader, Subset
from src.model.film_encoder import FilmEncoder
from src.data.dataset import FilmPairDataset, custom_collate_fn
from src.data.loader import(
    build_distance_matrix,
    group_by_genres,
    group_by_keywords,
    fetch_films_embeddings
)
from src.model.loss import TripletLoss
from src.main import PATH_TO_MODEL_WEIGHTS
from src.db import connection


EPOCHS = 20
MAX_PATIENCE_RATE = 4


def train():
    """
    Trains the projection head of the FilmEncoder model using Triplet Loss.

    Fetches precomputed embeddings, genres, keywords, and pairwise distance 
    matrices from PostgreSQL to construct a FilmPairDataset. Fine-tunes the 
    linear projection layer (384 -> 128) using semi-hard triplet sampling, 
    evaluating validation loss per epoch and saving model weights on improvement.
    """
    # init pool to get genres, keywords and embeddings from db
    connection.init_pool()

    model = FilmEncoder()   # uses model from HuggingFace + Linear Projection (384 -> 128)

    with connection.pool.connection() as conn:
        # fetch data needed to build a dataset
        genres: dict[str, list[int]] = group_by_genres(conn=conn)
        keywords: dict[str, list[int]] = group_by_keywords(conn=conn)
        embeddings: dict[int, list[float]] = fetch_films_embeddings(conn=conn)
        distance_matrix: dict[int, dict[int, float]] = build_distance_matrix(conn=conn)

    dataset = FilmPairDataset(
        matrix_distances=distance_matrix,
        genres=genres,
        keywords=keywords,
        embeddings=embeddings
    )

    # use 90% of data to train, 10% to validate
    training_dataset = Subset(dataset=dataset, indices=range(0, int(0.9 * len(dataset))))
    validation_dataset = Subset(dataset=dataset, indices=range(int(0.9 * len(dataset)), len(dataset)))

    training_dataloader = DataLoader(
        dataset=training_dataset, 
        shuffle=True, 
        batch_size=32, 
        collate_fn=custom_collate_fn    # use custom collate_fn method which deletes Nones
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset, 
        shuffle=True, 
        batch_size=16,
        collate_fn=custom_collate_fn
    )

    optimizer = AdamW(params=model.parameters(), lr=2e-5)
    triplet_loss = TripletLoss()
    best_loss = float("inf")
    patience_rate = 0

    for epoch in range(EPOCHS):
        total_train_loss = 0
        model.train()   # activate training mode to activate dropouts

        for anchors, positives, negatives in training_dataloader:
            # reset gradients before new batch
            optimizer.zero_grad()

            # move Tensors to GPU
            anchors = anchors.to(model._device)
            positives = positives.to(model._device)
            negatives = negatives.to(model._device)

            # make forward pass on vectors to make them 128 dim and to train Projection Head
            anchors = model(anchors)
            positives = model(positives)
            negatives = model(negatives)

            loss = triplet_loss(anchors, positives, negatives)
    
            # calculate gradinets and update weights
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
                

        mean_train_loss = total_train_loss / len(training_dataloader)

        model.eval()    # turn on dropped neurons
        total_val_loss = 0

        for anchors, positives, negatives in validation_dataloader:
            with torch.no_grad():
                # move Tensors to GPU
                anchors = anchors.to(model._device)
                positives = positives.to(model._device)
                negatives = negatives.to(model._device)

                anchors = model(anchors)
                positives = model(positives)
                negatives = model(negatives)
                loss = triplet_loss(anchors, positives, negatives)
                total_val_loss += loss.item()

        mean_val_loss = total_val_loss / len(validation_dataloader)
        print(f"Epoch {epoch + 1}, train_loss: {mean_train_loss}, val_loss: {mean_val_loss}")

        # if model becomes better on val data: reset patience rate and save weights of Projection Head
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            patience_rate = 0
            model.save(path=PATH_TO_MODEL_WEIGHTS)
        else:
            patience_rate += 1
            if patience_rate >= MAX_PATIENCE_RATE:
                break
                    
    connection.close_pool()


if __name__ == "__main__":
    train()
