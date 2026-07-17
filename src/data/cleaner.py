import pandas as pd


def clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads, filters, and sanitizes movie dataset CSV file for database ingestion.

    Extracts essential metadata columns, renames keys to align with the SQL schema, 
    and handles missing values (e.g., converting missing dates to None for native 
    SQL NULL compatibility).

    Args:
        file_path (str): The path to the target .csv file.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing the sanitized movie records.

    Raises:
        ValueError: If the provided file path does not end with '.csv'.
    """
    if not file_path.endswith(".csv"):
        raise ValueError(".csv file should be loaded to Pandas DataFrame")
    
    df = pd.read_csv(filepath_or_buffer=file_path)  # read data from .csv file 
    # get rid of unnecessary columns
    df = df[
        [
        "id",
        "title",
        "genres",
        "keywords",
        "overview",
        "release_date",
        "vote_average"
        ]
    ]

    # rename "id" column so it matches SQL schema
    df = df.rename(columns={
        "id": "film_id"
    })

    # clean data - fill empty or destroyed data
    df["overview"] = df["overview"].fillna("")
    df["release_date"] = df["release_date"].where(df["release_date"].notna(), None)
    df["vote_average"] = df["vote_average"].fillna(0.0)
    df["keywords"] = df["keywords"].fillna("[]")
    df["genres"] = df["genres"].fillna("[]")
    
    return df
