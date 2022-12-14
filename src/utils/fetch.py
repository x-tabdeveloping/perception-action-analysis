from typing import Dict

import pandas as pd
import firebase_admin
from firebase_admin import firestore

app = firebase_admin.initialize_app()
database = firestore.client()


def fetch_all_docs() -> Dict:
    """Fetches all records from the experimentData collection in the database.

    Returns
    -------
    Dict
        Mapping from document ids to documents in the collection.
    """
    docs = database.collection("experimentData").stream()
    return {doc.id: doc.to_dict() for doc in docs}


def docs_to_df(docs: Dict) -> pd.DataFrame:
    """Turns a collection of documents to a DataFrame"""
    records = []
    for doc_id, doc in docs.items():
        participant = doc["participant"]
        results = doc["results"]
        for result in results:
            record = {"doc_id": doc_id, **participant, **result}
            records.append(record)
    return pd.DataFrame.from_records(records)


def clean_column_names(experiment_data: pd.DataFrame) -> pd.DataFrame:
    """Pythonizes column names in the given experiment data frame."""
    return experiment_data.rename(
        columns={
            "screenWidth": "screen_width",
            "screenHeight": "screen_height",
            "mouseTrackingData": "mouse_tracking_data",
            "elapsedTime": "reaction_time",
        }
    )


def fetch_experiment_data() -> pd.DataFrame:
    """Fetches all documents from the experimentData collection in the
    form of a DataFrame"""
    docs = fetch_all_docs()
    df = docs_to_df(docs)
    return clean_column_names(df)
