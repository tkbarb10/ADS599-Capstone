import os
import warnings

from dotenv import load_dotenv
from google.cloud import bigquery

warnings.filterwarnings(action='ignore', message="Your application has authenticated using end user")


def get_client():
    """Load env vars and return a (client, project_name) tuple."""
    load_dotenv()
    project = os.environ.get("PROJECT_NAME")
    if not project:
        raise ValueError("PROJECT_NAME not set in .env")
    client = bigquery.Client(project=project)
    return client, project


def query_to_df(client, query):
    """Run a BigQuery query and return results as a pandas DataFrame."""
    return client.query(query).to_dataframe()