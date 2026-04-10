from pathlib import Path
from utils.bq import get_client, query_to_df

SQL_DIR = Path(__file__).parent.parent / "sql_scripts"

client, PROJECT_NAME = get_client()
query = (SQL_DIR / "radiology_data.sql").read_text().format(PROJECT_NAME=PROJECT_NAME)
df_radiology = query_to_df(client, query)
