from utils.bq import get_client, query_to_df
from utils.load_yaml_helper import get_sql_dir

SQL_DIR = get_sql_dir()

client, PROJECT_NAME = get_client()
query = (SQL_DIR / "vitals.sql").read_text().format(PROJECT_NAME=PROJECT_NAME)
df_vitals = query_to_df(client, query)
