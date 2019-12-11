from sqlalchemy import create_engine, MetaData
import os


def get_connection_string(environment_dict, no_db=False):
    connection_string = (
        f"{environment_dict.get('DBMS')}+{environment_dict.get('DB_DRIVER')}://"
        f"{environment_dict.get('DB_USER')}:{environment_dict.get('DB_PASSWORD')}"
        f"@{environment_dict.get('DB_HOST')}:{environment_dict.get('DB_PORT')}"
    )
    if not no_db:
        connection_string = "/".join(
            [connection_string, environment_dict.get("DB_NAME")]
        )
    return connection_string


def db_setup_core():
    con_str = get_connection_string(os.environ)
    engine = create_engine(con_str)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    return engine, metadata
