import argparse
import os
import pandas as pd
from sqlalchemy import create_engine
from time import time

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def main(args):
    # --- parse argument
    user = args.user
    password = args.password
    host = args.host
    port = args.port
    db = args.db
    table_name = args.table_name
    url = args.url

    # --- get parquet file from url
    print("Reading parquet file")
    csv_name = "output.csv"
    df = pd.read_parquet(url)
    df.to_csv(csv_name, index=False)
    #  os.system(f"wget {url} -O {csv_name}")

    # --- create db connection
    try:
        engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")
        connection = engine.connect()
        print("Connected to the database successfully")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return

    # --- iteratively push batches of data to database
    df_iter = pd.read_csv(csv_name, iterator=True, chunksize=100000)
    df = next(df_iter)

    # --- convert datetime columns to datetime
    for col in df.columns:
        if col.endswith("datetime"):
            df[col] = pd.to_datetime(df[col])

    df.head(n=0).to_sql(name=table_name, con=connection, if_exists="replace")
    df.to_sql(name=table_name, con=connection, if_exists="append")

    while True:
        try:
            t_start = time()

            df = next(df_iter)
            df.to_sql(name=table_name, con=engine, if_exists="append")

            t_end = time()
            print("inserted another chunk, took %.3f second" % (t_end - t_start))
        except StopIteration:
            print("Finished ingesting data into the postgres database")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest parquet file to postgres")

    parser.add_argument("--user", required=True, help="username for postgres")
    parser.add_argument("--password", required=True, help="password for postgres")
    parser.add_argument("--host", required=True, help="host for postgres")
    parser.add_argument("--port", required=True, help="port for postgres")
    parser.add_argument("--db", required=True, help="database for postgres")
    parser.add_argument("--table_name", required=True, help="table name for postgres")
    parser.add_argument("--url", required=True, help="URL of the parquet file")

    args = parser.parse_args()
    main(args)
