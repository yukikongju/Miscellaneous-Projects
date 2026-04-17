"""
Functions to fetch data from Appsflyer
"""

import requests
import logging
from typing import List, Optional
from date_utils import chunk_date_range


def fetch_appsflyer_master_api(
    token: str,
    #  app_id: str,
    start_date: str,
    end_date: str,
    groupings: Optional[List[str]],
    kpis: List[str],
):
    """
    Endpoint: https://hq1.appsflyer.com/api/master-agg-data/v4/app/all
    """
    base_url = "https://hq1.appsflyer.com/api/master-agg-data/v4/app/all"

    headers = {"accept": "application/json", "authorization": f"Bearer {token}"}

    date_chunks = chunk_date_range(
        start_date=start_date,
        end_date=end_date,
        max_chunk_day=30,
        date_format="%Y-%m-%d",
    )
    groupings_flatten = ",".join(groupings) if groupings is not None else ""
    kpis_flatten = ",".join(kpis)

    data = []
    for s_date, e_date in date_chunks:
        url = f"{base_url}?from={s_date}&to={e_date}&groupings={groupings_flatten}&kpis={kpis_flatten}&format=json"

        try:
            response = requests.get(url, headers=headers)
            logging.info(f"[STATUS CODE] {response.status_code}")
            if response.status_code == 200:
                data += response.json()
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch data from Appsflyer Master API. Exited with response {e}"
            )

    return data
