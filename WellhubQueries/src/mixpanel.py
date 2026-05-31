import ast
import base64
import json
import os
import sys
from typing import List

import pandas as pd
import requests
from dotenv import load_dotenv

from constants import COUNTRY_CODE_MAP

load_dotenv()

FUNNEL_ID = 90364672
PROJECT_ID = int(os.getenv("MIXPANEL_PROJECT_ID", "0"))


def _parse_funnel_json(raw: dict) -> pd.DataFrame:
    """Parse Mixpanel funnel JSON into a flat DataFrame.

    JSON structure: data[date][date][country_name][user_id] -> [step0, step1]
      - step0: screen_home
      - step1: create_record_wellhub (count > 0 means user triggered it)

    Returns columns: date, country_code, user_id, create_record_wellhub
    """
    records = []
    for date, date_data in raw["data"].items():
        inner = date_data.get(date, {})
        for country_name, country_data in inner.items():
            if country_name == "$overall":
                continue
            country_code = COUNTRY_CODE_MAP.get(country_name, country_name[:2].upper())
            for user_id, funnel_steps in country_data.items():
                if user_id in ("$overall", "undefined"):
                    continue
                if not isinstance(funnel_steps, list) or len(funnel_steps) < 2:
                    continue
                create_record = 1 if funnel_steps[1].get("count", 0) > 0 else 0
                records.append(
                    {
                        "date": date,
                        "country_code": country_code,
                        "user_id": user_id,
                        "create_record_wellhub": create_record,
                    }
                )

    df = pd.DataFrame(records, columns=["date", "country_code", "user_id", "create_record_wellhub"])
    return df.drop_duplicates(subset=["date", "user_id"]).reset_index(drop=True)


def get_mixpanel_funnel_dataframe_from_json(data):
    ## get breakdown columns name
    levels_types_map = {}
    for meta in data["meta"]["group_by_metadata"]:
        unpack = list(meta["bookmark"].values())[0]
        property_name = unpack["propertyName"]
        property_type = unpack["propertyType"]
        levels_types_map[property_name] = property_type
    # num_levels = len(levels_types_map)
    column_names = list(levels_types_map.keys())

    ## flatten json
    df = pd.json_normalize(data["data"])
    df = df.drop(columns=[c for c in df.columns if "$overall" in c])
    df = df.melt(var_name="key", value_name="value")
    df[column_names] = df["key"].str.split(".", expand=True).iloc[:, 1:]
    df["count"] = df["value"].apply(
        lambda x: ast.literal_eval(x)[0]["count"] if isinstance(x, str) else x[0]["count"]
    )
    df = df.drop(columns=["key", "value"])
    return df


def load_funnel_data_from_path(filepath: str) -> pd.DataFrame:
    """Load and parse a locally saved Mixpanel funnel JSON file."""
    with open(filepath) as f:
        raw = json.load(f)
    return _parse_funnel_json(raw)


def load_funnel_data_from_mixpanel(from_date: str, to_date: str) -> pd.DataFrame:
    """Fetch funnel data from the Mixpanel API and parse it.

    Reads MIXPANEL_USERNAME and MIXPANEL_PASSWORD from .env.
    Date format: YYYY-MM-DD
    """
    mp = MixpanelAPI(
        project_id=PROJECT_ID,
        username=os.getenv("MIXPANEL_USERNAME"),
        password=os.getenv("MIXPANEL_PASSWORD"),
    )
    raw = mp.query_funnel(FUNNEL_ID, from_date=from_date, to_date=to_date)
    return _parse_funnel_json(raw)


class MixpanelAPI:
    """
    EXAMPLE:

    MP = MixpanelAPI(project_id=2481461, username=..., password=...)
    data = MP.query_funnel(90364672, "2026-04-01", "2026-04-30")
    """

    def __init__(self, project_id: int, username: str, password: str):
        self.project_id = project_id
        self.username = username
        self.password = password
        auth_string = f"{self.username}:{self.password}"
        auth_base64 = base64.b64encode(auth_string.encode()).decode()
        self.headers = {
            "accept": "application/json",
            "authorization": f"Basic {auth_base64}",
            "Content-Type": "application/json",
        }

    def query_insights(self, insights_id: int, response_format: str):
        """
        The ID of your Insights report can be found from the url:
        https://mixpanel.com/project/<YOUR_PROJECT_ID>/view/<YOUR_WORKSPACE_ID>/app/boards#id=12345&editor-card-id=%22report-<YOUR_INSIGHTS_ID>%22

        Choose between "json" or "csv" as the return type from the query
        """
        if response_format not in {"json", "csv"}:
            raise ValueError("response_format must be 'json' or 'csv'")

        url = f"https://mixpanel.com/api/query/insights?project_id={self.project_id}&bookmark_id={insights_id}"
        request_body = {"queryLimits": {"limit": 50000}, "format": response_format}
        response = requests.post(url, headers=self.headers, data=json.dumps(request_body))

        print(f"query_insights status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response code: {response.status_code}. Terminating Process")
            sys.exit()

        if response_format == "csv":
            return response.text
        return json.loads(response.text)

    def query_funnel(
        self, funnel_id: int, from_date: str, to_date: str, response_format: str = "csv"
    ):
        """
        Fetch funnel data broken down by date, country, and user.
        Date format: YYYY-MM-DD (both inclusive).
        """
        url = (
            f"https://mixpanel.com/api/query/funnels"
            f"?project_id={self.project_id}"
            f"&funnel_id={funnel_id}"
            f"&from_date={from_date}"
            f"&to_date={to_date}"
            f"&unit=day"
            # f"&on=properties%5B%22%24country_code%22%5D"
            # f"&group_by_user_id=true"
        )
        response = requests.get(url, headers=self.headers)

        print(f"query_funnel status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response code: {response.status_code}. Terminating Process")
            sys.exit()

        return json.loads(response.text)

    def trigger_sync(self, put_url: str):
        response = requests.put(put_url, headers=self.headers)
        print(f"trigger_sync status: {response.status_code}")
        if response.status_code not in (200, 204):
            print("Failed to trigger sync")
            sys.exit()

    def get_insights_table(self, insight_id: int, groups: List[str]) -> pd.DataFrame:
        """
        Given an insight_id and a list of breakdowns, return the insights table as a DataFrame.

        Example:
            mp.get_insights_table(73654407, groups=["date", "utm source", "utm campaign", "country"])
        """
        data = self.query_insights(insights_id=insight_id, response_format="json")
        return self._parse_mixpanel_insights_table(data, groups)

    def _parse_mixpanel_insights_table(self, data: dict, groups: List[str]) -> pd.DataFrame:
        if "series" not in data:
            raise ValueError("The key 'series' is missing from the data.")

        series = data["series"]
        kpis = list(series.keys())
        df = pd.json_normalize(series)

        if len(kpis) > 1:
            new_cols = [col.split(".") for col in df.columns]
            new_cols = ".".join([subname.strip() for subname in new_cols])[2:]
            df.columns = new_cols
        df.columns = [col.split(".all")[0] for col in df.columns]

        overall_columns = [col for col in df.columns if "$overall" in col]
        df = df.drop(columns=overall_columns)

        df_long = df.melt(var_name="full_column", value_name="value")
        df_long[groups] = df_long["full_column"].str.split(".", expand=True)
        return df_long.drop(columns=["full_column"]).reset_index(drop=True)
