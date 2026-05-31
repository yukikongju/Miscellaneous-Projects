import ast
import base64
import json
import sys
from typing import List

import pandas as pd
import requests


def get_mixpanel_funnel_dataframe_from_json(data: dict):
    if not data:
        raise ValueError("No data provided to parse into DataFrame.")

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

    def query_insights(
        self,
        insights_id: int,
        response_format: str = "json",
        limit: int = 50000,
        timeout: int = 30,
    ) -> dict | str:
        """
        Fetch an Insights report by ID.

        The report ID can be found in the Mixpanel URL:
        https://mixpanel.com/project/<PROJECT_ID>/view/<WORKSPACE_ID>/app/boards
            #id=12345&editor-card-id=%22report-<INSIGHTS_ID>%22

        Args:
            insights_id: Mixpanel Insights report ID.
            response_format: "json" (default) or "csv".
            limit: Maximum number of rows to return (default: 50_000).
            timeout: Request timeout in seconds (default: 30).

        Returns:
            Parsed dict if response_format="json", raw CSV string if "csv".

        Raises:
            ValueError: If response_format is invalid or JSON parsing fails.
            requests.Timeout: If the request exceeds `timeout` seconds.
            requests.RequestException: On any network-level failure.
            RuntimeError: If Mixpanel returns a non-200 status code.
        """
        if response_format not in {"json", "csv"}:
            raise ValueError(f"response_format must be 'json' or 'csv', got {response_format!r}")

        url = "https://mixpanel.com/api/query/insights"
        params = {"project_id": self.project_id, "bookmark_id": insights_id}
        body = {"queryLimits": {"limit": limit}, "format": response_format}

        try:
            response = requests.post(
                url,
                headers=self.headers,
                params=params,
                json=body,  # sets Content-Type and serialises automatically
                timeout=timeout,
            )
        except requests.Timeout:
            raise requests.Timeout(
                f"query_insights timed out after {timeout}s (insights_id={insights_id})"
            )
        except requests.RequestException as e:
            raise requests.RequestException(f"query_insights network error: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"query_insights failed: HTTP {response.status_code} "
                f"(insights_id={insights_id})\n"
                f"Response body: {response.text[:500]}"
            )

        if response_format == "csv":
            return response.text

        try:
            return response.json()
        except ValueError as e:
            raise ValueError(
                f"query_insights: could not parse response as JSON "
                f"(insights_id={insights_id}). Body preview: {response.text[:200]}"
            ) from e

    def query_funnel(
        self,
        funnel_id: int,
        from_date: str,
        to_date: str,
        timeout: int = 30,
    ) -> dict:
        """
        Fetch funnel data broken down by date, country, and user.
        Date format: YYYY-MM-DD (both inclusive).

        Args:
            funnel_id: Mixpanel funnel ID.
            from_date: Start date in YYYY-MM-DD format (inclusive).
            to_date: End date in YYYY-MM-DD format (inclusive).
            response_format: Response format (default: "csv").
            timeout: Request timeout in seconds (default: 30).

        Returns:
            Parsed JSON response as a dict.

        Raises:
            requests.Timeout: If the request exceeds `timeout` seconds.
            requests.RequestException: On any network-level failure.
            ValueError: If the response body cannot be parsed as JSON.
            RuntimeError: If Mixpanel returns a non-200 status code.
        """
        url = "https://mixpanel.com/api/query/funnels"
        params = {
            "project_id": self.project_id,
            "funnel_id": funnel_id,
            "from_date": from_date,
            "to_date": to_date,
            "unit": "day",
        }

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=timeout)
        except requests.Timeout:
            raise requests.Timeout(
                f"query_funnel timed out after {timeout}s "
                f"(funnel_id={funnel_id}, {from_date} → {to_date})"
            )
        except requests.RequestException as e:
            raise requests.RequestException(f"query_funnel network error: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"query_funnel failed: HTTP {response.status_code} "
                f"(funnel_id={funnel_id}, {from_date} → {to_date})\n"
                f"Response body: {response.text[:500]}"
            )

        try:
            return response.json()
        except ValueError as e:
            raise ValueError(
                f"query_funnel: could not parse response as JSON "
                f"(funnel_id={funnel_id}). Body preview: {response.text[:200]}"
            ) from e

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
