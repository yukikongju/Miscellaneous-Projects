# Segmenting Users by Engagement Level

```{python}

# Compute total events per user
df["total_events"] = df["listening_sessions"] + df["play_content"] + df["toggle_favorite"]

# Define engagement levels (quartiles)
df["engagement_level"] = pd.qcut(df["total_events"], q=3, labels=["low", "medium", "high"])

# Compute correlation per engagement level
for level in df["engagement_level"].unique():
    subset = df[df["engagement_level"] == level]
    corr_matrix = subset.drop(columns=["user_id", "timebucket", "total_events", "engagement_level"]).corr()
    print(f"Correlation for {level} engagement users:\n", corr_matrix)

```
