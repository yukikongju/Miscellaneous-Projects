import pandas as pd

cols = [
    "Model",
    "English",
    "Twi",
    "Lingala",
    "Kinyarwanda",
    "Haussa",
    "Yoruba",
    "exp_setting",
    "ans_lang",
]
languages = ["English", "Twi", "Lingala", "Kinyarwanda", "Haussa", "Yoruba"]

# -------------------------
# TEXT ONLY
# -------------------------

df_text_only_ANS_EN = pd.DataFrame(
    [
        ["GPT 4o", 67.14, 54.29, 48.57, 42.69, 45.71, 60.06, "text-only", "EN"],
        ["gemini-2.5-flash", 78.85, 62.35, 61.52, 40.39, 52.63, 38.18, "text-only", "EN"],
        ["GPT 5.2", 70.05, 64.29, 57.14, 52.23, 58.57, 58.57, "text-only", "EN"],
        ["Gemini 3.1 Pro Preview", 66.76, 65.5, 52.75, 54.1, 63.04, 57.76, "text-only", "EN"],
        ["Aya Vision", 13.45, 7.32, 13.27, 13.86, 15.59, 9.89, "text-only", "EN"],
    ],
    columns=cols,
)

df_text_only_ANS_native = pd.DataFrame(
    [
        ["GPT 4o", 61.43, 42.86, 48.57, 39.45, 66.71, 55.71, "text-only", "native"],
        ["gemini-2.5-flash", 80.3, 65, 61.52, 35.4, 52.63, 38.18, "text-only", "native"],
        ["GPT 5.2", 67.14, 61.43, 54.29, 50.56, 61.43, 61.43, "text-only", "native"],
        ["Gemini 3.1 Pro Preview", 66.76, 65.5, 52.75, 54.1, 63.04, 57.76, "text-only", "native"],
        ["Aya Vision", 9.54, 5.9, 10.4, 9.35, 11.54, 9.56, "text-only", "native"],
    ],
    columns=cols,
)

# -------------------------
# IMAGE ONLY
# -------------------------

df_image_only_ANS_EN = pd.DataFrame(
    [
        ["GPT 4o", 33.33, 41.67, 39.58, 37.12, 45.83, 39.58, "image-only", "EN"],
        ["gemini-2.5-flash", 36.84, 44.44, 42.86, 25, 50, 47.83, "image-only", "EN"],
        ["GPT 5.2", 33.33, 56.25, 35.42, 30.19, 35.42, 31.25, "image-only", "EN"],
        ["Gemini 3.1 Pro Preview", 17.39, 24, 41.18, 21.05, 42.86, 55, "image-only", "EN"],
        ["Aya Vision", 11.65, 5.86, 14.27, 10.31, 12.45, 8.92, "image-only", "EN"],
    ],
    columns=cols,
)

df_image_only_ANS_native = pd.DataFrame(
    [
        ["GPT 4o", 37.5, 31.25, 43.75, 29.14, 47.92, 33.33, "image-only", "native"],
        ["gemini-2.5-flash", 36.84, 44.44, 47.96, 30.12, 42.45, 47.83, "image-only", "native"],
        ["GPT 5.2", 31.25, 56.25, 33.32, 26.34, 27.08, 41.67, "image-only", "native"],
        ["Gemini 3.1 Pro Preview", 17.39, 24, 45, 24.7, 42.86, 55, "image-only", "native"],
        ["Aya Vision", 8.72, 4.87, 12.54, 8.21, 10.74, 8.2, "image-only", "native"],
    ],
    columns=cols,
)

# -------------------------
# IMAGE + TEXT
# -------------------------

df_imagetext_ANS_EN = pd.DataFrame(
    [
        ["GPT 4o", 82.86, 65.71, 52.86, 51.68, 72.86, 70.01, "imagetext", "EN"],
        ["gemini-2.5-flash", 47.62, 56.56, 50.4, 50, 65.22, 45, "imagetext", "EN"],
        ["Gemini 3.1 Pro Preview", 42.86, 50, 36.36, 34.62, 38.1, 42.11, "imagetext", "EN"],
        ["GPT 5.2", 71.43, 80.02, 58.57, 56.84, 61.43, 67.14, "imagetext", "EN"],
        ["Aya Vision", 21.56, 16.56, 21.9, 20.6, 25.26, 19.34, "imagetext", "EN"],
    ],
    columns=cols,
)

df_imagetext_ANS_native = pd.DataFrame(
    [
        ["GPT 4o", 82.86, 62.86, 61.43, 59.89, 75.71, 61.43, "imagetext", "native"],
        ["gemini-2.5-flash", 47.62, 56.56, 50.4, 50, 65.22, 45, "imagetext", "native"],
        ["Gemini 3.1 Pro Preview", 42.86, 50, 36.36, 34.62, 38.1, 42.11, "imagetext", "native"],
        ["GPT 5.2", 72.86, 77.14, 61.43, 60.57, 55.71, 65.71, "imagetext", "native"],
        ["Aya Vision", 18.54, 14.96, 18.56, 20.3, 20.24, 17.54, "imagetext", "native"],
    ],
    columns=cols,
)


# ----------
dfs = [
    df_text_only_ANS_EN,
    df_text_only_ANS_native,
    df_image_only_ANS_EN,
    df_image_only_ANS_native,
    df_imagetext_ANS_EN,
    df_imagetext_ANS_native,
]
dfs = pd.concat(dfs)

# ---------- H1

# dfs['avg_score'] = dfs[languages].mean(axis=1)
df_pivot = dfs.drop(columns=["ans_lang"]).groupby(["Model", "exp_setting"]).mean().reset_index()
df_pivot.loc[:, "avg_score"] = df_pivot[languages].mean(axis=1)
df_pivot = df_pivot.pivot_table(index="Model", columns="exp_setting", values="avg_score")
df_pivot = df_pivot.round(2)
df_pivot = df_pivot[["text-only", "image-only", "imagetext"]]
# dfs.drop(columns=['ans_lang']).groupby(['Model', 'exp_setting']).mean().reset_index().pivot_table(index='Model', columns='exp_setting', values=['English', 'Twi', 'Lingala', 'Kinyarwanda', 'Haussa', 'Yoruba'], aggfunc='mean')
# dfs.drop(columns=['ans_lang']).groupby(['Model', 'exp_setting']).mean().reset_index().pivot_table(index='Model', columns='exp_setting', values='value', aggfunc='mean')

# ----------- H2
# \setlength\leftmargini{3.5em}
dfs = dfs[dfs["Model"] != "Aya Vision"]
# dfs.drop(columns=['Model']).loc[dfs['exp_setting'] == 'text-only', :].groupby(['ans_lang']).mean()
# dfs[dfs['Model'].str.lower().isin(['GPT 4o', 'GPT 5.2'])].drop(columns=['Model', 'exp_setting']).groupby(['ans_lang']).mean()
dfs[dfs["Model"].isin(["GPT 4o", "GPT 5.2"])].drop(columns=["Model", "exp_setting"]).groupby(
    ["ans_lang"]
).mean()
dfs.drop(columns=["Model", "exp_setting"]).groupby(["ans_lang"]).mean()
