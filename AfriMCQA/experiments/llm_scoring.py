"""
Script which computes in "evaluation/claude_evaluation/"

Analysis:
- Model Setup Performance: QUES_ANS_SETUP vs verdict (correct, partial, wrong) for each (model, exp_setup) pair
- QUES_ANS_SETUP vs category-verdict (model, exp_setup)
- QUES_ANS_SETUP vs augmented-verdict (model, exp_setup)
- H2 - Answer Language vs Accuracy => compare ANS_EN vs ANS_NATIVE
- Country/Language Breakdown

**Model Comparison Anlyses**
Refusal rate analysis — gpt-5_2 refuses much more on ImageOnly. Classifying Wrongs as "refusal" vs. "hallucination" vs. "wrong answer" distinguishes epistemic caution from confabulation.
Partial verdict analysis — Where does partial credit cluster? The Dukhan stool pattern (recognizes smoke/incense but misses the specific ritual) is a good example of culturally incomplete grounding.
Explanation type vs. verdict — The explanation_type column ([Encyclopedic], [Commonsense], [Functional], [Causal]) from gpt-4o lets you ask: when the model uses encyclopedic reasoning, does it do better than when it uses commonsense? This could reveal whether hallucinations are more frequent in a particular reasoning mode.

"""

import pandas as pd
import glob
import math

MODELS = ["gpt-5_2", "gpt-4o"]
EXPERIMENT_SETUPS = ["TextOnly", "ImageOnly", "ImageText"]

# =================================

CLAUDE_EVALUATION_PATH = "evaluation/claude_evaluation"
all_csv_files = glob.glob(f"{CLAUDE_EVALUATION_PATH}/*.csv")
print(all_csv_files)


# ================================= PERFORMANCE
def get_model_setup_performance(df: pd.DataFrame):
    """
    verdict              Correct  Partial  Wrong  Total  Correct_perc  Partial_perc  Wrong_perc
    QUES_ANS_SETUP
    """
    df_performance = (
        df[["QUES_ANS_SETUP", "verdict"]]
        .groupby(["QUES_ANS_SETUP"])
        .value_counts()
        .reset_index()
        .pivot_table(values="count", index="QUES_ANS_SETUP", columns="verdict")
        .fillna(0.0)
    )
    df_performance.loc[:, "Total"] = (
        df_performance["Correct"] + df_performance["Partial"] + df_performance["Wrong"]
    )
    for verdict in ["Correct", "Partial", "Wrong"]:
        df_performance.loc[:, f"{verdict}_perc"] = round(
            df_performance[verdict] / df_performance["Total"] * 100, 2
        )
    df_performance.loc[:, f"Correct_Partial_perc"] = round(
        (df_performance["Correct"] + df_performance["Partial"]) / df_performance["Total"] * 100, 2
    )
    return df_performance.fillna(0.0)


def get_category_verdict_model_performance(df: pd.DataFrame):
    category_maps = {
        "Cooking and food": "Food",
        "Vehicles and transportation Geography, building, and landmarks": "Landmarks",
        "Cooking and food Brands, products, and companies": "Brands",
        "Cooking and food Objects, materials, clothing": "Clothes",
        "Objects, materials, clothing": "Clothes",
        "Geography, building, and landmarks": "Landmarks",
        "Sports and recreation": "Sports",
        "Plants and animals": "Plants/Animals",
        "Tranditions, art, and history" "People, and everyday life": "Traditions",
        "Objects, materials, clothing Tranditions, art, and history": "Traditions",
        "Tranditions, art, and history": "Traditions",
        "People, and everyday life": "People",
        "Other": "Others",
    }
    df_category_performance = df.copy()
    df_category_performance.loc[:, "Category"] = df_category_performance["Category"].replace(
        category_maps
    )
    df_category_performance = (
        df_category_performance[["QUES_ANS_SETUP", "Category", "verdict"]]
        .groupby(["QUES_ANS_SETUP", "Category"])
        .value_counts()
        .reset_index()
        .pivot_table(values="count", index=["QUES_ANS_SETUP", "Category"], columns="verdict")
        .reset_index()
        .fillna(0.0)
    )
    return df_category_performance


def get_explanation_verdict_model_performance(df: pd.DataFrame):
    df_explanation = df.copy()
    df_explanation.loc[df["explanation_type"].isna(), "explanation_type"] = "None"
    df_explanation = (
        df_explanation[["QUES_ANS_SETUP", "explanation_type", "verdict"]]
        .groupby(["QUES_ANS_SETUP", "explanation_type"])
        .value_counts()
        .reset_index()
        .pivot_table(
            values="count", index=["QUES_ANS_SETUP", "explanation_type"], columns="verdict"
        )
        .reset_index()
        .fillna(0.0)
    )
    return df_explanation


dfs_performance = []
dfs_category = []
dfs_explanation = []
for model in MODELS:
    for exp_setup in EXPERIMENT_SETUPS:
        file_name = f"{CLAUDE_EVALUATION_PATH}/{model}_{exp_setup}_all_evaluated.csv"
        df = pd.read_csv(file_name)
        df_performance = get_model_setup_performance(df)
        df_category = get_category_verdict_model_performance(df)
        df_explanation = get_explanation_verdict_model_performance(df)

        dfs_category.append(df_category)
        dfs_explanation.append(df_explanation)

        print(
            "=" * 75,
        )
        print(f"{model=}")
        print(f"{exp_setup=}")
        print(
            "=" * 75,
        )
        print("-" * 75)
        # print(df_performance)
        print(df_category)
        print("-" * 75)
        print()

# dfs, groupby_col = dfs_category.copy(), 'Category'
dfs, groupby_col = dfs_explanation.copy(), "explanation_type"

dfs = pd.concat(dfs).reset_index().groupby([groupby_col]).sum().fillna(0.0)
verdicts = ["Correct", "Partial", "Wrong"]
dfs.loc[:, "Total"] = dfs["Correct"] + dfs["Partial"] + dfs["Wrong"]
for verdict in verdicts:
    dfs.loc[:, f"{verdict}_perc"] = round(dfs[verdict] / dfs["Total"] * 100, 2)
dfs.loc[:, f"Correct_Partial_perc"] = round(
    (dfs["Correct"] + dfs["Partial"]) / dfs["Total"] * 100, 2
)
dfs = dfs.drop(columns=["index", "QUES_ANS_SETUP"])
print(dfs)

# =================================
