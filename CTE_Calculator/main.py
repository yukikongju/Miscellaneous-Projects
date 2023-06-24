import pandas as pd


class Questionnaire:


    def __init__(self, csv_path: str, aggregate_columns: [str]):
        self.csv_path = csv_path
        self.aggregate_columns = aggregate_columns

        self.df = pd.read_csv(csv_path)

    def get_df_score(self) -> pd.DataFrame:
        pass


class QuestionnaireCTE(Questionnaire):

    def __init__(self, csv_path):
        super().__init__(csv_path, ['function_type'])

    def get_df_score(self) -> pd.DataFrame:
        # --- compute score by function types
        df_score = self.df.groupby(self.aggregate_columns).agg(
                question_count = pd.NamedAgg(column='function_type', aggfunc='count'),
                score = pd.NamedAgg(column='score', aggfunc='sum')).reset_index()

        # --- classify scores - excellent: 0.75-1.0 ; amelioration: 0.33-0.75; reexaminer: 0.0-0.33
        df_score['classification'] = df_score.apply(
                lambda x: self._get_score_classification(x['question_count'], x['score']), axis=1)

        return df_score

    def _get_score_classification(self, num_questions: int, score: int):
        """ 
        classify scores - excellent: 0.75-1.0 ; amelioration: 0.33-0.75; reexaminer: 0.0-0.33
        """
        MAX_QUESTION_VALUE = 3
        max_score = MAX_QUESTION_VALUE * num_questions
        excellent_score_threshold = int(round(max_score * 0.75))
        amelioration_score_threshold = int(round(max_score * 0.33))

        if score > excellent_score_threshold:
            return 'excellent'
        elif score > amelioration_score_threshold:
            return 'amelioration'
        else:
            return 'reexamination'


def main():
    cte_path = 'CTE_Calculator/questionnaires/cte_2023-06-24_1.csv'
    results_path = 'CTE_Calculator/results/cte_2023-06-24_1.csv'

    questionnaire = QuestionnaireCTE(cte_path)
    df_score = questionnaire.get_df_score()
    df_score.to_csv(results_path, index=False)


if __name__ == "__main__":
    main()

