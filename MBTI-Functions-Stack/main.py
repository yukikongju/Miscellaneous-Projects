import pandas as pd
import numpy as np
import unittest


class MBTIStackFinder(object):


    def __init__(self, file_path: str):
        self.file_path = file_path

        self._init_df_mbti()


    def _init_df_mbti(self, ):
        try:
            self.df_mbti = pd.read_csv(self.file_path)
        except:
            raise ValueError(f"File '{self.file_path}' not found. Please check")


    def get_true_mbti_function_stack(self, mbti: str):
        """ 
        Given mbti type, return function type as list
            [dominant,auxiliary,tertiary,inferior]
        """
        mbti = mbti.lower()

        # --- check if mbti type is valid
        personalities = self.df_mbti['mbti'].unique()
        if mbti not in personalities:
            raise ValueError(f"The MBTI type '{mbti}' is invalid. Please check")

        # ---
        row = self.df_mbti[self.df_mbti['mbti'] == mbti]
        
        # ---
        function_stack = [row['dominant'], row['auxiliary'], row['tertiary'], row['inferior']]

        return np.array(function_stack).flatten()



def get_mbti_function_stack(mbti: str):
    """ 
    Given MBTI type, return stacking using the following strategy:

    We alternate between E/I depending on the extraversion
    """
    dual_types = {'E': 'I', 'I': 'E', 
                  'N': 'S', 'S': 'N',
                  'F': 'T', 'T': 'F',
                  'P': 'J', 'J': 'P',
                  }

    # --- check if mbti type is valid
    mbti = mbti.upper()

    # --- get second and third letter
    first, second, third, fourth = mbti
    extraversion = first.lower()
    inverse_extraversion = dual_types.get(first).lower()

    if ((fourth == 'P') and (first == 'E')) or ((fourth == 'J') and (first == 'I')):
            dominant = second + extraversion
            auxiliary = third + inverse_extraversion
            tertiary = dual_types.get(third) + extraversion
            inferior = dual_types.get(second) + inverse_extraversion
    elif ((fourth == 'P') and (first == 'I')) or ((fourth == 'J') and (first == 'E')):
            dominant = third + extraversion
            auxiliary = second + inverse_extraversion
            tertiary = dual_types.get(second) + extraversion
            inferior = dual_types.get(third) + inverse_extraversion

    stack = np.array([dominant, auxiliary, tertiary, inferior])

    return stack
    

class TestMBTI(unittest.TestCase):

    def test_mbti_stack_functions(self):
        FILE_PATH = 'MBTI-Functions-Stack/mbti.csv'
        finder = MBTIStackFinder(FILE_PATH)

        # --- Analysts
        self.assertTrue((get_mbti_function_stack('INTJ') == finder.get_true_mbti_function_stack('INTJ')).any())
        self.assertTrue((get_mbti_function_stack('ENTJ') == finder.get_true_mbti_function_stack('ENTJ')).any())
        self.assertTrue((get_mbti_function_stack('INTP') == finder.get_true_mbti_function_stack('INTP')).any())
        self.assertTrue((get_mbti_function_stack('ENTP') == finder.get_true_mbti_function_stack('ENTP')).any())

        # --- Diplomats
        self.assertTrue((get_mbti_function_stack('INFJ') == finder.get_true_mbti_function_stack('INFJ')).any())
        self.assertTrue((get_mbti_function_stack('ENFJ') == finder.get_true_mbti_function_stack('ENFJ')).any())
        self.assertTrue((get_mbti_function_stack('INFP') == finder.get_true_mbti_function_stack('INFP')).any())
        self.assertTrue((get_mbti_function_stack('ENFP') == finder.get_true_mbti_function_stack('ENFP')).any())

        # --- Explorers
        self.assertTrue((get_mbti_function_stack('ISTJ') == finder.get_true_mbti_function_stack('ISTJ')).any())
        self.assertTrue((get_mbti_function_stack('ESTJ') == finder.get_true_mbti_function_stack('ESTJ')).any())
        self.assertTrue((get_mbti_function_stack('ISFJ') == finder.get_true_mbti_function_stack('ISFJ')).any())
        self.assertTrue((get_mbti_function_stack('ESFJ') == finder.get_true_mbti_function_stack('ESFJ')).any())

        # --- Sentinels
        self.assertTrue((get_mbti_function_stack('ISFP') == finder.get_true_mbti_function_stack('ISFP')).any())
        self.assertTrue((get_mbti_function_stack('ESFP') == finder.get_true_mbti_function_stack('ESFP')).any())
        self.assertTrue((get_mbti_function_stack('ISTP') == finder.get_true_mbti_function_stack('ISTP')).any())
        self.assertTrue((get_mbti_function_stack('ESTP') == finder.get_true_mbti_function_stack('ESTP')).any())
        


def main(file_path: str):
    finder = MBTIStackFinder(file_path)

    # --- Analysts
    print(finder.get_true_mbti_function_stack('INTJ'))
    print(finder.get_true_mbti_function_stack('ENTJ'))
    print(finder.get_true_mbti_function_stack('INTP'))
    print(finder.get_true_mbti_function_stack('ENTP'))

    # --- Diplomats
    print(finder.get_true_mbti_function_stack('INFJ'))
    print(finder.get_true_mbti_function_stack('ENFJ'))
    print(finder.get_true_mbti_function_stack('INFP'))
    print(finder.get_true_mbti_function_stack('ENFP'))

    # --- Explorers
    print(finder.get_true_mbti_function_stack('ISTJ'))
    print(finder.get_true_mbti_function_stack('ESTJ'))
    print(finder.get_true_mbti_function_stack('ISFJ'))
    print(finder.get_true_mbti_function_stack('ESFJ'))

    # --- Sentinels
    print(finder.get_true_mbti_function_stack('ISFP'))
    print(finder.get_true_mbti_function_stack('ESFP'))
    print(finder.get_true_mbti_function_stack('ISTP'))
    print(finder.get_true_mbti_function_stack('ESTP'))

    # ---
    print()

    # --- Analysts
    print(get_mbti_function_stack('INTJ') == finder.get_true_mbti_function_stack('INTJ'))
    print(get_mbti_function_stack('ENTJ') == finder.get_true_mbti_function_stack('ENTJ'))
    print(get_mbti_function_stack('INTP') == finder.get_true_mbti_function_stack('INTP'))
    print(get_mbti_function_stack('ENTP') == finder.get_true_mbti_function_stack('ENTP'))

    # --- Diplomats
    print(get_mbti_function_stack('INFJ') == finder.get_true_mbti_function_stack('INFJ'))
    print(get_mbti_function_stack('ENFJ') == finder.get_true_mbti_function_stack('ENFJ'))
    print(get_mbti_function_stack('INFP') == finder.get_true_mbti_function_stack('INFP'))
    print(get_mbti_function_stack('ENFP') == finder.get_true_mbti_function_stack('ENFP'))

    # --- Explorers
    print(get_mbti_function_stack('ISTJ') == finder.get_true_mbti_function_stack('ISTJ'))
    print(get_mbti_function_stack('ESTJ') == finder.get_true_mbti_function_stack('ESTJ'))
    print(get_mbti_function_stack('ISFJ') == finder.get_true_mbti_function_stack('ISFJ'))
    print(get_mbti_function_stack('ESFJ') == finder.get_true_mbti_function_stack('ESFJ'))

    # --- Sentinels
    print(get_mbti_function_stack('ISFP') == finder.get_true_mbti_function_stack('ISFP'))
    print(get_mbti_function_stack('ESFP') == finder.get_true_mbti_function_stack('ESFP'))
    print(get_mbti_function_stack('ISTP') == finder.get_true_mbti_function_stack('ISTP'))
    print(get_mbti_function_stack('ESTP') == finder.get_true_mbti_function_stack('ESTP'))


#  -------------------------------------------------------------------------


if __name__ == "__main__":
    FILE_PATH = 'MBTI-Functions-Stack/mbti.csv'
    main(file_path=FILE_PATH)
    unittest.main()
