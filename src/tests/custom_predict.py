import unittest

import pandas as pd

from src.predict import ArcheTypePredictor

class TestCustomPredict(unittest.TestCase):
    def test_main(self):

        TEST_FILE_PATH = "./table_samples/Book_5sentidoseditora.pt_September2020_CTA.json"
        df = pd.read_json(TEST_FILE_PATH, lines=True)
        print("before")
        print(df.head())
        args = {
            "model_name": "flan-t5-base-zs",
        }
        
        arch = ArcheTypePredictor(input_files = [df], user_args = args)
        new_df = arch.annotate_columns()
        print("After")
        print(new_df.head())