import unittest

import pandas as pd

from bookrec.data import (
    create_id_mappings,
    encode_interactions,
    split_interactions,
)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.interactions = pd.DataFrame(
            [
                {"User-ID": user, "ISBN": f"book-{item}", "Book-Rating": item}
                for user in (10, 20)
                for item in range(5)
            ]
        )

    def test_split_keeps_training_examples_for_each_user(self):
        train, validation, test = split_interactions(self.interactions)
        self.assertEqual((len(train), len(validation), len(test)), (6, 2, 2))
        self.assertEqual(set(train["User-ID"]), {10, 20})

    def test_encoding_drops_items_not_seen_in_training(self):
        train = self.interactions.iloc[:5]
        user_mapping, item_mapping = create_id_mappings(train)
        evaluation = pd.DataFrame(
            {
                "User-ID": [10, 10],
                "ISBN": ["book-0", "unseen"],
                "Book-Rating": [5, 5],
            }
        )

        encoded = encode_interactions(evaluation, user_mapping, item_mapping)

        self.assertEqual(len(encoded), 1)
        self.assertEqual(encoded.loc[0, "item"], item_mapping["book-0"])


if __name__ == "__main__":
    unittest.main()
