import unittest
import pandas as pd
import numpy as np
from app.data_preprocessing import load_data, handle_missing_values, feature_engineering, preprocess_data

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a dummy DataFrame for testing
        self.df = pd.DataFrame({
            'feature1': [10, 20, np.nan, 40, 50],
            'feature2': [1, 2, 3, 4, 5],
            'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })

    def test_load_data(self):
        # This test would typically involve mocking file I/O
        # For now, we'll assume load_data works with a valid path
        # and focus on other functions.
        pass

    def test_handle_missing_values_mean(self):
        df_processed = handle_missing_values(self.df.copy(), strategy='mean')
        self.assertFalse(df_processed['feature1'].isnull().any())
        self.assertAlmostEqual(df_processed['feature1'][2], self.df['feature1'].mean())

    def test_handle_missing_values_median(self):
        df_processed = handle_missing_values(self.df.copy(), strategy='median')
        self.assertFalse(df_processed['feature1'].isnull().any())
        self.assertAlmostEqual(df_processed['feature1'][2], self.df['feature1'].median())

    def test_feature_engineering(self):
        df_processed = feature_engineering(self.df.copy())
        self.assertIn('feature1_x_feature2', df_processed.columns)
        self.assertEqual(df_processed['feature1_x_feature2'][0], 10 * 1)

    def test_preprocess_data(self):
        X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(self.df.copy())
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertIsNotNone(scaler)
        self.assertIsNotNone(feature_columns)
        self.assertIn('categorical_feature_B', X_train.columns)
        self.assertIn('categorical_feature_C', X_train.columns)

if __name__ == '__main__':
    unittest.main()
