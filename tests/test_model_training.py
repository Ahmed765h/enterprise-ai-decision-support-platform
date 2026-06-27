import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from models.model_training import train_model, evaluate_model
from app.data_preprocessing import preprocess_data

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Create a dummy DataFrame for testing
        self.df = pd.DataFrame({
            'feature1': [10, 20, 30, 40, 50],
            'feature2': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        self.X_train, self.X_test, self.y_train, self.y_test, _, _ = preprocess_data(self.df.copy())

    def test_train_logistic_regression(self):
        model = train_model(self.X_train, self.y_train, model_name='LogisticRegression')
        self.assertIsNotNone(model)
        metrics = evaluate_model(model, self.X_test, self.y_test)
        self.assertIn('accuracy', metrics)

    def test_train_random_forest(self):
        model = train_model(self.X_train, self.y_train, model_name='RandomForestClassifier')
        self.assertIsNotNone(model)
        metrics = evaluate_model(model, self.X_test, self.y_test)
        self.assertIn('accuracy', metrics)

    def test_train_xgboost(self):
        model = train_model(self.X_train, self.y_train, model_name='XGBClassifier')
        self.assertIsNotNone(model)
        metrics = evaluate_model(model, self.X_test, self.y_test)
        self.assertIn('accuracy', metrics)

    def test_train_lightgbm(self):
        model = train_model(self.X_train, self.y_train, model_name='LGBMClassifier')
        self.assertIsNotNone(model)
        metrics = evaluate_model(model, self.X_test, self.y_test)
        self.assertIn('accuracy', metrics)

if __name__ == '__main__':
    unittest.main()
