import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb


class LoanDataProcessor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.features = None
        self.target = None
        self.scaler = StandardScaler()

    def preprocess(self):
        self.df.drop(columns=['customer_id'], inplace=True)
        self.create_features()
        self.apply_transformations()
        self.separate_target()
        self.scale_features()

    def create_features(self):
        self.df['loan_to_debt'] = self.df['loan_amt_outstanding'] / self.df['total_debt_outstanding']
        self.df['loan_to_income'] = self.df['loan_amt_outstanding'] / self.df['income']
        self.df['debt_to_income'] = self.df['total_debt_outstanding'] / self.df['income']
        self.df['loan_empyears_interac'] = self.df['loan_amt_outstanding'] * self.df['years_employed']
        self.df['debt_empyears_interac'] = self.df['total_debt_outstanding'] * self.df['years_employed']
        self.df['income_fico_interac'] = self.df['income'] * self.df['fico_score']
        self.df['crlines_fico_interac'] = self.df['credit_lines_outstanding'] * self.df['fico_score']
        self.df['crlines_income_interac'] = self.df['credit_lines_outstanding'] * self.df['income']
        self.df['crlines_empyears_interac'] = self.df['credit_lines_outstanding'] * self.df['years_employed']
        self.df['empyears_fico_interac'] = self.df['years_employed'] * self.df['fico_score']

    def apply_transformations(self):
        self.df['credit_lines_outstanding'] = np.log1p(self.df['credit_lines_outstanding'])
        self.df['total_debt_outstanding'] = np.log1p(self.df['total_debt_outstanding'])
        self.df['loan_to_debt'] = np.log1p(self.df['loan_to_debt'])
        self.df['debt_empyears_interac'] = np.log1p(self.df['debt_empyears_interac'])
        self.df['crlines_fico_interac'] = np.log1p(self.df['crlines_fico_interac'])
        self.df['crlines_income_interac'] = np.log1p(self.df['crlines_income_interac'])
        self.df['crlines_empyears_interac'] = np.log1p(self.df['crlines_empyears_interac'])

    def separate_target(self):
        self.features = self.df.drop(columns=['default'])
        self.target = self.df['default']

    def scale_features(self):
        self.features = pd.DataFrame(self.scaler.fit_transform(self.features), columns=self.features.columns)


class LoanModel:
    def __init__(self, processor):
        self.processor = processor
        self.model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.processor.features, self.processor.target, test_size=0.3, random_state=42
        )
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        self.model.fit(X_train_balanced, y_train_balanced)

        y_pred = self.model.predict(X_test)
        auc_score = roc_auc_score(y_test, y_pred)
        print(f"Model AUC: {auc_score:.4f}")

    def predict_expected_loss(self, new_data):
        new_data_scaled = self.processor.scaler.transform(new_data)
        probabilities = self.model.predict_proba(new_data_scaled)[:, 1]
        expected_loss = probabilities * (1 - 0.1)  # assuming a recovery rate of 10%
        return expected_loss


def main():
    # Load and preprocess data
    processor = LoanDataProcessor('Task_3_and_4_Loan_Data.csv')
    processor.preprocess()

    # Train model
    model = LoanModel(processor)
    model.train()

    # Example: Predict expected loss for new loans
    new_loan_data = pd.read_csv('New_Loan_Data.csv')  # This should be similar to the original dataset
    new_processor = LoanDataProcessor('New_Loan_Data.csv')
    new_processor.preprocess()

    expected_loss = model.predict_expected_loss(new_processor.features)
    print("Expected Loss for New Loans:", expected_loss)


if __name__ == "__main__":
    main()
