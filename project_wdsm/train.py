import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import early_stopping,log_evaluation,LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier


# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Data

train = pd.read_parquet("/Users/atharvjairath/Desktop/gatech/deep_learning/project_wdsm/data/split/WSDM_train.parquet")
test = pd.read_parquet("/Users/atharvjairath/Desktop/gatech/deep_learning/project_wdsm/data/split/WSDM_test.parquet")
val = pd.read_parquet("/Users/atharvjairath/Desktop/gatech/deep_learning/project_wdsm/data/split/WSDM_valid.parquet")


# delete column 'fold'
train = train.drop('fold', axis=1)
test = test.drop('fold', axis=1)
val = val.drop('fold', axis=1)





def compute_feats(df):
    for col in ["response_a","response_b","prompt"]:
        # response lenght is a key factor when choosing between two responses
        df[f"{col}_len"]=df[f"{col}"].str.len()

        # Some characters counting features 
        df[f"{col}_spaces"]=df[f"{col}"].str.count(r"\s")
        df[f"{col}_punct"]=df[f"{col}"].str.count(r",|\.|!")
        df[f"{col}_question_mark"]=df[f"{col}"].str.count(r"\?")
        df[f"{col}_quot"]=df[f"{col}"].str.count(r"'|\"")
        df[f"{col}_formatting_chars"]=df[f"{col}"].str.count(r"\*|\_")
        df[f"{col}_math_chars"]=df[f"{col}"].str.count(r"\-|\+|\=")
        df[f"{col}_curly_open"]=df[f"{col}"].str.count(r"\{")
        df[f"{col}_curly_close"]=df[f"{col}"].str.count(r"}")
        df[f"{col}_round_open"]=df[f"{col}"].str.count(r"\(")
        df[f"{col}_round_close"]=df[f"{col}"].str.count(r"\)")
        df[f"{col}_special_chars"]=df[f"{col}"].str.count(r"\W")
        df[f"{col}_digits"]=df[f"{col}"].str.count(r"\d")>0
        df[f"{col}_lower"]=df[f"{col}"].str.count("[a-z]").astype("float32")/df[f"{col}_len"]
        df[f"{col}_upper"]=df[f"{col}"].str.count("[A-Z]").astype("float32")/df[f"{col}_len"]
        df[f"{col}_chinese"]=df[f"{col}"].str.count(r'[\u4e00-\u9fff]+').astype("float32")/df[f"{col}_len"]

        # Feature that show how balanced are curly and round brackets
        df[f"{col}_round_balance"]=df[f"{col}_round_open"]-df[f"{col}_round_close"]
        df[f"{col}_curly_balance"]=df[f"{col}_curly_open"]-df[f"{col}_curly_close"]

        # Feature that tells if the string json is present somewhere (e.g. asking a json response or similar)
        # This for example could be expanded also to yaml, but analyses on train set are required to see if enough data is present for this to be really useful
        df[f"{col}_json"]=df[f"{col}"].str.lower().str.count("json")
    return df
    
train=compute_feats(train)
test=compute_feats(test)
val=compute_feats(val)


# Create directory for cached features
os.makedirs('cached_features', exist_ok=True)
preprocessor_path = 'cached_features/tfidf_preprocessor.joblib'
train_feats_path = 'cached_features/train_tfidf_feats.joblib'
test_feats_path = 'cached_features/test_tfidf_feats.joblib'

# Check if cached features exist
if os.path.exists(preprocessor_path) and os.path.exists(train_feats_path) and os.path.exists(test_feats_path):
    print('Loading cached TFIDF features...')
    preprocessor = joblib.load(preprocessor_path)
    train_feats = joblib.load(train_feats_path)
    test_feats = joblib.load(test_feats_path)
    print('Cached TFIDF features loaded.')
else:
    # TFIDF
    print('Generating TFIDF features...')
    vectorizer_char = TfidfVectorizer(sublinear_tf=True, analyzer='char', ngram_range=(1,2), max_features=50000)
    vectorizer_word = TfidfVectorizer(sublinear_tf=True, analyzer='word', min_df=3)
    preprocessor = ColumnTransformer(
        transformers=[
            ('prompt_feats', FeatureUnion([
                ('prompt_char', vectorizer_char),
                ('prompt_word', vectorizer_word)
            ]), 'prompt'),
            ('response_a_feats', FeatureUnion([
                ('response_a_char', vectorizer_char),
                ('response_a_word', vectorizer_word)
            ]), 'response_a'),
            ('response_b_feats', FeatureUnion([
                ('response_b_char', vectorizer_char),
                ('response_b_word', vectorizer_word)
            ]), 'response_b')
        ]
    )
    train_feats = preprocessor.fit_transform(train[["response_a","response_b","prompt"]])
    test_feats = preprocessor.transform(test[["response_a","response_b","prompt"]])
    print('TFIDF features generated...')
    
    # Save the preprocessor and features
    print('Saving TFIDF features...')
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(train_feats, train_feats_path)
    joblib.dump(test_feats, test_feats_path)
    print('TFIDF features saved.')



# Training Data

print('Generating training features...')
feats=list(train.columns)[8:]
# train["winner"]=(train["winner"]=="model_b").astype("int")
X_train=train[feats]
y_train=train["winner"]
print('Training features generated...')

print('Generating validation features...')
X_val=val[feats]
y_val=val["winner"]
print('Validation features generated...')


print('Creating LGBM model...')
# Create the model with early stopping
lgbm_model = LGBMClassifier(n_estimators=1000,  # Set a large number for early stopping
                        learning_rate=0.1,
                        early_stopping_rounds=15)  # Stop if no improvement in 15 rounds

print('Training LGBM model...')
# Train the model
lgbm_history = lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='binary_logloss')
print('LGBM model trained...')

# Training CatBoost
print('Creating CatBoost model...')
catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    early_stopping_rounds=15,
    verbose=100
)

print('Training CatBoost model...')
catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
print('CatBoost model trained...')

# Training XGBoost
print('Creating XGBoost model...')
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    early_stopping_rounds=15,
    eval_metric='logloss'
)

print('Training XGBoost model...')
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print('XGBoost model trained...')

# Skip using VotingClassifier as it tries to refit models with early stopping

# Testing

# Test LGBM
print('Testing LGBM model...')
X_test=test[feats]
lgbm_preds = lgbm_model.predict(X_test)
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
print('LGBM model tested...')

lgbm_accuracy = accuracy_score(test["winner"], lgbm_preds)
print(f"LGBM Accuracy: {lgbm_accuracy}")

# Calculate and print F1 score
lgbm_f1 = f1_score(test["winner"], lgbm_preds, average='weighted')
print(f"LGBM F1 Score: {lgbm_f1}")

# Print classification report
print("\nLGBM Classification Report:")
print(classification_report(test["winner"], lgbm_preds))

# Test CatBoost
print('Testing CatBoost model...')
catboost_preds = catboost_model.predict(X_test)
catboost_probs = catboost_model.predict_proba(X_test)[:, 1]
print('CatBoost model tested...')

catboost_accuracy = accuracy_score(test["winner"], catboost_preds)
print(f"CatBoost Accuracy: {catboost_accuracy}")

# Calculate and print F1 score
catboost_f1 = f1_score(test["winner"], catboost_preds, average='weighted')
print(f"CatBoost F1 Score: {catboost_f1}")

# Print classification report
print("\nCatBoost Classification Report:")
print(classification_report(test["winner"], catboost_preds))

# Test XGBoost
print('Testing XGBoost model...')
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
print('XGBoost model tested...')

xgb_accuracy = accuracy_score(test["winner"], xgb_preds)
print(f"XGBoost Accuracy: {xgb_accuracy}")

# Calculate and print F1 score
xgb_f1 = f1_score(test["winner"], xgb_preds, average='weighted')
print(f"XGBoost F1 Score: {xgb_f1}")

# Print classification report
print("\nXGBoost Classification Report:")
print(classification_report(test["winner"], xgb_preds))

# Create ensemble predictions using different methods

# 1. Majority voting ensemble
print("\nTesting Majority Voting Ensemble...")
majority_votes = np.vstack((lgbm_preds, catboost_preds, xgb_preds)).T
majority_preds = np.array([np.bincount(row).argmax() for row in majority_votes])

majority_accuracy = accuracy_score(test["winner"], majority_preds)
majority_f1 = f1_score(test["winner"], majority_preds, average='weighted')

print(f"Majority Voting Ensemble Accuracy: {majority_accuracy:.4f}")
print(f"Majority Voting Ensemble F1 Score: {majority_f1:.4f}")
print("\nMajority Voting Ensemble Classification Report:")
print(classification_report(test["winner"], majority_preds))

# 2. Weighted average ensemble
print("\nTesting Weighted Average Ensemble...")
# Find best weights based on individual model performance
total_accuracy = lgbm_accuracy + catboost_accuracy + xgb_accuracy
lgbm_weight = lgbm_accuracy / total_accuracy
catboost_weight = catboost_accuracy / total_accuracy
xgb_weight = xgb_accuracy / total_accuracy

print(f"Model weights: LGBM={lgbm_weight:.2f}, CatBoost={catboost_weight:.2f}, XGBoost={xgb_weight:.2f}")

# Weighted average of probabilities
weighted_probs = (lgbm_weight * lgbm_probs) + (catboost_weight * catboost_probs) + (xgb_weight * xgb_probs)
weighted_preds = (weighted_probs > 0.5).astype(int)

weighted_accuracy = accuracy_score(test["winner"], weighted_preds)
weighted_f1 = f1_score(test["winner"], weighted_preds, average='weighted')

print(f"Weighted Average Ensemble Accuracy: {weighted_accuracy:.4f}")
print(f"Weighted Average Ensemble F1 Score: {weighted_f1:.4f}")
print("\nWeighted Average Ensemble Classification Report:")
print(classification_report(test["winner"], weighted_preds))

# Compare all models
print("\nModel Comparison:")
print(f"LGBM Accuracy: {lgbm_accuracy:.4f}, F1 Score: {lgbm_f1:.4f}")
print(f"CatBoost Accuracy: {catboost_accuracy:.4f}, F1 Score: {catboost_f1:.4f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}, F1 Score: {xgb_f1:.4f}")
print(f"Majority Voting Ensemble Accuracy: {majority_accuracy:.4f}, F1 Score: {majority_f1:.4f}")
print(f"Weighted Average Ensemble Accuracy: {weighted_accuracy:.4f}, F1 Score: {weighted_f1:.4f}")

print('Done...')