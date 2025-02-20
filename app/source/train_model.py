# Script to train machine learning model.
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from source.ml.data import process_data
from source.ml.model import train_model, check_performance_sliced_data
from constants import DATA_PATH, MODEL_PATH

# Add code to load in the data.
data = pd.read_csv(DATA_PATH / 'census.csv', sep=',')
data_columns = [col.replace(' ', '') for col in data.columns]
data.columns = data_columns

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=0)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, encoder=encoder, lb=lb, training=False
)

# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(model, MODEL_PATH / 'model.joblib')

# Get train and test data performance
precision_train, recall_train, fbeta_train = check_performance_sliced_data(
        data_sliced=train, 
        model=model,
        categorical_features=cat_features, 
        label="salary",
        encoder=encoder, 
        lb=lb,
    )

print('Metrics on training set:')
print(f"precision = {precision_train}, recall = {recall_train}, f1 = {fbeta_train}")
print()

precision_test, recall_test, fbeta_test = check_performance_sliced_data(
        data_sliced=test, 
        model=model,
        categorical_features=cat_features, 
        label="salary",
        encoder=encoder, 
        lb=lb,
    )

print('Metrics on test set:')
print(f"precision = {precision_test}, recall = {recall_test}, f1 = {fbeta_test}")
print()

# Check model performance on slices for each type of categorical feature
for cat_feature in cat_features:
    print(f'Checking performance on {cat_feature}')

    for cat_value in data.loc[:, cat_feature].unique():
        data_sliced = data[data[cat_feature] == cat_value]
        precision_sliced, recall_sliced, fbeta_sliced = check_performance_sliced_data(
            data_sliced=data_sliced, 
            model=model,
            categorical_features=cat_features, 
            label="salary",
            encoder=encoder, 
            lb=lb,
        )

        print(f"For {cat_value}:")
        print(f"\tprecision = {precision_sliced}")
        print(f"\trecall = {recall_sliced}")
        print(f"\tf1 = {fbeta_sliced}")
        print()
