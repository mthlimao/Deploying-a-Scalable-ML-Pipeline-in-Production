from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from source.ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    return rf_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(X)

    return preds


def check_performance_sliced_data(
    data_sliced, 
    model,
    categorical_features, 
    label,
    encoder, 
    lb,
):
    """
    Checks model performance on sliced data.
    
    Inputs
    ------
    data_sliced: pd.DataFrame
        Sliced data.
    model : ???
        Trained machine learning model.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. 
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    # Get X and y slices
    X_sliced, y_sliced, _, _ = process_data(
        data_sliced, 
        categorical_features=categorical_features, 
        label=label,
        encoder=encoder, lb=lb, training=False
    )

    # Get slices predictions
    y_sliced_pred = inference(model, X_sliced)

    return compute_model_metrics(y_sliced, y_sliced_pred)
