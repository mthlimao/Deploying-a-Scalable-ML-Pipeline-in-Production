# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The trained model is a simple Random Forest Classifier using a simple train_test_split division for training and validation.

## Intended Use
The model is intended to predict whether an individual's income exceeds $50K/yr based on different aspects such as education, ocuppation, marital status and others. 

## Training Data
The training data used is the Adult Census Income, available in https://archive.ics.uci.edu/dataset/20/census+income

## Evaluation Data
The test data was obtained using a simple sklearn's train_test_split, with a test sixe of 20%.

## Metrics
Metrics on training set:
precision = 1.0, recall = 1.0, f1 = 1.0

Metrics on test set:
precision = 0.7311669128508124, recall = 0.6206896551724138, f1 = 0.671414038657172

## Ethical Considerations
The model may or may not have performance bias based on aspects such as race and sex. Further improvements are welcome.

## Caveats and Recommendations
While this model provides useful predictions for income classification, several limitations should be considered:

 - Bias and Fairness: The model may inherit biases from the Adult Census Income dataset, which could lead to disparities in predictions based on race, gender, or other demographic factors. Users should evaluate fairness metrics before deployment.
 - Feature Limitations: The model relies on predefined features and may not account for real-world economic changes, evolving job markets, or individual-specific factors not captured in the dataset.
 - Generalization: Since the model was trained on a historical dataset, it may not generalize well to new populations or regions with different socio-economic conditions.
 - Threshold Adjustments: Users may need to tune the classification threshold depending on their specific application needs to balance precision and recall.
 - Ongoing Monitoring: Regular model evaluation on updated datasets is recommended to ensure continued reliability and fairness in predictions.

Future improvements could include addressing bias mitigation strategies, incorporating additional socioeconomic features, and testing the model on more diverse datasets.