
from fileinput import filename
from gc import enable
import sys
from matplotlib.pyplot import axis
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, auc, roc_curve
import warnings
import pickle
warnings.filterwarnings("ignore")
from fastai.tabular.all import *
import subprocess
import tensorflow as tf
import torch

torch.backends.cudnn.benchmark = True

###################
#setting GPU Mem
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
if torch.cuda.is_available():
    print("GPU enabling...")
    torch.cuda.device('cuda')
else:
    print("No GPU")

# Load test data
test_data = pd.read_csv('test.csv')

# Split test data into features (X_test) and labels (y_test)
X_test = test_data.drop('Label', axis=1)
y_test = test_data['Label']

# Load the individual models
gbdt_model = pickle.load(open('../CICcatB_model.pkl', "rb"))
bagging_model = pickle.load(open('../CICbme_model.pkl', "rb"))
gbm_model = pickle.load(open('../CICbm_model.pkl', "rb"))
xgboost_model = pickle.load(open('../CICxgb_model.pkl', "rb"))
dnn_model = load_learner('../noGPU_DNN_CIC',cpu=True)

# Define the evaluation metric (e.g., accuracy)
def evaluate_model(weights):
    # Normalize weights
    weights_normalized = weights / np.sum(weights)

    # Combine predictions using weighted ensemble
    weighted_predictions = []
    for i in range(len(X_test)):
        predictions = []
        predictions.append(gbdt_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
        predictions.append(bagging_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
        predictions.append(gbm_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
        predictions.append(xgboost_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
        predictions.append(np.argmax(dnn_model.predict(X_test.iloc[i].values.reshape(1, -1)))[0])
        weighted_prediction = np.argmax(np.bincount(predictions, weights=weights_normalized))
        weighted_predictions.append(weighted_prediction)

    return accuracy_score(y_test, weighted_predictions)

# Define the optimization objective for Ax
def optimization_objective(weights):
    # Adjust weights by increment of 0.01
    weights_adjusted = [w + 0.01 for w in weights]

    # Normalize adjusted weights
    weights_normalized = weights_adjusted / np.sum(weights_adjusted)

    return {"objective": (evaluate_model(weights_normalized), 0.0)}

# Define the parameter search space for Ax
parameter_space = [
    {"name": "weight_1", "type": "range", "bounds": [0, 1]},
    {"name": "weight_2", "type": "range", "bounds": [0, 1]},
    {"name": "weight_3", "type": "range", "bounds": [0, 1]},
    {"name": "weight_4", "type": "range", "bounds": [0, 1]},
    {"name": "weight_5", "type": "range", "bounds": [0, 1]},
]

# Run the optimization with Ax
best_weights, best_accuracy = optimize(
    parameters=parameter_space,
    evaluation_function=optimization_objective,
    minimize=False,
)

# Adjust best weights by increment of 0.01
best_weights_adjusted = [w + 0.01 for w in best_weights]

# Normalize adjusted best weights
best_weights_normalized = best_weights_adjusted / np.sum(best_weights_adjusted)

# Combine predictions using the best weights
ensemble_predictions = []
for i in range(len(X_test)):
    predictions = []
    predictions.append(gbdt_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
    predictions.append(bagging_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
    predictions.append(gbm_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
    predictions.append(xgboost_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
    predictions.append(np.argmax(dnn_model.predict(X_test.iloc[i].values.reshape(1, -1)))[0])
    weighted_prediction = np.argmax(np.bincount(predictions, weights=best_weights_normalized))
    ensemble_predictions.append(weighted_prediction)

# Evaluate the ensemble model
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print("Ensemble Learning Accuracy:", ensemble_accuracy)


# Confusion matrix for GBDT model
cm_gbdt = confusion_matrix(y_test, gbdt_model.predict(X_test))
print("Confusion Matrix - GBDT Model:")
print(cm_gbdt)

# Confusion matrix for Bagging model
cm_bagging = confusion_matrix(y_test, bagging_model.predict(X_test))
print("Confusion Matrix - Bagging Model:")
print(cm_bagging)

# Confusion matrix for GBM model
cm_gbm = confusion_matrix(y_test, gbm_model.predict(X_test))
print("Confusion Matrix - GBM Model:")
print(cm_gbm)

# Confusion matrix for XGBoost model
cm_xgboost = confusion_matrix(y_test, xgboost_model.predict(X_test))
print("Confusion Matrix - XGBoost Model:")
print(cm_xgboost)

# Confusion matrix for DNN model
dnn_predictions = np.argmax(dnn_model.predict(X_test), axis=1)
cm_dnn = confusion_matrix(y_test, dnn_predictions)
print("Confusion Matrix - DNN Model:")
print(cm_dnn)

# Confusion matrix for ensemble model
ensemble_predictions = []
for i in range(len(X_test)):
    predictions = []
    predictions.append(gbdt_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
    predictions.append(bagging_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
    predictions.append(gbm_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
    predictions.append(xgboost_model.predict(X_test.iloc[i].values.reshape(1, -1))[0])
    predictions.append(np.argmax(dnn_model.predict(X_test.iloc[i].values.reshape(1, -1)))[0])
    weighted_prediction = np.argmax(np.bincount(predictions, weights=best_weights_normalized))
    ensemble_predictions.append(weighted_prediction)

cm_ensemble = confusion_matrix(y_test, ensemble_predictions)
print("Confusion Matrix - Ensemble Model:")
print(cm_ensemble)
