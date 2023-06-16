
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

# Set data and model paths
dataPath = '../train.csv'
modelPath = '../models/'

# Load data
df_test = pd.read_csv(dataPath + 'test.csv')

# Define column names
labels = ['SQL Injection', 'Infilteration', 'DoS attacks-SlowHTTPTest','DoS attacks-GoldenEye', 'Bot', 'DoS attacks-Slowloris','Brute Force -Web', 'DDOS attack-LOIC-UDP', 'Benign','Brute Force -XSS']

cat_names = ['Dst Port', 'Protocol']
y_names = 'Label'
cont_names = ['Flow Duration', 'Tot Fwd Pkts',
              'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
              'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
              'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
              'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
              'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
              'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
              'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
              'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags',
              'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
              'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
              'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
              'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
              'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
              'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
              'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
              'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
              'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
              'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
              'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
#chua doi feature
# Define data preprocessing steps
procs = [Categorify, FillMissing, Normalize]
y_block = CategoryBlock()

# Load pre-trained models
gmodel = load_learner('/home/dis/balancingDS/ICC_2023/VH_ICC/CIC/models/final_model/noGPU_DNN_CIC',cpu=True)
xgb_model_loaded = pickle.load(open('/home/dis/balancingDS/ICC_2023/VH_ICC/CIC/models/final_model/CICxgb_model.pkl', "rb"))
gbm_model_loaded = pickle.load(open('/home/dis/balancingDS/ICC_2023/VH_ICC/CIC/models/final_model/CICbm_model.pkl', "rb"))
bme_model_loaded = pickle.load(open('/home/dis/balancingDS/ICC_2023/VH_ICC/CIC/models/final_model/CICbme_model.pkl', "rb"))
catB_model_loaded = pickle.load(open('/home/dis/balancingDS/ICC_2023/VH_ICC/CIC/models/final_model/CICcatB_model.pkl', "rb"))


#Predict with GBM
start = time.time()
gbm_preds = gbm_model_loaded.predict_proba(df_test[df_test.columns[:-1]])
elapsed_gbm = time.time() - start


#Predict with BME
start = time.time()
bme_preds = bme_model_loaded.predict_proba(df_test[df_test.columns[:-1]])
elapsed_gbm = time.time() - start

#Predict with CatB
start = time.time()
catB_preds = catB_model_loaded.predict_proba(df_test[df_test.columns[:-1]])
elapsed_gbm = time.time() - start

#Predict with XGBoost
start = time.time()
xgb_preds = xgb_model_loaded.predict_proba(df_test[df_test.columns[:-1]])
elapsed_xgb = time.time() - start

#Predic with DNN
data = df_test
data[y_names] = data[y_names].astype('category')
yLabels = data['Label']
dl = gmodel.dls.test_dl(data, with_labels=True, drop_last=False)
start = time.time()
nn_preds, tests, clas_idx = gmodel.get_preds(dl=dl, with_loss=False, with_decoded=True)
loss, acc, precision, f1, recall, roc = gmodel.validate(dl=dl)
elapsed = time.time() - start
cm_nn = ClassificationInterpretation.from_learner(gmodel,dl=dl)#, y_preds=nn_preds)

# Soft Voting (Ensemble)
avgs = nn_preds*0.1 + xgb_preds*0.3 + gbm_preds*0.2 + bme_preds*0.2 + catB_preds*0.2
argmax = avgs.argmax(dim=1)

# Evaluate XGBoost
accuracy_xgb = accuracy(tensor(xgb_preds), tensor(tests))
precision_xgb = precision_score(tests, xgb_preds.argmax(axis=1), average='weighted')
f1_xgb = f1_score(tests, xgb_preds.argmax(axis=1), average='weighted')
recall_xgb = recall_score(tests, xgb_preds.argmax(axis=1), average='weighted')
Roc_xgb = roc_auc_score(tests, xgb_preds, multi_class="ovr", average='weighted')
cm_xgb = confusion_matrix(tests,np.argmax(xgb_preds, axis=1))

# Evaluate GBM
accuracy_gbm = accuracy(tensor(gbm_preds), tensor(tests))
precision_gbm = precision_score(tests, xgb_preds.argmax(axis=1), average='weighted')
f1_gbm = f1_score(tests, xgb_preds.argmax(axis=1), average='weighted')
recall_gbm = recall_score(tests, xgb_preds.argmax(axis=1), average='weighted')
Roc_gbm = roc_auc_score(tests, xgb_preds, multi_class="ovr", average='weighted')
cm_gbm = confusion_matrix(tests,np.argmax(gbm_preds, axis=1))

# Evaluate BME
accuracy_bme = accuracy(tensor(bme_preds), tensor(tests))
precision_bme = precision_score(tests, bme_preds.argmax(axis=1), average='weighted')
f1_bme = f1_score(tests, xgb_preds.argmax(axis=1), average='weighted')
recall_bme = recall_score(tests, bme_preds.argmax(axis=1), average='weighted')
Roc_bme = roc_auc_score(tests, bme_preds, multi_class="ovr", average='weighted')
cm_bme = confusion_matrix(tests,np.argmax(bme_preds, axis=1))

# Evaluate CatB
accuracy_catB = accuracy(tensor(catB_preds), tensor(tests))
precision_catB = precision_score(tests, catB_preds.argmax(axis=1), average='weighted')
f1_catB = f1_score(tests, catB_preds.argmax(axis=1), average='weighted')
recall_catB = recall_score(tests, catB_preds.argmax(axis=1), average='weighted')
Roc_catB = roc_auc_score(tests, catB_preds, multi_class="ovr", average='weighted')
cm_catB = confusion_matrix(tests,np.argmax(catB_preds, axis=1))


# Evaluate Ensemble Learning
start = time.time()
accuracy_ens = accuracy_score(tests, avgs.argmax(axis=1))
elapsed_ensemble = time.time() - start
precision_ens = precision_score(tests, avgs.argmax(axis=1), average='weighted')
f1_ens = f1_score(tests, avgs.argmax(axis=1), average='weighted')
recall_ens = recall_score(tests, avgs.argmax(axis=1), average='weighted')
Roc_ens = roc_auc_score(tests, avgs, multi_class="ovr", average='weighted')
cm_ens = confusion_matrix(tests,avgs.argmax(axis=1))

# Print results
print('DNN Results:')
print(f'Accuracy: {acc:.2%}, Precision: {precision:.2%}, F1: {f1:.2%}, Recall: {recall:.2%}, ROC-AUC: {roc:.2%}')
print('Confusion Matrix:')
#print(cm_nn)
print(cm_nn.confusion_matrix())

print('\nXGBoost Results:')
print(f'Accuracy: {accuracy_xgb:.2%}, Precision: {precision_xgb:.2%}, F1: {f1_xgb:.2%}, Recall: {recall_xgb:.2%}, ROC-AUC: {Roc_xgb:.2%}')
print('Confusion Matrix:')
print(cm_xgb)

print('\nGBM Results:')
print(f'Accuracy: {accuracy_gbm:.2%}, Precision: {precision_gbm:.2%}, F1: {f1_gbm:.2%}, Recall: {recall_gbm:.2%}, ROC-AUC: {Roc_gbm:.2%}')
print('Confusion Matrix:')
print(cm_gbm)

print('\nBME Results:')
print(f'Accuracy: {accuracy_bme:.2%}, Precision: {precision_bme:.2%}, F1: {f1_bme:.2%}, Recall: {recall_bme:.2%}, ROC-AUC: {Roc_bme:.2%}')
print('Confusion Matrix:')
print(cm_bme)

print('\ncatBoost Results:')
print(f'Accuracy: {accuracy_catB:.2%}, Precision: {precision_catB:.2%}, F1: {f1_catB:.2%}, Recall: {recall_catB:.2%}, ROC-AUC: {Roc_catB:.2%}')
print('Confusion Matrix:')
print(cm_catB)

print('\nEnsemble Learning Results:')
print(f'Accuracy: {accuracy_ens:.2%}, Precision: {precision_ens:.2%}, F1: {f1_ens:.2%}, Recall: {recall_ens:.2%}, ROC-AUC: {Roc_ens:.2%}')
print('Confusion Matrix:')
print(cm_ens)




