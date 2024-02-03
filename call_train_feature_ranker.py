import train_feature_ranker as fr
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pybaseball as pyb
import pandas as pd
import time
import inspect
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
import json
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os
from featureranker.utils import *
from featureranker.plots import *
from featureranker.rankers import *
# from featureranker.utils import view_data
# from featureranker.plots import *
# from featureranker.rankers import *
import glob
import numpy as np
from tqdm.auto import tqdm
import pickle
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
pyb.cache.enable()
pyb.cache.config.cache_type='csv'
pyb.cache.config.save()
# Check if CUDA is available

  
import pybaseball_v5_function_copy as main_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.ion()

# Initialize a dictionary to store the results for each year
results = {}

# Iterate over the years
year=2024
year_str = str(year)
# Call the main function for the current year
X_train_new, y_train_new = fr.main_function(year_str)




X_small, _, y_small, _ = train_test_split(X_train_new, y_train_new, test_size=0.95, random_state=42)  # Use 10% of the data
hypers = classification_hyper_param_search(X_small, y_small, 5, 20)  # For classification tasks
# # or
# hypers = regression_hyper_param_search(X_small, y_small, 3, 5)  # For regression tasks
xb_hypers = hypers[0]['best_params']
rf_hypers = hypers[1]['best_params']

ranking = classification_ranking(X_small, y_small, rf_hypers, xb_hypers)  # For classification tasks
# # or
# # #   ranking = regression_ranking(X_small, y_small, rf_hypers, xb_hypers)  # For regression tasks

scoring = voting(ranking)
plot_ranking(scoring, title='Feature Ranking')

import pickle

with open('scoring_full_all_years.pkl', 'wb') as f:
    pickle.dump(scoring, f) 