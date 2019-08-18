import pandas as pd
import os
import time
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
import numpy as np

import settings_model

warnings.simplefilter(action='ignore', category=FutureWarning)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
def lgbm_evaluate(**params):
    start = time.time()
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
        
    clf = LGBMClassifier(**params, 
                         n_estimators=10000, 
                         n_jobs=os.cpu_count(),
                         objective="multiclass",
                         num_class=n_classes,
                        )
        
    clf.fit(df_train[features].values, df_train["TARGET"].values, 
            eval_set = [(df_train[features].values, df_train["TARGET"].values),
                        (df_valid[features].values, df_valid["TARGET"].values)],
            early_stopping_rounds=10, verbose=0)
    
    # train_preds = clf.predict_proba(df_train[features].values, num_iteration=clf.best_iteration_)
    valid_preds = clf.predict_proba(df_valid[features].values, num_iteration=clf.best_iteration_)
    
#     print('Accuracy train {:.6f}'.format(sum(np.argmax(train_preds, axis=1) == df_train['TARGET'].values) / float(len(train_preds))))
    acc_valid = np.mean(np.argmax(valid_preds, axis=1) == df_valid['TARGET'].values)
    
    timestamp = time.time()
    with open(os.path.join(output_dir, f"clf_{acc_valid}_{timestamp}.p"), "wb") as fp:
        pickle.dump(clf, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_dir, f"params_{acc_valid}_{timestamp}.p"), "wb") as fp:
        pickle.dump(params, fp, protocol=pickle.HIGHEST_PROTOCOL)        
    with open(os.path.join(output_dir, f"preds_valid_{acc_valid}_{timestamp}.npy"), "wb") as fp:
        np.save(fp, params)
           
    return acc_valid

def optimize_lgbm():
    
    params_space = {'colsample_bytree': (0.9, 1.0),
                    'learning_rate': (0.01, 1.0), 
                    'num_leaves': (20, 1000), 
                    'subsample': (0.5, 1.0), 
                    'max_depth': (2, 1000), 
                    'reg_alpha': (0.0, 1.0), 
                    'reg_lambda': (0.0, 1.0), 
                    'min_split_gain': (0.0001, 1.),
                    'min_child_weight': (5., 200.),
                   }

    bo = BayesianOptimization(lgbm_evaluate, params_space)
    bo.maximize(init_points=50, n_iter=50)
    
    best_acc = bo.max['target']
    best_params = bo.max['params']
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['max_depth'] = int(best_params['max_depth'])
    
    print("Best validation acc: {}".format(best_acc))
    print('Best parameters found by optimization:\n')
    for k, v in best_params.items():
        print(color.BLUE + k + color.END + ' = ' + color.BOLD + str(v)+ color.END + '     [',params_space[k],']')
        
    return best_acc, best_params

def main(model_id):
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    print(model_id)
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    global output_dir
    output_dir = os.path.join(settings_model.root_path, "models", "siamese-cell",
                              f"{model_id}", "emb")
    global df_train
    global df_valid
    df_train = pd.read_csv(os.path.join(output_dir, "emb_train.csv"), header=None)
    df_valid = pd.read_csv(os.path.join(output_dir, "emb_valid.csv"), header=None)
    df_train.columns = df_train.columns.tolist()[:-1] + ["TARGET"]
    df_valid.columns = df_valid.columns.tolist()[:-1] + ["TARGET"]
    global features
    global n_classes
    global labels_valid
    features = df_train.columns.tolist()[:-1]
    n_classes = df_train["TARGET"].nunique()
    labels_valid = df_valid["TARGET"].unique()

    best_acc, best_params = optimize_lgbm()
    
if __name__ == "__main__":
    
    for model_id in ["U2OS_20190817_192038", "RPE_20190817_022451", 
                     "HEPG2_20190816_092727", "HUVEC_20190812_211524"]:
        main(model_id)

