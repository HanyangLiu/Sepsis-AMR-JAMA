import numpy as np
import argparse
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils.utils_data import load_data, RandomizedGroupKFold, select_subgroup, instance_filter
from utils.utils_evaluation import evaluate_multi, plot_prc, plot_roc, evaluate, model_explain_multiclass, \
    evaluate_binary
import pandas as pd
from tqdm import tqdm
from os.path import join


def train_model(X_train, y_train, model_name, rs=99):
    # train model
    if model_name == 'xgboost':
        model = XGBClassifier(objective='multi:logistic',
                              booster='gbtree',
                              verbosity=0,
                              random_state=rs,
                              subsample=0.8,
                              use_label_encoder=False)
    elif model_name == 'catboost':
        model = CatBoostClassifier(verbose=0, random_state=rs)
    elif model_name == 'logistic':
        model = LogisticRegression(max_iter=1000,
                                   random_state=rs,
                                   penalty='l1',
                                   solver='liblinear')
    elif model_name == 'svm':
        model = SVC(probability=True, random_state=rs)
    elif model_name == 'mlp':
        model = MLPClassifier(max_iter=1000, random_state=rs)
    else:
        raise NotImplementedError('Please specify the model!!')

    model.fit(X_train, y_train)
    # model.load_model('saved_models/multiclass_{}_{}_{}_complete_1.json'.format(args.model, prefix, rs))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--model', type=str, default='xgboost')  # 'catboost' or 'xgboost'
    parser.add_argument('--full_comorb', type=bool, default=True)
    parser.add_argument('--feats', type=str, default='last')
    parser.add_argument('--select_feats', type=bool, default=True)
    parser.add_argument('--PCA', type=bool, default=False)
    parser.add_argument('--n_repeat', type=int, default=5)
    args = parser.parse_args()
    print(args)

    # load data
    data_table = load_data(select_feats=args.select_feats, feats=args.feats, full_comorb=args.full_comorb)

    # get labels
    data_table = data_table[~data_table.UN]  # exclude cases with unknown labels
    data = data_table.drop(columns=['patient_id', 'admission_id', 'infection_id', 'SS', 'RS', 'RR', 'UN', 'GNB']).astype(float)
    admission_ids = data_table['admission_id']
    data_table.loc[data_table.SS, 'label'] = 0
    data_table.loc[data_table.RS, 'label'] = 1
    data_table.loc[data_table.RR, 'label'] = 2
    labels = data_table['label'].astype(int)

    # train/test split by stratifying so that each split has the same ratio of positive class
    cv = RandomizedGroupKFold(groups=admission_ids.to_numpy(), n_splits=5, random_state=42)
    train_ix, test_ix = cv[0]
    X_train, y_train = data.iloc[train_ix], labels.iloc[train_ix]
    X_test, y_test = data.iloc[test_ix], labels.iloc[test_ix]

    ###########
    # Using only complete cases
    complete_cases = pd.read_csv(join("../data_prepared/", "labels_complete_cases.csv"))
    complete = complete_cases["admission_instance"].tolist()
    X_train = X_train[X_train.index.isin(complete)]
    y_train = y_train[y_train.index.isin(complete)]
    X_test = X_test[X_test.index.isin(complete)]
    y_test = y_test[y_test.index.isin(complete)]
    ###########

    # PCA
    if args.PCA:
        pca = PCA(n_components=50, svd_solver='full')
        pca.fit(X_train)
        data_train, data_test = X_train.copy(), X_test.copy()
        data_train[data_train.columns] = pca.transform(X_train)
        data_test[data_test.columns] = pca.transform(X_test)
        X_train, X_test = data_train, data_test

    # train model
    models = []
    for i, rs in enumerate(tqdm(range(args.n_repeat))):
        prefix = "full" if args.full_comorb else "fam"
        model = train_model(X_train.values, y_train.values, model_name=args.model, rs=rs)
        # model.save_model('saved_models/multiclass_{}_{}_{}_complete_2.json'.format(args.model, prefix, rs))
        models.append(model)

    for infection_instance in ['initial', 'subsequent']:
        prefix = "full_comorb_{}".format(infection_instance) if args.full_comorb else "fam_comorb_{}".format(
            infection_instance)
        selected_instances = instance_filter(data.index.tolist(), mode=infection_instance)
        X_test_selected, y_test_selected = X_test[X_test.index.isin(selected_instances)], \
                                           y_test[y_test.index.isin(selected_instances)]

        # evaluate each class
        AUROC, AUPRC = [], []
        N = 1000
        TPRs, PRECs = np.empty(shape=(3, args.n_repeat, N)), np.empty(shape=(3, args.n_repeat, N))
        for i, rs in enumerate(range(args.n_repeat)):
            # train model
            model = models[i]
            # evaluate model
            auroc, auprc, TPR, PREC = evaluate_multi(model, X_test_selected, y_test_selected.astype(int), N=N,
                                                     verbose=0)
            AUROC.append(list(auroc.values()))
            AUPRC.append(list(auprc.values()))
            TPRs[:, i, :] = np.array([TPR[i] for i in TPR.keys()])
            PRECs[:, i, :] = np.array([PREC[i] for i in PREC.keys()])

            if i == 0: model_explain_multiclass(models[i], X_test_selected, prefix=prefix)

        for i in range(3):
            print('--------------------------------------------')
            print('5 Random Initialization Evaluation:')
            print('--------------------------------------------')
            print("AU-ROC:", "%0.4f" % np.mean(np.array(AUROC)[:, i]), "(%0.4f)" % np.std(np.array(AUROC)[:, i]),
                  "AU-PRC:", "%0.4f" % np.mean(np.array(AUPRC)[:, i]), "(%0.4f)" % np.std(np.array(AUPRC)[:, i]), )
            print('--------------------------------------------')

        tpr_mean, tpr_std = np.mean(TPRs, axis=1), np.std(TPRs, axis=1)
        prec_mean, prec_std = np.mean(PRECs, axis=1), np.std(PRECs, axis=1)

        prefix_name = 'Community_acquired' if infection_instance == 'initial' else 'Hospital_acquired'

        x = np.linspace(0, 1, N + 1)[:N]
        label_names = ['SS', 'RS', 'RR']
        plot_roc(x, tpr_mean, tpr_std,
                 auc=[np.mean(np.array(AUROC), axis=0), np.std(np.array(AUROC), axis=0)],
                 multiclass=True,
                 labels=label_names,
                 prefix=prefix_name)
        plot_prc(x, prec_mean, prec_std,
                 auc=[np.mean(np.array(AUPRC), axis=0), np.std(np.array(AUPRC), axis=0)],
                 multiclass=True,
                 labels=label_names,
                 prefix=prefix_name)

    df_results = evaluate_binary(models, X_test, y_test, args)
