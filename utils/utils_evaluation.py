import numpy as np
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
import matplotlib
import shap
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .utils_data import select_subgroup, instance_filter


def evaluate(model, X_test, y_test, verbose=0, plot=False, N=1000):
    # Testing
    y_prob = 1 - model.predict_proba(X_test)[:, 0]
    y_test[y_test > 0] = 1

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, thresh = metrics.precision_recall_curve(y_test, y_prob)

    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.auc(rec, prec)

    if verbose:
        print('--------------------------------------------')
        print('Evaluation of test set:')
        print("AU-ROC:", "%0.4f" % auroc,
              "AU-PRC:", "%0.4f" % auprc)
        print('--------------------------------------------')

    # interpolation
    x = np.linspace(0, 1, N + 1)[:N]
    TPR = np.interp(x, fpr, tpr)
    PREC = np.interp(x, rec.tolist()[::-1], prec.tolist()[::-1])

    if plot:
        plot_roc(fpr, tpr)
        plot_prc(rec, prec)

    return auroc, auprc, TPR, PREC


def evaluate_multi(model, X_test, y_test, verbose=0, plot=False, N=1000):
    # Testing
    y_probs = model.predict_proba(X_test)
    auroc, auprc, TPR, PREC = dict(), dict(), dict(), dict()

    for i in range(np.shape(y_probs)[1]):
        y_prob = y_probs[:, i]
        y = y_test == i

        # Evaluation
        fpr, tpr, _ = metrics.roc_curve(y, y_prob)
        prec, rec, _ = metrics.precision_recall_curve(y, y_prob)

        auroc[i] = metrics.auc(fpr, tpr)
        auprc[i] = metrics.auc(rec, prec)

        if verbose:
            print('--------------------------------------------')
            print('Evaluation of test set:')
            print("AU-ROC:", "%0.4f" % auroc[i],
                  "AU-PRC:", "%0.4f" % auprc[i])
            print('--------------------------------------------')

        # interpolation
        x = np.linspace(0, 1, N + 1)[:N]
        TPR[i] = np.interp(x, fpr, tpr)
        PREC[i] = np.interp(x, rec.tolist()[::-1], prec.tolist()[::-1])

        if plot:
            plot_roc(fpr, tpr)
            plot_prc(rec, prec)

    return auroc, auprc, TPR, PREC


def plot_roc(fpr, tpr, std=None, auc=None, multiclass=False, labels=None, prefix=None):
    font = {'fontname': 'Arial'}
    fig1, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.title('Receiver Operating Characteristic, {}'.format(prefix), **font)
    colors = ['limegreen', 'royalblue', 'darkviolet']
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not multiclass:
        plt.plot(fpr, tpr, label='AUC = %0.4f' % auc[0])
        if std.any(): plt.fill_between(fpr, tpr - std, tpr + std, alpha=0.5, label='std = %0.4f' % auc[1])
    else:
        for i in range(len(tpr)):
            plt.plot(fpr, tpr[i], label='{}, '.format(labels[i]) + 'AUC %0.2f' % auc[0][i] + ' (%0.3f)' % auc[1][i],
                     color=colors[i])
            if std.any(): plt.fill_between(fpr, tpr[i] - std[i], tpr[i] + std[i], alpha=0.5, color=colors[i])
    plt.legend(loc='lower right', fontsize=12)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=12, **font)
    plt.xlabel('False Positive Rate', fontsize=12, **font)
    plt.savefig('plot/{}-roc.pdf'.format(prefix))
    plt.show()


def plot_prc(rec, prec, std=None, auc=None, multiclass=False, labels=None, prefix=None):
    font = {'fontname': 'Arial'}
    fig1, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('Precision Recall Curve, {}'.format(prefix), **font)
    colors = ['limegreen', 'royalblue', 'darkviolet']
    if not multiclass:
        plt.plot(rec, prec, label='AUC = %0.4f' % auc[0])
        if std.any(): plt.fill_between(rec, prec - std, prec + std, alpha=0.5, label='std = %0.4f' % auc[1])
    else:
        for i in range(len(prec)):
            plt.plot(rec, prec[i], label='{}, '.format(labels[i]) + 'AUC %0.2f' % auc[0][i] + ' (%0.3f)' % auc[1][i],
                     color=colors[i])
            if std.any(): plt.fill_between(rec, prec[i] - std[i], prec[i] + std[i], alpha=0.5, color=colors[i])
    plt.legend(loc='upper left', fontsize=12)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision', fontsize=12, **font)
    plt.xlabel('Sensitivity', fontsize=12, **font)
    plt.savefig('plot/{}-prc.pdf'.format(prefix))
    plt.show()


def plot_thresh(prec, rec, thresh, y_prob):
    thresh = np.concatenate([thresh, np.array([1])])
    f1 = 2 * prec * rec / (prec + rec)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title('Precision/Sensitivity vs. Threshold')
    ax1.hist(y_prob, bins=100, label='Model Output')
    ax2.plot(thresh, prec, label='Precision')
    ax2.plot(thresh, rec, label='Sensitivity')
    ax2.plot(thresh, f1, label='F1')
    plt.xlim([0, 1])
    plt.legend(loc='upper center')
    plt.show()


def custom_summary_plot(model, data_eval):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_eval)
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(data_eval.columns, vals)),
                                      columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    feature_importance.set_index('col_name', inplace=True)

    topN = 30
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    barWidth = 0.5
    bars1 = list(feature_importance[0:topN]['feature_importance_vals'].values)
    r1 = np.arange(len(bars1))
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='Top {} Important Features'.format(topN))
    plt.xticks([r for r in range(len(bars1))], list(feature_importance[0:topN].index))
    fig.autofmt_xdate(rotation=40)
    plt.ylabel('Mean |SHAP| Values', fontweight='bold')
    plt.legend()
    plt.show()


def model_explain_all(model, data_eval, prefix=None):
    # custom_summary_plot(model, data_eval)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data_eval)

    plt.figure()
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.savefig("plot/{}-bar.pdf".format(prefix), bbox_inches="tight")

    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.savefig("plot/{}-beeswarm.pdf".format(prefix), bbox_inches="tight")


def model_explain_multiclass(model, data_eval, prefix=None, max_display=10):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_eval.values)
    feature_names = [name.split(',')[0] for name in data_eval.columns]

    inds = [feature_names.index('time_since_admission'), feature_names.index('resistance_history')]
    if 'initial' in prefix:
        for x in shap_values:
            x[:, inds] = 0

    name_dict = {
        'B96.5': 'History of P. aeruginosa',
        'resistance_history': "History of resistance in same admission",
        'time_since_admission': 'Time since admission',
        'B96.89': 'History of antibiotic administration',
        'lab_ALBUMIN': 'Albumin',
        'J15.1': 'History of pneumonia due to P. aeruginosa',
        'age_yrs': 'Age (years)',
        'lab_HEMOGLOBIN': 'Hemoglobin',
        'lab_LYMPHOCYTES': 'Lymphocyte count',
        'Z16.24': 'History of resistance to multiple antibiotics',
        'R65.21': 'History of severe sepsis or septic shock',
        'B96.20': 'History of E.coli',
        'Z16.12': 'History of ESBL',
        'A41.52': 'History of sepsis due to P. aeruginosa',
        'N39.0': 'History of urinary tract infection',
        'lab_PLATELETS': 'Platelet count',
        'lab_NEUTROPHILS': 'Neutrophil count',
        'J15.6': 'Pneumonia due to other GNB',
        'Y33.XXXA': 'Initial encounter',
        'vital_OXYGEN (O2) THERAPY': 'Vital oxygen therapy',
        'J04.10': 'History of acute tracheitis',
        'lab_PO2': 'Arterial partial pressure of O2',
        'lab_PCO2': 'Arterial partial pressure of CO2',
        'lab_LACTIC ACID': 'Lactic acid',
        'lab_ALANINE TRANSAMINASE (ALT)': 'ALT',
        'lab_CREATININE': "Creatinine",
        'lab_RESPIRATIONS': "Respirations",
        'lab_UREA NITROGEN (BUN)': 'Blood urea nitrogen',
        'vital_VENTILATOR PEEP VALUE': 'PEEP value (ventilator)',
        'vital_BODY MASS INDEX': 'BMI',
        'pneumonia_community': 'Community-acquired pneumonia',
        'vital_RESPIRATIONS': 'Respiratory rate',
        'readmission': 'Previous hospitalization',
        "A41.59": "History of other GNB sepsis",
        "lab_MONOCYTES": "Monocyte count",
        "lab_PH": "pH",
        "lab_WBCS": "WBC count",
        "vital_BLOOD PRESSURE": "Blood Pressure",
        "Z79.4": "History of long term use of insulin",
        "A41.51": "History of sepsis due to Escherichia coli"
    }

    names = []
    for name in feature_names:
        if name in name_dict:
            names.append(name_dict[name])
        else:
            names.append(name)

    feature_names = names

    plt.figure()
    shap.summary_plot(shap_values, data_eval.values, plot_type="bar", class_names=['SS', 'RS', 'RR'],
                      color=matplotlib.colors.ListedColormap(['limegreen', 'royalblue', 'darkviolet']),
                      feature_names=feature_names,
                      max_display=max_display,
                      show=False)
    plt.savefig("plot/{}-summary-all.svg".format(prefix), bbox_inches="tight")
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values[0], data_eval.values, feature_names=feature_names, max_display=max_display, show=False)
    plt.savefig("plot/{}-summary-SS.svg".format(prefix), bbox_inches="tight")
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values[1], data_eval.values, feature_names=feature_names, max_display=max_display, show=False)
    plt.savefig("plot/{}-summary-RS.svg".format(prefix), bbox_inches="tight")
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values[2], data_eval.values, feature_names=feature_names, max_display=max_display, show=False)
    plt.savefig("plot/{}-summary-RR.svg".format(prefix), bbox_inches="tight")
    plt.show()


def evaluate_false_samples(model, X_test, y_test):
    # Testing
    y_prob = model.predict_proba(X_test)[:, 1]
    best_metrics, best_thresh = search_best_f1(y_test, y_prob, granularity=0.001)
    print('--------------------------------------------')
    print('Evaluation of test set with best F1:')
    print("Sensitivity:", "%0.4f" % best_metrics[0],
          "Specificity:", "%0.4f" % best_metrics[1],
          "PPV:", "%0.4f" % best_metrics[2],
          "NPV:", "%0.4f" % best_metrics[3],
          "F1:", "%0.4f" % best_metrics[4],
          )
    print('--------------------------------------------')

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, thresh = metrics.precision_recall_curve(y_test, y_prob)
    plot_thresh(prec, rec, thresh, y_prob)

    try:
        predictions = y_prob > best_thresh
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CRO-S', 'CRO-R'])
        disp.plot()
        plt.show()
    except:
        pass

    # X_TP = X_test[np.logical_and(y_test == 1, y_test == (y_prob > best_thresh))]
    # X_TN = X_test[np.logical_and(y_test == 0, y_test == (y_prob > best_thresh))]
    # X_FP = X_test[np.logical_and(y_test == 1, y_test != (y_prob > best_thresh))]
    # X_FN = X_test[np.logical_and(y_test == 0, y_test != (y_prob > best_thresh))]
    #
    # model_explain_all(model, X_TP)
    # model_explain_all(model, X_TN)
    # model_explain_all(model, X_FP)
    # model_explain_all(model, X_FN)


def search_best_f1(y_test, y_prob, granularity=0.0001):
    def metric_eval(y_test, y_pred):
        C = metrics.confusion_matrix(y_test, y_pred)
        tn = np.float(C[0][0])
        fn = np.float(C[1][0])
        tp = np.float(C[1][1])
        fp = np.float(C[0][1])
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        PPV = tp / (tp + fp) if (tp + fp) != 0 else 0
        NPV = tn / (tn + fn) if (tn + fn) != 0 else 0
        f1 = metrics.f1_score(y_test, y_pred)
        acc = metrics.accuracy_score(y_test, y_pred)
        return sensitivity, specificity, PPV, NPV, f1, acc

    def get_F1(y_test, y_pred):
        C = metrics.confusion_matrix(y_test, y_pred)
        tp = np.float(C[1][1])
        fp = np.float(C[0][1])
        fn = np.float(C[1][0])
        precision = tp / (tp + fp) if (tp + fp) else 0
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) else 0
        return f1

    threshs = np.arange(0.0, 1.0, granularity)
    best_score = 0
    best_thresh = threshs[0]
    for i in range(threshs.shape[0]):
        score = get_F1(y_test, y_prob > threshs[i])
        if score > best_score:
            best_score, best_thresh = score, threshs[i]
        best_metrics = metric_eval(y_test, y_prob > best_thresh)

    return best_metrics, best_thresh


def evaluate_binary(models, X_test, y_test, args):
    # hospital_ids = ["all"] + X_test.columns.tolist()[:10]
    # hospital_ids = ['all', 'hospital_id_2574', 'hospital_id_3148', 'hospital_id_5107', 'hospital_id_6729']
    hospital_ids = ['all']
    rows = []
    # select subgroup in test set
    for g in range(12 + 1):
        if g != 0: continue
        sub_indices = select_subgroup(X_test, group=str(g))
        X_test_sub, y_test_sub = X_test[X_test.index.isin(sub_indices)], \
                                 y_test[y_test.index.isin(sub_indices)]
        print('Testing subgroup {}...'.format(g))

        # instance stage filtering
        for infection_instance in ['all', 'initial', 'subsequent']:
            prefix = 'Subgroup_{}-{}_instances'.format(g, infection_instance)
            print(prefix)
            selected_instances = instance_filter(X_test.index.tolist(), mode=infection_instance)
            X_test_selected, y_test_selected = X_test_sub[X_test_sub.index.isin(selected_instances)], \
                                               y_test_sub[y_test_sub.index.isin(selected_instances)]

            for hospital_id in hospital_ids:
                if hospital_id != "all":
                    # continue
                    selection = X_test_selected[hospital_id] == 1
                    X = X_test_selected[selection]
                    y = y_test_selected[selection]
                else:
                    X = X_test_selected
                    y = y_test_selected

                print('Hospital:', hospital_id)
                print('Number of instances tested {}...'.format(len(y)))
                print('Fraction in test set', "(%0.1f)" % (len(y) / len(y_test) * 100), '...')

                if len(X) == 0:
                    print("No infection instances... Skip...")
                    continue

                print('Positive rate', "(%0.1f)" % (sum(y) / len(y) * 100))
                print('--------------------------------------------')
                AUROC, AUPRC = [], []
                N = 1000
                TPRs, PRECs = np.empty(shape=(5, N)), np.empty(shape=(5, N))
                for i, rs in enumerate(range(args.n_repeat)):
                    # evaluate model
                    auroc, auprc, TPR, PREC = evaluate(models[i], X.values, y.astype(int).values, N=N, verbose=0)
                    AUROC.append(auroc)
                    AUPRC.append(auprc)
                    TPRs[i, :] = TPR
                    PRECs[i, :] = PREC
                    # if i == 0:
                    #     # model_explain_all(models[i], X_test_selected, prefix)
                    #     evaluate_false_samples(models[i], X_test_selected, y_test_selected.astype(int))

                print("AU-ROC:", "%0.4f" % np.mean(AUROC), "(%0.4f)" % np.std(AUROC),
                      "AU-PRC:", "%0.4f" % np.mean(AUPRC), "(%0.4f)" % np.std(AUPRC), )
                print('--------------------------------------------')

                rows.append({
                    "subgroup": g,
                    "n_instances": len(y),
                    "fraction_in_set": len(y) / len(y_test),
                    "positive_rate": sum(y) / len(y),
                    "infection_instance": infection_instance,
                    "hospital_id": hospital_id,
                    "auroc": "%0.2f" % np.mean(AUROC) + " (%0.3f)" % np.std(AUROC),
                    "auprc": "%0.2f" % np.mean(AUPRC) + " (%0.3f)" % np.std(AUPRC),
                    "auroc_mean": np.mean(AUROC),
                    "auroc_std": np.std(AUROC),
                    "auprc_mean": np.mean(AUPRC),
                    "auprc_std": np.std(AUPRC)
                })

                # tpr_mean, tpr_std = np.mean(TPRs, axis=0), np.std(TPRs, axis=0)
                # prec_mean, prec_std = np.mean(PRECs, axis=0), np.std(PRECs, axis=0)
                #
                # x = np.linspace(0, 1, N + 1)[:N]
                # plot_roc(x, tpr_mean, tpr_std, auc=[np.mean(AUROC), np.std(AUROC)], prefix=prefix)
                # plot_prc(x, prec_mean, prec_std, auc=[np.mean(AUPRC), np.std(AUPRC)], prefix=prefix)

    df_results = pd.DataFrame.from_dict(rows, orient='columns')
    df_results.to_csv("../data_analysis/subgroup_performance_hospital.csv")

    return df_results
