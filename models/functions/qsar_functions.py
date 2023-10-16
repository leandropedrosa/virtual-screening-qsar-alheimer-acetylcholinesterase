import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import permutation_test_score
import pandas as pd
# Algorítimos
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, plot_roc_curve, auc, roc_curve, roc_auc_score, confusion_matrix
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold
from nested_cv import NestedCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

class QSAR:
    def __init__(self):
        pass
    
    def stats(y_test, y_pred):
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[0,1])
        Kappa = metrics.cohen_kappa_score(y_test, y_pred, weights='linear')
        # Valores verdadeiros e falsos
        TN, FP, FN, TP = confusion_matrix.ravel()
        # Accuracy
        AC = (TP+TN)/(TP+FP+FN+TN)
        # Sensibilidade, taxa de acerto, recall ou taxa positiva verdadeira
        SE = TP/(TP+FN)
        # Especificidade ou taxa negativa verdadeira
        SP = TN/(TN+FP)
        # Precisão ou valor preditivo positivo
        PPV = TP/(TP+FP)
        # Valor preditivo negativo
        NPV = TN/(TN+FN)
        # Taxa de classificação correta
        CCR = (SE + SP)/2   
        # F1 Score
        F1_score = 2*(PPV*SE)/(PPV+SE)
        d = dict({'Kappa': Kappa,
             'AUC': CCR,
             'Sensibilidade': SE,
             'PPV': PPV,
             'Especificidade': SP,
             'NPV': NPV,
             'Acurácia': AC,
             'F1 Score':F1_score})
        return pd.DataFrame(d, columns=d.keys(), index=[0]).round(2)

    def report(model, y_test, y_pred, y_prob, algoritmo, descritores):
        # imprimir relatório de classificação
        print("Relatório de Classificação - " + descritores + "X" + algoritmo + ":\n", classification_report(y_test, y_pred, digits=4))
        # imprimir a área sob a curva
        print("AUC: {:.4f}\n".format(roc_auc_score(y_test, y_prob[:, 1])))

    def confusion_mtx(y_pred, y_test, algoritmo, descritores):
        fig, ax = plt.subplots()
        cm = metrics.confusion_matrix(y_test, y_pred)
        print('confusion_matrix:')
        print(cm)
        sns.heatmap(cm, annot=True, 
                    ax=ax, fmt='d', cmap='Reds')
        ax.set_title("Matriz de Confusão - "+descritores+'X'+algoritmo, fontsize=18)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted Label")
        plt.savefig('figures/'+descritores+'X'+algoritmo+'confusion-matrix.png', bbox_inches='tight',
                    transparent=False, format='png', dpi=300)
        plt.tight_layout()

    def precisao_recall(y_pred, y_test, algoritmo, descritores):
        pos_probs = y_pred[:, 1]
       
        precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, pos_probs)
        print('precisions:')
        print(precisions)
        print('recalls:')
        print(recalls)
        print('thresholds:')
        print(thresholds)
        fig, ax = plt.subplots(figsize = (12,3))
        plt.plot(thresholds, precisions[:-1], 'b--', label = 'Precisão')
        plt.plot(thresholds, recalls[:-1], 'g-', label = 'Recall')
        plt.xlabel('Threshold')
        plt.legend(loc = 'center right')
        plt.ylim([0,1])
        plt.title('Precisão x Recall - '+descritores+'X'+algoritmo, fontsize = 14)
        plt.savefig('figures/'+descritores+'X'+algoritmo+'precisao-recall.png', bbox_inches='tight',
                transparent=False, format='png', dpi=300)
        plt.show()

    def y_randomization(clf, X_test, y_test, descriptor, algoritmo):    
        X_test = pd.DataFrame(X_test)
        y_test = np.ravel(y_test)

        permutations = 10
        score, permutation_scores, pvalue = permutation_test_score(clf, X_test, y_test,
                                                                   cv=cross_val, scoring='balanced_accuracy',
                                                                   n_permutations=permutations,
                                                                   n_jobs=-1,
                                                                   verbose=1,
                                                                   random_state=seed)
        print('True score = ', score.round(2),
              '\n Média per. = ', np.mean(permutation_scores).round(2),
              '\np-value = ', pvalue.round(4))

        ###############################################################################
        # View histogram of permutation scores
        pl.subplots(figsize=(10,6))
        pl.hist(permutation_scores.round(2), label='Permutation scores')
        ylim = pl.ylim()
        pl.vlines(score, ylim[0], ylim[1], linestyle='--',
                  color='g', linewidth=3, label='Classification Score'
                  ' (pvalue %s)' % pvalue.round(4))
        pl.vlines(1.0 / 2, ylim[0], ylim[1], linestyle='--',
                  color='k', linewidth=3, label='Luck')
        pl.ylim(ylim)
        pl.legend()
        pl.xlabel('Score')
        pl.title('Aleatoriarização da variável Y '+algoritmo+'X'+descriptor, fontsize=12)
        pl.savefig('figures/y_randomization-'+descriptor+'X'+algoritimo+'.png', bbox_inches='tight', transparent=False, format='png', dpi=300)
        pl.show()
        
    def get_best_model_with_random_search(y_train, X_train, param_dist, cross_val, algoritimo, seed):
        best_model = None
        best_score = float('-inf')  # Inicializa com um valor muito baixo

        # Do the 5-fold loop with RandomizedSearchCV
        for train_index, test_index in cross_val.split(X_train, y_train):
            # Split the data
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

            # Create a RandomizedSearchCV object
            random_search = RandomizedSearchCV(
                algoritimo,
                param_distributions=param_dist, 
                n_iter=10,  # Você pode ajustar o número de iterações
                cv=cross_val,  # Usando a validação cruzada estratificada
                random_state=seed,
                verbose=1,
                n_jobs=None  # Use todos os núcleos disponíveis para acelerar a busca
            )

            # Fit the RandomizedSearchCV to your data
            random_search.fit(X_train_fold, y_train_fold)
            print(random_search.best_estimator_)
            pred = random_search.predict(X_test_fold)

            # calc statistics
            print("Accuracy = ", accuracy_score(y_test_fold, pred))
            print("MCC = ", matthews_corrcoef(y_test_fold, pred))
            print("Kappa = ", cohen_kappa_score(y_test_fold, pred))
            print("Confusion Matrix = \n",metrics.confusion_matrix(y_test_fold, pred))
            print("Classification Report = \n",metrics.classification_report(y_test_fold, pred))

            # Check if the current model is the best one
            if random_search.best_score_ > best_score:
                best_score = random_search.best_score_
                best_model = random_search

        return best_model

    def validation(y_ts, x_ts, cross_val, m):
        # Params
        pred = []
        ad = []
        index = []
        df_ts = pd.DataFrame(x_ts)

        # Do 5-fold loop
        for train_index, test_index in cross_val.split(df_ts, y_ts):

            fold_model = m.fit(df_ts.iloc[train_index], y_ts[train_index])
            fold_pred = m.predict(df_ts.iloc[test_index])
            fold_ad = m.predict_proba(df_ts.iloc[test_index])
            pred.append(fold_pred)
            ad.append(fold_ad)
            index.append(test_index)

        # Prepare results to export    
        fold_index = np.concatenate(index)    
        fold_pred = np.concatenate(pred)
        fold_ad = np.concatenate(ad)
        fold_ad = (np.amax(fold_ad, axis=1) >= 0.8).astype(str)
        five_fold_model = pd.DataFrame({'Prediction': fold_pred,'AD': fold_ad}, index=list(fold_index))
        five_fold_model.AD[five_fold_model.AD == 'False'] = np.nan
        five_fold_model.AD[five_fold_model.AD == 'True'] = five_fold_model.Prediction
        five_fold_model.sort_index(inplace=True)
        five_fold_model['y_train'] = pd.DataFrame(y_ts)
        five_fold_model_ad = five_fold_model.dropna().astype(int)
        coverage_5f = len(five_fold_model_ad) / len(five_fold_model)

        # model stats
        model = pd.DataFrame(stats(five_fold_model['y_train'], five_fold_model['Prediction']))
        model['Coverage'] = 1.0

        # model AD stats
        model_ad = five_fold_model.dropna(subset=['AD']).astype(int)
        coverage_model_ad = len(model_ad['AD']) / len(five_fold_model['y_train'])
        model_ad = pd.DataFrame(stats(model_ad['y_train'], model_ad['AD']))
        model_ad['Coverage'] = round(coverage_model_ad, 2)

        # print stats
        print('\033[1m' + '5-fold External Cross Validation Statistical Characteristcs of QSAR models developed '+descritores+ '\n' + '\033[0m')
        model_5f_stats = model.append(model_ad)
        model_5f_stats.set_index([[descritores, descritores+' AD']], drop=True, inplace=True)
        return model_5f_stats
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    def print_stats(morgan_stats, algoritimo, descritores):
        # Estatísticas de transposição
        morgan_stats_t = morgan_stats.T
        morgan_stats_t = morgan_stats_t.reset_index()
        morgan_stats_t = morgan_stats_t.rename(columns={'index': 'Stats'})

        # Fazer enredo
        plt.style.use('seaborn-colorblind')
        fig, ax1 = plt.subplots(figsize=(10,6))

        morgan_stats_t.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_xticklabels(labels=morgan_stats_t['Stats'].tolist(), fontsize=14, rotation=0)
        ax1.axhline(y=.8, color='indianred', ls='dashed')# xmin=0.25, xmax=0.75)
        ax1.legend_.remove()
        plt.title('Características estatísticas - '+descritores+'X'+algoritimo, fontsize=16)
        ax1.set_yticks(np.arange(0, 1.1, 0.1))
        ax1.tick_params(labelsize=12)

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, fontsize=16,
                    loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True,
                    shadow=True, ncol=2)
        fig.tight_layout()

        plt.savefig('figures/'+descritores+'X'+algoritimo+'statistics-'+descritores+'.png', bbox_inches='tight',
                    transparent=False, format='png', dpi=300)
        plt.show();
        
    def roc_auc(rf_best, cross_val, X_train, y_train, algoritimo, descritores):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        X_train = pd.DataFrame(X_train)

        fig, ax = plt.subplots(figsize=(10,6))
        for i, (train_index, test_index) in enumerate(cross_val.split(X_train, y_train)):
            rf_best.fit(X_train.iloc[train_index], y_train[train_index])
            viz = plot_roc_curve(rf_best, X_train.iloc[test_index], y_train[test_index],
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8, )

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Média ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Característica operacional do receptor "+descritores+"-"+algoritimo)
        ax.legend(loc="lower right")
        plt.savefig('figures/'+descritores+'X'+algoritimo+'roc-auc.png', bbox_inches='tight',
        transparent=False, format='png', dpi=300)

        plt.show()