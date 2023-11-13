from sklearn.metrics import (average_precision_score, roc_curve, roc_auc_score, precision_recall_curve, log_loss)
import numpy as np
import matplotlib.pyplot as plt

def plot_roc(labels, prediction_scores, legend, color):
    '''
    Function to plot ROC curve
    '''
    fpr, tpr, _   = roc_curve(labels, prediction_scores, pos_label=1)
    auc           = roc_auc_score(labels, prediction_scores)
    legend_string = legend + ' ($AUC = {:0.4f}$)'.format(auc)  
    plt.plot(fpr, tpr, label=legend_string, color=color)
    pass

def plot_prc(labels, prediction_scores, legend, color):
    '''
    Function to plot PRC curve
    '''
    precision, recall, thresholds = precision_recall_curve(labels, prediction_scores)
    average_precision = average_precision_score(labels, prediction_scores)
    legend_string = legend + ' ($AP = {:0.4f}$)'.format(average_precision)  
    plt.plot(recall, precision, label=legend_string, color=color)
    pass

def plot_ks(labels, prediction_scores, color):
    '''
    Function to plot KS plot
    '''
    fpr, tpr, thresholds = roc_curve(labels, prediction_scores, pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    thresholds[0] = 1
    plt.plot(thresholds, fnr, label='FNR (Class 1 Cum. Dist.)', color=color[0], lw=1.5, alpha=0.2)
    plt.plot(thresholds, tnr, label='TNR (Class 0 Cum. Dist.)', color=color[1], lw=1.5, alpha=0.2)

    kss = tnr - fnr
    ks = kss[np.argmax(np.abs(kss))]
    t_ = thresholds[np.argmax(np.abs(kss))]
    
    return ks, t_

def format_plot(title, xlabel, ylabel):
    '''
    Function to add format to plot
    '''
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid('on')
    plt.axis('square')
    plt.ylim((-0.05, 1.05))
    plt.legend()
    plt.tight_layout()
    pass


def plot_ks_2(labels, prediction_scores, legend, color):
    '''
    Function to plot KS plot
    '''
    fpr, tpr, thresholds = roc_curve(labels, prediction_scores, pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    thresholds[0] = 1
    
    kss = tnr - fnr
    ks = kss[np.argmax(np.abs(kss))]
    t_ = thresholds[np.argmax(np.abs(kss))]
    legend_string = f'{legend} ($KS = {ks:0.4f}$; $x = {t_:0.4f}$)'
    plt.plot(thresholds, kss, label=legend_string, color=color, lw=1.5)
    plt.vlines(t_, ks, 0, colors=color, linestyles='dashed', alpha=0.4)
    
    return ks, t_