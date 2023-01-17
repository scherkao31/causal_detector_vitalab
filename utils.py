import torch
import torch.nn as nn
import random
import numpy as np

import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
'''
def predict(classifier, val_loader_):
    criterion = nn.BCELoss()
    preds = []
    labs = []
    scores = []
    for inputs, _, labels, _, _ in val_loader_:
        outputs = classifier(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        labs += labels.tolist()
        preds += torch.round(outputs.squeeze(1)).tolist()
        scores += outputs.squeeze(1).tolist()
    acc = sklearn.metrics.accuracy_score(labs, preds)
    f1 = sklearn.metrics.f1_score(labs, preds)
    return labs, preds, acc, f1, scores
'''

def predict(classifier, val_loader_):
    criterion = nn.BCELoss()
    preds = []
    labs = []
    scores = []
    traj = []
    ades = []
    for inputs, inputs_agents, labels, ade, _ in val_loader_:
        outputs = classifier(inputs, inputs_agents)
        #print('out', outputs.type(), labels.type())
        loss = criterion(outputs, labels.unsqueeze(1))
        labs += labels.tolist()
        preds += torch.round(outputs.squeeze(1)).tolist()
        scores += outputs.squeeze(1).tolist()
        traj += inputs
        #print(ade)
        ades += ade.tolist()
    acc = sklearn.metrics.accuracy_score(labs, preds)
    f1 = sklearn.metrics.f1_score(labs, preds)
    return labs, preds, acc, f1, scores, traj, ades


def predict_stgat(classifier, val_loader_):
    
    preds = []
    labs = []
    scores = []
    traj = []
    ades = []
    for inputs, inputs_agents, labels, ade, _ in val_loader_:
        factual = inputs_agents[:, 0, :, :]
        outputs = classifier(inputs, factual, training_step_ = 3)
        #print('out', outputs.type(), labels.type())
        #loss = criterion(outputs, labels.unsqueeze(1))
        labs += labels.tolist()
        preds += torch.round(outputs.squeeze(1)).tolist()
        scores += outputs.squeeze(1).tolist()
        traj += inputs
        #print(ade)
        ades += ade.tolist()
    acc = sklearn.metrics.accuracy_score(labs, preds)
    f1 = sklearn.metrics.f1_score(labs, preds)
    return labs, preds, acc, f1, scores, traj, ades


def predict_ic(classifier, val_loader_):
    criterion = nn.BCELoss()
    preds = []
    labs = []
    scores = []
    traj = []
    ades = []
    
    for inputs, inputs_agents, labels in val_loader_:
        outputs = classifier(inputs, inputs_agents)
        labs += labels.tolist()
        preds += torch.round(outputs.squeeze(1)).tolist()
    acc = sklearn.metrics.accuracy_score(labs, preds)
    
    return acc

def predict_ic_stgat(classifier, val_loader_):
    preds = []
    labs = []
    
    for inputs, inputs_agents, labels in val_loader_:
        factual = inputs_agents[:, 0, :, :]
        
        outputs = classifier(inputs, factual, training_step_ = 3)
        labs += labels.tolist()
        preds += torch.round(outputs.squeeze(1)).tolist()
        
    acc = sklearn.metrics.accuracy_score(labs, preds)
    
    return acc
    

def ade_val(classifier, val_loader_, training_step_, causal = False):
    criterion = nn.MSELoss()
    losses = []
    for inputs, inputs_agents, labels, _, _ in val_loader_:
        outputs = classifier(inputs, inputs_agents, training_step = training_step_)
        #print('out', outputs.type(), labels.type())
        if causal == True:
            loss = criterion(outputs, inputs_agents)
        else:
            loss = criterion(outputs, inputs)
        losses += [loss.detach()]
        
    return np.mean(losses)

def ade_val_stgat(classifier, val_loader_, training_step_):
# First, we need to define a loss function and an optimizer
    mse = nn.MSELoss()
    loss_epoch = []
    # Loop over the training data in batches
    for inputs, inputs_agents, labels, _, _ in val_loader_:
        factual = inputs_agents[:, 0, :, :]
        # Forward pass
        outputs = classifier(inputs, factual, training_step_ = training_step_)
        
        target = factual[:, :, 1:20, :]
        loss = mse(outputs, target)
        
        loss_epoch += [loss.detach()]
        
    return np.mean(loss_epoch)
    
    
    
    
def plot_result_matrix(classifier, val_loader_, stgat = False):
    if stgat:
        target, pred, acc, f1, scores, _, _ = predict_stgat(classifier, val_loader_)
    else:
        target, pred, acc, f1, scores, _, _ = predict(classifier, val_loader_)

    cf_matrix = sklearn.metrics.confusion_matrix(target, pred)

    labels = ['True Neg','False Pos','False Neg','True Pos']
    categories = ['Zero', 'One']
    make_confusion_matrix(cf_matrix, 
                          group_names=labels,
                          categories=categories, 
                          )

    
def plot_graph_score_ade(classifier, val_loader_, size_points = 50.0, stgat = False):
    if stgat:
        target, pred, acc, f1, scores, trajs, ades = predict_stgat(classifier, val_loader_)
    else:
        target, pred, acc, f1, scores, trajs, ades = predict(classifier, val_loader_)

    ades = np.array(ades)
    scores = np.array(scores)

    pred_labels = torch.tensor(pred)
    target_labels = torch.tensor(target)

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.logical_and(pred_labels == 1, target_labels == 0)
    FP = torch.where(FP == 1)[0]
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.logical_and(pred_labels == 0, target_labels == 1)
    FN = torch.where(FN == 1)[0]

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    TP = np.logical_and(pred_labels == 1, target_labels == 1)
    TP = torch.where(TP == 1)[0]
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    TN = np.logical_and(pred_labels == 0, target_labels == 0)
    TN = torch.where(TN == 1)[0]

    results = [TP, TN, FP, FN]
    results_str = ['TP', 'TN', 'FP', 'FN']

    np.random.seed(19680801)

    fig, ax = plt.subplots()
    i = 0
    for color in ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']:
        x = scores[results[i]]
        n = len(x)
        y = ades[results[i]]
        scale = size_points #200.0 * np.random.rand(n)
        #print(scale)
        color = ax.scatter(x, y, c=color, s=scale, label= results_str[i],
                   alpha=0.3, edgecolors='none')
        ax.label='Inline label'
        i += 1
    plt.xlabel("Score prediction")
    plt.ylabel("Ade f-cf (causality)")
    plt.xlim([0, 1.])
    ax.legend()
    ax.grid(True)

    plt.show()
    
    
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            specificity = cf[0,0] / sum(cf[0,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nSpecificity={:0.3f}".format(
                accuracy,precision,recall,f1_score, specificity)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)