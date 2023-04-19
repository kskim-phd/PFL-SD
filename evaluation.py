import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import auc, roc_auc_score, roc_curve

currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-1])
root_current = '/'.join(currentpath)

def calculate_accuracy(y_true, y_score):
    N = y_true.shape[0]
    accuracy = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_score, axis=-1)) / N
    return accuracy

threshold = 0.5
patchnum = 30
train_no = 4  # choose training No. 2, 3, 4
ver = 2  # choose ver 1 or 2 (PFL-SD)

if train_no == 2:
    patientnum = 57
elif train_no == 3:
    patientnum = 53
elif train_no == 4:
    patientnum = 49

labels = ['snoring','stridor']
num_classes = 2
tprs = []
aucs = []
tprs_macro = []
aucs_macro = []
mean_aucs = []
mean_aucs_macro = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
colors = ["lightpink", "orchid", "orange", "gold", "yellowgreen", "powderblue", "teal", "mediumpurple",
              "lightslategray", "red", "green"]

NUM_FOLD = [1,2,3,4,5,6,7,8,9,10]
for fold in NUM_FOLD:
    TOTAL_AUC_SCORE = []

    if train_no == 2:
        clipwise_output1=np.load(root_current+"/evaluation/ver{}/{}train/{}fold/pred1_150sec.npy".format(ver,train_no,fold))  
        clipwise_output2=np.load(root_current+"/evaluation/ver{}/{}train/{}fold/pred2_150sec.npy".format(ver,train_no,fold))  
        clipwise_output3=np.load(root_current+"/evaluation/ver{}/{}train/{}fold/pred3_150sec.npy".format(ver,train_no,fold))  
    else :
        clipwise_output1=np.load(root_current+"/evaluation/ver{}/{}train/{}fold/pred1_150sec.npy".format(ver,train_no,fold))  
        clipwise_output2=np.load(root_current+"/evaluation/ver{}/{}train/{}fold/pred2_150sec.npy".format(ver,train_no,fold))  
        clipwise_output3=np.load(root_current+"/evaluation/ver{}/{}train/{}fold/pred3_150sec.npy".format(ver,train_no,fold))  
        clipwise_output4=np.load(root_current+"/evaluation/ver{}/{}train/{}fold/pred4_150sec.npy".format(ver,train_no,fold))  
    

    if ver == 1:
        if train_no == 2:
            clipwise_output = np.concatenate((clipwise_output1[:15],clipwise_output2,clipwise_output3,clipwise_output1[15:]),axis=0) #2train
        elif train_no == 3:
            clipwise_output = np.concatenate((clipwise_output1[:13],clipwise_output2,clipwise_output3,clipwise_output4,clipwise_output1[13:]),axis=0) #3train
        elif train_no == 4:
            clipwise_output = np.concatenate((clipwise_output1[:11],clipwise_output2,clipwise_output3,clipwise_output4,clipwise_output1[11:]),axis=0) #4train
    
    if ver == 2:
        if train_no == 2:
            clipwise_output = np.concatenate((clipwise_output1[:450],clipwise_output2,clipwise_output3,clipwise_output1[450:]),axis=0) #2train
        elif train_no == 3:
            clipwise_output = np.concatenate((clipwise_output1[:390],clipwise_output2,clipwise_output3,clipwise_output4,clipwise_output1[390:]),axis=0) #3train
        elif train_no == 4:
            clipwise_output = np.concatenate((clipwise_output1[:330],clipwise_output2,clipwise_output3,clipwise_output4,clipwise_output1[330:]),axis=0) #4train
    
    y_test = pd.read_csv(root_current+'/evaluation/label/{}train/snoring5sec1_final_target_2D.csv'.format(train_no),header=None, index_col=None)
    
    if ver == 1:
        y_score = clipwise_output
    
    if ver == 2:
        pred = []
        for i in range(int(clipwise_output.shape[0]/patchnum)):
            clipwise = np.mean(clipwise_output[patchnum*(i):patchnum*(i+1)],axis=0)
            pred.extend(clipwise) 
        pred = np.array(pred)
        pred = pred.reshape(patientnum,2)
        y_score = pred

    
    y_test = np.array(y_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    cm = metrics.confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(y_score, axis=-1), labels=None)
    Precision = metrics.precision_score(np.argmax(y_test, axis=-1), np.argmax(y_score, axis=-1))
    Recall = metrics.recall_score(np.argmax(y_test, axis=-1), np.argmax(y_score, axis=-1))
    accuracy = calculate_accuracy(np.argmax(y_test, axis=-1), np.argmax(y_score, axis=-1))

    print("################################ {}FOLD - CONFUSION MATRIX & INFO ###############################".format(fold))
    print(metrics.classification_report(np.argmax(y_test, axis=-1), np.argmax(y_score, axis=-1), target_names=labels, digits=4))
   
    Specificity = 100 * cm[0][0] / (cm[0][0] + cm[0][1])
    Sensitivity = 100 * cm[1][1] / (cm[1][1] + cm[1][0])
    PPV = 100 * cm[0][0] / (cm[0][0] + cm[1][0])
    NPV = 100 * cm[1][1] / (cm[1][1] + cm[0][1])
    F1_Scores = metrics.f1_score(np.argmax(y_test, axis=-1), np.argmax(y_score, axis=-1))
    print("Sensitivity: {}".format(Sensitivity))  
    print("Specificity: {}".format(Specificity))  
    print("precision: {}".format(Precision))
    print("recall: {}".format(Recall))  
    print("PPV: {}".format(PPV))
    print("NPV: {}".format(NPV))  
    print("F1 Scores: {}".format(F1_Scores))
    print('Confusion Matrix: ')
    print(cm)

    
    i=0
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], thresholds = roc_curve(y_test.ravel(), y_score.ravel())
    tprs.append(np.interp(mean_fpr, fpr["micro"], tpr["micro"]))
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    aucs.append(roc_auc["micro"])  # different

    plt.plot(fpr["micro"], tpr["micro"],
                label='FOLD {}  (AUC = %0.4f)'.format(fold) % (roc_auc["micro"]),
                color=colors[fold],
                linewidth=1.5)
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes


    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    tprs_macro.append(np.interp(mean_fpr, fpr["macro"], tpr["macro"]))
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    aucs_macro.append(roc_auc["macro"])
    """
    plt.plot(fpr["macro"], tpr["macro"],
                label='FOLD {}  (AUC = %0.4f)'.format(PRE_NUM_FOLD + 1) % (roc_auc["macro"]),
                color=colors[PRE_NUM_FOLD],
                linewidth=1.5)
    """

    print('FOLD{} micro {}'.format(fold, roc_auc_score(y_test, y_score, average='micro')))
    print('FOLD{} macro {}'.format(fold,  roc_auc["macro"]))

    #############
    # MEAN.
    #############

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gainsboro', label='Chance',
        alpha=.8)  # CENTER LINE.
###
mean_tpr_macro = np.mean(tprs_macro, axis=0)
mean_tpr_macro[-1] = 1.0
mean_auc_macro = auc(mean_fpr, mean_tpr_macro)

mean_aucs.append(mean_auc_macro)
std_auc_macro = np.std(aucs_macro)
#TOTAL_AUC_SCORE_macro.append(mean_auc_macro)
ax.plot(mean_fpr, mean_tpr_macro, color='tomato',
        label=r'Macro-avg ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_auc_macro, std_auc_macro), lw=2.5, alpha=.8, linestyle='dashed')
std_tpr_macro = np.std(tprs_macro, axis=0)
tprs_upper_macro = np.minimum(mean_tpr_macro + std_tpr_macro, 1)
tprs_lower_macro = np.maximum(mean_tpr_macro - std_tpr_macro, 0)
###

###
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

mean_aucs.append(mean_auc)
std_auc = np.std(aucs)
TOTAL_AUC_SCORE.append(mean_auc)
ax.plot(mean_fpr, mean_tpr, color='dodgerblue',
        label=r'Micro-avg ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2.5, alpha=.8, linestyle='dashed')

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])  # , xlabel='1-Specificity',ylabel='Sensitivity')  # , xlabel='False Positive Rate', ylabel='True Positive Rate')
# set labels and font size
ax.set_ylabel('Sensitivity', fontsize=13)
ax.set_xlabel('1-Specificity', fontsize=13)


plt.rcParams['hatch.linewidth'] = 0.5
plt.legend(loc="lower right", fontsize=10, frameon=True)  ###############
save_folder = root_current+'/evaluation/ver{}/{}train'.format(ver,train_no)
os.makedirs(save_folder, exist_ok=True)
ax.figure.savefig(
    save_folder + '/Micro_{}_Macro_{}_ROC_curve.png'.format(mean_auc, mean_auc_macro))
plt.show()
plt.close('all')