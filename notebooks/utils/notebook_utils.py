
from torch import nn
import torch

def calc_ae_perf_metrics(reconstructions, inputs, labels, threshold, print_all=True):

    loss_fn = nn.MSELoss(reduce=False)
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    
    Recall = -1
    Precision = -1
    Accuracy = -1
    F1 = -1
    FPR = -1
    TPR = -1

    preds = []
    targets = []
    
    for i in range(len(inputs)):
        label_batch = labels[i]
        reconstructions_batch = reconstructions[i]
        inputs_batch = inputs[i]
        
        loss = loss_fn(reconstructions_batch, inputs_batch)

        loss_batch = loss.mean(dim=(1,2))
        prediction_batch = loss_batch > threshold
        # print(prediction_batch)
        # break

        TN += torch.sum(torch.logical_and(torch.logical_not(prediction_batch), torch.squeeze(torch.logical_not(label_batch))))
        TP += torch.sum(torch.logical_and((prediction_batch), torch.squeeze(label_batch)))
        FN += torch.sum(torch.logical_and(torch.logical_not(prediction_batch), torch.squeeze(label_batch)))
        FP += torch.sum(torch.logical_and((prediction_batch), torch.squeeze(torch.logical_not(label_batch))))

        # TN += torch.sum(torch.logical_and((prediction_batch==0), (label_batch==0)))
        # TP += torch.sum(torch.logical_and((prediction_batch==1), (label_batch==1)))
        # FN += torch.sum(torch.logical_and((prediction_batch==0), (label_batch==1)))
        # FP += torch.sum(torch.logical_and((prediction_batch==1), (label_batch==0)))


    if TP + FN != 0:
        Recall = TP / (TP + FN)

    if TP + FP != 0:
        Precision = TP / (TP + FP)

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    if (TN + FP) != 0:
        FPR = FP / (TN + FP)

    if (TP + FN) != 0:
        TPR = TP / (TP + FN)

    if Precision + Recall != 0:
        F1 = 2*(Precision * Recall) / (Precision + Recall)

    if print_all:
        print(f"TP: {TP}")
        print(f"FP: {FP}")
        print(f"TN: {TN}")
        print(f"FN: {FN}")
        print()
        
        print(f"FPR: {FPR}")
        print(f"TPR = Recall: {TPR}")
        print(f"Recall: {Recall}")
        print(f"Precision: {Precision}")
        print(f"Accuracy: {Accuracy}")
        print(f"F1: {F1}")

    return FPR, TPR, Recall, Precision, Accuracy, F1