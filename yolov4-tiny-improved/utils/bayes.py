import numpy as np
import torch
import torch.nn.functional as F


def bayes_modify(image_pred, sure_mask, value_mask, pre_prob, con_prob):
    candidates = image_pred[value_mask, 5:25]
    fight = torch.topk(candidates, 2).values
    winner = torch.topk(candidates, 4).indices
    weak_index = [i for i in range(len(fight)) if fight[i, 1] * 1.5 >= fight[i, 0]]
    sure_obj = image_pred[sure_mask]
    class_conf, sure_label = torch.max(sure_obj[:, 5:25], 1)
    sure_labels = []
    for label in sure_label:
        sure_labels.append(label.item())
    sure_label = set(sure_labels)
    bayes_weight = pre_prob
    for label in sure_label:
        for target in range(20):
            weight = con_prob[target, label] / pre_prob[label]
            bayes_weight[target] *= F.sigmoid(torch.tensor(weight))

    bayes_weight = torch.tensor(bayes_weight, device="cuda")
    bayes_weight = F.softmax(bayes_weight, dim=0) * len(bayes_weight)

    value_obj = image_pred[value_mask]
    for index in weak_index:
        a, b, c, d = [winner[index, i].item() for i in range(4)]
        value_obj[index, [a, b, c, d]] *= bayes_weight[[a, b, c, d]]
    image_pred[value_mask] = value_obj

    return image_pred

