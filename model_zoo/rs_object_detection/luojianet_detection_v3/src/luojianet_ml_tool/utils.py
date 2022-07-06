# -*- coding:utf-8 -*-
from src.luojianet_metric_learning import losses, miners, distances, reducers, testers
from src.luojianet_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import numpy as np
import matplotlib.pyplot as plt


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    loss_record = []
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss_record.append(loss)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, loss,
                                                                                           mining_func.num_triplets))
    return loss_record


# convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


# compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(trainset, testset, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(trainset, model)
    test_embeddings, test_labels = get_all_embeddings(testset, model)
    print("Computing accuracy......")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings,
                                                  train_embeddings,
                                                  np.squeeze(test_labels),
                                                  np.squeeze(train_labels),
                                                  False)
    # print("Test set accuracy (MAP@5) = {}".format(accuracies["mean_average_precision_at_r"]))
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    # print("Test set map = {}".format(accuracies['mean_average_precision']))
    # return accuracies["mean_average_precision_at_r"]
    return accuracies["precision_at_1"]
    # return accuracies['mean_average_precision']


def StepLR(optimizer, lr, lr_list, cur_epoch):
    if cur_epoch not in lr_list:
        return lr

    print('######### lr change: {} -> {} #########'.format(lr, lr / 10))
    lr = lr / 10
    for index, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr

    return lr
