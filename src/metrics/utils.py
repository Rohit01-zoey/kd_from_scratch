""" Contains all the metrics used in the project. """
import torch

def accuracy(output, target):
    """ Computes the accuracy for a given output and target. 

    Args:
        output (tensor): The output of the model after softmaxing 
        target (tensor): the labels of the data

    Returns:
        float: the accuracy computer on the output and the targets
    """
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(target)
    return acc