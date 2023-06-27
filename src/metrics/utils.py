""" Contains all the metrics used in the project. """
import torch
import torch.nn.functional as F

def accuracy(output, target):
    """ Computes the accuracy for a given output and target. 

    Args:
        output (tensor): The output of the model after softmaxing. Shape must be (batch_size, num_classes)
        target (tensor): the labels of the data

    Returns:
        float: the accuracy computed on the output and the targets
    """
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(target)
    return acc

def top_k_accuracy(output, target, k=5):
    """ Computes the top-k accuracy for a given output and target. 

    Args:
        output (tensor): The output of the model after softmaxing. Shape must be (batch_size, num_classes)
        target (tensor): the labels of the data
        k (int, optional): The k value for the top-k accuracy. Defaults to 5.

    Returns:
        float: the top-k accuracy computed on the output and the targets
    """
    with torch.no_grad():
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].view(-1).float().sum(0)
        acc = correct_k / len(target)
    return acc


def ce_loss(output, target):
    """ Computes the cross entropy loss for a given output and target. 

    Args:
        output (tensor): The output of the model after softmaxing. Shape must be (batch_size, num_classes)
        target (tensor): the labels of the data

    Returns:
        float: the cross entropy loss computed on the output and the targets
    """
    return F.cross_entropy(output, target, reduction = 'mean') # return the cross entropy loss averaged across the batch

def kld_loss(output, target, T):
    """ Computes the KL divergence loss for a given output and target. 

    Args:
        output (tensor): The output of the model after softmaxing. Shape must be (batch_size, num_classes)
        target (tensor): the labels of the data
        T (float): the temperature for the softmax for the softening of the labels for the knowledge distillation

    Returns:
        float: the KL divergence loss computed on the output and the targets
    """    
    loss_KD = torch.nn.KLDivLoss(reduction='none')(torch.nn.functional.log_softmax(output/T, dim=-1), torch.nn.functional.softmax(target/T, dim=-1))
    loss =  T * T * torch.sum(loss_KD, dim=1) # multiply the loss by the temperature squared as suggested in the paper
    loss = torch.mean(loss) # get the mean of the loss across the batch
    return loss


def kd_loss(output, target, T, alpha):
    """ Computes the knowledge distillation loss for a given output and target. 

    Args:
        output (tensor): The output of the model after softmaxing. Shape must be (batch_size, num_classes)
        target (tensor): the labels of the data
        T (float): the temperature for the softmax for the softening of the labels for the knowledge distillation
        alpha (float): the weight for the cross entropy loss

    Returns:
        float: the knowledge distillation loss computed on the output and the targets
    """    
    return alpha * ce_loss(output, target) + (1 - alpha) * kld_loss(output, target, T) # compute the weighted sum of the cross entropy loss and the KL divergence loss