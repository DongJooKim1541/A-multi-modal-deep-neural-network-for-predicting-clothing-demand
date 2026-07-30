import torch
import numpy as np


def focal_loss(logits, labels, criterion_for_focal, alpha, gamma):
    """
    Compute focal loss for imbalanced classification.

    Args:
        logits: Model output logits [B, num_classes]
        labels: Ground truth labels [B]
        criterion_for_focal: CrossEntropyLoss(reduction='none')
        alpha: Focal loss alpha parameter (default 1)
        gamma: Focal loss gamma parameter (default 2)

    Returns:
        Scalar focal loss value
    """
    loss = criterion_for_focal(logits, labels)
    pt = torch.exp(-loss)
    focal = (alpha * (1 - pt) ** gamma * loss).mean()
    return focal


def combined_multitask_loss(pred_best_sex, pred_best_age, pred_view, pred_sales,
                             label_best_sex, label_best_age, label_view, label_sales,
                             criterion, criterion_for_focal, criterion_regression,
                             alpha, gamma, loss_alpha, loss_beta):
    """
    Compute combined multi-task loss: CE(best_sex) + focal(best_age) + α*MSE(view) + β*MSE(sales).

    Args:
        pred_*: Model predictions for each task
        label_*: Ground truth labels for each task
        criterion: CrossEntropyLoss() for best_sex
        criterion_for_focal: CrossEntropyLoss(reduction='none') for focal loss
        criterion_regression: MSELoss() for view and sales
        alpha, gamma: Focal loss parameters
        loss_alpha, loss_beta: Loss weights for view and sales

    Returns:
        Scalar combined loss value
    """
    # CE for best_sex
    loss_sex = criterion(pred_best_sex, label_best_sex)

    # Focal loss for best_age
    loss_age_ce = criterion_for_focal(pred_best_age, label_best_age)
    pt_age = torch.exp(-loss_age_ce)
    loss_age = (alpha * (1 - pt_age) ** gamma * loss_age_ce).mean()

    # MSE for view and sales (regression)
    loss_view = criterion_regression(pred_view, label_view)
    loss_sales = criterion_regression(pred_sales, label_sales)

    # Combined loss
    total_loss = loss_sex + loss_age + loss_alpha * loss_view + loss_beta * loss_sales
    return total_loss


def joint_accuracy(pred_best_sex, pred_best_age, label_best_sex, label_best_age):
    """
    Compute accuracy where both best_sex and best_age predictions are correct.

    Args:
        pred_best_sex, pred_best_age: Predictions [B, num_classes]
        label_best_sex, label_best_age: Labels [B]

    Returns:
        Number of samples where both predictions are correct
    """
    best_sex_pred = pred_best_sex.max(1, keepdim=True)[1]
    best_age_pred = pred_best_age.max(1, keepdim=True)[1]
    correct = (best_sex_pred.eq(label_best_sex.view_as(best_sex_pred)) &
               best_age_pred.eq(label_best_age.view_as(best_age_pred))).sum().item()
    return correct
