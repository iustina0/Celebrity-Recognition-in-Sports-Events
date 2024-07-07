import numpy as np
from scipy import interpolate
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import KFold


def evaluate_data(distances, labels, num_folds=10, far_target=1e-3):
    thresholds_roc = np.arange(0, 2, 0.01)
    true_positive_rate, false_positive_rate, precision, recall, accuracy, best_distances = \
        calculate_roc_values(
            thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
        )
    
    tar, far, best_threshold = calculate_val(
        distances=distances, labels=labels, far_target=far_target
    )

    roc_auc = auc(false_positive_rate, true_positive_rate)

    precision, recall, _ = precision_recall_curve(labels, -distances)

    return true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, tar, far, best_threshold


def calculate_val(distances, labels, far_target=1e-3):
    thresholds_val = np.arange(0.4, 2, 0.001)
    num_thresholds = len(thresholds_val)
    
    tar_train = np.zeros(num_thresholds)
    far_train = np.zeros(num_thresholds)
    youden_index = np.zeros(num_thresholds)

    for threshold_index, threshold in enumerate(thresholds_val):
        tar_train[threshold_index], far_train[threshold_index] = calculate_val_far(
            threshold=threshold, dist=distances, actual_issame=labels
        )
        youden_index[threshold_index] = tar_train[threshold_index] - far_train[threshold_index]

    best_threshold_index = np.argmax(youden_index)
    best_threshold = thresholds_val[best_threshold_index]
    best_tar = tar_train[best_threshold_index]
    best_far = far_train[best_threshold_index]

    return best_tar, best_far, best_threshold

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)

    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    num_same = np.sum(actual_issame)
    num_diff = np.sum(np.logical_not(actual_issame))

    if num_diff == 0:
        num_diff = 1
    if num_same == 0:
        return 0, 0

    tar = float(true_accept) / float(num_same)
    far = float(false_accept) / float(num_diff)

    return tar, far


def calculate_metrics(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)

    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, precision, recall, accuracy


def calculate_roc_values(thresholds, distances, labels, num_folds=10):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    true_positive_rates = np.zeros((num_folds, num_thresholds))
    false_positive_rates = np.zeros((num_folds, num_thresholds))
    precision = np.zeros(num_folds)
    recall = np.zeros(num_folds)
    accuracy = np.zeros(num_folds)
    best_distances = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        accuracies_trainset = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            _, _, _, _, accuracies_trainset[threshold_index] = calculate_metrics(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        best_threshold_index = np.argmax(accuracies_trainset)
        for threshold_index, threshold in enumerate(thresholds):
            true_positive_rates[fold_index, threshold_index], false_positive_rates[fold_index, threshold_index], _, _,\
                _ = calculate_metrics(
                    threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
                )

        _, _, precision[fold_index], recall[fold_index], accuracy[fold_index] = calculate_metrics(
            threshold=thresholds[best_threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
        )

        true_positive_rate = np.mean(true_positive_rates, 0)
        false_positive_rate = np.mean(false_positive_rates, 0)
        best_distances[fold_index] = thresholds[best_threshold_index]

    return true_positive_rate, false_positive_rate, precision, recall, accuracy, best_distances