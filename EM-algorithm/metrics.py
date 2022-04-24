from typing import List, Tuple
import numpy as np
from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """
    intersection_w = []
    pred_w = []
    for pred, ref in zip(predicted, reference):
        ha = set(ref.possible) | set(ref.sure)
        intersection_w.append(len(set(pred) & ha))
        pred_w.append(len(set(pred)))
    
    return (sum(intersection_w), sum(pred_w))


def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    intersection_w = []
    sure_w = []
    for pred, ref in zip(predicted, reference):
        intersection_w.append(len(set(pred) & set(ref.sure)))
        sure_w.append(len(set(ref.sure)))
    
    return (sum(intersection_w), sum(sure_w))


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    numerator = compute_recall(reference, predicted)[0] + compute_precision(reference,predicted)[0]
    denominator = compute_recall(reference, predicted)[1] + compute_precision(reference,predicted)[1]
    return 1 - numerator/denominator