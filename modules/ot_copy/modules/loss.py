import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    
import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F


def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[0]
    query_copies = query.repeat(int(num_pos), 1)
    diff = ((pos_vecs - query_copies) ** 2).sum(1)

    min_pos, _ = diff.min(0)
    max_pos, _ = diff.max(0)
    return min_pos, max_pos


def mean_squared_error_loss(anchor_vector, pos_vectors, neg_vectors, pos_overlaps, neg_overlaps, alpha=1.0):
    """
    First calculate the cosine similarity between two batch of vectors.
    Then calculate the mean squared error between the similarity and overlaps.

    Args:
        anchor_vector: (torch.Tensor) scan1 vectors in shape (1, vec_size).
        pos_vectors: (torch.Tensor) scan2 vectors in shape (num, vec_size).
        neg_vectors: (torch.Tensor) scan3 vectors in shape (num, vec_size).
        pos_overlaps: (torch.Tensor) the overlaps between two anchor vector and pos_vectors (1, num_pos).
        neg_overlaps: (torch.Tensor) the overlaps between two anchor vector and neg_vectors (1, num_neg).
        alpha: (float) the parameter balance the loss between positive loss and negative loss.
    """
    num = pos_vectors.shape[0] + neg_vectors.shape[0]
    pos_similarities = F.cosine_similarity(anchor_vector, pos_vectors, dim=1)
    neg_similarities = F.cosine_similarity(anchor_vector, neg_vectors, dim=1)

    loss_pos = F.mse_loss(pos_similarities, pos_overlaps, reduction='sum')
    loss_neg = F.mse_loss(neg_similarities, neg_overlaps, reduction='sum')
    loss = (loss_pos + alpha * loss_neg) / num
    return loss


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    if pos_vecs.shape[0] == 0:
        return -1

    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    if use_min:
        positive = min_pos
    else:
        positive = max_pos
    num_neg = neg_vecs.shape[0]
    num_pos= pos_vecs.shape[0]
    query_copies = q_vec.repeat(int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(int(num_neg), 1)

    negative = ((neg_vecs - query_copies) ** 2).sum(1).unsqueeze(1)

    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(1).unsqueeze(1)

    loss = loss.clamp(min=0.0)

    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(0)

    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()

    return triplet_loss

def triplet_loss_inv(q_vec, pos_vecs, neg_vecs, margin, use_min=True, lazy=False, ignore_zero_loss=False):

    min_neg, max_neg = best_pos_distance(q_vec, neg_vecs)

    if use_min:
        negative = min_neg
    else:
        negative = max_neg
    num_neg = neg_vecs.shape[0]
    num_pos= pos_vecs.shape[0]
    query_copies = q_vec.repeat(int(num_pos), 1)
    negative = negative.view(-1, 1)
    negative = negative.repeat(int(num_pos), 1)

    loss = margin - negative + ((pos_vecs - query_copies) ** 2).sum(1).unsqueeze(1)

    loss = loss.clamp(min=0.0)

    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(0)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()
    return triplet_loss


def triplet_loss_wrapper(q_vec, pos_vecs, neg_vecs, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    return triplet_loss(q_vec, pos_vecs, neg_vecs, m1, use_min, lazy, ignore_zero_loss)
