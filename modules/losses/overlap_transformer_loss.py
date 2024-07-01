import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_distance(v1, v2, axis=1, metric='euclidean'):
    """
    Compute the Euclidean or Cosine distance between 2 batch of vectors.

    Args:
        v1: (torch.Tensor) the first batch of vectors in shape (1, q_size).
        v2: (torch.Tensor) the second batch of vectors in shape (num, q_size).
        axis: (int) the sum axis (default is 1 for vector 1 * n, switch to 0 if vector has dimension n * 1).
        metric: (string) the metric. euclidean: Euclidean distances between 2 vectors. cosine: cosine similarity between
        2 vectors, note the distance is computed between 0 and 1.
    Output:
        (torch.Tensor) the distances between vectors in shape (num, 1).
    """
    if metric == 'euclidean':
        return ((v1 - v2) ** 2).sum(axis, keepdim=True)
    elif metric == 'cosine':
        similarity = F.cosine_similarity(v1, v2, dim=axis).unsqueeze(1)
        return 1.0 - similarity                                         # 0 - 2, 0: same vectors, 2: opposite vectors
    else:
        raise ValueError('Invalid metric! Choose either "euclidean" or "cosine".')


def best_pos_distance(query, pos_vecs, axis=1, metric='euclidean'):
    """
    Find the closest positive vector between query and pos_vecs set.

    Args:
        query: (torch.Tensor) the query vector in shape (1, q_size).
        pos_vecs: (torch.Tensor) the query positive vector set in shape (num_pos, q_size).
        axis: (int) the sum axis (default is 1 for vector 1 * n, switch to 0 if vector has dimension n * 1).
        metric: (string) the metric. euclidean: Euclidean distances between 2 vectors. cosine: cosine similarity between
        2 vectors, note the distance is computed between 0 and 1.
    Outputs:
        (torch.Tensor) the closest distance between query and pos_vecs set in shape (1,).
        (torch.Tensor) the furthest distance between query and pos_vecs set in shape (1,).
    """
    diff = compute_distance(query, pos_vecs, axis, metric)
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
        alpha: (float) the parameter balance the losses between positive losses and negative losses.
    Output:
        (torch.Tensor) the mean squared error between the similarity and overlaps.
    """
    num = pos_vectors.shape[0] + neg_vectors.shape[0]
    pos_similarities = (F.cosine_similarity(anchor_vector, pos_vectors, dim=1) + 1.0) / 2.0
    neg_similarities = (F.cosine_similarity(anchor_vector, neg_vectors, dim=1) + 1.0) / 2.0

    loss_pos = F.mse_loss(pos_similarities, pos_overlaps, reduction='sum')
    loss_neg = F.mse_loss(neg_similarities, neg_overlaps, reduction='sum')
    loss = (loss_pos + alpha * loss_neg) / num
    return loss


def overlap_loss(q_vec, pos_vecs, overlaps, metric='euclidean'):
    """
    First calculate the similarity between two batch of vectors.
    Then calculate the mean squared error between the similarity and overlaps.
    !!! Note, the loss make the margin very tight. Hard to train. !!!

    Args:
        q_vec: (torch.Tensor) scan1 vectors in shape (1, vec_size).
        pos_vecs: (torch.Tensor) scan2 vectors in shape (num, vec_size).
        overlaps: (float) the overlaps between two vectors.
        metric: (string) the metric. euclidean: Euclidean distance, cosine: Cosine similarity
    Outputs:
        (torch.Tensor) the (mean) squared error between the similarity and overlaps.
    """
    similarity = compute_distance(q_vec, pos_vecs, axis=1, metric=metric)

    # if metric == 'euclidean':
    #     similarity = F.normalize(similarity, dim=0) * torch.pi / 2   # normalized between 0 - pi/2
    #     similarity = torch.cos(similarity)                           # between 1 - 0
    #
    # if metric == 'cosine':
    #     # similarity = 1.0 - similarity / 2.0                          # between 1 - 0
    #     similarity = torch.clamp(1.0 - similarity, min=-1.0, max=1.0)
    #     similarity = 1.0 - torch.arccos(similarity) / torch.pi               # between 1 - 0
    #
    # if len(overlaps.shape) == 1:
    #     overlaps = overlaps.unsqueeze(1)
    #
    # # loss = F.mse_loss(diff, overlaps)
    # loss = ((similarity - overlaps) ** 2).sum()

    if metric == 'cosine':
        similarity = torch.clamp(similarity, min=0.0, max=2.0)              # avoid negative numbers

    if len(overlaps.shape) == 1:
        overlaps = overlaps.unsqueeze(1)

    loss = (similarity * (overlaps ** 1)).sum()                             # higher overlap, higher weight
    return loss


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=True,
                 metric='euclidean'):
    """
    Calculate the (lazy) triplet losses for a query vector and a positive vector set and a negative vector set.

    Args:
        q_vec: (torch.Tensor) the query vector in shape (1, q_size).
        pos_vecs: (torch.Tensor) the query positive vector set in shape (num_pos, q_size).
        neg_vecs: (torch.Tensor) the query negative vector set in shape (num_neg, q_size).
        margin: (float) the margin parameters.
        use_min: (bool) decide to use the positive pair with max distance or min distance.
        lazy: (bool) use lazy triplet losses ot not.
        ignore_zero_loss: if count the 0 triplet losses pair when calculate the mean.
        metric: (string) the metric. euclidean: Euclidean distance, cosine: Cosine similarity between vectors.
    Output:
        (torch.Tensor) the triplet loss.
    """
    if pos_vecs.shape[1] == 0:
        return -1

    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs, axis=1, metric=metric)

    # the PointNetVLAD use min_pos, the implementation in cattaneod use max_pos instead, (I prefer max_pos)
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    negative = compute_distance(q_vec, neg_vecs, axis=1, metric=metric)

    # negative if correctly distinguish, only count fail pairs
    loss = margin + positive - negative
    loss = loss.clamp(min=0.0)

    # lazy triplet use max to find the closest (smallest) negative pair
    if lazy:
        loss = loss.max(0)[0]
    else:
        loss = loss.sum(0)

    if ignore_zero_loss:                                      # this part is useless now as loss is 1-d tensor
        hard_triplet = torch.gt(loss, 1e-16).float()    # 1 is element > 0, otherwise 0
        num_hard_triplet = torch.sum(hard_triplet)
        loss = loss.sum() / (num_hard_triplet + 1e-16)
    else:
        loss = loss.mean()

    return loss


def triplet_confidence_loss(q_vec, pos_vecs, neg_vecs, pos_overlaps, margin, alpha=1.0, use_min=False, lazy=False,
                            ignore_zero_loss=True, metric='euclidean'):
    """
    Calculate the (lazy) triplet losses combine with a confidence loss function to minimize the overlaps and similarity
    between query and positive vectors.

    Args:
        q_vec: (torch.Tensor) the query vector in shape (1, q_size).
        pos_vecs: (torch.Tensor) the query positive vector set in shape (num_pos, q_size).
        neg_vecs: (torch.Tensor) the query negative vector set in shape (num_neg, q_size).
        pos_overlaps: (torch.Tensor) the positive pairs overlap values in shape (num_pos,).
        margin: (float) the margin parameters.
        alpha: (float) the parameter balance the losses between triplet loss and overlaps loss.
        use_min: (bool) decide to use the positive pair with max distance or min distance.
        lazy: (bool) use lazy triplet losses ot not.
        ignore_zero_loss: if count the 0 triplet losses pair when calculate the mean.
        metric: (string) the metric. euclidean: Euclidean distance, cosine: Cosine similarity between vectors.
    Output:
        (torch.Tensor) the triplet loss.
    """
    assert metric == 'euclidean' or metric == 'cosine', "metric must be 'euclidean' or 'cosine'."

    tri_loss = triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min, lazy, ignore_zero_loss, metric=metric)
    sim_loss = overlap_loss(q_vec, pos_vecs, pos_overlaps, metric=metric)
    loss = tri_loss + alpha * sim_loss
    return loss


def quadruplet_loss(q_vec, pos_vecs, neg_vecs, neg_vec_rand, margin1, margin2,
                    use_min=False, lazy=False, ignore_zero_loss=False):
    """
    Calculate the (lazy) quadruplet losses for a query vector and a positive vector set and a negative vector set.

    Args:
        q_vec: (torch.Tensor) the query vector in shape (1, q_size).
        pos_vecs: (torch.Tensor) the query positive vector set in shape (num_pos, q_size).
        neg_vecs: (torch.Tensor) the query negative vector set in shape (num_neg, q_size).
        neg_vec_rand: (torch.Tensor) a random negative vector from neg_vecs set to prevent the gap between neg_max and
                                    the other samples.
        margin1: (float) the margin parameters for query triplet losses.
        margin2: (float) the margin parameters for random negative pair triplet losses.
        use_min: (bool) decide to use the positive pair with max distance or min distance.
        lazy: (bool) use lazy triplet losses ot not.
        ignore_zero_loss: if count the 0 triplet losses pair when calculate the mean.
    Output:
        (torch.Tensor) the quadruplet loss.
    """
    # in case no positive pair
    if pos_vecs.shape[1] == 0:
        return -1

    # calculate the triple losses for query vector
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # the PointNetVLAD use min_pos, the implementation in cattaneod use max_pos instead, (I prefer max_pos)
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    positive = positive.view(-1, 1)
    negative = ((neg_vecs - q_vec) ** 2).sum(1)

    # negative if correctly distinguish, only count fail pairs
    loss_pos = margin1 + positive - negative
    loss_pos = loss_pos.clamp(min=0.0)

    # lazy triplet use max to find the closest (smallest) negative pair
    if lazy:
        loss_pos = loss_pos.max(1)[0]
    else:
        loss_pos = loss_pos.sum(0)

    if ignore_zero_loss:
        hard_triplet_pos = torch.gt(loss_pos, 1e-16).float()  # 1 is element > 0, otherwise 0
        num_hard_triplet_pos = torch.sum(hard_triplet_pos)
        loss_pos = loss_pos.sum() / num_hard_triplet_pos
    else:
        loss_pos = loss_pos.mean()

    # calculate the random negative vector losses
    negative_rand = ((neg_vecs - neg_vec_rand) ** 2).sum(1)
    loss_other = margin2 + positive - negative_rand
    loss_other = loss_other.clamp(0.0)

    # lazy triplet use max to find the closest (smallest) negative pair
    if lazy:
        loss_other = loss_other.max(1)[0]
    else:
        loss_other = loss_other.sum(0)

    if ignore_zero_loss:
        hard_triplet_other = torch.gt(loss_other, 1e-16).float()    # 1 is element > 0, otherwise 0
        num_hard_triplet_other = torch.sum(hard_triplet_other)
        loss_other = loss_other.sum() / num_hard_triplet_other
    else:
        loss_other = loss_other.mean()

    loss = loss_pos + loss_other
    return loss


if __name__ == '__main__':
    q = torch.tensor([1.0, 0.0]).unsqueeze(0)
    pos_vectors = torch.tensor([1.0, 0.0]).repeat(3, 1)
    neg_vectors = torch.tensor([0.5, 0.87]).repeat(3, 1)
    neg_vector_rand = neg_vectors[torch.randperm(neg_vectors.size(0))][0]
    pos_overlaps = torch.tensor([0.9]).repeat(3)
    neg_overlaps = torch.tensor([0.1]).repeat(3)
    #
    # tri_loss = triplet_loss(q, pos_vectors, neg_vectors, margin=0.5)
    # quad_loss = quadruplet_loss(q, pos_vectors, neg_vectors, neg_vector_rand, 0.5, 0.5)

    # print(tri_loss)
    # print(compute_distance(q, neg_vectors, axis=1, metric='euclidean'))

    print(triplet_loss(q, pos_vectors, neg_vectors, margin=0.5, ignore_zero_loss=True, metric='cosine'))
    # margin = 0.5 in euclidean space approximately equal to 0.72 in rad and 41 deg, same as margin 0.25 in cosine
    # margin = 2.0 im euclidean space approximately equal to pi/2 in rad amd 90 deg, same as margin 1.0 in cosine




