import torch
import torch.nn as nn
import torch.nn.functional as F


def best_pos_distance(query, pos_vecs, axis=1):
    """
    Find the closest positive vector between query and pos_vecs set.

    Args:
        query: (torch.Tensor) the query vector in shape (1, q_size).
        pos_vecs: (torch.Tensor) the query positive vector set in shape (num_pos, q_size).
        axis: (int) the sum axis (default is 1 for vector 1 * n, switch to 0 if vector has dimension n * 1).
    """
    diff = ((pos_vecs - query) ** 2).sum(axis)
    min_pos, _ = diff.min()
    max_pos, _ = diff.max()
    return min_pos, max_pos


def mean_squared_error_loss(vec1, vec2, overlaps):
    """
    First calculate the cosine similarity between two batch of vectors.
    Then calculate the mean squared error between the similarity and overlaps.

    Args:
        vec1: (torch.Tensor) scan1 vectors in shape (num, vec_size).
        vec2: (torch.Tensor) scan2 vectors in shape (num, vec_size).
        overlaps: (float) the overlaps between two vectors.
    """
    similarity = F.cosine_similarity(vec1, vec2)
    loss = F.mse_loss(similarity, overlaps)
    return loss


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    """
    Calculate the (lazy) triplet loss for a query vector and a positive vector set and a negative vector set.

    Args:
        q_vec: (torch.Tensor) the query vector in shape (1, q_size).
        pos_vecs: (torch.Tensor) the query positive vector set in shape (num_pos, q_size).
        neg_vecs: (torch.Tensor) the query negative vector set in shape (num_neg, q_size).
        margin: (float) the margin parameters.
        use_min: (bool) decide to use the positive pair with max distance or min distance.
        lazy: (bool) use lazy triplet loss ot not.
        ignore_zero_loss: if count the 0 triplet loss pair when calculate the mean.
    """
    if pos_vecs.shape[1] == 0:
        return -1

    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # the PointNetVLAD use min_pos, the implementation in cattaneod use max_pos instead, (I prefer max_pos)
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    positive = positive.view(-1, 1)
    negative = ((neg_vecs - q_vec) ** 2).sum(1)

    # negative if correctly distinguish, only count fail pairs
    loss = margin + positive - negative
    loss = loss.clamp(min=0.0)

    # lazy triplet use max to find the closest (smallest) negative pair
    if lazy:
        loss = loss.max(1)[0]
    else:
        loss = loss.sum(0)

    if ignore_zero_loss:
        hard_triplet = torch.gt(loss, 1e-16).float()    # 1 is element > 0, otherwise 0
        num_hard_triplet = torch.sum(hard_triplet)
        loss = loss.sum() / num_hard_triplet
    else:
        loss = loss.mean()

    return loss


def quadruplet_loss(q_vec, pos_vecs, neg_vecs, neg_vec_rand, margin1, margin2,
                    use_min=False, lazy=False, ignore_zero_loss=False):
    """
    Calculate the (lazy) quadruplet loss for a query vector and a positive vector set and a negative vector set.

    Args:
        q_vec: (torch.Tensor) the query vector in shape (1, q_size).
        pos_vecs: (torch.Tensor) the query positive vector set in shape (num_pos, q_size).
        neg_vecs: (torch.Tensor) the query negative vector set in shape (num_neg, q_size).
        neg_vec_rand: (torch.Tensor) a random negative vector from neg_vecs set to prevent the gap between neg_max and
                                    the other samples.
        margin1: (float) the margin parameters for query triplet loss.
        margin2: (float) the margin parameters for random negative pair triplet loss.
        use_min: (bool) decide to use the positive pair with max distance or min distance.
        lazy: (bool) use lazy triplet loss ot not.
        ignore_zero_loss: if count the 0 triplet loss pair when calculate the mean.
    """
    # in case no positive pair
    if pos_vecs.shape[1] == 0:
        return -1

    # calculate the triple loss for query vector
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

    # calculate the random negative vector loss
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
    q = torch.tensor([1.1, 1.2, 1.3]).unsqueeze(0)
    pos_vectors = torch.tensor([1, 2, 3]).repeat(3, 1).transpose(0, 1)
    neg_vectors = torch.tensor([0.95, 1.7, 3.8]).repeat(3, 1).transpose(0, 1)
    neg_vector_rand = neg_vectors[torch.randperm(neg_vectors.size(0))][0]

    print(neg_vector_rand)

    tri_loss = triplet_loss(q, pos_vectors, neg_vectors, margin=0.5)
    quad_loss = quadruplet_loss(q, pos_vectors, neg_vectors, neg_vector_rand, 0.5, 0.5)

