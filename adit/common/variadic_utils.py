import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.nn import functional as F

def area_under_prc(pred, target):
    """
    Area under precision-recall curve (PRC).

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    order = pred.argsort(descending=True)
    target = target[order]
    precision = target.cumsum(0) / torch.arange(1, len(target) + 1, device=target.device)
    auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
    return auprc

def f1_max(pred, target):
    """
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()

def spearmanr(pred, target):
    """
    Spearman correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """

    def get_ranking(input):
        input_set, input_inverse = input.unique(return_inverse=True)
        order = input_inverse.argsort()
        ranking = torch.zeros(len(input_inverse), device=input.device)
        ranking[order] = torch.arange(1, len(input) + 1, dtype=torch.float, device=input.device)

        # for elements that have the same value, replace their rankings with the mean of their rankings
        mean_ranking = scatter_mean(ranking, input_inverse, dim=0, dim_size=len(input_set))
        ranking = mean_ranking[input_inverse]
        return ranking

    pred = get_ranking(pred)
    target = get_ranking(target)
    covariance = (pred * target).mean() - pred.mean() * target.mean()
    pred_std = pred.std(unbiased=False)
    target_std = target.std(unbiased=False)
    spearmanr = covariance / (pred_std * target_std + 1e-10)
    return spearmanr

def pearsonr(pred, target):
    """
    Pearson correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """
    pred_mean = pred.float().mean()
    target_mean = target.float().mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    pred_normalized = pred_centered / pred_centered.norm(2)
    target_normalized = target_centered / target_centered.norm(2)
    pearsonr = pred_normalized @ target_normalized
    return pearsonr

def multi_slice(starts, ends):
    """
    Compute the union of indexes in multiple slices.

    Example::

        >>> mask = multi_slice(torch.tensor([0, 1, 4]), torch.tensor([2, 3, 6]), 6)
        >>> assert (mask == torch.tensor([0, 1, 2, 4, 5]).all()

    Parameters:
        starts (LongTensor): start indexes of slices
        ends (LongTensor): end indexes of slices
    """
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    slices, order = slices.sort()
    values = values[order]
    depth = values.cumsum(0)
    valid = ((values == 1) & (depth == 1)) | ((values == -1) & (depth == 0))
    slices = slices[valid]

    starts, ends = slices.view(-1, 2).t()
    size = ends - starts
    indexes = variadic_arange(size)
    indexes = indexes + starts.repeat_interleave(size)
    return indexes


def multi_slice_mask(starts, ends, length):
    """
    Compute the union of multiple slices into a binary mask.

    Example::

        >>> mask = multi_slice_mask(torch.tensor([0, 1, 4]), torch.tensor([2, 3, 6]), 6)
        >>> assert (mask == torch.tensor([1, 1, 1, 0, 1, 1])).all()

    Parameters:
        starts (LongTensor): start indexes of slices
        ends (LongTensor): end indexes of slices
        length (int): length of mask
    """
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    if slices.numel():
        assert slices.min() >= 0 and slices.max() <= length
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def as_mask(indexes, length):
    """
    Convert indexes into a binary mask.

    Parameters:
        indexes (LongTensor): positive indexes
        length (int): maximal possible value of indexes
    """
    mask = torch.zeros(length, dtype=torch.bool, device=indexes.device)
    mask[indexes] = 1
    return mask


def extend(data, size, input, input_size):
    """
    Extend variadic-sized data with variadic-sized input.
    This is a variadic variant of ``torch.cat([data, input], dim=-1)``.

    Example::

        >>> data = torch.tensor([0, 1, 2, 3, 4])
        >>> size = torch.tensor([3, 2])
        >>> input = torch.tensor([-1, -2, -3])
        >>> input_size = torch.tensor([1, 2])
        >>> new_data, new_size = extend(data, size, input, input_size)
        >>> assert (new_data == torch.tensor([0, 1, 2, -1, 3, 4, -2, -3])).all()
        >>> assert (new_size == torch.tensor([4, 4])).all()

    Parameters:
        data (Tensor): variadic data
        size (LongTensor): size of data
        input (Tensor): variadic input
        input_size (LongTensor): size of input

    Returns:
        (Tensor, LongTensor): output data, output size
    """
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def variadic_sum(input, size):
    """
    Compute sum over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
    """
    index2sample = torch.repeat_interleave(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    value = scatter_add(input, index2sample, dim=0)
    return value


def variadic_mean(input, size):
    """
    Compute mean over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
    """
    index2sample = torch.repeat_interleave(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    value = scatter_mean(input, index2sample, dim=0)
    return value


def variadic_max(input, size):
    """
    Compute max over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`

    Returns
        (Tensor, LongTensor): max values and indexes
    """
    index2sample = torch.repeat_interleave(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    value, index = scatter_max(input, index2sample, dim=0)
    index = index + (size - size.cumsum(0)).view([-1] + [1] * (index.ndim - 1))
    return value, index


def variadic_arange(size):
    """
    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``torch.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Parameters:
        size (LongTensor): size of intervals of shape :math:`(N,)`
    """
    starts = size.cumsum(0) - size

    range = torch.arange(size.sum(), device=size.device)
    range = range - starts.repeat_interleave(size)
    return range


def variadic_meshgrid(input1, size1, input2, size2):
    """
    Compute the Cartesian product for two batches of sets with variadic sizes.

    Suppose there are :math:`N` sets in each input,
    and the sizes of all sets are summed to :math:`B_1` and :math:`B_2` respectively.

    Parameters:
        input1 (Tensor): input of shape :math:`(B_1, ...)`
        size1 (LongTensor): size of :attr:`input1` of shape :math:`(N,)`
        input2 (Tensor): input of shape :math:`(B_2, ...)`
        size2 (LongTensor): size of :attr:`input2` of shape :math:`(N,)`

    Returns
        (Tensor, Tensor): the first and the second elements in the Cartesian product
    """
    grid_size = size1 * size2
    local_index = variadic_arange(grid_size)
    local_inner_size = size2.repeat_interleave(grid_size)
    offset1 = (size1.cumsum(0) - size1).repeat_interleave(grid_size)
    offset2 = (size2.cumsum(0) - size2).repeat_interleave(grid_size)
    index1 = torch.div(local_index, local_inner_size, rounding_mode="floor") + offset1
    index2 = local_index % local_inner_size + offset2
    return input1[index1], input2[index2]


def variadic_to_padded(input, size, value=0):
    """
    Convert a variadic tensor to a padded tensor.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
        value (scalar): fill value for padding

    Returns:
        (Tensor, BoolTensor): padded tensor and mask
    """
    num_sample = len(size)
    max_size = size.max()
    starts = torch.arange(num_sample, device=size.device) * max_size
    ends = starts + size
    mask = multi_slice_mask(starts, ends, num_sample * max_size)
    mask = mask.view(num_sample, max_size)
    shape = (num_sample, max_size) + input.shape[1:]
    padded = torch.full(shape, value, dtype=input.dtype, device=size.device)
    padded[mask] = input
    return padded, mask


def padded_to_variadic(padded, size):
    """
    Convert a padded tensor to a variadic tensor.

    Parameters:
        padded (Tensor): padded tensor of shape :math:`(N, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
    """
    num_sample, max_size = padded.shape[:2]
    starts = torch.arange(num_sample, device=size.device) * max_size
    ends = starts + size
    mask = multi_slice_mask(starts, ends, num_sample * max_size)
    mask = mask.view(num_sample, max_size)
    return padded[mask]
