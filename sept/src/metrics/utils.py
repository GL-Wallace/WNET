import torch


def sort_predictions(predictions, probability, k=6):
    """Sort the predictions based on the probability of each mode.
    Args:
        predictions (torch.Tensor): The predicted trajectories [b, k, t, 2].
        probability (torch.Tensor): The probability of each mode [b, k].
    Returns:
        torch.Tensor: The sorted predictions [b, k', t, 2].
    """ # argsort() 返回的是元素排序后的索引
    indices = torch.argsort(probability, dim=-1, descending=True) # 使用 torch.argsort 对每个批次中的模式概率进行降序排序。indices 是排序后的索引。
    sorted_prob = probability[torch.arange(probability.size(0))[:, None], indices] # 利用排序索引对概率进行排序，得到排序后的概率（也是降序）
    sorted_predictions = predictions[  #使用同样的排序索引对预测轨迹进行排序
        torch.arange(predictions.size(0))[:, None], indices
    ]
    return sorted_predictions[:, :k], sorted_prob[:, :k]
