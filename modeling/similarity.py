'''
GPU based metrics:
 - Cosine similarity
 - Jackart simiarity
 - Levenstein similarity
 - L1 and L2 distance
 -
'''


import torch
from torch import nn
import numpy as np
import time

# TODO: add CPU based implementations of every metric (numpy based)

def cosine_gpu(tensor_1, tensor_2):
    """
    Expects input tensors to contain vector representations of entities in rows (not in columns!).
    Expects two dimensional tensors (even in case of vector it should have the second dimension).
    :param tensor_1:
    :param tensor_2:
    :return:
    """
    # dot prod of tensors
    dot = torch.matmul(tensor_1, tensor_2.t())
    # calc vector norm of entities
    norm_tensor_1 = torch.norm(tensor_1, p=2, dim=1)
    norm_tensor_2 = torch.norm(tensor_2, p=2, dim=1)
    # divide dot prod by respective vector norms
    cos = torch.div(dot, torch.unsqueeze(norm_tensor_1, dim=1))
    cos = torch.div(cos, norm_tensor_2)
    return cos

def dist_sort(dist_matrix, top_n):
    """
    Sorts distance matrix by similarity value in descending order (left to right) and provide two outputs:
     - similarity values of top n most similar items to entity
     - index values of top n most similar items to entity
    Puts resulting matrices on cpu and converts to numpy
    """
    values_sorted = torch.sort(dist_matrix, descending=True)[0][:, :top_n].cpu().detach().numpy()
    index_sorted = torch.sort(dist_matrix, descending=True)[1][:, :top_n].cpu().detach().numpy()
    return values_sorted, index_sorted




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.rand([10, 2]).to(device)
b = torch.rand([100, 2]).to(device)
res = cosine_gpu(a, b)


x = torch.Tensor([6, 5, 4])
y = torch.Tensor([3, 2, 1])
x.reshape(-1, 1) - y

x = np.array([[1, 2, 3], [1, 2, 3]])
y = np.array([[1, 2, 3], [1, 2, 3]])
np.subtract.outer(x, y).shape