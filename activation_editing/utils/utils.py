"""
Uncategorized utility functions that are used by other modules.
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Union, Dict, Optional


# CONSTANTS
EPSILON = 1e-7


# Functions
def add_special_tokens(model: Union[nn.Module, PreTrainedModel],
                       tokenizer: PreTrainedTokenizerBase,
                       special_tokens_dict: Dict[str, str]):
    """
    Add special tokens while also handling padding tokens for decoder-only models
    :param model: A model.
    :param tokenizer: The model's corresponding tokenizer.
    :param special_tokens_dict: A dictionary in the form {special_tokens_name: special_token_text}
    :return: None
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if 'pad_token' in special_tokens_dict.keys():
        model.config.pad_token_id = tokenizer.pad_token_id
        input_embeddings = model.get_input_embeddings()
        input_embeddings.padding_idx = tokenizer.pad_token_id
        input_embeddings._fill_padding_idx_with_zero()
        model.tie_weights()
    return


def projection_on_vector_support(u: torch.Tensor,
                                 v: torch.Tensor,
                                 dim: Optional[int] = None) -> torch.Tensor:
    """
    Project  u on to the support of v. Assuming that both u and v go through the origin.
    :param u: ('torch.Tensor') The vector(s) to project.
    :param v: ('torch.Tensor') The vector(s) with the target support.
    :param dim: (int) The dimension along which calculations are done.
    :return: ('torch.Tensor') The projection of u on the support of v.
    """
    return torch.sum(u*v, dim=dim, keepdim=True) / (v.norm(dim=dim, keepdim=True)**2) * v


def projection_on_plane(u: torch.Tensor,
                        v: torch.Tensor,
                        dim: Optional[int] = None) -> torch.Tensor:
    """
    Project u onto the plane orthogonal to v. Assuming that both u and v go through the origin.
    :param u: ('torch.Tensor') The vector(s) to project.
    :param v: ('torch.Tensor') The vector(s) orthogonal to the target plane.
    :param dim: (int) The dimension along which calculations are done.
    :return: ('torch.Tensor') The projection of u on the plane orthogonal to v.
    """
    return u - projection_on_vector_support(u, v, dim=dim)


def batched_identity_mat(*size: int, dtype=None, device=None):
    """
    Create a batch of identity matrices with the dimension of each matrix being the last size.
    :param size: The size of each dimension. For example (m, n, p) --> Return tensor of shape (m, n, p, p).
    :param dtype: The type of the returned tensor.
    :param device: The device to send the returned tensor to.
    :return: Batched identity matrices of shape (*size[:-1], size[-1], size[-1])
    """
    identity = torch.eye(size[-1], dtype=dtype, device=device)
    identity = identity.reshape((*([1] * (len(size)-1)), *identity.shape))
    identity = identity.repeat(*size[:-1], 1, 1)
    return identity


def get_householder_matrix(v: torch.Tensor):
    """
    Create a matrix that reflects vectors about a hyperplane whose normal vector is v.
    :param v: A unit vector (i.e. norm(v) == 1) of shape (1, n), which is orthogonal to the target hyperplane.
    :return: A Householder reflection matrix of shape (n, n).
    """
    identity = torch.eye(v.shape[-1], dtype=v.dtype, device=v.device)
    return identity - 2 * torch.matmul(v.T, v) / (torch.norm(v, dim=-1)**2)


def batched_dot_product(a: torch.Tensor, b: torch.Tensor, dim=-1, keepdim=True) -> torch.Tensor:
    """
    Perform batched dot product. Somehow PyTorch doesn't have an optimized API for this usecase
    :param a: a tensor.
    :param b: another tensor.
    :param dim: The dimension along which dot product will be computed.
    :param keepdim: Whether or not to keep the number of dimension after calculation.
    :return: a tensor of dot product between a and b along the given dimension.
    """
    return torch.sum(a * b, dim=dim, keepdim=keepdim)


def calculate_angle(a: torch.Tensor, b: torch.Tensor):
    """
    Calculate the angle (in radians) between two set of vectors, pair-wise, along the last dimension.
    :param a: a tensor.
    :param b: another tensor.
    :return: The angle (in radians) between a and b.
    """
    cos = batched_dot_product(a, b, dim=-1, keepdim=True) / (a.norm(dim=-1, keepdim=True)*b.norm(dim=-1, keepdim=True))
    cos = torch.clamp(cos, -1 + EPSILON, 1 - EPSILON)   # For numerical stability
    return torch.arccos(cos)
