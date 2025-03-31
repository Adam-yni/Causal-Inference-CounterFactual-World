# Copyright (c) 2020, Ioana Bica

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def equivariant_layer(x, h_dim, layer_id, treatment_id):
    # Sum along axis 1
    xm = torch.sum(x, dim=1, keepdim=True)

    # Dense layers
    l_gamma = nn.Linear(x.size(-1), h_dim, bias=True)
    l_lambda = nn.Linear(x.size(-1), h_dim, bias=False)

    # Naming layers (optional, for debugging or saving models)
    l_gamma.name = f'eqv_{layer_id}_treatment_{treatment_id}_gamma'
    l_lambda.name = f'eqv_{layer_id}_treatment_{treatment_id}_lambda'

    # Apply layers
    gamma_out = l_gamma(x)
    lambda_out = l_lambda(xm)

    # Compute output
    out = gamma_out - lambda_out
    return out


def invariant_layer(x, h_dim, treatment_id):
    # Dense layer with ELU activation
    rep_layer_1 = nn.Linear(x.size(-1), h_dim)
    rep_layer_1.name = f'inv_treatment_{treatment_id}'

    # Apply layer and activation
    rep_out = F.elu(rep_layer_1(x))

    # Sum along axis 1
    rep_sum = torch.sum(rep_out, dim=1)

    return rep_sum


def sample_Z(m, n):
    # Uniform random sampling
    return torch.rand(m, n)


def sample_X(X, size):
    # Random sampling of indices
    start_idx = np.random.randint(0, X.shape[0], size)
    return start_idx


def sample_dosages(batch_size, num_treatments, num_dosages):
    # Uniform random sampling for dosages
    dosage_samples = torch.rand(batch_size, num_treatments, num_dosages)
    return dosage_samples