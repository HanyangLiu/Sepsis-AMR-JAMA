# Code adapted from https://github.com/WenjieDu/PyPOTS/blob/main/pypots/nn/modules/grud/layers.py

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from typing import Tuple, Union, Optional


class TemporalDecay(nn.Module):
    """The module used to generate the temporal decay factor gamma in the GRU-D model.
    Please refer to the original paper :cite:`che2018GRUD` for more details.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input

    output_size : int,
        the feature dimension of the output

    diag : bool,
        whether to product the weight with an identity matrix before forward processing

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Forward processing of this NN module.

        Parameters
        ----------
        delta : tensor, shape [n_samples, n_steps, n_features]
            The time gaps.

        Returns
        -------
        gamma : tensor, of the same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class BackboneGRUD(nn.Module):
    def __init__(
            self,
            n_steps: int,
            n_features: int,
            rnn_hidden_size: int,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        # create models
        self.rnn_cell = nn.GRUCell(
            self.n_features * 2 + self.rnn_hidden_size, self.rnn_hidden_size
        )
        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True
        )

    def forward(
            self, X, missing_mask, deltas, empirical_mean, X_filledLOCF
    ) -> Tuple[torch.Tensor, ...]:
        """Forward processing of GRU-D.

        Parameters
        ----------
        X:

        missing_mask:

        deltas:

        empirical_mean:

        X_filledLOCF:

        Returns
        -------
        classification_pred:

        logits:

        """

        hidden_state = torch.zeros((X.size()[0], self.rnn_hidden_size), device=X.device)

        representation_collector = []
        for t in range(self.n_steps):
            # for data, [batch, time, features]
            x = X[:, t, :]  # values
            m = missing_mask[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap
            x_filledLOCF = X_filledLOCF[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            hidden_state = hidden_state * gamma_h
            representation_collector.append(hidden_state)

            x_h = gamma_x * x_filledLOCF + (1 - gamma_x) * empirical_mean
            x_replaced = m * x + (1 - m) * x_h
            data_input = torch.cat([x_replaced, hidden_state, m], dim=1)
            hidden_state = self.rnn_cell(data_input, hidden_state)

        representation_collector = torch.stack(representation_collector, dim=1)

        return representation_collector, hidden_state


class _GRUD(nn.Module):
    """
    Imputation module. Not used for classification.
    """
    def __init__(
            self,
            n_steps: int,
            n_features: int,
            rnn_hidden_size: int,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        # create models
        self.backbone = BackboneGRUD(
            n_steps,
            n_features,
            rnn_hidden_size,
        )
        self.output_projection = nn.Linear(rnn_hidden_size, n_features)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        """Forward processing of GRU-D.

        Parameters
        ----------
        inputs :
            The input data.

        training :
            Whether in training mode.

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        X = inputs["X"]
        missing_mask = inputs["missing_mask"]
        deltas = inputs["deltas"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_filledLOCF"]

        hidden_states, _ = self.backbone(
            X, missing_mask, deltas, empirical_mean, X_filledLOCF
        )

        # project back the original data space
        reconstruction = self.output_projection(hidden_states)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            results["loss"] = calc_mse(reconstruction, X, missing_mask)

        return results


def calc_mse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------
    mse = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is 1/2=0.5.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.square(predictions - targets) * masks) / (
            lib.sum(masks) + 1e-12
        )
    else:
        return lib.mean(lib.square(predictions - targets))


def _check_inputs(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
    check_shape: bool = True,
):
    # check type
    assert isinstance(predictions, type(targets)), (
        f"types of `predictions` and `targets` must match, but got"
        f"`predictions`: {type(predictions)}, `target`: {type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    # check shape
    prediction_shape = predictions.shape
    target_shape = targets.shape
    if check_shape:
        assert (
            prediction_shape == target_shape
        ), f"shape of `predictions` and `targets` must match, but got {prediction_shape} and {target_shape}"
    # check NaN
    assert not lib.isnan(
        predictions
    ).any(), "`predictions` mustn't contain NaN values, but detected NaN in it"
    assert not lib.isnan(
        targets
    ).any(), "`targets` mustn't contain NaN values, but detected NaN in it"

    if masks is not None:
        # check type
        assert isinstance(masks, type(targets)), (
            f"types of `masks`, `predictions`, and `targets` must match, but got"
            f"`masks`: {type(masks)}, `targets`: {type(targets)}"
        )
        # check shape, masks shape must match targets
        mask_shape = masks.shape
        assert mask_shape == target_shape, (
            f"shape of `masks` must match `targets` shape, "
            f"but got `mask`: {mask_shape} that is different from `targets`: {target_shape}"
        )
        # check NaN
        assert not lib.isnan(
            masks
        ).any(), "`masks` mustn't contain NaN values, but detected NaN in it"

    return lib
