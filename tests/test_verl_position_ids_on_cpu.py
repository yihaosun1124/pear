import pickle

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


def position_rows(lengths):
    return [torch.arange(4 * length).reshape(4, length) + row * 100000 for row, length in enumerate(lengths)]


@pytest.mark.parametrize("lengths", [(5140,) * 8, (8, 8)])
def test_uniform_mrope_positions_survive_batch_selection(lengths):
    rows = position_rows(lengths)
    # Equal-length trajectories can arrive packed along the four mRoPE channels.
    positions = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    batch = TensorDict({"position_ids": positions}, batch_size=[len(rows)])

    tu.maybe_fix_3d_position_ids(batch)
    selected = tu.index_select_tensor_dict(batch, [len(rows) - 1, 0, len(rows) - 1])

    assert selected["position_ids"].shape[1] == 4
    for actual, index in zip(selected["position_ids"].unbind(), [len(rows) - 1, 0, len(rows) - 1]):
        torch.testing.assert_close(actual, rows[index], rtol=0, atol=0)


def test_mrope_positions_survive_tensordict_consolidation_and_pickle():
    rows = position_rows([6, 9, 5, 10])
    positions = tu.nested_tensor_from_tensor_list(rows, ragged_idx=2)
    batch = TensorDict({"position_ids": positions}, batch_size=[len(rows)]).consolidate()
    batch = pickle.loads(pickle.dumps(batch))

    tu.maybe_fix_3d_position_ids(batch)

    assert batch["position_ids"].shape[1] == 4
    for actual, expected in zip(batch["position_ids"].unbind(), rows):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("ragged_idx", [1, 2])
def test_nested_tensor_constructor_keeps_the_declared_ragged_dimension(ragged_idx):
    rows = [torch.arange(3 * length).reshape((length, 3) if ragged_idx == 1 else (3, length)) for length in [2, 5]]
    positions = tu.nested_tensor_from_tensor_list(rows, ragged_idx=ragged_idx)

    assert positions.shape[2 if ragged_idx == 1 else 1] == 3
    for actual, expected in zip(positions.unbind(), rows):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_text_position_ids_need_no_mrope_layout_repair():
    positions = torch.nested.as_nested_tensor([torch.arange(3), torch.arange(5)], layout=torch.jagged)
    batch = TensorDict({"position_ids": positions}, batch_size=[2])

    tu.maybe_fix_3d_position_ids(batch)

    assert batch["position_ids"] is positions
