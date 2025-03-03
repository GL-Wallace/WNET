# Copyright (c) Carizon. All rights reserved.

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def qcnet_viz_collate(
    batch: List[Dict[str, torch.tensor]],
) -> Dict[str, Any]:
    """QCNet collate function is used for Carizon driving data.

    Args:
        batch: Bactched data to be collated with.
    """
    sample: Dict[str, Any] = dict()

    sample["agent_features"] = {}
    sample["agent_features"]["gcs"] = {}
    sample["agent_features"]["acs"] = {}
    sample["agent_features"]["gcs"]["poses"] = {}
    sample["agent_features"]["acs"]["poses"] = {}
    sample["agent_features"]["gcs"]["vels"] = {}
    sample["agent_features"]["agent_properties"] = {}
    sample["agent_features"]["state_valid_masks"] = {}

    num_agent_classes = 16
    num_map_classes = 4
    num_arrow_classes = 11

    # 1.1.1 pad historical trajectories
    sample["agent_features"]["gcs"]["poses"]["his"] = pad_sequence(
        [
            b["agent_features"]["gcs"]["his_pose"][:, :, [0, 1, 4]]
            for b in batch
        ],
        batch_first=True,
    ).float()

    # 1.1.2 pad future trajectories (gt in gcs)
    sample["agent_features"]["gcs"]["poses"]["fut"] = pad_sequence(
        [
            b["agent_features"]["gcs"]["fut_pose"][:, :, [0, 1, 4]]
            for b in batch
        ],
        batch_first=True,
    ).float()

    # 1.1.3 pad future trajectories (gt in acs)
    sample["agent_features"]["acs"]["poses"]["fut"] = pad_sequence(
        [
            b["agent_features"]["acs"]["fut_pose"][:, :, [0, 1, 4]]
            for b in batch
        ],
        batch_first=True,
    ).float()

    # 1.2 pad historical velocities
    sample["agent_features"]["gcs"]["vels"]["his"] = pad_sequence(
        [b["agent_features"]["gcs"]["his_pose"][:, :, 2:4] for b in batch],
        batch_first=True,
    ).float()

    # 1.3.1 pad historical valid masks
    sample["agent_features"]["state_valid_masks"]["his"] = pad_sequence(
        [b["agent_features"]["his_valid_masks"] for b in batch],
        batch_first=True,
        padding_value=False,
    ).bool()

    # 1.3.2 pad future valid masks
    sample["agent_features"]["state_valid_masks"]["fut"] = pad_sequence(
        [b["agent_features"]["fut_valid_masks"] for b in batch],
        batch_first=True,
        padding_value=False,
    ).bool()

    # 1.4 pad current valid masks
    sample["agent_features"]["agent_valid_masks"] = pad_sequence(
        [b["agent_features"]["his_valid_masks"][:, -1] for b in batch],
        batch_first=True,
        padding_value=False,
    ).bool()

    agent_classes = pad_sequence(
        [b["agent_features"]["types"] for b in batch],
        batch_first=True,
    ).long()
    sample["agent_features"]["agent_properties"]["classes"] = F.one_hot(
        agent_classes, num_classes=num_agent_classes
    ).float()

    # 1.5.2 pad agent ids
    sample["agent_features"]["track_ids"] = pad_sequence(
        [b["agent_features"]["track_ids"] for b in batch],
        batch_first=True,
    ).long()

    # 1.6 pad agent categories
    sample["agent_features"]["categories"] = pad_sequence(
        [b["agent_features"]["category"] for b in batch],
        batch_first=True,
    ).long()

    # map polygon features padding
    sample["map_features"] = {}
    sample["map_features"]["gcs"] = {}

    # 2.1 polygon poses
    sample["map_features"]["gcs"]["pl_poses"] = pad_sequence(
        [b["map_features"]["map_polygon"]["position"] for b in batch],
        batch_first=True,
    ).float()

    # 2.2 polygon types
    pl_types = pad_sequence(
        [b["map_features"]["map_polygon"]["type"] for b in batch],
        batch_first=True,
    ).long()
    sample["map_features"]["pl_types"] = F.one_hot(
        pl_types, num_classes=num_map_classes
    ).float()

    # 2.2 polygon arrow types
    arrow_types = pad_sequence(
        [b["map_features"]["map_polygon"]["arrow_type"] for b in batch],
        batch_first=True,
    ).long()
    sample["map_features"]["arrow_types"] = F.one_hot(
        arrow_types, num_classes=num_arrow_classes
    ).float()

    # 2.3 polygon valid masks
    sample["map_features"]["pl_valid_masks"] = pad_sequence(
        [b["map_features"]["map_polygon"]["valid_mask"] for b in batch],
        batch_first=True,
    ).bool()

    # 3.1 point poses
    sample["map_features"]["gcs"]["pt_poses"] = pad_sequence(
        [b["map_features"]["map_point"]["position"] for b in batch],
        batch_first=True,
    ).float()

    # 3.3 point magnitude
    sample["map_features"]["pt_magnitudes"] = pad_sequence(
        [b["map_features"]["map_point"]["magnitude"] for b in batch],
        batch_first=True,
    ).float()

    # 3.4 polygon valid masks
    sample["map_features"]["pt_valid_masks"] = pad_sequence(
        [b["map_features"]["map_point"]["valid_mask"] for b in batch],
        batch_first=True,
    ).bool()

    # 4.1 polygon to polygon rpe
    sample["map_features"]["r_pl2pl"] = pad_sequence(
        [b["map_encoder"]["r_pl2pl"] for b in batch],
        batch_first=True,
    ).float()

    # 4.2 polygon to point rpe
    sample["map_features"]["r_pt2pl"] = pad_sequence(
        [b["map_encoder"]["r_pt2pl"] for b in batch],
        batch_first=True,
    ).float()

    # 4.3 agent to polygon rpe
    sample["agent_features"]["r_a2pl"] = pad_sequence(
        [b["agent_encoder"]["r_a2pl"] for b in batch],
        batch_first=True,
    ).float()

    # 4.4 agent to self embeddings
    sample["agent_features"]["x_pt_continuous_embs"] = pad_sequence(
        [b["agent_encoder"]["x_pt_continuous_embs"] for b in batch],
        batch_first=True,
    ).float()

    # 4.5 agent to agent rpe
    sample["agent_features"]["r_a2a"] = pad_sequence(
        [b["agent_encoder"]["r_a2a"] for b in batch],
        batch_first=True,
    ).float()

    # 4.6 agent to time rpe
    sample["agent_features"]["r_a2t"] = pad_sequence(
        [b["agent_encoder"]["r_a2t"] for b in batch],
        batch_first=True,
    ).float()

    # 4.7 mode to polygon rpe
    sample["map_features"]["rpe_m2pl"] = pad_sequence(
        [b["qcnet_header"]["rpe_m2pl"] for b in batch],
        batch_first=True,
    ).float()

    # 4.8 mode to time rpe
    sample["map_features"]["rpe_m2t"] = pad_sequence(
        [b["qcnet_header"]["rpe_m2t"] for b in batch],
        batch_first=True,
    ).float()

    # 4.9 mode to agent rpe
    sample["map_features"]["rpe_m2a"] = pad_sequence(
        [b["qcnet_header"]["rpe_m2a"] for b in batch],
        batch_first=True,
    ).float()

    # ##### agentencoder mask1.0
    agent_state_valid_mask = sample["agent_features"]["state_valid_masks"][
        "his"
    ]  # [B, N, T]
    B, N, T = agent_state_valid_mask.shape
    M = sample["map_features"]["pl_valid_masks"].shape[1]

    a2t_valid_mask = (
        agent_state_valid_mask[..., None]
        .repeat(1, 1, 1, T)
        .reshape(B * N, T, T)
    )  # [B*N, T, T]

    causal_mask = torch.tril(
        torch.ones(T, T, device=a2t_valid_mask.device)
    )  # [T, T]

    a2t_valid_mask = torch.logical_and(
        a2t_valid_mask, causal_mask[None, :, :]
    )  # [B*N, T, T]
    a2t_invalid_mask = ~a2t_valid_mask

    a2pl_valid_mask = torch.logical_and(
        agent_state_valid_mask.permute(0, 2, 1)[..., None],  # [B,T,N,1]
        sample["map_features"]["pl_valid_masks"][
            :, None, None, :
        ],  # [B,1,1,M]
    )
    a2pl_invalid_mask = ~a2pl_valid_mask.reshape(B * T, N, M)  # [B*T, N, M]

    a2a_valid_mask = agent_state_valid_mask.permute(0, 2, 1)  # [B, T, N]
    a2a_valid_mask = torch.logical_and(
        a2a_valid_mask[:, :, :, None],  # [B, T, N, 1]
        a2a_valid_mask[:, :, None, :],  # [B, T, 1, N]
    )  # [B, T, N, N]
    a2a_invalid_mask = ~a2a_valid_mask.reshape(B * T, N, N)  # [B*T, N, N]

    sample["agent_features"]["invalid_mask_a2t"] = a2t_invalid_mask
    sample["agent_features"]["invalid_mask_a2pl"] = a2pl_invalid_mask
    sample["agent_features"]["invalid_mask_a2a"] = a2a_invalid_mask

    B, M, P = sample["map_features"]["pt_valid_masks"].shape
    invalid_mask_pt2pl = ~(
        sample["map_features"]["pt_valid_masks"].reshape(B * M, 1, P).bool()
    )
    invalid_mask_pl2pl = ~torch.logical_and(
        sample["map_features"]["pl_valid_masks"][:, :, None].bool(),
        sample["map_features"]["pl_valid_masks"][:, None, :].bool(),
    )
    sample["map_features"]["invalid_mask_pt2pl"] = invalid_mask_pt2pl
    sample["map_features"]["invalid_mask_pl2pl"] = invalid_mask_pl2pl

    # ##### qcnethead mask1.0
    agent_state_valid_mask = sample["agent_features"]["state_valid_masks"][
        "his"
    ]  # [B, N, HT]
    agent_valid_mask = sample["agent_features"]["agent_valid_masks"]  # [B, N]

    B, N, HT = agent_state_valid_mask.shape
    M = sample["map_features"]["pl_valid_masks"].shape[1]
    # K = 6
    K = 3

    invalid_mask_m2t = ~(
        agent_state_valid_mask[:, None, :, :].repeat(1, K, 1, 1).bool()
    ).reshape(
        B * K * N, 1, HT
    )  # [B*K*N, 1, HT]
    invalid_mask_m2pl = ~(
        sample["map_features"]["pl_valid_masks"][:, None, None, :]
        .repeat(1, K, N, 1)
        .reshape(B * K * N, 1, M)
        .bool()
    )  # [B*K*N, 1, M]
    invalid_mask_m2a = ~(
        torch.logical_and(
            agent_valid_mask[:, None, :],
            agent_valid_mask[:, :, None],
        )[:, None, :, :]
        .repeat(1, K, 1, 1)
        .reshape(B * K, N, N)
        .bool()
    )  # [B*K, N, N]
    sample["map_features"]["invalid_mask_m2t"] = invalid_mask_m2t
    sample["map_features"]["invalid_mask_m2pl"] = invalid_mask_m2pl
    sample["map_features"]["invalid_mask_m2a"] = invalid_mask_m2a

    # 3.4 point types
    sample["map_features"]["pt_types"] = pad_sequence(
        [b["map_features"]["map_point"]["point_type"] for b in batch],
        batch_first=True,
    ).long()

    sample["agent_features"]["lateral"] = pad_sequence(
        [b["lateral"] for b in batch],
        batch_first=True,
    ).long()

    return sample


def pnp_e2e_collate(
    batch: List[Dict[str, torch.tensor]],
) -> Dict[str, Any]:
    """PnP collate function is used for Carizon driving data.

    Args:
        batch: Bactched data to be collated with.
    """
    sample: Dict[str, Any] = dict()

    # agent features padding
    sample["map_encoder"] = {}
    sample["agent_encoder"] = {}
    sample["qcnet_header"] = {}
    sample["agent_features"] = {}
    sample["agent_features"]["acs"] = {}
    sample["agent_features"]["acs"]["poses"] = {}
    sample["agent_features"]["agent_properties"] = {}
    sample["agent_features"]["state_valid_masks"] = {}

    sample["agent_features"]["gcs"] = {}
    sample["agent_features"]["gcs"]["poses"] = {}

    # 1.1.1 pad historical trajectories
    sample["agent_features"]["gcs"]["poses"]["his"] = pad_sequence(
        [
            b["agent_features"]["gcs"]["his_pose"][:, :, [0, 1, 4]]
            for b in batch
        ],
        batch_first=True,
    ).float()

    # 1.1.3 pad future trajectories (gt in acs)
    sample["agent_features"]["acs"]["poses"]["fut"] = pad_sequence(
        [
            b["agent_features"]["acs"]["fut_pose"][:, :, [0, 1, 4]]
            for b in batch
        ],
        batch_first=True,
    ).float()

    # 1.3.1 pad historical valid masks
    sample["agent_features"]["state_valid_masks"]["his"] = pad_sequence(
        [b["agent_features"]["his_valid_masks"] for b in batch],
        batch_first=True,
        padding_value=False,
    ).bool()

    # 1.3.2 pad future valid masks
    sample["agent_features"]["state_valid_masks"]["fut"] = pad_sequence(
        [b["agent_features"]["fut_valid_masks"] for b in batch],
        batch_first=True,
        padding_value=False,
    ).bool()

    # 1.4 pad current valid masks
    sample["agent_features"]["agent_valid_masks"] = pad_sequence(
        [b["agent_features"]["his_valid_masks"][:, -1] for b in batch],
        batch_first=True,
        padding_value=False,
    ).bool()

    # 1.5.1 pad agent types
    sample["agent_features"]["agent_properties"]["classes"] = pad_sequence(
        [b["agent_features"]["types"] for b in batch],
        batch_first=True,
    ).long()

    # 1.5.2 pad agent ids
    sample["agent_features"]["track_ids"] = pad_sequence(
        [b["agent_features"]["track_ids"] for b in batch],
        batch_first=True,
    ).long()

    # 1.6 pad agent categories
    sample["agent_features"]["categories"] = pad_sequence(
        [b["agent_features"]["category"] for b in batch],
        batch_first=True,
    ).long()

    # map polygon features padding
    sample["map_features"] = {}
    sample["map_features"]["gcs"] = {}

    # 2.2 polygon types
    sample["map_features"]["pl_types"] = pad_sequence(
        [b["map_features"]["map_polygon"]["type"] for b in batch],
        batch_first=True,
    ).long()

    # 2.2.2 polygon arrow types
    sample["map_features"]["arrow_types"] = pad_sequence(
        [b["map_features"]["map_polygon"]["arrow_type"] for b in batch],
        batch_first=True,
    ).long()

    # 2.2.3 light status types
    sample["map_features"]["light_status"] = pad_sequence(
        [b["map_features"]["map_polygon"]["light_status"] for b in batch],
        batch_first=True,
    ).long()

    # 2.2.4 signal turning types
    sample["map_features"]["signal_type"] = pad_sequence(
        [b["map_features"]["map_polygon"]["signal_type"] for b in batch],
        batch_first=True,
    ).long()

    # 2.3 polygon valid masks
    sample["map_features"]["pl_valid_masks"] = pad_sequence(
        [b["map_features"]["map_polygon"]["valid_mask"] for b in batch],
        batch_first=True,
    ).bool()

    # 3.1 point poses
    sample["map_features"]["gcs"]["pt_poses"] = pad_sequence(
        [b["map_features"]["map_point"]["position"] for b in batch],
        batch_first=True,
    ).float()

    # 2.4 point magnitude
    sample["map_features"]["pt_magnitudes"] = pad_sequence(
        [b["map_features"]["map_point"]["magnitude"] for b in batch],
        batch_first=True,
    ).float()

    sample["map_features"]["pt_valid_masks"] = pad_sequence(
        [b["map_features"]["map_point"]["valid_mask"] for b in batch],
        batch_first=True,
    ).bool()

    # 4.1 polygon to polygon rpe
    sample["map_encoder"]["r_pl2pl"] = pad_sequence(
        [b["map_encoder"]["r_pl2pl"] for b in batch],
        batch_first=True,
    ).float()

    # 4.2 polygon to point rpe
    sample["map_encoder"]["r_pt2pl"] = pad_sequence(
        [b["map_encoder"]["r_pt2pl"] for b in batch],
        batch_first=True,
    ).float()

    # 4.3 agent to polygon rpe
    sample["agent_encoder"]["r_a2pl"] = pad_sequence(
        [b["agent_encoder"]["r_a2pl"] for b in batch],
        batch_first=True,
    ).float()

    # 4.4 agent to self embeddings
    sample["agent_encoder"]["x_pt_continuous_embs"] = pad_sequence(
        [b["agent_encoder"]["x_pt_continuous_embs"] for b in batch],
        batch_first=True,
    ).float()

    # 4.5 agent to agent rpe
    sample["agent_encoder"]["r_a2a"] = pad_sequence(
        [b["agent_encoder"]["r_a2a"] for b in batch],
        batch_first=True,
    ).float()

    # 4.6 agent to time rpe
    sample["agent_encoder"]["r_a2t"] = pad_sequence(
        [b["agent_encoder"]["r_a2t"] for b in batch],
        batch_first=True,
    ).float()

    # 4.7 mode to polygon rpe
    sample["qcnet_header"]["rpe_m2pl"] = pad_sequence(
        [b["qcnet_header"]["rpe_m2pl"] for b in batch],
        batch_first=True,
    ).float()

    # 4.8 mode to time rpe
    sample["qcnet_header"]["rpe_m2t"] = pad_sequence(
        [b["qcnet_header"]["rpe_m2t"] for b in batch],
        batch_first=True,
    ).float()

    # 4.9 mode to agent rpe
    sample["qcnet_header"]["rpe_m2a"] = pad_sequence(
        [b["qcnet_header"]["rpe_m2a"] for b in batch],
        batch_first=True,
    ).float()

    return sample
