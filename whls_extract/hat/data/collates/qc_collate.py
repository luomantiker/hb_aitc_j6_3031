from typing import Optional

import torch

__all__ = ["collate_qc_argoverse2"]


def collate_qc_argoverse2(
    batch: dict,
    ori_historical_sec: int = 5,
    ori_future_sec: int = 6,
    ori_sample_fre: int = 10,
    sample_fre: int = 2,
    stage: str = "train",
    pl_N: Optional[int] = None,
    pt_N: Optional[int] = None,
    pt_N_downsample_nums: int = 50,
    agent_num: Optional[int] = None,
    add_noise: bool = False,
    num_historical_steps: int = 50,
):
    """
    Collate function for QCNet with Argoverse2 dataset.

    This function preprocesses a batch of data, creating tensors for agents,
    map polygons, map points, and their relationships,
    to be used as input to the QCNet model.

    Args:
        batch: Batch of data containing agent and map information.
        ori_historical_sec: Original data historical sampling duration
        ori_future_sec: Original data future sampling duration (in seconds)
        ori_sample_fre: Original data sampling frequency
        sample_fre: Current sampling frequency
        stage: The stage of processing, e.g., 'train', 'val', 'test'.
        pl_N: Number of polygons for map polygon to agent cross attention.
        pt_N: Number of points for map polygon to agent cross attention.
        pt_N_downsample_nums: Number of map points to downsample from the.
        agent_num: Number of agents in the dataset.
        add_noise: Flag to add noise to the agent's position and velocity.
        num_historical_steps: Number of historical steps.

    Returns:
        dict: A dictionary containing processed agent and map data.
    """
    return_data = {
        "agent": {},
        "map_polygon": {},
        "map_point": {},
        "map_point_to_map_polygon": {},
        "map_polygon_to_map_polygon": {},
    }
    B = len(batch)
    input_dim = 2
    FS = sample_fre  # Downsampling Frequency
    HT = ori_sample_fre * ori_historical_sec  # Original Historical Input Steps
    samp_his_steps = ori_historical_sec * FS  # Time Steps of Historical Data
    his_samp_list = torch.arange(
        HT // samp_his_steps - 1, HT, HT // samp_his_steps
    )
    fu_samp_list = torch.arange(
        HT - 1 + HT // samp_his_steps,
        (ori_historical_sec + ori_future_sec) * ori_sample_fre,
        HT // samp_his_steps,
    )
    samp_list = torch.cat((his_samp_list, fu_samp_list))
    av_cur_pos = []
    for i in range(B):
        av_idx = batch[i]["agent"]["av_index"]
        av_cur = batch[i]["agent"]["position"][av_idx, HT - 1, :input_dim]
        av_cur_pos.append(av_cur)
    max_pl = max([b["map_polygon"]["num_nodes"] for b in batch])
    if pl_N is None or (stage == "train" and pl_N >= max_pl):
        pl_N = max_pl
        pl_idx = [
            torch.arange(b["map_polygon"]["num_nodes"]).long() for b in batch
        ]
    else:
        pl_idx = []
        for i in range(B):
            pl_pos = batch[i]["map_polygon"]["position"]
            if pl_pos.shape[0] > pl_N:
                pl_dist = torch.norm(pl_pos - av_cur_pos[i], p=2, dim=-1)
                if stage != "train":
                    _, pl_k = torch.topk(-pl_dist, k=pl_N, dim=-1)
                else:
                    k1 = 20
                    _, pl_k1 = torch.topk(-pl_dist, k=k1, dim=-1)
                    all_indices = torch.arange(pl_pos.shape[0])
                    remain_idx = all_indices[~torch.isin(all_indices, pl_k1)]
                    pl_k2 = remain_idx[
                        torch.randperm(len(remain_idx))[: pl_N - k1]
                    ]

                    pl_k = torch.cat([pl_k1, pl_k2])
            else:
                pl_k = torch.arange(pl_pos.shape[0])
            pl_idx.append(pl_k)
    if pt_N is None:
        pt_N = max(
            [
                max([p.shape[0] for p in b["map_point"]["position"]])
                for b in batch
            ]
        )

    mp = {}
    mp["position"] = torch.zeros([B, pl_N, pt_N, 2])
    mp["orientation"] = torch.zeros([B, pl_N, pt_N])
    mp["magnitude"] = torch.zeros([B, pl_N, pt_N])
    mp["height"] = torch.zeros([B, pl_N, pt_N])
    mp["pt_type"] = torch.zeros([B, pl_N, pt_N]).long()
    mp["side"] = torch.zeros([B, pl_N, pt_N]).long()
    mp["mask"] = torch.zeros([B, pl_N, pt_N]).bool()

    if pt_N_downsample_nums is not None and pt_N_downsample_nums < pt_N:
        step = pt_N / pt_N_downsample_nums
        pt_index = torch.arange(0, pt_N, step).long()[:pt_N_downsample_nums]
        mp_s = {}
        mp_s["position"] = torch.zeros([B, pl_N, pt_N_downsample_nums, 2])
        mp_s["orientation"] = torch.zeros([B, pl_N, pt_N_downsample_nums])
        mp_s["magnitude"] = torch.zeros([B, pl_N, pt_N_downsample_nums])
        mp_s["height"] = torch.zeros([B, pl_N, pt_N_downsample_nums])
        mp_s["pt_type"] = torch.zeros([B, pl_N, pt_N_downsample_nums]).long()
        mp_s["side"] = torch.zeros([B, pl_N, pt_N_downsample_nums]).long()
        mp_s["mask"] = torch.zeros([B, pl_N, pt_N_downsample_nums]).bool()

    for i, b in enumerate(batch):
        pl_num = b["map_polygon"]["num_nodes"]
        av_idx = batch[i]["agent"]["av_index"]
        av_pos0 = batch[i]["agent"]["position"][av_idx][0]
        pt = b["map_point"]

        for j in range(pl_idx[i].shape[0]):
            if pl_idx is not None:
                pt_j = pl_idx[i][j]
            pt_num = len(pt["position"][pt_j])

            mp["position"][i, j, :pt_num, :] = (
                pt["position"][pt_j][:pt_N, :input_dim]
                - av_pos0[None, :input_dim]
            )
            mp["orientation"][i, j, :pt_num] = pt["orientation"][pt_j][:pt_N]
            mp["magnitude"][i, j, :pt_num] = pt["magnitude"][pt_j][:pt_N]
            mp["height"][i, j, :pt_num] = pt["height"][pt_j][:pt_N]
            mp["pt_type"][i, j, :pt_num] = pt["type"][pt_j][:pt_N]
            mp["side"][i, j, :pt_num] = pt["side"][pt_j][:pt_N]
            mp["mask"][i, j, :pt_num] = True

            if (
                pt_N_downsample_nums is not None
                and pt_N_downsample_nums < pt_N
            ):
                mp_s["position"][i, j, :pt_N_downsample_nums, :] = mp[
                    "position"
                ][i, j, pt_index, :]
                mp_s["orientation"][i, j, :pt_N_downsample_nums] = mp[
                    "orientation"
                ][i, j, pt_index]
                mp_s["magnitude"][i, j, :pt_N_downsample_nums] = mp[
                    "magnitude"
                ][i, j, pt_index]
                mp_s["height"][i, j, :pt_N_downsample_nums] = mp["height"][
                    i, j, pt_index
                ]
                mp_s["pt_type"][i, j, :pt_N_downsample_nums] = mp["pt_type"][
                    i, j, pt_index
                ]
                mp_s["side"][i, j, :pt_N_downsample_nums] = mp["side"][
                    i, j, pt_index
                ]
                mp_s["mask"][i, j, :pt_N_downsample_nums] = mp["mask"][
                    i, j, pt_index
                ]

    if pt_N_downsample_nums is not None and pt_N_downsample_nums < pt_N:
        return_data["map_point"] = mp_s
    else:
        return_data["map_point"] = mp

    mpl = {}
    mpl["position"] = torch.zeros([B, pl_N, 2])
    mpl["orientation"] = torch.zeros([B, pl_N])
    mpl["height"] = torch.zeros([B, pl_N])
    mpl["pl_type"] = torch.zeros([B, pl_N]).long()
    mpl["is_intersection"] = torch.zeros([B, pl_N]).long()
    mpl["valid_mask"] = torch.zeros([B, pl_N]).bool()
    for i, b in enumerate(batch):
        pl_num = b["map_polygon"]["num_nodes"]
        pl = b["map_polygon"]
        av_idx = batch[i]["agent"]["av_index"]
        av_pos0 = batch[i]["agent"]["position"][av_idx][0]
        pl_pos = pl["position"][:, :input_dim] - av_pos0[None, :input_dim]
        mpl["position"][i, :pl_num, :] = pl_pos[pl_idx[i]]
        mpl["orientation"][i, :pl_num] = pl["orientation"][pl_idx[i]]
        mpl["height"][i, :pl_num] = pl["height"][pl_idx[i]]
        mpl["pl_type"][i, :pl_num] = pl["type"][pl_idx[i]]
        mpl["is_intersection"][i, :pl_num] = pl["is_intersection"][pl_idx[i]]
        mpl["valid_mask"][i, :pl_num] = True

    return_data["map_polygon"] = mpl
    type_pl2pl = torch.zeros(B, pl_N, pl_N).long()
    for i, b in enumerate(batch):
        edge_pl = b["map_polygon_to_map_polygon"]["edge_index"]
        pl_type = b["map_polygon_to_map_polygon"]["type"]

        for j in range(min(pl_N, edge_pl.shape[1])):
            if edge_pl[1, j] < pl_N and edge_pl[0, j] < pl_N:
                type_pl2pl[i, edge_pl[1, j], edge_pl[0, j]] = pl_type[j]

    return_data["type_pl2pl"] = type_pl2pl
    max_A = max([b["agent"]["num_nodes"] for b in batch])
    if agent_num is None:
        agent_num = max_A
    if stage == "train":
        agent_num = min(agent_num, max_A, 80)
    if agent_num is None or (stage == "train" and agent_num >= max_A):
        A = max_A
        a_idx = [torch.arange(b["agent"]["num_nodes"]).long() for b in batch]
    else:
        A = agent_num
        a_idx = []
        for i in range(B):
            agent_pos = batch[i]["agent"]["position"][:, HT - 1, :input_dim]
            if agent_pos.shape[0] > A:
                a_dist = torch.norm(agent_pos - av_cur_pos[i], p=2, dim=-1)
                if stage != "train":
                    _, a_k = torch.topk(-a_dist, k=A, dim=-1)
                else:
                    k1 = 10
                    _, a_k1 = torch.topk(-a_dist, k=k1, dim=-1)
                    all_indices = torch.arange(agent_pos.shape[0])
                    remain_idx = all_indices[~torch.isin(all_indices, a_k1)]
                    a_k2 = remain_idx[
                        torch.randperm(len(remain_idx))[: A - k1]
                    ]

                    a_k = torch.cat([a_k1, a_k2])
            else:
                a_k = torch.arange(agent_pos.shape[0])
            focal_idx = (batch[i]["agent"]["category"] == 3).nonzero()

            if focal_idx[0][0] not in a_k:
                a_k = torch.cat([a_k[:-1], focal_idx[0]], dim=-1)
            assert focal_idx[0][0] in a_k
            a_idx.append(a_k)
    # Total Steps of Historical and Forecast Data After Downsampling
    T = batch[0]["agent"]["valid_mask"].shape[1] // ori_sample_fre * FS
    agent = {}
    agent["num_nodes"] = [b["agent"]["num_nodes"] for b in batch]
    agent["valid_mask"] = torch.zeros([B, A, T], dtype=torch.bool)
    agent["predict_mask"] = torch.zeros([B, A, T], dtype=torch.bool)
    agent["agent_type"] = torch.zeros([B, A, 1], dtype=torch.long)
    agent["position"] = torch.zeros([B, A, T, 2], dtype=torch.float)
    agent["heading"] = torch.zeros([B, A, T], dtype=torch.float)
    agent["velocity"] = torch.zeros([B, A, T, 2], dtype=torch.float)
    agent["category"] = torch.zeros([B, A, 1], dtype=torch.long)

    for i, _b in enumerate(batch):
        a = batch[i]["agent"]["num_nodes"]
        av_idx = batch[i]["agent"]["av_index"]
        av_pos0 = batch[i]["agent"]["position"][av_idx][0]
        pos = (
            batch[i]["agent"]["position"][a_idx[i], :, :2] - av_pos0[None, :2]
        )
        pos = pos[:, samp_list, :]
        valid_mask = batch[i]["agent"]["valid_mask"][a_idx[i]]
        valid_mask = valid_mask[:, samp_list]
        vel = batch[i]["agent"]["velocity"][a_idx[i], :, :2]
        vel = vel[:, samp_list, :]
        if add_noise:
            noise1 = (
                torch.randn((a, samp_his_steps, 2)).to(valid_mask.device)
                * 0.01
            )
            noise2 = (
                torch.randn((a, samp_his_steps, 2)).to(valid_mask.device)
                * 0.002
            )
            pos[:, :samp_his_steps] = pos[:, :samp_his_steps] + noise1
            vel[:, :samp_his_steps] = vel[:, :samp_his_steps] + noise2
        pos = torch.where(valid_mask.unsqueeze(-1), pos, 0)
        agent["valid_mask"][i, :a] = valid_mask
        predict_mask = batch[i]["agent"]["predict_mask"][a_idx[i]]
        agent["predict_mask"][i, :a] = predict_mask[:, samp_list]
        agent["agent_type"][i, :a, 0] = batch[i]["agent"]["type"][a_idx[i]]

        agent["position"][i, :a] = pos
        heading = batch[i]["agent"]["heading"][a_idx[i]]
        agent["heading"][i, :a] = heading[:, samp_list]
        agent["velocity"][i, :a] = vel
        agent["category"][i, :a, 0] = batch[i]["agent"]["category"][a_idx[i]]

    mask_dst = agent["predict_mask"].any(dim=-1, keepdim=True)

    mask_src = agent["valid_mask"][:, :, :samp_his_steps]
    mask_ta = agent["valid_mask"].transpose(1, 2)[
        :, :samp_his_steps
    ]  # [B, T, A]
    agent["valid_mask_a2a"] = mask_ta[:, :, :, None] & mask_ta[:, :, None, :]
    return_data["agent"] = agent
    decoder = {}
    decoder["mask_dst"] = mask_dst  # [B, A, 1]
    decoder["mask_a2m"] = mask_dst[:, :, :] & mask_src[:, None, :, -1]
    return_data["decoder"] = decoder
    return return_data
