"""
Last modified date: 2023.04.12
Author: Jialiang Zhang
Description: energy functions
"""

import torch


def cal_energy(hand_model, object_model, w_dis=100.0, w_pen=100.0, w_prior=0.5, w_spen=10.0, w_tpen=40.0, on_table=False, verbose=False):

    # E_dis
    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = object_model.device
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)

    # E_fc
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)
    transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0], [0, 0, 1, 0, 0, 0, -1, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)
    g = torch.cat([
        torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3),
        (hand_model.contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)
    ],
                  dim=2).float().to(device)
    norm = torch.norm(contact_normal @ g, dim=[1, 2])
    E_fc = norm * norm

    # E_pen
    object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    object_surface_points = object_model.surface_points_tensor * object_scale  # (n_objects * batch_size_each, num_samples, 3)
    distances = hand_model.cal_distance(object_surface_points)
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)

    # E_prior
    E_prior = torch.norm((hand_model.hand_pose[:, 6:] - hand_model.pose_distrib[0]) / hand_model.pose_distrib[1], dim=-1)

    # E_spen
    E_spen = hand_model.self_penetration()

    if on_table:
        # E_tpen
        plane_distances = hand_model.cal_dis_plane(object_model.plane_parameters)  # [B, 778]
        plane_distances[plane_distances > 0] = 0
        E_tpen = -plane_distances.sum(-1)
    else:
        E_tpen = torch.zeros_like(E_spen, device=E_spen.device)

    if verbose:
        return E_fc + w_dis * E_dis + w_pen * E_pen + w_prior * E_prior + w_spen * E_spen + w_tpen * E_tpen, E_fc, E_dis, E_pen, E_prior, E_spen, E_tpen
    else:
        return E_fc + w_dis * E_dis + w_pen * E_pen + w_prior * E_prior + w_spen * E_spen + w_tpen * E_tpen
