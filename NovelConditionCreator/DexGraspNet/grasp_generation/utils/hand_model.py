"""
Last modified date: 2023.04.12
Author: Jialiang Zhang, Ruicheng Wang
Description: Class HandModel
"""

import json
import numpy as np
import torch
import trimesh as tm
import open3d as o3d
import pymeshlab
from manopth.manolayer import ManoLayer
import plotly.graph_objects as go
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_points


class HandModel:

    def __init__(self, mano_root, contact_indices_path, pose_distrib_path, device='cpu'):
        """
        Create a Hand Model for MANO
        
        Parameters
        ----------
        mano_root: str
            base directory of MANO_RIGHT.pkl
        contact_indices_path: str
            path to hand-selected contact candidates
        pose_distrib_path: str
            path to a multivariate gaussian distribution of the `thetas` of MANO
        device: str | torch.Device
            device for torch tensors
        """

        # load MANO

        self.device = device
        self.manolayer = ManoLayer(mano_root=mano_root, flat_hand_mean=True, use_pca=False).to(device=self.device)

        # load contact points and pose distribution

        with open(contact_indices_path, 'r') as f:
            self.contact_indices = json.load(f)
        self.contact_indices = torch.tensor(self.contact_indices, dtype=torch.long, device=self.device)
        self.n_contact_candidates = len(self.contact_indices)

        self.pose_distrib = torch.load(pose_distrib_path, map_location=device)

        # parameters

        self.hand_pose = None
        self.contact_point_indices = None
        self.vertices = None
        self.keypoints = None
        self.contact_points = None

    def set_parameters(self, hand_pose, contact_point_indices=None):
        """
        Set translation, rotation, thetas, and contact points of grasps
        
        Parameters
        ----------
        hand_pose: (B, 3+3+45) torch.FloatTensor
            translation, rotation in axis angles, and `thetas`
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.vertices, self.keypoints = self.manolayer.forward(
            th_trans=self.hand_pose[:, :3],
            th_pose_coeffs=self.hand_pose[:, 3:],
        )
        self.vertices /= 1000.0
        self.keypoints /= 1000.0
        self.contact_point_indices = contact_point_indices
        self.contact_points = self.vertices[torch.arange(len(hand_pose)).unsqueeze(1), self.contact_indices[self.contact_point_indices]]

    def cal_dis_plane(self, p):
        """
        Calculate signed distances from each MANO vertex to table plane (above+, below-)
        
        Parameters
        ----------
        p: (B, 4) torch.Tensor
            plane parameters in object reference frame: (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
        Returns
        -------
        dis: (B, 778) torch.Tensor
            signed distances from each MANO vertex to table plane
        """
        dis = (p[:, :3].unsqueeze(1) * self.vertices).sum(2) + p[:, 3].unsqueeze(1)
        return dis

    def cal_distance(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes
        
        Interiors are positive, exteriors are negative
        
        Use the inner product of the ObjectPoint-to-HandNearestNeighbour vector 
        and the vertex normal of the HandNearestNeighbour to approximate the sdf
        
        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """
        # Alternative 1: directly using Kaolin results in a time-consuming for-loop along the batch dimension
        # Alternative 2: discarding the inner product with the vertex normal will mess up the optimization severely
        # we reserve the implementation of the second alternative as comments below
        mesh = Meshes(verts=self.vertices, faces=self.manolayer.th_faces.unsqueeze(0).repeat(len(x), 1, 1))
        normals = mesh.verts_normals_packed().view(-1, 778, 3)
        knn_result = knn_points(x, self.vertices, K=1)
        knn_idx = (torch.arange(len(x)).unsqueeze(1), knn_result.idx[:, :, 0])
        dis = -((x - self.vertices[knn_idx]) * normals[knn_idx].detach()).sum(dim=-1)
        # interior = ((x - self.vertices[knn_idx]) * normals[knn_idx]).sum(dim=-1) < 0
        # dis = torch.sqrt(knn_result.dists[:, :, 0] + 1e-8)
        # dis = torch.where(interior, dis, -dis)
        return dis

    def self_penetration(self):
        """
        Calculate self penetration energy
        
        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        dis = (self.keypoints.unsqueeze(1) - self.keypoints.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        loss = -torch.where(dis < 0.018, dis, torch.zeros_like(dis))
        return loss.sum((1, 2))

    def get_contact_candidates(self):
        """
        Get all contact candidates
        
        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        return self.vertices[torch.arange(len(self.vertices)).unsqueeze(1), self.contact_indices.unsqueeze(0)]

    def get_penetraion_keypoints(self):
        """
        Get MANO keypoints
        
        Returns
        -------
        points: (N, 21, 3) torch.Tensor
            MANO keypoints
        """
        return self.keypoints

    def get_plotly_data(self, i, opacity=0.5, color='lightblue', with_keypoints=False, with_contact_points=False, pose=None):
        """
        Get visualization data for plotly.graph_objects
        
        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_keypoints: bool
            whether to visualize keypoints
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix
        
        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        v = self.vertices[i].detach().cpu().numpy()
        if pose is not None:
            v = v @ pose[:3, :3].T + pose[:3, 3]
        f = self.manolayer.th_faces
        if isinstance(f, torch.Tensor):
            f = f.detach().cpu().numpy()
        hand_plotly = [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], text=list(range(len(v))), color=color, opacity=opacity, hovertemplate='%{text}')]
        if with_keypoints:
            keypoints = self.keypoints[i].detach().cpu().numpy()
            if pose is not None:
                keypoints = keypoints @ pose[:3, :3].T + pose[:3, 3]
            hand_plotly.append(go.Scatter3d(x=keypoints[:, 0], y=keypoints[:, 1], z=keypoints[:, 2], mode='markers', marker=dict(color='red', size=5)))
            for penetration_keypoint in keypoints:
                mesh = tm.primitives.Capsule(radius=0.009, height=0)
                v = mesh.vertices + penetration_keypoint
                f = mesh.faces
                hand_plotly.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='burlywood', opacity=0.5))
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            hand_plotly.append(go.Scatter3d(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2], mode='markers', marker=dict(color='red', size=5)))
        return hand_plotly

    def get_trimesh_data(self, i, pose=None, open3d=True):
        """
        Get visualization data for trimesh
        
        Parameters
        ----------
        i: int
            index of data
        
        Returns
        -------
        data: trimesh.Trimesh
        """
        v = self.vertices[i].detach().cpu().numpy()
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            v = v @ pose[:3, :3].T + pose[:3, 3]
        f = self.manolayer.th_faces.detach().cpu().numpy()
        right_extra_faces = np.array([[118, 239, 122], [119, 79, 215], [78, 121, 214], [120, 79, 119], [239, 118, 279], [108, 79, 120], [118, 117, 279], [239, 234, 122],
                                      [78, 214, 79], [234, 92, 122], [214, 215, 79], [279, 117, 215], [117, 119, 215], [92, 38, 122]])
        f = np.concatenate((f, right_extra_faces), axis=0)
        mesh = tm.Trimesh(v, f)
        if open3d:
            mesh = mesh.as_open3d
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
        # mesh.triangles = o3d.utility.Vector3iVector(f.astype(np.int64))
        return mesh

    def get_pymeshlab_data(self, i, pose=None):
        """
        Get visualization data for trimesh
        
        Parameters
        ----------
        i: int
            index of data
        
        Returns
        -------
        data: trimesh.Trimesh
        """
        v = self.vertices[i].detach().cpu().numpy()
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            v = v @ pose[:3, :3].T + pose[:3, 3]
        f = self.manolayer.th_faces.detach().cpu().numpy()
        right_extra_faces = np.array([[118, 239, 122], [119, 79, 215], [78, 121, 214], [120, 79, 119], [239, 118, 279], [108, 79, 120], [118, 117, 279], [239, 234, 122],
                                      [78, 214, 79], [234, 92, 122], [214, 215, 79], [279, 117, 215], [117, 119, 215], [92, 38, 122]])
        f = np.concatenate((f, right_extra_faces), axis=0)
        mesh = pymeshlab.Mesh(vertex_matrix=v, face_matrix=f)
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
        # mesh.triangles = o3d.utility.Vector3iVector(f.astype(np.int64))
        return mesh
