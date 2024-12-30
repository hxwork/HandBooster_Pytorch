import os
import cv2

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import pyrender
from pyrender.shader_program import ShaderProgramCache
import trimesh
import matplotlib.pyplot as plt

from seg import *


def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0., alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.zeros_like(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


class CustomShaderCache():

    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram('shaders/mesh.vert', 'shaders/mesh.frag', defines=defines)
        return self.program


def render_hand_obj_property(shape, hand_verts, hand_faces, obj_mesh, obj_pose, cam_param):
    # hand mesh
    hand_mesh = trimesh.Trimesh(hand_verts, hand_faces, process=False)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    hand_mesh.apply_transform(rot)
    hand_mesh = pyrender.Mesh.from_trimesh(hand_mesh, smooth=True)

    # new scene
    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]))

    # add hand mesh
    scene.add(hand_mesh, 'hand_mesh')

    # add obj mesh
    for o in range(len(obj_pose)):
        if np.all(obj_pose[o] == 0.0):
            continue
        pose = np.vstack((obj_pose[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
        pose[1] *= -1
        pose[2] *= -1
        scene.add(obj_mesh[o], pose=pose)

    # add camera
    focal = [cam_param[0], cam_param[1]]
    princpt = [cam_param[2], cam_param[3]]
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=shape[1], viewport_height=shape[0], point_size=1.0)

    # render
    renderer._renderer._program_cache = ShaderProgramCache(shader_dir='/data/code/DatasetPreprocessing/dexycb/shaders')
    normal, depth = renderer.render(scene)
    renderer.delete()

    # post processing
    depth = depth.astype(np.float32)

    return normal, depth


# render hand mesh and textured object mesh
def render_hand_obj_condition(shape, hand_verts, hand_faces, obj_mesh, obj_pose, hand_type, cam_param):
    # hand mesh
    hand_mesh = trimesh.Trimesh(hand_verts, hand_faces, process=False)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    hand_mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    hand_mesh = pyrender.Mesh.from_trimesh(hand_mesh, material=material, smooth=True)

    # new scene
    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]), ambient_light=np.array([0.3, 0.3, 0.3]))

    # add hand mesh
    scene.add(hand_mesh, 'hand_mesh')

    # add object mesh
    for o in range(len(obj_pose)):
        if np.all(obj_pose[o] == 0.0):
            continue
        pose = np.vstack((obj_pose[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
        pose[1] *= -1
        pose[2] *= -1
        scene.add(obj_mesh[o], pose=pose)

    # add camera
    focal = [cam_param[0], cam_param[1]]
    princpt = [cam_param[2], cam_param[3]]
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=shape[1], viewport_height=shape[0], point_size=1.0)

    # render RGB and depth
    flag = pyrender.constants.RenderFlags.OFFSCREEN | pyrender.RenderFlags.RGBA
    rgb, _ = renderer.render(scene, flags=flag)
    renderer.delete()

    # post process
    rgb = rgb[:, :, :3].astype(np.uint8)

    return rgb
