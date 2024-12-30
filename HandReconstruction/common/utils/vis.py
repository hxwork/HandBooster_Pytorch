import os
import cv2
import copy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
import pyrender
import trimesh

from common.utils.preprocessing import generate_patch_image

# from seg import *


def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1, thickness=2):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(kp_mask, p1, p2, color=colors[l], thickness=thickness, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_joints_3d(kpt_3d, kps_lines, output_path):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        # colors
        if i1 in palm_joints_idx:
            line_c = palm_color
            i1_point_c = palm_color
        elif i1 in dmz_joints_idx:
            line_c = dmz_color
            i1_point_c = dmz_color
        elif i1 in sz_joints_idx:
            line_c = sz_color
            i1_point_c = sz_color
        elif i1 in zz_joints_idx:
            line_c = zz_color
            i1_point_c = zz_color
        elif i1 in wmz_joints_idx:
            line_c = wmz_color
            i1_point_c = wmz_color
        elif i1 in xmz_joints_idx:
            line_c = xmz_color
            i1_point_c = xmz_color

        if i2 in palm_joints_idx:
            i2_point_c = palm_color
        elif i2 in dmz_joints_idx:
            i2_point_c = dmz_color
        elif i2 in sz_joints_idx:
            i2_point_c = sz_color
        elif i2 in zz_joints_idx:
            i2_point_c = zz_color
        elif i2 in wmz_joints_idx:
            i2_point_c = wmz_color
        elif i2 in xmz_joints_idx:
            i2_point_c = xmz_color

        # lines
        ax.plot(x, z, -y, c=line_c[::-1] / 255., linewidth=2)
        # points
        ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=i1_point_c[::-1] / 255., marker='o')
        ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=i2_point_c[::-1] / 255., marker='o')

    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    # legend
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    # manually define a new patch
    p_line = mpatches.Patch(color=palm_color[::-1] / 255., label='P: Palm')
    t_line = mpatches.Patch(color=dmz_color[::-1] / 255., label='T: Thumb')
    i_line = mpatches.Patch(color=sz_color[::-1] / 255., label='I: Index')
    m_line = mpatches.Patch(color=zz_color[::-1] / 255., label='M: Middle')
    r_line = mpatches.Patch(color=wmz_color[::-1] / 255., label='R: Ring')
    l_line = mpatches.Patch(color=xmz_color[::-1] / 255., label='L: Little')
    # handles is a list, so append manual patch
    handles = [p_line, t_line, i_line, m_line, r_line, l_line]
    # plot the legend
    # plt.legend(handles=handles, loc='lower center', ncols=3, bbox_to_anchor=(0.5, 1))
    plt.legend(handles=handles, loc='lower left', ncols=1, bbox_to_anchor=(0.04, 0.05))

    plt.savefig(output_path, dpi=400)


def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)


def vis_mesh_finger_colors(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    # cmap = plt.get_cmap('rainbow')
    # colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    # colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    colors = np.zeros_like(mesh_vertex)
    colors[dmz_verts_idx] = dmz_color
    colors[sz_verts_idx] = sz_color
    colors[zz_verts_idx] = zz_color
    colors[wmz_verts_idx] = wmz_color
    colors[xmz_verts_idx] = xmz_color
    colors[palm_verts_idx] = palm_color
    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i].tolist(), thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)


# def save_obj(v, f, file_name='output.obj'):
#     obj_file = open(file_name, 'w')
#     for i in range(len(v)):
#         obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
#     for i in range(len(f)):
#         obj_file.write('f ' + str(f[i][0] + 1) + '/' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + '/' + str(f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '/' +
#                        str(f[i][2] + 1) + '\n')
#     obj_file.close()


def save_obj(v, f, vc=None, file_name='output.obj'):
    mesh = trimesh.Trimesh(v, f, process=False)
    if vc is not None:
        mesh.visual.vertex_colors = vc
    mesh.export(file_name)


def render_mesh(img, mesh, face, cam_param, add_offset=False, return_mesh=False):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    ori_mesh = copy.deepcopy(mesh)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # add x-axis offset
    if add_offset:
        xoffset = 2 * (320 - princpt[0])  # 308.5481
        T = np.array([[1, 0, xoffset], [0, 1, 0]], dtype=np.float32)
        valid_mask = cv2.warpAffine(valid_mask.astype(np.uint8), T, (640, 480))[:, :, None]
        rgb = cv2.warpAffine(rgb, T, (640, 480))

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)

    if return_mesh:
        return img, ori_mesh
    else:
        return img


def render_mesh_seg(img, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    face_colors = np.zeros_like(face)
    face_colors[xmz_faces_idx] = xmz_color
    face_colors[wmz_faces_idx] = wmz_color
    face_colors[zz_faces_idx] = zz_color
    face_colors[sz_faces_idx] = sz_color
    face_colors[dmz_faces_idx] = dmz_color
    face_colors[palm_faces_idx] = palm_color

    mesh.visual.face_colors = face_colors
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene()
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img, valid_mask


def color_mesh_occ(mesh, face, occ_prob):
    occ_prob = occ_prob * 1.9 - 1
    # occ_prob = occ_prob * 1.95 - 1
    # use the coolwarm colormap that is built-in, and goes from blue to red
    cmap = mpl.cm.coolwarm
    # cmap = mpl.cm.RdYlBu_r
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    rgb_values = cmap(occ_prob)[:, :3] * 255
    rgb_values = rgb_values.astype(np.uint8)
    print(rgb_values)

    face_colors = np.zeros_like(face)
    face_colors[xmz_faces_idx] = rgb_values[4]
    face_colors[wmz_faces_idx] = rgb_values[3]
    face_colors[zz_faces_idx] = rgb_values[2]
    face_colors[sz_faces_idx] = rgb_values[1]
    face_colors[dmz_faces_idx] = rgb_values[0]
    # face_colors[palm_faces_idx] = np.array([211, 174, 144])
    face_colors[palm_faces_idx] = np.array([160, 106, 79])
    # mesh
    mesh = trimesh.Trimesh(mesh, face, process=False, face_colors=face_colors)

    # vertex_colors = np.zeros_like(face)
    # vertex_colors[xmz_verts_idx] = rgb_values[4]
    # vertex_colors[wmz_verts_idx] = rgb_values[3]
    # vertex_colors[zz_verts_idx] = rgb_values[2]
    # vertex_colors[sz_verts_idx] = rgb_values[1]
    # vertex_colors[dmz_verts_idx] = rgb_values[0]
    # vertex_colors[palm_verts_idx] = np.array([180, 153, 132])  # mesh
    # mesh = trimesh.Trimesh(mesh, face, process=False, vertex_colors=vertex_colors)

    # mesh.visual.face_colors = face_colors
    return mesh


def render_filter_mesh(ori_img, gen_img, bbox, pred_ori_mesh, pred_gen_mesh, gt_mesh, face, cam_param, do_flip):
    # mesh
    pred_ori_mesh = trimesh.Trimesh(pred_ori_mesh, face)
    pred_gen_mesh = trimesh.Trimesh(pred_gen_mesh, face)
    gt_mesh = trimesh.Trimesh(gt_mesh, face)

    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    pred_ori_mesh.apply_transform(rot)
    pred_gen_mesh.apply_transform(rot)
    gt_mesh.apply_transform(rot)

    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    pred_ori_mesh = pyrender.Mesh.from_trimesh(pred_ori_mesh, material=material, smooth=True)
    pred_gen_mesh = pyrender.Mesh.from_trimesh(pred_gen_mesh, material=material, smooth=True)
    gt_mesh = pyrender.Mesh.from_trimesh(gt_mesh, material=material, smooth=True)

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    camera = pyrender.IntrinsicsCamera(fx=cam_param[0], fy=cam_param[1], cx=cam_param[2], cy=cam_param[3])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)
    # renderer = pyrender.OffscreenRenderer(viewport_width=ori_img.shape[1], viewport_height=ori_img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render pred ori results
    pred_ori_mesh_node = scene.add(pred_ori_mesh, 'pred_ori_mesh')
    pred_ori_rgb, pred_ori_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    scene.remove_node(pred_ori_mesh_node)

    # render pred gen results
    pred_gen_mesh_node = scene.add(pred_gen_mesh, 'pred_gen_mesh')
    pred_gen_rgb, pred_gen_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    scene.remove_node(pred_gen_mesh_node)

    # render gt results
    scene.add(gt_mesh, 'gt_mesh')
    gt_rgb, gt_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    # pred ori image
    pred_ori_rgb = pred_ori_rgb[:, :, :3].astype(np.float32)
    pred_ori_rgb, _, _ = generate_patch_image(pred_ori_rgb, bbox, scale=1, rot=0, do_flip=do_flip, out_shape=(128, 128))
    valid_mask = (pred_ori_depth > 0)[:, :, None].astype(np.uint8)
    valid_mask, _, _ = generate_patch_image(valid_mask, bbox, scale=1, rot=0, do_flip=do_flip, out_shape=(128, 128), warp_flag=cv2.INTER_NEAREST)
    valid_mask = valid_mask[:, :, None]
    pred_ori_img = pred_ori_rgb * valid_mask + ori_img * (1 - valid_mask)

    # pred gen image
    pred_gen_rgb = pred_gen_rgb[:, :, :3].astype(np.float32)
    pred_gen_rgb, _, _ = generate_patch_image(pred_gen_rgb, bbox, scale=1, rot=0, do_flip=do_flip, out_shape=(128, 128))
    valid_mask = (pred_gen_depth > 0)[:, :, None].astype(np.uint8)
    valid_mask, _, _ = generate_patch_image(valid_mask, bbox, scale=1, rot=0, do_flip=do_flip, out_shape=(128, 128), warp_flag=cv2.INTER_NEAREST)
    valid_mask = valid_mask[:, :, None]
    pred_gen_img = pred_gen_rgb * valid_mask + gen_img * (1 - valid_mask)

    # gt image
    gt_rgb = gt_rgb[:, :, :3].astype(np.float32)
    gt_rgb, _, _ = generate_patch_image(gt_rgb, bbox, scale=1, rot=0, do_flip=do_flip, out_shape=(128, 128))
    valid_mask = (gt_depth > 0)[:, :, None].astype(np.uint8)
    valid_mask, _, _ = generate_patch_image(valid_mask, bbox, scale=1, rot=0, do_flip=do_flip, out_shape=(128, 128), warp_flag=cv2.INTER_NEAREST)
    valid_mask = valid_mask[:, :, None]
    gt_img = gt_rgb * valid_mask + ori_img * (1 - valid_mask)

    img_list = [ori_img, gt_img, ori_img, pred_ori_img, gen_img, pred_gen_img]
    for i in range(len(img_list)):
        img_list[i] = cv2.copyMakeBorder(np.copy(img_list[i]), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cat_img = np.concatenate(img_list, axis=1)[:, :, ::-1].astype(np.uint8)

    return cat_img


def render_badcase_mesh(ori_img, bbox, pred_mesh, gt_mesh, face, cam_param, do_flip):
    # mesh
    pred_mesh = trimesh.Trimesh(pred_mesh, face)
    gt_mesh = trimesh.Trimesh(gt_mesh, face)

    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    pred_mesh.apply_transform(rot)
    gt_mesh.apply_transform(rot)

    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    pred_mesh = pyrender.Mesh.from_trimesh(pred_mesh, material=material, smooth=True)
    gt_mesh = pyrender.Mesh.from_trimesh(gt_mesh, material=material, smooth=True)

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    camera = pyrender.IntrinsicsCamera(fx=cam_param[0], fy=cam_param[1], cx=cam_param[2], cy=cam_param[3])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)
    # renderer = pyrender.OffscreenRenderer(viewport_width=ori_img.shape[1], viewport_height=ori_img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render pred ori results
    pred_mesh_node = scene.add(pred_mesh, 'pred_mesh')
    pred_rgb, pred_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    scene.remove_node(pred_mesh_node)

    # render gt results
    scene.add(gt_mesh, 'gt_mesh')
    gt_rgb, gt_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    # pred ori image
    pred_rgb = pred_rgb[:, :, :3].astype(np.float32)
    pred_rgb, _, _ = generate_patch_image(pred_rgb, bbox, scale=1, rot=0, do_flip=do_flip, out_shape=(128, 128))
    valid_mask = (pred_depth > 0)[:, :, None].astype(np.uint8)
    valid_mask, _, _ = generate_patch_image(valid_mask, bbox, scale=1, rot=0, do_flip=do_flip, out_shape=(128, 128), warp_flag=cv2.INTER_NEAREST)
    valid_mask = valid_mask[:, :, None]
    pred_img = pred_rgb * valid_mask + ori_img * (1 - valid_mask)

    # gt image
    gt_rgb = gt_rgb[:, :, :3].astype(np.float32)
    gt_rgb, _, _ = generate_patch_image(gt_rgb, bbox, scale=1, rot=0, do_flip=do_flip, out_shape=(128, 128))
    valid_mask = (gt_depth > 0)[:, :, None].astype(np.uint8)
    valid_mask, _, _ = generate_patch_image(valid_mask, bbox, scale=1, rot=0, do_flip=do_flip, out_shape=(128, 128), warp_flag=cv2.INTER_NEAREST)
    valid_mask = valid_mask[:, :, None]
    gt_img = gt_rgb * valid_mask + ori_img * (1 - valid_mask)

    img_list = [ori_img, gt_img, ori_img, pred_img]
    for i in range(len(img_list)):
        img_list[i] = cv2.copyMakeBorder(np.copy(img_list[i]), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cat_img = np.concatenate(img_list, axis=1)[:, :, ::-1].astype(np.uint8)

    return cat_img
