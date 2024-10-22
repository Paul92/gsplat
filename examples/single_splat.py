import enum
import math

import torch

from gsplat.rendering import rasterization, rasterization_pgsr

from gsplat import rasterization
import tifffile
import tyro

class Rasterization(enum.Enum):
    GS3D = "gs3d"
    GS2D = "gs2d"
    RADE = "rade"
    RADE_INRIA = "rade_inria"
    PGSR = "pgsr"


def make_rotation_quat(angle):
    # Define the angle in radians
    angle = math.radians(angle)

    # Calculate the quaternion components
    qw = math.cos(angle / 2)
    qx = 0.0
    qy = math.sin(angle / 2)
    qz = 0.0

    # Create the quaternion tensor
    quaternion = torch.tensor([[qw, qx, qy, qz]], dtype=torch.float32)

    return quaternion


def run(rasterization_type: Rasterization):
    # Check if GPU is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    quats = make_rotation_quat(45)

    # Define the single splat in the center of the scene
    means = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32).to(device)  # [N, 3]
    quats = torch.tensor(quats, dtype=torch.float32).to(device)  # [N, 4]
    scales = torch.tensor([[1.0, 1.0, 0.1]], dtype=torch.float32).to(device)  # [N, 3]
    opacities = torch.tensor([1.0], dtype=torch.float32).to(device)  # [N]
    colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).to(device)  # [N, 3]

    # Define the camera parameters
    viewmats = torch.eye(4, dtype=torch.float32).unsqueeze(0).to(device)  # [C, 4, 4]
    viewmats[0, 2, 3] = 5.0  # Move the camera back along the z-axis

    Ks = torch.tensor([[[300.0, 0.0, 150.0], [0.0, 300.0, 100.0], [0.0, 0.0, 1.0]]], dtype=torch.float32).to(device)  # [C, 3, 3]

    # Image dimensions
    width = 300
    height = 200

    render_colors = None
    render_alphas = None
    render_depths = None
    render_normals = None

    if rasterization_type == Rasterization.GS3D:
        print("GS3D")
        # Call the rasterization function
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=0.0,
            eps2d=0.3,
            sh_degree=None,
            packed=False,
            tile_size=16,
            backgrounds=None,
            render_mode="RGB+ED",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="classic",
            channel_chunk=32,
            distributed=False,
            ortho=False,
            covars=None,
        )
        tifffile.imwrite("render_depths_test_3dgs.tiff", render_colors[0,:,:,-1].detach().cpu().numpy())


    elif rasterization_type == Rasterization.PGSR:
        render_colors, render_alphas, meta = rasterization_pgsr(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=0.0,
            far_plane=10000.0,
            radius_clip=0.0,
            eps2d=0.3,
            sh_degree=None,
            packed=False,
            tile_size=16,
            backgrounds=None,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="classic",
            channel_chunk=32,
            distributed=False,
            covars=None,
        )

        for key, value in meta.items():
            if torch.is_tensor(value):
                tifffile.imwrite(f"{key}.tiff", value.detach().cpu().numpy())
            else:
                print(f"{key} is None", key)


    # Save the rendered image
    if render_colors is not None:
        tifffile.imwrite("render_colors_test.tiff", render_colors[0].detach().cpu().numpy())
    if render_alphas is not None:
        tifffile.imwrite("render_alphas_test.tiff", render_alphas[0].detach().cpu().numpy())
    if render_depths is not None:
        tifffile.imwrite("render_depths_test.tiff", render_depths[0].detach().cpu().numpy())
    if render_normals is not None:
        tifffile.imwrite("render_normals_test.tiff", render_normals[0].detach().cpu().numpy())

   # render_normals.mean().backward()



if __name__ == "__main__":
    tyro.cli(run)