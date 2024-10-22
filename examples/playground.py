import math
import torch

from gsplat.rendering import compute_all_map
from gsplat.utils import normalized_quat_to_rotmat
from torch import Tensor
from gsplat.cuda._wrapper import _make_lazy_cuda_func,  isect_tiles, \
    isect_offset_encode, _FullyFusedProjection
from torch.autograd import Function
from torch.autograd import gradcheck
from typing import Tuple


torch.use_deterministic_algorithms(True)
import sys

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def get_rotation(quats):
    return torch.nn.functional.normalize(quats)

def get_rotation_matrix(quats):
    return normalized_quat_to_rotmat(get_rotation(quats))

def get_scaling(scales):
    return torch.exp(scales)

def get_smallest_axis(quats, scales, return_idx=False):
    rotation_matrices = get_rotation_matrix(quats)
    smallest_axis_idx = get_scaling(scales).min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
    smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
    if return_idx:
        return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
    return smallest_axis.squeeze(dim=2)

def get_normal(view_cam, quats, scales, means3d):
    normal_global = get_smallest_axis(quats, scales)
    gaussian_to_cam_global = view_cam.camera_center - means3d
    neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
    normal_global[neg_mask] = -normal_global[neg_mask]
    return normal_global


width = 765
height = 572

zfar = 100.0
znear = 0.01

start = 0
end = None

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the single splat in the center of the scene
means3D = torch.load("means.pt", weights_only=True).to(device=device)
quats = torch.load("quats.pt", weights_only=True).to(device=device)
scales = torch.load("scales.pt", weights_only=True).to(device=device)
opacities = torch.load("opacities.pt", weights_only=True).to(device=device).contiguous()
colors = torch.load("colors.pt", weights_only=True).to(device=device)
viewmats = torch.load("viewmats.pt", weights_only=True).to(device=device).contiguous()
Ks = torch.load("Ks.pt", weights_only=True).to(device=device)

# if start is not None and end is not None:
#     means = means3D[start:end]
#     quats = quats[start:end]
#     scales = scales[start:end]
#     opacities = opacities[start:end]
#     colors = colors[start:end]
#     viewmats = viewmats[start:end]
#     Ks = Ks[start:end]
#
# means3D = means3D.double()
# quats = quats.double()
# scales = scales.double()
# opacities = opacities.double()
# colors = colors.double()
# viewmats = viewmats.double()
# Ks = Ks.double()


from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer

screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
screenspace_points_abs = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
try:
    screenspace_points.retain_grad()
    screenspace_points_abs.retain_grad()
except:
    pass


FoVx = focal2fov(Ks[0, 0], width)
FoVy = focal2fov(Ks[0, 1], height)

# Set up rasterization configuration
tanfovx = math.tan(FoVx * 0.5)
tanfovy = math.tan(FoVy * 0.5)

means2D = screenspace_points
means2D_abs = screenspace_points_abs

background_color = torch.tensor([0.0, 0.0, 0.0], device=device)

projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx,
                                        fovY=FoVy).transpose(0, 1).cuda()
full_proj_transform = (viewmats.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
camera_center = viewmats.unsqueeze(0).inverse()[3, :3]

raster_settings = PlaneGaussianRasterizationSettings(
    image_height=int(height),
    image_width=int(width),
    tanfovx=FoVx,
    tanfovy=FoVy,
    bg=background_color,
    scale_modifier=1,
    viewmatrix=viewmats,
    projmatrix=full_proj_transform,
    sh_degree=3,
    campos=camera_center,
    prefiltered=False,
    render_geo=True,
    debug=True
)

rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

rendered_image, radii, out_observe, _, _ = rasterizer(
    means3D=means3D,
    means2D=means2D,
    means2D_abs=means2D_abs,
    colors_precopm=colors,
    opacities=opacities,
    scales=scales,
    rotations=quats,
    cov3D_precomp=None)

return_dict = {"render": rendered_image,
               "viewspace_points": screenspace_points,
               "viewspace_points_abs": screenspace_points_abs,
               "visibility_filter": radii > 0,
               "radii": radii,
               "out_observe": out_observe}


all_maps = compute_all_map(means3D, quats, scales, viewmats[0])

rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
    means3D=means3D,
    means2D=means2D,
    means2D_abs=means2D_abs,
    shs=shs,
    colors_precomp=colors_precomp,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    all_map=input_all_map,
    cov3D_precomp=cov3D_precomp)

rendered_normal = out_all_map[0:3]
rendered_alpha = out_all_map[3:4, ]
rendered_distance = out_all_map[4:5, ]

return_dict = {"render": rendered_image,
               "viewspace_points": screenspace_points,
               "viewspace_points_abs": screenspace_points_abs,
               "visibility_filter": radii > 0,
               "radii": radii,
               "out_observe": out_observe,
               "rendered_normal": rendered_normal,
               "plane_depth": plane_depth,
               "rendered_distance": rendered_distance
               }



# fully_fused_projection_rade = _FullyFusedProjectionRade.apply
#
# input_rade = (means, quats, scales, opacities, viewmats, Ks, width, height, 0.3, 0.01, 1e10, 0.0, False, False)
#
# gradcheck_result = gradcheck(fully_fused_projection_rade, input_rade, eps=1e-5)
#
# if not gradcheck_result:
#     raise Exception("Gradcheck failed")
#
# print("Gradcheck passed")
# sys.exit(0)





# Forward pass
radii, means2d, conics, depths, camera_planes, normals, ray_planes, ts = fully_fused_projection_rade(means, quats, scales, opacities, viewmats,
                                                                                                Ks, width, height,
                                                                                                0.3,
                                                                                                0.01,
                                                                                                1e10,
                                                                                                0.0,
                                                                                                False,
                                                                                                False,
                                                                                                )

means2d = means2d.double()
conics = conics.double()
depths = depths.double()
camera_planes = camera_planes.double()
ray_planes = ray_planes.double()
normals = normals.double()
ts = ts.double()



tile_size=16

C = viewmats.shape[0]

# Identify intersecting tiles
tile_width = math.ceil(width / float(tile_size))
tile_height = math.ceil(height / float(tile_size))
tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
    means2d,
    radii.int(),
    depths,
    tile_size,
    tile_width,
    tile_height,
    packed=False,
    n_cameras=C,
    camera_ids=None,
    gaussian_ids=None,
)
# print("rank", world_rank, "Before isect_offset_encode")
isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)


rasterize_to_pixels = _RasterizeToPixelsRADE.apply

input = (means2d,
         conics,
         colors,
         opacities,
         camera_planes,
         ray_planes,
         normals,
         ts,
         Ks[0].double(),
         None,
         None,
         width,
         height,
         tile_size,
         isect_offsets,
         flatten_ids,
         False)

gradcheck_result = gradcheck(rasterize_to_pixels, input, eps=1e-5)

if not gradcheck_result:
    raise Exception("Gradcheck failed")

print("Gradcheck passed")
sys.exit(0)























double_means = means.double().requires_grad_()
double_quats = quats.double()#.requires_grad_()
double_scales = scales.double()#.requires_grad_()
double_viewmats = viewmats.double()#.requires_grad_()
double_opacities = torch.ones(means.shape[0], dtype=torch.float64).to(device=device)#.requires_grad_()
double_Ks = Ks.double()#.requires_grad_()































test = _FullyFusedProjectionRade.apply
input = (double_means, double_quats, double_scales, double_viewmats, double_opacities, double_Ks)

outputs = test(*input)
#print(outputs)


#gradcheck_result = gradcheck(test, input, eps=1e-6, atol=1e-4, nondet_tol=1e-5)
#print("GRADCHECK 2", gradcheck_result)
#
#if not gradcheck_result:
#    raise Exception("Gradcheck failed")


n_start = 89
n_end = 90

#n_start = 0
#n_end = 50

means = torch.load("means.pt").to(device=device)[n_start:n_end]
quats = torch.load("quats.pt").to(device=device)[n_start:n_end]
scales = torch.load("scales.pt").to(device=device)[n_start:n_end]
viewmats = torch.load("viewmats.pt").to(device=device)
Ks = torch.load("Ks.pt").to(device=device)


width=764
height=572
eps2d=0.3
near_plane=0.01
far_plane=10000000000.0
radius_clip=0.0
calc_compensations=False
ortho=False

double_means = means.double().requires_grad_()
double_quats = quats.double()#.requires_grad_()
double_scales = scales.double()#.requires_grad_()
double_viewmats = viewmats.double()#.requires_grad_()
double_opacities = torch.ones(means.shape[0], dtype=torch.float64).to(device=device)#.requires_grad_()
double_Ks = Ks.double()#.requires_grad_()



input_rade = (double_means, double_quats, double_scales, double_viewmats, double_opacities, double_Ks, width, height)
input_3dgs = (double_means, None, double_quats, double_scales, double_viewmats, double_Ks, width, height)

outputs = test(*input)

gradcheck_result = gradcheck(test, input, eps=1e-6, atol=1e-4)
print("GRADCHECK", gradcheck_result)

if not gradcheck_result:
    raise Exception("Gradcheck failed")
