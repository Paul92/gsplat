import torch
from torch.autograd import gradcheck

from gsplat.cuda._wrapper import _RasterizeToPixelsPGSR

from gsplat.rendering import compute_all_map

input = torch.load("input.pt", weights_only=True)
params = torch.load("params.pt", weights_only=True)

[means,
          quats,
          scales,
          opacities,
          colors,
          viewmats,
          Ks,
          width,
          height] = params

[means2d,
         conics,
         colors,
         opacities,
         all_maps,
         width,
         height,
         Ks,
         tile_size,
         isect_offsets,
         flatten_ids,
         backgrounds,
         masks,
         packed,
         absgrad] = input

tile_height, tile_width = isect_offsets.shape[1:3]


all_map = compute_all_map(means, quats, scales, viewmats[0])

final_input = (
    means2d.contiguous(),
    conics.contiguous(),
    colors.contiguous(),
    opacities.contiguous(),
    backgrounds,
    masks,
    all_map,
    width,
    height,
    Ks,
    tile_size,
    isect_offsets.contiguous(),
    flatten_ids.contiguous(),
    absgrad,
)

fun = _RasterizeToPixelsPGSR.apply

gradcheck_result = gradcheck(fun, final_input, eps=1e-5)

if not gradcheck_result:
    raise Exception("Gradcheck failed")

print("Gradcheck passed")
