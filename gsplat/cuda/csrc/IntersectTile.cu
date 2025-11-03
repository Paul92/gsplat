#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

// for CUB_WRAPPER
#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "Intersect.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

// Computes the two intersection coordinates of the ellipse with a fixed line.
// For horizontal lines, we solve for x at a given y = coord.
// For vertical lines, we solve for y at a given x = coord.
// The ellipse is defined by a*(u - p_u)^2 + 2*b*(u - p_u)*(v - p_v) + c*(v - p_v)^2 = t,
// where (p_u, p_v) are the center coordinates (with u = x and v = y when we solve for x,
// and vice versa when solving for y). The function returns a vec2 containing the lower and upper
// intersection values.
template <typename scalar_t>
__device__ inline vec2 isect_ellipse(
    const scalar_t a,
    const scalar_t b,
    const scalar_t disc,
    const scalar_t thres,
    const scalar_t mean2d_x,
    const scalar_t mean2d_y,
    const scalar_t coord
) {
    // coord is the "line" coordinate along the axis you scan
    scalar_t arg = coord - mean2d_y;
    scalar_t sqrt_term = sqrt(arg * arg * disc + thres * a);

    return {
        mean2d_x + (-b * arg - sqrt_term) / a,
        mean2d_x + (-b * arg + sqrt_term) / a
    };
}

template <typename scalar_t>
__global__ void intersect_tiles_kernel_accutile(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t I,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t* __restrict__ image_ids,    // [nnz] optional
    const int64_t* __restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t* __restrict__ means2d,                   // [..., N, 2] or [nnz, 2]
    const scalar_t* __restrict__ opacities,                 // [..., N] or [nnz]
    const scalar_t* __restrict__ conics,                    // [..., N, 3] or [nnz, 3]
    const int32_t* __restrict__ radii,                      // [..., N, 2] or [nnz, 2]
    const scalar_t* __restrict__ depths,                    // [..., N] or [nnz]
    const int64_t* __restrict__ cum_tiles_per_gauss,        // [..., N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    const uint32_t image_n_bits,
    int32_t* __restrict__ tiles_per_gauss, // [..., N] or [nnz]
    int64_t* __restrict__ isect_ids,       // [n_isects]
    int32_t* __restrict__ flatten_ids      // [n_isects]
) {
    // Parallelize over I * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : I * N)) {
        return;
    }

    // Validate Gaussian.
    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    // Load center and conics.
    vec2 mean2d{means2d[2 * idx + 0], means2d[2 * idx + 1]};
    float a = conics[idx * 3];
    float b = conics[idx * 3 + 1];
    float c = conics[idx * 3 + 2];

    // Load current index, encoded depth ID and encoded camera ID
    int64_t iid; // image id
    if (packed) {
        // parallelize over nnz
        iid = image_ids[idx];
    } else {
        // parallelize over I * N
        iid = idx / N;
    }
    const int64_t iid_enc = iid << (32 + tile_n_bits);

    // tolerance for negative depth
    int32_t depth_i32 = *(int32_t *)&(depths[idx]);  // Bit-level reinterpret
    int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);  // Zero-extend to 64-bit

    // Calculate discriminant.
    const float disc = b * b - a * c;
    if (a <= 0 || c <= 0 || disc >= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    int64_t cur_idx = 0;
    if (!first_pass) {
        cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    }

    // Compute threshold
    const float opacity = max(1e-6f, opacities[idx]);
    const float thres = 2.f * log(opacity * 255.f);

    // Compute the extrema of the ellipse along each axis
    float x_extent = sqrt(-(b * b * thres) / (disc * a));
    x_extent = (b < 0) ? x_extent : -x_extent;
    float y_extent = sqrt(-(b * b * thres) / (disc * c));
    y_extent = (b < 0) ? y_extent : -y_extent;

    // Critical coordinates where the derivative vanishes
    vec2 arg_min, arg_max;
    arg_min.x = mean2d.y - y_extent;
    arg_min.y = mean2d.x - x_extent;
    arg_max.x = mean2d.y + y_extent;
    arg_max.y = mean2d.x + x_extent;

    // Compute the bounding box of the ellipse (for the Y-axis intersection invert a and c, and mean2d respectively)
    vec2 box_min, box_max;
    box_min.x = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, arg_min.x).x;
    box_min.y = isect_ellipse(c, b, disc, thres, mean2d.y, mean2d.x, arg_min.y).x;
    box_max.x = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, arg_max.x).y;
    box_max.y = isect_ellipse(c, b, disc, thres, mean2d.y, mean2d.x, arg_max.y).y;

    // Convert bounding box to tile grid coordinates
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(box_min.x / tile_size)), tile_width);
    tile_min.y = min(max(0, (uint32_t)floor(box_min.y / tile_size)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(box_max.x / tile_size)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(box_max.y / tile_size)), tile_height);

    // Process tiles. Choose iteration based on which dimension is smaller.
    int32_t tiles_count = 0;
    bool use_horizontal = (tile_max.y - tile_min.y < tile_max.x - tile_min.x);
    if (use_horizontal) {
        // Invert X and Y
        tile_min = {tile_min.y, tile_min.x};
        tile_max = {tile_max.y, tile_max.x};

        box_min = {box_min.y, box_min.x};
        box_max = {box_max.y, box_max.x};

        arg_min = {arg_min.y, arg_min.x};
        arg_max = {arg_max.y, arg_max.x};
    } else {
        // Invert a and c, and mean2d respectively
        a = c;
        mean2d = {mean2d.y, mean2d.x};
    }

    // Initialize last line and last intersect.
    vec2 last_intersect;
    last_intersect.x = box_max.y;
    last_intersect.y = box_min.y;
    float last_line = tile_min.x * tile_size;
    if (box_min.x <= last_line) {
        last_intersect = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, last_line);
    }

    for (int32_t i = tile_min.x; i < tile_max.x; ++i) {
        float isect_coord_min, isect_coord_max;
        // Compute current line intersection.
        const float cur_line = min(last_line + tile_size, box_max.x);
        vec2 cur_intersect;
        cur_intersect.x = box_max.y;
        cur_intersect.y = box_min.y;

        // If current line is not above the maximum ellipse extrema, compute current line ellipse intersection
        if (cur_line <= box_max.x) {
            cur_intersect = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, cur_line);
        } 

        // If the minimum extrema lies in current row / column, skip ellipse intersection.
        if (last_line <= arg_min.y && arg_min.y < cur_line) {
            isect_coord_min = box_min.y;
        } else {
            isect_coord_min = min(last_intersect.x, cur_intersect.x);
        }

        // If the maximum extrema lies in current row / column, skip ellipse intersection.
        if (last_line <= arg_max.y && arg_max.y < cur_line) {
            isect_coord_max = box_max.y;
        } else {
            isect_coord_max = max(last_intersect.y, cur_intersect.y);
        }

        // Map intersections to tile indices.
        uint32_t isect_tile_min = min(tile_max.y, max(tile_min.y, (uint32_t)floor(isect_coord_min / tile_size)));
        uint32_t isect_tile_max = min(tile_max.y, max(tile_min.y, (uint32_t)ceil(isect_coord_max / tile_size)));

        // Accumulate tiles count
        tiles_count += (isect_tile_max - isect_tile_min);

        // If second pass, fill isect_ids and flatten_ids with the intersected tile IDs
        if (!first_pass) {
            for (int32_t j = isect_tile_min; j < isect_tile_max; ++j) {
                int64_t tile_id = use_horizontal ? (i * tile_width + j) : (j * tile_width + i);
                isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[cur_idx] = static_cast<int32_t>(idx);
                cur_idx++;
            }
        }
        // Update last line and last intersect.
        last_intersect = cur_intersect;
        last_line = cur_line;
    }

    // First pass only writes out tiles_per_gauss.
    if (first_pass) {
        tiles_per_gauss[idx] = static_cast<int32_t>(tiles_count);
    }
}

template <typename scalar_t>
__global__ void intersect_tiles_kernel_snugbox(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t I,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t* __restrict__ image_ids,    // [nnz] optional
    const int64_t* __restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t* __restrict__ means2d,                   // [..., N, 2] or [nnz, 2]
    const scalar_t* __restrict__ opacities,                 // [..., N] or [nnz]
    const scalar_t* __restrict__ conics,                    // [..., N, 3] or [nnz, 3]
    const int32_t* __restrict__ radii,                      // [..., N, 2] or [nnz, 2]
    const scalar_t* __restrict__ depths,                    // [..., N] or [nnz]
    const int64_t* __restrict__ cum_tiles_per_gauss,        // [..., N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    const uint32_t image_n_bits,
    int32_t* __restrict__ tiles_per_gauss, // [..., N] or [nnz]
    int64_t* __restrict__ isect_ids,       // [n_isects]
    int32_t* __restrict__ flatten_ids      // [n_isects]
) {
    // Parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : I * N)) {
        return;
    }

    // Validate Gaussian.
    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    // Load center and conics.
    vec2 mean2d{means2d[2 * idx + 0], means2d[2 * idx + 1]};
    float a = conics[idx * 3];
    float b = conics[idx * 3 + 1];
    float c = conics[idx * 3 + 2];

    // Calculate discriminant.
    const float disc = b * b - a * c;
    if (a <= 0 || c <= 0 || disc >= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }
    
    // Compute threshold.
    const float opacity = max(1e-6f, opacities[idx]);
    const float thres   = 2.f * log(opacity * 255.f);

    // Compute the extrema of the ellipse along each axis.
    float x_extent = sqrt(-(b * b * thres) / (disc * a));
    x_extent = (b < 0) ? x_extent : -x_extent;
    float y_extent = sqrt(-(b * b * thres) / (disc * c));
    y_extent = (b < 0) ? y_extent : -y_extent;

    // Critical coordinates where the derivative vanishes.
    vec2 arg_min, arg_max;
    arg_min.x = mean2d.y - y_extent;
    arg_min.y = mean2d.x - x_extent;
    arg_max.x = mean2d.y + y_extent;
    arg_max.y = mean2d.x + x_extent;

    // Compute the bounding box of the ellipse (for the Y-axis intersection invert a and c, and mean2d respectively).
    vec2 box_min, box_max;
    box_min.x = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, arg_min.x).x;
    box_min.y = isect_ellipse(c, b, disc, thres, mean2d.y, mean2d.x, arg_min.y).x;
    box_max.x = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, arg_max.x).y;
    box_max.y = isect_ellipse(c, b, disc, thres, mean2d.y, mean2d.x, arg_max.y).y;

    // Convert bounding box to tile grid coordinates.
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(box_min.x / tile_size)), tile_width);
    tile_min.y = min(max(0, (uint32_t)floor(box_min.y / tile_size)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(box_max.x / tile_size)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(box_max.y / tile_size)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }

    int64_t iid; // image id
    if (packed) {
        // parallelize over nnz
        iid = image_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over I * N
        iid = idx / N;
        // gid = idx % N;
    }
    const int64_t iid_enc = iid << (32 + tile_n_bits);

    // tolerance for negative depth
    int32_t depth_i32 = *(int32_t *)&(depths[idx]);
    int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);

    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // image id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [C * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}


// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation

template <typename scalar_t>
__global__ void intersect_tile_kernel(
    // if the data is [...,  N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over I * N, only used if packed is False
    const uint32_t I,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ image_ids,    // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t *__restrict__ means2d,            // [..., N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [..., N, 2] or [nnz, 2]
    const scalar_t *__restrict__ depths,             // [..., N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [..., N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    const uint32_t image_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [..., N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // parallelize over I * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : I * N)) {
        return;
    }

    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2 mean2d = glm::make_vec2(means2d + 2 * idx);

    float tile_radius_x = radius_x / static_cast<float>(tile_size);
    float tile_radius_y = radius_y / static_cast<float>(tile_size);
    float tile_x = mean2d.x / static_cast<float>(tile_size);
    float tile_y = mean2d.y / static_cast<float>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius_x)), tile_width);
    tile_min.y =
        min(max(0, (uint32_t)floor(tile_y - tile_radius_y)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }

    int64_t iid; // image id
    if (packed) {
        // parallelize over nnz
        iid = image_ids[idx];
    } else {
        // parallelize over I * N
        iid = idx / N;
    }
    const int64_t iid_enc = iid << (32 + tile_n_bits);

    // tolerance for negative depth
    int32_t depth_i32 = *(int32_t *)&(depths[idx]);  // Bit-level reinterpret
    int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);  // Zero-extend to 64-bit
    // int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // image id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [I * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}

enum class IntersectKind : uint8_t {
    AxisAligned = 0,
    SnugBox = 1,
    AccuTile = 2
 };

void launch_intersect_tile_kernel(
    // inputs
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor opacities,                  // [..., N] or [nnz]
    const at::Tensor conics,                     // [..., N, 3] or [nnz, 3]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids,     // [n_isects]
    // options
    IntersectKind intersect_kind = IntersectKind::AxisAligned
) {
    bool packed = means2d.dim() == 2;

    uint32_t N, nnz;
    int64_t n_elements;
    if (packed) {
        nnz = means2d.size(0); // total number of gaussians
        n_elements = nnz;
    } else {
        N = means2d.size(-2); // number of gaussians per image
        n_elements = I * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the image id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t image_n_bits = std::bit_width(I);
    uint32_t image_n_bits = (uint32_t)floor(log2(I)) + 1;
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    // the first 32 bits are used for the image id and tile id altogether, so
    // check if we have enough bits for them.
    assert(image_n_bits + tile_n_bits <= 32);

    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    switch(intersect_kind) {
        case IntersectKind::AxisAligned:
            AT_DISPATCH_FLOATING_TYPES(
                means2d.scalar_type(),
                "intersect_tile_kernel",
                [&]() {
                    intersect_tile_kernel<scalar_t>
                        <<<grid,
                        threads,
                        shmem_size,
                        at::cuda::getCurrentCUDAStream()>>>(
                            packed,
                            I,
                            N,
                            nnz,
                            image_ids.has_value()
                                ? image_ids.value().data_ptr<int64_t>()
                                : nullptr,
                            gaussian_ids.has_value()
                                ? gaussian_ids.value().data_ptr<int64_t>()
                                : nullptr,
                            means2d.data_ptr<scalar_t>(),
                            radii.data_ptr<int32_t>(),
                            depths.data_ptr<scalar_t>(),
                            cum_tiles_per_gauss.has_value()
                                ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                                : nullptr,
                            tile_size,
                            tile_width,
                            tile_height,
                            tile_n_bits,
                            image_n_bits,
                            tiles_per_gauss.has_value()
                                ? tiles_per_gauss.value().data_ptr<int32_t>()
                                : nullptr,
                            isect_ids.has_value()
                                ? isect_ids.value().data_ptr<int64_t>()
                                : nullptr,
                            flatten_ids.has_value()
                                ? flatten_ids.value().data_ptr<int32_t>()
                                : nullptr
                        );
                }
            );
            break;
        case IntersectKind::SnugBox:
                    AT_DISPATCH_FLOATING_TYPES(
                means2d.scalar_type(),
                "intersect_tiles_kernel_snugbox",
                [&]() {
                    intersect_tiles_kernel_snugbox<scalar_t>
                        <<<grid,
                        threads,
                        shmem_size,
                        at::cuda::getCurrentCUDAStream()>>>(
                            packed,
                            I,
                            N,
                            nnz,
                            image_ids.has_value()
                                ? image_ids.value().data_ptr<int64_t>()
                                : nullptr,
                            gaussian_ids.has_value()
                                ? gaussian_ids.value().data_ptr<int64_t>()
                                : nullptr,
                            means2d.data_ptr<scalar_t>(),
                            opacities.data_ptr<scalar_t>(),
                            conics.data_ptr<scalar_t>(),
                            radii.data_ptr<int32_t>(),
                            depths.data_ptr<scalar_t>(),
                            cum_tiles_per_gauss.has_value()
                                ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                                : nullptr,
                            tile_size,
                            tile_width,
                            tile_height,
                            tile_n_bits,
                            image_n_bits,
                            tiles_per_gauss.has_value()
                                ? tiles_per_gauss.value().data_ptr<int32_t>()
                                : nullptr,
                            isect_ids.has_value()
                                ? isect_ids.value().data_ptr<int64_t>()
                                : nullptr,
                            flatten_ids.has_value()
                                ? flatten_ids.value().data_ptr<int32_t>()
                                : nullptr
                        );
                }
            );
            break;
        case IntersectKind::AccuTile:
                    AT_DISPATCH_FLOATING_TYPES(
                means2d.scalar_type(),
                "intersect_tiles_kernel_accutile",
                [&]() {
                    intersect_tiles_kernel_accutile<scalar_t>
                        <<<grid,
                        threads,
                        shmem_size,
                        at::cuda::getCurrentCUDAStream()>>>(
                            packed,
                            I,
                            N,
                            nnz,
                            image_ids.has_value()
                                ? image_ids.value().data_ptr<int64_t>()
                                : nullptr,
                            gaussian_ids.has_value()
                                ? gaussian_ids.value().data_ptr<int64_t>()
                                : nullptr,
                            means2d.data_ptr<scalar_t>(),
                            opacities.data_ptr<scalar_t>(),
                            conics.data_ptr<scalar_t>(),
                            radii.data_ptr<int32_t>(),
                            depths.data_ptr<scalar_t>(),
                            cum_tiles_per_gauss.has_value()
                                ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                                : nullptr,
                            tile_size,
                            tile_width,
                            tile_height,
                            tile_n_bits,
                            image_n_bits,
                            tiles_per_gauss.has_value()
                                ? tiles_per_gauss.value().data_ptr<int32_t>()
                                : nullptr,
                            isect_ids.has_value()
                                ? isect_ids.value().data_ptr<int64_t>()
                                : nullptr,
                            flatten_ids.has_value()
                                ? flatten_ids.value().data_ptr<int32_t>()
                                : nullptr
                        );
                }
            );
            break;

    }
}

__global__ void intersect_offset_kernel(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t I,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [I, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    uint32_t image_n_bits = (uint32_t)floor(log2f(float(I))) + 1;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t iid_curr = isect_id_curr >> (tile_n_bits);
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = iid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < I * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (bid, iid,
        // tile_id) tuple changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t iid_prev = isect_id_prev >> (tile_n_bits);
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = iid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

void launch_intersect_offset_kernel(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [I, tile_height, tile_width]
) {
    int64_t n_elements = isect_ids.size(0); // total number of intersections
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        offsets.fill_(0);
        return;
    }

    uint32_t n_tiles = tile_width * tile_height;
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    intersect_offset_kernel<<<
        grid,
        threads,
        shmem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        n_elements,
        isect_ids.data_ptr<int64_t>(),
        I,
        n_tiles,
        tile_n_bits,
        offsets.data_ptr<int32_t>()
    );
}

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
// DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
void radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
) {
    if (n_isects <= 0) {
        return;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int64_t> d_keys(
        isect_ids.data_ptr<int64_t>(), isect_ids_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int32_t> d_values(
        flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>()
    );
    CUB_WRAPPER(
        cub::DeviceRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        0,
        32 + tile_n_bits + image_n_bits,
        at::cuda::getCurrentCUDAStream()
    );
    switch (d_keys.selector) {
    case 0: // sorted items are stored in isect_ids
        isect_ids_sorted.set_(isect_ids);
        break;
    case 1: // sorted items are stored in isect_ids_sorted
        break;
    }
    switch (d_values.selector) {
    case 0: // sorted items are stored in flatten_ids
        flatten_ids_sorted.set_(flatten_ids);
        break;
    case 1: // sorted items are stored in flatten_ids_sorted
        break;
    }
}

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedRadixSort.html
// DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
void segmented_radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t n_segments,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    const at::Tensor offsets,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
) {
    if (n_isects <= 0) {
        return;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int64_t> d_keys(
        isect_ids.data_ptr<int64_t>(), isect_ids_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int32_t> d_values(
        flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>()
    );
    // image dimensions are contiguous in the isect_ids, 
    // so we can use DeviceSegmentedRadixSort to only sort the lower 
    // (tile_n_bits + 32) bits
    CUB_WRAPPER(
        cub::DeviceSegmentedRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        n_segments, // number of segments
        offsets.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>() + 1,
        0,
        32 + tile_n_bits,
        at::cuda::getCurrentCUDAStream()
    );
    switch (d_keys.selector) {
    case 0: // sorted items are stored in isect_ids
        isect_ids_sorted.set_(isect_ids);
        break;
    case 1: // sorted items are stored in isect_ids_sorted
        break;
    }
    switch (d_values.selector) {
    case 0: // sorted items are stored in flatten_ids
        flatten_ids_sorted.set_(flatten_ids);
        break;
    case 1: // sorted items are stored in flatten_ids_sorted
        break;
    }
}

} // namespace gsplat
