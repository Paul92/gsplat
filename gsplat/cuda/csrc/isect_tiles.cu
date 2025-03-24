#include "bindings.h"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Gaussian Tile Intersection
 ****************************************************************************/

// Computes the two intersection coordinates of the ellipse with a fixed line.
// For horizontal lines, we solve for x at a given y = coord.
// For vertical lines, we solve for y at a given x = coord.
// The ellipse is defined by a*(u - p_u)^2 + 2*b*(u - p_u)*(v - p_v) + c*(v - p_v)^2 = t,
// where (p_u, p_v) are the center coordinates (with u = x and v = y when we solve for x,
// and vice versa when solving for y). The function returns a vec2 containing the lower and upper
// intersection values.
template <typename T>
__device__ inline vec2<T> isect_ellipse(
    const T a,
    const T b, 
    const T disc, 
    const T thres, 
    const T mean2d_x,
    const T mean2d_y,
    const T coord
) {
    T arg = coord - mean2d_y;
    T sqrt_term = sqrt(arg * arg * disc + thres * a);

    return {
        mean2d_x + (-b * arg - sqrt_term) / a,
        mean2d_x + (-b * arg + sqrt_term) / a
    };
}

template <typename T>
__device__ inline void isect_tiles_accutile(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t* __restrict__ camera_ids,   // [nnz] optional
    const int64_t* __restrict__ gaussian_ids, // [nnz] optional
    // data
    const T* __restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const T* __restrict__ opacities,                 // [C, N] or [nnz]
    const T* __restrict__ conics,                    // [C, N, 3] or [nnz, 3]
    const int32_t* __restrict__ radii,               // [C, N] or [nnz]
    const T* __restrict__ depths,                    // [C, N] or [nnz]
    const int64_t* __restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t* __restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t* __restrict__ isect_ids,       // [n_isects]
    int32_t* __restrict__ flatten_ids      // [n_isects]
) {
    // For now we'll upcast float16 and bfloat16 to float32.
    using OpT = typename OpType<T>::type;

    // Parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    // Validate Gaussian.
    const OpT radius = radii[idx];
    if (radius <= 0) {
        if (first_pass)
            tiles_per_gauss[idx] = 0;
        return;
    }

    // Load center and conics.
    vec2<OpT> mean2d = glm::make_vec2(means2d + 2 * idx);
    OpT a = conics[idx * 3];
    OpT b = conics[idx * 3 + 1];
    OpT c = conics[idx * 3 + 2];

    // Load current index, encoded depth ID and encoded camera ID
    int64_t cid, cid_enc, depth_id_enc, cur_idx;
    if (!first_pass) {
        if (packed) {
            // Parallelize over nnz.
            cid = camera_ids[idx];
        } else {
            // Parallelize over C * N.
            cid = idx / N;
        }
        cid_enc = cid << (32 + tile_n_bits);
        depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
        cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    }

    // Calculate discriminant.
    OpT disc = b * b - a * c;
    if (a <= 0 || c <= 0 || disc >= 0) {
        return;
    }
    
    // Compute threshold.
    OpT thres = 2.0 * log(opacities[idx] * 255.0);

    // Compute the extrema of the ellipse along each axis.
    OpT x_extent = sqrt(-(b * b * thres) / (disc * a));
    x_extent = (b < 0) ? x_extent : -x_extent;
    OpT y_extent = sqrt(-(b * b * thres) / (disc * c));
    y_extent = (b < 0) ? y_extent : -y_extent;

    // Critical coordinates where the derivative vanishes.
    vec2<OpT> arg_min, arg_max;
    arg_min.x = mean2d.y - y_extent;
    arg_min.y = mean2d.x - x_extent;
    arg_max.x = mean2d.y + y_extent;
    arg_max.y = mean2d.x + x_extent;

    // Compute the bounding box of the ellipse (for the Y-axis intersection invert a and c, and mean2d respectively).
    vec2<OpT> box_min, box_max;
    box_min.x = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, arg_min.x).x;
    box_min.y = isect_ellipse(c, b, disc, thres, mean2d.y, mean2d.x, arg_min.y).x;
    box_max.x = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, arg_max.x).y;
    box_max.y = isect_ellipse(c, b, disc, thres, mean2d.y, mean2d.x, arg_max.y).y;

    // Convert bounding box to tile grid coordinates.
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(box_min.x / static_cast<OpT>(tile_size))), tile_width);
    tile_min.y = min(max(0, (uint32_t)floor(box_min.y / static_cast<OpT>(tile_size))), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(box_max.x / static_cast<OpT>(tile_size))), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(box_max.y / static_cast<OpT>(tile_size))), tile_height);

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
    vec2<OpT> last_intersect;
    last_intersect.x = box_max.y;
    last_intersect.y = box_min.y;
    OpT last_line = tile_min.x * tile_size;
    if (box_min.x <= last_line) {
        last_intersect = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, last_line);
    }

    for (int32_t i = tile_min.x; i < tile_max.x; ++i) {
        OpT isect_coord_min, isect_coord_max;
        // Compute current line intersection.
        OpT cur_line = min(last_line + tile_size, box_max.x);
        vec2<OpT> cur_intersect;
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
                
                isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
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

template <typename T>
__device__ inline void isect_tiles_snugbox(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const T *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const T* __restrict__ opacities,                 // [C, N] or [nnz]
    const T* __restrict__ conics,                    // [C, N, 3] or [nnz, 3]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const T *__restrict__ depths,                    // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // For now we'll upcast float16 and bfloat16 to float32.
    using OpT = typename OpType<T>::type;

    // Parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    // Validate Gaussian.
    const OpT radius = radii[idx];
    if (radius <= 0) {
        if (first_pass)
            tiles_per_gauss[idx] = 0;
        return;
    }

    // Load center and conics.
    vec2<OpT> mean2d = glm::make_vec2(means2d + 2 * idx);
    OpT a = conics[idx * 3];
    OpT b = conics[idx * 3 + 1];
    OpT c = conics[idx * 3 + 2];

    // Calculate discriminant.
    OpT disc = b * b - a * c;
    if (a <= 0 || c <= 0 || disc >= 0) {
        return;
    }
    
    // Compute threshold.
    OpT thres = 2.0 * log(opacities[idx] * 255.0);

    // Compute the extrema of the ellipse along each axis.
    OpT x_extent = sqrt(-(b * b * thres) / (disc * a));
    x_extent = (b < 0) ? x_extent : -x_extent;
    OpT y_extent = sqrt(-(b * b * thres) / (disc * c));
    y_extent = (b < 0) ? y_extent : -y_extent;

    // Critical coordinates where the derivative vanishes.
    vec2<OpT> arg_min, arg_max;
    arg_min.x = mean2d.y - y_extent;
    arg_min.y = mean2d.x - x_extent;
    arg_max.x = mean2d.y + y_extent;
    arg_max.y = mean2d.x + x_extent;

    // Compute the bounding box of the ellipse (for the Y-axis intersection invert a and c, and mean2d respectively).
    vec2<OpT> box_min, box_max;
    box_min.x = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, arg_min.x).x;
    box_min.y = isect_ellipse(c, b, disc, thres, mean2d.y, mean2d.x, arg_min.y).x;
    box_max.x = isect_ellipse(a, b, disc, thres, mean2d.x, mean2d.y, arg_max.x).y;
    box_max.y = isect_ellipse(c, b, disc, thres, mean2d.y, mean2d.x, arg_max.y).y;

    // Convert bounding box to tile grid coordinates.
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(box_min.x / static_cast<OpT>(tile_size))), tile_width);
    tile_min.y = min(max(0, (uint32_t)floor(box_min.y / static_cast<OpT>(tile_size))), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(box_max.x / static_cast<OpT>(tile_size))), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(box_max.y / static_cast<OpT>(tile_size))), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [C * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}

template <typename T>
__device__ inline void isect_tiles_original(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const T *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const T *__restrict__ depths,                    // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    const OpT radius = radii[idx];
    if (radius <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2<OpT> mean2d = glm::make_vec2(means2d + 2 * idx);

    OpT tile_radius = radius / static_cast<OpT>(tile_size);
    OpT tile_x = mean2d.x / static_cast<OpT>(tile_size);
    OpT tile_y = mean2d.y / static_cast<OpT>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius)), tile_width);
    tile_min.y =
        min(max(0, (uint32_t)floor(tile_y - tile_radius)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [C * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}

template <typename T>
__global__ void isect_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const T *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const T* __restrict__ opacities,                 // [C, N] or [nnz]
    const T* __restrict__ conics,                    // [C, N, 3] or [nnz, 3]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const T *__restrict__ depths,                    // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids,     // [n_isects]
    int32_t isect_method
) {
    switch (isect_method)
    {
    case 1:
        isect_tiles_snugbox(
            packed,
            C,
            N,
            nnz,
            camera_ids,
            gaussian_ids,
            means2d,
            opacities,
            conics,
            radii,
            depths,
            cum_tiles_per_gauss,
            tile_size,
            tile_width,
            tile_height,
            tile_n_bits,
            tiles_per_gauss,
            isect_ids,
            flatten_ids
        );
        break;

    case 2:
        isect_tiles_accutile(
            packed,
            C,
            N,
            nnz,
            camera_ids,
            gaussian_ids,
            means2d,
            opacities,
            conics,
            radii,
            depths,
            cum_tiles_per_gauss,
            tile_size,
            tile_width,
            tile_height,
            tile_n_bits,
            tiles_per_gauss,
            isect_ids,
            flatten_ids
        );
        break;
    
    default:
        isect_tiles_original(
            packed,
            C,
            N,
            nnz,
            camera_ids,
            gaussian_ids,
            means2d,
            radii,
            depths,
            cum_tiles_per_gauss,
            tile_size,
            tile_width,
            tile_height,
            tile_n_bits,
            tiles_per_gauss,
            isect_ids,
            flatten_ids
        );
        break;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_tensor(
    const torch::Tensor &means2d,                    // [C, N, 2] or [nnz, 2]
    const torch::Tensor &opacities,                  // [C, N] or [nnz]
    const torch::Tensor &conics,                     // [C, N, 3] or [nnz, 3]
    const torch::Tensor &radii,                      // [C, N] or [nnz]
    const torch::Tensor &depths,                     // [C, N] or [nnz]
    const at::optional<torch::Tensor> &camera_ids,   // [nnz]
    const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool double_buffer,
    const uint32_t isect_method                      // 0: original, 1: snugbox, 2: accutile
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(depths);
    if (camera_ids.has_value()) {
        GSPLAT_CHECK_INPUT(camera_ids.value());
    }
    if (gaussian_ids.has_value()) {
        GSPLAT_CHECK_INPUT(gaussian_ids.value());
    }
    bool packed = means2d.dim() == 2;
    uint32_t N = 0, nnz = 0, total_elems = 0;
    int64_t *camera_ids_ptr = nullptr;
    int64_t *gaussian_ids_ptr = nullptr;
    if (packed) {
        nnz = means2d.size(0);
        total_elems = nnz;
        TORCH_CHECK(
            camera_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, camera_ids and gaussian_ids must be provided."
        );
        camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
        gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
    } else {
        N = means2d.size(1); // number of gaussians
        total_elems = C * N;
    }
    uint32_t n_tiles = tile_width * tile_height;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);
    // first pass: compute number of tiles per gaussian
    torch::Tensor tiles_per_gauss =
        torch::empty_like(depths, depths.options().dtype(torch::kInt32));
    int64_t n_isects;
    torch::Tensor cum_tiles_per_gauss;
    if (total_elems) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_total_elems",
            [&]() {
                isect_tiles<<<
                    (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    opacities.data_ptr<scalar_t>(),
                    conics.data_ptr<scalar_t>(),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    tiles_per_gauss.data_ptr<int32_t>(),
                    nullptr,
                    nullptr,
                    isect_method
                );
            }
        );
        cum_tiles_per_gauss = torch::cumsum(tiles_per_gauss.view({-1}), 0);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }
    // second pass: compute isect_ids and flatten_ids as a packed tensor
    torch::Tensor isect_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt32));
    if (n_isects) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_n_isects",
            [&]() {
                isect_tiles<<<
                    (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    opacities.data_ptr<scalar_t>(),
                    conics.data_ptr<scalar_t>(),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    cum_tiles_per_gauss.data_ptr<int64_t>(),
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    nullptr,
                    isect_ids.data_ptr<int64_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    isect_method
                );
            }
        );
    }
    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
        torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);
        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>()
            );
            cub::DoubleBuffer<int32_t> d_values(
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>()
            );
            GSPLAT_CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                d_keys,
                d_values,
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
            switch (d_keys.selector) {
            case 0: // sorted items are stored in isect_ids
                isect_ids_sorted = isect_ids;
                break;
            case 1: // sorted items are stored in isect_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n",
            // d_values.selector);
        } else {
            GSPLAT_CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>(),
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
        }
        return std::make_tuple(
            tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted
        );
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

__global__ void isect_offset_encode(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t C,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [C, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t cid_curr = isect_id_curr >> tile_n_bits;
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = cid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (cid,
        // tile_id) pair changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t cid_prev = isect_id_prev >> tile_n_bits;
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

torch::Tensor isect_offset_encode_tensor(
    const torch::Tensor &isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    GSPLAT_DEVICE_GUARD(isect_ids);
    GSPLAT_CHECK_INPUT(isect_ids);

    uint32_t n_isects = isect_ids.size(0);
    torch::Tensor offsets = torch::empty(
        {C, tile_height, tile_width}, isect_ids.options().dtype(torch::kInt32)
    );
    if (n_isects) {
        uint32_t n_tiles = tile_width * tile_height;
        uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        isect_offset_encode<<<
            (n_isects + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
            GSPLAT_N_THREADS,
            0,
            stream>>>(
            n_isects,
            isect_ids.data_ptr<int64_t>(),
            C,
            n_tiles,
            tile_n_bits,
            offsets.data_ptr<int32_t>()
        );
    } else {
        offsets.fill_(0);
    }
    return offsets;
}

} // namespace gsplat
