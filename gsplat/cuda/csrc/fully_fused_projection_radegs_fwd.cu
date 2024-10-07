#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/pca.hpp>

namespace gsplat {

namespace cg = cooperative_groups;

using namespace glm;

/****************************************************************************
 * Projection of Gaussians (Single Batch) Forward Pass RaDe-GS
 ****************************************************************************/


template <typename T, bool INTE=false>
__global__ void fully_fused_projection_fwd_radegs_kernel(
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ quats,    // [N, 4] optional
    const T *__restrict__ scales,   // [N, 3] optional
    const T *__restrict__ opacities,// [N, 1] optional
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T eps2d,
    const T near_plane,
    const T far_plane,
    const T radius_clip,
    const bool ortho,
    float kernel_size,
    // outputs
    int32_t *__restrict__ radii,  // [C, N]
    T *__restrict__ means2d,      // [C, N, 2]
    T *__restrict__ depths,       // [C, N]
    T *__restrict__ conic_opacities,       // [C, N, 4]
    T *__restrict__ compensations, // [C, N] optional
    // RaDe-GS specific outputs. TODO: check sizes
    T *__restrict__ camera_plane,  // [C, N, 6] # TODO: NOT USED
    T *__restrict__ output_normal, // [C, N, 3]
    T *__restrict__ ray_plane,     // [C, N, 3]
    T *__restrict__ coef,          // [C, N, 1]
    T *__restrict__ invraycov3Ds,  // [C, N, 6] optional
    T *__restrict__ ts             // [C, N]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    const T tan_fovx = image_width / (2.f * Ks[0]);
    const T tan_fovy = image_height / (2.f * Ks[4]);

    // glm is column-major but input is row-major
    mat3<T> R = mat3<T>(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    mat3<T> W = R;
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3<T> covar;
    // compute from quaternions and scales
    quats += gid * 4;
    scales += gid * 3;
    quat_scale_to_covar_preci<T>(
        glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
    );
    mat3<T> covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // perspective projection
    mat2<T> covar2d;
    vec2<T> mean2d;

//    printf("Doing perspective projection\n");
//    printf("     mean_c: %f %f %f\n", mean_c.x, mean_c.y, mean_c.z);
//    printf("     covar_c: %f %f %f\n", covar_c[0][0], covar_c[0][1], covar_c[0][2]);
//    printf("               %f %f %f\n", covar_c[1][0], covar_c[1][1], covar_c[1][2]);
//    printf("               %f %f %f\n", covar_c[2][0], covar_c[2][1], covar_c[2][2]);
//    printf("     Ks: %f %f %f\n", Ks[0], Ks[1], Ks[2]);
//    printf("         %f %f %f\n", Ks[3], Ks[4], Ks[5]);
//    printf("         %f %f %f\n", Ks[6], Ks[7], Ks[8]);
//    printf("     image_width: %d\n", image_width);
//    printf("     image_height: %d\n", image_height);


    if (ortho){
        ortho_proj<T>(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
    } else {
        persp_proj<T>(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
    }
//    printf("     resulted\n");
//    printf("     covar2d: %f %f\n", covar2d[0][0], covar2d[0][1]);
//    printf("               %f %f\n", covar2d[1][0], covar2d[1][1]);
//    printf("     mean2d: %f %f\n", mean2d.x, mean2d.y);

    means2d[idx * 2] = 0;
    means2d[idx * 2 + 1] = 0;

    T compensation;
    T det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2<T> covar2d_inv;
    inverse(covar2d, covar2d_inv);

    // take 3 sigma as the radius (non differentiable)
    T b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    T v1 = b + sqrt(max(0.01f, b * b - det));
    T radius = ceil(3.f * sqrt(v1));

    // T v2 = b - sqrt(max(0.1f, b * b - det));
    // T radius = ceil(3.f * sqrt(max(v1, v2)));

    if (radius <= radius_clip) {
        radii[idx] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
        mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
        radii[idx] = 0;
        return;
    }

    // write to outputs
    radii[idx] = (int32_t)radius;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    ts[idx] = sqrt(mean_c.x * mean_c.x + mean_c.y * mean_c.y + mean_c.z * mean_c.z);
    depths[idx] = mean_c.z;
    conic_opacities[idx * 4] = covar2d_inv[0][0];
    conic_opacities[idx * 4 + 1] = covar2d_inv[0][1];
    conic_opacities[idx * 4 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }

    // Shift rade-gs specific pointers
    camera_plane += idx * 6;
    output_normal += idx * 3;
    ray_plane += idx * 2;
    coef += idx;
    invraycov3Ds += idx * 6;

    // RaDe-GS variables
    mat3<T> cov = covar_c; // covariance in camera space
    mat3<T> Vrk = covar; // covariance in world space
    const T focal_x = Ks[0];
    const T focal_y = Ks[4];

    // RaDe-GS specific computations
    const T det_0 = max(1e-6, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
    const T det_1 = max(1e-6, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
    *coef = sqrt(det_0 / (det_1+1e-6) + 1e-6);
    if (det_0 <= 1e-6 || det_1 <= 1e-6){
        *coef = 0.0f;
    }

    conic_opacities[idx * 4 + 3] = opacities[idx] * (*coef);

    t = {mean_c.x, mean_c.y, mean_c.z};

    const T limx = 1.3f * tan_fovx;
    const T limy = 1.3f * tan_fovy;
    T txtz = t.x / t.z;
    T tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;
    txtz = t.x / t.z;
    tytz = t.y / t.z;


    mat3<T> Vrk_eigen_vector;
    vec3<T> Vrk_eigen_value;
    // TODO: use glm_modification
    int D = glm::findEigenvaluesSymReal(Vrk,Vrk_eigen_value,Vrk_eigen_vector);

    unsigned int min_id = Vrk_eigen_value[0]>Vrk_eigen_value[1]? (Vrk_eigen_value[1]>Vrk_eigen_value[2]?2:1):(Vrk_eigen_value[0]>Vrk_eigen_value[2]?2:0);

    mat3<T> Vrk_inv;
    bool well_conditioned = Vrk_eigen_value[min_id]>0.00000001;
        vec3<T> eigenvector_min;
    if(well_conditioned)
    {
        mat3<T> diag = glm::mat3( 1/Vrk_eigen_value[0], 0, 0,
                                    0, 1/Vrk_eigen_value[1], 0,
                                    0, 0, 1/Vrk_eigen_value[2] );
        Vrk_inv = Vrk_eigen_vector * diag * glm::transpose(Vrk_eigen_vector);
    }
    else
    {
        eigenvector_min = Vrk_eigen_vector[min_id];
        Vrk_inv = glm::outerProduct(eigenvector_min,eigenvector_min);
    }

    mat3<T> cov_cam_inv = glm::transpose(W) * Vrk_inv * W;
    vec3<T> uvh = {txtz, tytz, 1};
    vec3<T> uvh_m = cov_cam_inv * uvh;
    vec3<T> uvh_mn = glm::normalize(uvh_m);

    if(isnan(uvh_mn.x)|| D==0)
    {
        for(int ch = 0; ch < 6; ch++)
            camera_plane[ch] = 0;
        output_normal[0] = 0;
        output_normal[1] = 0;
        output_normal[2] = 0;
        ray_plane[0] = 0;
        ray_plane[1] = 0;
    }
    else
    {
        T u2 = txtz * txtz;
        T v2 = tytz * tytz;
        T uv = txtz * tytz;

        T l = sqrt(t.x*t.x+t.y*t.y+t.z*t.z);
        mat3<T> nJ = mat3<T>(
            1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
            0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
            t.x/l, t.y/l, t.z/l);

        mat3<T> nJ_inv = mat3<T>(
            v2 + 1,    -uv,         0,
            -uv,    u2 + 1,        0,
            -txtz,    -tytz,        0
        );

        if constexpr (INTE)
        {
            mat3<T> inv_cov_ray;
            if(well_conditioned)
            {
                T ltz = u2+v2+1;
                mat3<T> nJ_inv_full = t.z/(u2+v2+1) * \
                                        mat3<T>(
                                            v2 + 1,    -uv,         txtz/l*ltz,
                                            -uv,    u2 + 1,        tytz/l*ltz,
                                            -txtz,    -tytz,        1/l*ltz);
                mat3<T> T2 = W * glm::transpose(nJ_inv_full);
                inv_cov_ray = glm::transpose(T2) * Vrk_inv * T2;
            }
            else
            {
                mat3<T> T2 = W * nJ;
                mat3<T> cov_ray = glm::transpose(T2) * Vrk_inv * T2;
                mat3<T> cov_eigen_vector;
                mat3<T> cov_eigen_value;
                // TODO: use glm_modification
                glm::findEigenvaluesSymReal(cov_ray,cov_eigen_value,cov_eigen_vector);
                unsigned int min_id = cov_eigen_value[0]>cov_eigen_value[1]? (cov_eigen_value[1]>cov_eigen_value[2]?2:1):(cov_eigen_value[0]>cov_eigen_value[2]?2:0);
                T lambda1 = cov_eigen_value[(min_id+1)%3];
                T lambda2 = cov_eigen_value[(min_id+2)%3];
                T lambda3 = cov_eigen_value[min_id];
                mat3<T> new_cov_eigen_vector = mat3<T>();
                new_cov_eigen_vector[0] = cov_eigen_vector[(min_id+1)%3];
                new_cov_eigen_vector[1] = cov_eigen_vector[(min_id+2)%3];
                new_cov_eigen_vector[2] = cov_eigen_vector[min_id];
                glm::vec3 r3 = glm::vec3(new_cov_eigen_vector[0][2],new_cov_eigen_vector[1][2],new_cov_eigen_vector[2][2]);

                mat3<T> cov2d = mat3<T>(
                    1/lambda1,0,-r3[0]/r3[2]/lambda1,
                    0,1/lambda2,-r3[1]/r3[2]/lambda2,
                    -r3[0]/r3[2]/lambda1,-r3[1]/r3[2]/lambda2,0
                );
                mat3<T> inv_cov_ray = new_cov_eigen_vector * cov2d * glm::transpose(new_cov_eigen_vector);
            }
            mat3<T> scale = mat3<T>(1/focal_x,0,0,
                                        0, 1/focal_y,0,
                                        0,0,1);
            inv_cov_ray = scale * inv_cov_ray * scale;
            invraycov3Ds[0] = inv_cov_ray[0][0];
            invraycov3Ds[1] = inv_cov_ray[0][1];
            invraycov3Ds[2] = inv_cov_ray[0][2];
            invraycov3Ds[3] = inv_cov_ray[1][1];
            invraycov3Ds[4] = inv_cov_ray[1][2];
            invraycov3Ds[5] = inv_cov_ray[2][2];
        }


        T vbn = glm::dot(uvh_mn, uvh);
        T factor_normal = l / (u2+v2+1);
        vec3<T> plane = nJ_inv * (uvh_mn/max(vbn,0.0000001f));
        T nl = u2+v2+1;
        vec2<T> camera_plane_x =//;// {(-(v2 + 1)*t.z+plane[0]*t.x)/nl/focal_x, 0};
        {(-(v2 + 1)*t.z+plane[0]*t.x)/nl/focal_x, (uv*t.z+plane[1]*t.x)/nl/focal_y};
        vec2<T> camera_plane_y =//;// {0, 0};
         {(uv*t.z+plane[0]*t.y)/nl/focal_x,  (-(u2 + 1)*t.z+plane[1]*t.y)/nl/focal_y};
        vec2<T> camera_plane_z =//;// {0, 0};
         {(t.x+plane[0]*t.z)/nl/focal_x, (t.y+plane[1]*t.z)/nl/focal_y};

        ray_plane[0] = plane[0]*l/nl/focal_x;
        ray_plane[1] = plane[1]*l/nl/focal_y;

        camera_plane[0] =  camera_plane_x.x;
        camera_plane[1] =  camera_plane_x.y;
        camera_plane[2] =  camera_plane_y.x;
        camera_plane[3] =  camera_plane_y.y;
        camera_plane[4] =  camera_plane_z.x;
        camera_plane[5] =  camera_plane_z.y;

        vec3<T> ray_normal_vector = {-plane[0]*factor_normal, -plane[1]*factor_normal, -1};
        vec3<T> cam_normal_vector = nJ * ray_normal_vector;
        vec3<T> normal_vector = glm::normalize(cam_normal_vector);

        output_normal[0] = normal_vector.x;
        output_normal[1] = normal_vector.y;
        output_normal[2] = normal_vector.z;
    }
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_fwd_radegs_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const torch::Tensor &opacities,             // [N, 1]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const bool ortho
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats.value());
    GSPLAT_CHECK_INPUT(scales.value());
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii =
        torch::empty({C, N}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor conic_opacities = torch::empty({C, N, 4}, means.options());
    torch::Tensor compensations;
    torch::Tensor camera_plane = torch::empty({C, N, 6}, means.options());
    torch::Tensor output_normal = torch::empty({C, N, 3}, means.options());
    torch::Tensor ray_plane = torch::empty({C, N, 2}, means.options());
    torch::Tensor coef = torch::empty({C, N}, means.options());
    torch::Tensor invraycov3Ds = torch::empty({C, N, 6}, means.options());
    torch::Tensor ts = torch::empty({C, N}, means.options());

//    printf("CREATED measn2d %d %d %d\n", means2d.size(0), means2d.size(1), means2d.size(2));

    const float kernel_size = 0.1f;

    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({C, N}, means.options());
    }

    if (C && N) {
        if (means.scalar_type() == torch::kFloat32) {
            using T = float;
            fully_fused_projection_fwd_radegs_kernel<T>
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
            GSPLAT_N_THREADS,
            0,
            stream>>>(
                    C,
                    N,
                    means.data_ptr<T>(),
                    quats.has_value() ? quats.value().data_ptr<T>() : nullptr,
                    scales.has_value() ? scales.value().data_ptr<T>() : nullptr,
                    opacities.data_ptr<T>(),
                    viewmats.data_ptr<T>(),
                    Ks.data_ptr<T>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    ortho,
                    kernel_size,
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<T>(),
                    depths.data_ptr<T>(),
                    conic_opacities.data_ptr<T>(),
                    calc_compensations ? compensations.data_ptr<T>() : nullptr,
                    camera_plane.data_ptr<T>(),  // [C, N, 6]
                    output_normal.data_ptr<T>(), // [C, N, 3]
                    ray_plane.data_ptr<T>(),     // [C, N, 3]
                    coef.data_ptr<T>(),          // [C, N, 1]
                    invraycov3Ds.data_ptr<T>(),   // [C, N, 6]
                    ts.data_ptr<T>()
            );
        } else if (means.scalar_type() == torch::kFloat64) {
            using T = double;

            fully_fused_projection_fwd_radegs_kernel<T>
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
            GSPLAT_N_THREADS,
            0,
            stream>>>(
                    C,
                    N,
                    means.data_ptr<T>(),
                    quats.has_value() ? quats.value().data_ptr<T>() : nullptr,
                    scales.has_value() ? scales.value().data_ptr<T>() : nullptr,
                    opacities.data_ptr<T>(),
                    viewmats.data_ptr<T>(),
                    Ks.data_ptr<T>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    ortho,
                    kernel_size,
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<T>(),
                    depths.data_ptr<T>(),
                    conic_opacities.data_ptr<T>(),
                    calc_compensations ? compensations.data_ptr<T>() : nullptr,
                    camera_plane.data_ptr<T>(),  // [C, N, 6]
                    output_normal.data_ptr<T>(), // [C, N, 3]
                    ray_plane.data_ptr<T>(),     // [C, N, 3]
                    coef.data_ptr<T>(),          // [C, N, 1]
                    invraycov3Ds.data_ptr<T>(),   // [C, N, 6]
                    ts.data_ptr<T>()
            );
        } else {
            throw std::runtime_error("Unsupported scalar type");
        }
    }
    return std::make_tuple(radii, means2d, depths, conic_opacities, compensations, camera_plane, output_normal, ray_plane, coef, invraycov3Ds, ts);
}


} // namespace gsplat