//#pragma once
// This file is compiled many times for different simd architectures (SSE, NEON...).
// Macro 'NS' defines a namespace in which everything will be located, so it should be set before including this file.
// Macros 'USE_XXX' define template instantiation of simd_fvec, simd_ivec classes.
// Template parameter S defines width of vectors used. Usualy it is equal to ray packet size.

#include <vector>

#include "TextureAtlasRef.h"

#include "simd/simd_vec.h"

namespace Ray {
namespace NS {

const int ray_packet_layout_x[] = { 0, 1, 0, 1,
                                    2, 3, 2, 3,
                                    0, 1, 0, 1,
                                    2, 3, 2, 3 };

const int ray_packet_layout_y[] = { 0, 0, 1, 1,
                                    0, 0, 1, 1,
                                    2, 2, 3, 3,
                                    2, 2, 3, 3 };

template <int S>
struct ray_packet_t {
    // directions of rays in packet
    simd_fvec<S> d[3];
    // origins of rays in packet
    simd_fvec<S> o[3];
    // color of Ray and ior of medium
    simd_fvec<S> c[3], ior;
    // derivatives
    simd_fvec<S> do_dx[3], dd_dx[3], do_dy[3], dd_dy[3];
    // 16-bit pixel coordinates of rays in packet ((x << 16) | y)
    simd_ivec<S> xy;
};

template <int S>
struct hit_data_t {
    simd_ivec<S> mask;
    simd_ivec<S> obj_index;
    simd_ivec<S> prim_index;
    simd_fvec<S> t, u, v;
    // 16-bit pixel coordinates of rays in packet ((x << 16) | y)
    simd_ivec<S> xy;

    hit_data_t(eUninitialize) {}
    force_inline hit_data_t() {
        mask = { 0 };
        obj_index = { -1 };
        prim_index = { -1 };
        t = { MAX_DIST };
    }
};

struct environment_t {
    float env_col[3];
    float env_clamp;
    uint32_t env_map;
};

template <int S>
struct derivatives_t {
    simd_fvec<S> do_dx[3], dd_dx[3], do_dy[3], dd_dy[3];
    simd_fvec<S> duv_dx[2], duv_dy[2];
    simd_fvec<S> dndx[3], dndy[3];
    simd_fvec<S> ddn_dx, ddn_dy;
};

// Generating rays
template <int DimX, int DimY>
void GeneratePrimaryRays(const int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t<DimX * DimY>> &out_rays);
template <int DimX, int DimY>
void SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const transform_t &tr, const uint32_t *vtx_indices, const vertex_t *vertices,
                              const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t<DimX * DimY>> &out_rays, aligned_vector<hit_data_t<DimX * DimY>> &out_inters);

// Sorting rays
template <int S>
void SortRays(ray_packet_t<S> *rays, simd_ivec<S> *ray_masks, int &secondary_rays_count, const float root_min[3], const float cell_size[3],
              simd_ivec<S> *hash_values, int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp, uint32_t *skeleton);

// Intersect primitives
template <int S>
bool IntersectTris(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t<S> &out_inter);
template <int S>
bool IntersectTris(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t<S> &out_inter);

// Traverse acceleration structure
// stack-less cpu-style traversal of outer nodes
template <int S>
bool Traverse_MacroTree_Stackless_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                      const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<S> &inter);
// stack-less cpu-style traversal of inner nodes
template <int S>
bool Traverse_MicroTree_Stackless_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                      const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter);

// traditional bvh traversal with stack for outer nodes
template <int S>
bool Traverse_MacroTree_WithStack(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                  const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                  const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<S> &inter);
// traditional bvh traversal with stack for inner nodes
template <int S>
__declspec(noinline) bool Traverse_MicroTree_WithStack(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                  const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter);

// Transform
template <int S>
ray_packet_t<S> TransformRay(const ray_packet_t<S> &r, const float *xform);
template <int S>
void TransformPoint(const simd_fvec<S> p[3], const float *xform, simd_fvec<S> out_p[3]);
template <int S>
void TransformNormal(const simd_fvec<S> n[3], const float *inv_xform, simd_fvec<S> out_n[3]);
template <int S>
void TransformUVs(const simd_fvec<S> _uvs[2], float sx, float sy, const texture_t &t, const simd_ivec<S> &mip_level, const simd_ivec<S> &mask, simd_fvec<S> out_res[2]);

// Sample texture
template <int S>
void SampleNearest(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<S> uvs[2], const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleBilinear(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<S> uvs[2], const simd_ivec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleBilinear(const Ref::TextureAtlas &atlas, const simd_fvec<S> uvs[2], const simd_ivec<S> &page, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleTrilinear(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<S> uvs[2], const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleAnisotropic(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<S> uvs[2], const simd_fvec<S> duv_dx[2], const simd_fvec<S> duv_dy[2], const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]);
template <int S>
void SampleLatlong_RGBE(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<S> dir[3], const simd_ivec<S> &mask, simd_fvec<S> out_rgb[3]);

// Compute punctual lights contribution
template <int S>
void ComputeDirectLighting(const simd_fvec<S> P[3], const simd_fvec<S> N[3], const simd_fvec<S> B[3], const simd_fvec<S> plane_N[3],
                           const float *halton, const int hi, const simd_fvec<S> &rand_offset, const simd_fvec<S> &rand_offset2,
                           const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                           const uint32_t *vtx_indices, const vertex_t *vertices, const bvh_node_t *nodes, uint32_t node_index,
                           const tri_accel_t *tris, const uint32_t *tri_indices, const light_t *lights,
                           const uint32_t *li_indices, uint32_t light_node_index, const simd_ivec<S> &ray_mask, simd_fvec<S> *out_col);

// Compute derivatives at hit point
template <int S>
void ComputeDerivatives(const simd_fvec<S> I[3], const simd_fvec<S> &t, const simd_fvec<S> do_dx[3], const simd_fvec<S> do_dy[3], const simd_fvec<S> dd_dx[3], const simd_fvec<S> dd_dy[3],
                        const simd_fvec<S> p1[3], const simd_fvec<S> p2[3], const simd_fvec<S> p3[3], const simd_fvec<S> n1[3], const simd_fvec<S> n2[3], const simd_fvec<S> n3[3],
                        const simd_fvec<S> u1[2], const simd_fvec<S> u2[2], const simd_fvec<S> u3[2], const simd_fvec<S> plane_N[3], derivatives_t<S> &out_der);

// Shade
template <int S>
void ShadeSurface(const simd_ivec<S> &index, const pass_info_t &pi, const float *halton, const hit_data_t<S> &inter, const ray_packet_t<S> &ray,
                  const environment_t &env, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                  const mesh_t *meshes, const transform_t *transforms, const uint32_t *vtx_indices, const vertex_t *vertices,
                  const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                  const material_t *materials, const texture_t *textures, const Ref::TextureAtlas &tex_atlas,
                  const light_t *lights, const uint32_t *li_indices, uint32_t light_node_index, simd_fvec<S> out_rgba[4], simd_ivec<S> *out_secondary_masks, ray_packet_t<S> *out_secondary_rays, int *out_secondary_rays_count);
}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cassert>

namespace Ray {
namespace NS {
template <int S>
force_inline void _IntersectTri(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t &tri, uint32_t prim_index, hit_data_t<S> &inter) {
    const int _next_u[] = { 1, 0, 0 },
              _next_v[] = { 2, 2, 1 };

    int w = (tri.ci & TRI_W_BITS),
        u = _next_u[w],
        v = _next_v[w];

    // from "Ray-Triangle Intersection Algorithm for Modern CPU Architectures" [2007]

    simd_fvec<S> det = r.d[u] * tri.nu + r.d[v] * tri.nv + r.d[w];
    simd_fvec<S> dett = tri.np - (r.o[u] * tri.nu + r.o[v] * tri.nv + r.o[w]);
    simd_fvec<S> Du = r.d[u] * dett - (tri.pu - r.o[u]) * det;
    simd_fvec<S> Dv = r.d[v] * dett - (tri.pv - r.o[v]) * det;
    simd_fvec<S> detu = tri.e1v * Du - tri.e1u * Dv;
    simd_fvec<S> detv = tri.e0u * Dv - tri.e0v * Du;

    simd_fvec<S> tmpdet0 = det - detu - detv;

    //////////////////////////////////////////////////////////////////////////

    simd_fvec<S> mm = ((tmpdet0 > -HIT_EPS) & (detu > -HIT_EPS) & (detv > -HIT_EPS)) |
                      ((tmpdet0 < HIT_EPS) & (detu < HIT_EPS) & (detv < HIT_EPS));

    simd_ivec<S> imask = reinterpret_cast<const simd_ivec<S>&>(mm) & ray_mask;

    if (imask.all_zeros()) return; // no intersection found

    simd_fvec<S> rdet = 1.0f / det;
    simd_fvec<S> t = dett * rdet;

    simd_fvec<S> t_valid = (t < inter.t) & (t > 0.0f);
    imask = imask & reinterpret_cast<const simd_ivec<S>&>(t_valid);

    if (imask.all_zeros()) return; // all intersections further than needed

    simd_fvec<S> bar_u = detu * rdet;
    simd_fvec<S> bar_v = detv * rdet;

    const auto &fmask = reinterpret_cast<const simd_fvec<S>&>(imask);

    inter.mask = inter.mask | imask;

    where(imask, inter.prim_index) = simd_ivec<S>{ reinterpret_cast<const int&>(prim_index) };
    where(fmask, inter.t) = t;
    where(fmask, inter.u) = bar_u;
    where(fmask, inter.v) = bar_v;
}

template <int S>
force_inline simd_ivec<S> bbox_test(const simd_fvec<S> o[3], const simd_fvec<S> inv_d[3], const simd_fvec<S> &t, const float _bbox_min[3], const float _bbox_max[3]) {
    simd_fvec<S> low, high, tmin, tmax;
    
    low = inv_d[0] * (_bbox_min[0] - o[0]);
    high = inv_d[0] * (_bbox_max[0] - o[0]);
    tmin = min(low, high);
    tmax = max(low, high);

    low = inv_d[1] * (_bbox_min[1] - o[1]);
    high = inv_d[1] * (_bbox_max[1] - o[1]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    low = inv_d[2] * (_bbox_min[2] - o[2]);
    high = inv_d[2] * (_bbox_max[2] - o[2]);
    tmin = max(tmin, min(low, high));
    tmax = min(tmax, max(low, high));

    simd_fvec<S> mask = (tmin <= tmax) & (tmin <= t) & (tmax > 0.0f);
    
    return reinterpret_cast<const simd_ivec<S>&>(mask);
}

template <int S>
force_inline simd_ivec<S> bbox_test(const simd_fvec<S> p[3], const float _bbox_min[3], const float _bbox_max[3]) {
    simd_fvec<S> mask = (p[0] > _bbox_min[0]) & (p[0] < _bbox_max[0]) &
                        (p[1] > _bbox_min[1]) & (p[1] < _bbox_max[1]) &
                        (p[2] > _bbox_min[2]) & (p[2] < _bbox_max[2]);
    return reinterpret_cast<const simd_ivec<S>&>(mask);
}

template <int S>
force_inline simd_ivec<S> bbox_test(const simd_fvec<S> o[3], const simd_fvec<S> inv_d[3], const simd_fvec<S> &t, const bvh_node_t &node) {
    return bbox_test(o, inv_d, t, node.bbox[0], node.bbox[1]);
}

template <int S>
force_inline simd_ivec<S> bbox_test(const simd_fvec<S> p[3], const bvh_node_t &node) {
    return bbox_test(p, node.bbox[0], node.bbox[1]);
}

template <int S>
force_inline uint32_t near_child(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t &node) {
    const auto dir_neg_mask = r.d[node.flags & SPACE_AXIS_BITS] < 0.0f;
    const auto mask = reinterpret_cast<const simd_ivec<S>&>(dir_neg_mask);
    if (mask.all_zeros(ray_mask)) {
        return node.left_child;
    } else {
        assert(and_not(mask, ray_mask).all_zeros());
        return node.right_child;
    }
}

force_inline uint32_t other_child(const bvh_node_t &node, uint32_t cur_child) {
    return node.left_child == cur_child ? node.right_child : node.left_child;
}

force_inline bool is_leaf_node(const bvh_node_t &node) {
    return node.prim_count != 0;
}

enum eTraversalSource { FromParent, FromChild, FromSibling };

template <int S>
struct TraversalState {
    struct {
        simd_ivec<S> mask;
        uint32_t cur;
        eTraversalSource src;
    } queue[S];

    int index = 0, num = 1;

    force_inline void select_near_child(const ray_packet_t<S> &r, const bvh_node_t &node) {
        const auto dir_neg_mask = r.d[node.flags & SPACE_AXIS_BITS] < 0.0f;
        const auto mask1 = reinterpret_cast<const simd_ivec<S>&>(dir_neg_mask) & queue[index].mask;
        if (mask1.all_zeros()) {
            queue[index].cur = node.left_child;
        } else {
            simd_ivec<S> mask2 = and_not(mask1, queue[index].mask);
            if (mask2.all_zeros()) {
                queue[index].cur = node.right_child;
            } else {
                queue[num].cur = node.left_child;
                queue[num].mask = mask2;
                queue[num].src = queue[index].src;
                num++;
                queue[index].cur = node.right_child;
                queue[index].mask = mask1;
            }
        }
    }
};

template <int S>
struct TraversalStateStack {
    struct {
        simd_ivec<S> mask;
        uint32_t stack[MAX_STACK_SIZE];
        uint32_t stack_size;
    } queue[S];

    force_inline void push_children(const ray_packet_t<S> &r, const bvh_node_t &node) {
        const auto dir_neg_mask = r.d[node.flags & SPACE_AXIS_BITS] < 0.0f;
        const auto mask1 = reinterpret_cast<const simd_ivec<S>&>(dir_neg_mask) & queue[index].mask;
        if (mask1.all_zeros()) {
            queue[index].stack[queue[index].stack_size++] = node.right_child;
            queue[index].stack[queue[index].stack_size++] = node.left_child;
        } else {
            simd_ivec<S> mask2 = and_not(mask1, queue[index].mask);
            if (mask2.all_zeros()) {
                queue[index].stack[queue[index].stack_size++] = node.left_child;
                queue[index].stack[queue[index].stack_size++] = node.right_child;
            } else {
                queue[num].stack_size = queue[index].stack_size;
                memcpy(queue[num].stack, queue[index].stack, sizeof(uint32_t) * queue[index].stack_size);
                queue[num].stack[queue[num].stack_size++] = node.right_child;
                queue[num].stack[queue[num].stack_size++] = node.left_child;
                queue[num].mask = mask2;
                num++;
                queue[index].stack[queue[index].stack_size++] = node.left_child;
                queue[index].stack[queue[index].stack_size++] = node.right_child;
                queue[index].mask = mask1;
            }
        }
    }

    force_inline void push_children(const bvh_node_t &node) {
        queue[index].stack[queue[index].stack_size++] = node.left_child;
        queue[index].stack[queue[index].stack_size++] = node.right_child;
    }

    int index = 0, num = 1;
};

template <int S>
force_inline void safe_invert(const simd_fvec<S> v[3], simd_fvec<S> inv_v[3]) {
    inv_v[0] = { 1.0f / v[0] };
    where(v[0] <= FLT_EPS & v[0] >= 0, inv_v[0]) = MAX_DIST;
    where(v[0] >= -FLT_EPS & v[0] < 0, inv_v[0]) = -MAX_DIST;

    inv_v[1] = { 1.0f / v[1] };
    where(v[1] <= FLT_EPS & v[1] >= 0, inv_v[1]) = MAX_DIST;
    where(v[1] >= -FLT_EPS & v[1] < 0, inv_v[1]) = -MAX_DIST;

    inv_v[2] = { 1.0f / v[2] };
    where(v[2] <= FLT_EPS & v[2] >= 0, inv_v[2]) = MAX_DIST;
    where(v[2] >= -FLT_EPS & v[2] < 0, inv_v[2]) = -MAX_DIST;
}

template <int S>
force_inline simd_fvec<S> dot(const simd_fvec<S> v1[3], const simd_fvec<S> v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template <int S>
force_inline simd_fvec<S> dot(const simd_fvec<S> v1[3], const float v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template <int S>
force_inline simd_fvec<S> dot(const float v1[3], const simd_fvec<S> v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}


template <int S>
force_inline void cross(const simd_fvec<S> v1[3], const simd_fvec<S> v2[3], simd_fvec<S> res[3]) {
    res[0] = v1[1] * v2[2] - v1[2] * v2[1];
    res[1] = v1[2] * v2[0] - v1[0] * v2[2];
    res[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

template <int S>
force_inline void normalize(simd_fvec<S> v[3]) {
    simd_fvec<S> l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= l;
    v[1] /= l;
    v[2] /= l;
}

template <int S>
force_inline simd_fvec<S> length(const simd_fvec<S> v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

template <int S>
force_inline simd_fvec<S> clamp(const simd_fvec<S> &v, float min, float max) {
    simd_fvec<S> ret = v;
    where(ret < min, ret) = min;
    where(ret > max, ret) = max;
    return ret;
}

template <int S>
force_inline simd_ivec<S> clamp(const simd_ivec<S> &v, int min, int max) {
    simd_ivec<S> ret = v;
    where(ret < min, ret) = min;
    where(ret > max, ret) = max;
    return ret;
}

force_inline int hash(int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

template <int S>
force_inline simd_ivec<S> hash(const simd_ivec<S> &x) {
    simd_ivec<S> ret;
    ret = ((x >> 16) ^ x) * 0x45d9f3b;
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = (ret >> 16) ^ ret;
    return ret;
}

force_inline float length(const simd_fvec2 &x) {
    return sqrtf(x[0] * x[0] + x[1] * x[1]);
}

force_inline float floor(float x) {
    return (float)((int)x - (x < 0.0f));
}

template <int S>
force_inline void reflect(const simd_fvec<S> I[3], const simd_fvec<S> N[3], const simd_fvec<S> &dot_N_I, simd_fvec<S> res[3]) {
    res[0] = I[0] - 2.0f * dot_N_I * N[0];
    res[1] = I[1] - 2.0f * dot_N_I * N[1];
    res[2] = I[2] - 2.0f * dot_N_I * N[2];
}

template <int S>
force_inline simd_ivec<S> get_ray_hash(const ray_packet_t<S> &r, const simd_ivec<S> &mask, const float root_min[3], const float cell_size[3]) {
    simd_ivec<S> x = clamp((simd_ivec<S>)((r.o[0] - root_min[0]) / cell_size[0]), 0, 255),
                 y = clamp((simd_ivec<S>)((r.o[1] - root_min[1]) / cell_size[1]), 0, 255),
                 z = clamp((simd_ivec<S>)((r.o[2] - root_min[2]) / cell_size[2]), 0, 255);

    simd_ivec<S> omega_index = clamp((simd_ivec<S>)((1.0f + r.d[2]) / omega_step), 0, 32),
                 phi_index_i = clamp((simd_ivec<S>)((1.0f + r.d[1]) / phi_step), 0, 16),
                 phi_index_j = clamp((simd_ivec<S>)((1.0f + r.d[0]) / phi_step), 0, 16);

    simd_ivec<S> o, p;

    ITERATE(S, {
        if (mask[i]) {
            x[i] = morton_table_256[x[i]];
            y[i] = morton_table_256[y[i]];
            z[i] = morton_table_256[z[i]];
            o[i] = morton_table_16[omega_table[omega_index[i]]];
            p[i] = morton_table_16[phi_table[phi_index_i[i]][phi_index_j[i]]];
        } else {
            o[i] = p[i] = 0xFFFFFFFF;
            x[i] = y[i] = z[i] = 0xFFFFFFFF;
        }
    });

    return (o << 25) | (p << 24) | (y << 2) | (z << 1) | (x << 0);
}

force_inline void _radix_sort_lsb(ray_chunk_t *begin, ray_chunk_t *end, ray_chunk_t *begin1, unsigned maxshift) {
    ray_chunk_t *end1 = begin1 + (end - begin);

    for (unsigned shift = 0; shift <= maxshift; shift += 8) {
        size_t count[0x100] = {};
        for (ray_chunk_t *p = begin; p != end; p++) {
            count[(p->hash >> shift) & 0xFF]++;
        }
        ray_chunk_t *bucket[0x100], *q = begin1;
        for (int i = 0; i < 0x100; q += count[i++]) {
            bucket[i] = q;
        }
        for (ray_chunk_t *p = begin; p != end; p++) {
            *bucket[(p->hash >> shift) & 0xFF]++ = *p;
        }
        std::swap(begin, begin1);
        std::swap(end, end1);
    }
}

force_inline void radix_sort(ray_chunk_t *begin, ray_chunk_t *end, ray_chunk_t *begin1) {
    _radix_sort_lsb(begin, end, begin1, 24);
}

template <int S>
force_inline simd_fvec<S> construct_float(const simd_ivec<S> &_m) {
    const simd_ivec<S> ieeeMantissa = { 0x007FFFFF }; // binary32 mantissa bitmask
    const simd_ivec<S> ieeeOne = { 0x3F800000 };      // 1.0 in IEEE binary32

    simd_ivec<S> m = _m & ieeeMantissa;       // Keep only mantissa bits (fractional part)
    m = m | ieeeOne;                          // Add fractional part to 1.0

    simd_fvec<S>  f = reinterpret_cast<simd_fvec<S> &>(m);  // Range [1:2]
    return f - simd_fvec<S>{ 1.0f };                        // Range [0:1]
}

}
}

template <int DimX, int DimY>
void Ray::NS::GeneratePrimaryRays(const int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t<DimX * DimY>> &out_rays) {
    const int S = DimX * DimY;
    static_assert(S <= 16, "!");

    simd_fvec<S> ww = { (float)w }, hh = { (float)h };

    float k = float(h) / w;

    float focus_distance = cam.focus_distance;
    float fov_k = std::tan(0.5f * cam.fov * PI / 180.0f) * focus_distance;

    simd_fvec<S> fwd[3] = { { cam.fwd[0] }, { cam.fwd[1] }, { cam.fwd[2] } },
                 side[3] = { { cam.side[0] }, { cam.side[1] }, { cam.side[2] } },
                 up[3] = { { cam.up[0] }, { cam.up[1] }, { cam.up[2] } },
                 cam_origin[3] = { { cam.origin[0] }, { cam.origin[1] }, { cam.origin[2] } };

    auto get_pix_dirs = [k, fov_k, focus_distance, &fwd, &side, &up, &cam_origin, &ww, &hh](const simd_fvec<S> &x, const simd_fvec<S> &y, const simd_fvec<S> origin[3], simd_fvec<S> d[3]) {
        auto _dx = 2 * fov_k * x / ww - fov_k;
        auto _dy = 2 * fov_k  * -y / hh + fov_k;

        d[0] = cam_origin[0] + _dx * side[0] + k * _dy * up[0] + fwd[0] * focus_distance;
        d[1] = cam_origin[1] + _dx * side[1] + k * _dy * up[1] + fwd[1] * focus_distance;
        d[2] = cam_origin[2] + _dx * side[2] + k * _dy * up[2] + fwd[2] * focus_distance;

        d[0] = d[0] - origin[0];
        d[1] = d[1] - origin[1];
        d[2] = d[2] - origin[2];

        simd_fvec<DimX * DimY> len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        d[0] /= len;
        d[1] /= len;
        d[2] /= len;
    };

    simd_ivec<S> off_x, off_y;

    for (int i = 0; i < S; i++) {
        off_x[i] = ray_packet_layout_x[i];
        off_y[i] = ray_packet_layout_y[i];
    }

    size_t i = 0;
    out_rays.resize(r.w * r.h / S + ((r.w * r.h) % S != 0));

    for (int y = r.y; y < r.y + r.h - (r.h & (DimY - 1)); y += DimY) {
        for (int x = r.x; x < r.x + r.w - (r.w & (DimX - 1)); x += DimX) {
            auto &out_r = out_rays[i++];

            simd_ivec<S> ixx = x + off_x, iyy = simd_ivec<S>(y) + off_y;

            simd_ivec<S> index = iyy * w + ixx;
            const int hi = (iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

            simd_fvec<S> fxx = (simd_fvec<S>)ixx,
                         fyy = (simd_fvec<S>)iyy;

            simd_ivec<S> hash_val = hash(index);
            simd_fvec<S> rxx = construct_float(hash_val);
            simd_fvec<S> ryy = construct_float(hash(hash_val));
            simd_fvec<S> sxx, syy;

            for (int i = 0; i < S; i++) {
                float _unused;
                sxx[i] = cam.focus_factor * (-0.5f + std::modf(halton[hi + 2 + 0] + rxx[i], &_unused));
                syy[i] = cam.focus_factor * (-0.5f + std::modf(halton[hi + 2 + 1] + ryy[i], &_unused));
                rxx[i] = std::modf(halton[hi + 0] + rxx[i], &_unused);
                ryy[i] = std::modf(halton[hi + 1] + ryy[i], &_unused);
            }

            if (cam.filter == Tent) {
                auto temp = rxx;
                rxx = 1.0f - sqrt(2.0f - 2.0f * temp);
                where(temp < 0.5f, rxx) = sqrt(2.0f * temp) - 1.0f;

                temp = ryy;
                ryy = 1.0f - sqrt(2.0f - 2.0f * temp);
                where(temp < 0.5f, ryy) = sqrt(2.0f * temp) - 1.0f;

                rxx += 0.5f;
                ryy += 0.5f;
            }

            fxx += rxx;
            fyy += ryy;

            simd_fvec<S> _origin[3] = { { cam_origin[0] + side[0] * sxx + up[0] * syy },
                                        { cam_origin[1] + side[1] * sxx + up[1] * syy },
                                        { cam_origin[2] + side[2] * sxx + up[2] * syy } };

            simd_fvec<S> _d[3], _dx[3], _dy[3];
            get_pix_dirs(fxx, fyy, _origin, _d);
            get_pix_dirs(fxx + 1.0f, fyy, _origin, _dx);
            get_pix_dirs(fxx, fyy + 1.0f, _origin, _dy);

            for (int j = 0; j < 3; j++) {
                out_r.d[j] = _d[j];
                out_r.o[j] = _origin[j];
                out_r.c[j] = { 1.0f };

                out_r.do_dx[j] = { 0.0f };
                out_r.dd_dx[j] = _dx[j] - out_r.d[j];
                out_r.do_dy[j] = { 0.0f };
                out_r.dd_dy[j] = _dy[j] - out_r.d[j];
            }

            out_r.ior = { 1.0f };
            out_r.xy = (ixx << 16) | iyy;
        }
    }
}

template <int DimX, int DimY>
void Ray::NS::SampleMeshInTextureSpace(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const transform_t &tr, const uint32_t *vtx_indices, const vertex_t *vertices,
                                       const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t<DimX * DimY>> &out_rays, aligned_vector<hit_data_t<DimX * DimY>> &out_inters) {
    const int S = DimX * DimY;
    static_assert(S <= 16, "!");

    out_rays.resize(r.w * r.h / S + ((r.w * r.h) % S != 0));
    out_inters.resize(out_rays.size());

    simd_ivec<S> off_x, off_y;

    for (int i = 0; i < S; i++) {
        off_x[i] = ray_packet_layout_x[i];
        off_y[i] = ray_packet_layout_y[i];
    }

    size_t i = 0;
    for (int y = r.y; y < r.y + r.h - (r.h & (DimY - 1)); y += DimY) {
        for (int x = r.x; x < r.x + r.w - (r.w & (DimX - 1)); x += DimX) {
            simd_ivec<S> ixx = x + off_x, iyy = simd_ivec<S>(y) + off_y;

            auto &out_ray = out_rays[i];
            auto &out_inter = out_inters[i];
            i++;

            out_ray.xy = (ixx << 16) | iyy;
            out_ray.c[0] = out_ray.c[1] = out_ray.c[2] = 1.0f;
            out_ray.do_dx[0] = out_ray.do_dx[1] = out_ray.do_dx[2] = 0.0f;
            out_ray.dd_dx[0] = out_ray.dd_dx[1] = out_ray.dd_dx[2] = 0.0f;
            out_ray.do_dy[0] = out_ray.do_dy[1] = out_ray.do_dy[2] = 0.0f;
            out_ray.dd_dy[0] = out_ray.dd_dy[1] = out_ray.dd_dy[2] = 0.0f;
            out_inter.mask = 0;
            out_inter.xy = out_ray.xy;
        }
    }

    simd_ivec2 irect_min = { r.x, r.y }, irect_max = { r.x + r.w - 1, r.y + r.h - 1 };
    simd_fvec2 size = { (float)w, (float)h };

    for (uint32_t tri = mesh.tris_index; tri < mesh.tris_index + mesh.tris_count; tri++) {
        const auto &v0 = vertices[vtx_indices[tri * 3 + 0]];
        const auto &v1 = vertices[vtx_indices[tri * 3 + 1]];
        const auto &v2 = vertices[vtx_indices[tri * 3 + 2]];

        const simd_fvec2 t0 = simd_fvec2{ v0.t[uv_layer][0], 1.0f - v0.t[uv_layer][1] } *size;
        const simd_fvec2 t1 = simd_fvec2{ v1.t[uv_layer][0], 1.0f - v1.t[uv_layer][1] } *size;
        const simd_fvec2 t2 = simd_fvec2{ v2.t[uv_layer][0], 1.0f - v2.t[uv_layer][1] } *size;

        simd_fvec2 bbox_min = t0, bbox_max = t0;

        bbox_min = min(bbox_min, t1);
        bbox_min = min(bbox_min, t2);

        bbox_max = max(bbox_max, t1);
        bbox_max = max(bbox_max, t2);

        simd_ivec2 ibbox_min = (simd_ivec2)(bbox_min),
                   ibbox_max = simd_ivec2{ (int)std::round(bbox_max[0]), (int)std::round(bbox_max[1]) };

        if (ibbox_max[0] < irect_min[0] || ibbox_max[1] < irect_min[1] ||
            ibbox_min[0] > irect_max[0] || ibbox_min[1] > irect_max[1]) continue;

        ibbox_min = max(ibbox_min, irect_min);
        ibbox_max = min(ibbox_max, irect_max);

        ibbox_min[0] -= ibbox_min[0] % DimX;
        ibbox_min[1] -= ibbox_min[1] % DimY;
        ibbox_max[0] += ((ibbox_max[0] + 1) % DimX) ? (DimX - (ibbox_max[0] - 1) % DimX) : 0;
        ibbox_max[1] += ((ibbox_max[1] + 1) % DimY) ? (DimY - (ibbox_max[1] - 1) % DimY) : 0;

        const simd_fvec2 d01 = t0 - t1, d12 = t1 - t2, d20 = t2 - t0;

        float area = d01[0] * d20[1] - d20[0] * d01[1];
        if (area < FLT_EPS) continue;

        float inv_area = 1.0f / area;

        for (int y = ibbox_min[1]; y <= ibbox_max[1]; y += DimY) {
            for (int x = ibbox_min[0]; x <= ibbox_max[0]; x += DimX) {
                simd_ivec<S> ixx = x + off_x, iyy = simd_ivec<S>(y) + off_y;

                int i = ((y - r.y) / DimY) * (r.w / DimX) + (x - r.x) / DimX;
                auto &out_ray = out_rays[i];
                auto &out_inter = out_inters[i];

                simd_ivec<S> index = iyy * w + ixx;
                const int hi = (iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

                simd_ivec<S> hash_val = hash(index);
                simd_fvec<S> rxx = construct_float(hash_val);
                simd_fvec<S> ryy = construct_float(hash(hash_val));

                for (int i = 0; i < S; i++) {
                    float _unused;
                    rxx[i] = std::modf(halton[hi + 0] + rxx[i], &_unused);
                    ryy[i] = std::modf(halton[hi + 1] + ryy[i], &_unused);
                }

                simd_fvec<S> fxx = (simd_fvec<S>)ixx + rxx,
                             fyy = (simd_fvec<S>)iyy + ryy;

                simd_fvec<S> u = d01[0] * (fyy - t0[1]) - d01[1] * (fxx - t0[0]),
                             v = d12[0] * (fyy - t1[1]) - d12[1] * (fxx - t1[0]),
                             w = d20[0] * (fyy - t2[1]) - d20[1] * (fxx - t2[0]);

                auto fmask = (u >= -FLT_EPS) & (v >= -FLT_EPS) & (w >= -FLT_EPS);
                const auto &imask = reinterpret_cast<const simd_ivec<S> &>(fmask);

                if (imask.not_all_zeros()) {
                    u *= inv_area; v *= inv_area; w *= inv_area;

                    simd_fvec<S> _p[3] = { v0.p[0] * v + v1.p[0] * w + v2.p[0] * u,
                                           v0.p[1] * v + v1.p[1] * w + v2.p[1] * u,
                                           v0.p[2] * v + v1.p[2] * w + v2.p[2] * u },
                                 _n[3] = { v0.n[0] * v + v1.n[0] * w + v2.n[0] * u,
                                           v0.n[1] * v + v1.n[1] * w + v2.n[1] * u,
                                           v0.n[2] * v + v1.n[2] * w + v2.n[2] * u };

                    simd_fvec<S> p[3], n[3];

                    TransformPoint(_p, tr.xform, p);
                    TransformNormal(_n, tr.inv_xform, n);

                    ITERATE_3({ where(fmask, out_ray.o[i]) = p[i] + n[i]; })
                    ITERATE_3({ where(fmask, out_ray.d[i]) = -n[i]; })
                    where(fmask, out_ray.ior) = 1.0f;

                    out_inter.mask = out_inter.mask | imask;
                    where(imask, out_inter.prim_index) = tri;
                    where(imask, out_inter.obj_index) = obj_index;
                    where(fmask, out_inter.t) = 1.0f;
                    where(fmask, out_inter.u) = w;
                    where(fmask, out_inter.v) = u;
                }
            }
        }
    }
}


template <int S>
void Ray::NS::SortRays(ray_packet_t<S> *rays, simd_ivec<S> *ray_masks, int &secondary_rays_count, const float root_min[3], const float cell_size[3],
                       simd_ivec<S> *hash_values, int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp, uint32_t *skeleton) {
    // From "Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing" [2010]

    // compute Ray hash values
    for (int i = 0; i < secondary_rays_count; i++) {
        hash_values[i] = get_ray_hash(rays[i], ray_masks[i], root_min, cell_size);
    }

    // set head flags
    head_flags[0] = 1;
    for (int i = 1; i < secondary_rays_count * S; i++) {
        head_flags[i] = hash_values[i / S][i % S] != hash_values[(i - 1) / S][(i - 1) % S];
    }

    int chunks_count = 0;

    {   // perform exclusive scan on head flags
        uint32_t cur_sum = 0;
        for (int i = 0; i < secondary_rays_count * S; i++) {
            scan_values[i] = cur_sum;
            cur_sum += head_flags[i];
        }
        chunks_count = cur_sum;
    }

    // init Ray chunks hash and base index
    for (int i = 0; i < secondary_rays_count * S; i++) {
        if (head_flags[i]) {
            chunks[scan_values[i]].hash = reinterpret_cast<const uint32_t &>(hash_values[i / S][i % S]);
            chunks[scan_values[i]].base = (uint32_t)i;
        }
    }

    // init Ray chunks size
    for (int i = 0; i < chunks_count - 1; i++) {
        chunks[i].size = chunks[i + 1].base - chunks[i].base;
    }
    chunks[chunks_count - 1].size = (uint32_t)secondary_rays_count * S - chunks[chunks_count - 1].base;

    radix_sort(&chunks[0], &chunks[0] + chunks_count, &chunks_temp[0]);

    {   // perform exclusive scan on chunks size
        uint32_t cur_sum = 0;
        for (int i = 0; i < chunks_count; i++) {
            scan_values[i] = cur_sum;
            cur_sum += chunks[i].size;
        }
    }

    std::fill(&skeleton[0], &skeleton[0] + secondary_rays_count * S, 1);
    std::fill(&head_flags[0], &head_flags[0] + secondary_rays_count * S, 0);

    // init skeleton and head flags array
    for (int i = 0; i < chunks_count; i++) {
        skeleton[scan_values[i]] = chunks[i].base;
        head_flags[scan_values[i]] = 1;
    }

    {   // perform a segmented scan on skeleton array
        uint32_t cur_sum = 0;
        for (int i = 0; i < secondary_rays_count * S; i++) {
            if (head_flags[i]) cur_sum = 0;
            cur_sum += skeleton[i];
            scan_values[i] = cur_sum;
        }
    }

    {   // reorder rays
        int j, k;
        for (int i = 0; i < secondary_rays_count * S; i++) {
            while (i != (j = scan_values[i])) {
                k = scan_values[j];

                {
                    int jj = j / S, _jj = j % S,
                        kk = k / S, _kk = k % S;

                    std::swap(hash_values[jj][_jj], hash_values[kk][_kk]);

                    std::swap(rays[jj].d[0][_jj], rays[kk].d[0][_kk]);
                    std::swap(rays[jj].d[1][_jj], rays[kk].d[1][_kk]);
                    std::swap(rays[jj].d[2][_jj], rays[kk].d[2][_kk]);

                    std::swap(rays[jj].o[0][_jj], rays[kk].o[0][_kk]);
                    std::swap(rays[jj].o[1][_jj], rays[kk].o[1][_kk]);
                    std::swap(rays[jj].o[2][_jj], rays[kk].o[2][_kk]);

                    std::swap(rays[jj].c[0][_jj], rays[kk].c[0][_kk]);
                    std::swap(rays[jj].c[1][_jj], rays[kk].c[1][_kk]);
                    std::swap(rays[jj].c[2][_jj], rays[kk].c[2][_kk]);

                    std::swap(rays[jj].do_dx[0][_jj], rays[kk].do_dx[0][_kk]);
                    std::swap(rays[jj].do_dx[1][_jj], rays[kk].do_dx[1][_kk]);
                    std::swap(rays[jj].do_dx[2][_jj], rays[kk].do_dx[2][_kk]);

                    std::swap(rays[jj].dd_dx[0][_jj], rays[kk].dd_dx[0][_kk]);
                    std::swap(rays[jj].dd_dx[1][_jj], rays[kk].dd_dx[1][_kk]);
                    std::swap(rays[jj].dd_dx[2][_jj], rays[kk].dd_dx[2][_kk]);

                    std::swap(rays[jj].do_dy[0][_jj], rays[kk].do_dy[0][_kk]);
                    std::swap(rays[jj].do_dy[1][_jj], rays[kk].do_dy[1][_kk]);
                    std::swap(rays[jj].do_dy[2][_jj], rays[kk].do_dy[2][_kk]);

                    std::swap(rays[jj].dd_dy[0][_jj], rays[kk].dd_dy[0][_kk]);
                    std::swap(rays[jj].dd_dy[1][_jj], rays[kk].dd_dy[1][_kk]);
                    std::swap(rays[jj].dd_dy[2][_jj], rays[kk].dd_dy[2][_kk]);

                    std::swap(rays[jj].xy[_jj], rays[kk].xy[_kk]);

                    std::swap(ray_masks[jj][_jj], ray_masks[kk][_kk]);
                }

                std::swap(scan_values[i], scan_values[j]);
            }
        }
    }

    // remove non-active rays
    while (secondary_rays_count && ray_masks[secondary_rays_count - 1].all_zeros()) {
        secondary_rays_count--;
    }
}

template <int S>
bool Ray::NS::IntersectTris(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t<S> &out_inter) {
    hit_data_t<S> inter = { Uninitialize };
    inter.mask = { 0 };
    inter.obj_index = { reinterpret_cast<const int&>(obj_index) };
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        _IntersectTri(r, ray_mask, tris[i], i, inter);
    }

    const auto &fmask = reinterpret_cast<const simd_fvec<S>&>(inter.mask);

    out_inter.mask = out_inter.mask | inter.mask;

    where(inter.mask, out_inter.obj_index) = inter.obj_index;
    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(fmask, out_inter.u) = inter.u;
    where(fmask, out_inter.v) = inter.v;

    return inter.mask.not_all_zeros();
}

template <int S>
bool Ray::NS::IntersectTris(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t<S> &out_inter) {
    hit_data_t<S> inter = { Uninitialize };
    inter.mask = { 0 };
    inter.obj_index = { reinterpret_cast<const int&>(obj_index) };
    inter.t = out_inter.t;

    for (uint32_t i = 0; i < num_tris; i++) {
        uint32_t index = indices[i];
        _IntersectTri(r, ray_mask, tris[index], index, inter);
    }

    const auto &fmask = reinterpret_cast<const simd_fvec<S>&>(inter.mask);

    out_inter.mask = out_inter.mask | inter.mask;

    where(inter.mask, out_inter.obj_index) = inter.obj_index;
    where(inter.mask, out_inter.prim_index) = inter.prim_index;

    out_inter.t = inter.t; // already contains min value

    where(fmask, out_inter.u) = inter.u;
    where(fmask, out_inter.v) = inter.v;

    return inter.mask.not_all_zeros();
}

template <int S>
bool Ray::NS::Traverse_MacroTree_Stackless_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t root_index,
                                               const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                               const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3];
    safe_invert(r.d, inv_d);

    TraversalState<S> st;

    st.queue[0].mask = ray_mask;

    st.queue[0].src = FromSibling;
    st.queue[0].cur = root_index;

    if (!is_leaf_node(nodes[root_index])) {
        st.queue[0].src = FromParent;
        st.select_near_child(r, nodes[root_index]);
    }

    while (st.index < st.num) {
        uint32_t &cur = st.queue[st.index].cur;
        eTraversalSource &src = st.queue[st.index].src;

        switch (src) {
        case FromChild:
            if (cur == root_index || cur == 0xffffffff) {
                st.index++;
                continue;
            }
            if (cur == near_child(r, st.queue[st.index].mask, nodes[nodes[cur].parent])) {
                cur = other_child(nodes[nodes[cur].parent], cur);
                src = FromSibling;
            } else {
                cur = nodes[cur].parent;
                src = FromChild;
            }
            break;
        case FromSibling: {
            auto mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                auto mask2 = and_not(mask1, st.queue[st.index].mask);
                if (mask2.not_all_zeros()) {
                    st.queue[st.num].cur = nodes[cur].parent;
                    st.queue[st.num].mask = mask2;
                    st.queue[st.num].src = FromChild;
                    st.num++;
                    st.queue[st.index].mask = mask1;
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                        const auto &mi = mesh_instances[mi_indices[i]];
                        const auto &m = meshes[mi.mesh_index];
                        const auto &tr = transforms[mi.tr_index];

                        auto bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max) & st.queue[st.index].mask;
                        if (bbox_mask.all_zeros()) continue;

                        ray_packet_t<S> _r = TransformRay(r, tr.inv_xform);

                        res |= Traverse_MicroTree_Stackless_CPU(_r, bbox_mask, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter);
                    }

                    cur = nodes[cur].parent;
                    src = FromChild;
                } else {
                    src = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        case FromParent: {
            auto mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                cur = other_child(nodes[nodes[cur].parent], cur);
                src = FromSibling;
            } else {
                auto mask2 = and_not(mask1, st.queue[st.index].mask);
                if (mask2.not_all_zeros()) {
                    st.queue[st.num].cur = other_child(nodes[nodes[cur].parent], cur);
                    st.queue[st.num].mask = mask2;
                    st.queue[st.num].src = FromSibling;
                    st.num++;
                    st.queue[st.index].mask = mask1;
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                        const auto &mi = mesh_instances[mi_indices[i]];
                        const auto &m = meshes[mi.mesh_index];
                        const auto &tr = transforms[mi.tr_index];

                        auto bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max) & st.queue[st.index].mask;
                        if (bbox_mask.all_zeros()) continue;

                        ray_packet_t<S> _r = TransformRay(r, tr.inv_xform);

                        res |= Traverse_MicroTree_Stackless_CPU(_r, bbox_mask, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter);
                    }

                    cur = other_child(nodes[nodes[cur].parent], cur);
                    src = FromSibling;
                } else {
                    src = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        }
    }
    return res;
}

template <int S>
bool Ray::NS::Traverse_MicroTree_Stackless_CPU(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t root_index,
                                               const tri_accel_t *tris, const uint32_t *indices, int obj_index, hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3];
    safe_invert(r.d, inv_d);

    TraversalState<S> st;

    st.queue[0].mask = ray_mask;

    st.queue[0].src = FromSibling;
    st.queue[0].cur = root_index;

    if (!is_leaf_node(nodes[root_index])) {
        st.queue[0].src = FromParent;
        st.select_near_child(r, nodes[root_index]);
    }

    while (st.index < st.num) {
        uint32_t &cur = st.queue[st.index].cur;
        eTraversalSource &src = st.queue[st.index].src;

        switch (src) {
        case FromChild:
            if (cur == root_index || cur == 0xffffffff) {
                st.index++;
                continue;
            }
            if (cur == near_child(r, st.queue[st.index].mask, nodes[nodes[cur].parent])) {
                cur = other_child(nodes[nodes[cur].parent], cur);
                src = FromSibling;
            } else {
                cur = nodes[cur].parent;
                src = FromChild;
            }
            break;
        case FromSibling: {
            auto mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                cur = nodes[cur].parent;
                src = FromChild;
            } else {
                auto mask2 = and_not(mask1, st.queue[st.index].mask);
                if (mask2.not_all_zeros()) {
                    st.queue[st.num].cur = nodes[cur].parent;
                    st.queue[st.num].mask = mask2;
                    st.queue[st.num].src = FromChild;
                    st.num++;
                    st.queue[st.index].mask = mask1;
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    res |= IntersectTris(r, st.queue[st.index].mask, tris, &indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter);

                    cur = nodes[cur].parent;
                    src = FromChild;
                } else {
                    src = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        case FromParent: {
            auto mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                cur = other_child(nodes[nodes[cur].parent], cur);
                src = FromSibling;
            } else {
                auto mask2 = and_not(mask1, st.queue[st.index].mask);
                if (mask2.not_all_zeros()) {
                    st.queue[st.num].cur = other_child(nodes[nodes[cur].parent], cur);
                    st.queue[st.num].mask = mask2;
                    st.queue[st.num].src = FromSibling;
                    st.num++;
                    st.queue[st.index].mask = mask1;
                }

                if (is_leaf_node(nodes[cur])) {
                    // process leaf
                    res |= IntersectTris(r, st.queue[st.index].mask, tris, &indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter);

                    cur = other_child(nodes[nodes[cur].parent], cur);
                    src = FromSibling;
                } else {
                    src = FromParent;
                    st.select_near_child(r, nodes[cur]);
                }
            }
        }
        break;
        }
    }
    return res;
}

template <int S>
bool Ray::NS::Traverse_MacroTree_WithStack(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                           const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                           const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3];
    safe_invert(r.d, inv_d);

    TraversalStateStack<S> st;

    st.queue[0].mask = ray_mask;
    st.queue[0].stack_size = 0;
    st.queue[0].stack[st.queue[0].stack_size++] = node_index;

    while (st.index < st.num) {
        uint32_t *stack = &st.queue[st.index].stack[0];
        uint32_t &stack_size = st.queue[st.index].stack_size;
        while (stack_size) {
            uint32_t cur = stack[--stack_size];

            auto mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                continue;
            }

            auto mask2 = and_not(mask1, st.queue[st.index].mask);
            if (mask2.not_all_zeros()) {
                st.queue[st.num].mask = mask2;
                st.queue[st.num].stack_size = stack_size;
                memcpy(st.queue[st.num].stack, st.queue[st.index].stack, sizeof(uint32_t) * stack_size);
                st.num++;
                st.queue[st.index].mask = mask1;
            }

            if (!is_leaf_node(nodes[cur])) {
                st.push_children(r, nodes[cur]);
            } else {
                for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                    const auto &mi = mesh_instances[mi_indices[i]];
                    const auto &m = meshes[mi.mesh_index];
                    const auto &tr = transforms[mi.tr_index];

                    auto bbox_mask = bbox_test(r.o, inv_d, inter.t, mi.bbox_min, mi.bbox_max) & st.queue[st.index].mask;
                    if (bbox_mask.all_zeros()) continue;

                    ray_packet_t<S> _r = TransformRay(r, tr.inv_xform);

                    res |= Traverse_MicroTree_WithStack(_r, bbox_mask, nodes, m.node_index, tris, tri_indices, (int)mi_indices[i], inter);
                }
            }
        }
        st.index++;
    }

    return res;
}

template <int S>
bool Ray::NS::Traverse_MicroTree_WithStack(const ray_packet_t<S> &r, const simd_ivec<S> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                           const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t<S> &inter) {
    bool res = false;

    simd_fvec<S> inv_d[3];
    safe_invert(r.d, inv_d);

    TraversalStateStack<S> st;

    st.queue[0].mask = ray_mask;
    st.queue[0].stack_size = 0;
    st.queue[0].stack[st.queue[0].stack_size++] = node_index;

    while (st.index < st.num) {
        uint32_t *stack = &st.queue[st.index].stack[0];
        uint32_t &stack_size = st.queue[st.index].stack_size;
        while (stack_size) {
            uint32_t cur = stack[--stack_size];

            auto mask1 = bbox_test(r.o, inv_d, inter.t, nodes[cur]) & st.queue[st.index].mask;

            if (mask1.all_zeros()) {
                continue;
            }

            auto mask2 = and_not(mask1, st.queue[st.index].mask);
            if (mask2.not_all_zeros()) {
                st.queue[st.num].mask = mask2;
                st.queue[st.num].stack_size = stack_size;
                memcpy(st.queue[st.num].stack, st.queue[st.index].stack, sizeof(uint32_t) * stack_size);
                st.num++;
                st.queue[st.index].mask = mask1;
            }

            if (!is_leaf_node(nodes[cur])) {
                st.push_children(r, nodes[cur]);
            } else {
                res |= IntersectTris(r, st.queue[st.index].mask, tris, &tri_indices[nodes[cur].prim_index], nodes[cur].prim_count, obj_index, inter);
            }
        }
        st.index++;
    }

    return res;
}

template <int S>
force_inline Ray::NS::ray_packet_t<S> Ray::NS::TransformRay(const ray_packet_t<S> &r, const float *xform) {
    ray_packet_t<S> _r = r;

    _r.o[0] = r.o[0] * xform[0] + r.o[1] * xform[4] + r.o[2] * xform[8] + xform[12];
    _r.o[1] = r.o[0] * xform[1] + r.o[1] * xform[5] + r.o[2] * xform[9] + xform[13];
    _r.o[2] = r.o[0] * xform[2] + r.o[1] * xform[6] + r.o[2] * xform[10] + xform[14];

    _r.d[0] = r.d[0] * xform[0] + r.d[1] * xform[4] + r.d[2] * xform[8];
    _r.d[1] = r.d[0] * xform[1] + r.d[1] * xform[5] + r.d[2] * xform[9];
    _r.d[2] = r.d[0] * xform[2] + r.d[1] * xform[6] + r.d[2] * xform[10];

    return _r;
}

template <int S>
void Ray::NS::TransformPoint(const simd_fvec<S> p[3], const float *xform, simd_fvec<S> out_p[3]) {
    out_p[0] = xform[0] * p[0] + xform[4] * p[1] + xform[8] * p[2] + xform[12];
    out_p[1] = xform[1] * p[0] + xform[5] * p[1] + xform[9] * p[2] + xform[13];
    out_p[2] = xform[2] * p[0] + xform[6] * p[1] + xform[10] * p[2] + xform[14];
}

template <int S>
void Ray::NS::TransformNormal(const simd_fvec<S> n[3], const float *inv_xform, simd_fvec<S> out_n[3]) {
    out_n[0] = n[0] * inv_xform[0] + n[1] * inv_xform[1] + n[2] * inv_xform[2];
    out_n[1] = n[0] * inv_xform[4] + n[1] * inv_xform[5] + n[2] * inv_xform[6];
    out_n[2] = n[0] * inv_xform[8] + n[1] * inv_xform[9] + n[2] * inv_xform[10];
}

template <int S>
void Ray::NS::TransformUVs(const simd_fvec<S> uvs[2], float sx, float sy, const texture_t &t, const simd_ivec<S> &mip_level, const simd_ivec<S> &mask, simd_fvec<S> out_res[2]) {
    simd_ivec<S> ipos[2];

    ITERATE(S, { ipos[0][i] = (int)t.pos[mip_level[i]][0];
                 ipos[1][i] = (int)t.pos[mip_level[i]][1]; });

    simd_ivec<S> isize[2] = { (int)t.size[0], (int)t.size[1] };

    out_res[0] = (static_cast<simd_fvec<S>>(ipos[0]) + (uvs[0] - floor(uvs[0])) * static_cast<simd_fvec<S>>(isize[0] >> mip_level) + 1.0f) / sx;
    out_res[1] = (static_cast<simd_fvec<S>>(ipos[1]) + (uvs[1] - floor(uvs[1])) * static_cast<simd_fvec<S>>(isize[1] >> mip_level) + 1.0f) / sy;
}

template <int S>
void Ray::NS::SampleNearest(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<S> uvs[2], const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    simd_ivec<S> _lod = (simd_ivec<S>)lod;

    simd_fvec<S> _uvs[2];
    TransformUVs(uvs, atlas.size_x(), atlas.size_y(), t, _lod, mask, _uvs);

    where(_lod > MAX_MIP_LEVEL, _lod) = MAX_MIP_LEVEL;

    for (int i = 0; i < S; i++) {
        if (!mask[i]) continue;

        int page = t.page[_lod[i]];

        const auto &pix = atlas.Get(page, _uvs[0][i], _uvs[1][i]);

        out_rgba[0][i] = (float)pix.r;
        out_rgba[1][i] = (float)pix.g;
        out_rgba[2][i] = (float)pix.b;
        out_rgba[3][i] = (float)pix.a;
    }

    const float k = 1.0f / 255.0f;
    ITERATE_4({ out_rgba[i] *= k; })
}

template <int S>
void Ray::NS::SampleBilinear(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<S> uvs[2], const simd_ivec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    simd_fvec<S> _uvs[2];
    TransformUVs(uvs, atlas.size_x(), atlas.size_y(), t, lod, mask, _uvs);

    _uvs[0] = _uvs[0] * atlas.size_x() - 0.5f;
    _uvs[1] = _uvs[1] * atlas.size_y() - 0.5f;

    simd_fvec<S> k[2] = { _uvs[0] - floor(_uvs[0]), _uvs[1] - floor(_uvs[1]) };

    simd_fvec<S> p0[4], p1[4];

    for (int i = 0; i < S; i++) {
        if (!mask[i]) continue;

        int page = t.page[lod[i]];

        const auto &p00 = atlas.Get(page, int(_uvs[0][i]), int(_uvs[1][i]));
        const auto &p01 = atlas.Get(page, int(_uvs[0][i] + 1), int(_uvs[1][i]));
        const auto &p10 = atlas.Get(page, int(_uvs[0][i]), int(_uvs[1][i] + 1));
        const auto &p11 = atlas.Get(page, int(_uvs[0][i] + 1), int(_uvs[1][i] + 1));

        p0[0][i] = p01.r * k[0][i] + p00.r * (1 - k[0][i]);
        p0[1][i] = p01.g * k[0][i] + p00.g * (1 - k[0][i]);
        p0[2][i] = p01.b * k[0][i] + p00.b * (1 - k[0][i]);
        p0[3][i] = p01.a * k[0][i] + p00.a * (1 - k[0][i]);

        p1[0][i] = p11.r * k[0][i] + p10.r * (1 - k[0][i]);
        p1[1][i] = p11.g * k[0][i] + p10.g * (1 - k[0][i]);
        p1[2][i] = p11.b * k[0][i] + p10.b * (1 - k[0][i]);
        p1[3][i] = p11.a * k[0][i] + p10.a * (1 - k[0][i]);
    }

    const float _k = 1.0f / 255.0f;

    out_rgba[0] = (p1[0] * k[1] + p0[0] * (1.0f - k[1])) * _k;
    out_rgba[1] = (p1[1] * k[1] + p0[1] * (1.0f - k[1])) * _k;
    out_rgba[2] = (p1[2] * k[1] + p0[2] * (1.0f - k[1])) * _k;
    out_rgba[3] = (p1[3] * k[1] + p0[3] * (1.0f - k[1])) * _k;
}

template <int S>
void Ray::NS::SampleBilinear(const Ref::TextureAtlas &atlas, const simd_fvec<S> uvs[2], const simd_ivec<S> &page, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    simd_fvec<S> _p00[4], _p01[4], _p10[4], _p11[4];

    const simd_fvec<S> k[2] = { uvs[0] - floor(uvs[0]), uvs[1] - floor(uvs[1]) };

    for (int i = 0; i < S; i++) {
        if (!mask[i]) continue;

        const auto &p00 = atlas.Get(page[i], int(uvs[0][i] + 0), int(uvs[1][i] + 0));
        const auto &p01 = atlas.Get(page[i], int(uvs[0][i] + 1), int(uvs[1][i] + 0));
        const auto &p10 = atlas.Get(page[i], int(uvs[0][i] + 0), int(uvs[1][i] + 1));
        const auto &p11 = atlas.Get(page[i], int(uvs[0][i] + 1), int(uvs[1][i] + 1));

        _p00[0][i] = to_norm_float(p00.r);
        _p00[1][i] = to_norm_float(p00.g);
        _p00[2][i] = to_norm_float(p00.b);
        _p00[3][i] = to_norm_float(p00.a);

        _p01[0][i] = to_norm_float(p01.r);
        _p01[1][i] = to_norm_float(p01.g);
        _p01[2][i] = to_norm_float(p01.b);
        _p01[3][i] = to_norm_float(p01.a);

        _p10[0][i] = to_norm_float(p10.r);
        _p10[1][i] = to_norm_float(p10.g);
        _p10[2][i] = to_norm_float(p10.b);
        _p10[3][i] = to_norm_float(p10.a);

        _p11[0][i] = to_norm_float(p11.r);
        _p11[1][i] = to_norm_float(p11.g);
        _p11[2][i] = to_norm_float(p11.b);
        _p11[3][i] = to_norm_float(p11.a);
    }

    const simd_fvec<S> p0X[4] = { _p01[0] * k[0] + _p00[0] * (1 - k[0]),
                                  _p01[1] * k[0] + _p00[1] * (1 - k[0]),
                                  _p01[2] * k[0] + _p00[2] * (1 - k[0]),
                                  _p01[3] * k[0] + _p00[3] * (1 - k[0]) };
    const simd_fvec<S> p1X[4] = { _p11[0] * k[0] + _p10[0] * (1 - k[0]),
                                  _p11[1] * k[0] + _p10[1] * (1 - k[0]),
                                  _p11[2] * k[0] + _p10[2] * (1 - k[0]),
                                  _p11[3] * k[0] + _p10[3] * (1 - k[0]) };

    out_rgba[0] = p1X[0] * k[1] + p0X[0] * (1.0f - k[1]);
    out_rgba[1] = p1X[1] * k[1] + p0X[1] * (1.0f - k[1]);
    out_rgba[2] = p1X[2] * k[1] + p0X[2] * (1.0f - k[1]);
    out_rgba[3] = p1X[3] * k[1] + p0X[3] * (1.0f - k[1]);
}

template <int S>
void Ray::NS::SampleTrilinear(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<S> uvs[2], const simd_fvec<S> &lod, const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    simd_fvec<S> col1[4];
    SampleBilinear(atlas, t, uvs, (simd_ivec<S>)floor(lod), mask, col1);
    simd_fvec<S> col2[4];
    SampleBilinear(atlas, t, uvs, (simd_ivec<S>)ceil(lod), mask, col2);

    simd_fvec<S> k = lod - floor(lod);

    ITERATE_4({ out_rgba[i] = col1[i] * (1.0f - k) + col2[i] * k; })
}

template <int S>
void Ray::NS::SampleAnisotropic(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<S> uvs[2], const simd_fvec<S> duv_dx[2], const simd_fvec<S> duv_dy[2], const simd_ivec<S> &mask, simd_fvec<S> out_rgba[4]) {
    simd_fvec<S> _duv_dx[2] = { abs(duv_dx[0] * (float)t.size[0]),
                                abs(duv_dx[1] * (float)t.size[1]) };

    simd_fvec<S> l1 = sqrt(_duv_dx[0] * _duv_dx[0] + _duv_dx[1] * _duv_dx[1]);

    simd_fvec<S> _duv_dy[2] = { abs(duv_dy[0] * (float)t.size[0]),
                                abs(duv_dy[1] * (float)t.size[1]) };

    simd_fvec<S> l2 = sqrt(_duv_dy[0] * _duv_dy[0] + _duv_dy[1] * _duv_dy[1]);

    simd_fvec<S> lod,
                 k = l2 / l1,
                 step[2] = { duv_dx[0], duv_dx[1] };

    ITERATE(S, { lod[i] = log2(std::min(_duv_dy[0][i], _duv_dy[1][i])); })

    auto _mask = l1 <= l2;
    where(_mask, k) = l1 / l2;
    where(_mask, step[0]) = duv_dy[0];
    where(_mask, step[1]) = duv_dy[1];

    ITERATE(S, {
        if (reinterpret_cast<const simd_ivec<S>&>(_mask)[i]) {
            lod[i] = log2(std::min(_duv_dx[0][i], _duv_dx[1][i]));
        }
    })

    where(lod < 0.0f, lod) = 0.0f;
    where(lod > (float)MAX_MIP_LEVEL, lod) = (float)MAX_MIP_LEVEL;

    const auto imask = mask == 0;
    where(reinterpret_cast<const simd_fvec<S>&>(imask), lod) = 0.0f;

    simd_fvec<S> _uvs[2] = { uvs[0] - step[0] * 0.5f, uvs[1] - step[1] * 0.5f };

    auto num = (simd_ivec<S>)(2.0f / k);
    where(num < 1, num) = 1;
    where(num > 4, num) = 4;

    step[0] /= (simd_fvec<S>)num;
    step[1] /= (simd_fvec<S>)num;

    ITERATE_4({ out_rgba[i] = 0.0f; })

    simd_ivec<S> lod1 = (simd_ivec<S>)floor(lod);
    simd_ivec<S> lod2 = (simd_ivec<S>)ceil(lod);

    simd_ivec<S> page1, page2;
    simd_fvec<S> pos1[2], pos2[2], size1[2], size2[2];

    ITERATE(S, {
        page1[i] = t.page[lod1[i]];
        page2[i] = t.page[lod2[i]];

        pos1[0][i] = t.pos[lod1[i]][0] + 0.5f;
        pos1[1][i] = t.pos[lod1[i]][1] + 0.5f;
        pos2[0][i] = t.pos[lod2[i]][0] + 0.5f;
        pos2[1][i] = t.pos[lod2[i]][1] + 0.5f;
        size1[0][i] = float(t.size[0] >> lod1[i]);
        size1[1][i] = float(t.size[1] >> lod1[i]);
        size2[0][i] = float(t.size[0] >> lod2[i]);
        size2[1][i] = float(t.size[1] >> lod2[i]);
    })

    const simd_fvec<S> kz = lod - floor(lod);

    auto kz_big_enough = kz > 0.0001f;
    bool skip_z = reinterpret_cast<simd_ivec<S> &>(kz_big_enough).all_zeros();

    for (int j = 0; j < 4; j++) {
        auto imask = (num > j) & mask;
        if (imask.all_zeros()) break;

        const auto &fmask = reinterpret_cast<const simd_fvec<S>&>(imask);

        _uvs[0] = _uvs[0] - floor(_uvs[0]);
        _uvs[1] = _uvs[1] - floor(_uvs[1]);

        simd_fvec<S> col[4];

        simd_fvec<S> _uvs1[2] = { pos1[0] + _uvs[0] * size1[0], pos1[1] + _uvs[1] * size1[1] };
        SampleBilinear(atlas, _uvs1, page1, imask, col);
        ITERATE_4({ where(fmask, out_rgba[i]) = out_rgba[i] + (1.0f - kz) * col[i]; })

        if (!skip_z) {
            simd_fvec<S> _uvs2[2] = { pos2[0] + _uvs[0] * size2[0], pos2[1] + _uvs[1] * size2[1] };
            SampleBilinear(atlas, _uvs2, page2, imask, col);
            ITERATE_4({ where(fmask, out_rgba[i]) = out_rgba[i] + kz * col[i]; })
        }

        _uvs[0] = _uvs[0] + step[0];
        _uvs[1] = _uvs[1] + step[1];
    }

    const auto fnum = static_cast<simd_fvec<S>>(num);
    ITERATE_4({ out_rgba[i] /= fnum; })
}

template <int S>
void Ray::NS::SampleLatlong_RGBE(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<S> dir[3], const simd_ivec<S> &mask, simd_fvec<S> out_rgb[3]) {
    
    simd_fvec<S> theta, u = 0.0f;
    simd_fvec<S> r = sqrt(dir[0] * dir[0] + dir[2] * dir[2]);
    simd_fvec<S> y = clamp(dir[1], -1.0f, 1.0f);

    where(r > FLT_EPS, u) = clamp(dir[0] / r, -1.0f, 1.0f);

    ITERATE(S, {
        theta[i] = std::acos(y[i]) / PI;
        u[i] = 0.5f * std::acos(u[i]) / PI;
    })

    where(dir[2] < 0.0f, u) = 1.0f - u;

    simd_fvec<S> uvs[2] = { u * t.size[0] + float(t.pos[0][0]) + 1.0f,
                            theta * t.size[1] + float(t.pos[0][1]) + 1.0f };

    const simd_fvec<S> k[2] = { uvs[0] - floor(uvs[0]), uvs[1] - floor(uvs[1]) };

    simd_fvec<S> _p00[3], _p01[3], _p10[3], _p11[3];

    for (int i = 0; i < S; i++) {
        if (!mask[i]) continue;

        const auto &p00 = atlas.Get(t.page[0], int(uvs[0][i] + 0), int(uvs[1][i] + 0));
        const auto &p01 = atlas.Get(t.page[0], int(uvs[0][i] + 1), int(uvs[1][i] + 0));
        const auto &p10 = atlas.Get(t.page[0], int(uvs[0][i] + 0), int(uvs[1][i] + 1));
        const auto &p11 = atlas.Get(t.page[0], int(uvs[0][i] + 1), int(uvs[1][i] + 1));

        float f = std::pow(2.0f, float(p00.a) - 128.0f);
        _p00[0][i] = to_norm_float(p00.r) * f;
        _p00[1][i] = to_norm_float(p00.g) * f;
        _p00[2][i] = to_norm_float(p00.b) * f;

        f = std::pow(2.0f, float(p01.a) - 128.0f);
        _p01[0][i] = to_norm_float(p01.r) * f;
        _p01[1][i] = to_norm_float(p01.g) * f;
        _p01[2][i] = to_norm_float(p01.b) * f;

        f = std::pow(2.0f, float(p10.a) - 128.0f);
        _p10[0][i] = to_norm_float(p10.r) * f;
        _p10[1][i] = to_norm_float(p10.g) * f;
        _p10[2][i] = to_norm_float(p10.b) * f;

        f = std::pow(2.0f, float(p11.a) - 128.0f);
        _p11[0][i] = to_norm_float(p11.r) * f;
        _p11[1][i] = to_norm_float(p11.g) * f;
        _p11[2][i] = to_norm_float(p11.b) * f;
    }

    const simd_fvec<S> p0X[3] = { _p01[0] * k[0] + _p00[0] * (1 - k[0]),
                                  _p01[1] * k[0] + _p00[1] * (1 - k[0]),
                                  _p01[2] * k[0] + _p00[2] * (1 - k[0]) };
    const simd_fvec<S> p1X[3] = { _p11[0] * k[0] + _p10[0] * (1 - k[0]),
                                  _p11[1] * k[0] + _p10[1] * (1 - k[0]),
                                  _p11[2] * k[0] + _p10[2] * (1 - k[0]) };

    out_rgb[0] = p1X[0] * k[1] + p0X[0] * (1.0f - k[1]);
    out_rgb[1] = p1X[1] * k[1] + p0X[1] * (1.0f - k[1]);
    out_rgb[2] = p1X[2] * k[1] + p0X[2] * (1.0f - k[1]);
}

template <int S>
void Ray::NS::ComputeDirectLighting(const simd_fvec<S> P[3], const simd_fvec<S> N[3], const simd_fvec<S> B[3], const simd_fvec<S> plane_N[3],
                                    const float *halton, const int hi, const simd_fvec<S> &rand_offset, const simd_fvec<S> &rand_offset2,
                                    const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                    const uint32_t *vtx_indices, const vertex_t *vertices, const bvh_node_t *nodes, uint32_t node_index,
                                    const tri_accel_t *tris, const uint32_t *tri_indices, const light_t *lights,
                                    const uint32_t *li_indices, uint32_t light_node_index, const simd_ivec<S> &ray_mask, simd_fvec<S> *out_col) {
    TraversalStateStack<S> st;

    st.queue[0].mask = ray_mask;
    st.queue[0].stack_size = 0;
    st.queue[0].stack[st.queue[0].stack_size++] = light_node_index;

    while (st.index < st.num) {
        uint32_t *stack = &st.queue[st.index].stack[0];
        uint32_t &stack_size = st.queue[st.index].stack_size;
        while (stack_size) {
            uint32_t cur = stack[--stack_size];

            auto mask1 = bbox_test(P, nodes[cur]) & st.queue[st.index].mask;
            if (mask1.all_zeros()) {
                continue;
            }

            auto mask2 = and_not(mask1, st.queue[st.index].mask);
            if (mask2.not_all_zeros()) {
                st.queue[st.num].mask = mask2;
                st.queue[st.num].stack_size = stack_size;
                memcpy(st.queue[st.num].stack, st.queue[st.index].stack, sizeof(uint32_t) * stack_size);
                st.num++;
                st.queue[st.index].mask = mask1;
            }

            if (!is_leaf_node(nodes[cur])) {
                st.push_children(nodes[cur]);
            } else {
                for (uint32_t i = nodes[cur].prim_index; i < nodes[cur].prim_index + nodes[cur].prim_count; i++) {
                    const light_t &l = lights[li_indices[i]];

                    simd_fvec<S> L[3] = { { P[0] - l.pos[0] }, { P[1] - l.pos[1] }, { P[2] - l.pos[2] } };
                    simd_fvec<S> distance = length(L);
                    simd_fvec<S> d = max(distance - l.radius, simd_fvec<S>{ 0.0f });
                    ITERATE_3({ L[i] /= distance; })

                    simd_fvec<S> V[3], TT[3], BB[3];

                    cross(L, B, TT);
                    cross(L, TT, BB);

                    for (int i = 0; i < S; i++) {
                        if (!mask1[i]) continue;

                        float _unused;
                        const float z = std::modf(halton[hi + 0] + rand_offset[i], &_unused);

                        const float dir = std::sqrt(z);
                        const float phi = 2 * PI * std::modf(halton[hi + 1] + rand_offset2[i], &_unused);

                        const float cos_phi = std::cos(phi);
                        const float sin_phi = std::sin(phi);

                        V[0][i] = dir * sin_phi * BB[0][i] + std::sqrt(1.0f - dir) * L[0][i] + dir * cos_phi * TT[0][i];
                        V[1][i] = dir * sin_phi * BB[1][i] + std::sqrt(1.0f - dir) * L[1][i] + dir * cos_phi * TT[1][i];
                        V[2][i] = dir * sin_phi * BB[2][i] + std::sqrt(1.0f - dir) * L[2][i] + dir * cos_phi * TT[2][i];
                    }

                    ITERATE_3({ L[i] = l.pos[i] + V[i] * l.radius - P[i]; })
                    normalize(L);

                    simd_fvec<S> denom = d / l.radius + 1.0f;
                    simd_fvec<S> atten = 1.0f / (denom * denom);

                    atten = (atten - LIGHT_ATTEN_CUTOFF / l.brightness) / (1.0f - LIGHT_ATTEN_CUTOFF);
                    atten = max(atten, simd_fvec<S>{ 0.0f });

                    simd_fvec<S> _dot1 = max(dot(L, N), simd_fvec<S>{ 0.0f });
                    simd_fvec<S> _dot2 = dot(L, l.dir);

                    auto fmask = reinterpret_cast<const simd_fvec<S> &>(mask1) & (_dot1 > FLT_EPS) & (_dot2 > l.spot) & (l.brightness * atten > FLT_EPS);
                    const auto &imask = reinterpret_cast<const simd_ivec<S> &>(fmask);
                    if (imask.not_all_zeros()) {
                        ray_packet_t<S> r;

                        r.o[0] = P[0] + HIT_BIAS * plane_N[0];
                        r.o[1] = P[1] + HIT_BIAS * plane_N[1];
                        r.o[2] = P[2] + HIT_BIAS * plane_N[2];

                        r.d[0] = L[0];
                        r.d[1] = L[1];
                        r.d[2] = L[2];

                        hit_data_t<S> sh_inter;
                        sh_inter.t = distance;
                        Traverse_MacroTree_WithStack(r, imask, nodes, node_index, mesh_instances, mi_indices, meshes, transforms, tris, tri_indices, sh_inter);

                        simd_fvec<S> _v = 1.0f;
                        where(reinterpret_cast<const simd_fvec<S>&>(sh_inter.mask), _v) = 0.0f;

                        where(fmask, out_col[0]) = out_col[0] + l.col[0] * _dot1 * _v * atten;
                        where(fmask, out_col[1]) = out_col[1] + l.col[1] * _dot1 * _v * atten;
                        where(fmask, out_col[2]) = out_col[2] + l.col[2] * _dot1 * _v * atten;
                    }
                }
            }
        }
        st.index++;
    }
}

template <int S>
void Ray::NS::ComputeDerivatives(const simd_fvec<S> I[3], const simd_fvec<S> &t, const simd_fvec<S> do_dx[3], const simd_fvec<S> do_dy[3], const simd_fvec<S> dd_dx[3], const simd_fvec<S> dd_dy[3],
                                 const simd_fvec<S> p1[3], const simd_fvec<S> p2[3], const simd_fvec<S> p3[3], const simd_fvec<S> n1[3], const simd_fvec<S> n2[3], const simd_fvec<S> n3[3],
                                 const simd_fvec<S> u1[2], const simd_fvec<S> u2[2], const simd_fvec<S> u3[2], const simd_fvec<S> plane_N[3], derivatives_t<S> &out_der) {
    // From 'Tracing Ray Differentials' [1999]

    simd_fvec<S> temp[3];

    simd_fvec<S> dot_I_N = -dot(I, plane_N);
    simd_fvec<S> inv_dot = 1.0f / dot_I_N;
    where(abs(dot_I_N) < FLT_EPS, inv_dot) = { 0.0f };

    ITERATE_3({ temp[i] = do_dx[i] + t * dd_dx[i]; })

    simd_fvec<S> dt_dx = -dot(temp, plane_N) * inv_dot;
    ITERATE_3({ out_der.do_dx[i] = temp[i] + dt_dx * I[i]; })
    ITERATE_3({ out_der.dd_dx[i] = dd_dx[i]; })

    ITERATE_3({ temp[i] = do_dy[i] + t * dd_dy[i]; })

    simd_fvec<S> dt_dy = -dot(temp, plane_N) * inv_dot;
    ITERATE_3({ out_der.do_dy[i] = temp[i] + dt_dy * I[i]; })
    ITERATE_3({ out_der.dd_dy[i] = dd_dy[i]; })

    // From 'Physically Based Rendering: ...' book

    simd_fvec<S> duv13[2] = { u1[0] - u3[0], u1[1] - u3[1] },
                 duv23[2] = { u2[0] - u3[0], u2[1] - u3[1] };
    simd_fvec<S> dp13[3] = { p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2] },
                 dp23[3] = { p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2] };

    simd_fvec<S> det_uv = duv13[0] * duv23[1] - duv13[1] * duv23[0];
    simd_fvec<S> inv_det_uv = 1.0f / det_uv;
    where(abs(det_uv) < FLT_EPS, inv_det_uv) = 0.0f;

    const simd_fvec<S> dpdu[3] = { (duv23[1] * dp13[0] - duv13[1] * dp23[0]) * inv_det_uv,
                                   (duv23[1] * dp13[1] - duv13[1] * dp23[1]) * inv_det_uv,
                                   (duv23[1] * dp13[2] - duv13[1] * dp23[2]) * inv_det_uv };
    const simd_fvec<S> dpdv[3] = { (-duv23[0] * dp13[0] + duv13[0] * dp23[0]) * inv_det_uv,
                                   (-duv23[0] * dp13[1] + duv13[0] * dp23[1]) * inv_det_uv,
                                   (-duv23[0] * dp13[2] + duv13[0] * dp23[2]) * inv_det_uv };

    simd_fvec<S> A[2][2] = { { dpdu[0], dpdu[1] },{ dpdv[0], dpdv[1] } };
    simd_fvec<S> Bx[2] = { out_der.do_dx[0], out_der.do_dx[1] };
    simd_fvec<S> By[2] = { out_der.do_dy[0], out_der.do_dy[1] };

    auto mask1 = abs(plane_N[0]) > abs(plane_N[1]) & abs(plane_N[0]) > abs(plane_N[2]);
    where(mask1, A[0][0]) = dpdu[1];
    where(mask1, A[0][1]) = dpdu[2];
    where(mask1, A[1][0]) = dpdv[1];
    where(mask1, A[1][1]) = dpdv[2];
    where(mask1, Bx[0]) = out_der.do_dx[1];
    where(mask1, Bx[1]) = out_der.do_dx[2];
    where(mask1, By[0]) = out_der.do_dy[1];
    where(mask1, By[1]) = out_der.do_dy[2];

    auto mask2 = abs(plane_N[1]) > abs(plane_N[0]) & abs(plane_N[1]) > abs(plane_N[2]);
    where(mask2, A[0][1]) = dpdu[2];
    where(mask2, A[1][1]) = dpdv[2];
    where(mask2, Bx[1]) = out_der.do_dx[2];
    where(mask2, By[1]) = out_der.do_dy[2];

    simd_fvec<S> det = A[0][0] * A[1][1] - A[1][0] * A[0][1];
    simd_fvec<S> inv_det = 1.0f / det;
    where(abs(det) < FLT_EPS, inv_det) = { 0.0f };

    ITERATE_2({ out_der.duv_dx[i] = (A[i][0] * Bx[0] - A[i][1] * Bx[1]) * inv_det; })
    ITERATE_2({ out_der.duv_dy[i] = (A[i][0] * By[0] - A[i][1] * By[1]) * inv_det; })

    // Derivative for normal

    simd_fvec<S> dn1[3] = { n1[0] - n3[0], n1[1] - n3[1], n1[2] - n3[2] },
                 dn2[3] = { n2[0] - n3[0], n2[1] - n3[1], n2[2] - n3[2] };
    simd_fvec<S> dndu[3] = { (duv23[1] * dn1[0] - duv13[1] * dn2[0]) * inv_det_uv,
                             (duv23[1] * dn1[1] - duv13[1] * dn2[1]) * inv_det_uv,
                             (duv23[1] * dn1[2] - duv13[1] * dn2[2]) * inv_det_uv };
    simd_fvec<S> dndv[3] = { (-duv23[0] * dn1[0] + duv13[0] * dn2[0]) * inv_det_uv,
                             (-duv23[0] * dn1[1] + duv13[0] * dn2[1]) * inv_det_uv,
                             (-duv23[0] * dn1[2] + duv13[0] * dn2[2]) * inv_det_uv };

    ITERATE_3({ out_der.dndx[i] = dndu[i] * out_der.duv_dx[0] + dndv[i] * out_der.duv_dx[1]; })
    ITERATE_3({ out_der.dndy[i] = dndu[i] * out_der.duv_dy[0] + dndv[i] * out_der.duv_dy[1]; })

    out_der.ddn_dx = dot(out_der.dd_dx, plane_N) + dot(I, out_der.dndx);
    out_der.ddn_dy = dot(out_der.dd_dy, plane_N) + dot(I, out_der.dndy);
}

template <int S>
void Ray::NS::ShadeSurface(const simd_ivec<S> &px_index, const pass_info_t &pi, const float *halton, const hit_data_t<S> &inter, const ray_packet_t<S> &ray,
                           const environment_t &env, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                           const mesh_t *meshes, const transform_t *transforms, const uint32_t *vtx_indices, const vertex_t *vertices,
                           const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                           const material_t *materials, const texture_t *textures, const Ref::TextureAtlas &tex_atlas,
                           const light_t *lights, const uint32_t *li_indices, uint32_t light_node_index, simd_fvec<S> out_rgba[4], simd_ivec<S> *out_secondary_masks, ray_packet_t<S> *out_secondary_rays, int *out_secondary_rays_count) {
    out_rgba[3] = { 1.0f };
    
    auto ino_hit = inter.mask ^ simd_ivec<S>(-1);
    if (ino_hit.not_all_zeros()) {
        simd_fvec<S> env_col[4] = { { 0.0f }, { 0.0f }, { 0.0f }, { 0.0f } };
        if (pi.should_add_environment()) {
            SampleLatlong_RGBE(tex_atlas, textures[env.env_map], ray.d, ino_hit, env_col);

            if (env.env_clamp > FLT_EPS) {
                ITERATE_3({ env_col[i] = min(env_col[i], simd_fvec<S>{ env.env_clamp }); })
            }
            env_col[3] = 1.0f;
        }

        auto fno_hit = reinterpret_cast<const simd_fvec<S>&>(ino_hit);

        where(fno_hit, out_rgba[0]) = ray.c[0] * env_col[0] * env.env_col[0];
        where(fno_hit, out_rgba[1]) = ray.c[1] * env_col[1] * env.env_col[1];
        where(fno_hit, out_rgba[2]) = ray.c[2] * env_col[2] * env.env_col[2];
        where(fno_hit, out_rgba[3]) = env_col[3];
    }
    
    if (inter.mask.all_zeros()) return;

    const auto *I = ray.d;
    const simd_fvec<S> P[3] = { ray.o[0] + inter.t * I[0], ray.o[1] + inter.t * I[1], ray.o[2] + inter.t * I[2] };

    const simd_fvec<S> w = 1.0f - inter.u - inter.v;

    simd_fvec<S> p1[3], p2[3], p3[3],
                 n1[3], n2[3], n3[3],
                 u1[2], u2[2], u3[2],
                 b1[3], b2[3], b3[3];

    simd_ivec<S> mat_index = { -1 };

    simd_fvec<S> inv_xform1[3], inv_xform2[3], inv_xform3[3];

    simd_fvec<S> N[3];

    for (int i = 0; i < S; i++) {
        if (ino_hit[i]) continue;

        const auto &v1 = vertices[vtx_indices[inter.prim_index[i] * 3 + 0]];
        const auto &v2 = vertices[vtx_indices[inter.prim_index[i] * 3 + 1]];
        const auto &v3 = vertices[vtx_indices[inter.prim_index[i] * 3 + 2]];

        p1[0][i] = v1.p[0]; p1[1][i] = v1.p[1]; p1[2][i] = v1.p[2];
        p2[0][i] = v2.p[0]; p2[1][i] = v2.p[1]; p2[2][i] = v2.p[2];
        p3[0][i] = v3.p[0]; p3[1][i] = v3.p[1]; p3[2][i] = v3.p[2];

        n1[0][i] = v1.n[0]; n1[1][i] = v1.n[1]; n1[2][i] = v1.n[2];
        n2[0][i] = v2.n[0]; n2[1][i] = v2.n[1]; n2[2][i] = v2.n[2];
        n3[0][i] = v3.n[0]; n3[1][i] = v3.n[1]; n3[2][i] = v3.n[2];

        u1[0][i] = v1.t[0][0]; u1[1][i] = v1.t[0][1];
        u2[0][i] = v2.t[0][0]; u2[1][i] = v2.t[0][1];
        u3[0][i] = v3.t[0][0]; u3[1][i] = v3.t[0][1];

        b1[0][i] = v1.b[0]; b1[1][i] = v1.b[1]; b1[2][i] = v1.b[2];
        b2[0][i] = v2.b[0]; b2[1][i] = v2.b[1]; b2[2][i] = v2.b[2];
        b3[0][i] = v3.b[0]; b3[1][i] = v3.b[1]; b3[2][i] = v3.b[2];

        const auto &tri = tris[inter.prim_index[i]];
        uint32_t mi = tri.mi;
        mat_index[i] = reinterpret_cast<const int&>(mi);

        float _plane_N[3];
        ExtractPlaneNormal(tri, _plane_N);

        N[0][i] = _plane_N[0];
        N[1][i] = _plane_N[1];
        N[2][i] = _plane_N[2];

        const auto *tr = &transforms[mesh_instances[inter.obj_index[i]].tr_index];

        inv_xform1[0][i] = tr->inv_xform[0]; inv_xform1[1][i] = tr->inv_xform[1]; inv_xform1[2][i] = tr->inv_xform[2];
        inv_xform2[0][i] = tr->inv_xform[4]; inv_xform2[1][i] = tr->inv_xform[5]; inv_xform2[2][i] = tr->inv_xform[6];
        inv_xform3[0][i] = tr->inv_xform[8]; inv_xform3[1][i] = tr->inv_xform[9]; inv_xform3[2][i] = tr->inv_xform[10];
    }

    simd_fvec<S> plane_N[3] = { dot(N, inv_xform1), dot(N, inv_xform2), dot(N, inv_xform3) };

    N[0] = n1[0] * w + n2[0] * inter.u + n3[0] * inter.v;
    N[1] = n1[1] * w + n2[1] * inter.u + n3[1] * inter.v;
    N[2] = n1[2] * w + n2[2] * inter.u + n3[2] * inter.v;

    simd_fvec<S> uvs[2] = { u1[0] * w + u2[0] * inter.u + u3[0] * inter.v,
                            u1[1] * w + u2[1] * inter.u + u3[1] * inter.v };

    derivatives_t<S> surf_der;
    ComputeDerivatives(I, inter.t, ray.do_dx, ray.do_dy, ray.dd_dx, ray.dd_dy, p1, p2, p3, n1, n2, n3, u1, u2, u3, plane_N, surf_der);

    ////////////////////////////////////////////////////////

    simd_fvec<S> B[3] = { b1[0] * w + b2[0] * inter.u + b3[0] * inter.v,
                          b1[1] * w + b2[1] * inter.u + b3[1] * inter.v,
                          b1[2] * w + b2[2] * inter.u + b3[2] * inter.v };

    simd_fvec<S> T[3];
    cross(B, N, T);

    simd_fvec<S> _dot_I_N = dot(I, N);

    // used to randomize halton sequence among pixels
    simd_ivec<S> rand_hash = hash(px_index), rand_hash2, rand_hash3;
    simd_fvec<S> rand_offset = construct_float(rand_hash), rand_offset2, rand_offset3;

    const int hi = (pi.iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT + pi.bounce * 2;

    simd_ivec<S> secondary_mask = { 0 };

    {
        simd_ivec<S> ray_queue[S];
        int index = 0;
        int num = 1;

        ray_queue[0] = inter.mask;

        while (index != num) {
            uint32_t first_oi = 0xffffffff, first_mi = 0xffffffff;

            for (int i = 0; i < S; i++) {
                if (!ray_queue[index][i]) continue;

                if (first_oi == 0xffffffff)
                    first_oi = inter.obj_index[i];

                if (first_mi == 0xffffffff)
                    first_mi = mat_index[i];
            }
    
            auto same_mi = mat_index == first_mi;
            auto diff_mi = and_not(same_mi, ray_queue[index]);

            if (diff_mi.not_all_zeros()) {
                ray_queue[num] = diff_mi;
                num++;
            }

            /////////////////////////////////////////

            const auto *mat = &materials[first_mi];

            while (mat->type == MixMaterial) {
                simd_fvec<S> mix[4];
                SampleBilinear(tex_atlas, textures[mat->textures[MAIN_TEXTURE]], uvs, { 0 }, same_mi, mix);
                mix[0] *= mat->strength;

                first_mi = 0xffffffff;

                for (int i = 0; i < S; i++) {
                    if (!same_mi[i]) continue;

                    float _unused;
                    const float r = std::modf(halton[hi + 0] + rand_offset[i], &_unused);

                    rand_hash[i] = hash(rand_hash[i]);
                    rand_offset = construct_float(rand_hash);

                    // shlick fresnel
                    float RR = mat->fresnel + (1.0f - mat->fresnel) * std::pow(1.0f + _dot_I_N[i], 5.0f);
                    if (RR < 0.0f) RR = 0.0f;
                    else if (RR > 1.0f) RR = 1.0f;

                    mat_index[i] = (r * RR < mix[0][i]) ? mat->textures[MIX_MAT1] : mat->textures[MIX_MAT2];
                    if (first_mi == 0xffffffff) {
                        first_mi = mat_index[i];
                    }
                }

                auto _same_mi = mat_index == first_mi;
                diff_mi = and_not(_same_mi, same_mi);
                same_mi = _same_mi;

                if (diff_mi.not_all_zeros()) {
                    ray_queue[num] = diff_mi;
                    num++;
                }

                mat = &materials[first_mi];
            }

            rand_hash2 = hash(rand_hash);
            rand_offset2 = construct_float(rand_hash2);

            rand_hash3 = hash(rand_hash2);
            rand_offset3 = construct_float(rand_hash3);

            simd_fvec<S> tex_normal[4], tex_albedo[4];

            SampleBilinear(tex_atlas, textures[mat->textures[NORMALS_TEXTURE]], uvs, { 0 }, same_mi, tex_normal);

            tex_normal[0] = tex_normal[0] * 2.0f - 1.0f;
            tex_normal[1] = tex_normal[1] * 2.0f - 1.0f;
            tex_normal[2] = tex_normal[2] * 2.0f - 1.0f;

            SampleAnisotropic(tex_atlas, textures[mat->textures[MAIN_TEXTURE]], uvs, surf_der.duv_dx, surf_der.duv_dy, same_mi, tex_albedo);

            tex_albedo[0] = pow(tex_albedo[0] * mat->main_color[0], 2.2f);
            tex_albedo[1] = pow(tex_albedo[1] * mat->main_color[1], 2.2f);
            tex_albedo[2] = pow(tex_albedo[2] * mat->main_color[2], 2.2f);

            simd_fvec<S> temp[3];
            temp[0] = tex_normal[0] * B[0] + tex_normal[2] * N[0] + tex_normal[1] * T[0];
            temp[1] = tex_normal[0] * B[1] + tex_normal[2] * N[1] + tex_normal[1] * T[1];
            temp[2] = tex_normal[0] * B[2] + tex_normal[2] * N[2] + tex_normal[1] * T[2];

            //////////////////////////////////////////

            simd_fvec<S> __N[3] = { dot(temp, inv_xform1), dot(temp, inv_xform2), dot(temp, inv_xform3) },
                         __B[3] = { dot(B, inv_xform1), dot(B, inv_xform2), dot(B, inv_xform3) },
                         __T[3] = { dot(T, inv_xform1), dot(T, inv_xform2), dot(T, inv_xform3) };

            //////////////////////////////////////////

            if (mat->type == DiffuseMaterial) {
                if (pi.should_add_direct_light()) {
                    ComputeDirectLighting(P, __N, __B, plane_N, halton, hi, rand_offset, rand_offset2, mesh_instances, mi_indices,
                                          meshes, transforms, vtx_indices, vertices, nodes, node_index, tris,
                                          tri_indices, lights, li_indices, light_node_index, same_mi, out_rgba);

                    if (pi.should_consider_albedo()) {
                        const auto &mask = reinterpret_cast<const simd_fvec<S>&>(same_mi);

                        where(mask, out_rgba[0]) = ray.c[0] * tex_albedo[0] * out_rgba[0];
                        where(mask, out_rgba[1]) = ray.c[1] * tex_albedo[1] * out_rgba[1];
                        where(mask, out_rgba[2]) = ray.c[2] * tex_albedo[2] * out_rgba[2];
                    }
                }

                simd_fvec<S> rc[3] = { ray.c[0], ray.c[1], ray.c[2] };

                if (pi.should_consider_albedo()) {
                    ITERATE_3({ rc[i] *= tex_albedo[i]; })
                }

                simd_fvec<S> V[3], p;

                for (int i = 0; i < S; i++) {
                    if (!same_mi[i]) continue;

                    float _unused;
                    const float z = std::modf(halton[hi + 0] + rand_offset[i], &_unused);

                    const float dir = std::sqrt(z);
                    const float phi = 2 * PI * std::modf(halton[hi + 1] + rand_offset2[i], &_unused);

                    const float cos_phi = std::cos(phi);
                    const float sin_phi = std::sin(phi);

                    V[0][i] = dir * sin_phi * __B[0][i] + std::sqrt(1.0f - dir) * __N[0][i] + dir * cos_phi * __T[0][i];
                    V[1][i] = dir * sin_phi * __B[1][i] + std::sqrt(1.0f - dir) * __N[1][i] + dir * cos_phi * __T[1][i];
                    V[2][i] = dir * sin_phi * __B[2][i] + std::sqrt(1.0f - dir) * __N[2][i] + dir * cos_phi * __T[2][i];
                
                    p[i] = std::modf(halton[hi + 0] + rand_offset3[i], &_unused);
                }

                const simd_fvec<S> thr = max(rc[0], max(rc[1], rc[2]));
                const auto new_ray_mask = (p < thr / RAY_TERM_THRES) & reinterpret_cast<const simd_fvec<S>&>(same_mi);

                if (reinterpret_cast<const simd_ivec<S>&>(new_ray_mask).not_all_zeros()) {
                    const int index = *out_secondary_rays_count;
                    auto &r = out_secondary_rays[index];

                    // modify weight of non-terminated Ray
                    where(thr < RAY_TERM_THRES, rc[0]) = rc[0] * (RAY_TERM_THRES / thr);
                    where(thr < RAY_TERM_THRES, rc[1]) = rc[1] * (RAY_TERM_THRES / thr);
                    where(thr < RAY_TERM_THRES, rc[2]) = rc[2] * (RAY_TERM_THRES / thr);

                    secondary_mask = secondary_mask | reinterpret_cast<const simd_ivec<S>&>(new_ray_mask);

                    where(new_ray_mask, r.o[0]) = P[0] + HIT_BIAS * __N[0];
                    where(new_ray_mask, r.o[1]) = P[1] + HIT_BIAS * __N[1];
                    where(new_ray_mask, r.o[2]) = P[2] + HIT_BIAS * __N[2];

                    where(new_ray_mask, r.d[0]) = V[0];
                    where(new_ray_mask, r.d[1]) = V[1];
                    where(new_ray_mask, r.d[2]) = V[2];

                    where(new_ray_mask, r.c[0]) = rc[0];
                    where(new_ray_mask, r.c[1]) = rc[1];
                    where(new_ray_mask, r.c[2]) = rc[2];
                    where(new_ray_mask, r.ior) = ray.ior;

                    where(new_ray_mask, r.do_dx[0]) = surf_der.do_dx[0];
                    where(new_ray_mask, r.do_dx[1]) = surf_der.do_dx[1];
                    where(new_ray_mask, r.do_dx[2]) = surf_der.do_dx[2];

                    where(new_ray_mask, r.do_dy[0]) = surf_der.do_dy[0];
                    where(new_ray_mask, r.do_dy[1]) = surf_der.do_dy[1];
                    where(new_ray_mask, r.do_dy[2]) = surf_der.do_dy[2];

                    where(new_ray_mask, r.dd_dx[0]) = surf_der.dd_dx[0] - 2 * (dot(I, __N) * surf_der.dndx[0] + surf_der.ddn_dx * __N[0]);
                    where(new_ray_mask, r.dd_dx[1]) = surf_der.dd_dx[1] - 2 * (dot(I, __N) * surf_der.dndx[1] + surf_der.ddn_dx * __N[1]);
                    where(new_ray_mask, r.dd_dx[2]) = surf_der.dd_dx[2] - 2 * (dot(I, __N) * surf_der.dndx[2] + surf_der.ddn_dx * __N[2]);

                    where(new_ray_mask, r.dd_dy[0]) = surf_der.dd_dy[0] - 2 * (dot(I, __N) * surf_der.dndy[0] + surf_der.ddn_dy * __N[0]);
                    where(new_ray_mask, r.dd_dy[1]) = surf_der.dd_dy[1] - 2 * (dot(I, __N) * surf_der.dndy[1] + surf_der.ddn_dy * __N[1]);
                    where(new_ray_mask, r.dd_dy[2]) = surf_der.dd_dy[2] - 2 * (dot(I, __N) * surf_der.dndy[2] + surf_der.ddn_dy * __N[2]);
                }
            } else if (mat->type == GlossyMaterial) {
                simd_fvec<S> V[3], TT[3], BB[3];
                simd_fvec<S> _NN[3] = { __N[0], __N[1], __N[2] };

                simd_fvec<S> dot_I_N2 = dot(I, __N);

                where(dot_I_N2 < 0, _NN[0]) = -__N[0];
                where(dot_I_N2 < 0, _NN[1]) = -__N[1];
                where(dot_I_N2 < 0, _NN[2]) = -__N[2];
                where(dot_I_N2 < 0, dot_I_N2) = -dot_I_N2;

                reflect(I, _NN, dot_I_N2, V);

                cross(V, B, TT);
                cross(V, TT, BB);

                simd_fvec<S> rc[3] = { ray.c[0], ray.c[1], ray.c[2] }, p;

                const float h = 1.0f - std::cos(0.5f * PI * mat->roughness * mat->roughness);

                for (int i = 0; i < S; i++) {
                    float _unused;
                    const float z = h * std::modf(halton[hi + 0] + rand_offset[i], &_unused);
                    const float dir = std::sqrt(z);

                    const float phi = 2 * PI * std::modf(halton[hi + 1] + rand_offset2[i], &_unused);
                    const float cos_phi = std::cos(phi);
                    const float sin_phi = std::sin(phi);

                    V[0][i] = dir * sin_phi * BB[0][i] + std::sqrt(1.0f - dir) * V[0][i] + dir * cos_phi * TT[0][i];
                    V[1][i] = dir * sin_phi * BB[1][i] + std::sqrt(1.0f - dir) * V[1][i] + dir * cos_phi * TT[1][i];
                    V[2][i] = dir * sin_phi * BB[2][i] + std::sqrt(1.0f - dir) * V[2][i] + dir * cos_phi * TT[2][i];
                
                    p[i] = std::modf(halton[hi + 0] + rand_offset3[i], &_unused);
                }

                const simd_fvec<S> thr = max(rc[0], max(rc[1], rc[2]));
                const auto new_ray_mask = (p < thr / RAY_TERM_THRES) & reinterpret_cast<const simd_fvec<S>&>(same_mi);

                if (reinterpret_cast<const simd_ivec<S>&>(new_ray_mask).not_all_zeros()) {
                    const int index = *out_secondary_rays_count;
                    auto &r = out_secondary_rays[index];

                    // modify weight of non-terminated Ray
                    where(thr < RAY_TERM_THRES, rc[0]) = rc[0] * (RAY_TERM_THRES / thr);
                    where(thr < RAY_TERM_THRES, rc[1]) = rc[1] * (RAY_TERM_THRES / thr);
                    where(thr < RAY_TERM_THRES, rc[2]) = rc[2] * (RAY_TERM_THRES / thr);

                    secondary_mask = secondary_mask | reinterpret_cast<const simd_ivec<S>&>(new_ray_mask);

                    where(new_ray_mask, r.o[0]) = P[0] + HIT_BIAS * __N[0];
                    where(new_ray_mask, r.o[1]) = P[1] + HIT_BIAS * __N[1];
                    where(new_ray_mask, r.o[2]) = P[2] + HIT_BIAS * __N[2];

                    where(new_ray_mask, r.d[0]) = V[0];
                    where(new_ray_mask, r.d[1]) = V[1];
                    where(new_ray_mask, r.d[2]) = V[2];

                    where(new_ray_mask, r.c[0]) = rc[0];
                    where(new_ray_mask, r.c[1]) = rc[1];
                    where(new_ray_mask, r.c[2]) = rc[2];
                    where(new_ray_mask, r.ior) = ray.ior;

                    where(new_ray_mask, r.do_dx[0]) = surf_der.do_dx[0];
                    where(new_ray_mask, r.do_dx[1]) = surf_der.do_dx[1];
                    where(new_ray_mask, r.do_dx[2]) = surf_der.do_dx[2];

                    where(new_ray_mask, r.do_dy[0]) = surf_der.do_dy[0];
                    where(new_ray_mask, r.do_dy[1]) = surf_der.do_dy[1];
                    where(new_ray_mask, r.do_dy[2]) = surf_der.do_dy[2];

                    where(new_ray_mask, r.dd_dx[0]) = surf_der.dd_dx[0] - 2.0f * (dot_I_N2 * surf_der.dndx[0] + surf_der.ddn_dx * __N[0]);
                    where(new_ray_mask, r.dd_dx[1]) = surf_der.dd_dx[1] - 2.0f * (dot_I_N2 * surf_der.dndx[1] + surf_der.ddn_dx * __N[1]);
                    where(new_ray_mask, r.dd_dx[2]) = surf_der.dd_dx[2] - 2.0f * (dot_I_N2 * surf_der.dndx[2] + surf_der.ddn_dx * __N[2]);

                    where(new_ray_mask, r.dd_dy[0]) = surf_der.dd_dy[0] - 2.0f * (dot_I_N2 * surf_der.dndy[0] + surf_der.ddn_dy * __N[0]);
                    where(new_ray_mask, r.dd_dy[1]) = surf_der.dd_dy[1] - 2.0f * (dot_I_N2 * surf_der.dndy[1] + surf_der.ddn_dy * __N[1]);
                    where(new_ray_mask, r.dd_dy[2]) = surf_der.dd_dy[2] - 2.0f * (dot_I_N2 * surf_der.dndy[2] + surf_der.ddn_dy * __N[2]);
                }
            } else if (mat->type == RefractiveMaterial) {
                simd_fvec<S> _NN[3] = { __N[0], __N[1], __N[2] };

                simd_fvec<S> dot_I_N2 = dot(I, __N);

                where(dot_I_N2 > 0, _NN[0]) = -__N[0];
                where(dot_I_N2 > 0, _NN[1]) = -__N[1];
                where(dot_I_N2 > 0, _NN[2]) = -__N[2];
                
                simd_fvec<S> eta = ray.ior;
                where(dot_I_N2 <= 0, eta) = eta / mat->ior;
                where(dot_I_N2 < 0, dot_I_N2) = -dot_I_N2;

                simd_fvec<S> _I[3] = { -I[0], -I[1], -I[2] };

                simd_fvec<S> cosi = dot(_I, _NN);
                simd_fvec<S> cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
                simd_fvec<S> m = eta * cosi - sqrt(cost2);

                simd_fvec<S> V[3] = { eta * I[0] + m * _NN[0], eta * I[1] + m * _NN[1], eta * I[2] + m * _NN[2] };

                simd_fvec<S> TT[3], BB[3];

                cross(V, __B, TT);
                cross(V, TT, BB);

                normalize(TT);
                normalize(BB);

                simd_fvec<S> rc[3] = { ray.c[0], ray.c[1], ray.c[2] }, p;

                for (int i = 0; i < S; i++) {
                    const float z = 1.0f - halton[hi + 0] * mat->roughness;
                    const float temp = std::sqrt(1.0f - z * z);

                    const float phi = halton[(((hash(hi) + pi.iteration) & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT + pi.bounce * 2) + 0] * 2 * PI;
                    const float cos_phi = std::cos(phi);
                    const float sin_phi = std::sin(phi);

                    V[0][i] = temp * sin_phi * BB[0][i] + z * V[0][i] + temp * cos_phi * TT[0][i];
                    V[1][i] = temp * sin_phi * BB[1][i] + z * V[1][i] + temp * cos_phi * TT[1][i];
                    V[2][i] = temp * sin_phi * BB[2][i] + z * V[2][i] + temp * cos_phi * TT[2][i];

                    rc[0][i] *= z;
                    rc[1][i] *= z;
                    rc[2][i] *= z;

                    float _unused;
                    p[i] = std::modf(halton[hi + 0] + rand_offset3[i], &_unused);
                }

                simd_fvec<S> k = (eta - eta * eta * dot(I, plane_N) / dot(V, plane_N));
                simd_fvec<S> dmdx = k * surf_der.ddn_dx;
                simd_fvec<S> dmdy = k * surf_der.ddn_dy;

                simd_fvec<S> thres = dot(rc, rc);

                const simd_fvec<S> thr = max(rc[0], max(rc[1], rc[2]));
                const auto new_ray_mask = (cost2 >= 0.0f) & (p < thr / RAY_TERM_THRES) & reinterpret_cast<const simd_fvec<S>&>(same_mi);

                if (reinterpret_cast<const simd_ivec<S>&>(new_ray_mask).not_all_zeros()) {
                    const int index = *out_secondary_rays_count;
                    auto &r = out_secondary_rays[index];

                    // modify weight of non-terminated Ray
                    where(thr < RAY_TERM_THRES, rc[0]) = rc[0] * (RAY_TERM_THRES / thr);
                    where(thr < RAY_TERM_THRES, rc[1]) = rc[1] * (RAY_TERM_THRES / thr);
                    where(thr < RAY_TERM_THRES, rc[2]) = rc[2] * (RAY_TERM_THRES / thr);

                    secondary_mask = secondary_mask | reinterpret_cast<const simd_ivec<S>&>(new_ray_mask);

                    where(new_ray_mask, r.o[0]) = P[0] + HIT_BIAS * I[0];
                    where(new_ray_mask, r.o[1]) = P[1] + HIT_BIAS * I[1];
                    where(new_ray_mask, r.o[2]) = P[2] + HIT_BIAS * I[2];

                    where(new_ray_mask, r.d[0]) = V[0];
                    where(new_ray_mask, r.d[1]) = V[1];
                    where(new_ray_mask, r.d[2]) = V[2];

                    where(new_ray_mask, r.c[0]) = rc[0];
                    where(new_ray_mask, r.c[1]) = rc[1];
                    where(new_ray_mask, r.c[2]) = rc[2];
                    where(new_ray_mask, r.ior) = mat->ior;

                    where(new_ray_mask, r.do_dx[0]) = surf_der.do_dx[0];
                    where(new_ray_mask, r.do_dx[1]) = surf_der.do_dx[1];
                    where(new_ray_mask, r.do_dx[2]) = surf_der.do_dx[2];

                    where(new_ray_mask, r.do_dy[0]) = surf_der.do_dy[0];
                    where(new_ray_mask, r.do_dy[1]) = surf_der.do_dy[1];
                    where(new_ray_mask, r.do_dy[2]) = surf_der.do_dy[2];

                    where(new_ray_mask, r.dd_dx[0]) = eta * surf_der.dd_dx[0] - (m * surf_der.dndx[0] + dmdx * plane_N[0]);
                    where(new_ray_mask, r.dd_dx[1]) = eta * surf_der.dd_dx[1] - (m * surf_der.dndx[1] + dmdx * plane_N[1]);
                    where(new_ray_mask, r.dd_dx[2]) = eta * surf_der.dd_dx[2] - (m * surf_der.dndx[2] + dmdx * plane_N[2]);

                    where(new_ray_mask, r.dd_dy[0]) = eta * surf_der.dd_dy[0] - (m * surf_der.dndy[0] + dmdy * plane_N[0]);
                    where(new_ray_mask, r.dd_dy[1]) = eta * surf_der.dd_dy[1] - (m * surf_der.dndy[1] + dmdy * plane_N[1]);
                    where(new_ray_mask, r.dd_dy[2]) = eta * surf_der.dd_dy[2] - (m * surf_der.dndy[2] + dmdy * plane_N[2]);
                }
            } else if (mat->type == EmissiveMaterial) {
                const auto &mask = reinterpret_cast<const simd_fvec<S>&>(same_mi);

                where(mask, out_rgba[0]) = mat->strength * ray.c[0] * tex_albedo[0];
                where(mask, out_rgba[1]) = mat->strength * ray.c[1] * tex_albedo[1];
                where(mask, out_rgba[2]) = mat->strength * ray.c[2] * tex_albedo[2];
            }

            index++;
        }
    }

    if (secondary_mask.not_all_zeros()) {
        const int index = (*out_secondary_rays_count)++;
        out_secondary_masks[index] = secondary_mask;
        out_secondary_rays[index].xy = ray.xy;
    }
}
