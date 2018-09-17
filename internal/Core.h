#pragma once

#include <cstdint>

#include "../SceneBase.h"
#include "../Types.h"

#ifdef __GNUC__
#define force_inline __attribute__((always_inline)) inline
#endif
#ifdef _MSC_VER
#define force_inline __forceinline
#endif

#include "simd/aligned_allocator.h"

namespace Ray {
enum eUninitialize { Uninitialize };

union rays_id_t {
    uint32_t id;
    struct {
        uint16_t x, y;
    };
};

struct tri_accel_t {
    float nu, nv;
    float np;
    float pu, pv;
    int32_t ci;
    float e0u, e0v;
    float e1u, e1v;
    uint32_t mi;
    uint32_t pad;
};
static_assert(sizeof(tri_accel_t) == 48, "!");

const uint8_t TRI_W_BITS = 0b00000011;
const uint8_t TRI_AXIS_ALIGNED_BIT = 0b00000100;
const uint8_t TRI_INV_NORMAL_BIT = 0b00001000;

const float HIT_BIAS = 0.001f;
const float HIT_EPS = 0.000001f;
const float FLT_EPS = 0.0000001f;

const float PI = 3.141592653589793238463f;

const float MAX_DIST = 3.402823466e+38F;

const int MAX_BOUNCES = 4;

const float RAY_TERM_THRES = 0.01f;

const float LIGHT_ATTEN_CUTOFF = 0.001f;

const uint32_t SLAB_BITS = 0b11111100;
const uint32_t SLAB_BIT_POS_X = 0b00000100,
               SLAB_BIT_NEG_X = 0b00001000,
               SLAB_BIT_POS_Y = 0b00010000,
               SLAB_BIT_NEG_Y = 0b00100000,
               SLAB_BIT_POS_Z = 0b01000000,
               SLAB_BIT_NEG_Z = 0b10000000;
const uint32_t SPACE_AXIS_BITS = 0b00000011;

struct bvh_node_t {
    uint32_t prim_index, prim_count,
             left_child, right_child, parent,
             flags;
    // Flags contain:
    // (flags & SLAB_BIT_*) - bounding box slab relation to parent, used for optimized intersection test
    // (flags & SPACE_AXIS_BITS) - axis with maximal child's centroids distance, used to guess closest child of node
    float bbox[2][3];
};
static_assert(sizeof(bvh_node_t) == 48, "!");

const int MAX_MIP_LEVEL = 11;
const int NUM_MIP_LEVELS = MAX_MIP_LEVEL + 1;
const int MAX_TEXTURE_SIZE = (1 << NUM_MIP_LEVELS);

struct texture_t {
    uint16_t size[2];
    uint8_t page[NUM_MIP_LEVELS];
    uint16_t pos[NUM_MIP_LEVELS][2];
};
static_assert(sizeof(texture_t) == 64, "!");

const int MAX_MATERIAL_TEXTURES = 7;

const int NORMALS_TEXTURE = 0;
const int MAIN_TEXTURE = 1;

const int MIX_MAT1 = 2;
const int MIX_MAT2 = 3;

const int MAX_STACK_SIZE = 32;

struct material_t {
    uint32_t type;
    uint32_t textures[MAX_MATERIAL_TEXTURES];
    float main_color[3], pad;
    union {
        float roughness;
        float strength;
    };
    union {
        float fresnel;
        float ior;
    };
    float pad1[2];
};
static_assert(sizeof(material_t) == 64, "!");

struct light_t {
    float pos[3], radius;
    float col[3], brightness;
    float dir[3], spot;
};
static_assert(sizeof(light_t) == 48, "!");

struct prim_t;

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, alignof(T)>>;

// Creates struct of precomputed triangle data for faster Plucker intersection test
void PreprocessTri(const float *p, int stride, tri_accel_t *acc);
// Extructs planar trianble normal from precomputed triangle data
void ExtractPlaneNormal(const tri_accel_t &tri, float *out_normal);

// Builds BVH for mesh and precomputes triangle data
uint32_t PreprocessMesh(const float *attrs, size_t attrs_count, const uint32_t *indices, size_t indices_count, eVertexLayout layout,
                        bool allow_spatial_splits, std::vector<bvh_node_t> &out_nodes, std::vector<tri_accel_t> &out_tris, std::vector<uint32_t> &out_indices);

// Builds BVH for arbitrary primitives (using their bounding boxes), used to build macro tree
uint32_t PreprocessPrims(const prim_t *prims, size_t prims_count, const float *positions, size_t stride,
                         bool allow_spatial_splits, std::vector<bvh_node_t> &out_nodes, std::vector<uint32_t> &out_indices);

bool NaiivePluckerTest(const float p[9], const float o[3], const float d[3]);

void ConstructCamera(eCamType type, eFilterType filter, const float origin[3], const float fwd[3], float fov, float gamma, float focus_distance, float focus_factor, camera_t *cam);

// Applies 4x4 matrix matrix transform to bounding box
void TransformBoundingBox(const float bbox[2][3], const float *xform, float out_bbox[2][3]);

void InverseMatrix(const float mat[16], float out_mat[16]);

// Arrays of prime numbers, used to generate halton sequence for sampling
const int PrimesCount = 24;
const int g_primes[PrimesCount] =     { 2, 3, 5, 7,  11, 13, 17, 19, 23, 29,  31,  37,  41,  43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89  };
const int g_prime_sums[PrimesCount] = { 0, 2, 5, 10, 17, 28, 41, 58, 77, 100, 129, 160, 197, 238, 281, 328, 381, 440, 501, 568, 639, 712, 791, 874 };

const int HALTON_COUNT = PrimesCount;
const int HALTON_2D_COUNT = PrimesCount / 2;
const int HALTON_SEQ_LEN = 256;

// Sampling stages must be independent from each other (otherwise it may lead to artifacts), so different halton sequences at each ray bounce must be used.
// This leads to limited number of bounces. Can be easily fixed by generating huge primes table above, but i do not need it for now.
static_assert(MAX_BOUNCES + 2 <= HALTON_2D_COUNT, "!");

struct vertex_t {
    float p[3], n[3], b[3], t[2][2];
};
static_assert(sizeof(vertex_t) == 52, "!");

struct mesh_t {
    uint32_t node_index, node_count;
    uint32_t tris_index, tris_count;
};
static_assert(sizeof(mesh_t) == 16, "!");

struct transform_t {
    float xform[16], inv_xform[16];
};
static_assert(sizeof(transform_t) == 128, "!");

struct mesh_instance_t {
    float bbox_min[3];
    uint32_t tr_index;
    float bbox_max[3];
    uint32_t mesh_index;
};
static_assert(sizeof(mesh_instance_t) == 32, "!");

extern const float uint8_to_float_table[];

force_inline float to_norm_float(uint8_t v) {
    return uint8_to_float_table[v];
}

extern const uint8_t morton_table_16[];
extern const int morton_table_256[];

extern const float omega_step;
extern const char omega_table[];

extern const float phi_step;
extern const char phi_table[][17];

struct ray_chunk_t {
    uint32_t hash, base, size;
};

struct pass_info_t {
    int index;
    int iteration, bounce;
    uint32_t flags = 0;

    force_inline bool should_add_direct_light() const {
        // skip for primary bounce if we want only indirect light contribution
        return !(flags & SkipDirectLight) || bounce > 2;
    }

    force_inline bool should_add_environment() const {
        return !(flags & NoBackground) || bounce > 2;
    }

    force_inline bool should_consider_albedo() const {
        // do not use albedo in lightmap mode for primary bounce
        return !(flags & LightingOnly) || bounce > 2;
    }
};
static_assert(sizeof(pass_info_t) == 16, "!");

}
