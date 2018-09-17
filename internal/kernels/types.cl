R"(

/* This file should be passed in cl program first */

typedef struct _ray_packet_t {
    float4 o, d;
    float4 c;
    float3 do_dx, dd_dx, do_dy, dd_dy;
} ray_packet_t;

typedef struct _camera_t {
    float4 origin, fwd;
    float4 side, up;
    int flags;
} camera_t;

typedef struct _tri_accel_t {
    float nu, nv;
    float np;
    float pu, pv;
    int ci;
    float e0u, e0v;
    float e1u, e1v;
    uint mi;
    int pad1;
} tri_accel_t;

typedef struct _hit_data_t {
    int mask, obj_index, prim_index;
    float t, u, v;
    float2 ray_id;
} hit_data_t;

typedef struct _bvh_node_t {
    uint prim_index, prim_count,
         left_child, right_child, parent,
         flags;
    // Flags contain:
    // (flags & SLAB_BIT_*) - bounding box slab relation to parent, used for optimized intersection test
    // (flags & SPACE_AXIS_BITS) - axis with maximal child's centroids distance, used to guess closest child of node
    float bbox[2][3];
} bvh_node_t;

typedef struct _vertex_t {
    float p[3], n[3], b[3], t[2][2];
} vertex_t;

typedef struct _mesh_t {
    uint node_index, node_count;
    uint tris_index, tris_count;
} mesh_t;

typedef struct _transform_t {
    float16 xform, inv_xform;
} transform_t;

typedef struct _mesh_instance_t {
    float bbox_min[3];
    uint tr_index;
    float bbox_max[3];
    uint mesh_index;
} mesh_instance_t;

typedef struct _texture_t {
    ushort size[2];
    uchar page[12];
    ushort pos[12][2];
} texture_t;

typedef struct _material_t {
    uint type;
    uint textures[MAX_MATERIAL_TEXTURES];
    float3 main_color;
    union {
        float roughness;
        float strength;
    };
    union {
        float fresnel;
        float ior;
    };
} material_t;

typedef struct _light_t {
    float4 pos_and_radius;
    float4 col_and_brightness;
    float4 dir_and_spot;
} light_t;

typedef struct _environment_t {
    float4 env_col_and_clamp;
    uint env_map;
    float pad[3];
} environment_t;

typedef struct _ray_chunk_t {
    uint hash, base, size;
} ray_chunk_t;

typedef struct _pass_info_t {
    int index;
    int iteration, bounce;
    uint flags;
} pass_info_t;

typedef struct _derivatives_t {
    float3 do_dx, do_dy, dd_dx, dd_dy;
    float2 duv_dx, duv_dy;
    float3 dndx, dndy;
    float ddn_dx, ddn_dy;
} derivatives_t;

__kernel void TypesCheck(ray_packet_t r, camera_t c, tri_accel_t t, hit_data_t i,
                         bvh_node_t b, vertex_t v, mesh_t m, mesh_instance_t mi, transform_t tr,
                         texture_t tex, material_t mat, light_t l, environment_t env, ray_chunk_t ch,
                         pass_info_t pi) {}

)"