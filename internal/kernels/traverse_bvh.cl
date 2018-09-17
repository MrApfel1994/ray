R"(

bool __bbox_test(const float3 o, const float3 inv_d, const float t, const float3 bbox_min, const float3 bbox_max) {
    float low = inv_d.x * (bbox_min.x - o.x);
    float high = inv_d.x * (bbox_max.x - o.x);
    float tmin = fmin(low, high);
    float tmax = fmax(low, high);

    low = inv_d.y * (bbox_min.y - o.y);
    high = inv_d.y * (bbox_max.y - o.y);
    tmin = fmax(tmin, fmin(low, high));
    tmax = fmin(tmax, fmax(low, high));

    low = inv_d.z * (bbox_min.z - o.z);
    high = inv_d.z * (bbox_max.z - o.z);
    tmin = fmax(tmin, fmin(low, high));
    tmax = fmin(tmax, fmax(low, high));

    return tmin <= tmax && tmin <= t && tmax > 0;
}

bool _bbox_test(const float3 o, const float3 inv_d, const float t, __global const float *bbox_min, __global const float *bbox_max) {
    float low = inv_d.x * (bbox_min[0] - o.x);
    float high = inv_d.x * (bbox_max[0] - o.x);
    float tmin = fmin(low, high);
    float tmax = fmax(low, high);

    low = inv_d.y * (bbox_min[1] - o.y);
    high = inv_d.y * (bbox_max[1] - o.y);
    tmin = fmax(tmin, fmin(low, high));
    tmax = fmin(tmax, fmax(low, high));

    low = inv_d.z * (bbox_min[2] - o.z);
    high = inv_d.z * (bbox_max[2] - o.z);
    tmin = fmax(tmin, fmin(low, high));
    tmax = fmin(tmax, fmax(low, high));

    return tmin <= tmax && tmin <= t && tmax > 0;
}

bool _is_point_inside(const float3 p, __global const float *bbox_min, __global const float *bbox_max) {
    return p.x > bbox_min[0] && p.x < bbox_max[0] &&
           p.y > bbox_min[1] && p.y < bbox_max[1] &&
           p.z > bbox_min[2] && p.z < bbox_max[2];
}


bool bbox_test(const float3 o, const float3 inv_d, const float t, __global const bvh_node_t *node) {
    return _bbox_test(o, inv_d, t, node->bbox[0], node->bbox[1]);
}

bool is_point_inside(const float3 p, __global const bvh_node_t *node) {
    return _is_point_inside(p, node->bbox[0], node->bbox[1]);
}

float3 safe_invert(const float3 v) {
    float3 inv_v = 1.0f / v;

    if (v.x <= FLT_EPS && v.x >= 0) {
        inv_v.x = FLT_MAX;
    } else if (v.x >= -FLT_EPS && v.x < 0) {
        inv_v.x = -FLT_MAX;
    }

    if (v.y <= FLT_EPS && v.y >= 0) {
        inv_v.y = FLT_MAX;
    } else if (v.y >= -FLT_EPS && v.y < 0) {
        inv_v.y = -FLT_MAX;
    }

    if (v.z <= FLT_EPS && v.z >= 0) {
        inv_v.z = FLT_MAX;
    } else if (v.z >= -FLT_EPS && v.z < 0) {
        inv_v.z = -FLT_MAX;
    }

    return inv_v;
}

#define near_child(rd, n)   \
    (rd)[(n)->flags & SPACE_AXIS_BITS] < 0 ? (n)->right_child : (n)->left_child

#define far_child(rd, n)    \
    (rd)[(n)->flags & SPACE_AXIS_BITS] < 0 ? (n)->left_child : (n)->right_child

void Traverse_MicroTree_Stackless(const float3 r_o, const float3 r_d, const float3 inv_d, uint obj_index,
                                  __global const bvh_node_t *nodes, uint node_index,
                                  __global const tri_accel_t *tris, __global const uint *tri_indices, 
                                  hit_data_t *inter) {
    const float *rd = (const float *)&r_d;

    uint cur = node_index;
    uint last = node_index;

    if (!nodes[cur].prim_count) {
        cur = near_child(rd, &nodes[cur]);
    }

    while (cur != 0xffffffff) {
        __global const bvh_node_t *n = &nodes[cur];
        
        if (n->prim_count) {
            IntersectTris(r_o, r_d, tris, tri_indices, n->prim_index, n->prim_count, obj_index, inter);
            last = cur; cur = n->parent;
            continue;
        }

        uint near = near_child(rd, n);
        uint far = far_child(rd, n);

        if (last == far) {
            last = cur; cur = n->parent;
            continue;
        }

        uint try_child = (last == n->parent) ? near : far;
        if (bbox_test(r_o, inv_d, inter->t, &nodes[try_child])) {
            last = cur; cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur; cur = n->parent;
            }
        }
    }
}

void Traverse_MicroTree_WithLocalStack(const float3 r_o, const float3 r_d, const float3 inv_d, uint obj_index,
                                       __global const bvh_node_t *nodes, uint node_index,
                                       __global const tri_accel_t *tris, __global const uint *tri_indices, 
                                       __local uint *stack, hit_data_t *inter) {
    const float *rd = (const float *)&r_d;

    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        __global const bvh_node_t *n = &nodes[cur];

        if (!bbox_test(r_o, inv_d, inter->t, n)) continue;

        if (!n->prim_count) {
            stack[stack_size++] = far_child(rd, n);
            stack[stack_size++] = near_child(rd, n);
        } else {
            IntersectTris(r_o, r_d, tris, tri_indices, n->prim_index, n->prim_count, obj_index, inter);
        }
    }
}

void Traverse_MicroTree_WithPrivateStack(const float3 r_o, const float3 r_d, const float3 inv_d, uint obj_index,
                                         __global const bvh_node_t *nodes, uint node_index,
                                         __global const tri_accel_t *tris, __global const uint *tri_indices, 
                                         uint *stack, hit_data_t *inter) {
    const float *rd = (const float *)&r_d;

    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        __global const bvh_node_t *n = &nodes[cur];

        if (!bbox_test(r_o, inv_d, inter->t, n)) continue;

        if (!n->prim_count) {
            stack[stack_size++] = far_child(rd, n);
            stack[stack_size++] = near_child(rd, n);
        } else {
            IntersectTris(r_o, r_d, tris, tri_indices, n->prim_index, n->prim_count, obj_index, inter);
        }
    }
}

void Traverse_MicroTreeImg_WithLocalStack(const float3 r_o, const float3 r_d, const float3 inv_d, uint obj_index,
                                          __read_only image1d_buffer_t nodes, uint node_index,
                                          __global const tri_accel_t *tris, __global const uint *tri_indices, 
                                          __local uint *stack, uint4 *node_data1, float4 *node_data2, hit_data_t *inter) {
    const float *rd = (const float *)&r_d;

    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        /*
            struct bvh_node_t {
                xxx4 node_data1;  // { prim_index, prim_count, left_child, right_child }
                xxx4 node_data2;  // { parent,     flags,      bbox[0][0], bbox[0][1]  }
                xxx4 node_data3;  // { bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]  }
            };
        */

        *node_data2 = read_imagef(nodes, cur * 3 + 1);
        {   float4 node_data3 = read_imagef(nodes, cur * 3 + 2);
            if (!__bbox_test(r_o, inv_d, inter->t, (float3)((*node_data2).zw, node_data3.x), node_data3.yzw)) continue;
        }

        *node_data1 = read_imageui(nodes, cur * 3 + 0);

        if (!(*node_data1).y) {
            uint space_axis = as_uint((*node_data2).y) & SPACE_AXIS_BITS;
            stack[stack_size++] = rd[space_axis] < 0 ? (*node_data1).z : (*node_data1).w;
            stack[stack_size++] = rd[space_axis] < 0 ? (*node_data1).w : (*node_data1).z;
        } else {
            IntersectTris(r_o, r_d, tris, tri_indices, (*node_data1).x, (*node_data1).y, obj_index, inter);
        }
    }
}

void Traverse_MicroTreeImg_WithPrivateStack(const float3 r_o, const float3 r_d, const float3 inv_d, uint obj_index,
                                            __read_only image1d_buffer_t nodes, uint node_index,
                                            __global const tri_accel_t *tris, __global const uint *tri_indices, 
                                            uint *stack, uint4 *node_data1, float4 *node_data2, hit_data_t *inter) {
    const float *rd = (const float *)&r_d;

    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        /*
            struct bvh_node_t {
                xxx4 node_data1;  // { prim_index, prim_count, left_child, right_child }
                xxx4 node_data2;  // { parent,     flags,      bbox[0][0], bbox[0][1]  }
                xxx4 node_data3;  // { bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]  }
            };
        */

        *node_data2 = read_imagef(nodes, cur * 3 + 1);
        {   float4 node_data3 = read_imagef(nodes, cur * 3 + 2);
            if (!__bbox_test(r_o, inv_d, inter->t, (float3)((*node_data2).zw, node_data3.x), node_data3.yzw)) continue;
        }

        *node_data1 = read_imageui(nodes, cur * 3 + 0);

        if (!(*node_data1).y) {
            uint space_axis = as_uint((*node_data2).y) & SPACE_AXIS_BITS;
            stack[stack_size++] = rd[space_axis] < 0 ? (*node_data1).z : (*node_data1).w;
            stack[stack_size++] = rd[space_axis] < 0 ? (*node_data1).w : (*node_data1).z;
        } else {
            IntersectTris(r_o, r_d, tris, tri_indices, (*node_data1).x, (*node_data1).y, obj_index, inter);
        }
    }
}

float Traverse_MicroTree_Occlusion_Stackless(const float3 r_o, const float3 r_d, const float3 inv_d, float max_dist,
                                             __global const bvh_node_t *nodes, uint node_index,
                                             __global const tri_accel_t *tris, __global const uint *tri_indices) {
    const float *rd = (const float *)&r_d;

    uint cur = node_index;
    uint last = node_index;

    if (!nodes[cur].prim_count) {
        cur = near_child(rd, &nodes[cur]);
    }

    while (cur != 0xffffffff) {
        __global const bvh_node_t *n = &nodes[cur];
        
        if (n->prim_count) {
            if (IntersectTris_Occlusion(r_o, r_d, max_dist, tris, tri_indices, n->prim_index, n->prim_count) < 1)  {
                return 0;
            }
            last = cur; cur = n->parent;
            continue;
        }

        uint near = near_child(rd, n);
        uint far = far_child(rd, n);

        if (last == far) {
            last = cur; cur = n->parent;
            continue;
        }

        uint try_child = (last == n->parent) ? near : far;
        if (bbox_test(r_o, inv_d, max_dist, &nodes[try_child])) {
            last = cur; cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur; cur = n->parent;
            }
        }
    }

    return 1;
}

float Traverse_MicroTree_Occlusion_WithLocalStack(const float3 r_o, const float3 r_d, const float3 inv_d, float max_dist,
                                             __global const bvh_node_t *nodes, uint node_index,
                                             __global const tri_accel_t *tris, __global const uint *tri_indices,
                                             __local uint *stack) {
    const float *rd = (const float *)&r_d;

    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        __global const bvh_node_t *n = &nodes[cur];

        if (!bbox_test(r_o, inv_d, max_dist, n)) continue;

        if (!n->prim_count) {
            stack[stack_size++] = far_child(rd, n);
            stack[stack_size++] = near_child(rd, n);
        } else {
            if (IntersectTris_Occlusion(r_o, r_d, max_dist, tris, tri_indices, n->prim_index, n->prim_count) < 1)  {
                return 0;
            }
        }
    }

    return 1;
}

float Traverse_MicroTree_Occlusion_WithPrivateStack(const float3 r_o, const float3 r_d, const float3 inv_d, float max_dist,
                                                    __global const bvh_node_t *nodes, uint node_index,
                                                    __global const tri_accel_t *tris, __global const uint *tri_indices,
                                                    uint *stack) {
    const float *rd = (const float *)&r_d;

    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        __global const bvh_node_t *n = &nodes[cur];

        if (!bbox_test(r_o, inv_d, max_dist, n)) continue;

        if (!n->prim_count) {
            stack[stack_size++] = far_child(rd, n);
            stack[stack_size++] = near_child(rd, n);
        } else {
            if (IntersectTris_Occlusion(r_o, r_d, max_dist, tris, tri_indices, n->prim_index, n->prim_count) < 1)  {
                return 0;
            }
        }
    }

    return 1;
}

)" // workaround for 16k string literal limitation on msvc
R"(

void Traverse_MacroTree_Stackless(const float3 orig_r_o, const float3 orig_r_d, const float3 orig_r_inv_d, 
                                  __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, 
                                  __global const mesh_t *meshes, __global const transform_t *transforms, 
                                  __global const bvh_node_t *nodes, uint node_index, 
                                  __global const tri_accel_t *tris, __global const uint *tri_indices,
                                  hit_data_t *inter) {
    const float *orig_rd = (const float *)&orig_r_d;

    uint cur = node_index;
    uint last = node_index;

    if (!nodes[cur].prim_count) {
        cur = near_child(orig_rd, &nodes[cur]);
    }

    while (cur != 0xffffffff) {
        __global const bvh_node_t *n = &nodes[cur];

        if (n->prim_count) {
            for (uint i = n->prim_index; i < n->prim_index + n->prim_count; i++) {
                __global const mesh_instance_t *mi = &mesh_instances[mi_indices[i]];
                __global const mesh_t *m = &meshes[mi->mesh_index];
                __global const transform_t *tr = &transforms[mi->tr_index];

                if (!_bbox_test(orig_r_o, orig_r_inv_d, inter->t, mi->bbox_min, mi->bbox_max)) continue;

                const float3 r_o = TransformPoint(orig_r_o, &tr->inv_xform);
                const float3 r_d = TransformDirection(orig_r_d, &tr->inv_xform);
                const float3 inv_d = safe_invert(r_d);
                
                Traverse_MicroTree_Stackless(r_o, r_d, inv_d, mi_indices[i], nodes, m->node_index, tris, tri_indices, inter);
            }

            last = cur; cur = n->parent;
            continue;
        }

        uint near = near_child(orig_rd, n);
        uint far = far_child(orig_rd, n);

        if (last == far) {
            last = cur; cur = n->parent;
            continue;
        }

        uint try_child = (last == n->parent) ? near : far;
        if (bbox_test(orig_r_o, orig_r_inv_d, inter->t, &nodes[try_child])) {
            last = cur; cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur; cur = n->parent;
            }
        }
    }
}

void Traverse_MacroTree_WithLocalStack(const float3 orig_r_o, const float3 orig_r_d, const float3 orig_r_inv_d, 
                                       __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, 
                                       __global const mesh_t *meshes, __global const transform_t *transforms, 
                                       __global const bvh_node_t *nodes, uint node_index, 
                                       __global const tri_accel_t *tris, __global const uint *tri_indices,
                                       __local uint *stack, hit_data_t *inter) {
    const float *orig_rd = (const float *)&orig_r_d;
    
    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        __global const bvh_node_t *n = &nodes[cur];

        if (!bbox_test(orig_r_o, orig_r_inv_d, inter->t, n)) continue;

        if (!n->prim_count) {
            stack[stack_size++] = far_child(orig_rd, n);
            stack[stack_size++] = near_child(orig_rd, n);
        } else {
            for (uint i = n->prim_index; i < n->prim_index + n->prim_count; i++) {
                __global const mesh_instance_t *mi = &mesh_instances[mi_indices[i]];
                __global const mesh_t *m = &meshes[mi->mesh_index];
                __global const transform_t *tr = &transforms[mi->tr_index];

                if (!_bbox_test(orig_r_o, orig_r_inv_d, inter->t, mi->bbox_min, mi->bbox_max)) continue;

                const float3 r_o = TransformPoint(orig_r_o, &tr->inv_xform);
                const float3 r_d = TransformDirection(orig_r_d, &tr->inv_xform);
                const float3 inv_d = safe_invert(r_d);
                
                Traverse_MicroTree_WithLocalStack(r_o, r_d, inv_d, mi_indices[i], nodes, m->node_index, tris, tri_indices, &stack[stack_size], inter);
            }
        }
    }
}

void Traverse_MacroTree_WithPrivateStack(const float3 orig_r_o, const float3 orig_r_d, const float3 orig_r_inv_d, 
                                         __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, 
                                         __global const mesh_t *meshes, __global const transform_t *transforms, 
                                         __global const bvh_node_t *nodes, uint node_index, 
                                         __global const tri_accel_t *tris, __global const uint *tri_indices,
                                         uint *stack, hit_data_t *inter) {
    const float *orig_rd = (const float *)&orig_r_d;
    
    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        __global const bvh_node_t *n = &nodes[cur];

        if (!bbox_test(orig_r_o, orig_r_inv_d, inter->t, n)) continue;

        if (!n->prim_count) {
            stack[stack_size++] = far_child(orig_rd, n);
            stack[stack_size++] = near_child(orig_rd, n);
        } else {
            for (uint i = n->prim_index; i < n->prim_index + n->prim_count; i++) {
                __global const mesh_instance_t *mi = &mesh_instances[mi_indices[i]];
                __global const mesh_t *m = &meshes[mi->mesh_index];
                __global const transform_t *tr = &transforms[mi->tr_index];

                if (!_bbox_test(orig_r_o, orig_r_inv_d, inter->t, mi->bbox_min, mi->bbox_max)) continue;

                const float3 r_o = TransformPoint(orig_r_o, &tr->inv_xform);
                const float3 r_d = TransformDirection(orig_r_d, &tr->inv_xform);
                const float3 inv_d = safe_invert(r_d);
                
                Traverse_MicroTree_WithPrivateStack(r_o, r_d, inv_d, mi_indices[i], nodes, m->node_index, tris, tri_indices, &stack[stack_size], inter);
            }
        }
    }
}

void Traverse_MacroTreeImg_WithLocalStack(const float3 orig_r_o, const float3 orig_r_d, const float3 orig_r_inv_d, 
                                          __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, 
                                          __global const mesh_t *meshes, __global const transform_t *transforms, 
                                          __read_only image1d_buffer_t nodes, uint node_index, 
                                          __global const tri_accel_t *tris, __global const uint *tri_indices,
                                          __local uint *stack, hit_data_t *inter) {
    const float *orig_rd = (const float *)&orig_r_d;

    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        /*
            struct bvh_node_t {
                xxx4 node_data1;  // { prim_index, prim_count, left_child, right_child }
                xxx4 node_data2;  // { parent,     flags,      bbox[0][0], bbox[0][1]  }
                xxx4 node_data3;  // { bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]  }
            };
        */

        float4 node_data2 = read_imagef(nodes, cur * 3 + 1);
        {   float4 node_data3 = read_imagef(nodes, cur * 3 + 2);
            if (!__bbox_test(orig_r_o, orig_r_inv_d, inter->t, (float3)(node_data2.zw, node_data3.x), node_data3.yzw)) continue;
        }

        uint4 node_data1 = read_imageui(nodes, cur * 3 + 0);

        if (!node_data1.y) {
            int space_axis = as_int(node_data2.y) & SPACE_AXIS_BITS;
            stack[stack_size++] = orig_rd[space_axis] < 0 ? node_data1.z : node_data1.w;
            stack[stack_size++] = orig_rd[space_axis] < 0 ? node_data1.w : node_data1.z;
        } else {
            const uint index_start = node_data1.x;
            const uint index_end = node_data1.x + node_data1.y;
            for (uint i = index_start; i < index_end; i++) {
                __global const mesh_instance_t *mi = &mesh_instances[mi_indices[i]];
                __global const mesh_t *m = &meshes[mi->mesh_index];
                __global const transform_t *tr = &transforms[mi->tr_index];

                if (!_bbox_test(orig_r_o, orig_r_inv_d, inter->t, mi->bbox_min, mi->bbox_max)) continue;

                const float3 r_o = TransformPoint(orig_r_o, &tr->inv_xform);
                const float3 r_d = TransformDirection(orig_r_d, &tr->inv_xform);
                const float3 inv_d = safe_invert(r_d);
                
                Traverse_MicroTreeImg_WithLocalStack(r_o, r_d, inv_d, mi_indices[i], nodes, m->node_index, tris, tri_indices,
                                                     &stack[stack_size], &node_data1, &node_data2, inter);
            }
        }
    }
}

)" // workaround for 16k string literal limitation on msvc
R"(

void Traverse_MacroTreeImg_WithPrivateStack(const float3 orig_r_o, const float3 orig_r_d, const float3 orig_r_inv_d, 
                                            __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, 
                                            __global const mesh_t *meshes, __global const transform_t *transforms, 
                                            __read_only image1d_buffer_t nodes, uint node_index, 
                                            __global const tri_accel_t *tris, __global const uint *tri_indices,
                                            uint *stack, hit_data_t *inter) {
    const float *orig_rd = (const float *)&orig_r_d;

    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        /*
            struct bvh_node_t {
                xxx4 node_data1;  // { prim_index, prim_count, left_child, right_child }
                xxx4 node_data2;  // { parent,     flags,      bbox[0][0], bbox[0][1]  }
                xxx4 node_data3;  // { bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]  }
            };
        */

        float4 node_data2 = read_imagef(nodes, cur * 3 + 1);
        {   float4 node_data3 = read_imagef(nodes, cur * 3 + 2);
            if (!__bbox_test(orig_r_o, orig_r_inv_d, inter->t, (float3)(node_data2.zw, node_data3.x), node_data3.yzw)) continue;
        }

        uint4 node_data1 = read_imageui(nodes, cur * 3 + 0);

        if (!node_data1.y) {
            int space_axis = as_int(node_data2.y) & SPACE_AXIS_BITS;
            stack[stack_size++] = orig_rd[space_axis] < 0 ? node_data1.z : node_data1.w;
            stack[stack_size++] = orig_rd[space_axis] < 0 ? node_data1.w : node_data1.z;
        } else {
            const uint index_start = node_data1.x;
            const uint index_end = node_data1.x + node_data1.y;
            for (uint i = index_start; i < index_end; i++) {
                __global const mesh_instance_t *mi = &mesh_instances[mi_indices[i]];
                __global const mesh_t *m = &meshes[mi->mesh_index];
                __global const transform_t *tr = &transforms[mi->tr_index];

                if (!_bbox_test(orig_r_o, orig_r_inv_d, inter->t, mi->bbox_min, mi->bbox_max)) continue;

                const float3 r_o = TransformPoint(orig_r_o, &tr->inv_xform);
                const float3 r_d = TransformDirection(orig_r_d, &tr->inv_xform);
                const float3 inv_d = safe_invert(r_d);
                
                Traverse_MicroTreeImg_WithPrivateStack(r_o, r_d, inv_d, mi_indices[i], nodes, m->node_index, tris, tri_indices,
                                                       &stack[stack_size], &node_data1, &node_data2, inter);
            }
        }
    }
}

float Traverse_MacroTree_Occlusion_Stackless(const float3 orig_r_o, const float3 orig_r_d, const float3 orig_r_inv_d, float max_dist, 
                                             __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, 
                                             __global const mesh_t *meshes, __global const transform_t *transforms, 
                                             __global const bvh_node_t *nodes, uint node_index, 
                                             __global const tri_accel_t *tris, __global const uint *tri_indices) {
    const float *orig_rd = (const float *)&orig_r_d;

    uint cur = node_index;
    uint last = node_index;

    if (!nodes[cur].prim_count) {
        cur = near_child(orig_rd, &nodes[cur]);
    }

    while (cur != 0xffffffff) {
        __global const bvh_node_t *n = &nodes[cur];

        if (n->prim_count) {
            for (uint i = n->prim_index; i < n->prim_index + n->prim_count; i++) {
                __global const mesh_instance_t *mi = &mesh_instances[mi_indices[i]];
                __global const mesh_t *m = &meshes[mi->mesh_index];
                __global const transform_t *tr = &transforms[mi->tr_index];

                if (!_bbox_test(orig_r_o, orig_r_inv_d, max_dist, mi->bbox_min, mi->bbox_max)) continue;

                const float3 r_o = TransformPoint(orig_r_o, &tr->inv_xform);
                const float3 r_d = TransformDirection(orig_r_d, &tr->inv_xform);
                const float3 inv_d = safe_invert(r_d);

                if (Traverse_MicroTree_Occlusion_Stackless(r_o, r_d, inv_d, max_dist, nodes, m->node_index, tris, tri_indices) < 1) {
                    return 0;
                }
            }

            last = cur; cur = n->parent;
            continue;
        }

        uint near = near_child(orig_rd, n);
        uint far = far_child(orig_rd, n);

        if (last == far) {
            last = cur; cur = n->parent;
            continue;
        }

        uint try_child = (last == n->parent) ? near : far;
        if (bbox_test(orig_r_o, orig_r_inv_d, max_dist, &nodes[try_child])) {
            last = cur; cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur; cur = n->parent;
            }
        }
    }

    return 1;
}

float Traverse_MacroTree_Occlusion_WithLocalStack(const float3 orig_r_o, const float3 orig_r_d, const float3 orig_r_inv_d, float max_dist,
                                                  __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, 
                                                  __global const mesh_t *meshes, __global const transform_t *transforms, 
                                                  __global const bvh_node_t *nodes, uint node_index, 
                                                  __global const tri_accel_t *tris, __global const uint *tri_indices,
                                                  __local uint *stack) {
    const float *orig_rd = (const float *)&orig_r_d;
    
    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        __global const bvh_node_t *n = &nodes[cur];

        if (!bbox_test(orig_r_o, orig_r_inv_d, max_dist, n)) continue;

        if (!n->prim_count) {
            stack[stack_size++] = far_child(orig_rd, n);
            stack[stack_size++] = near_child(orig_rd, n);
        } else {
            for (uint i = n->prim_index; i < n->prim_index + n->prim_count; i++) {
                __global const mesh_instance_t *mi = &mesh_instances[mi_indices[i]];
                __global const mesh_t *m = &meshes[mi->mesh_index];
                __global const transform_t *tr = &transforms[mi->tr_index];

                if (!_bbox_test(orig_r_o, orig_r_inv_d, max_dist, mi->bbox_min, mi->bbox_max)) continue;

                const float3 r_o = TransformPoint(orig_r_o, &tr->inv_xform);
                const float3 r_d = TransformDirection(orig_r_d, &tr->inv_xform);
                const float3 inv_d = safe_invert(r_d);
                
                if (Traverse_MicroTree_Occlusion_WithLocalStack(r_o, r_d, inv_d, max_dist, nodes, m->node_index, tris, tri_indices, &stack[stack_size]) < 1) {
                    return 0;
                }
            }
        }
    }

    return 1;
}

float Traverse_MacroTree_Occlusion_WithPrivateStack(const float3 orig_r_o, const float3 orig_r_d, const float3 orig_r_inv_d, float max_dist,
                                                    __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, 
                                                    __global const mesh_t *meshes, __global const transform_t *transforms, 
                                                    __global const bvh_node_t *nodes, uint node_index, 
                                                    __global const tri_accel_t *tris, __global const uint *tri_indices, uint *stack) {
    const float *orig_rd = (const float *)&orig_r_d;
    
    uint stack_size = 0;
    stack[stack_size++] = node_index;

    while (stack_size) {
        uint cur = stack[--stack_size];

        __global const bvh_node_t *n = &nodes[cur];

        if (!bbox_test(orig_r_o, orig_r_inv_d, max_dist, n)) continue;

        if (!n->prim_count) {
            stack[stack_size++] = far_child(orig_rd, n);
            stack[stack_size++] = near_child(orig_rd, n);
        } else {
            for (uint i = n->prim_index; i < n->prim_index + n->prim_count; i++) {
                __global const mesh_instance_t *mi = &mesh_instances[mi_indices[i]];
                __global const mesh_t *m = &meshes[mi->mesh_index];
                __global const transform_t *tr = &transforms[mi->tr_index];

                if (!_bbox_test(orig_r_o, orig_r_inv_d, max_dist, mi->bbox_min, mi->bbox_max)) continue;

                const float3 r_o = TransformPoint(orig_r_o, &tr->inv_xform);
                const float3 r_d = TransformDirection(orig_r_d, &tr->inv_xform);
                const float3 inv_d = safe_invert(r_d);
                
                if (Traverse_MicroTree_Occlusion_WithPrivateStack(r_o, r_d, inv_d, max_dist, nodes, m->node_index, tris, tri_indices, &stack[stack_size]) < 1) {
                    return 0;
                }
            }
        }
    }

    return 1;
}

#undef near_child
#undef far_child

)"