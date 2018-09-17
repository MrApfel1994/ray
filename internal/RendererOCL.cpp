#include "RendererOCL.h"

#include <sys/types.h>
#include <sys/stat.h>

#include <chrono>
#include <random>
#include <string>
#include <utility>

#include "CoreOCL.h"
#include "Halton.h"
#include "SceneOCL.h"
#include "TextureAtlasOCL.h"

namespace Ray {
namespace Ocl {
const char *cl_src_types =
#include "kernels/types.cl"
    ;
const char *cl_src_primary_ray_gen =
#include "kernels/primary_ray_gen.cl"
    ;
const char *cl_src_intersect =
#include "kernels/intersect.cl"
    ;
const char *cl_src_traverse =
#include "kernels/traverse_bvh.cl"
    ;
const char *cl_src_trace =
#include "kernels/trace.cl"
    ;
const char *cl_src_sort =
#include "kernels/sort.cl"
    ;
const char *cl_src_texture =
#include "kernels/texture.cl"
    ;
const char *cl_src_shade =
#include "kernels/shade.cl"
    ;
const char *cl_src_postprocess =
#include "kernels/postprocess.cl"
    ;
const char *cl_src_transform =
#include "kernels/transform.cl"
    ;
}
}

#define USE_IMG_BUFFERS 1

Ray::Ocl::Renderer::Renderer(int w, int h, int platform_index, int device_index) : w_(w), h_(h), loaded_halton_(-1) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) throw std::runtime_error("Cannot create OpenCL renderer!");

    if (platform_index == -1) {
        platform_index = 0;
        for (size_t i = 0; i < platforms.size(); i++) {
            auto s = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
            if (s.find("NVIDIA") != std::string::npos || s.find("AMD") != std::string::npos) {
                platform_index = (int)i;
                break;
            }
        }
    }

    platform_ = platforms[platform_index];

    std::vector<cl::Device> devices;
    platform_.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) throw std::runtime_error("Cannot create OpenCL renderer!");

    if (device_index == -1) {
        device_index = 0;
    }

    device_ = devices[device_index];
    //if (device_ != devices[0]) throw std::runtime_error("Cannot create OpenCL renderer!");

    // get properties
    max_compute_units_ = device_.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    max_clock_ = device_.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    mem_size_ = device_.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

    max_work_group_size_ = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    scan_portion_ = max_work_group_size_;
    seg_scan_portion_ = std::min(max_work_group_size_, (size_t)64);

    // local group size 8x8 seems to be optimal for traversing BVH in most scenes
    trace_group_size_x_ = 8;
    trace_group_size_y_ = std::min((size_t)8, max_work_group_size_/8);

    if (device_.getInfo(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, &max_image_buffer_size_) != CL_SUCCESS) {
        max_image_buffer_size_ = 0;
    }

    tri_rast_x_ = tri_rast_y_ = 8;
    tri_bin_size_ = 1024;

    {
        // create context
        cl_int error = CL_SUCCESS;
        context_ = cl::Context(devices, nullptr, nullptr, nullptr, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        queue_ = cl::CommandQueue(context_, device_, cl::QueueProperties::None, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
    }

    {
        // load kernels

        std::string cl_src_defines;
        cl_src_defines += "#define TRI_W_BITS " + std::to_string(TRI_W_BITS) + "\n";
        cl_src_defines += "#define TRI_AXIS_ALIGNED_BIT " + std::to_string(TRI_AXIS_ALIGNED_BIT) + "\n";
        cl_src_defines += "#define TRI_INV_NORMAL_BIT " + std::to_string(TRI_INV_NORMAL_BIT) + "\n";
        cl_src_defines += "#define HIT_BIAS " + std::to_string(HIT_BIAS) + "f\n";
        cl_src_defines += "#define HIT_EPS " + std::to_string(HIT_EPS) + "f\n";
        cl_src_defines += "#define FLT_EPS " + std::to_string(FLT_EPS) + "f\n";
        cl_src_defines += "#define PI " + std::to_string(PI) + "f\n";
        cl_src_defines += "#define RAY_TERM_THRES " + std::to_string(RAY_TERM_THRES) + "f\n";
        cl_src_defines += "#define HALTON_SEQ_LEN " + std::to_string(HALTON_SEQ_LEN) + "\n";
        cl_src_defines += "#define HALTON_COUNT " + std::to_string(HALTON_COUNT) + "\n";
        cl_src_defines += "#define MAX_MIP_LEVEL " + std::to_string(MAX_MIP_LEVEL) + "\n";
        cl_src_defines += "#define NUM_MIP_LEVELS " + std::to_string(NUM_MIP_LEVELS) + "\n";
        cl_src_defines += "#define MAX_TEXTURE_SIZE " + std::to_string(MAX_TEXTURE_SIZE) + "\n";
        cl_src_defines += "#define MAX_MATERIAL_TEXTURES " + std::to_string(MAX_MATERIAL_TEXTURES) + "\n";
        cl_src_defines += "#define DiffuseMaterial " + std::to_string(DiffuseMaterial) + "\n";
        cl_src_defines += "#define GlossyMaterial " + std::to_string(GlossyMaterial) + "\n";
        cl_src_defines += "#define RefractiveMaterial " + std::to_string(RefractiveMaterial) + "\n";
        cl_src_defines += "#define EmissiveMaterial " + std::to_string(EmissiveMaterial) + "\n";
        cl_src_defines += "#define MixMaterial " + std::to_string(MixMaterial) + "\n";
        cl_src_defines += "#define TransparentMaterial " + std::to_string(TransparentMaterial) + "\n";
        cl_src_defines += "#define MAIN_TEXTURE " + std::to_string(MAIN_TEXTURE) + "\n";
        cl_src_defines += "#define NORMALS_TEXTURE " + std::to_string(NORMALS_TEXTURE) + "\n";
        cl_src_defines += "#define MIX_MAT1 " + std::to_string(MIX_MAT1) + "\n";
        cl_src_defines += "#define MIX_MAT2 " + std::to_string(MIX_MAT2) + "\n";
        cl_src_defines += "#define SCAN_PORTION " + std::to_string(scan_portion_) + "\n";
        cl_src_defines += "#define SEG_SCAN_PORTION " + std::to_string(seg_scan_portion_) + "\n";
        cl_src_defines += "#define TRACE_GROUP_SIZE_X " + std::to_string(trace_group_size_x_) + "\n";
        cl_src_defines += "#define TRACE_GROUP_SIZE_Y " + std::to_string(trace_group_size_y_) + "\n";
        cl_src_defines += "#define CAM_USE_TENT_FILTER " + std::to_string(CAM_USE_TENT_FILTER) + "\n";
        cl_src_defines += "#define MAX_STACK_SIZE " + std::to_string(MAX_STACK_SIZE) + "\n";
        cl_src_defines += "#define LIGHT_ATTEN_CUTOFF " + std::to_string(LIGHT_ATTEN_CUTOFF) + "\n";
        cl_src_defines += "#define SkipDirectLight " + std::to_string(SkipDirectLight) + "\n";
        cl_src_defines += "#define SkipIndirectLight " + std::to_string(SkipIndirectLight) + "\n";
        cl_src_defines += "#define LightingOnly " + std::to_string(LightingOnly) + "\n";
        cl_src_defines += "#define NoBackground " + std::to_string(NoBackground) + "\n";
        cl_src_defines += "#define TRI_RAST_X " + std::to_string(tri_rast_x_) + "\n";
        cl_src_defines += "#define TRI_RAST_Y " + std::to_string(tri_rast_y_) + "\n";
        cl_src_defines += "#define SLAB_BIT_POS_X " + std::to_string(SLAB_BIT_POS_X) + "\n";
        cl_src_defines += "#define SLAB_BIT_NEG_X " + std::to_string(SLAB_BIT_NEG_X) + "\n";
        cl_src_defines += "#define SLAB_BIT_POS_Y " + std::to_string(SLAB_BIT_POS_Y) + "\n";
        cl_src_defines += "#define SLAB_BIT_NEG_Y " + std::to_string(SLAB_BIT_NEG_Y) + "\n";
        cl_src_defines += "#define SLAB_BIT_POS_Z " + std::to_string(SLAB_BIT_POS_Z) + "\n";
        cl_src_defines += "#define SLAB_BIT_NEG_Z " + std::to_string(SLAB_BIT_NEG_Z) + "\n";
        cl_src_defines += "#define SPACE_AXIS_BITS " + std::to_string(SPACE_AXIS_BITS) + "\n";

        cl_int error = CL_SUCCESS;
        cl::Program::Sources srcs = {
            cl_src_defines,
            cl_src_types, cl_src_transform, cl_src_primary_ray_gen,
            cl_src_intersect, cl_src_traverse, cl_src_trace, cl_src_sort,
            cl_src_texture, cl_src_shade, cl_src_postprocess
        };

        program_ = cl::Program(context_, srcs, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        //std::string build_opts = "-Werror -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math ";// = "-cl-opt-disable ";
        std::string build_opts = "-cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math ";// = "-cl-opt-disable ";

        struct stat info = { 0 };
        if (stat("./.dumps", &info) == 0 && info.st_mode & S_IFDIR) {
            build_opts += "-save-temps=./.dumps/ ";
        }

        error = program_.build(build_opts.c_str());
        if (error == CL_INVALID_BUILD_OPTIONS) {
            // -cl-strict-aliasing not supported sometimes, try to build without it
            build_opts = "-Werror -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -cl-std=CL1.2 ";
            program_ = cl::Program(context_, srcs, &error);
            if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
            error = program_.build(build_opts.c_str());
        }

        if (error != CL_SUCCESS) {
            auto build_log = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_);
#if defined(_MSC_VER)
            __debugbreak();
#endif
            throw std::runtime_error("Cannot create OpenCL renderer!");
        }

        prim_rays_gen_kernel_ = cl::Kernel(program_, "GeneratePrimaryRays", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        sample_mesh_reset_bins_kernel_ = cl::Kernel(program_, "SampleMeshInTextureSpace_ResetBins", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        sample_mesh_bin_stage_kernel_ = cl::Kernel(program_, "SampleMeshInTextureSpace_BinStage", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        sample_mesh_raster_stage_kernel_ = cl::Kernel(program_, "SampleMeshInTextureSpace_RasterStage", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        texture_debug_page_kernel_ = cl::Kernel(program_, "TextureDebugPage", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        shade_primary_kernel_ = cl::Kernel(program_, "ShadePrimary", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        shade_secondary_kernel_ = cl::Kernel(program_, "ShadeSecondary", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        trace_primary_rays_kernel_ = cl::Kernel(program_, "TracePrimaryRays", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        trace_primary_rays_img_kernel_ = cl::Kernel(program_, "TracePrimaryRaysImg", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        compute_ray_hashes_kernel_ = cl::Kernel(program_, "ComputeRayHashes", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        set_head_flags_kernel_ = cl::Kernel(program_, "SetHeadFlags", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        excl_scan_kernel_ = cl::Kernel(program_, "ExclusiveScan", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        incl_scan_kernel_ = cl::Kernel(program_, "InclusiveScan", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        excl_seg_scan_kernel_ = cl::Kernel(program_, "ExclusiveSegScan", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        incl_seg_scan_kernel_ = cl::Kernel(program_, "InclusiveSegScan", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        init_chunk_hash_and_base_kernel_ = cl::Kernel(program_, "InitChunkHashAndBase", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        init_chunk_size_kernel_ = cl::Kernel(program_, "InitChunkSize", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        init_skel_and_head_flags_kernel_ = cl::Kernel(program_, "InitSkeletonAndHeadFlags", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        init_count_table_kernel_ = cl::Kernel(program_, "InitCountTable", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        write_sorted_chunks_kernel_ = cl::Kernel(program_, "WriteSortedChunks", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        add_partial_sums_kernel_ = cl::Kernel(program_, "AddPartialSums", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        add_seg_partial_sums_kernel_ = cl::Kernel(program_, "AddSegPartialSums", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        reorder_rays_kernel_ = cl::Kernel(program_, "ReorderRays", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        trace_secondary_rays_kernel_ = cl::Kernel(program_, "TraceSecondaryRays", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        trace_secondary_rays_img_kernel_ = cl::Kernel(program_, "TraceSecondaryRaysImg", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        mix_incremental_kernel_ = cl::Kernel(program_, "MixIncremental", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
        post_process_kernel_ = cl::Kernel(program_, "PostProcess", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

#if !defined(NDEBUG)
        cl::Kernel types_check = cl::Kernel(program_, "TypesCheck", &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        char buf[512];
        int argc = 0;
        if (types_check.setArg(argc++, sizeof(ray_packet_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(Ocl::camera_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(tri_accel_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(hit_data_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(bvh_node_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(vertex_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(mesh_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(mesh_instance_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(transform_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(texture_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(material_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(light_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(environment_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(ray_chunk_t), buf) != CL_SUCCESS ||
                types_check.setArg(argc++, sizeof(pass_info_t), buf) != CL_SUCCESS) {
#if defined(_MSC_VER)
            __debugbreak();
#endif
            throw std::runtime_error("Cannot create OpenCL renderer!");
        }
#endif
    }

    {
        // create buffers
        cl_int error = CL_SUCCESS;
        Resize(w, h);

        //secondary_inters_count_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &error);
        //if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        secondary_rays_count_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        std::vector<pixel_color_t> color_table;
        /*for (int i = 0; i < 256; i++) {
            color_table.push_back({ Ray::U_0_p1(), Ray::U_0_p1(), Ray::U_0_p1(), 1 });
        }*/
        for (int i = 0; i < 64; i++) {
            color_table.push_back( { float(i) / 63, float(i) / 63, float(i) / 63, 1 });
        }

        color_table_buf_ = cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(pixel_color_t) * color_table.size(), nullptr, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        error = queue_.enqueueWriteBuffer(color_table_buf_, CL_TRUE, 0, sizeof(pixel_color_t) * color_table.size(), &color_table[0]);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

        halton_seq_buf_ = cl::Buffer(context_, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(float) * HALTON_COUNT * HALTON_SEQ_LEN, nullptr, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");
    }

    auto rand_func = std::bind(std::uniform_int_distribution<int>(), std::mt19937(0));
    permutations_ = Ray::ComputeRadicalInversePermutations(g_primes, PrimesCount, rand_func);
}

void Ray::Ocl::Renderer::Resize(int w, int h) {
    const int num_pixels = w * h;

    cl_int error = CL_SUCCESS;
    prim_rays_buf_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS, sizeof(ray_packet_t) * num_pixels, nullptr, &error);
    secondary_rays_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(ray_packet_t) * num_pixels, nullptr, &error);
    prim_inters_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(hit_data_t) * num_pixels, nullptr, &error);
    
    const size_t min_scan_portion = std::min(scan_portion_, seg_scan_portion_);
    const size_t max_scan_portion = std::max(scan_portion_, seg_scan_portion_);

    const size_t _rem = num_pixels % max_scan_portion;
    const size_t _num = num_pixels + (_rem == 0 ? 0 : (max_scan_portion - _rem));
    
    ray_hashes_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num, nullptr, &error);
    head_flags_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(uint32_t) * _num, nullptr, &error);
    scan_values_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(uint32_t) * _num, nullptr, &error);

    chunks_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(ray_chunk_t) * _num, nullptr, &error);
    chunks2_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(ray_chunk_t) * _num, nullptr, &error);
    counters_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(ray_chunk_t) * _num, nullptr, &error);

    const size_t _rem1 = (_num / min_scan_portion) % max_scan_portion;
    const size_t _num1 = _num / min_scan_portion + (_rem1 == 0 ? 0 : (max_scan_portion - _rem1));
    partial_sums_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num1, nullptr, &error);
    partial_flags_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num1, nullptr, &error);
    scan_values2_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num1, nullptr, &error);

    const size_t _rem2 = (_num1 / min_scan_portion) % max_scan_portion;
    const size_t _num2 = _num1 / min_scan_portion + (_rem2 == 0 ? 0 : (max_scan_portion - _rem2));
    partial_sums2_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num2, nullptr, &error);
    partial_flags2_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num2, nullptr, &error);
    scan_values3_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num2, nullptr, &error);

    const size_t _rem3 = (_num2 / min_scan_portion) % max_scan_portion;
    const size_t _num3 = _num2 / min_scan_portion + (_rem3 == 0 ? 0 : (max_scan_portion - _rem3));
    partial_sums3_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num3, nullptr, &error);
    partial_flags3_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num3, nullptr, &error);
    scan_values4_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num3, nullptr, &error);

    const size_t _rem4 = (_num3 / min_scan_portion) % max_scan_portion;
    const size_t _num4 = _num3 / min_scan_portion + (_rem4 == 0 ? 0 : (max_scan_portion - _rem4));
    partial_sums4_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num4, nullptr, &error);
    partial_flags4_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num4, nullptr, &error);

    skeleton_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _num, nullptr, &error);

    {
        size_t _w = (w / tri_rast_x_) + ((w % tri_rast_x_) == 0 ? 0 : 1),
               _h = (h / tri_rast_y_) + ((h % tri_rast_y_) == 0 ? 0 : 1);
        tri_bin_buf_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * _w * _h * tri_bin_size_, nullptr, &error);
    }

    temp_buf_ = cl::Image2D(context_, CL_MEM_READ_WRITE, cl::ImageFormat { CL_RGBA, CL_FLOAT }, (size_t)w, (size_t)h, 0, nullptr, &error);
    clean_buf_ = cl::Image2D(context_, CL_MEM_READ_WRITE, cl::ImageFormat { CL_RGBA, CL_FLOAT }, (size_t)w, (size_t)h, 0, nullptr, &error);
    final_buf_ = cl::Image2D(context_, CL_MEM_READ_WRITE, cl::ImageFormat { CL_RGBA, CL_FLOAT }, (size_t)w, (size_t)h, 0, nullptr, &error);
    if (error != CL_SUCCESS) throw std::runtime_error("Cannot create OpenCL renderer!");

    frame_pixels_.resize((size_t)4 * w * h);

    w_ = w;
    h_ = h;
}

void Ray::Ocl::Renderer::Clear(const pixel_color_t &c) {
    static_assert(sizeof(pixel_color_t) == sizeof(cl_float4), "!");
    queue_.enqueueFillImage(clean_buf_, *(cl_float4 *)&c, {}, { (size_t)w_, (size_t)h_, 1 });
    queue_.enqueueFillImage(final_buf_, *(cl_float4 *)&c, {}, { (size_t)w_, (size_t)h_, 1 });
}

std::shared_ptr<Ray::SceneBase> Ray::Ocl::Renderer::CreateScene() {
#if USE_IMG_BUFFERS
    return std::make_shared<Ocl::Scene>(context_, queue_, max_image_buffer_size_);
#else
    return std::make_shared<Ocl::Scene>(context_, queue_, 0);
#endif
}

void Ray::Ocl::Renderer::RenderScene(const std::shared_ptr<SceneBase> &_s, RegionContext &region) {
    auto s = std::dynamic_pointer_cast<Ocl::Scene>(_s);
    if (!s) return;

    uint32_t macro_tree_root = s->macro_nodes_start_;
    bvh_node_t root_node;
    s->nodes_.Get(macro_tree_root, root_node);

    cl_float3 root_min = { root_node.bbox[0][0], root_node.bbox[0][1], root_node.bbox[0][2] },
              root_max = { root_node.bbox[1][0], root_node.bbox[1][1], root_node.bbox[1][2] };
    cl_float3 cell_size = { (root_max.s[0] - root_min.s[0]) / 255, (root_max.s[1] - root_min.s[1]) / 255, (root_max.s[2] - root_min.s[2]) / 255 };

    region.iteration++;
    if (!region.halton_seq || region.iteration % HALTON_SEQ_LEN == 0) {
        UpdateHaltonSequence(region.iteration, region.halton_seq);
    }

    if (region.iteration != loaded_halton_) {
        if (CL_SUCCESS != queue_.enqueueWriteBuffer(halton_seq_buf_, CL_TRUE, 0, sizeof(float) * HALTON_COUNT * HALTON_SEQ_LEN, &region.halton_seq[0])) {
            return;
        }
        loaded_halton_ = region.iteration;
    }

    const auto &cam = s->cams_[s->current_cam()].cam;

    pass_info_t pass_info;

    pass_info.iteration = region.iteration;
    pass_info.bounce = 2;
    pass_info.flags = cam.pass_flags;

    Ray::Ocl::camera_t cl_cam = { cam };

    cl_int error = CL_SUCCESS;

    const auto time_start = std::chrono::high_resolution_clock::now();

    std::chrono::time_point<std::chrono::high_resolution_clock> time_after_ray_gen;

    if (cam.type != Geo) {
        if (!kernel_GeneratePrimaryRays((cl_int)region.iteration, cl_cam, region.rect(), w_, h_, halton_seq_buf_, prim_rays_buf_)) return;

        queue_.finish();
        time_after_ray_gen = std::chrono::high_resolution_clock::now();

        if (s->nodes_.img_buf().get() != nullptr) {
            if (!kernel_TracePrimaryRaysImg(prim_rays_buf_, region.rect(), w_,
                                            s->mesh_instances_.buf(), s->mi_indices_.buf(), s->meshes_.buf(), s->transforms_.buf(),
                                            s->nodes_.img_buf(), (cl_uint)s->macro_nodes_start_, s->tris_.buf(), s->tri_indices_.buf(), prim_inters_buf_)) return;
        } else {
            if (!kernel_TracePrimaryRays(prim_rays_buf_, region.rect(), w_,
                                         s->mesh_instances_.buf(), s->mi_indices_.buf(), s->meshes_.buf(), s->transforms_.buf(),
                                         s->nodes_.buf(), (cl_uint)s->macro_nodes_start_, s->tris_.buf(), s->tri_indices_.buf(), prim_inters_buf_)) return;
        }
    } else {
        if (!kernel_SampleMesh_ResetBins(w_, h_, tri_bin_buf_)) return;

        mesh_instance_t mi;
        if (queue_.enqueueReadBuffer(s->mesh_instances_.buf(), CL_TRUE, cam.mi_index * sizeof(mesh_instance_t), sizeof(mesh_instance_t), &mi) != CL_SUCCESS) {
            return;
        }

        mesh_t mesh;
        if (queue_.enqueueReadBuffer(s->meshes_.buf(), CL_TRUE, mi.mesh_index * sizeof(mesh), sizeof(mesh), &mesh) != CL_SUCCESS) {
            return;
        }

        if (!kernel_SampleMesh_BinStage((cl_int)cam.uv_index, mesh.tris_index, mesh.tris_count, s->vtx_indices_.buf(), s->vertices_.buf(), w_, h_, tri_bin_buf_)) return;

        if (!kernel_SampleMesh_RasterStage((cl_int)cam.uv_index, (cl_int)region.iteration, (cl_uint)mi.tr_index, s->transforms_.buf(), s->vtx_indices_.buf(),
                                           s->vertices_.buf(), (cl_int)w_, (cl_int)h_, halton_seq_buf_, tri_bin_buf_,
                                           prim_rays_buf_, prim_inters_buf_)) return;

        time_after_ray_gen = std::chrono::high_resolution_clock::now();
    }

    cl_int secondary_rays_count = 0;
    if (queue_.enqueueWriteBuffer(secondary_rays_count_buf_, CL_TRUE, 0, sizeof(cl_int),
                                    &secondary_rays_count) != CL_SUCCESS) return;

    queue_.finish();
    const auto time_after_prim_trace = std::chrono::high_resolution_clock::now();

    if (!kernel_ShadePrimary(pass_info, halton_seq_buf_, region.rect(), w_,
                                prim_inters_buf_, prim_rays_buf_,
                                s->mesh_instances_.buf(), s->mi_indices_.buf(), s->meshes_.buf(),
                                s->transforms_.buf(), s->vtx_indices_.buf(), s->vertices_.buf(),
                                s->nodes_.buf(), (cl_uint)s->macro_nodes_start_, s->tris_.buf(), s->tri_indices_.buf(),
                                s->env_, s->materials_.buf(), s->textures_.buf(), s->texture_atlas_.atlas(),
                                s->lights_.buf(), s->li_indices_.buf(), (cl_uint)s->light_nodes_start_,
                                temp_buf_, secondary_rays_buf_, secondary_rays_count_buf_)) return;
    
    if (queue_.enqueueReadBuffer(secondary_rays_count_buf_, CL_TRUE, 0, sizeof(cl_int),
                                    &secondary_rays_count) != CL_SUCCESS) return;

    queue_.finish();
    const auto time_after_prim_shade = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> secondary_sort_time{}, secondary_trace_time{}, secondary_shade_time{};

    for (int bounce = 0; bounce < MAX_BOUNCES && secondary_rays_count && !(pass_info.flags & SkipIndirectLight); bounce++) {
        auto time_secondary_sort_start = std::chrono::high_resolution_clock::now();

        if (secondary_rays_count > (cl_int)scan_portion_ * 64) {
            if (!SortRays(secondary_rays_buf_, secondary_rays_count, root_min, cell_size, ray_hashes_buf_, head_flags_buf_,
                          partial_sums_buf_, partial_sums2_buf_, partial_sums3_buf_, partial_sums4_buf_, partial_flags_buf_,
                          partial_flags2_buf_, partial_flags3_buf_, partial_flags4_buf_, scan_values_buf_, scan_values2_buf_,
                          scan_values3_buf_, scan_values4_buf_, chunks_buf_, chunks2_buf_, counters_buf_, skeleton_buf_, prim_rays_buf_)) return;
            std::swap(prim_rays_buf_, secondary_rays_buf_);
        }

        queue_.finish();
        auto time_secondary_trace_start = std::chrono::high_resolution_clock::now();

        if (s->nodes_.img_buf().get() != nullptr) {
            if (!kernel_TraceSecondaryRaysImg(secondary_rays_buf_, secondary_rays_count,
                s->mesh_instances_.buf(), s->mi_indices_.buf(), s->meshes_.buf(), s->transforms_.buf(),
                s->nodes_.img_buf(), (cl_uint)s->macro_nodes_start_, s->tris_.buf(), s->tri_indices_.buf(), prim_inters_buf_)) return;
        } else {
            if (!kernel_TraceSecondaryRays(secondary_rays_buf_, secondary_rays_count,
                s->mesh_instances_.buf(), s->mi_indices_.buf(), s->meshes_.buf(), s->transforms_.buf(),
                s->nodes_.buf(), (cl_uint)s->macro_nodes_start_, s->tris_.buf(), s->tri_indices_.buf(), prim_inters_buf_)) return;
        }

        cl_int new_secondary_rays_count = 0;
        if (queue_.enqueueWriteBuffer(secondary_rays_count_buf_, CL_TRUE, 0, sizeof(cl_int), &new_secondary_rays_count) != CL_SUCCESS) return;

#if 0
        pixel_color_t c = { 0, 0, 0, 0 };
        queue_.enqueueFillImage(temp_buf_, *(cl_float4 *)&c, {}, { (size_t)w_, (size_t)h_, 1 });
#endif
        queue_.finish();
        auto time_secondary_shade_start = std::chrono::high_resolution_clock::now();

        if (queue_.enqueueCopyImage(temp_buf_, final_buf_, { 0, 0, 0 }, { 0, 0, 0 },
    { (size_t)w_, (size_t)h_, 1 }) != CL_SUCCESS) return;

        pass_info.bounce = bounce + 3;

        if (!kernel_ShadeSecondary(pass_info, halton_seq_buf_, prim_inters_buf_, secondary_rays_buf_, (int)secondary_rays_count, w_, h_,
                                    s->mesh_instances_.buf(), s->mi_indices_.buf(), s->meshes_.buf(),
                                    s->transforms_.buf(), s->vtx_indices_.buf(), s->vertices_.buf(),
                                    s->nodes_.buf(), (cl_uint)s->macro_nodes_start_, s->tris_.buf(), s->tri_indices_.buf(),
                                    s->env_, s->materials_.buf(), s->textures_.buf(), s->texture_atlas_.atlas(),
                                    s->lights_.buf(), s->li_indices_.buf(), (cl_uint)s->light_nodes_start_,
                                    final_buf_, temp_buf_, prim_rays_buf_, secondary_rays_count_buf_)) return;

        if (queue_.enqueueReadBuffer(secondary_rays_count_buf_, CL_TRUE, 0, sizeof(cl_int),
                                     &secondary_rays_count) != CL_SUCCESS) return;

        queue_.finish();
        auto time_secondary_shade_end = std::chrono::high_resolution_clock::now();

        secondary_sort_time += std::chrono::duration<double, std::micro>{ time_secondary_trace_start - time_secondary_sort_start };
        secondary_trace_time += std::chrono::duration<double, std::micro>{ time_secondary_shade_start - time_secondary_trace_start };
        secondary_shade_time += std::chrono::duration<double, std::micro>{ time_secondary_shade_end - time_secondary_shade_start };

        std::swap(final_buf_, temp_buf_);
        std::swap(secondary_rays_buf_, prim_rays_buf_);
    }

    stats_.time_primary_ray_gen_us += (unsigned long long)std::chrono::duration<double, std::micro>{ time_after_ray_gen - time_start }.count();
    stats_.time_primary_trace_us += (unsigned long long)std::chrono::duration<double, std::micro>{ time_after_prim_trace - time_after_ray_gen }.count();
    stats_.time_primary_shade_us += (unsigned long long)std::chrono::duration<double, std::micro>{ time_after_prim_shade - time_after_prim_trace }.count();
    stats_.time_secondary_sort_us += (unsigned long long)secondary_sort_time.count();
    stats_.time_secondary_trace_us += (unsigned long long)secondary_trace_time.count();
    stats_.time_secondary_shade_us += (unsigned long long)secondary_shade_time.count();

    float k = 1.0f / region.iteration;

    if (!kernel_MixIncremental(clean_buf_, temp_buf_, (cl_float)k, final_buf_)) return;
    std::swap(final_buf_, clean_buf_);

    if (!kernel_Postprocess(clean_buf_, w_, h_, (cl_float)cam.gamma, final_buf_)) return;

    error = queue_.enqueueReadImage(final_buf_, CL_TRUE, {}, { (size_t)w_, (size_t)h_, 1 }, 0, 0, &frame_pixels_[0]);
}

bool Ray::Ocl::Renderer::kernel_GeneratePrimaryRays(const cl_int iteration, const Ray::Ocl::camera_t &cam, const Ray::rect_t &rect, cl_int w, cl_int h, const cl::Buffer &halton, const cl::Buffer &out_rays) {
    cl_uint argc = 0;
    if (prim_rays_gen_kernel_.setArg(argc++, iteration) != CL_SUCCESS ||
            prim_rays_gen_kernel_.setArg(argc++, cam) != CL_SUCCESS ||
            prim_rays_gen_kernel_.setArg(argc++, w) != CL_SUCCESS ||
            prim_rays_gen_kernel_.setArg(argc++, h) != CL_SUCCESS ||
            prim_rays_gen_kernel_.setArg(argc++, halton) != CL_SUCCESS ||
            prim_rays_gen_kernel_.setArg(argc++, out_rays) != CL_SUCCESS) {
        return false;
    }
    return CL_SUCCESS == queue_.enqueueNDRangeKernel(prim_rays_gen_kernel_, cl::NDRange{ (size_t)rect.x, (size_t)rect.y }, cl::NDRange{ (size_t)rect.w, (size_t)rect.h });
}

bool Ray::Ocl::Renderer::kernel_SampleMesh_ResetBins(cl_int w, cl_int h, const cl::Buffer &tri_bin_buf) {
    cl_uint argc = 0;
    if (sample_mesh_reset_bins_kernel_.setArg(argc++, tri_bin_buf) != CL_SUCCESS) {
        return false;
    }

    size_t _w = (w / tri_rast_x_) + ((w % tri_rast_x_) == 0 ? 0 : 1),
           _h = (h / tri_rast_y_) + ((h % tri_rast_y_) == 0 ? 0 : 1);

    return CL_SUCCESS == queue_.enqueueNDRangeKernel(sample_mesh_reset_bins_kernel_, { 0 }, { _w * _h });
}

bool Ray::Ocl::Renderer::kernel_SampleMesh_BinStage(cl_int uv_layer, uint32_t tris_index, uint32_t tris_count, const cl::Buffer &vtx_indices,
                                                    const cl::Buffer &vertices, cl_int w, cl_int h, const cl::Buffer &tri_bin_buf) {
    cl_uint argc = 0;
    if (sample_mesh_bin_stage_kernel_.setArg(argc++, uv_layer) != CL_SUCCESS ||
        sample_mesh_bin_stage_kernel_.setArg(argc++, (cl_uint)tris_index) != CL_SUCCESS ||
        sample_mesh_bin_stage_kernel_.setArg(argc++, vtx_indices) != CL_SUCCESS ||
        sample_mesh_bin_stage_kernel_.setArg(argc++, vertices) != CL_SUCCESS ||
        sample_mesh_bin_stage_kernel_.setArg(argc++, w) != CL_SUCCESS ||
        sample_mesh_bin_stage_kernel_.setArg(argc++, h) != CL_SUCCESS ||
        sample_mesh_bin_stage_kernel_.setArg(argc++, tri_bin_buf) != CL_SUCCESS) {
        return false;
    }

    return CL_SUCCESS == queue_.enqueueNDRangeKernel(sample_mesh_bin_stage_kernel_, { 0 }, { (size_t)tris_count });
}

bool Ray::Ocl::Renderer::kernel_SampleMesh_RasterStage(cl_int uv_layer, cl_int iteration, cl_uint tr_index, const cl::Buffer &transforms,
                                                       const cl::Buffer &vtx_indices, const cl::Buffer &vertices, cl_int w, cl_int h,
                                                       const cl::Buffer &halton_seq, const cl::Buffer &tri_bin_buf,
                                                       const cl::Buffer &out_rays, const cl::Buffer &out_inters) {
    cl_uint argc = 0;
    if (sample_mesh_raster_stage_kernel_.setArg(argc++, uv_layer) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, iteration) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, tr_index) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, vtx_indices) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, vertices) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, w) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, h) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, halton_seq) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, tri_bin_buf) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, out_rays) != CL_SUCCESS ||
        sample_mesh_raster_stage_kernel_.setArg(argc++, out_inters) != CL_SUCCESS) {
        return false;
    }

    return CL_SUCCESS == queue_.enqueueNDRangeKernel(sample_mesh_raster_stage_kernel_, { 0, 0 }, { (size_t)w, (size_t)h });
}

bool Ray::Ocl::Renderer::kernel_TextureDebugPage(const cl::Image2DArray &textures, cl_int page, const cl::Image2D &frame_buf) {
    cl_uint argc = 0;
    if (texture_debug_page_kernel_.setArg(argc++, textures) != CL_SUCCESS ||
            texture_debug_page_kernel_.setArg(argc++, page) != CL_SUCCESS ||
            texture_debug_page_kernel_.setArg(argc++, frame_buf) != CL_SUCCESS) {
        return false;
    }

    auto w = frame_buf.getImageInfo<CL_IMAGE_WIDTH>(),
         h = frame_buf.getImageInfo<CL_IMAGE_HEIGHT>();

    return CL_SUCCESS == queue_.enqueueNDRangeKernel(texture_debug_page_kernel_, cl::NullRange, cl::NDRange { (size_t)w, (size_t)h });
}

bool Ray::Ocl::Renderer::kernel_ShadePrimary(const pass_info_t pi, const cl::Buffer &halton,
        const Ray::rect_t &rect, cl_int w,
        const cl::Buffer &intersections, const cl::Buffer &rays,
        const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes,
        const cl::Buffer &transforms, const cl::Buffer &vtx_indices, const cl::Buffer &vertices,
        const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices,
        const environment_t &env, const cl::Buffer &materials, const cl::Buffer &textures, const cl::Image2DArray &texture_atlas,
        const cl::Buffer &lights, const cl::Buffer &li_indices, cl_uint light_node_index, const cl::Image2D &frame_buf,
        const cl::Buffer &secondary_rays, const cl::Buffer &secondary_rays_count) {
    cl_uint argc = 0;
    if (shade_primary_kernel_.setArg(argc++, pi) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, halton) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, w) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, intersections) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, mesh_instances) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, mi_indices) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, meshes) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, vtx_indices) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, vertices) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, nodes) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, node_index) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, tris) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, tri_indices) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, env) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, materials) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, textures) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, texture_atlas) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, lights) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, li_indices) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, light_node_index) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, frame_buf) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, secondary_rays) != CL_SUCCESS ||
            shade_primary_kernel_.setArg(argc++, secondary_rays_count) != CL_SUCCESS) {
        return false;
    }

    int border_x = rect.w % trace_group_size_x_, border_y = rect.h % trace_group_size_y_;

    cl::NDRange global = { (size_t)(rect.w - border_x), (size_t)(rect.h - border_y) };
    cl::NDRange local = { trace_group_size_x_, trace_group_size_y_ };

    if (rect.w - border_x > 0 && rect.h - border_y > 0) {
        if (queue_.enqueueNDRangeKernel(shade_primary_kernel_, { (size_t)rect.x, (size_t)rect.y }, global, local) != CL_SUCCESS) {
            return false;
        }
    }

    if (border_x) {
        if (queue_.enqueueNDRangeKernel(shade_primary_kernel_, { (size_t)(rect.x + rect.w - border_x), (size_t)rect.y },
                                                               { (size_t)(border_x), (size_t)(rect.h - border_y) },
                                                               { (size_t)(border_x), trace_group_size_y_ }) != CL_SUCCESS) {
            return false;
        }
    }

    if (border_y) {
        if (queue_.enqueueNDRangeKernel(shade_primary_kernel_, { (size_t)rect.x, (size_t)(rect.y + rect.h - border_y) },
                                                               { (size_t)(rect.w - border_x), (size_t)(border_y) },
                                                               { trace_group_size_x_, (size_t)(border_y) }) != CL_SUCCESS) {
            return false;
        }

        if (border_x) {
            if (queue_.enqueueNDRangeKernel(shade_primary_kernel_, { (size_t)(rect.x + rect.w - border_x), (size_t)(rect.y + rect.h - border_y) },
                                                                   { (size_t)(border_x), (size_t)(border_y) }) != CL_SUCCESS) {
                return false;
            }
        }
    }

    return true;
}

bool Ray::Ocl::Renderer::kernel_ShadeSecondary(const pass_info_t pi, const cl::Buffer &halton,
        const cl::Buffer &intersections, const cl::Buffer &rays,
        int rays_count, int w, int h,
        const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes,
        const cl::Buffer &transforms, const cl::Buffer &vtx_indices, const cl::Buffer &vertices,
        const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices,
        const environment_t &env, const cl::Buffer &materials, const cl::Buffer &textures, const cl::Image2DArray &texture_atlas,
        const cl::Buffer &lights, const cl::Buffer &li_indices, cl_uint light_node_index,
        const cl::Image2D &frame_buf, const cl::Image2D &frame_buf2,
        const cl::Buffer &secondary_rays, const cl::Buffer &secondary_rays_count) {
    if (rays_count == 0) return true;

    cl_uint argc = 0;
    if (shade_secondary_kernel_.setArg(argc++, pi) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, halton) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, intersections) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, mesh_instances) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, mi_indices) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, meshes) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, vtx_indices) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, vertices) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, nodes) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, node_index) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, tris) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, tri_indices) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, env) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, materials) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, textures) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, texture_atlas) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, lights) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, li_indices) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, light_node_index) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, frame_buf) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, frame_buf2) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, secondary_rays) != CL_SUCCESS ||
            shade_secondary_kernel_.setArg(argc++, secondary_rays_count) != CL_SUCCESS) {
        return false;
    }

    size_t group_size = std::min((size_t)64, max_work_group_size_);

    int remaining = rays_count % group_size;

    cl::NDRange global = { (size_t)(rays_count - remaining) };
    cl::NDRange local = { group_size };

    if (rays_count - remaining > 0) {
        if (queue_.enqueueNDRangeKernel(shade_secondary_kernel_, cl::NullRange, global, local) != CL_SUCCESS) {
            return false;
        }
    }

    if (remaining) {
        if (queue_.enqueueNDRangeKernel(shade_secondary_kernel_, { (size_t)(rays_count - remaining) }, { (size_t)(remaining) }) != CL_SUCCESS) {
            return false;
        }
    }

    return true;
}

bool Ray::Ocl::Renderer::kernel_TracePrimaryRays(const cl::Buffer &rays, const Ray::rect_t &rect, cl_int w, const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
        const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections) {
    cl_uint argc = 0;
    if (trace_primary_rays_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, w) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, mesh_instances) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, mi_indices) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, meshes) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, nodes) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, node_index) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, tris) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, tri_indices) != CL_SUCCESS ||
            trace_primary_rays_kernel_.setArg(argc++, intersections) != CL_SUCCESS) {
        return false;
    }

    int border_x = rect.w % trace_group_size_x_, border_y = rect.h % trace_group_size_y_;

    cl::NDRange global = { (size_t)(rect.w - border_x), (size_t)(rect.h - border_y) };
    cl::NDRange local = { trace_group_size_x_, trace_group_size_y_ };

    if (rect.w - border_x > 0 && rect.h - border_y > 0) {
        if (queue_.enqueueNDRangeKernel(trace_primary_rays_kernel_, { (size_t)rect.x, (size_t)rect.y }, global, local) != CL_SUCCESS) {
            return false;
        }
    }

    if (border_x) {
        if (queue_.enqueueNDRangeKernel(trace_primary_rays_kernel_, { (size_t)(rect.x + rect.w - border_x), (size_t)rect.y },
                                                                    { (size_t)(border_x), (size_t)(rect.h - border_y) },
                                                                    { (size_t)(border_x), trace_group_size_y_ }) != CL_SUCCESS) {
            return false;
        }
    }

    if (border_y) {
        if (queue_.enqueueNDRangeKernel(trace_primary_rays_kernel_, { (size_t)rect.x, (size_t)(rect.y + rect.h - border_y) },
                                                                    { (size_t)(rect.w - border_x), (size_t)(border_y) },
                                                                    { trace_group_size_x_, (size_t)(border_y) }) != CL_SUCCESS) {
            return false;
        }

        if (border_x) {
            if (queue_.enqueueNDRangeKernel(trace_primary_rays_kernel_, { (size_t)(rect.x + rect.w - border_x), (size_t)(rect.y + rect.h - border_y) },
                                                                        { (size_t)(border_x), (size_t)(border_y) }) != CL_SUCCESS) {
                return false;
            }
        }
    }

    return true;
}

bool Ray::Ocl::Renderer::kernel_TracePrimaryRaysImg(const cl::Buffer &rays, const Ray::rect_t &rect, cl_int w,
                                                    const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
                                                    const cl::Image1DBuffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections) {
    cl_uint argc = 0;
    if (trace_primary_rays_img_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
        trace_primary_rays_img_kernel_.setArg(argc++, w) != CL_SUCCESS ||
        trace_primary_rays_img_kernel_.setArg(argc++, mesh_instances) != CL_SUCCESS ||
        trace_primary_rays_img_kernel_.setArg(argc++, mi_indices) != CL_SUCCESS ||
        trace_primary_rays_img_kernel_.setArg(argc++, meshes) != CL_SUCCESS ||
        trace_primary_rays_img_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
        trace_primary_rays_img_kernel_.setArg(argc++, nodes) != CL_SUCCESS ||
        trace_primary_rays_img_kernel_.setArg(argc++, node_index) != CL_SUCCESS ||
        trace_primary_rays_img_kernel_.setArg(argc++, tris) != CL_SUCCESS ||
        trace_primary_rays_img_kernel_.setArg(argc++, tri_indices) != CL_SUCCESS ||
        trace_primary_rays_img_kernel_.setArg(argc++, intersections) != CL_SUCCESS) {
        return false;
    }

    int border_x = rect.w % trace_group_size_x_, border_y = rect.h % trace_group_size_y_;

    cl::NDRange global = { (size_t)(rect.w - border_x), (size_t)(rect.h - border_y) };
    cl::NDRange local = { trace_group_size_x_, trace_group_size_y_ };

    if (rect.w - border_x > 0 && rect.h - border_y > 0) {
        if (queue_.enqueueNDRangeKernel(trace_primary_rays_img_kernel_, { (size_t)rect.x, (size_t)rect.y }, global, local) != CL_SUCCESS) {
            return false;
        }
    }

    if (border_x) {
        if (queue_.enqueueNDRangeKernel(trace_primary_rays_img_kernel_, { (size_t)(rect.x + rect.w - border_x), (size_t)rect.y },
                                                                        { (size_t)(border_x), (size_t)(rect.h - border_y) },
                                                                        { (size_t)(border_x), trace_group_size_y_ }) != CL_SUCCESS) {
            return false;
        }
    }

    if (border_y) {
        if (queue_.enqueueNDRangeKernel(trace_primary_rays_img_kernel_, { (size_t)rect.x, (size_t)(rect.y + rect.h - border_y) },
                                                                        { (size_t)(rect.w - border_x), (size_t)(border_y) },
                                                                        { trace_group_size_x_, (size_t)(border_y) } ) != CL_SUCCESS) {
            return false;
        }

        if (border_x) {
            if (queue_.enqueueNDRangeKernel(trace_primary_rays_img_kernel_, { (size_t)(rect.x + rect.w - border_x), (size_t)(rect.y + rect.h - border_y) },
                                                                            { (size_t)(border_x), (size_t)(border_y) }) != CL_SUCCESS) {
                return false;
            }
        }
    }

    return true;
}

bool Ray::Ocl::Renderer::kernel_TraceSecondaryRays(const cl::Buffer &rays, cl_int rays_count,
        const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
        const cl::Buffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections) {
    cl_uint argc = 0;
    if (trace_secondary_rays_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
        trace_secondary_rays_kernel_.setArg(argc++, mesh_instances) != CL_SUCCESS ||
        trace_secondary_rays_kernel_.setArg(argc++, mi_indices) != CL_SUCCESS ||
        trace_secondary_rays_kernel_.setArg(argc++, meshes) != CL_SUCCESS ||
        trace_secondary_rays_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
        trace_secondary_rays_kernel_.setArg(argc++, nodes) != CL_SUCCESS ||
        trace_secondary_rays_kernel_.setArg(argc++, node_index) != CL_SUCCESS ||
        trace_secondary_rays_kernel_.setArg(argc++, tris) != CL_SUCCESS ||
        trace_secondary_rays_kernel_.setArg(argc++, tri_indices) != CL_SUCCESS ||
        trace_secondary_rays_kernel_.setArg(argc++, intersections) != CL_SUCCESS) {
        return false;
    }

    size_t group_size = trace_group_size_x_ * trace_group_size_y_;

    int remaining = rays_count % group_size;

    cl::NDRange global = { (size_t)(rays_count - remaining) };
    cl::NDRange local = { (size_t)(group_size) };

    if (rays_count - remaining > 0) {
        if (queue_.enqueueNDRangeKernel(trace_secondary_rays_kernel_, cl::NullRange, global, local) != CL_SUCCESS) {
            return false;
        }
    }

    if (remaining) {
        if (queue_.enqueueNDRangeKernel(trace_secondary_rays_kernel_, { (size_t)(rays_count - remaining) }, { (size_t)(remaining) }) != CL_SUCCESS) {
            return false;
        }
    }

    return true;
}

bool Ray::Ocl::Renderer::kernel_TraceSecondaryRaysImg(const cl::Buffer &rays, cl_int rays_count,
                                                      const cl::Buffer &mesh_instances, const cl::Buffer &mi_indices, const cl::Buffer &meshes, const cl::Buffer &transforms,
                                                      const cl::Image1DBuffer &nodes, cl_uint node_index, const cl::Buffer &tris, const cl::Buffer &tri_indices, const cl::Buffer &intersections) {
    cl_uint argc = 0;
    if (trace_secondary_rays_img_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
        trace_secondary_rays_img_kernel_.setArg(argc++, mesh_instances) != CL_SUCCESS ||
        trace_secondary_rays_img_kernel_.setArg(argc++, mi_indices) != CL_SUCCESS ||
        trace_secondary_rays_img_kernel_.setArg(argc++, meshes) != CL_SUCCESS ||
        trace_secondary_rays_img_kernel_.setArg(argc++, transforms) != CL_SUCCESS ||
        trace_secondary_rays_img_kernel_.setArg(argc++, nodes) != CL_SUCCESS ||
        trace_secondary_rays_img_kernel_.setArg(argc++, node_index) != CL_SUCCESS ||
        trace_secondary_rays_img_kernel_.setArg(argc++, tris) != CL_SUCCESS ||
        trace_secondary_rays_img_kernel_.setArg(argc++, tri_indices) != CL_SUCCESS ||
        trace_secondary_rays_img_kernel_.setArg(argc++, intersections) != CL_SUCCESS) {
        return false;
    }

    size_t group_size = trace_group_size_x_ * trace_group_size_y_;

    int remaining = rays_count % group_size;

    cl::NDRange global = { (size_t)(rays_count - remaining) };
    cl::NDRange local = { (size_t)(group_size) };

    if (rays_count - remaining > 0) {
        if (queue_.enqueueNDRangeKernel(trace_secondary_rays_img_kernel_, cl::NullRange, global, local) != CL_SUCCESS) {
            return false;
        }
    }

    if (remaining) {
        if (queue_.enqueueNDRangeKernel(trace_secondary_rays_img_kernel_, { (size_t)(rays_count - remaining) }, { (size_t)(remaining) }) != CL_SUCCESS) {
            return false;
        }
    }

    return true;
}

bool Ray::Ocl::Renderer::kernel_ComputeRayHashes(const cl::Buffer &rays, cl_int rays_count, cl_float3 root_min, cl_float3 cell_size, const cl::Buffer &out_hashes) {
    cl_uint argc = 0;
    if (compute_ray_hashes_kernel_.setArg(argc++, rays) != CL_SUCCESS ||
        compute_ray_hashes_kernel_.setArg(argc++, root_min) != CL_SUCCESS ||
        compute_ray_hashes_kernel_.setArg(argc++, cell_size) != CL_SUCCESS ||
        compute_ray_hashes_kernel_.setArg(argc++, out_hashes) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)(rays_count) };

    return queue_.enqueueNDRangeKernel(compute_ray_hashes_kernel_, cl::NullRange, global, cl::NullRange) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_SetHeadFlags(const cl::Buffer &hashes, cl_int hashes_count, const cl::Buffer &out_head_flags) {
    cl_uint argc = 0;
    if (set_head_flags_kernel_.setArg(argc++, hashes) != CL_SUCCESS ||
        set_head_flags_kernel_.setArg(argc++, out_head_flags) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)hashes_count };
    cl::NDRange local = cl::NullRange;

    return queue_.enqueueNDRangeKernel(set_head_flags_kernel_, cl::NullRange, global, local) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_ExclusiveScan(const cl::Buffer &values, cl_int count, cl_int offset, cl_int stride,
                                              const cl::Buffer &out_scan_values, const cl::Buffer &out_partial_sums) {
    cl_uint argc = 0;
    if (excl_scan_kernel_.setArg(argc++, values) != CL_SUCCESS ||
        excl_scan_kernel_.setArg(argc++, offset) != CL_SUCCESS ||
        excl_scan_kernel_.setArg(argc++, stride) != CL_SUCCESS ||
        excl_scan_kernel_.setArg(argc++, out_scan_values) != CL_SUCCESS ||
        excl_scan_kernel_.setArg(argc++, out_partial_sums) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)(count) };
    cl::NDRange local = { scan_portion_ };

    return queue_.enqueueNDRangeKernel(excl_scan_kernel_, cl::NullRange, global, local) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_InclusiveScan(const cl::Buffer &values, cl_int count, const cl::Buffer &out_scan_values, const cl::Buffer &out_partial_sums) {
    cl_uint argc = 0;
    if (incl_scan_kernel_.setArg(argc++, values) != CL_SUCCESS ||
        incl_scan_kernel_.setArg(argc++, out_scan_values) != CL_SUCCESS ||
        incl_scan_kernel_.setArg(argc++, out_partial_sums) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)(count) };
    cl::NDRange local = { scan_portion_ };

    return queue_.enqueueNDRangeKernel(incl_scan_kernel_, cl::NullRange, global, local) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_ExclusiveSegScan(const cl::Buffer &values, const cl::Buffer &flags, cl_int count, const cl::Buffer &out_scan_values, const cl::Buffer &out_partial_sums) {
    cl_uint argc = 0;
    if (excl_seg_scan_kernel_.setArg(argc++, values) != CL_SUCCESS ||
        excl_seg_scan_kernel_.setArg(argc++, flags) != CL_SUCCESS ||
        excl_seg_scan_kernel_.setArg(argc++, out_scan_values) != CL_SUCCESS ||
        excl_seg_scan_kernel_.setArg(argc++, out_partial_sums) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)(count) };
    cl::NDRange local = { seg_scan_portion_ };

    return queue_.enqueueNDRangeKernel(excl_seg_scan_kernel_, cl::NullRange, global, local) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_InclusiveSegScan(const cl::Buffer &values, const cl::Buffer &flags, cl_int count, const cl::Buffer &out_scan_values,
                                                 const cl::Buffer &out_partial_sums, const cl::Buffer &out_partial_flags) {
    cl_uint argc = 0;
    if (incl_seg_scan_kernel_.setArg(argc++, values) != CL_SUCCESS ||
        incl_seg_scan_kernel_.setArg(argc++, flags) != CL_SUCCESS ||
        incl_seg_scan_kernel_.setArg(argc++, out_scan_values) != CL_SUCCESS ||
        incl_seg_scan_kernel_.setArg(argc++, out_partial_sums) != CL_SUCCESS ||
        incl_seg_scan_kernel_.setArg(argc++, out_partial_flags) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)(count) };
    cl::NDRange local = { seg_scan_portion_ };

    return queue_.enqueueNDRangeKernel(incl_seg_scan_kernel_, cl::NullRange, global, local) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_AddPartialSums(const cl::Buffer &values, cl_int count, const cl::Buffer &partial_sums) {
    cl_uint argc = 0;
    if (add_partial_sums_kernel_.setArg(argc++, values) != CL_SUCCESS ||
        add_partial_sums_kernel_.setArg(argc++, partial_sums) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)(count) };
    cl::NDRange local = { scan_portion_ };

    return queue_.enqueueNDRangeKernel(add_partial_sums_kernel_, cl::NullRange, global, local) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_AddSegPartialSums(const cl::Buffer &flags, const cl::Buffer &values, const cl::Buffer &partial_sums, cl_int count, cl_int _group_size) {
    cl_uint argc = 0;
    if (add_seg_partial_sums_kernel_.setArg(argc++, flags) != CL_SUCCESS ||
        add_seg_partial_sums_kernel_.setArg(argc++, values) != CL_SUCCESS ||
        add_seg_partial_sums_kernel_.setArg(argc++, partial_sums) != CL_SUCCESS ||
        add_seg_partial_sums_kernel_.setArg(argc++, _group_size) != CL_SUCCESS) {
        return false;
    }

    size_t group_size = std::min((size_t)32, max_work_group_size_);

    int remaining = count % group_size;

    cl::NDRange global = { (size_t)(count - remaining) };
    cl::NDRange local = { (size_t)(group_size) };

    if (count - remaining > 0) {
        if (queue_.enqueueNDRangeKernel(add_seg_partial_sums_kernel_, cl::NullRange, global, local) != CL_SUCCESS) {
            return false;
        }
    }

    if (remaining) {
        if (queue_.enqueueNDRangeKernel(add_seg_partial_sums_kernel_, { (size_t)(count - remaining) }, { (size_t)(remaining) }) != CL_SUCCESS) {
            return false;
        }
    }

    return true;
}

bool Ray::Ocl::Renderer::kernel_InitChunkHashAndBase(const cl::Buffer &chunks, cl_int count, const cl::Buffer &hash_values, const cl::Buffer &head_flags, const cl::Buffer &scan_values) {
    cl_uint argc = 0;
    if (init_chunk_hash_and_base_kernel_.setArg(argc++, chunks) != CL_SUCCESS ||
        init_chunk_hash_and_base_kernel_.setArg(argc++, hash_values) != CL_SUCCESS ||
        init_chunk_hash_and_base_kernel_.setArg(argc++, head_flags) != CL_SUCCESS ||
        init_chunk_hash_and_base_kernel_.setArg(argc++, scan_values) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)(count) };

    return queue_.enqueueNDRangeKernel(init_chunk_hash_and_base_kernel_, cl::NullRange, global, cl::NullRange) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_InitChunkSize(const cl::Buffer &chunks, cl_int count, cl_int ray_count) {
    cl_uint argc = 0;
    if (init_chunk_size_kernel_.setArg(argc++, chunks) != CL_SUCCESS ||
        init_chunk_size_kernel_.setArg(argc++, ray_count) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)count };

    return queue_.enqueueNDRangeKernel(init_chunk_size_kernel_, cl::NullRange, global, cl::NullRange) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_InitSkeletonAndHeadFlags(const cl::Buffer &scan_values, const cl::Buffer &chunks, cl_int count, const cl::Buffer &skeleton, const cl::Buffer &head_flags) {
    cl_uint argc = 0;
    if (init_skel_and_head_flags_kernel_.setArg(argc++, scan_values) != CL_SUCCESS ||
        init_skel_and_head_flags_kernel_.setArg(argc++, chunks) != CL_SUCCESS ||
        init_skel_and_head_flags_kernel_.setArg(argc++, skeleton) != CL_SUCCESS ||
        init_skel_and_head_flags_kernel_.setArg(argc++, head_flags) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)(count) };

    return queue_.enqueueNDRangeKernel(init_skel_and_head_flags_kernel_, cl::NullRange, global, cl::NullRange) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_InitCountTable(const cl::Buffer &chunks, cl_int count, cl_int group_size, cl_int shift, const cl::Buffer &counters) {
    cl_uint argc = 0;
    if (init_count_table_kernel_.setArg(argc++, chunks) != CL_SUCCESS ||
        init_count_table_kernel_.setArg(argc++, shift) != CL_SUCCESS ||
        init_count_table_kernel_.setArg(argc++, counters) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)count };
    cl::NDRange local = { (size_t)group_size };

    return queue_.enqueueNDRangeKernel(init_count_table_kernel_, cl::NullRange, global, local) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_WriteSortedChunks(const cl::Buffer &chunks_in, const cl::Buffer &offsets, const cl::Buffer &counts, cl_int count, cl_int shift, cl_int _group_size, const cl::Buffer &chunks_out) {
    cl_uint argc = 0;
    if (write_sorted_chunks_kernel_.setArg(argc++, chunks_in) != CL_SUCCESS ||
        write_sorted_chunks_kernel_.setArg(argc++, offsets) != CL_SUCCESS ||
        write_sorted_chunks_kernel_.setArg(argc++, counts) != CL_SUCCESS ||
        write_sorted_chunks_kernel_.setArg(argc++, shift) != CL_SUCCESS ||
        write_sorted_chunks_kernel_.setArg(argc++, _group_size) != CL_SUCCESS ||
        write_sorted_chunks_kernel_.setArg(argc++, count) != CL_SUCCESS ||
        write_sorted_chunks_kernel_.setArg(argc++, chunks_out) != CL_SUCCESS) {
        return false;
    }

    size_t group_size = std::min((size_t)8, max_work_group_size_);

    int remaining = count % group_size;

    if (count - remaining > 0) {
        if (queue_.enqueueNDRangeKernel(write_sorted_chunks_kernel_, cl::NullRange, { (size_t)(count - remaining) }, { group_size }) != CL_SUCCESS) {
            return false;
        }
    }

    if (remaining) {
        if (queue_.enqueueNDRangeKernel(write_sorted_chunks_kernel_, { (size_t)(count - remaining) }, { (size_t)(remaining) }) != CL_SUCCESS) {
            return false;
        }
    }

    return true;
}

bool Ray::Ocl::Renderer::kernel_ReorderRays(const cl::Buffer &in_rays, const cl::Buffer &in_indices, cl_int count, const cl::Buffer &out_rays) {
    cl_uint argc = 0;
    if (reorder_rays_kernel_.setArg(argc++, in_rays) != CL_SUCCESS ||
        reorder_rays_kernel_.setArg(argc++, in_indices) != CL_SUCCESS ||
        reorder_rays_kernel_.setArg(argc++, out_rays) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)count };

    return queue_.enqueueNDRangeKernel(reorder_rays_kernel_, cl::NullRange, global, cl::NullRange) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_MixIncremental(const cl::Image2D &fbuf1, const cl::Image2D &fbuf2, cl_float k, const cl::Image2D &res) {
    cl_uint argc = 0;
    if (mix_incremental_kernel_.setArg(argc++, fbuf1) != CL_SUCCESS ||
        mix_incremental_kernel_.setArg(argc++, fbuf2) != CL_SUCCESS ||
        mix_incremental_kernel_.setArg(argc++, k) != CL_SUCCESS ||
        mix_incremental_kernel_.setArg(argc++, res) != CL_SUCCESS) {
        return false;
    }

    const auto w = fbuf1.getImageInfo<CL_IMAGE_WIDTH>(),
               h = fbuf1.getImageInfo<CL_IMAGE_HEIGHT>();

    cl::NDRange global = { (size_t)w, (size_t)h };
    cl::NDRange local = cl::NullRange;

    return queue_.enqueueNDRangeKernel(mix_incremental_kernel_, cl::NullRange, global, local) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::kernel_Postprocess(const cl::Image2D &frame_buf, cl_int w, cl_int h, cl_float gamma, const cl::Image2D &out_pixels) {
    cl_uint argc = 0;
    if (post_process_kernel_.setArg(argc++, frame_buf) != CL_SUCCESS ||
            post_process_kernel_.setArg(argc++, w) != CL_SUCCESS ||
            post_process_kernel_.setArg(argc++, h) != CL_SUCCESS ||
            post_process_kernel_.setArg(argc++, gamma) != CL_SUCCESS ||
            post_process_kernel_.setArg(argc++, out_pixels) != CL_SUCCESS) {
        return false;
    }

    cl::NDRange global = { (size_t)w, (size_t)h };
    cl::NDRange local = cl::NullRange;//{ (size_t)8, std::min((size_t)8, max_work_group_size_ / 8) };

    return queue_.enqueueNDRangeKernel(post_process_kernel_, cl::NullRange, global, local) == CL_SUCCESS;
}

void Ray::Ocl::Renderer::UpdateHaltonSequence(int iteration, std::unique_ptr<float[]> &seq) {
    if (!seq) {
        seq.reset(new float[HALTON_COUNT * HALTON_SEQ_LEN]);
    }

    for (int i = 0; i < HALTON_SEQ_LEN; i++) {
        seq[2 * (i * HALTON_2D_COUNT + 0 ) + 0] = Ray::ScrambledRadicalInverse<2 >(&permutations_[0  ], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 0 ) + 1] = Ray::ScrambledRadicalInverse<3 >(&permutations_[2  ], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 1 ) + 0] = Ray::ScrambledRadicalInverse<5 >(&permutations_[5  ], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 1 ) + 1] = Ray::ScrambledRadicalInverse<7 >(&permutations_[10 ], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 2 ) + 0] = Ray::ScrambledRadicalInverse<11>(&permutations_[17 ], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 2 ) + 1] = Ray::ScrambledRadicalInverse<13>(&permutations_[28 ], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 3 ) + 0] = Ray::ScrambledRadicalInverse<17>(&permutations_[41 ], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 3 ) + 1] = Ray::ScrambledRadicalInverse<19>(&permutations_[58 ], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 4 ) + 0] = Ray::ScrambledRadicalInverse<23>(&permutations_[77 ], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 4 ) + 1] = Ray::ScrambledRadicalInverse<29>(&permutations_[100], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 5 ) + 0] = Ray::ScrambledRadicalInverse<31>(&permutations_[129], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 5 ) + 1] = Ray::ScrambledRadicalInverse<37>(&permutations_[160], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 6 ) + 0] = Ray::ScrambledRadicalInverse<41>(&permutations_[197], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 6 ) + 1] = Ray::ScrambledRadicalInverse<43>(&permutations_[238], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 7 ) + 0] = Ray::ScrambledRadicalInverse<47>(&permutations_[281], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 7 ) + 1] = Ray::ScrambledRadicalInverse<53>(&permutations_[328], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 8 ) + 0] = Ray::ScrambledRadicalInverse<59>(&permutations_[381], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 8 ) + 1] = Ray::ScrambledRadicalInverse<61>(&permutations_[440], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 9 ) + 0] = Ray::ScrambledRadicalInverse<67>(&permutations_[501], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 9 ) + 1] = Ray::ScrambledRadicalInverse<71>(&permutations_[568], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 10) + 0] = Ray::ScrambledRadicalInverse<73>(&permutations_[639], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 10) + 1] = Ray::ScrambledRadicalInverse<79>(&permutations_[712], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 11) + 0] = Ray::ScrambledRadicalInverse<83>(&permutations_[791], (uint64_t)(iteration + i));
        seq[2 * (i * HALTON_2D_COUNT + 11) + 1] = Ray::ScrambledRadicalInverse<89>(&permutations_[874], (uint64_t)(iteration + i));
    }
}

bool Ray::Ocl::Renderer::ExclusiveScan_CPU(const cl::Buffer &values, cl_int count, cl_int offset, cl_int stride, const cl::Buffer &out_scan_values) {
    std::vector<uint32_t> _values((stride/4) * count), _scan_values(count);
    
    if (queue_.enqueueReadBuffer(values, CL_TRUE, 0, stride * count, &_values[0]) != CL_SUCCESS) return false;

    uint32_t cur_sum = 0;
    for (size_t i = 0; i < (size_t)count; i++) {
        _scan_values[i] = cur_sum;
        cur_sum += _values[i * (stride / 4) + (offset / 4)];
    }
    
    return queue_.enqueueWriteBuffer(out_scan_values, CL_TRUE, 0, sizeof(uint32_t) * count, &_scan_values[0]) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::ExclusiveScan_GPU(const cl::Buffer &values, cl_int count, cl_int offset, cl_int stride,
                                           const cl::Buffer &partial_sums, const cl::Buffer &partial_sums2, const cl::Buffer &scan_values2,
                                           const cl::Buffer &scan_values3, const cl::Buffer &scan_values4, const cl::Buffer &out_scan_values) {
    size_t _rem = count % scan_portion_;
    size_t num = count + (_rem == 0 ? 0 : scan_portion_ - _rem);

    if (!kernel_ExclusiveScan(values, (cl_int)num, offset, stride, out_scan_values, partial_sums)) return false;

    size_t new_num = num / scan_portion_;
    size_t rem = new_num % scan_portion_;
    size_t _new_num = new_num + (rem == 0 ? 0 : scan_portion_ - rem);

    if (new_num != _new_num) {
        if (queue_.enqueueFillBuffer(partial_sums, (uint32_t)0, sizeof(uint32_t) * new_num, sizeof(uint32_t) * (_new_num - new_num)) != CL_SUCCESS) return false;
    }

    if (!kernel_InclusiveScan(partial_sums, (cl_int)_new_num, scan_values2, partial_sums2)) return false;

    if (_new_num > scan_portion_) {
        size_t new_num1 = _new_num / scan_portion_;
        size_t rem1 = new_num1 % scan_portion_;
        size_t new_num2 = new_num1 + (rem1 == 0 ? 0 : scan_portion_ - rem1);

        if (new_num1 != new_num2) {
            if (queue_.enqueueFillBuffer(partial_sums2, (uint32_t)0, sizeof(uint32_t) * new_num1, sizeof(uint32_t) * (new_num2 - new_num1)) != CL_SUCCESS) return false;
        }

        if (!kernel_InclusiveScan(partial_sums2, (cl_int)new_num2, scan_values3, partial_sums)) return false;

        if (new_num2 > scan_portion_) {
            size_t new_num3 = new_num2 / scan_portion_;
            size_t rem3 = new_num3 % scan_portion_;
            size_t new_num4 = new_num3 + (rem3 == 0 ? 0 : scan_portion_ - rem3);

            if (new_num4 != scan_portion_) {
                return false;
            }

            if (new_num3 != new_num4) {
                if (queue_.enqueueFillBuffer(partial_sums, (uint32_t)0, sizeof(uint32_t) * new_num3, sizeof(uint32_t) * (new_num4 - new_num3)) != CL_SUCCESS) return false;
            }

            if (!kernel_InclusiveScan(partial_sums, (cl_int)new_num4, scan_values4, partial_sums2)) return false;
            if (!kernel_AddPartialSums(scan_values3, (cl_int)new_num2, scan_values4)) return false;
        }

        if (!kernel_AddPartialSums(scan_values2, (cl_int)_new_num, scan_values3)) return false;
    }

    if (!kernel_AddPartialSums(out_scan_values, (cl_int)num, scan_values2)) return false;

    return true;
}

bool Ray::Ocl::Renderer::InclusiveSegScan_CPU(const cl::Buffer &flags, const cl::Buffer &values, cl_int count, const cl::Buffer &out_scan_values) {
    std::vector<uint32_t> _flags(count), _values(count), _scan_values(count);

    if (queue_.enqueueReadBuffer(flags, CL_TRUE, 0, sizeof(uint32_t) * count, &_flags[0]) != CL_SUCCESS) return false;
    if (queue_.enqueueReadBuffer(values, CL_TRUE, 0, sizeof(uint32_t) * count, &_values[0]) != CL_SUCCESS) return false;

    uint32_t cur_sum = 0;
    for (size_t i = 0; i < (size_t)count; i++) {
        if (_flags[i]) cur_sum = 0;
        cur_sum += _values[i];
        _scan_values[i] = cur_sum;
    }

    return queue_.enqueueWriteBuffer(out_scan_values, CL_TRUE, 0, sizeof(uint32_t) * count, &_scan_values[0]) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::InclusiveSegScan_GPU(const cl::Buffer &flags, const cl::Buffer &values, cl_int count, const cl::Buffer &partial_sums,
                                              const cl::Buffer &partial_sums2, const cl::Buffer &partial_sums3, const cl::Buffer &partial_sums4,
                                              const cl::Buffer &partial_flags, const cl::Buffer &partial_flags2, const cl::Buffer &partial_flags3,
                                              const cl::Buffer &partial_flags4, const cl::Buffer &scan_values2, const cl::Buffer &scan_values3,
                                              const cl::Buffer &scan_values4, const cl::Buffer &out_scan_values) {
    size_t _rem = count % seg_scan_portion_;
    size_t num = count + (_rem == 0 ? 0 : seg_scan_portion_ - _rem);

    if (!kernel_InclusiveSegScan(values, flags, (cl_int)num, out_scan_values, partial_sums, partial_flags)) return false;

    size_t new_num = num / seg_scan_portion_;
    size_t rem = new_num % seg_scan_portion_;
    size_t _new_num = new_num + (rem == 0 ? 0 : seg_scan_portion_ - rem);

    if (new_num != _new_num) {
        if (queue_.enqueueFillBuffer(partial_sums, (uint32_t)0, sizeof(uint32_t) * new_num, sizeof(uint32_t) * (_new_num - new_num)) != CL_SUCCESS) return false;
        if (queue_.enqueueFillBuffer(partial_flags, (uint32_t)0, sizeof(uint32_t) * new_num, sizeof(uint32_t) * (_new_num - new_num)) != CL_SUCCESS) return false;
    }

    if (!kernel_InclusiveSegScan(partial_sums, partial_flags, (cl_int)_new_num, scan_values2, partial_sums2, partial_flags2)) return false;

    if (_new_num > seg_scan_portion_) {
        size_t new_num1 = _new_num / seg_scan_portion_;
        size_t rem1 = new_num1 % seg_scan_portion_;
        size_t new_num2 = new_num1 + (rem1 == 0 ? 0 : seg_scan_portion_ - rem1);

        if (new_num1 != new_num2) {
            if (queue_.enqueueFillBuffer(partial_sums2, (uint32_t)0, sizeof(uint32_t) * new_num1, sizeof(uint32_t) * (new_num2 - new_num1)) != CL_SUCCESS) return false;
            if (queue_.enqueueFillBuffer(partial_flags2, (uint32_t)0, sizeof(uint32_t) * new_num1, sizeof(uint32_t) * (new_num2 - new_num1)) != CL_SUCCESS) return false;
        }

        if (!kernel_InclusiveSegScan(partial_sums2, partial_flags2, (cl_int)new_num2, scan_values3, partial_sums3, partial_flags3)) return false;

        if (new_num2 > seg_scan_portion_) {
            size_t new_num3 = new_num2 / seg_scan_portion_;
            size_t rem3 = new_num3 % seg_scan_portion_;
            size_t new_num4 = new_num3 + (rem3 == 0 ? 0 : seg_scan_portion_ - rem3);

            if (new_num4 != seg_scan_portion_) {
                return false;
            }

            if (new_num3 != new_num4) {
                if (queue_.enqueueFillBuffer(partial_sums, (uint32_t)0, sizeof(uint32_t) * new_num3, sizeof(uint32_t) * (new_num4 - new_num3)) != CL_SUCCESS) return false;
                if (queue_.enqueueFillBuffer(partial_flags, (uint32_t)0, sizeof(uint32_t) * new_num3, sizeof(uint32_t) * (new_num4 - new_num3)) != CL_SUCCESS) return false;
            }

            if (!kernel_InclusiveSegScan(partial_sums, partial_flags, (cl_int)new_num4, scan_values4, partial_sums4, partial_flags4)) return false;
            if (!kernel_AddSegPartialSums(partial_flags, scan_values3, scan_values4, (cl_int)(new_num2/seg_scan_portion_), (cl_int)seg_scan_portion_)) return false;
        }

        if (!kernel_AddSegPartialSums(partial_flags, scan_values2, scan_values3, (cl_int)(new_num/seg_scan_portion_), (cl_int)seg_scan_portion_)) return false;
    }

    if (!kernel_AddSegPartialSums(flags, out_scan_values, scan_values2, (cl_int)(num/seg_scan_portion_), (cl_int)seg_scan_portion_)) return false;

    return true;
}

bool Ray::Ocl::Renderer::ReorderRays_CPU(const cl::Buffer &scan_values, const cl::Buffer &rays, cl_int count) {
    std::vector<uint32_t> _scan_values(count);
    std::vector<ray_packet_t> _rays(count);

    if (queue_.enqueueReadBuffer(scan_values, CL_TRUE, 0, sizeof(uint32_t) * count, &_scan_values[0]) != CL_SUCCESS) return false;
    if (queue_.enqueueReadBuffer(rays, CL_TRUE, 0, sizeof(ray_packet_t) * count, &_rays[0]) != CL_SUCCESS) return false;

    uint32_t j, k;
    for (uint32_t i = 0; i < (uint32_t)count; i++) {
        while (i != (j = _scan_values[i])) {
            k = _scan_values[j];
            std::swap(_rays[j], _rays[k]);
            std::swap(_scan_values[i], _scan_values[j]);
        }
    }

    return queue_.enqueueWriteBuffer(rays, CL_TRUE, 0, sizeof(ray_packet_t) * count, &_rays[0]) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::PerformRadixSort_CPU(const cl::Buffer &chunks, cl_int count) {
    std::vector<ray_chunk_t> _chunks(count), _chunks_temp(count);

    if (queue_.enqueueReadBuffer(chunks, CL_TRUE, 0, sizeof(ray_chunk_t) * count, &_chunks[0]) != CL_SUCCESS) return false;

    {
        ray_chunk_t *begin = &_chunks[0];
        ray_chunk_t *end = &_chunks[0] + count;
        ray_chunk_t *begin1 = &_chunks_temp[0];
        ray_chunk_t *end1 = begin1 + (end - begin);

        for (unsigned shift = 0; shift <= 24; shift += 8) {
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

    return queue_.enqueueWriteBuffer(chunks, CL_TRUE, 0, sizeof(ray_chunk_t) * count, &_chunks[0]) == CL_SUCCESS;
}

bool Ray::Ocl::Renderer::PerformRadixSort_GPU(const cl::Buffer &chunks, const cl::Buffer &chunks2, cl_int count, const cl::Buffer &counters,
                                              const cl::Buffer &partial_sums, const cl::Buffer &partial_sums2, const cl::Buffer &scan_values,
                                              const cl::Buffer &scan_values2, const cl::Buffer &scan_values3, const cl::Buffer &scan_values4) {
    size_t group_size = std::min(max_work_group_size_, (size_t)64);

    size_t _rem = count % group_size;
    size_t num = count + (_rem == 0 ? 0 : group_size - _rem);

    if (_rem) {
        if (queue_.enqueueFillBuffer(chunks, (uint32_t)0xFFFFFFFF, sizeof(ray_chunk_t) * count, sizeof(ray_chunk_t) * (group_size - _rem)) != CL_SUCCESS) return false;
        if (queue_.enqueueFillBuffer(chunks2, (uint32_t)0xFFFFFFFF, sizeof(ray_chunk_t) * count, sizeof(ray_chunk_t) * (group_size - _rem)) != CL_SUCCESS) return false;
    }

    size_t group_count = num / group_size;

    const auto *_chunks1 = &chunks, *_chunks2 = &chunks2;

    for (int shift = 0; shift <= 28; shift += 4) {
        if (!kernel_InitCountTable(*_chunks1, (cl_int)num, (cl_int)group_size, shift, counters) ||
            !ExclusiveScan_GPU(counters, (cl_int)group_count * 0x10, 0, 4, partial_sums, partial_sums2, scan_values2, scan_values3, scan_values4, scan_values) ||
            !kernel_WriteSortedChunks(*_chunks1, scan_values, counters, (cl_int)group_count, shift, (cl_int)group_size, *_chunks2)) return false;

        std::swap(_chunks1, _chunks2);
    }

    return true;
}

bool Ray::Ocl::Renderer::SortRays(const cl::Buffer &in_rays, cl_int rays_count, cl_float3 root_min, cl_float3 cell_size, const cl::Buffer &ray_hashes,
                                  const cl::Buffer &head_flags, const cl::Buffer &partial_sums, const cl::Buffer &partial_sums2, const cl::Buffer &partial_sums3, const cl::Buffer &partial_sums4,
                                  const cl::Buffer &partial_flags, const cl::Buffer &partial_flags2, const cl::Buffer &partial_flags3, const cl::Buffer &partial_flags4,
                                  const cl::Buffer &scan_values, const cl::Buffer &scan_values2, const cl::Buffer &scan_values3, const cl::Buffer &scan_values4,
                                  const cl::Buffer &chunks, const cl::Buffer &chunks2, const cl::Buffer &counters, const cl::Buffer &skeleton, const cl::Buffer &out_rays) {
    if (!kernel_ComputeRayHashes(in_rays, rays_count, root_min, cell_size, ray_hashes)) return false;

    if (!kernel_SetHeadFlags(ray_hashes, rays_count, head_flags)) return false;

    if (!ExclusiveScan_GPU(head_flags, rays_count, 0, 4, partial_sums, partial_sums2, scan_values2, scan_values3, scan_values4, scan_values)) return false;

    uint32_t chunks_count = 0;

    {   // get chunks count
        uint32_t last_scan_value = 0;
        if (queue_.enqueueReadBuffer(scan_values, CL_FALSE, sizeof(uint32_t) * (rays_count - 1), sizeof(uint32_t), &last_scan_value) != CL_SUCCESS) return false;

        uint32_t last_flag = 0;
        if (queue_.enqueueReadBuffer(head_flags, CL_TRUE, sizeof(uint32_t) * (rays_count - 1), sizeof(uint32_t), &last_flag) != CL_SUCCESS) return false;

        chunks_count = last_scan_value + last_flag;
    }

    if (!kernel_InitChunkHashAndBase(chunks, rays_count, ray_hashes, head_flags, scan_values)) return false;
    if (!kernel_InitChunkSize(chunks, (cl_int)chunks_count, (cl_int)rays_count)) return false;

    if (!PerformRadixSort_GPU(chunks, chunks2, (cl_int)chunks_count, counters, partial_sums, partial_sums2,
                              scan_values, scan_values2, scan_values3, scan_values4)) return false;

    if (!ExclusiveScan_GPU(chunks, (cl_int)chunks_count, offsetof(ray_chunk_t, size), sizeof(ray_chunk_t), partial_sums, partial_sums2, scan_values2, scan_values3,
                           scan_values4, scan_values)) return false;

    if (queue_.enqueueFillBuffer(skeleton, (uint32_t)1, 0, sizeof(uint32_t) * rays_count) != CL_SUCCESS) return false;
    if (queue_.enqueueFillBuffer(head_flags, (uint32_t)0, 0, sizeof(uint32_t) * rays_count) != CL_SUCCESS) return false;

    if (!kernel_InitSkeletonAndHeadFlags(scan_values, chunks, (cl_int)chunks_count, skeleton, head_flags)) return false;

    if (!InclusiveSegScan_GPU(head_flags, skeleton, (cl_int)rays_count, partial_sums, partial_sums2, partial_sums3, partial_sums4,
                              partial_flags, partial_flags2, partial_flags3, partial_flags4, scan_values2,
                              scan_values3, scan_values4, scan_values)) return false;

    if (!kernel_ReorderRays(in_rays, scan_values, (cl_int)rays_count, out_rays)) return false;

    return true;
}

std::vector<Ray::Ocl::Platform> Ray::Ocl::Renderer::QueryPlatforms() {
    std::vector<Ray::Ocl::Platform> out_platforms;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (size_t i = 0; i < platforms.size(); i++) {
        out_platforms.emplace_back();

        auto v = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
        auto n = platforms[i].getInfo<CL_PLATFORM_NAME>();

        out_platforms.back().vendor = v;
        out_platforms.back().name = n;

        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        for (size_t j = 0; j < devices.size(); j++) {
            auto n = devices[j].getInfo<CL_DEVICE_NAME>();
            out_platforms.back().devices.push_back({ n });
        }
    }

    return out_platforms;
}