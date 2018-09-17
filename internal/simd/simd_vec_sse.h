//#pragma once

#include <type_traits>

#include <immintrin.h>
#include <xmmintrin.h>

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target ("sse2")
#endif

namespace Ray {
namespace NS {

template <int S>
class simd_vec<typename std::enable_if<S == 4, float>::type, S> {
    union {
        __m128 vec_;
        float comp_[4];
    };

    friend class simd_vec<int, S>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(float f) {
        vec_ = _mm_set1_ps(f);
    }
    force_inline simd_vec(float f1, float f2, float f3, float f4) {
        vec_ = _mm_setr_ps(f1, f2, f3, f4);
    }
    force_inline simd_vec(const float *f) {
        vec_ = _mm_loadu_ps(f);
    }
    force_inline simd_vec(const float *f, simd_mem_aligned_tag) {
        vec_ = _mm_load_ps(f);
    }

    force_inline float &operator[](int i) { return comp_[i]; }
    force_inline float operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<float, S> &operator+=(const simd_vec<float, S> &rhs) {
        vec_ = _mm_add_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, S> &operator+=(float rhs) {
        vec_ = _mm_add_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, S> &operator-=(const simd_vec<float, S> &rhs) {
        vec_ = _mm_sub_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, S> &operator-=(float rhs) {
        vec_ = _mm_sub_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, S> &operator*=(const simd_vec<float, S> &rhs) {
        vec_ = _mm_mul_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, S> &operator*=(float rhs) {
        vec_ = _mm_mul_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, S> &operator/=(const simd_vec<float, S> &rhs) {
        vec_ = _mm_div_ps(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<float, S> &operator/=(float rhs) {
        vec_ = _mm_div_ps(vec_, _mm_set1_ps(rhs));
        return *this;
    }

    force_inline simd_vec<float, S> operator-() const {
        simd_vec<float, S> temp;
        __m128 m = _mm_set1_ps(-0.0f);
        temp.vec_ = _mm_xor_ps(vec_, m);
        return temp;
    }

    force_inline simd_vec<float, S> operator<(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_cmplt_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, S> operator<=(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_cmple_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, S> operator>(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_cmpgt_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, S> operator>=(const simd_vec<float, S> &rhs) const {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_cmpge_ps(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<float, S> operator<(float rhs) const {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_cmplt_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, S> operator<=(float rhs) const {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_cmple_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, S> operator>(float rhs) const {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_cmpgt_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline simd_vec<float, S> operator>=(float rhs) const {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_cmpge_ps(vec_, _mm_set1_ps(rhs));
        return ret;
    }

    force_inline operator simd_vec<int, S>() const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_cvtps_epi32(vec_);
        return ret;
    }

    force_inline simd_vec<float, S> sqrt() const {
        simd_vec<float, S> temp;
        temp.vec_ = _mm_sqrt_ps(vec_);
        return temp;
    }

    force_inline void copy_to(float *f) const {
        _mm_storeu_ps(f, vec_);
    }

    force_inline void copy_to(float *f, simd_mem_aligned_tag) const {
        _mm_store_ps(f, vec_);
    }

    force_inline void blend_to(const simd_vec<float, S> &mask, const simd_vec<float, S> &v1) {
#if 0 // requires sse4.1
        vec_ = _mm_blendv_ps(vec_, v1.vec_, mask.vec_);
#else
        __m128 temp1 = _mm_and_ps(mask.vec_, v1.vec_);
        __m128 temp2 = _mm_andnot_ps(mask.vec_, vec_);
        vec_ = _mm_or_ps(temp1, temp2);
#endif
    }

    force_inline static simd_vec<float, S> min(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm_min_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, S> max(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm_max_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, S> and_not(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm_andnot_ps(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<float, S> floor(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
#if 1
        __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_));
        __m128 r = _mm_sub_ps(t, _mm_and_ps(_mm_cmplt_ps(v1.vec_, t), _mm_set1_ps(1.0f)));
        temp.vec_ = r;
#else
        temp.vec_ = _mm_floor_ps(v1.vec_);
#endif
        return temp;
    }

    force_inline static simd_vec<float, S> ceil(const simd_vec<float, S> &v1) {
        simd_vec<float, S> temp;
        __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(v1.vec_));
        __m128 r = _mm_add_ps(t, _mm_and_ps(_mm_cmpgt_ps(v1.vec_, t), _mm_set1_ps(1.0f)));
        temp.vec_ = r;
        return temp;
    }

    friend force_inline simd_vec<float, S> operator&(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm_and_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator|(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm_or_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator^(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> temp;
        temp.vec_ = _mm_xor_ps(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<float, S> operator+(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_add_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator-(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_sub_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator*(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_mul_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator/(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_div_ps(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator+(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_add_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, S> operator-(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_sub_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, S> operator*(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_mul_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, S> operator/(const simd_vec<float, S> &v1, float v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_div_ps(v1.vec_, _mm_set1_ps(v2));
        return ret;
    }

    friend force_inline simd_vec<float, S> operator+(float v1, const simd_vec<float, S> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<float, S> operator-(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_sub_ps(_mm_set1_ps(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator*(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_mul_ps(_mm_set1_ps(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<float, S> operator/(float v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_div_ps(_mm_set1_ps(v1), v2.vec_);
        return ret;
    }

    friend force_inline float dot(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        __m128 r1, r2;
        r1 = _mm_mul_ps(v1.vec_, v2.vec_);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
        r1 = _mm_add_ps(r1, r2);
        r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_add_ps(r1, r2);
        return _mm_cvtss_f32(r1);
    }

    friend force_inline simd_vec<float, S> clamp(const simd_vec<float, S> &v1, float min, float max) {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_max_ps(_mm_set1_ps(min), _mm_min_ps(v1.vec_, _mm_set1_ps(max)));
        return ret;
    }

    friend force_inline simd_vec<float, S> pow(const simd_vec<float, S> &v1, const simd_vec<float, S> &v2) {
        simd_vec<float, S> ret;
        ITERATE_4({ ret.comp_[i] = std::pow(v1.comp_[i], v2.comp_[i]); })
        return ret;
    }

    friend force_inline simd_vec<float, S> normalize(const simd_vec<float, S> &v1) {
        return v1 / v1.length();
    }

    friend force_inline const float *value_ptr(const simd_vec<float, S> &v1) {
        return &v1.comp_[0];
    }

    static int size() { return S; }
    static bool is_native() { return true; }
};

template <int S>
class simd_vec<typename std::enable_if<S == 4, int>::type, S> {
    union {
        __m128i vec_;
        int comp_[4];
    };

    friend class simd_vec<float, S>;
public:
    force_inline simd_vec() = default;
    force_inline simd_vec(int f) {
        vec_ = _mm_set1_epi32(f);
    }
    force_inline simd_vec(int i1, int i2, int i3, int i4) {
        vec_ = _mm_setr_epi32(i1, i2, i3, i4);
    }
    force_inline simd_vec(const int *f) {
        vec_ = _mm_loadu_si128((const __m128i *)f);
    }
    force_inline simd_vec(const int *f, simd_mem_aligned_tag) {
        vec_ = _mm_load_si128((const __m128i *)f);
    }

    force_inline int &operator[](int i) { return comp_[i]; }
    force_inline int operator[](int i) const { return comp_[i]; }

    force_inline simd_vec<int, S> &operator+=(const simd_vec<int, S> &rhs) {
        vec_ = _mm_add_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, S> &operator+=(int rhs) {
        vec_ = _mm_add_epi32(vec_, _mm_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, S> &operator-=(const simd_vec<int, S> &rhs) {
        vec_ = _mm_sub_epi32(vec_, rhs.vec_);
        return *this;
    }

    force_inline simd_vec<int, S> &operator-=(int rhs) {
        vec_ = _mm_sub_epi32(vec_, _mm_set1_epi32(rhs));
        return *this;
    }

    force_inline simd_vec<int, S> &operator*=(const simd_vec<int, S> &rhs) {
        ITERATE_4({ comp_[i] = comp_[i] * rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator*=(int rhs) {
        ITERATE_4({ comp_[i] = comp_[i] * rhs; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator/=(const simd_vec<int, S> &rhs) {
        ITERATE_4({ comp_[i] = comp_[i] / rhs.comp_[i]; })
        return *this;
    }

    force_inline simd_vec<int, S> &operator/=(int rhs) {
        ITERATE_4({ comp_[i] = comp_[i] / rhs; })
        return *this;
    }

    force_inline simd_vec<int, S> operator==(int rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_cmpeq_epi32(vec_, _mm_set1_epi32(rhs));
        return ret;
    }

    force_inline simd_vec<int, S> operator==(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_cmpeq_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, S> operator!=(int rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpeq_epi32(vec_, _mm_set1_epi32(rhs)), _mm_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, S> operator!=(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpeq_epi32(vec_, rhs.vec_), _mm_set1_epi32(~0));
        return ret;
    }

    force_inline simd_vec<int, S> operator<(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_cmplt_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, S> operator<=(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpgt_epi32(vec_, rhs.vec_), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, S> operator>(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_cmpgt_epi32(vec_, rhs.vec_);
        return ret;
    }

    force_inline simd_vec<int, S> operator>=(const simd_vec<int, S> &rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmplt_epi32(vec_, rhs.vec_), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, S> operator<(int rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_cmplt_epi32(vec_, _mm_set1_epi32(rhs));
        return ret;
    }

    force_inline simd_vec<int, S> operator<=(int rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmpgt_epi32(vec_, _mm_set1_epi32(rhs)), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline simd_vec<int, S> operator>(int rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_cmpgt_epi32(vec_, _mm_set1_epi32(rhs));
        return ret;
    }

    force_inline simd_vec<int, S> operator>=(int rhs) const {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_andnot_si128(_mm_cmplt_epi32(vec_, _mm_set1_epi32(rhs)), _mm_set_epi32(~0, ~0, ~0, ~0));
        return ret;
    }

    force_inline operator simd_vec<float, S>() const {
        simd_vec<float, S> ret;
        ret.vec_ = _mm_cvtepi32_ps(vec_);
        return ret;
    }

    force_inline void copy_to(int *f) const {
        _mm_storeu_si128((__m128i *)f, vec_);
    }

    force_inline void copy_to(int *f, simd_mem_aligned_tag) const {
        _mm_store_si128((__m128i *)f, vec_);
    }

    force_inline void blend_to(const simd_vec<int, S> &mask, const simd_vec<int, S> &v1) {
#if 0 // requires sse4.1
        vec_ = _mm_blendv_epi8(vec_, v1.vec_, mask.vec_);
#else
        __m128i temp1 = _mm_and_si128(mask.vec_, v1.vec_);
        __m128i temp2 = _mm_andnot_si128(mask.vec_, vec_);
        vec_ = _mm_or_si128(temp1, temp2);
#endif
    }

    force_inline bool all_zeros() const {
#if 0 // requires sse4.1
        if (!_mm_test_all_zeros(vec_, vec_)) return false;
#else
        if (_mm_movemask_epi8(_mm_cmpeq_epi32(vec_, _mm_setzero_si128())) != 0xFFFF) return false;
#endif
        return true;
    }

    force_inline bool all_zeros(const simd_vec<int, S> &mask) const { 
#if 0 // requires sse4.1
        if (!_mm_test_all_zeros(vec_, mask.vec_)) return false;
#else
        if (_mm_movemask_epi8(_mm_cmpeq_epi32(_mm_and_si128(vec_, mask.vec_), _mm_setzero_si128())) != 0xFFFF) return false;
#endif
        return true;
    }

    force_inline bool not_all_zeros() const {
        return !all_zeros();
    }

    force_inline static simd_vec<int, S> min(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        temp.vec_ = _mm_min_si128(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, S> max(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        temp.vec_ = _mm_max_epi32(v1.vec_, v2.vec_);
        return temp;
    }

    force_inline static simd_vec<int, S> and_not(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        temp.vec_ = _mm_andnot_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, S> operator&(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        temp.vec_ = _mm_and_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, S> operator|(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        temp.vec_ = _mm_or_si128(v1.vec_, v2.vec_);
        return temp;
    }

    friend force_inline simd_vec<int, S> operator^(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> temp;
        temp.vec_ = _mm_xor_si128(v1.vec_, v2.vec_);;
        return temp;
    }

    friend force_inline simd_vec<int, S> operator+(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_add_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, S> operator-(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_sub_epi32(v1.vec_, v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, S> operator*(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator/(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator+(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_add_epi32(v1.vec_, _mm_set1_epi32(v2));
        return ret;
    }

    friend force_inline simd_vec<int, S> operator-(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_sub_epi32(v1.vec_, _mm_set1_epi32(v2));
        return ret;
    }

    friend force_inline simd_vec<int, S> operator*(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] * v2; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator/(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ITERATE_4({ ret.vec_ = v1.comp_[i] / v2; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator+(int v1, const simd_vec<int, S> &v2) {
        return operator+(v2, v1);
    }

    friend force_inline simd_vec<int, S> operator-(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_sub_epi32(_mm_set1_epi32(v1), v2.vec_);
        return ret;
    }

    friend force_inline simd_vec<int, S> operator*(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE_4({ ret.comp_[i] = v1 * v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator/(int v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
        ITERATE_4({ ret.comp_[i] = v1 / v2.comp_[i]; })
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>>(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
#if 0
        ret.vec_ = _mm_srlv_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] >> v2.comp_[i]; })
#endif
        return ret;
    }

    friend force_inline simd_vec<int, S> operator>>(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_srli_epi32(v1.vec_, v2);
        return ret;
    }

    friend force_inline simd_vec<int, S> operator<<(const simd_vec<int, S> &v1, const simd_vec<int, S> &v2) {
        simd_vec<int, S> ret;
#if 0
        ret.vec_ = _mm_sllv_epi32(v1.vec_, v2.vec_);
#else
        ITERATE_4({ ret.comp_[i] = v1.comp_[i] << v2.comp_[i]; })
#endif
            return ret;
    }

    friend force_inline simd_vec<int, S> operator<<(const simd_vec<int, S> &v1, int v2) {
        simd_vec<int, S> ret;
        ret.vec_ = _mm_slli_epi32(v1.vec_, v2);
        return ret;
    }

    static int size() { return S; }
    static bool is_native() { return true; }
};

#if defined(USE_SSE)
using native_simd_fvec = simd_fvec<4>;
using native_simd_ivec = simd_ivec<4>;
#endif

}
}

#ifdef __GNUC__
#pragma GCC pop_options
#endif