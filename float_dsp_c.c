#define BUILDING_FDSP 1
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fdsp.h"

typedef union _f32 {
    float f;
    uint32_t i;
} _f32;

FDSP_EXPORT void vector_fmul_c(float *__restrict dst, const float *__restrict src0, const float *__restrict src1, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++)
        dst[i] = src0[i] * src1[i];
}

FDSP_EXPORT void vector_fmac_scalar_c(float *__restrict dst, const float *__restrict src, float mul, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++)
        dst[i] += src[i] * mul;
}

FDSP_EXPORT void vector_fmul_scalar_c(float *__restrict dst, const float *__restrict src, float mul, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++)
        dst[i] = src[i] * mul;
}

FDSP_EXPORT void vector_fmul_window_c(float *__restrict dst, const float *__restrict src0, const float *__restrict src1, const float *__restrict win, unsigned int len)
{
    unsigned int i;
    for(i=0; i<len; i++) {
        float s0 = src0[i];
        float s1 = src1[len-1-i];
        float wi = win[i];
        float wj = win[2*len-1-i];
        dst[i        ] = s0*wj - s1*wi;
        dst[2*len-i-1] = s0*wi + s1*wj;
    }
}

FDSP_EXPORT void vector_fmul_copy_c(float *__restrict dst, const float *__restrict src, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++)
        dst[i] = src[i];
}

FDSP_EXPORT void vector_fmul_add_c(float *__restrict dst, const float *__restrict src0, const float *__restrict src1, const float *__restrict src2, unsigned int len){
    unsigned int i;
    for (i = 0; i < len; i++)
        dst[i] = src0[i] * src1[i] + src2[i];
}

FDSP_EXPORT void vector_fmul_reverse_c(float *__restrict dst, const float *__restrict src0, const float *__restrict src1, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++)
        dst[i] = src0[i] * src1[len - i - 1];
}

FDSP_EXPORT void vector_fmul_cf_c(FFTComplex *__restrict dst, FFTComplex *__restrict src0, float *__restrict src1, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++) {
        dst[i].re = src0[i].re * src1[i];
        dst[i].im = src0[i].im * src1[i];
    }
}

FDSP_EXPORT void butterflies_float_c(float *__restrict __restrict v1, float *__restrict __restrict v2, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++) {
        float s = v1[i] + v2[i];
        float t = v1[i] - v2[i];
        v1[i] = s;
        v2[i] = t;
    }
}

FDSP_EXPORT float scalarproduct_float_c(float *__restrict v1, float *__restrict v2, unsigned int len)
{
    float p = 0.0;
    unsigned int i;
    for (i = 0; i < len; i++) {
        p += v1[i] * v2[i];
    }
    return p;
}

FDSP_EXPORT void calc_power_spectrum_c(float *__restrict psd, const FFTComplex *__restrict vin, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len/2; i++) {
        psd[i] = ((vin[i].re * vin[i].re) + (vin[i].im * vin[i].im));
    }
}

FDSP_EXPORT void vector_clipf_c(float *__restrict dst, const float *__restrict src,
                                float min, float max, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++) {
        float f = src[i];
        if (f < min) f = min;
        if (f > max) f = max;
        dst[i] = f;
    }
}

FDSP_EXPORT void sbr_sum64x5_c(float *__restrict z)
{
    unsigned int k;
    for (k = 0; k < 64; k++) {
        float f = z[k] + z[k + 64] + z[k + 128] + z[k + 192] + z[k + 256];
        z[k] = f;
    }
}

FDSP_EXPORT void sbrenc_sum128x5_c(float *__restrict z)
{
    unsigned int k;
    for (k = 0; k < 128; k++) {
        float f = z[k] + z[k + 128] + z[k + 256] + z[k + 384] + z[k + 512];
        z[k] = f;
    }
}

FDSP_EXPORT void sbr_qmf_pre_shuffle_c(float *__restrict z)
{
    unsigned int k;
    float tmp1 = z[0], tmp2 = z[1];
    //z[64] = z[0];
    //z[65] = z[1];
    for (k = 0; k < 32; k++) {
        z[64+2*k  ] = -z[64 - k];
        z[64+2*k+1] =  z[ k + 1];
    }
    z[64] = tmp1;
}

FDSP_EXPORT void sbr_ldqmf_pre_shuffle_c(float *z)
{
    unsigned int k;
    for (k = 0; k < 16; k++) {
        z[64+k   ] =  z[47-k] + z[48+k];
        z[64+k+16] =  z[31-k] - z[k];
        z[64+k+32] =  z[47-k] - z[48+k];
        z[64+k+48] = -z[31-k] - z[k];
    }
}

FDSP_EXPORT void sbr_qmf_post_shuffle_c(FFTComplex W[32], float *__restrict z)
{
    unsigned int k;
    for (k = 0; k < 32; k++) {
        W[k].re = z[k];
        W[k].im = z[63-k];
    }
}

FDSP_EXPORT void sbr_qmf_deint_bfly_c(float *__restrict v, const float *__restrict src0, const float *__restrict src1)
{
    unsigned int i;
    for (i = 0; i < 64; i++) {
        v[      i] = src1[63 - i] + src0[i];
        v[127 - i] = src1[63 - i] - src0[i];
    }
}

FDSP_EXPORT void sbrenc_qmf_deint_bfly_c(float *__restrict v, const float *__restrict src0, const float *__restrict src1)
{
    unsigned int i;
    for (i = 0; i < 64; i++) {
        v[      i] = -src1[63 - i] + src0[i];
        v[127 - i] =  src1[63 - i] + src0[i];
    }
}

FDSP_EXPORT void sbr_qmf_deint_neg_c(float *__restrict v, const float *__restrict src)
{
    unsigned int i;
    for (i = 0; i < 32; i++) {
        v[     i] =  src[63 - 2*i    ];
        v[63 - i] = -src[63 - 2*i - 1];
    }
}

FDSP_EXPORT void sbr_hf_g_filt_c(FFTComplex *__restrict Y, FFTComplex (*__restrict X_high)[40],
                     const float *__restrict g_filt, size_t m_max, size_t ixh)
{
    size_t m;
    for (m = 0; m < m_max; m++) {
        Y[m].re = X_high[m][ixh].re * g_filt[m];
        Y[m].im = X_high[m][ixh].im * g_filt[m];
    }
}

FDSP_EXPORT void sbr_hf_gen_c(FFTComplex *__restrict X_high, FFTComplex *__restrict X_low,
                              float alpha[4], unsigned int start, unsigned int end)
{
    int i;

    for (i = start; i < end; i++) {
        X_high[i].re = X_low[i].re +
            X_low[i - 2].re * alpha[0] + X_low[i - 2].im * alpha[1] +
            X_low[i - 1].re * alpha[2] + X_low[i - 1].im * alpha[3];
        X_high[i].im = X_low[i].im +
            X_low[i - 2].im * alpha[0] - X_low[i - 2].re * alpha[1] +
            X_low[i - 1].im * alpha[2] - X_low[i - 1].re * alpha[3];
    }
}

FDSP_EXPORT void sbr_autocorrelate_c(const FFTComplex x[40], float phi[5], unsigned int ac_len)
{
    float real_sum2 = 0.0f, imag_sum2 = 0.0f;
    float real_sum1 = 0.0f, imag_sum1 = 0.0f, real_sum0 = 0.0f;
    unsigned int i;
    for (i = 1; i < ac_len; i++) {
        real_sum0 +=  x[i].re * x[i  ].re + x[i].im * x[i  ].im;
        real_sum1 +=  x[i].re * x[i+1].re + x[i].im * x[i+1].im;
        imag_sum1 += -x[i].re * x[i+1].im + x[i].im * x[i+1].re;
        real_sum2 +=  x[i].re * x[i+2].re + x[i].im * x[i+2].im;
        imag_sum2 += -x[i].re * x[i+2].im + x[i].im * x[i+2].re;
    }
    phi[2] = real_sum2;
    phi[3] = imag_sum2;
    phi[4] = real_sum0;
    phi[0] = real_sum1;
    phi[1] = imag_sum1;
}

FDSP_EXPORT void sbr_qmf_synthesis_window_c(float *__restrict out, float *__restrict v, float *__restrict sbr_qmf_window)
{
    unsigned int n;
    for (n = 0; n < 64; n++) {
        out[n] = v[n]*sbr_qmf_window[n] + v[n+19*64]*sbr_qmf_window[n+9*64];
    }
    for (n = 0; n < 128; n++) {
        float t  = v[n+ 3*64]*sbr_qmf_window[n+  64];
              t += v[n+ 7*64]*sbr_qmf_window[n+3*64];
              t += v[n+11*64]*sbr_qmf_window[n+5*64];
              t += v[n+15*64]*sbr_qmf_window[n+7*64];
        out[n&63] += t;
    }
}

FDSP_EXPORT void sbr_qmf_synthesis_window_ds_c(float *__restrict out, float *__restrict v, float *__restrict sbr_qmf_window)
{
    unsigned int n;
    for (n = 0; n < 32; n++) {
        out[n] = v[n]*sbr_qmf_window[n] + v[n+19*32]*sbr_qmf_window[n+9*32];
    }
    for (n = 0; n < 64; n++) {
        float t  = v[n+ 3*32]*sbr_qmf_window[n+  32];
              t += v[n+ 7*32]*sbr_qmf_window[n+3*32];
              t += v[n+11*32]*sbr_qmf_window[n+5*32];
              t += v[n+15*32]*sbr_qmf_window[n+7*32];
        out[n&31] += t;
    }
}

FDSP_EXPORT void aacenc_calc_expspec_c(float *__restrict expspec, float *__restrict mdct_spectrum, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++) {
        _f32 tmp;
        uint32_t ix;
        float x, y, t;

        x = mdct_spectrum[i];
        tmp.f = x;
        tmp.i &= 0x7fffffff;
        x = tmp.f;
        ix = tmp.i;

        ix  = 0x4f58cae5 - (ix >> 2);
        tmp.i = ix;
        y = tmp.f;
        t = y * y;
        y  = y * 0.25f * (5.0f - (x * t * t));   // 1st iteration
        t = y * y;
        y  = y * 0.25f * (5.0f - (x * t * t));   // 2nd iteration
        t = y * y;
        y  = y * 0.25f * (5.0f - (x * t * t));   // 3rd iteration
        y = y * x;
        expspec[i] = y;
    }
}

FDSP_EXPORT void aac_update_ltp_c(float *dst, const float *buf, const float *win, unsigned int len)
{
    unsigned int i, len2 = 2*len-1;
    for (i = 0; i < len; i++) {
        dst[     i] = buf[i] * win[len2 - i];
        dst[len2-i] = buf[i] * win[i];
    }
}

FDSP_EXPORT void vorbis_inverse_coupling_c(float *__restrict mag, float *__restrict ang, unsigned int blocksize)
{
    unsigned int i;
    for (i = 0;  i < blocksize; i++) {
        if (mag[i] >= 0.0f) {
            if (ang[i] > 0.0f) {
                ang[i] = mag[i] - ang[i];
            } else {
                float temp = ang[i];
                ang[i]     = mag[i];
                mag[i]    += temp;
            }
        } else {
            if (ang[i] > 0.0f) {
                ang[i] = mag[i] + ang[i];
            } else {
                float temp = ang[i];
                ang[i]     = mag[i];
                mag[i]    -= temp;
            }
        }
    }
}

FDSP_EXPORT void flac_decorrelate_mono_c(int16_t *out, int32_t **in, unsigned long len, int shift)
{
    unsigned long i;

    for (i = 0; i < len; i++) {
        out[i] = (int16_t)in[0][i];
    }
}

FDSP_EXPORT void flac_decorrelate_indep2_c(int16_t *out, int32_t **in, unsigned long len, int shift)
{
    unsigned long i;

    for (i = 0; i < len; i++) {
        int a= in[0][i];
        int b= in[1][i];
        out[2*i+0] = a;
        out[2*i+1] = b;
    }
}

FDSP_EXPORT void flac_decorrelate_ls_c(int16_t *out, int32_t **in, unsigned long len, int shift)
{
    unsigned long i;

    for (i = 0; i < len; i++) {
        int a= in[0][i];
        int b= in[1][i];
        out[2*i+0] = a;
        out[2*i+1] = a-b;
    }
}

FDSP_EXPORT void flac_decorrelate_rs_c(int16_t *out, int32_t **in, unsigned long len, int shift)
{
    unsigned long i;

    for (i = 0; i < len; i++) {
        int a= in[0][i];
        int b= in[1][i];
        out[2*i+0] = a+b;
        out[2*i+1] = b;
    }
}

FDSP_EXPORT void flac_decorrelate_ms_c(int16_t *out, int32_t **in, unsigned long len, int shift)
{
    unsigned long i;

    for (i = 0; i < len; i++) {
        int a= in[0][i];
        int b= in[1][i];
        a -= b>>1;
        out[2*i+0] = a+b;
        out[2*i+1] = a;
    }
}

FDSP_EXPORT void tta_decorrelate_ms_c(int16_t *out, int32_t **in, unsigned long len, int shift)
{
    unsigned long i;

    for (i = 0; i < len; i++) {
        int32_t a = in[0][i], a2 = a;
        int32_t b = in[1][i];
        a2 += (a2 < 0);
        b += a2>>1;
        out[2*i+0] = (int16_t)(b-a);
        out[2*i+1] = (int16_t)b;
    }
}

FDSP_EXPORT void conv_fltp_to_flt_2ch_c(float *__restrict dst, float *__restrict src[2], unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++) {
        dst[2*i] = src[0][i];
        dst[2*i+1] = src[1][i];
    }
}

FDSP_EXPORT void conv_flt_to_fltp_2ch_c(float *__restrict dst[2], float *__restrict src, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++) {
        dst[0][i] = src[2*i];
        dst[1][i] = src[2*i+1];
    }
}

FDSP_EXPORT void conv_s16p_to_s16_2ch_c(short *__restrict dst, short *__restrict src[2], unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++) {
        dst[2*i] = src[0][i];
        dst[2*i+1] = src[1][i];
    }
}

FDSP_EXPORT void conv_s16_to_s16p_2ch_c(short *__restrict dst[2], short *__restrict src, unsigned int len)
{
    unsigned int i;
    for (i = 0; i < len; i++) {
        dst[0][i] = src[2*i];
        dst[1][i] = src[2*i+1];
    }
}

