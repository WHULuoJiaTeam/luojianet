#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define LOCAL_CACHE_THREAD 16
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void GlobalHWMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);  // C4
  if (X >= size.z) {
    return;
  }
  float4 result = (float4)0.f;
  for (int h = 0; h < size.x; h++) {
    for (int w = 0; w < size.y; w++) {
      result += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h)));
    }
  }
  result /= size.x * size.y;
  WRITE_IMAGE(dst_data, (int2)(X, 0), TO_FLT4(result));
}

__kernel void LocalHWMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);  // C4
  int localy = get_local_id(1);
  int localz = get_local_id(2);
  if (X >= size.z) return;
  __local float4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];
  temp[localy][localz] = (float4)0.f;
  for (int h = localy; h < size.x; h += LOCAL_CACHE_THREAD) {
    for (int w = localz; w < size.y; w += LOCAL_CACHE_THREAD) {
      temp[localy][localz] += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h)));
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (localz == 0) {
    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
      temp[localy][0] += temp[localy][i];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float4 result = temp[0][0];
  for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
    result += temp[i][0];
  }
  result /= size.x * size.y;
  WRITE_IMAGE(dst_data, (int2)(X, 0), TO_FLT4(result));
}

__kernel void GlobalWCMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, float4 mask) {
  int X = get_global_id(0);  // H
  if (X >= size.x) {
    return;
  }
  float4 result = (float4)0.f;
  for (int w = 0; w < size.y; w++) {
    for (int c = 0; c < size.z; c++) {
      result += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X)));
    }
  }

  result /= size.y * size.w;
  FLT4 result2 = (FLT4)(0.f);
  result2.x = dot(TO_FLT4(result), (FLT4)(1.f));
  WRITE_IMAGE(dst_data, (int2)(0, X), result2);
}

__kernel void LocalWCMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, float4 mask) {
  int X = get_global_id(0);  // H
  int localy = get_local_id(1);
  int localz = get_local_id(2);
  if (X >= size.x) return;
  __local float4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];
  temp[localy][localz] = (float4)0.f;
  for (int w = localy; w < size.y; w += LOCAL_CACHE_THREAD) {
    for (int c = localz; c < size.z; c += LOCAL_CACHE_THREAD) {
      temp[localy][localz] += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X)));
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (localz == 0) {
    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
      temp[localy][0] += temp[localy][i];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float4 result = temp[0][0];
  for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
    result += temp[i][0];
  }
  result /= size.y * size.w;
  FLT4 result2 = (FLT4)(0.f);
  result2.x = dot(TO_FLT4(result), (FLT4)(1.f));
  WRITE_IMAGE(dst_data, (int2)(0, X), result2);
}

__kernel void GlobalHWSumSquare(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);
  if (X >= size.z) {
    return;
  }
  FLT4 result = (FLT4)0.f;
  for (int h = 0; h < size.x; h++) {
    for (int w = 0; w < size.y; w++) {
      FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h));
      result += current * current;
    }
  }
  WRITE_IMAGE(dst_data, (int2)(X, 0), result);
}

__kernel void LocalHWSumSquare(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);
  int localy = get_local_id(1);
  int localz = get_local_id(2);
  if (X >= size.z) return;
  __local FLT4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];
  temp[localy][localz] = (FLT4)0.f;
  for (int h = localy; h < size.x; h += LOCAL_CACHE_THREAD) {
    for (int w = localz; w < size.y; w += LOCAL_CACHE_THREAD) {
      FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h));
      temp[localy][localz] += current * current;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (localz == 0) {
    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
      temp[localy][0] += temp[localy][i];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  FLT4 result = temp[0][0];
  for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
    result += temp[i][0];
  }
  WRITE_IMAGE(dst_data, (int2)(X, 0), result);
}

__kernel void GlobalWCSumSquare(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size,
                                float4 mask) {
  int X = get_global_id(0);
  if (X >= size.x) {
    return;
  }
  FLT4 result = (FLT4)0.f;
  for (int w = 0; w < size.y; w++) {
    for (int c = 0; c < size.z; c++) {
      FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X));
      result += current * current;
    }
  }

  FLT4 result2 = (FLT4)(0.f);
  result2.x = dot(result, (FLT4)(1.f));
  WRITE_IMAGE(dst_data, (int2)(0, X), result2);
}

__kernel void LocalWCSumSquare(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size,
                               float4 mask) {
  int X = get_global_id(0);
  int localy = get_local_id(1);
  int localz = get_local_id(2);
  if (X >= size.x) return;
  __local FLT4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];
  temp[localy][localz] = (FLT4)0.f;
  for (int w = localy; w < size.y; w += LOCAL_CACHE_THREAD) {
    for (int c = localz; c < size.z; c += LOCAL_CACHE_THREAD) {
      FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X));
      temp[localy][localz] += current * current;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (localz == 0) {
    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
      temp[localy][0] += temp[localy][i];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  FLT4 result = temp[0][0];
  for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {
    result += temp[i][0];
  }
  FLT4 result2 = (FLT4)(0.f);
  result2.x = dot(result, (FLT4)(1.f));
  WRITE_IMAGE(dst_data, (int2)(0, X), result2);
}

__kernel void GlobalCMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, float4 mask) {
  int X = get_global_id(0);  // H
  int Y = get_global_id(1);  // W
  if (X >= size.x || Y >= size.y) {
    return;
  }
  float4 result = (float4)0.f;
  for (int c = 0; c < size.z; c++) {
    result += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(Y * size.z + c, X)));
  }

  result /= size.w;
  FLT4 result2 = (FLT4)(0.f);
  result2.x = dot(TO_FLT4(result), (FLT4)(1.f));
  WRITE_IMAGE(dst_data, (int2)(Y, X), result2);
}

#define GlobalHW(Method)                                                                                       \
  __kernel void GlobalHW##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) { \
    int X = get_global_id(0);                                                                                  \
    if (X >= size.z) {                                                                                         \
      return;                                                                                                  \
    }                                                                                                          \
    FLT4 result = (FLT4)Init##Method;                                                                          \
    for (int h = 0; h < size.x; h++) {                                                                         \
      for (int w = 0; w < size.y; w++) {                                                                       \
        FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h));                              \
        Do##Method(result, current);                                                                           \
      }                                                                                                        \
    }                                                                                                          \
    WRITE_IMAGE(dst_data, (int2)(X, 0), result);                                                               \
  }

#define GlobalWC(Method)                                                                                     \
  __kernel void GlobalWC##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, \
                                 float4 mask) {                                                              \
    int X = get_global_id(0);                                                                                \
    if (X >= size.x) {                                                                                       \
      return;                                                                                                \
    }                                                                                                        \
    FLT4 result = (FLT4)Init##Method;                                                                        \
    FLT4 maskFLT4 = TO_FLT4(mask);                                                                           \
    for (int w = 0; w < size.y; w++) {                                                                       \
      int c = 0;                                                                                             \
      for (; c < size.z - 1; c++) {                                                                          \
        FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X));                            \
        Do##Method(result, current);                                                                         \
      }                                                                                                      \
      FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X));                              \
      current += maskFLT4;                                                                                   \
      Do##Method(result, current);                                                                           \
    }                                                                                                        \
    Do##Method(result.x, result.y);                                                                          \
    Do##Method(result.x, result.z);                                                                          \
    Do##Method(result.x, result.w);                                                                          \
    FLT4 result2 = (FLT4)(0.f);                                                                              \
    result2.x = TO_FLT(result.x);                                                                            \
    WRITE_IMAGE(dst_data, (int2)(0, X), result2);                                                            \
  }

#define LocalHW(Method)                                                                                       \
  __kernel void LocalHW##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) { \
    int X = get_global_id(0);                                                                                 \
    int localy = get_local_id(1);                                                                             \
    int localz = get_local_id(2);                                                                             \
    if (X >= size.z) return;                                                                                  \
    __local float4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];                                              \
    temp[localy][localz] = (float4)Init##Method;                                                              \
    for (int h = localy; h < size.x; h += LOCAL_CACHE_THREAD) {                                               \
      for (int w = localz; w < size.y; w += LOCAL_CACHE_THREAD) {                                             \
        float4 current = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h)));           \
        Do##Method(temp[localy][localz], current);                                                            \
      }                                                                                                       \
    }                                                                                                         \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                             \
    if (localz == 0) {                                                                                        \
      for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {                                                          \
        Do##Method(temp[localy][0], temp[localy][i]);                                                         \
      }                                                                                                       \
    }                                                                                                         \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                             \
    float4 result = temp[0][0];                                                                               \
    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {                                                            \
      Do##Method(result, temp[i][0]);                                                                         \
    }                                                                                                         \
    WRITE_IMAGE(dst_data, (int2)(X, 0), TO_FLT4(result));                                                     \
  }

#define LocalWC(Method)                                                                                     \
  __kernel void LocalWC##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, \
                                float4 mask) {                                                              \
    int X = get_global_id(0);                                                                               \
    int localy = get_local_id(1);                                                                           \
    int localz = get_local_id(2);                                                                           \
    if (X >= size.x) return;                                                                                \
    __local float4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];                                            \
    temp[localy][localz] = (float4)Init##Method;                                                            \
    for (int w = localy; w < size.y; w += LOCAL_CACHE_THREAD) {                                             \
      int c = localz;                                                                                       \
      for (; c < size.z - 1; c += LOCAL_CACHE_THREAD) {                                                     \
        float4 current = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X)));         \
        Do##Method(temp[localy][localz], current);                                                          \
      }                                                                                                     \
      if (c == size.z - 1) {                                                                                \
        float4 current = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X)));         \
        current += mask;                                                                                    \
        Do##Method(temp[localy][localz], current);                                                          \
      }                                                                                                     \
    }                                                                                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                           \
    if (localz == 0) {                                                                                      \
      for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {                                                        \
        Do##Method(temp[localy][0], temp[localy][i]);                                                       \
      }                                                                                                     \
    }                                                                                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                           \
    float4 result = temp[0][0];                                                                             \
    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {                                                          \
      Do##Method(result, temp[i][0]);                                                                       \
    }                                                                                                       \
    Do##Method(result.x, result.y);                                                                         \
    Do##Method(result.x, result.z);                                                                         \
    Do##Method(result.x, result.w);                                                                         \
    FLT4 result2 = (FLT4)(0.f);                                                                             \
    result2.x = TO_FLT(result.x);                                                                           \
    WRITE_IMAGE(dst_data, (int2)(0, X), result2);                                                           \
  }

// HWC
__kernel void GlobalHWCMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  float4 value = (float4)0.f;
  for (int h = 0; h < size.x; h++) {
    for (int w = 0; w < size.y; w++) {
      for (int c4 = 0; c4 < size.z; c4++) {
        value += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c4, h)));
      }
    }
  }
  float4 result = (float4)0.f;
  result.x = dot((float4)(1.0f), value) / (size.x * size.y * size.w);
  WRITE_IMAGE(dst_data, (int2)(0, 0), TO_FLT4(result));
}

#define DoHWCSum(a, B) ((a) = dot((float4)(1.0f), (B)))
#define DoHWCMax(a, B) ((a) = max((B).x, max((B).y, max((B).z, (B).w))))
#define DoHWCMin(a, B) ((a) = min((B).x, min((B).y, min((B).z, (B).w))))
#define DoHWCProd(a, B) ((a) = (B).x * (B).y * (B).z * (B).w)

#define GlobalHWC(Method)                                                                                       \
  __kernel void GlobalHWC##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) { \
    float4 value = (float4)0.f;                                                                                 \
    for (int h = 0; h < size.x; h++) {                                                                          \
      for (int w = 0; w < size.y; w++) {                                                                        \
        for (int c4 = 0; c4 < size.z; c4++) {                                                                   \
          Do##Method(value, convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c4, h))));        \
        }                                                                                                       \
      }                                                                                                         \
    }                                                                                                           \
    float4 result = (float4)0.f;                                                                                \
    DoHWC##Method(result.x, value);                                                                             \
    WRITE_IMAGE(dst_data, (int2)(0, 0), TO_FLT4(result));                                                       \
  }

// H
__kernel void GlobalHMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int w = get_global_id(0);
  int c4 = get_global_id(1);
  float4 result = (float4)0.f;
  for (int h = 0; h < size.x; h++) {
    result += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c4, h)));
  }
  result /= size.x;
  WRITE_IMAGE(dst_data, (int2)(w * size.z + c4, 0), TO_FLT4(result));
}

#define GlobalH(Method)                                                                                       \
  __kernel void GlobalH##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) { \
    int w = get_global_id(0);                                                                                 \
    int c4 = get_global_id(1);                                                                                \
    float4 result = (float4)Init##Method;                                                                     \
    for (int h = 0; h < size.x; h++) {                                                                        \
      Do##Method(result, convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c4, h))));         \
    }                                                                                                         \
    WRITE_IMAGE(dst_data, (int2)(w * size.z + c4, 0), TO_FLT4(result));                                       \
  }

// W
__kernel void GlobalWMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int h = get_global_id(0);
  int c4 = get_global_id(1);
  float4 result = (float4)(0.f);
  for (int w = 0; w < size.y; w++) {
    result += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c4, h)));
  }
  result /= size.y;
  WRITE_IMAGE(dst_data, (int2)(c4, h), TO_FLT4(result));
}

#define GlobalW(Method)                                                                                       \
  __kernel void GlobalW##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) { \
    int h = get_global_id(0);                                                                                 \
    int c4 = get_global_id(1);                                                                                \
    float4 result = (float4)Init##Method;                                                                     \
    for (int w = 0; w < size.y; w++) {                                                                        \
      Do##Method(result, convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c4, h))));         \
    }                                                                                                         \
    WRITE_IMAGE(dst_data, (int2)(c4, h), TO_FLT4(result));                                                    \
  }

// C
#define DoCSum(a, B) ((a) = dot((float4)(1.0f), (B)))
#define DoCMax(a, B) ((a) = max((B).x, max((B).y, max((B).z, (B).w))))
#define DoCMin(a, B) ((a) = min((B).x, min((B).y, min((B).z, (B).w))))
#define DoCProd(a, B) ((a) = (B).x * (B).y * (B).z * (B).w)

#define GlobalC(Method)                                                                                           \
  __kernel void GlobalC##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size,       \
                                float4 mask) {                                                                    \
    int h = get_global_id(0);                                                                                     \
    int w = get_global_id(1);                                                                                     \
    float4 value = (float4)Init##Method;                                                                          \
    for (int c4 = 0; c4 < size.z - 1; c4++) {                                                                     \
      Do##Method(value, convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c4, h))));              \
    }                                                                                                             \
    Do##Method(value, convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + size.z - 1, h))) + mask); \
    float4 result = (float4)0.f;                                                                                  \
    DoC##Method(result.x, value);                                                                                 \
    WRITE_IMAGE(dst_data, (int2)(w * size.z, h), TO_FLT4(result));                                                \
  }

#define DoSum(A, B) A += B
#define InitSum 0.f
GlobalHWC(Sum) GlobalHW(Sum) GlobalWC(Sum) GlobalH(Sum) GlobalW(Sum) GlobalC(Sum) LocalHW(Sum) LocalWC(Sum)

#define DoMin(A, B) A = min(A, B)
#define InitMin 10000.f
  GlobalHWC(Min) GlobalHW(Min) GlobalWC(Min) GlobalH(Min) GlobalW(Min) GlobalC(Min) LocalHW(Min) LocalWC(Min)

#define DoMax(A, B) A = max(A, B)
#define InitMax -10000.f
    GlobalHWC(Max) GlobalHW(Max) GlobalWC(Max) GlobalH(Max) GlobalW(Max) GlobalC(Max) LocalHW(Max) LocalWC(Max)

#define DoProd(A, B) A *= B
#define InitProd 1.f
      GlobalHWC(Prod) GlobalHW(Prod) GlobalWC(Prod) GlobalH(Prod) GlobalW(Prod) GlobalC(Prod) LocalHW(Prod)
        LocalWC(Prod)
