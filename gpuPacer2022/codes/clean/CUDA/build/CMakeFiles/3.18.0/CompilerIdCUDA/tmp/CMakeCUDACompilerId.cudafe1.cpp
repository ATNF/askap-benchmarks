# 1 "CMakeCUDACompilerId.cu"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
# 1
#pragma GCC diagnostic push
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"
# 1
#pragma GCC diagnostic ignored "-Wunused-function"
# 1
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
# 1
#pragma GCC diagnostic pop
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"

# 1
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false
#if defined(__nv_is_extended_device_lambda_closure_type) && defined(__nv_is_extended_host_device_lambda_closure_type)
#endif

# 1
# 61 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
#pragma GCC diagnostic push
# 64
#pragma GCC diagnostic ignored "-Wunused-function"
# 68 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_types.h"
#if 0
# 68
enum cudaRoundMode { 
# 70
cudaRoundNearest, 
# 71
cudaRoundZero, 
# 72
cudaRoundPosInf, 
# 73
cudaRoundMinInf
# 74
}; 
#endif
# 100 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 100
struct char1 { 
# 102
signed char x; 
# 103
}; 
#endif
# 105 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 105
struct uchar1 { 
# 107
unsigned char x; 
# 108
}; 
#endif
# 111 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 111
struct __attribute((aligned(2))) char2 { 
# 113
signed char x, y; 
# 114
}; 
#endif
# 116 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 116
struct __attribute((aligned(2))) uchar2 { 
# 118
unsigned char x, y; 
# 119
}; 
#endif
# 121 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 121
struct char3 { 
# 123
signed char x, y, z; 
# 124
}; 
#endif
# 126 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 126
struct uchar3 { 
# 128
unsigned char x, y, z; 
# 129
}; 
#endif
# 131 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 131
struct __attribute((aligned(4))) char4 { 
# 133
signed char x, y, z, w; 
# 134
}; 
#endif
# 136 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 136
struct __attribute((aligned(4))) uchar4 { 
# 138
unsigned char x, y, z, w; 
# 139
}; 
#endif
# 141 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 141
struct short1 { 
# 143
short x; 
# 144
}; 
#endif
# 146 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 146
struct ushort1 { 
# 148
unsigned short x; 
# 149
}; 
#endif
# 151 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 151
struct __attribute((aligned(4))) short2 { 
# 153
short x, y; 
# 154
}; 
#endif
# 156 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 156
struct __attribute((aligned(4))) ushort2 { 
# 158
unsigned short x, y; 
# 159
}; 
#endif
# 161 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 161
struct short3 { 
# 163
short x, y, z; 
# 164
}; 
#endif
# 166 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 166
struct ushort3 { 
# 168
unsigned short x, y, z; 
# 169
}; 
#endif
# 171 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 171
struct __attribute((aligned(8))) short4 { short x; short y; short z; short w; }; 
#endif
# 172 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 172
struct __attribute((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
# 174 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 174
struct int1 { 
# 176
int x; 
# 177
}; 
#endif
# 179 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 179
struct uint1 { 
# 181
unsigned x; 
# 182
}; 
#endif
# 184 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 184
struct __attribute((aligned(8))) int2 { int x; int y; }; 
#endif
# 185 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 185
struct __attribute((aligned(8))) uint2 { unsigned x; unsigned y; }; 
#endif
# 187 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 187
struct int3 { 
# 189
int x, y, z; 
# 190
}; 
#endif
# 192 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 192
struct uint3 { 
# 194
unsigned x, y, z; 
# 195
}; 
#endif
# 197 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 197
struct __attribute((aligned(16))) int4 { 
# 199
int x, y, z, w; 
# 200
}; 
#endif
# 202 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 202
struct __attribute((aligned(16))) uint4 { 
# 204
unsigned x, y, z, w; 
# 205
}; 
#endif
# 207 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 207
struct long1 { 
# 209
long x; 
# 210
}; 
#endif
# 212 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 212
struct ulong1 { 
# 214
unsigned long x; 
# 215
}; 
#endif
# 222 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 222
struct __attribute((aligned((2) * sizeof(long)))) long2 { 
# 224
long x, y; 
# 225
}; 
#endif
# 227 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 227
struct __attribute((aligned((2) * sizeof(unsigned long)))) ulong2 { 
# 229
unsigned long x, y; 
# 230
}; 
#endif
# 234 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 234
struct long3 { 
# 236
long x, y, z; 
# 237
}; 
#endif
# 239 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 239
struct ulong3 { 
# 241
unsigned long x, y, z; 
# 242
}; 
#endif
# 244 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 244
struct __attribute((aligned(16))) long4 { 
# 246
long x, y, z, w; 
# 247
}; 
#endif
# 249 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 249
struct __attribute((aligned(16))) ulong4 { 
# 251
unsigned long x, y, z, w; 
# 252
}; 
#endif
# 254 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 254
struct float1 { 
# 256
float x; 
# 257
}; 
#endif
# 276 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 276
struct __attribute((aligned(8))) float2 { float x; float y; }; 
#endif
# 281 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 281
struct float3 { 
# 283
float x, y, z; 
# 284
}; 
#endif
# 286 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 286
struct __attribute((aligned(16))) float4 { 
# 288
float x, y, z, w; 
# 289
}; 
#endif
# 291 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 291
struct longlong1 { 
# 293
long long x; 
# 294
}; 
#endif
# 296 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 296
struct ulonglong1 { 
# 298
unsigned long long x; 
# 299
}; 
#endif
# 301 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 301
struct __attribute((aligned(16))) longlong2 { 
# 303
long long x, y; 
# 304
}; 
#endif
# 306 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 306
struct __attribute((aligned(16))) ulonglong2 { 
# 308
unsigned long long x, y; 
# 309
}; 
#endif
# 311 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 311
struct longlong3 { 
# 313
long long x, y, z; 
# 314
}; 
#endif
# 316 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 316
struct ulonglong3 { 
# 318
unsigned long long x, y, z; 
# 319
}; 
#endif
# 321 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 321
struct __attribute((aligned(16))) longlong4 { 
# 323
long long x, y, z, w; 
# 324
}; 
#endif
# 326 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 326
struct __attribute((aligned(16))) ulonglong4 { 
# 328
unsigned long long x, y, z, w; 
# 329
}; 
#endif
# 331 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 331
struct double1 { 
# 333
double x; 
# 334
}; 
#endif
# 336 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 336
struct __attribute((aligned(16))) double2 { 
# 338
double x, y; 
# 339
}; 
#endif
# 341 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 341
struct double3 { 
# 343
double x, y, z; 
# 344
}; 
#endif
# 346 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 346
struct __attribute((aligned(16))) double4 { 
# 348
double x, y, z, w; 
# 349
}; 
#endif
# 363 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char1 
# 363
char1; 
#endif
# 364 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar1 
# 364
uchar1; 
#endif
# 365 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char2 
# 365
char2; 
#endif
# 366 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar2 
# 366
uchar2; 
#endif
# 367 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char3 
# 367
char3; 
#endif
# 368 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar3 
# 368
uchar3; 
#endif
# 369 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char4 
# 369
char4; 
#endif
# 370 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar4 
# 370
uchar4; 
#endif
# 371 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short1 
# 371
short1; 
#endif
# 372 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort1 
# 372
ushort1; 
#endif
# 373 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short2 
# 373
short2; 
#endif
# 374 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort2 
# 374
ushort2; 
#endif
# 375 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short3 
# 375
short3; 
#endif
# 376 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort3 
# 376
ushort3; 
#endif
# 377 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short4 
# 377
short4; 
#endif
# 378 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort4 
# 378
ushort4; 
#endif
# 379 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int1 
# 379
int1; 
#endif
# 380 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint1 
# 380
uint1; 
#endif
# 381 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int2 
# 381
int2; 
#endif
# 382 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint2 
# 382
uint2; 
#endif
# 383 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int3 
# 383
int3; 
#endif
# 384 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint3 
# 384
uint3; 
#endif
# 385 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int4 
# 385
int4; 
#endif
# 386 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint4 
# 386
uint4; 
#endif
# 387 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long1 
# 387
long1; 
#endif
# 388 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong1 
# 388
ulong1; 
#endif
# 389 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long2 
# 389
long2; 
#endif
# 390 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong2 
# 390
ulong2; 
#endif
# 391 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long3 
# 391
long3; 
#endif
# 392 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong3 
# 392
ulong3; 
#endif
# 393 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long4 
# 393
long4; 
#endif
# 394 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong4 
# 394
ulong4; 
#endif
# 395 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float1 
# 395
float1; 
#endif
# 396 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float2 
# 396
float2; 
#endif
# 397 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float3 
# 397
float3; 
#endif
# 398 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float4 
# 398
float4; 
#endif
# 399 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong1 
# 399
longlong1; 
#endif
# 400 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong1 
# 400
ulonglong1; 
#endif
# 401 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong2 
# 401
longlong2; 
#endif
# 402 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong2 
# 402
ulonglong2; 
#endif
# 403 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong3 
# 403
longlong3; 
#endif
# 404 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong3 
# 404
ulonglong3; 
#endif
# 405 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong4 
# 405
longlong4; 
#endif
# 406 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong4 
# 406
ulonglong4; 
#endif
# 407 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double1 
# 407
double1; 
#endif
# 408 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double2 
# 408
double2; 
#endif
# 409 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double3 
# 409
double3; 
#endif
# 410 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double4 
# 410
double4; 
#endif
# 418 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 418
struct dim3 { 
# 420
unsigned x, y, z; 
# 432
}; 
#endif
# 434 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef dim3 
# 434
dim3; 
#endif
# 143 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/lib/gcc/x86_64-pc-linux-gnu/11.1.0/include/stddef.h" 3
typedef long ptrdiff_t; 
# 209 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/lib/gcc/x86_64-pc-linux-gnu/11.1.0/include/stddef.h" 3
typedef unsigned long size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
# 426 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/lib/gcc/x86_64-pc-linux-gnu/11.1.0/include/stddef.h" 3
typedef 
# 415 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/lib/gcc/x86_64-pc-linux-gnu/11.1.0/include/stddef.h" 3
struct { 
# 416
long long __max_align_ll __attribute((__aligned__(__alignof__(long long)))); 
# 417
long double __max_align_ld __attribute((__aligned__(__alignof__(long double)))); 
# 426 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/lib/gcc/x86_64-pc-linux-gnu/11.1.0/include/stddef.h" 3
} max_align_t; 
# 433
typedef __decltype((nullptr)) nullptr_t; 
# 199 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 199
enum cudaError { 
# 206
cudaSuccess, 
# 212
cudaErrorInvalidValue, 
# 218
cudaErrorMemoryAllocation, 
# 224
cudaErrorInitializationError, 
# 231
cudaErrorCudartUnloading, 
# 238
cudaErrorProfilerDisabled, 
# 246
cudaErrorProfilerNotInitialized, 
# 253
cudaErrorProfilerAlreadyStarted, 
# 260
cudaErrorProfilerAlreadyStopped, 
# 269 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorInvalidConfiguration, 
# 275
cudaErrorInvalidPitchValue = 12, 
# 281
cudaErrorInvalidSymbol, 
# 289
cudaErrorInvalidHostPointer = 16, 
# 297
cudaErrorInvalidDevicePointer, 
# 303
cudaErrorInvalidTexture, 
# 309
cudaErrorInvalidTextureBinding, 
# 316
cudaErrorInvalidChannelDescriptor, 
# 322
cudaErrorInvalidMemcpyDirection, 
# 332 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorAddressOfConstant, 
# 341 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorTextureFetchFailed, 
# 350 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorTextureNotBound, 
# 359 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSynchronizationError, 
# 365
cudaErrorInvalidFilterSetting, 
# 371
cudaErrorInvalidNormSetting, 
# 379
cudaErrorMixedDeviceExecution, 
# 387
cudaErrorNotYetImplemented = 31, 
# 396 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorMemoryValueTooLarge, 
# 403
cudaErrorStubLibrary = 34, 
# 410
cudaErrorInsufficientDriver, 
# 417
cudaErrorCallRequiresNewerDriver, 
# 423
cudaErrorInvalidSurface, 
# 429
cudaErrorDuplicateVariableName = 43, 
# 435
cudaErrorDuplicateTextureName, 
# 441
cudaErrorDuplicateSurfaceName, 
# 451 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorDevicesUnavailable, 
# 464 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorIncompatibleDriverContext = 49, 
# 470
cudaErrorMissingConfiguration = 52, 
# 479 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorPriorLaunchFailure, 
# 486
cudaErrorLaunchMaxDepthExceeded = 65, 
# 494
cudaErrorLaunchFileScopedTex, 
# 502
cudaErrorLaunchFileScopedSurf, 
# 517 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSyncDepthExceeded, 
# 529 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchPendingCountExceeded, 
# 535
cudaErrorInvalidDeviceFunction = 98, 
# 541
cudaErrorNoDevice = 100, 
# 548
cudaErrorInvalidDevice, 
# 553
cudaErrorDeviceNotLicensed, 
# 562 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSoftwareValidityNotEstablished, 
# 567
cudaErrorStartupFailure = 127, 
# 572
cudaErrorInvalidKernelImage = 200, 
# 582 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorDeviceUninitialized, 
# 587
cudaErrorMapBufferObjectFailed = 205, 
# 592
cudaErrorUnmapBufferObjectFailed, 
# 598
cudaErrorArrayIsMapped, 
# 603
cudaErrorAlreadyMapped, 
# 611
cudaErrorNoKernelImageForDevice, 
# 616
cudaErrorAlreadyAcquired, 
# 621
cudaErrorNotMapped, 
# 627
cudaErrorNotMappedAsArray, 
# 633
cudaErrorNotMappedAsPointer, 
# 639
cudaErrorECCUncorrectable, 
# 645
cudaErrorUnsupportedLimit, 
# 651
cudaErrorDeviceAlreadyInUse, 
# 657
cudaErrorPeerAccessUnsupported, 
# 663
cudaErrorInvalidPtx, 
# 668
cudaErrorInvalidGraphicsContext, 
# 674
cudaErrorNvlinkUncorrectable, 
# 681
cudaErrorJitCompilerNotFound, 
# 688
cudaErrorUnsupportedPtxVersion, 
# 695
cudaErrorJitCompilationDisabled, 
# 700
cudaErrorUnsupportedExecAffinity, 
# 705
cudaErrorInvalidSource = 300, 
# 710
cudaErrorFileNotFound, 
# 715
cudaErrorSharedObjectSymbolNotFound, 
# 720
cudaErrorSharedObjectInitFailed, 
# 725
cudaErrorOperatingSystem, 
# 732
cudaErrorInvalidResourceHandle = 400, 
# 738
cudaErrorIllegalState, 
# 745
cudaErrorSymbolNotFound = 500, 
# 753
cudaErrorNotReady = 600, 
# 761
cudaErrorIllegalAddress = 700, 
# 770 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchOutOfResources, 
# 781 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchTimeout, 
# 787
cudaErrorLaunchIncompatibleTexturing, 
# 794
cudaErrorPeerAccessAlreadyEnabled, 
# 801
cudaErrorPeerAccessNotEnabled, 
# 814 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSetOnActiveProcess = 708, 
# 821
cudaErrorContextIsDestroyed, 
# 828
cudaErrorAssert, 
# 835
cudaErrorTooManyPeers, 
# 841
cudaErrorHostMemoryAlreadyRegistered, 
# 847
cudaErrorHostMemoryNotRegistered, 
# 856 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorHardwareStackError, 
# 864
cudaErrorIllegalInstruction, 
# 873 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorMisalignedAddress, 
# 884 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorInvalidAddressSpace, 
# 892
cudaErrorInvalidPc, 
# 903 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchFailure, 
# 912 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorCooperativeLaunchTooLarge, 
# 917
cudaErrorNotPermitted = 800, 
# 923
cudaErrorNotSupported, 
# 932 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSystemNotReady, 
# 939
cudaErrorSystemDriverMismatch, 
# 948 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorCompatNotSupportedOnDevice, 
# 953
cudaErrorMpsConnectionFailed, 
# 958
cudaErrorMpsRpcFailure, 
# 964
cudaErrorMpsServerNotReady, 
# 969
cudaErrorMpsMaxClientsReached, 
# 974
cudaErrorMpsMaxConnectionsReached, 
# 979
cudaErrorStreamCaptureUnsupported = 900, 
# 985
cudaErrorStreamCaptureInvalidated, 
# 991
cudaErrorStreamCaptureMerge, 
# 996
cudaErrorStreamCaptureUnmatched, 
# 1002
cudaErrorStreamCaptureUnjoined, 
# 1009
cudaErrorStreamCaptureIsolation, 
# 1015
cudaErrorStreamCaptureImplicit, 
# 1021
cudaErrorCapturedEvent, 
# 1028
cudaErrorStreamCaptureWrongThread, 
# 1033
cudaErrorTimeout, 
# 1039
cudaErrorGraphExecUpdateFailure, 
# 1049 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorExternalDevice, 
# 1054
cudaErrorUnknown = 999, 
# 1062
cudaErrorApiFailureBase = 10000
# 1063
}; 
#endif
# 1068 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1068
enum cudaChannelFormatKind { 
# 1070
cudaChannelFormatKindSigned, 
# 1071
cudaChannelFormatKindUnsigned, 
# 1072
cudaChannelFormatKindFloat, 
# 1073
cudaChannelFormatKindNone, 
# 1074
cudaChannelFormatKindNV12
# 1075
}; 
#endif
# 1080 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1080
struct cudaChannelFormatDesc { 
# 1082
int x; 
# 1083
int y; 
# 1084
int z; 
# 1085
int w; 
# 1086
cudaChannelFormatKind f; 
# 1087
}; 
#endif
# 1092 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
typedef struct cudaArray *cudaArray_t; 
# 1097
typedef const cudaArray *cudaArray_const_t; 
# 1099
struct cudaArray; 
# 1104
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
# 1109
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
# 1111
struct cudaMipmappedArray; 
# 1121 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1121
struct cudaArraySparseProperties { 
# 1122
struct { 
# 1123
unsigned width; 
# 1124
unsigned height; 
# 1125
unsigned depth; 
# 1126
} tileExtent; 
# 1127
unsigned miptailFirstLevel; 
# 1128
unsigned long long miptailSize; 
# 1129
unsigned flags; 
# 1130
unsigned reserved[4]; 
# 1131
}; 
#endif
# 1136 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1136
enum cudaMemoryType { 
# 1138
cudaMemoryTypeUnregistered, 
# 1139
cudaMemoryTypeHost, 
# 1140
cudaMemoryTypeDevice, 
# 1141
cudaMemoryTypeManaged
# 1142
}; 
#endif
# 1147 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1147
enum cudaMemcpyKind { 
# 1149
cudaMemcpyHostToHost, 
# 1150
cudaMemcpyHostToDevice, 
# 1151
cudaMemcpyDeviceToHost, 
# 1152
cudaMemcpyDeviceToDevice, 
# 1153
cudaMemcpyDefault
# 1154
}; 
#endif
# 1161 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1161
struct cudaPitchedPtr { 
# 1163
void *ptr; 
# 1164
size_t pitch; 
# 1165
size_t xsize; 
# 1166
size_t ysize; 
# 1167
}; 
#endif
# 1174 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1174
struct cudaExtent { 
# 1176
size_t width; 
# 1177
size_t height; 
# 1178
size_t depth; 
# 1179
}; 
#endif
# 1186 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1186
struct cudaPos { 
# 1188
size_t x; 
# 1189
size_t y; 
# 1190
size_t z; 
# 1191
}; 
#endif
# 1196 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1196
struct cudaMemcpy3DParms { 
# 1198
cudaArray_t srcArray; 
# 1199
cudaPos srcPos; 
# 1200
cudaPitchedPtr srcPtr; 
# 1202
cudaArray_t dstArray; 
# 1203
cudaPos dstPos; 
# 1204
cudaPitchedPtr dstPtr; 
# 1206
cudaExtent extent; 
# 1207
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1208
}; 
#endif
# 1213 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1213
struct cudaMemcpy3DPeerParms { 
# 1215
cudaArray_t srcArray; 
# 1216
cudaPos srcPos; 
# 1217
cudaPitchedPtr srcPtr; 
# 1218
int srcDevice; 
# 1220
cudaArray_t dstArray; 
# 1221
cudaPos dstPos; 
# 1222
cudaPitchedPtr dstPtr; 
# 1223
int dstDevice; 
# 1225
cudaExtent extent; 
# 1226
}; 
#endif
# 1231 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1231
struct cudaMemsetParams { 
# 1232
void *dst; 
# 1233
size_t pitch; 
# 1234
unsigned value; 
# 1235
unsigned elementSize; 
# 1236
size_t width; 
# 1237
size_t height; 
# 1238
}; 
#endif
# 1243 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1243
enum cudaAccessProperty { 
# 1244
cudaAccessPropertyNormal, 
# 1245
cudaAccessPropertyStreaming, 
# 1246
cudaAccessPropertyPersisting
# 1247
}; 
#endif
# 1260 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1260
struct cudaAccessPolicyWindow { 
# 1261
void *base_ptr; 
# 1262
size_t num_bytes; 
# 1263
float hitRatio; 
# 1264
cudaAccessProperty hitProp; 
# 1265
cudaAccessProperty missProp; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1266
}; 
#endif
# 1278 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
typedef void (*cudaHostFn_t)(void * userData); 
# 1283
#if 0
# 1283
struct cudaHostNodeParams { 
# 1284
cudaHostFn_t fn; 
# 1285
void *userData; 
# 1286
}; 
#endif
# 1291 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1291
enum cudaStreamCaptureStatus { 
# 1292
cudaStreamCaptureStatusNone, 
# 1293
cudaStreamCaptureStatusActive, 
# 1294
cudaStreamCaptureStatusInvalidated
# 1296
}; 
#endif
# 1302 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1302
enum cudaStreamCaptureMode { 
# 1303
cudaStreamCaptureModeGlobal, 
# 1304
cudaStreamCaptureModeThreadLocal, 
# 1305
cudaStreamCaptureModeRelaxed
# 1306
}; 
#endif
# 1308 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1308
enum cudaSynchronizationPolicy { 
# 1309
cudaSyncPolicyAuto = 1, 
# 1310
cudaSyncPolicySpin, 
# 1311
cudaSyncPolicyYield, 
# 1312
cudaSyncPolicyBlockingSync
# 1313
}; 
#endif
# 1318 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1318
enum cudaStreamAttrID { 
# 1319
cudaStreamAttributeAccessPolicyWindow = 1, 
# 1320
cudaStreamAttributeSynchronizationPolicy = 3
# 1321
}; 
#endif
# 1326 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1326
union cudaStreamAttrValue { 
# 1327
cudaAccessPolicyWindow accessPolicyWindow; 
# 1328
cudaSynchronizationPolicy syncPolicy; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1329
}; 
#endif
# 1334 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1334
enum cudaStreamUpdateCaptureDependenciesFlags { 
# 1335
cudaStreamAddCaptureDependencies, 
# 1336
cudaStreamSetCaptureDependencies
# 1337
}; 
#endif
# 1342 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1342
enum cudaUserObjectFlags { 
# 1343
cudaUserObjectNoDestructorSync = 1
# 1344
}; 
#endif
# 1349 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1349
enum cudaUserObjectRetainFlags { 
# 1350
cudaGraphUserObjectMove = 1
# 1351
}; 
#endif
# 1356 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
struct cudaGraphicsResource; 
# 1361
#if 0
# 1361
enum cudaGraphicsRegisterFlags { 
# 1363
cudaGraphicsRegisterFlagsNone, 
# 1364
cudaGraphicsRegisterFlagsReadOnly, 
# 1365
cudaGraphicsRegisterFlagsWriteDiscard, 
# 1366
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
# 1367
cudaGraphicsRegisterFlagsTextureGather = 8
# 1368
}; 
#endif
# 1373 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1373
enum cudaGraphicsMapFlags { 
# 1375
cudaGraphicsMapFlagsNone, 
# 1376
cudaGraphicsMapFlagsReadOnly, 
# 1377
cudaGraphicsMapFlagsWriteDiscard
# 1378
}; 
#endif
# 1383 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1383
enum cudaGraphicsCubeFace { 
# 1385
cudaGraphicsCubeFacePositiveX, 
# 1386
cudaGraphicsCubeFaceNegativeX, 
# 1387
cudaGraphicsCubeFacePositiveY, 
# 1388
cudaGraphicsCubeFaceNegativeY, 
# 1389
cudaGraphicsCubeFacePositiveZ, 
# 1390
cudaGraphicsCubeFaceNegativeZ
# 1391
}; 
#endif
# 1396 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1396
enum cudaKernelNodeAttrID { 
# 1397
cudaKernelNodeAttributeAccessPolicyWindow = 1, 
# 1398
cudaKernelNodeAttributeCooperative
# 1399
}; 
#endif
# 1404 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1404
union cudaKernelNodeAttrValue { 
# 1405
cudaAccessPolicyWindow accessPolicyWindow; 
# 1406
int cooperative; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1407
}; 
#endif
# 1412 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1412
enum cudaResourceType { 
# 1414
cudaResourceTypeArray, 
# 1415
cudaResourceTypeMipmappedArray, 
# 1416
cudaResourceTypeLinear, 
# 1417
cudaResourceTypePitch2D
# 1418
}; 
#endif
# 1423 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1423
enum cudaResourceViewFormat { 
# 1425
cudaResViewFormatNone, 
# 1426
cudaResViewFormatUnsignedChar1, 
# 1427
cudaResViewFormatUnsignedChar2, 
# 1428
cudaResViewFormatUnsignedChar4, 
# 1429
cudaResViewFormatSignedChar1, 
# 1430
cudaResViewFormatSignedChar2, 
# 1431
cudaResViewFormatSignedChar4, 
# 1432
cudaResViewFormatUnsignedShort1, 
# 1433
cudaResViewFormatUnsignedShort2, 
# 1434
cudaResViewFormatUnsignedShort4, 
# 1435
cudaResViewFormatSignedShort1, 
# 1436
cudaResViewFormatSignedShort2, 
# 1437
cudaResViewFormatSignedShort4, 
# 1438
cudaResViewFormatUnsignedInt1, 
# 1439
cudaResViewFormatUnsignedInt2, 
# 1440
cudaResViewFormatUnsignedInt4, 
# 1441
cudaResViewFormatSignedInt1, 
# 1442
cudaResViewFormatSignedInt2, 
# 1443
cudaResViewFormatSignedInt4, 
# 1444
cudaResViewFormatHalf1, 
# 1445
cudaResViewFormatHalf2, 
# 1446
cudaResViewFormatHalf4, 
# 1447
cudaResViewFormatFloat1, 
# 1448
cudaResViewFormatFloat2, 
# 1449
cudaResViewFormatFloat4, 
# 1450
cudaResViewFormatUnsignedBlockCompressed1, 
# 1451
cudaResViewFormatUnsignedBlockCompressed2, 
# 1452
cudaResViewFormatUnsignedBlockCompressed3, 
# 1453
cudaResViewFormatUnsignedBlockCompressed4, 
# 1454
cudaResViewFormatSignedBlockCompressed4, 
# 1455
cudaResViewFormatUnsignedBlockCompressed5, 
# 1456
cudaResViewFormatSignedBlockCompressed5, 
# 1457
cudaResViewFormatUnsignedBlockCompressed6H, 
# 1458
cudaResViewFormatSignedBlockCompressed6H, 
# 1459
cudaResViewFormatUnsignedBlockCompressed7
# 1460
}; 
#endif
# 1465 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1465
struct cudaResourceDesc { 
# 1466
cudaResourceType resType; 
# 1468
union { 
# 1469
struct { 
# 1470
cudaArray_t array; 
# 1471
} array; 
# 1472
struct { 
# 1473
cudaMipmappedArray_t mipmap; 
# 1474
} mipmap; 
# 1475
struct { 
# 1476
void *devPtr; 
# 1477
cudaChannelFormatDesc desc; 
# 1478
size_t sizeInBytes; 
# 1479
} linear; 
# 1480
struct { 
# 1481
void *devPtr; 
# 1482
cudaChannelFormatDesc desc; 
# 1483
size_t width; 
# 1484
size_t height; 
# 1485
size_t pitchInBytes; 
# 1486
} pitch2D; 
# 1487
} res; 
# 1488
}; 
#endif
# 1493 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1493
struct cudaResourceViewDesc { 
# 1495
cudaResourceViewFormat format; 
# 1496
size_t width; 
# 1497
size_t height; 
# 1498
size_t depth; 
# 1499
unsigned firstMipmapLevel; 
# 1500
unsigned lastMipmapLevel; 
# 1501
unsigned firstLayer; 
# 1502
unsigned lastLayer; 
# 1503
}; 
#endif
# 1508 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1508
struct cudaPointerAttributes { 
# 1514
cudaMemoryType type; 
# 1525 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
int device; 
# 1531
void *devicePointer; 
# 1540 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
void *hostPointer; 
# 1541
}; 
#endif
# 1546 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1546
struct cudaFuncAttributes { 
# 1553
size_t sharedSizeBytes; 
# 1559
size_t constSizeBytes; 
# 1564
size_t localSizeBytes; 
# 1571
int maxThreadsPerBlock; 
# 1576
int numRegs; 
# 1583
int ptxVersion; 
# 1590
int binaryVersion; 
# 1596
int cacheModeCA; 
# 1603
int maxDynamicSharedSizeBytes; 
# 1612 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
int preferredShmemCarveout; 
# 1613
}; 
#endif
# 1618 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1618
enum cudaFuncAttribute { 
# 1620
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
# 1621
cudaFuncAttributePreferredSharedMemoryCarveout, 
# 1622
cudaFuncAttributeMax
# 1623
}; 
#endif
# 1628 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1628
enum cudaFuncCache { 
# 1630
cudaFuncCachePreferNone, 
# 1631
cudaFuncCachePreferShared, 
# 1632
cudaFuncCachePreferL1, 
# 1633
cudaFuncCachePreferEqual
# 1634
}; 
#endif
# 1640 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1640
enum cudaSharedMemConfig { 
# 1642
cudaSharedMemBankSizeDefault, 
# 1643
cudaSharedMemBankSizeFourByte, 
# 1644
cudaSharedMemBankSizeEightByte
# 1645
}; 
#endif
# 1650 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1650
enum cudaSharedCarveout { 
# 1651
cudaSharedmemCarveoutDefault = (-1), 
# 1652
cudaSharedmemCarveoutMaxShared = 100, 
# 1653
cudaSharedmemCarveoutMaxL1 = 0
# 1654
}; 
#endif
# 1659 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1659
enum cudaComputeMode { 
# 1661
cudaComputeModeDefault, 
# 1662
cudaComputeModeExclusive, 
# 1663
cudaComputeModeProhibited, 
# 1664
cudaComputeModeExclusiveProcess
# 1665
}; 
#endif
# 1670 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1670
enum cudaLimit { 
# 1672
cudaLimitStackSize, 
# 1673
cudaLimitPrintfFifoSize, 
# 1674
cudaLimitMallocHeapSize, 
# 1675
cudaLimitDevRuntimeSyncDepth, 
# 1676
cudaLimitDevRuntimePendingLaunchCount, 
# 1677
cudaLimitMaxL2FetchGranularity, 
# 1678
cudaLimitPersistingL2CacheSize
# 1679
}; 
#endif
# 1684 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1684
enum cudaMemoryAdvise { 
# 1686
cudaMemAdviseSetReadMostly = 1, 
# 1687
cudaMemAdviseUnsetReadMostly, 
# 1688
cudaMemAdviseSetPreferredLocation, 
# 1689
cudaMemAdviseUnsetPreferredLocation, 
# 1690
cudaMemAdviseSetAccessedBy, 
# 1691
cudaMemAdviseUnsetAccessedBy
# 1692
}; 
#endif
# 1697 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1697
enum cudaMemRangeAttribute { 
# 1699
cudaMemRangeAttributeReadMostly = 1, 
# 1700
cudaMemRangeAttributePreferredLocation, 
# 1701
cudaMemRangeAttributeAccessedBy, 
# 1702
cudaMemRangeAttributeLastPrefetchLocation
# 1703
}; 
#endif
# 1708 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1708
enum cudaOutputMode { 
# 1710
cudaKeyValuePair, 
# 1711
cudaCSV
# 1712
}; 
#endif
# 1717 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1717
enum cudaFlushGPUDirectRDMAWritesOptions { 
# 1718
cudaFlushGPUDirectRDMAWritesOptionHost = (1 << 0), 
# 1719
cudaFlushGPUDirectRDMAWritesOptionMemOps
# 1720
}; 
#endif
# 1725 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1725
enum cudaGPUDirectRDMAWritesOrdering { 
# 1726
cudaGPUDirectRDMAWritesOrderingNone, 
# 1727
cudaGPUDirectRDMAWritesOrderingOwner = 100, 
# 1728
cudaGPUDirectRDMAWritesOrderingAllDevices = 200
# 1729
}; 
#endif
# 1734 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1734
enum cudaFlushGPUDirectRDMAWritesScope { 
# 1735
cudaFlushGPUDirectRDMAWritesToOwner = 100, 
# 1736
cudaFlushGPUDirectRDMAWritesToAllDevices = 200
# 1737
}; 
#endif
# 1742 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1742
enum cudaFlushGPUDirectRDMAWritesTarget { 
# 1743
cudaFlushGPUDirectRDMAWritesTargetCurrentDevice
# 1744
}; 
#endif
# 1750 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1750
enum cudaDeviceAttr { 
# 1752
cudaDevAttrMaxThreadsPerBlock = 1, 
# 1753
cudaDevAttrMaxBlockDimX, 
# 1754
cudaDevAttrMaxBlockDimY, 
# 1755
cudaDevAttrMaxBlockDimZ, 
# 1756
cudaDevAttrMaxGridDimX, 
# 1757
cudaDevAttrMaxGridDimY, 
# 1758
cudaDevAttrMaxGridDimZ, 
# 1759
cudaDevAttrMaxSharedMemoryPerBlock, 
# 1760
cudaDevAttrTotalConstantMemory, 
# 1761
cudaDevAttrWarpSize, 
# 1762
cudaDevAttrMaxPitch, 
# 1763
cudaDevAttrMaxRegistersPerBlock, 
# 1764
cudaDevAttrClockRate, 
# 1765
cudaDevAttrTextureAlignment, 
# 1766
cudaDevAttrGpuOverlap, 
# 1767
cudaDevAttrMultiProcessorCount, 
# 1768
cudaDevAttrKernelExecTimeout, 
# 1769
cudaDevAttrIntegrated, 
# 1770
cudaDevAttrCanMapHostMemory, 
# 1771
cudaDevAttrComputeMode, 
# 1772
cudaDevAttrMaxTexture1DWidth, 
# 1773
cudaDevAttrMaxTexture2DWidth, 
# 1774
cudaDevAttrMaxTexture2DHeight, 
# 1775
cudaDevAttrMaxTexture3DWidth, 
# 1776
cudaDevAttrMaxTexture3DHeight, 
# 1777
cudaDevAttrMaxTexture3DDepth, 
# 1778
cudaDevAttrMaxTexture2DLayeredWidth, 
# 1779
cudaDevAttrMaxTexture2DLayeredHeight, 
# 1780
cudaDevAttrMaxTexture2DLayeredLayers, 
# 1781
cudaDevAttrSurfaceAlignment, 
# 1782
cudaDevAttrConcurrentKernels, 
# 1783
cudaDevAttrEccEnabled, 
# 1784
cudaDevAttrPciBusId, 
# 1785
cudaDevAttrPciDeviceId, 
# 1786
cudaDevAttrTccDriver, 
# 1787
cudaDevAttrMemoryClockRate, 
# 1788
cudaDevAttrGlobalMemoryBusWidth, 
# 1789
cudaDevAttrL2CacheSize, 
# 1790
cudaDevAttrMaxThreadsPerMultiProcessor, 
# 1791
cudaDevAttrAsyncEngineCount, 
# 1792
cudaDevAttrUnifiedAddressing, 
# 1793
cudaDevAttrMaxTexture1DLayeredWidth, 
# 1794
cudaDevAttrMaxTexture1DLayeredLayers, 
# 1795
cudaDevAttrMaxTexture2DGatherWidth = 45, 
# 1796
cudaDevAttrMaxTexture2DGatherHeight, 
# 1797
cudaDevAttrMaxTexture3DWidthAlt, 
# 1798
cudaDevAttrMaxTexture3DHeightAlt, 
# 1799
cudaDevAttrMaxTexture3DDepthAlt, 
# 1800
cudaDevAttrPciDomainId, 
# 1801
cudaDevAttrTexturePitchAlignment, 
# 1802
cudaDevAttrMaxTextureCubemapWidth, 
# 1803
cudaDevAttrMaxTextureCubemapLayeredWidth, 
# 1804
cudaDevAttrMaxTextureCubemapLayeredLayers, 
# 1805
cudaDevAttrMaxSurface1DWidth, 
# 1806
cudaDevAttrMaxSurface2DWidth, 
# 1807
cudaDevAttrMaxSurface2DHeight, 
# 1808
cudaDevAttrMaxSurface3DWidth, 
# 1809
cudaDevAttrMaxSurface3DHeight, 
# 1810
cudaDevAttrMaxSurface3DDepth, 
# 1811
cudaDevAttrMaxSurface1DLayeredWidth, 
# 1812
cudaDevAttrMaxSurface1DLayeredLayers, 
# 1813
cudaDevAttrMaxSurface2DLayeredWidth, 
# 1814
cudaDevAttrMaxSurface2DLayeredHeight, 
# 1815
cudaDevAttrMaxSurface2DLayeredLayers, 
# 1816
cudaDevAttrMaxSurfaceCubemapWidth, 
# 1817
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
# 1818
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
# 1819
cudaDevAttrMaxTexture1DLinearWidth, 
# 1820
cudaDevAttrMaxTexture2DLinearWidth, 
# 1821
cudaDevAttrMaxTexture2DLinearHeight, 
# 1822
cudaDevAttrMaxTexture2DLinearPitch, 
# 1823
cudaDevAttrMaxTexture2DMipmappedWidth, 
# 1824
cudaDevAttrMaxTexture2DMipmappedHeight, 
# 1825
cudaDevAttrComputeCapabilityMajor, 
# 1826
cudaDevAttrComputeCapabilityMinor, 
# 1827
cudaDevAttrMaxTexture1DMipmappedWidth, 
# 1828
cudaDevAttrStreamPrioritiesSupported, 
# 1829
cudaDevAttrGlobalL1CacheSupported, 
# 1830
cudaDevAttrLocalL1CacheSupported, 
# 1831
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
# 1832
cudaDevAttrMaxRegistersPerMultiprocessor, 
# 1833
cudaDevAttrManagedMemory, 
# 1834
cudaDevAttrIsMultiGpuBoard, 
# 1835
cudaDevAttrMultiGpuBoardGroupID, 
# 1836
cudaDevAttrHostNativeAtomicSupported, 
# 1837
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
# 1838
cudaDevAttrPageableMemoryAccess, 
# 1839
cudaDevAttrConcurrentManagedAccess, 
# 1840
cudaDevAttrComputePreemptionSupported, 
# 1841
cudaDevAttrCanUseHostPointerForRegisteredMem, 
# 1842
cudaDevAttrReserved92, 
# 1843
cudaDevAttrReserved93, 
# 1844
cudaDevAttrReserved94, 
# 1845
cudaDevAttrCooperativeLaunch, 
# 1846
cudaDevAttrCooperativeMultiDeviceLaunch, 
# 1847
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
# 1848
cudaDevAttrCanFlushRemoteWrites, 
# 1849
cudaDevAttrHostRegisterSupported, 
# 1850
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
# 1851
cudaDevAttrDirectManagedMemAccessFromHost, 
# 1852
cudaDevAttrMaxBlocksPerMultiprocessor = 106, 
# 1853
cudaDevAttrMaxPersistingL2CacheSize = 108, 
# 1854
cudaDevAttrMaxAccessPolicyWindowSize, 
# 1855
cudaDevAttrReservedSharedMemoryPerBlock = 111, 
# 1856
cudaDevAttrSparseCudaArraySupported, 
# 1857
cudaDevAttrHostRegisterReadOnlySupported, 
# 1858
cudaDevAttrMaxTimelineSemaphoreInteropSupported, 
# 1859
cudaDevAttrMemoryPoolsSupported, 
# 1860
cudaDevAttrGPUDirectRDMASupported, 
# 1861
cudaDevAttrGPUDirectRDMAFlushWritesOptions, 
# 1862
cudaDevAttrGPUDirectRDMAWritesOrdering, 
# 1863
cudaDevAttrMemoryPoolSupportedHandleTypes, 
# 1864
cudaDevAttrMax
# 1865
}; 
#endif
# 1870 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1870
enum cudaMemPoolAttr { 
# 1880 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaMemPoolReuseFollowEventDependencies = 1, 
# 1887
cudaMemPoolReuseAllowOpportunistic, 
# 1895
cudaMemPoolReuseAllowInternalDependencies, 
# 1906 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
cudaMemPoolAttrReleaseThreshold, 
# 1912
cudaMemPoolAttrReservedMemCurrent, 
# 1919
cudaMemPoolAttrReservedMemHigh, 
# 1925
cudaMemPoolAttrUsedMemCurrent, 
# 1932
cudaMemPoolAttrUsedMemHigh
# 1933
}; 
#endif
# 1938 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1938
enum cudaMemLocationType { 
# 1939
cudaMemLocationTypeInvalid, 
# 1940
cudaMemLocationTypeDevice
# 1941
}; 
#endif
# 1948 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1948
struct cudaMemLocation { 
# 1949
cudaMemLocationType type; 
# 1950
int id; 
# 1951
}; 
#endif
# 1956 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1956
enum cudaMemAccessFlags { 
# 1957
cudaMemAccessFlagsProtNone, 
# 1958
cudaMemAccessFlagsProtRead, 
# 1959
cudaMemAccessFlagsProtReadWrite = 3
# 1960
}; 
#endif
# 1965 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1965
struct cudaMemAccessDesc { 
# 1966
cudaMemLocation location; 
# 1967
cudaMemAccessFlags flags; 
# 1968
}; 
#endif
# 1973 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1973
enum cudaMemAllocationType { 
# 1974
cudaMemAllocationTypeInvalid, 
# 1978
cudaMemAllocationTypePinned, 
# 1979
cudaMemAllocationTypeMax = 2147483647
# 1980
}; 
#endif
# 1985 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1985
enum cudaMemAllocationHandleType { 
# 1986
cudaMemHandleTypeNone, 
# 1987
cudaMemHandleTypePosixFileDescriptor, 
# 1988
cudaMemHandleTypeWin32, 
# 1989
cudaMemHandleTypeWin32Kmt = 4
# 1990
}; 
#endif
# 1995 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1995
struct cudaMemPoolProps { 
# 1996
cudaMemAllocationType allocType; 
# 1997
cudaMemAllocationHandleType handleTypes; 
# 1998
cudaMemLocation location; 
# 2005
void *win32SecurityAttributes; 
# 2006
unsigned char reserved[64]; 
# 2007
}; 
#endif
# 2012 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2012
struct cudaMemPoolPtrExportData { 
# 2013
unsigned char reserved[64]; 
# 2014
}; 
#endif
# 2019 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2019
struct cudaMemAllocNodeParams { 
# 2024
cudaMemPoolProps poolProps; 
# 2025
const cudaMemAccessDesc *accessDescs; 
# 2026
size_t accessDescCount; 
# 2027
size_t bytesize; 
# 2028
void *dptr; 
# 2029
}; 
#endif
# 2034 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2034
enum cudaGraphMemAttributeType { 
# 2039
cudaGraphMemAttrUsedMemCurrent = 1, 
# 2046
cudaGraphMemAttrUsedMemHigh, 
# 2053
cudaGraphMemAttrReservedMemCurrent, 
# 2060
cudaGraphMemAttrReservedMemHigh
# 2061
}; 
#endif
# 2067 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2067
enum cudaDeviceP2PAttr { 
# 2068
cudaDevP2PAttrPerformanceRank = 1, 
# 2069
cudaDevP2PAttrAccessSupported, 
# 2070
cudaDevP2PAttrNativeAtomicSupported, 
# 2071
cudaDevP2PAttrCudaArrayAccessSupported
# 2072
}; 
#endif
# 2079 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2079
struct CUuuid_st { 
# 2080
char bytes[16]; 
# 2081
}; 
#endif
# 2082 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef CUuuid_st 
# 2082
CUuuid; 
#endif
# 2084 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef CUuuid_st 
# 2084
cudaUUID_t; 
#endif
# 2089 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2089
struct cudaDeviceProp { 
# 2091
char name[256]; 
# 2092
cudaUUID_t uuid; 
# 2093
char luid[8]; 
# 2094
unsigned luidDeviceNodeMask; 
# 2095
size_t totalGlobalMem; 
# 2096
size_t sharedMemPerBlock; 
# 2097
int regsPerBlock; 
# 2098
int warpSize; 
# 2099
size_t memPitch; 
# 2100
int maxThreadsPerBlock; 
# 2101
int maxThreadsDim[3]; 
# 2102
int maxGridSize[3]; 
# 2103
int clockRate; 
# 2104
size_t totalConstMem; 
# 2105
int major; 
# 2106
int minor; 
# 2107
size_t textureAlignment; 
# 2108
size_t texturePitchAlignment; 
# 2109
int deviceOverlap; 
# 2110
int multiProcessorCount; 
# 2111
int kernelExecTimeoutEnabled; 
# 2112
int integrated; 
# 2113
int canMapHostMemory; 
# 2114
int computeMode; 
# 2115
int maxTexture1D; 
# 2116
int maxTexture1DMipmap; 
# 2117
int maxTexture1DLinear; 
# 2118
int maxTexture2D[2]; 
# 2119
int maxTexture2DMipmap[2]; 
# 2120
int maxTexture2DLinear[3]; 
# 2121
int maxTexture2DGather[2]; 
# 2122
int maxTexture3D[3]; 
# 2123
int maxTexture3DAlt[3]; 
# 2124
int maxTextureCubemap; 
# 2125
int maxTexture1DLayered[2]; 
# 2126
int maxTexture2DLayered[3]; 
# 2127
int maxTextureCubemapLayered[2]; 
# 2128
int maxSurface1D; 
# 2129
int maxSurface2D[2]; 
# 2130
int maxSurface3D[3]; 
# 2131
int maxSurface1DLayered[2]; 
# 2132
int maxSurface2DLayered[3]; 
# 2133
int maxSurfaceCubemap; 
# 2134
int maxSurfaceCubemapLayered[2]; 
# 2135
size_t surfaceAlignment; 
# 2136
int concurrentKernels; 
# 2137
int ECCEnabled; 
# 2138
int pciBusID; 
# 2139
int pciDeviceID; 
# 2140
int pciDomainID; 
# 2141
int tccDriver; 
# 2142
int asyncEngineCount; 
# 2143
int unifiedAddressing; 
# 2144
int memoryClockRate; 
# 2145
int memoryBusWidth; 
# 2146
int l2CacheSize; 
# 2147
int persistingL2CacheMaxSize; 
# 2148
int maxThreadsPerMultiProcessor; 
# 2149
int streamPrioritiesSupported; 
# 2150
int globalL1CacheSupported; 
# 2151
int localL1CacheSupported; 
# 2152
size_t sharedMemPerMultiprocessor; 
# 2153
int regsPerMultiprocessor; 
# 2154
int managedMemory; 
# 2155
int isMultiGpuBoard; 
# 2156
int multiGpuBoardGroupID; 
# 2157
int hostNativeAtomicSupported; 
# 2158
int singleToDoublePrecisionPerfRatio; 
# 2159
int pageableMemoryAccess; 
# 2160
int concurrentManagedAccess; 
# 2161
int computePreemptionSupported; 
# 2162
int canUseHostPointerForRegisteredMem; 
# 2163
int cooperativeLaunch; 
# 2164
int cooperativeMultiDeviceLaunch; 
# 2165
size_t sharedMemPerBlockOptin; 
# 2166
int pageableMemoryAccessUsesHostPageTables; 
# 2167
int directManagedMemAccessFromHost; 
# 2168
int maxBlocksPerMultiProcessor; 
# 2169
int accessPolicyMaxWindowSize; 
# 2170
size_t reservedSharedMemPerBlock; 
# 2171
}; 
#endif
# 2267 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 2264
struct cudaIpcEventHandle_st { 
# 2266
char reserved[64]; 
# 2267
} cudaIpcEventHandle_t; 
#endif
# 2275 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 2272
struct cudaIpcMemHandle_st { 
# 2274
char reserved[64]; 
# 2275
} cudaIpcMemHandle_t; 
#endif
# 2280 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2280
enum cudaExternalMemoryHandleType { 
# 2284
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
# 2288
cudaExternalMemoryHandleTypeOpaqueWin32, 
# 2292
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
# 2296
cudaExternalMemoryHandleTypeD3D12Heap, 
# 2300
cudaExternalMemoryHandleTypeD3D12Resource, 
# 2304
cudaExternalMemoryHandleTypeD3D11Resource, 
# 2308
cudaExternalMemoryHandleTypeD3D11ResourceKmt, 
# 2312
cudaExternalMemoryHandleTypeNvSciBuf
# 2313
}; 
#endif
# 2355 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2355
struct cudaExternalMemoryHandleDesc { 
# 2359
cudaExternalMemoryHandleType type; 
# 2360
union { 
# 2366
int fd; 
# 2382 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
struct { 
# 2386
void *handle; 
# 2391
const void *name; 
# 2392
} win32; 
# 2397
const void *nvSciBufObject; 
# 2398
} handle; 
# 2402
unsigned long long size; 
# 2406
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2407
}; 
#endif
# 2412 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2412
struct cudaExternalMemoryBufferDesc { 
# 2416
unsigned long long offset; 
# 2420
unsigned long long size; 
# 2424
unsigned flags; 
# 2425
}; 
#endif
# 2430 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2430
struct cudaExternalMemoryMipmappedArrayDesc { 
# 2435
unsigned long long offset; 
# 2439
cudaChannelFormatDesc formatDesc; 
# 2443
cudaExtent extent; 
# 2448
unsigned flags; 
# 2452
unsigned numLevels; 
# 2453
}; 
#endif
# 2458 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2458
enum cudaExternalSemaphoreHandleType { 
# 2462
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
# 2466
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
# 2470
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
# 2474
cudaExternalSemaphoreHandleTypeD3D12Fence, 
# 2478
cudaExternalSemaphoreHandleTypeD3D11Fence, 
# 2482
cudaExternalSemaphoreHandleTypeNvSciSync, 
# 2486
cudaExternalSemaphoreHandleTypeKeyedMutex, 
# 2490
cudaExternalSemaphoreHandleTypeKeyedMutexKmt, 
# 2494
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd, 
# 2498
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
# 2499
}; 
#endif
# 2504 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2504
struct cudaExternalSemaphoreHandleDesc { 
# 2508
cudaExternalSemaphoreHandleType type; 
# 2509
union { 
# 2516
int fd; 
# 2532 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
struct { 
# 2536
void *handle; 
# 2541
const void *name; 
# 2542
} win32; 
# 2546
const void *nvSciSyncObj; 
# 2547
} handle; 
# 2551
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2552
}; 
#endif
# 2653 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2653
struct cudaExternalSemaphoreSignalParams { 
# 2654
struct { 
# 2658
struct { 
# 2662
unsigned long long value; 
# 2663
} fence; 
# 2664
union { 
# 2669
void *fence; 
# 2670
unsigned long long reserved; 
# 2671
} nvSciSync; 
# 2675
struct { 
# 2679
unsigned long long key; 
# 2680
} keyedMutex; 
# 2681
unsigned reserved[12]; 
# 2682
} params; 
# 2693 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
unsigned flags; 
# 2694
unsigned reserved[16]; 
# 2695
}; 
#endif
# 2700 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2700
struct cudaExternalSemaphoreWaitParams { 
# 2701
struct { 
# 2705
struct { 
# 2709
unsigned long long value; 
# 2710
} fence; 
# 2711
union { 
# 2716
void *fence; 
# 2717
unsigned long long reserved; 
# 2718
} nvSciSync; 
# 2722
struct { 
# 2726
unsigned long long key; 
# 2730
unsigned timeoutMs; 
# 2731
} keyedMutex; 
# 2732
unsigned reserved[10]; 
# 2733
} params; 
# 2744 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
unsigned flags; 
# 2745
unsigned reserved[16]; 
# 2746
}; 
#endif
# 2758 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaError 
# 2758
cudaError_t; 
#endif
# 2763 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUstream_st *
# 2763
cudaStream_t; 
#endif
# 2768 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUevent_st *
# 2768
cudaEvent_t; 
#endif
# 2773 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaGraphicsResource *
# 2773
cudaGraphicsResource_t; 
#endif
# 2778 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaOutputMode 
# 2778
cudaOutputMode_t; 
#endif
# 2783 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUexternalMemory_st *
# 2783
cudaExternalMemory_t; 
#endif
# 2788 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUexternalSemaphore_st *
# 2788
cudaExternalSemaphore_t; 
#endif
# 2793 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUgraph_st *
# 2793
cudaGraph_t; 
#endif
# 2798 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUgraphNode_st *
# 2798
cudaGraphNode_t; 
#endif
# 2803 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUuserObject_st *
# 2803
cudaUserObject_t; 
#endif
# 2808 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUfunc_st *
# 2808
cudaFunction_t; 
#endif
# 2813 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUmemPoolHandle_st *
# 2813
cudaMemPool_t; 
#endif
# 2818 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2818
enum cudaCGScope { 
# 2819
cudaCGScopeInvalid, 
# 2820
cudaCGScopeGrid, 
# 2821
cudaCGScopeMultiGrid
# 2822
}; 
#endif
# 2827 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2827
struct cudaLaunchParams { 
# 2829
void *func; 
# 2830
dim3 gridDim; 
# 2831
dim3 blockDim; 
# 2832
void **args; 
# 2833
size_t sharedMem; 
# 2834
cudaStream_t stream; 
# 2835
}; 
#endif
# 2840 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2840
struct cudaKernelNodeParams { 
# 2841
void *func; 
# 2842
dim3 gridDim; 
# 2843
dim3 blockDim; 
# 2844
unsigned sharedMemBytes; 
# 2845
void **kernelParams; 
# 2846
void **extra; 
# 2847
}; 
#endif
# 2852 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2852
struct cudaExternalSemaphoreSignalNodeParams { 
# 2853
cudaExternalSemaphore_t *extSemArray; 
# 2854
const cudaExternalSemaphoreSignalParams *paramsArray; 
# 2855
unsigned numExtSems; 
# 2856
}; 
#endif
# 2861 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2861
struct cudaExternalSemaphoreWaitNodeParams { 
# 2862
cudaExternalSemaphore_t *extSemArray; 
# 2863
const cudaExternalSemaphoreWaitParams *paramsArray; 
# 2864
unsigned numExtSems; 
# 2865
}; 
#endif
# 2870 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2870
enum cudaGraphNodeType { 
# 2871
cudaGraphNodeTypeKernel, 
# 2872
cudaGraphNodeTypeMemcpy, 
# 2873
cudaGraphNodeTypeMemset, 
# 2874
cudaGraphNodeTypeHost, 
# 2875
cudaGraphNodeTypeGraph, 
# 2876
cudaGraphNodeTypeEmpty, 
# 2877
cudaGraphNodeTypeWaitEvent, 
# 2878
cudaGraphNodeTypeEventRecord, 
# 2879
cudaGraphNodeTypeExtSemaphoreSignal, 
# 2880
cudaGraphNodeTypeExtSemaphoreWait, 
# 2881
cudaGraphNodeTypeMemAlloc, 
# 2882
cudaGraphNodeTypeMemFree, 
# 2883
cudaGraphNodeTypeCount
# 2884
}; 
#endif
# 2889 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
typedef struct CUgraphExec_st *cudaGraphExec_t; 
# 2894
#if 0
# 2894
enum cudaGraphExecUpdateResult { 
# 2895
cudaGraphExecUpdateSuccess, 
# 2896
cudaGraphExecUpdateError, 
# 2897
cudaGraphExecUpdateErrorTopologyChanged, 
# 2898
cudaGraphExecUpdateErrorNodeTypeChanged, 
# 2899
cudaGraphExecUpdateErrorFunctionChanged, 
# 2900
cudaGraphExecUpdateErrorParametersChanged, 
# 2901
cudaGraphExecUpdateErrorNotSupported, 
# 2902
cudaGraphExecUpdateErrorUnsupportedFunctionChange
# 2903
}; 
#endif
# 2909 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2909
enum cudaGetDriverEntryPointFlags { 
# 2910
cudaEnableDefault, 
# 2911
cudaEnableLegacyStream, 
# 2912
cudaEnablePerThreadDefaultStream
# 2913
}; 
#endif
# 2918 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2918
enum cudaGraphDebugDotFlags { 
# 2919
cudaGraphDebugDotFlagsVerbose = (1 << 0), 
# 2920
cudaGraphDebugDotFlagsKernelNodeParams = (1 << 2), 
# 2921
cudaGraphDebugDotFlagsMemcpyNodeParams = (1 << 3), 
# 2922
cudaGraphDebugDotFlagsMemsetNodeParams = (1 << 4), 
# 2923
cudaGraphDebugDotFlagsHostNodeParams = (1 << 5), 
# 2924
cudaGraphDebugDotFlagsEventNodeParams = (1 << 6), 
# 2925
cudaGraphDebugDotFlagsExtSemasSignalNodeParams = (1 << 7), 
# 2926
cudaGraphDebugDotFlagsExtSemasWaitNodeParams = (1 << 8), 
# 2927
cudaGraphDebugDotFlagsKernelNodeAttributes = (1 << 9), 
# 2928
cudaGraphDebugDotFlagsHandles = (1 << 10)
# 2929
}; 
#endif
# 2934 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2934
enum cudaGraphInstantiateFlags { 
# 2935
cudaGraphInstantiateFlagAutoFreeOnLaunch = 1
# 2936
}; 
#endif
# 84 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 84
enum cudaSurfaceBoundaryMode { 
# 86
cudaBoundaryModeZero, 
# 87
cudaBoundaryModeClamp, 
# 88
cudaBoundaryModeTrap
# 89
}; 
#endif
# 94 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 94
enum cudaSurfaceFormatMode { 
# 96
cudaFormatModeForced, 
# 97
cudaFormatModeAuto
# 98
}; 
#endif
# 103 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 103
struct surfaceReference { 
# 108
cudaChannelFormatDesc channelDesc; 
# 109
}; 
#endif
# 114 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
typedef unsigned long long 
# 114
cudaSurfaceObject_t; 
#endif
# 84 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 84
enum cudaTextureAddressMode { 
# 86
cudaAddressModeWrap, 
# 87
cudaAddressModeClamp, 
# 88
cudaAddressModeMirror, 
# 89
cudaAddressModeBorder
# 90
}; 
#endif
# 95 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 95
enum cudaTextureFilterMode { 
# 97
cudaFilterModePoint, 
# 98
cudaFilterModeLinear
# 99
}; 
#endif
# 104 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 104
enum cudaTextureReadMode { 
# 106
cudaReadModeElementType, 
# 107
cudaReadModeNormalizedFloat
# 108
}; 
#endif
# 113 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 113
struct textureReference { 
# 118
int normalized; 
# 122
cudaTextureFilterMode filterMode; 
# 126
cudaTextureAddressMode addressMode[3]; 
# 130
cudaChannelFormatDesc channelDesc; 
# 134
int sRGB; 
# 138
unsigned maxAnisotropy; 
# 142
cudaTextureFilterMode mipmapFilterMode; 
# 146
float mipmapLevelBias; 
# 150
float minMipmapLevelClamp; 
# 154
float maxMipmapLevelClamp; 
# 158
int disableTrilinearOptimization; 
# 159
int __cudaReserved[14]; 
# 160
}; 
#endif
# 165 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 165
struct cudaTextureDesc { 
# 170
cudaTextureAddressMode addressMode[3]; 
# 174
cudaTextureFilterMode filterMode; 
# 178
cudaTextureReadMode readMode; 
# 182
int sRGB; 
# 186
float borderColor[4]; 
# 190
int normalizedCoords; 
# 194
unsigned maxAnisotropy; 
# 198
cudaTextureFilterMode mipmapFilterMode; 
# 202
float mipmapLevelBias; 
# 206
float minMipmapLevelClamp; 
# 210
float maxMipmapLevelClamp; 
# 214
int disableTrilinearOptimization; 
# 215
}; 
#endif
# 220 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
typedef unsigned long long 
# 220
cudaTextureObject_t; 
#endif
# 84 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/library_types.h"
typedef 
# 54
enum cudaDataType_t { 
# 56
CUDA_R_16F = 2, 
# 57
CUDA_C_16F = 6, 
# 58
CUDA_R_16BF = 14, 
# 59
CUDA_C_16BF, 
# 60
CUDA_R_32F = 0, 
# 61
CUDA_C_32F = 4, 
# 62
CUDA_R_64F = 1, 
# 63
CUDA_C_64F = 5, 
# 64
CUDA_R_4I = 16, 
# 65
CUDA_C_4I, 
# 66
CUDA_R_4U, 
# 67
CUDA_C_4U, 
# 68
CUDA_R_8I = 3, 
# 69
CUDA_C_8I = 7, 
# 70
CUDA_R_8U, 
# 71
CUDA_C_8U, 
# 72
CUDA_R_16I = 20, 
# 73
CUDA_C_16I, 
# 74
CUDA_R_16U, 
# 75
CUDA_C_16U, 
# 76
CUDA_R_32I = 10, 
# 77
CUDA_C_32I, 
# 78
CUDA_R_32U, 
# 79
CUDA_C_32U, 
# 80
CUDA_R_64I = 24, 
# 81
CUDA_C_64I, 
# 82
CUDA_R_64U, 
# 83
CUDA_C_64U
# 84
} cudaDataType; 
# 92
typedef 
# 87
enum libraryPropertyType_t { 
# 89
MAJOR_VERSION, 
# 90
MINOR_VERSION, 
# 91
PATCH_LEVEL
# 92
} libraryPropertyType; 
# 115 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
extern "C" {
# 117
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 118
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 119
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 120
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 121
extern cudaError_t cudaDeviceSynchronize(); 
# 122
extern cudaError_t cudaGetLastError(); 
# 123
extern cudaError_t cudaPeekAtLastError(); 
# 124
extern const char *cudaGetErrorString(cudaError_t error); 
# 125
extern const char *cudaGetErrorName(cudaError_t error); 
# 126
extern cudaError_t cudaGetDeviceCount(int * count); 
# 127
extern cudaError_t cudaGetDevice(int * device); 
# 128
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 129
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 130
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 131
__attribute__((unused)) extern cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 132
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 133
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
# 134
__attribute__((unused)) extern cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
# 135
extern cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
# 136
__attribute__((unused)) extern cudaError_t cudaEventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
# 137
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 138
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 139
extern cudaError_t cudaFree(void * devPtr); 
# 140
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 141
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 142
__attribute__((unused)) extern cudaError_t cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 143
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 144
__attribute__((unused)) extern cudaError_t cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 145
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 146
__attribute__((unused)) extern cudaError_t cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 147
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 148
__attribute__((unused)) extern cudaError_t cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 149
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 150
__attribute__((unused)) extern cudaError_t cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 151
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 152
__attribute__((unused)) extern cudaError_t cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 153
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 174 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void *cudaGetParameterBuffer(size_t alignment, size_t size); 
# 202 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void *cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
# 203
__attribute__((unused)) extern cudaError_t cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 204
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
# 222 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern cudaError_t cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 223
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
# 226
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
# 227
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 229
__attribute__((unused)) extern unsigned long long cudaCGGetIntrinsicHandle(cudaCGScope scope); 
# 230
__attribute__((unused)) extern cudaError_t cudaCGSynchronize(unsigned long long handle, unsigned flags); 
# 231
__attribute__((unused)) extern cudaError_t cudaCGSynchronizeGrid(unsigned long long handle, unsigned flags); 
# 232
__attribute__((unused)) extern cudaError_t cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned long long handle); 
# 233
__attribute__((unused)) extern cudaError_t cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned long long handle); 
# 234
}
# 236
template< class T> static inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
# 237
template< class T> static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
# 238
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
# 239
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 266 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern "C" {
# 301 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceReset(); 
# 322 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSynchronize(); 
# 409 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value); 
# 444 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 467 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const cudaChannelFormatDesc * fmtDesc, int device); 
# 501 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 538 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
# 582 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
# 613 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 657 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
# 684 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
# 714 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
# 762 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
# 803 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
# 846 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
# 910 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
# 946 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcCloseMemHandle(void * devPtr); 
# 978 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope); 
# 1022 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadExit(); 
# 1048 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSynchronize(); 
# 1097 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value); 
# 1130 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
# 1166 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 1213 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
# 1276 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetLastError(); 
# 1324 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaPeekAtLastError(); 
# 1340 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern const char *cudaGetErrorName(cudaError_t error); 
# 1356 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern const char *cudaGetErrorString(cudaError_t error); 
# 1384 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceCount(int * count); 
# 1662 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
# 1857 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 1875 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int device); 
# 1899 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool); 
# 1919 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetMemPool(cudaMemPool_t * memPool, int device); 
# 1967 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int device, int flags); 
# 2007 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
# 2028 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
# 2065 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetDevice(int device); 
# 2086 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDevice(int * device); 
# 2117 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetValidDevices(int * device_arr, int len); 
# 2182 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetDeviceFlags(unsigned flags); 
# 2226 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceFlags(unsigned * flags); 
# 2266 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreate(cudaStream_t * pStream); 
# 2298 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 2344 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
# 2371 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
# 2396 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
# 2411 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaCtxResetPersistingL2Cache(); 
# 2431 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src); 
# 2452 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue * value_out); 
# 2476 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue * value); 
# 2510 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 2541 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags = 0); 
# 2549
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
# 2616 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
# 2640 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream); 
# 2665 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamQuery(cudaStream_t stream); 
# 2749 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
# 2788 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode); 
# 2839 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode); 
# 2867 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
# 2905 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
# 2937 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus, unsigned long long * pId); 
# 2992 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out = 0, cudaGraph_t * graph_out = 0, const cudaGraphNode_t ** dependencies_out = 0, size_t * numDependencies_out = 0); 
# 3025 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t * dependencies, size_t numDependencies, unsigned flags = 0); 
# 3062 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventCreate(cudaEvent_t * event); 
# 3099 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 3139 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
# 3186 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned flags = 0); 
# 3218 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventQuery(cudaEvent_t event); 
# 3248 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventSynchronize(cudaEvent_t event); 
# 3277 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 3320 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
# 3500 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
# 3555 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
# 3615 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
# 3639 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
# 3792 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
# 3859 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 3935 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 3958 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
# 4025 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 4082 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 4183 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
# 4230 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
# 4285 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
# 4318 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 4355 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
# 4381 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForDevice(double * d); 
# 4405 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForHost(double * d); 
# 4473 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
# 4530 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
# 4559 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int numBlocks, int blockSize); 
# 4604 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
# 4725 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
# 4758 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 4791 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocHost(void ** ptr, size_t size); 
# 4834 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
# 4883 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
# 4912 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFree(void * devPtr); 
# 4935 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeHost(void * ptr); 
# 4958 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeArray(cudaArray_t array); 
# 4981 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
# 5047 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
# 5140 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned flags); 
# 5163 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostUnregister(void * ptr); 
# 5208 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
# 5230 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostGetFlags(unsigned * pFlags, void * pHost); 
# 5269 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
# 5411 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
# 5553 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
# 5586 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
# 5691 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms * p); 
# 5722 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
# 5840 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
# 5866 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
# 5888 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemGetInfo(size_t * free, size_t * total); 
# 5914 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
# 5943 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t hArray, unsigned planeIdx); 
# 5971 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaArray_t array); 
# 6001 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t mipmap); 
# 6046 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 6081 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
# 6130 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 6180 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 6230 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
# 6277 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 6320 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
# 6363 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
# 6420 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6455 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
# 6518 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6576 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6633 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6684 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6735 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6764 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset(void * devPtr, int value, size_t count); 
# 6798 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
# 6844 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
# 6880 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
# 6921 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
# 6974 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
# 7002 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
# 7029 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol); 
# 7099 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
# 7215 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
# 7274 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
# 7313 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
# 7373 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
# 7415 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
# 7458 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 7509 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7559 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 7628 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocAsync(void ** devPtr, size_t size, cudaStream_t hStream); 
# 7654 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeAsync(void * devPtr, cudaStream_t hStream); 
# 7679 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep); 
# 7717 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
# 7755 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
# 7770 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc * descList, size_t count); 
# 7783 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags * flags, cudaMemPool_t memPool, cudaMemLocation * location); 
# 7803 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolCreate(cudaMemPool_t * memPool, const cudaMemPoolProps * poolProps); 
# 7825 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool); 
# 7861 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocFromPoolAsync(void ** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream); 
# 7886 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned flags); 
# 7913 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, cudaMemAllocationHandleType handleType, unsigned flags); 
# 7936 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData * exportData, void * ptr); 
# 7965 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData * exportData); 
# 8117 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
# 8158 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
# 8200 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
# 8222 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice); 
# 8286 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
# 8321 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
# 8360 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 8395 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 8427 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
# 8465 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
# 8494 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
# 8565 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = ((2147483647) * 2U) + 1U); 
# 8624 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
# 8662 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 8702 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
# 8728 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaUnbindTexture(const textureReference * texref); 
# 8757 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
# 8787 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
# 8832 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 8857 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
# 8892 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
# 8922 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
# 9140 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
# 9160 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject); 
# 9180 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
# 9200 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
# 9221 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
# 9266 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
# 9286 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
# 9305 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
# 9339 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDriverGetVersion(int * driverVersion); 
# 9364 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 9411 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
# 9508 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
# 9541 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
# 9566 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 9586 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst); 
# 9609 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue * value_out); 
# 9633 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue * value); 
# 9683 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
# 9742 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 9811 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 9879 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 9911 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
# 9937 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 9976 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10022 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 10068 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 10115 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
# 10138 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
# 10161 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 10202 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
# 10225 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
# 10248 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 10288 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
# 10315 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
# 10352 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies); 
# 10395 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
# 10422 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
# 10449 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
# 10495 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
# 10522 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
# 10549 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
# 10598 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 10631 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams * params_out); 
# 10658 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 10707 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 10740 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams * params_out); 
# 10767 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 10844 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaMemAllocNodeParams * nodeParams); 
# 10871 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams * params_out); 
# 10931 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dptr); 
# 10955 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void * dptr_out); 
# 10983 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGraphMemTrim(int device); 
# 11020 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
# 11054 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
# 11082 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
# 11110 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
# 11141 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
# 11172 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
# 11203 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
# 11237 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges); 
# 11268 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies); 
# 11300 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes); 
# 11331 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 11362 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 11392 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node); 
# 11430 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphNode_t * pErrorNode, char * pLogBuffer, size_t bufferSize); 
# 11473 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned long long flags); 
# 11517 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 11567 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 11622 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
# 11685 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
# 11746 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 11800 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 11839 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 11885 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph); 
# 11929 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
# 11973 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
# 12020 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
# 12067 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
# 12142 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t * hErrorNode_out, cudaGraphExecUpdateResult * updateResult_out); 
# 12167 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 12198 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 12221 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
# 12242 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphDestroy(cudaGraph_t graph); 
# 12261 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char * path, unsigned flags); 
# 12297 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t destroy, unsigned initialRefcount, unsigned flags); 
# 12321 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned count = 1); 
# 12349 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned count = 1); 
# 12377 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1, unsigned flags = 0); 
# 12402 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1); 
# 12468 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned long long flags); 
# 12473
extern cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
# 12649 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetFuncBySymbol(cudaFunction_t * functionPtr, const void * symbolPtr); 
# 12791 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
}
# 104 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/channel_descriptor.h"
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 105
{ 
# 106
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 107
} 
# 109
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
# 110
{ 
# 111
int e = (((int)sizeof(unsigned short)) * 8); 
# 113
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 114
} 
# 116
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
# 117
{ 
# 118
int e = (((int)sizeof(unsigned short)) * 8); 
# 120
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 121
} 
# 123
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
# 124
{ 
# 125
int e = (((int)sizeof(unsigned short)) * 8); 
# 127
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 128
} 
# 130
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
# 131
{ 
# 132
int e = (((int)sizeof(unsigned short)) * 8); 
# 134
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 135
} 
# 137
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
# 138
{ 
# 139
int e = (((int)sizeof(char)) * 8); 
# 144
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 146
} 
# 148
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
# 149
{ 
# 150
int e = (((int)sizeof(signed char)) * 8); 
# 152
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 153
} 
# 155
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
# 156
{ 
# 157
int e = (((int)sizeof(unsigned char)) * 8); 
# 159
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 160
} 
# 162
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
# 163
{ 
# 164
int e = (((int)sizeof(signed char)) * 8); 
# 166
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 167
} 
# 169
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
# 170
{ 
# 171
int e = (((int)sizeof(unsigned char)) * 8); 
# 173
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 174
} 
# 176
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
# 177
{ 
# 178
int e = (((int)sizeof(signed char)) * 8); 
# 180
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 181
} 
# 183
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
# 184
{ 
# 185
int e = (((int)sizeof(unsigned char)) * 8); 
# 187
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 188
} 
# 190
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
# 191
{ 
# 192
int e = (((int)sizeof(signed char)) * 8); 
# 194
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 195
} 
# 197
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
# 198
{ 
# 199
int e = (((int)sizeof(unsigned char)) * 8); 
# 201
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 202
} 
# 204
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
# 205
{ 
# 206
int e = (((int)sizeof(short)) * 8); 
# 208
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 209
} 
# 211
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
# 212
{ 
# 213
int e = (((int)sizeof(unsigned short)) * 8); 
# 215
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 216
} 
# 218
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
# 219
{ 
# 220
int e = (((int)sizeof(short)) * 8); 
# 222
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 223
} 
# 225
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
# 226
{ 
# 227
int e = (((int)sizeof(unsigned short)) * 8); 
# 229
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 230
} 
# 232
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
# 233
{ 
# 234
int e = (((int)sizeof(short)) * 8); 
# 236
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 237
} 
# 239
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
# 240
{ 
# 241
int e = (((int)sizeof(unsigned short)) * 8); 
# 243
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 244
} 
# 246
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
# 247
{ 
# 248
int e = (((int)sizeof(short)) * 8); 
# 250
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 251
} 
# 253
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
# 254
{ 
# 255
int e = (((int)sizeof(unsigned short)) * 8); 
# 257
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 258
} 
# 260
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
# 261
{ 
# 262
int e = (((int)sizeof(int)) * 8); 
# 264
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 265
} 
# 267
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
# 268
{ 
# 269
int e = (((int)sizeof(unsigned)) * 8); 
# 271
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 272
} 
# 274
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
# 275
{ 
# 276
int e = (((int)sizeof(int)) * 8); 
# 278
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 279
} 
# 281
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
# 282
{ 
# 283
int e = (((int)sizeof(unsigned)) * 8); 
# 285
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 286
} 
# 288
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
# 289
{ 
# 290
int e = (((int)sizeof(int)) * 8); 
# 292
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 293
} 
# 295
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
# 296
{ 
# 297
int e = (((int)sizeof(unsigned)) * 8); 
# 299
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 300
} 
# 302
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
# 303
{ 
# 304
int e = (((int)sizeof(int)) * 8); 
# 306
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 307
} 
# 309
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
# 310
{ 
# 311
int e = (((int)sizeof(unsigned)) * 8); 
# 313
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 314
} 
# 376 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
# 377
{ 
# 378
int e = (((int)sizeof(float)) * 8); 
# 380
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 381
} 
# 383
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
# 384
{ 
# 385
int e = (((int)sizeof(float)) * 8); 
# 387
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 388
} 
# 390
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
# 391
{ 
# 392
int e = (((int)sizeof(float)) * 8); 
# 394
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 395
} 
# 397
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
# 398
{ 
# 399
int e = (((int)sizeof(float)) * 8); 
# 401
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 402
} 
# 404
static inline cudaChannelFormatDesc cudaCreateChannelDescNV12() 
# 405
{ 
# 406
int e = (((int)sizeof(char)) * 8); 
# 408
return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindNV12); 
# 409
} 
# 79 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
# 80
{ 
# 81
cudaPitchedPtr s; 
# 83
(s.ptr) = d; 
# 84
(s.pitch) = p; 
# 85
(s.xsize) = xsz; 
# 86
(s.ysize) = ysz; 
# 88
return s; 
# 89
} 
# 106 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_functions.h"
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
# 107
{ 
# 108
cudaPos p; 
# 110
(p.x) = x; 
# 111
(p.y) = y; 
# 112
(p.z) = z; 
# 114
return p; 
# 115
} 
# 132 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/driver_functions.h"
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
# 133
{ 
# 134
cudaExtent e; 
# 136
(e.width) = w; 
# 137
(e.height) = h; 
# 138
(e.depth) = d; 
# 140
return e; 
# 141
} 
# 73 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_functions.h"
static inline char1 make_char1(signed char x); 
# 75
static inline uchar1 make_uchar1(unsigned char x); 
# 77
static inline char2 make_char2(signed char x, signed char y); 
# 79
static inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
# 81
static inline char3 make_char3(signed char x, signed char y, signed char z); 
# 83
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
# 85
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
# 87
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
# 89
static inline short1 make_short1(short x); 
# 91
static inline ushort1 make_ushort1(unsigned short x); 
# 93
static inline short2 make_short2(short x, short y); 
# 95
static inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
# 97
static inline short3 make_short3(short x, short y, short z); 
# 99
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
# 101
static inline short4 make_short4(short x, short y, short z, short w); 
# 103
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
# 105
static inline int1 make_int1(int x); 
# 107
static inline uint1 make_uint1(unsigned x); 
# 109
static inline int2 make_int2(int x, int y); 
# 111
static inline uint2 make_uint2(unsigned x, unsigned y); 
# 113
static inline int3 make_int3(int x, int y, int z); 
# 115
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
# 117
static inline int4 make_int4(int x, int y, int z, int w); 
# 119
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
# 121
static inline long1 make_long1(long x); 
# 123
static inline ulong1 make_ulong1(unsigned long x); 
# 125
static inline long2 make_long2(long x, long y); 
# 127
static inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
# 129
static inline long3 make_long3(long x, long y, long z); 
# 131
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
# 133
static inline long4 make_long4(long x, long y, long z, long w); 
# 135
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
# 137
static inline float1 make_float1(float x); 
# 139
static inline float2 make_float2(float x, float y); 
# 141
static inline float3 make_float3(float x, float y, float z); 
# 143
static inline float4 make_float4(float x, float y, float z, float w); 
# 145
static inline longlong1 make_longlong1(long long x); 
# 147
static inline ulonglong1 make_ulonglong1(unsigned long long x); 
# 149
static inline longlong2 make_longlong2(long long x, long long y); 
# 151
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y); 
# 153
static inline longlong3 make_longlong3(long long x, long long y, long long z); 
# 155
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z); 
# 157
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w); 
# 159
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w); 
# 161
static inline double1 make_double1(double x); 
# 163
static inline double2 make_double2(double x, double y); 
# 165
static inline double3 make_double3(double x, double y, double z); 
# 167
static inline double4 make_double4(double x, double y, double z, double w); 
# 73 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/vector_functions.hpp"
static inline char1 make_char1(signed char x) 
# 74
{ 
# 75
char1 t; (t.x) = x; return t; 
# 76
} 
# 78
static inline uchar1 make_uchar1(unsigned char x) 
# 79
{ 
# 80
uchar1 t; (t.x) = x; return t; 
# 81
} 
# 83
static inline char2 make_char2(signed char x, signed char y) 
# 84
{ 
# 85
char2 t; (t.x) = x; (t.y) = y; return t; 
# 86
} 
# 88
static inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
# 89
{ 
# 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
# 91
} 
# 93
static inline char3 make_char3(signed char x, signed char y, signed char z) 
# 94
{ 
# 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 96
} 
# 98
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
# 99
{ 
# 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 101
} 
# 103
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
# 104
{ 
# 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 106
} 
# 108
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
# 109
{ 
# 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 111
} 
# 113
static inline short1 make_short1(short x) 
# 114
{ 
# 115
short1 t; (t.x) = x; return t; 
# 116
} 
# 118
static inline ushort1 make_ushort1(unsigned short x) 
# 119
{ 
# 120
ushort1 t; (t.x) = x; return t; 
# 121
} 
# 123
static inline short2 make_short2(short x, short y) 
# 124
{ 
# 125
short2 t; (t.x) = x; (t.y) = y; return t; 
# 126
} 
# 128
static inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
# 129
{ 
# 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
# 131
} 
# 133
static inline short3 make_short3(short x, short y, short z) 
# 134
{ 
# 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 136
} 
# 138
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
# 139
{ 
# 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 141
} 
# 143
static inline short4 make_short4(short x, short y, short z, short w) 
# 144
{ 
# 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 146
} 
# 148
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
# 149
{ 
# 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 151
} 
# 153
static inline int1 make_int1(int x) 
# 154
{ 
# 155
int1 t; (t.x) = x; return t; 
# 156
} 
# 158
static inline uint1 make_uint1(unsigned x) 
# 159
{ 
# 160
uint1 t; (t.x) = x; return t; 
# 161
} 
# 163
static inline int2 make_int2(int x, int y) 
# 164
{ 
# 165
int2 t; (t.x) = x; (t.y) = y; return t; 
# 166
} 
# 168
static inline uint2 make_uint2(unsigned x, unsigned y) 
# 169
{ 
# 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
# 171
} 
# 173
static inline int3 make_int3(int x, int y, int z) 
# 174
{ 
# 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 176
} 
# 178
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
# 179
{ 
# 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 181
} 
# 183
static inline int4 make_int4(int x, int y, int z, int w) 
# 184
{ 
# 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 186
} 
# 188
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
# 189
{ 
# 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 191
} 
# 193
static inline long1 make_long1(long x) 
# 194
{ 
# 195
long1 t; (t.x) = x; return t; 
# 196
} 
# 198
static inline ulong1 make_ulong1(unsigned long x) 
# 199
{ 
# 200
ulong1 t; (t.x) = x; return t; 
# 201
} 
# 203
static inline long2 make_long2(long x, long y) 
# 204
{ 
# 205
long2 t; (t.x) = x; (t.y) = y; return t; 
# 206
} 
# 208
static inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
# 209
{ 
# 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
# 211
} 
# 213
static inline long3 make_long3(long x, long y, long z) 
# 214
{ 
# 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 216
} 
# 218
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
# 219
{ 
# 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 221
} 
# 223
static inline long4 make_long4(long x, long y, long z, long w) 
# 224
{ 
# 225
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 226
} 
# 228
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
# 229
{ 
# 230
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 231
} 
# 233
static inline float1 make_float1(float x) 
# 234
{ 
# 235
float1 t; (t.x) = x; return t; 
# 236
} 
# 238
static inline float2 make_float2(float x, float y) 
# 239
{ 
# 240
float2 t; (t.x) = x; (t.y) = y; return t; 
# 241
} 
# 243
static inline float3 make_float3(float x, float y, float z) 
# 244
{ 
# 245
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 246
} 
# 248
static inline float4 make_float4(float x, float y, float z, float w) 
# 249
{ 
# 250
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 251
} 
# 253
static inline longlong1 make_longlong1(long long x) 
# 254
{ 
# 255
longlong1 t; (t.x) = x; return t; 
# 256
} 
# 258
static inline ulonglong1 make_ulonglong1(unsigned long long x) 
# 259
{ 
# 260
ulonglong1 t; (t.x) = x; return t; 
# 261
} 
# 263
static inline longlong2 make_longlong2(long long x, long long y) 
# 264
{ 
# 265
longlong2 t; (t.x) = x; (t.y) = y; return t; 
# 266
} 
# 268
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) 
# 269
{ 
# 270
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
# 271
} 
# 273
static inline longlong3 make_longlong3(long long x, long long y, long long z) 
# 274
{ 
# 275
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 276
} 
# 278
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) 
# 279
{ 
# 280
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 281
} 
# 283
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w) 
# 284
{ 
# 285
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 286
} 
# 288
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) 
# 289
{ 
# 290
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 291
} 
# 293
static inline double1 make_double1(double x) 
# 294
{ 
# 295
double1 t; (t.x) = x; return t; 
# 296
} 
# 298
static inline double2 make_double2(double x, double y) 
# 299
{ 
# 300
double2 t; (t.x) = x; (t.y) = y; return t; 
# 301
} 
# 303
static inline double3 make_double3(double x, double y, double z) 
# 304
{ 
# 305
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 306
} 
# 308
static inline double4 make_double4(double x, double y, double z, double w) 
# 309
{ 
# 310
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 311
} 
# 27 "/usr/include/string.h" 3
extern "C" {
# 42 "/usr/include/string.h" 3
extern void *memcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 43
 __attribute((__nonnull__(1, 2))); 
# 46
extern void *memmove(void * __dest, const void * __src, size_t __n) throw()
# 47
 __attribute((__nonnull__(1, 2))); 
# 54
extern void *memccpy(void *__restrict__ __dest, const void *__restrict__ __src, int __c, size_t __n) throw()
# 56
 __attribute((__nonnull__(1, 2))); 
# 62
extern void *memset(void * __s, int __c, size_t __n) throw() __attribute((__nonnull__(1))); 
# 65
extern int memcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 66
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 70
extern "C++" {
# 72
extern void *memchr(void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 73
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 74
extern const void *memchr(const void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 75
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 90 "/usr/include/string.h" 3
}
# 101
extern "C++" void *rawmemchr(void * __s, int __c) throw() __asm__("rawmemchr")
# 102
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 103
extern "C++" const void *rawmemchr(const void * __s, int __c) throw() __asm__("rawmemchr")
# 104
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 112
extern "C++" void *memrchr(void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 113
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 114
extern "C++" const void *memrchr(const void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 115
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 125
extern char *strcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 126
 __attribute((__nonnull__(1, 2))); 
# 128
extern char *strncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 130
 __attribute((__nonnull__(1, 2))); 
# 133
extern char *strcat(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 134
 __attribute((__nonnull__(1, 2))); 
# 136
extern char *strncat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 137
 __attribute((__nonnull__(1, 2))); 
# 140
extern int strcmp(const char * __s1, const char * __s2) throw()
# 141
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 143
extern int strncmp(const char * __s1, const char * __s2, size_t __n) throw()
# 144
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 147
extern int strcoll(const char * __s1, const char * __s2) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 150
extern size_t strxfrm(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 152
 __attribute((__nonnull__(2))); 
# 39 "/usr/include/xlocale.h" 3
typedef 
# 27
struct __locale_struct { 
# 30
struct __locale_data *__locales[13]; 
# 33
const unsigned short *__ctype_b; 
# 34
const int *__ctype_tolower; 
# 35
const int *__ctype_toupper; 
# 38
const char *__names[13]; 
# 39
} *__locale_t; 
# 42
typedef __locale_t locale_t; 
# 162 "/usr/include/string.h" 3
extern int strcoll_l(const char * __s1, const char * __s2, __locale_t __l) throw()
# 163
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 165
extern size_t strxfrm_l(char * __dest, const char * __src, size_t __n, __locale_t __l) throw()
# 166
 __attribute((__nonnull__(2, 4))); 
# 172
extern char *strdup(const char * __s) throw()
# 173
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 180
extern char *strndup(const char * __string, size_t __n) throw()
# 181
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 210 "/usr/include/string.h" 3
extern "C++" {
# 212
extern char *strchr(char * __s, int __c) throw() __asm__("strchr")
# 213
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 214
extern const char *strchr(const char * __s, int __c) throw() __asm__("strchr")
# 215
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 230 "/usr/include/string.h" 3
}
# 237
extern "C++" {
# 239
extern char *strrchr(char * __s, int __c) throw() __asm__("strrchr")
# 240
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 241
extern const char *strrchr(const char * __s, int __c) throw() __asm__("strrchr")
# 242
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 257 "/usr/include/string.h" 3
}
# 268
extern "C++" char *strchrnul(char * __s, int __c) throw() __asm__("strchrnul")
# 269
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 270
extern "C++" const char *strchrnul(const char * __s, int __c) throw() __asm__("strchrnul")
# 271
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 281
extern size_t strcspn(const char * __s, const char * __reject) throw()
# 282
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 285
extern size_t strspn(const char * __s, const char * __accept) throw()
# 286
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 289
extern "C++" {
# 291
extern char *strpbrk(char * __s, const char * __accept) throw() __asm__("strpbrk")
# 292
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 293
extern const char *strpbrk(const char * __s, const char * __accept) throw() __asm__("strpbrk")
# 294
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 309 "/usr/include/string.h" 3
}
# 316
extern "C++" {
# 318
extern char *strstr(char * __haystack, const char * __needle) throw() __asm__("strstr")
# 319
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 320
extern const char *strstr(const char * __haystack, const char * __needle) throw() __asm__("strstr")
# 321
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 336 "/usr/include/string.h" 3
}
# 344
extern char *strtok(char *__restrict__ __s, const char *__restrict__ __delim) throw()
# 345
 __attribute((__nonnull__(2))); 
# 350
extern char *__strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 353
 __attribute((__nonnull__(2, 3))); 
# 355
extern char *strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 357
 __attribute((__nonnull__(2, 3))); 
# 363
extern "C++" char *strcasestr(char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 364
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 365
extern "C++" const char *strcasestr(const char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 367
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 378 "/usr/include/string.h" 3
extern void *memmem(const void * __haystack, size_t __haystacklen, const void * __needle, size_t __needlelen) throw()
# 380
 __attribute((__pure__)) __attribute((__nonnull__(1, 3))); 
# 384
extern void *__mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 386
 __attribute((__nonnull__(1, 2))); 
# 387
extern void *mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 389
 __attribute((__nonnull__(1, 2))); 
# 395
extern size_t strlen(const char * __s) throw()
# 396
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 402
extern size_t strnlen(const char * __string, size_t __maxlen) throw()
# 403
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 409
extern char *strerror(int __errnum) throw(); 
# 434 "/usr/include/string.h" 3
extern char *strerror_r(int __errnum, char * __buf, size_t __buflen) throw()
# 435
 __attribute((__nonnull__(2))); 
# 441
extern char *strerror_l(int __errnum, __locale_t __l) throw(); 
# 447
extern void __bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 451
extern void bcopy(const void * __src, void * __dest, size_t __n) throw()
# 452
 __attribute((__nonnull__(1, 2))); 
# 455
extern void bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 458
extern int bcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 459
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 463
extern "C++" {
# 465
extern char *index(char * __s, int __c) throw() __asm__("index")
# 466
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 467
extern const char *index(const char * __s, int __c) throw() __asm__("index")
# 468
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 483 "/usr/include/string.h" 3
}
# 491
extern "C++" {
# 493
extern char *rindex(char * __s, int __c) throw() __asm__("rindex")
# 494
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 495
extern const char *rindex(const char * __s, int __c) throw() __asm__("rindex")
# 496
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 511 "/usr/include/string.h" 3
}
# 519
extern int ffs(int __i) throw() __attribute((const)); 
# 524
extern int ffsl(long __l) throw() __attribute((const)); 
# 526
__extension__ extern int ffsll(long long __ll) throw()
# 527
 __attribute((const)); 
# 532
extern int strcasecmp(const char * __s1, const char * __s2) throw()
# 533
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 536
extern int strncasecmp(const char * __s1, const char * __s2, size_t __n) throw()
# 537
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 543
extern int strcasecmp_l(const char * __s1, const char * __s2, __locale_t __loc) throw()
# 545
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 547
extern int strncasecmp_l(const char * __s1, const char * __s2, size_t __n, __locale_t __loc) throw()
# 549
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 4))); 
# 555
extern char *strsep(char **__restrict__ __stringp, const char *__restrict__ __delim) throw()
# 557
 __attribute((__nonnull__(1, 2))); 
# 562
extern char *strsignal(int __sig) throw(); 
# 565
extern char *__stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 566
 __attribute((__nonnull__(1, 2))); 
# 567
extern char *stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 568
 __attribute((__nonnull__(1, 2))); 
# 572
extern char *__stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 574
 __attribute((__nonnull__(1, 2))); 
# 575
extern char *stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 577
 __attribute((__nonnull__(1, 2))); 
# 582
extern int strverscmp(const char * __s1, const char * __s2) throw()
# 583
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 586
extern char *strfry(char * __string) throw() __attribute((__nonnull__(1))); 
# 589
extern void *memfrob(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 597
extern "C++" char *basename(char * __filename) throw() __asm__("basename")
# 598
 __attribute((__nonnull__(1))); 
# 599
extern "C++" const char *basename(const char * __filename) throw() __asm__("basename")
# 600
 __attribute((__nonnull__(1))); 
# 642 "/usr/include/string.h" 3
}
# 29 "/usr/include/time.h" 3
extern "C" {
# 30 "/usr/include/bits/types.h" 3
typedef unsigned char __u_char; 
# 31
typedef unsigned short __u_short; 
# 32
typedef unsigned __u_int; 
# 33
typedef unsigned long __u_long; 
# 36
typedef signed char __int8_t; 
# 37
typedef unsigned char __uint8_t; 
# 38
typedef signed short __int16_t; 
# 39
typedef unsigned short __uint16_t; 
# 40
typedef signed int __int32_t; 
# 41
typedef unsigned __uint32_t; 
# 43
typedef signed long __int64_t; 
# 44
typedef unsigned long __uint64_t; 
# 52
typedef long __quad_t; 
# 53
typedef unsigned long __u_quad_t; 
# 133 "/usr/include/bits/types.h" 3
typedef unsigned long __dev_t; 
# 134
typedef unsigned __uid_t; 
# 135
typedef unsigned __gid_t; 
# 136
typedef unsigned long __ino_t; 
# 137
typedef unsigned long __ino64_t; 
# 138
typedef unsigned __mode_t; 
# 139
typedef unsigned long __nlink_t; 
# 140
typedef long __off_t; 
# 141
typedef long __off64_t; 
# 142
typedef int __pid_t; 
# 143
typedef struct { int __val[2]; } __fsid_t; 
# 144
typedef long __clock_t; 
# 145
typedef unsigned long __rlim_t; 
# 146
typedef unsigned long __rlim64_t; 
# 147
typedef unsigned __id_t; 
# 148
typedef long __time_t; 
# 149
typedef unsigned __useconds_t; 
# 150
typedef long __suseconds_t; 
# 152
typedef int __daddr_t; 
# 153
typedef int __key_t; 
# 156
typedef int __clockid_t; 
# 159
typedef void *__timer_t; 
# 162
typedef long __blksize_t; 
# 167
typedef long __blkcnt_t; 
# 168
typedef long __blkcnt64_t; 
# 171
typedef unsigned long __fsblkcnt_t; 
# 172
typedef unsigned long __fsblkcnt64_t; 
# 175
typedef unsigned long __fsfilcnt_t; 
# 176
typedef unsigned long __fsfilcnt64_t; 
# 179
typedef long __fsword_t; 
# 181
typedef long __ssize_t; 
# 184
typedef long __syscall_slong_t; 
# 186
typedef unsigned long __syscall_ulong_t; 
# 190
typedef __off64_t __loff_t; 
# 191
typedef __quad_t *__qaddr_t; 
# 192
typedef char *__caddr_t; 
# 195
typedef long __intptr_t; 
# 198
typedef unsigned __socklen_t; 
# 30 "/usr/include/bits/time.h" 3
struct timeval { 
# 32
__time_t tv_sec; 
# 33
__suseconds_t tv_usec; 
# 34
}; 
# 25 "/usr/include/bits/timex.h" 3
struct timex { 
# 27
unsigned modes; 
# 28
__syscall_slong_t offset; 
# 29
__syscall_slong_t freq; 
# 30
__syscall_slong_t maxerror; 
# 31
__syscall_slong_t esterror; 
# 32
int status; 
# 33
__syscall_slong_t constant; 
# 34
__syscall_slong_t precision; 
# 35
__syscall_slong_t tolerance; 
# 36
timeval time; 
# 37
__syscall_slong_t tick; 
# 38
__syscall_slong_t ppsfreq; 
# 39
__syscall_slong_t jitter; 
# 40
int shift; 
# 41
__syscall_slong_t stabil; 
# 42
__syscall_slong_t jitcnt; 
# 43
__syscall_slong_t calcnt; 
# 44
__syscall_slong_t errcnt; 
# 45
__syscall_slong_t stbcnt; 
# 47
int tai; 
# 50
int:32; int:32; int:32; int:32; 
# 51
int:32; int:32; int:32; int:32; 
# 52
int:32; int:32; int:32; 
# 53
}; 
# 90 "/usr/include/bits/time.h" 3
extern "C" {
# 93
extern int clock_adjtime(__clockid_t __clock_id, timex * __utx) throw(); 
# 95
}
# 59 "/usr/include/time.h" 3
typedef __clock_t clock_t; 
# 75 "/usr/include/time.h" 3
typedef __time_t time_t; 
# 91 "/usr/include/time.h" 3
typedef __clockid_t clockid_t; 
# 103 "/usr/include/time.h" 3
typedef __timer_t timer_t; 
# 120 "/usr/include/time.h" 3
struct timespec { 
# 122
__time_t tv_sec; 
# 123
__syscall_slong_t tv_nsec; 
# 124
}; 
# 133
struct tm { 
# 135
int tm_sec; 
# 136
int tm_min; 
# 137
int tm_hour; 
# 138
int tm_mday; 
# 139
int tm_mon; 
# 140
int tm_year; 
# 141
int tm_wday; 
# 142
int tm_yday; 
# 143
int tm_isdst; 
# 146
long tm_gmtoff; 
# 147
const char *tm_zone; 
# 152
}; 
# 161
struct itimerspec { 
# 163
timespec it_interval; 
# 164
timespec it_value; 
# 165
}; 
# 168
struct sigevent; 
# 174
typedef __pid_t pid_t; 
# 189 "/usr/include/time.h" 3
extern clock_t clock() throw(); 
# 192
extern time_t time(time_t * __timer) throw(); 
# 195
extern double difftime(time_t __time1, time_t __time0) throw()
# 196
 __attribute((const)); 
# 199
extern time_t mktime(tm * __tp) throw(); 
# 205
extern size_t strftime(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp) throw(); 
# 213
extern char *strptime(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp) throw(); 
# 223
extern size_t strftime_l(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp, __locale_t __loc) throw(); 
# 230
extern char *strptime_l(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp, __locale_t __loc) throw(); 
# 239
extern tm *gmtime(const time_t * __timer) throw(); 
# 243
extern tm *localtime(const time_t * __timer) throw(); 
# 249
extern tm *gmtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 254
extern tm *localtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 261
extern char *asctime(const tm * __tp) throw(); 
# 264
extern char *ctime(const time_t * __timer) throw(); 
# 272
extern char *asctime_r(const tm *__restrict__ __tp, char *__restrict__ __buf) throw(); 
# 276
extern char *ctime_r(const time_t *__restrict__ __timer, char *__restrict__ __buf) throw(); 
# 282
extern char *__tzname[2]; 
# 283
extern int __daylight; 
# 284
extern long __timezone; 
# 289
extern char *tzname[2]; 
# 293
extern void tzset() throw(); 
# 297
extern int daylight; 
# 298
extern long timezone; 
# 304
extern int stime(const time_t * __when) throw(); 
# 319 "/usr/include/time.h" 3
extern time_t timegm(tm * __tp) throw(); 
# 322
extern time_t timelocal(tm * __tp) throw(); 
# 325
extern int dysize(int __year) throw() __attribute((const)); 
# 334 "/usr/include/time.h" 3
extern int nanosleep(const timespec * __requested_time, timespec * __remaining); 
# 339
extern int clock_getres(clockid_t __clock_id, timespec * __res) throw(); 
# 342
extern int clock_gettime(clockid_t __clock_id, timespec * __tp) throw(); 
# 345
extern int clock_settime(clockid_t __clock_id, const timespec * __tp) throw(); 
# 353
extern int clock_nanosleep(clockid_t __clock_id, int __flags, const timespec * __req, timespec * __rem); 
# 358
extern int clock_getcpuclockid(pid_t __pid, clockid_t * __clock_id) throw(); 
# 363
extern int timer_create(clockid_t __clock_id, sigevent *__restrict__ __evp, timer_t *__restrict__ __timerid) throw(); 
# 368
extern int timer_delete(timer_t __timerid) throw(); 
# 371
extern int timer_settime(timer_t __timerid, int __flags, const itimerspec *__restrict__ __value, itimerspec *__restrict__ __ovalue) throw(); 
# 376
extern int timer_gettime(timer_t __timerid, itimerspec * __value) throw(); 
# 380
extern int timer_getoverrun(timer_t __timerid) throw(); 
# 386
extern int timespec_get(timespec * __ts, int __base) throw()
# 387
 __attribute((__nonnull__(1))); 
# 403 "/usr/include/time.h" 3
extern int getdate_err; 
# 412 "/usr/include/time.h" 3
extern tm *getdate(const char * __string); 
# 426 "/usr/include/time.h" 3
extern int getdate_r(const char *__restrict__ __string, tm *__restrict__ __resbufp); 
# 430
}
# 88 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern "C" {
# 91
extern clock_t clock() throw(); 
# 96 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern void *memset(void *, int, size_t) throw(); 
# 97 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern void *memcpy(void *, const void *, size_t) throw(); 
# 99 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
}
# 115 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern "C" {
# 213 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int abs(int a) throw(); 
# 221 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long labs(long a) throw(); 
# 229 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llabs(long long a) throw(); 
# 279 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fabs(double x) throw(); 
# 320 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fabsf(float x) throw(); 
# 330 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline int min(const int a, const int b); 
# 337
extern inline unsigned umin(const unsigned a, const unsigned b); 
# 344
extern inline long long llmin(const long long a, const long long b); 
# 351
extern inline unsigned long long ullmin(const unsigned long long a, const unsigned long long b); 
# 372 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fminf(float x, float y) throw(); 
# 392 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fmin(double x, double y) throw(); 
# 405 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline int max(const int a, const int b); 
# 413
extern inline unsigned umax(const unsigned a, const unsigned b); 
# 420
extern inline long long llmax(const long long a, const long long b); 
# 427
extern inline unsigned long long ullmax(const unsigned long long a, const unsigned long long b); 
# 448 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fmaxf(float x, float y) throw(); 
# 468 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fmax(double, double) throw(); 
# 512 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sin(double x) throw(); 
# 545 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cos(double x) throw(); 
# 564 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincos(double x, double * sptr, double * cptr) throw(); 
# 580 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincosf(float x, float * sptr, float * cptr) throw(); 
# 625 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double tan(double x) throw(); 
# 694 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sqrt(double x) throw(); 
# 766 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rsqrt(double x); 
# 836 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rsqrtf(float x); 
# 892 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log2(double x) throw(); 
# 917 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double exp2(double x) throw(); 
# 942 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float exp2f(float x) throw(); 
# 969 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double exp10(double x) throw(); 
# 992 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float exp10f(float x) throw(); 
# 1038 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double expm1(double x) throw(); 
# 1083 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float expm1f(float x) throw(); 
# 1138 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float log2f(float x) throw(); 
# 1192 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log10(double x) throw(); 
# 1263 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log(double x) throw(); 
# 1366 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log1p(double x) throw(); 
# 1472 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float log1pf(float x) throw(); 
# 1536 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double floor(double x) throw(); 
# 1575 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double exp(double x) throw(); 
# 1606 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cosh(double x) throw(); 
# 1656 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sinh(double x) throw(); 
# 1686 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double tanh(double x) throw(); 
# 1721 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double acosh(double x) throw(); 
# 1759 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float acoshf(float x) throw(); 
# 1775 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double asinh(double x) throw(); 
# 1791 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float asinhf(float x) throw(); 
# 1845 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double atanh(double x) throw(); 
# 1899 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float atanhf(float x) throw(); 
# 1958 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double ldexp(double x, int exp) throw(); 
# 2014 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float ldexpf(float x, int exp) throw(); 
# 2066 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double logb(double x) throw(); 
# 2121 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float logbf(float x) throw(); 
# 2152 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int ilogb(double x) throw(); 
# 2183 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int ilogbf(float x) throw(); 
# 2259 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double scalbn(double x, int n) throw(); 
# 2335 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float scalbnf(float x, int n) throw(); 
# 2411 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double scalbln(double x, long n) throw(); 
# 2487 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float scalblnf(float x, long n) throw(); 
# 2565 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double frexp(double x, int * nptr) throw(); 
# 2640 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float frexpf(float x, int * nptr) throw(); 
# 2654 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double round(double x) throw(); 
# 2671 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float roundf(float x) throw(); 
# 2689 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lround(double x) throw(); 
# 2707 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lroundf(float x) throw(); 
# 2725 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llround(double x) throw(); 
# 2743 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llroundf(float x) throw(); 
# 2795 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rintf(float x) throw(); 
# 2812 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lrint(double x) throw(); 
# 2829 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lrintf(float x) throw(); 
# 2846 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llrint(double x) throw(); 
# 2863 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llrintf(float x) throw(); 
# 2916 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double nearbyint(double x) throw(); 
# 2969 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float nearbyintf(float x) throw(); 
# 3031 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double ceil(double x) throw(); 
# 3043 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double trunc(double x) throw(); 
# 3058 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float truncf(float x) throw(); 
# 3084 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fdim(double x, double y) throw(); 
# 3110 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fdimf(float x, float y) throw(); 
# 3146 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double atan2(double y, double x) throw(); 
# 3177 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double atan(double x) throw(); 
# 3200 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double acos(double x) throw(); 
# 3232 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double asin(double x) throw(); 
# 3278 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double hypot(double x, double y) throw(); 
# 3376 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float hypotf(float x, float y) throw(); 
# 4108 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cbrt(double x) throw(); 
# 4194 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cbrtf(float x) throw(); 
# 4249 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rcbrt(double x); 
# 4299 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rcbrtf(float x); 
# 4359 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sinpi(double x); 
# 4419 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sinpif(float x); 
# 4471 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cospi(double x); 
# 4523 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cospif(float x); 
# 4553 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincospi(double x, double * sptr, double * cptr); 
# 4583 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincospif(float x, float * sptr, float * cptr); 
# 4895 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double pow(double x, double y) throw(); 
# 4951 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double modf(double x, double * iptr) throw(); 
# 5010 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fmod(double x, double y) throw(); 
# 5096 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double remainder(double x, double y) throw(); 
# 5186 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float remainderf(float x, float y) throw(); 
# 5240 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double remquo(double x, double y, int * quo) throw(); 
# 5294 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float remquof(float x, float y, int * quo) throw(); 
# 5335 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double j0(double x) throw(); 
# 5377 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float j0f(float x) throw(); 
# 5446 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double j1(double x) throw(); 
# 5515 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float j1f(float x) throw(); 
# 5558 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double jn(int n, double x) throw(); 
# 5601 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float jnf(int n, float x) throw(); 
# 5653 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double y0(double x) throw(); 
# 5705 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float y0f(float x) throw(); 
# 5757 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double y1(double x) throw(); 
# 5809 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float y1f(float x) throw(); 
# 5862 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double yn(int n, double x) throw(); 
# 5915 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float ynf(int n, float x) throw(); 
# 6104 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erf(double x) throw(); 
# 6186 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erff(float x) throw(); 
# 6250 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfinv(double y); 
# 6307 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfinvf(float y); 
# 6346 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfc(double x) throw(); 
# 6384 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfcf(float x) throw(); 
# 6512 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double lgamma(double x) throw(); 
# 6575 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfcinv(double y); 
# 6631 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfcinvf(float y); 
# 6689 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double normcdfinv(double y); 
# 6747 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float normcdfinvf(float y); 
# 6790 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double normcdf(double y); 
# 6833 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float normcdff(float y); 
# 6908 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfcx(double x); 
# 6983 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfcxf(float x); 
# 7117 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float lgammaf(float x) throw(); 
# 7226 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double tgamma(double x) throw(); 
# 7335 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float tgammaf(float x) throw(); 
# 7348 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double copysign(double x, double y) throw(); 
# 7361 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float copysignf(float x, float y) throw(); 
# 7380 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double nextafter(double x, double y) throw(); 
# 7399 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float nextafterf(float x, float y) throw(); 
# 7415 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double nan(const char * tagp) throw(); 
# 7431 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float nanf(const char * tagp) throw(); 
# 7438 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isinff(float) throw(); 
# 7439 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isnanf(float) throw(); 
# 7449 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __finite(double) throw(); 
# 7450 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __finitef(float) throw(); 
# 7451 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __signbit(double) throw(); 
# 7452 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isnan(double) throw(); 
# 7453 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isinf(double) throw(); 
# 7456 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __signbitf(float) throw(); 
# 7615 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fma(double x, double y, double z) throw(); 
# 7773 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fmaf(float x, float y, float z) throw(); 
# 7784 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __signbitl(long double) throw(); 
# 7790 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __finitel(long double) throw(); 
# 7791 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isinfl(long double) throw(); 
# 7792 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isnanl(long double) throw(); 
# 7842 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float acosf(float x) throw(); 
# 7882 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float asinf(float x) throw(); 
# 7922 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float atanf(float x) throw(); 
# 7955 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float atan2f(float y, float x) throw(); 
# 7979 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cosf(float x) throw(); 
# 8021 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sinf(float x) throw(); 
# 8063 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float tanf(float x) throw(); 
# 8094 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float coshf(float x) throw(); 
# 8144 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sinhf(float x) throw(); 
# 8174 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float tanhf(float x) throw(); 
# 8225 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float logf(float x) throw(); 
# 8275 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float expf(float x) throw(); 
# 8326 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float log10f(float x) throw(); 
# 8381 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float modff(float x, float * iptr) throw(); 
# 8689 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float powf(float x, float y) throw(); 
# 8758 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sqrtf(float x) throw(); 
# 8817 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float ceilf(float x) throw(); 
# 8878 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float floorf(float x) throw(); 
# 8936 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fmodf(float x, float y) throw(); 
# 8951 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 278 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/x86_64-pc-linux-gnu/bits/c++config.h" 3
namespace std { 
# 280
typedef unsigned long size_t; 
# 281
typedef long ptrdiff_t; 
# 284
typedef __decltype((nullptr)) nullptr_t; 
# 286
}
# 300 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/x86_64-pc-linux-gnu/bits/c++config.h" 3
namespace std { 
# 302
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 303
}
# 304
namespace __gnu_cxx { 
# 306
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 307
}
# 67 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/cpp_type_traits.h" 3
extern "C++" {
# 69
namespace std __attribute((__visibility__("default"))) { 
# 73
struct __true_type { }; 
# 74
struct __false_type { }; 
# 76
template< bool > 
# 77
struct __truth_type { 
# 78
typedef __false_type __type; }; 
# 81
template<> struct __truth_type< true>  { 
# 82
typedef __true_type __type; }; 
# 86
template< class _Sp, class _Tp> 
# 87
struct __traitor { 
# 89
enum { __value = ((bool)_Sp::__value) || ((bool)_Tp::__value)}; 
# 90
typedef typename __truth_type< __value> ::__type __type; 
# 91
}; 
# 94
template< class , class > 
# 95
struct __are_same { 
# 97
enum { __value}; 
# 98
typedef __false_type __type; 
# 99
}; 
# 101
template< class _Tp> 
# 102
struct __are_same< _Tp, _Tp>  { 
# 104
enum { __value = 1}; 
# 105
typedef __true_type __type; 
# 106
}; 
# 109
template< class _Tp> 
# 110
struct __is_void { 
# 112
enum { __value}; 
# 113
typedef __false_type __type; 
# 114
}; 
# 117
template<> struct __is_void< void>  { 
# 119
enum { __value = 1}; 
# 120
typedef __true_type __type; 
# 121
}; 
# 126
template< class _Tp> 
# 127
struct __is_integer { 
# 129
enum { __value}; 
# 130
typedef __false_type __type; 
# 131
}; 
# 138
template<> struct __is_integer< bool>  { 
# 140
enum { __value = 1}; 
# 141
typedef __true_type __type; 
# 142
}; 
# 145
template<> struct __is_integer< char>  { 
# 147
enum { __value = 1}; 
# 148
typedef __true_type __type; 
# 149
}; 
# 152
template<> struct __is_integer< signed char>  { 
# 154
enum { __value = 1}; 
# 155
typedef __true_type __type; 
# 156
}; 
# 159
template<> struct __is_integer< unsigned char>  { 
# 161
enum { __value = 1}; 
# 162
typedef __true_type __type; 
# 163
}; 
# 167
template<> struct __is_integer< wchar_t>  { 
# 169
enum { __value = 1}; 
# 170
typedef __true_type __type; 
# 171
}; 
# 185 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/cpp_type_traits.h" 3
template<> struct __is_integer< char16_t>  { 
# 187
enum { __value = 1}; 
# 188
typedef __true_type __type; 
# 189
}; 
# 192
template<> struct __is_integer< char32_t>  { 
# 194
enum { __value = 1}; 
# 195
typedef __true_type __type; 
# 196
}; 
# 200
template<> struct __is_integer< short>  { 
# 202
enum { __value = 1}; 
# 203
typedef __true_type __type; 
# 204
}; 
# 207
template<> struct __is_integer< unsigned short>  { 
# 209
enum { __value = 1}; 
# 210
typedef __true_type __type; 
# 211
}; 
# 214
template<> struct __is_integer< int>  { 
# 216
enum { __value = 1}; 
# 217
typedef __true_type __type; 
# 218
}; 
# 221
template<> struct __is_integer< unsigned>  { 
# 223
enum { __value = 1}; 
# 224
typedef __true_type __type; 
# 225
}; 
# 228
template<> struct __is_integer< long>  { 
# 230
enum { __value = 1}; 
# 231
typedef __true_type __type; 
# 232
}; 
# 235
template<> struct __is_integer< unsigned long>  { 
# 237
enum { __value = 1}; 
# 238
typedef __true_type __type; 
# 239
}; 
# 242
template<> struct __is_integer< long long>  { 
# 244
enum { __value = 1}; 
# 245
typedef __true_type __type; 
# 246
}; 
# 249
template<> struct __is_integer< unsigned long long>  { 
# 251
enum { __value = 1}; 
# 252
typedef __true_type __type; 
# 253
}; 
# 270 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/cpp_type_traits.h" 3
template<> struct __is_integer< __int128>  { enum { __value = 1}; typedef __true_type __type; }; template<> struct __is_integer< unsigned __int128>  { enum { __value = 1}; typedef __true_type __type; }; 
# 287 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/cpp_type_traits.h" 3
template< class _Tp> 
# 288
struct __is_floating { 
# 290
enum { __value}; 
# 291
typedef __false_type __type; 
# 292
}; 
# 296
template<> struct __is_floating< float>  { 
# 298
enum { __value = 1}; 
# 299
typedef __true_type __type; 
# 300
}; 
# 303
template<> struct __is_floating< double>  { 
# 305
enum { __value = 1}; 
# 306
typedef __true_type __type; 
# 307
}; 
# 310
template<> struct __is_floating< long double>  { 
# 312
enum { __value = 1}; 
# 313
typedef __true_type __type; 
# 314
}; 
# 319
template< class _Tp> 
# 320
struct __is_pointer { 
# 322
enum { __value}; 
# 323
typedef __false_type __type; 
# 324
}; 
# 326
template< class _Tp> 
# 327
struct __is_pointer< _Tp *>  { 
# 329
enum { __value = 1}; 
# 330
typedef __true_type __type; 
# 331
}; 
# 336
template< class _Tp> 
# 337
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> >  { 
# 339
}; 
# 344
template< class _Tp> 
# 345
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> >  { 
# 347
}; 
# 352
template< class _Tp> 
# 353
struct __is_char { 
# 355
enum { __value}; 
# 356
typedef __false_type __type; 
# 357
}; 
# 360
template<> struct __is_char< char>  { 
# 362
enum { __value = 1}; 
# 363
typedef __true_type __type; 
# 364
}; 
# 368
template<> struct __is_char< wchar_t>  { 
# 370
enum { __value = 1}; 
# 371
typedef __true_type __type; 
# 372
}; 
# 375
template< class _Tp> 
# 376
struct __is_byte { 
# 378
enum { __value}; 
# 379
typedef __false_type __type; 
# 380
}; 
# 383
template<> struct __is_byte< char>  { 
# 385
enum { __value = 1}; 
# 386
typedef __true_type __type; 
# 387
}; 
# 390
template<> struct __is_byte< signed char>  { 
# 392
enum { __value = 1}; 
# 393
typedef __true_type __type; 
# 394
}; 
# 397
template<> struct __is_byte< unsigned char>  { 
# 399
enum { __value = 1}; 
# 400
typedef __true_type __type; 
# 401
}; 
# 404
enum class byte: unsigned char; 
# 407
template<> struct __is_byte< byte>  { 
# 409
enum { __value = 1}; 
# 410
typedef __true_type __type; 
# 411
}; 
# 423 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/cpp_type_traits.h" 3
template< class > struct iterator_traits; 
# 426
template< class _Tp> 
# 427
struct __is_nonvolatile_trivially_copyable { 
# 429
enum { __value = __is_trivially_copyable(_Tp)}; 
# 430
}; 
# 435
template< class _Tp> 
# 436
struct __is_nonvolatile_trivially_copyable< volatile _Tp>  { 
# 438
enum { __value}; 
# 439
}; 
# 442
template< class _OutputIter, class _InputIter> 
# 443
struct __memcpyable { 
# 445
enum { __value}; 
# 446
}; 
# 448
template< class _Tp> 
# 449
struct __memcpyable< _Tp *, _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 451
}; 
# 453
template< class _Tp> 
# 454
struct __memcpyable< _Tp *, const _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 456
}; 
# 463
template< class _Iter1, class _Iter2> 
# 464
struct __memcmpable { 
# 466
enum { __value}; 
# 467
}; 
# 470
template< class _Tp> 
# 471
struct __memcmpable< _Tp *, _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 473
}; 
# 475
template< class _Tp> 
# 476
struct __memcmpable< const _Tp *, _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 478
}; 
# 480
template< class _Tp> 
# 481
struct __memcmpable< _Tp *, const _Tp *>  : public __is_nonvolatile_trivially_copyable< _Tp>  { 
# 483
}; 
# 491
template< class _Tp, bool _TreatAsBytes = __is_byte< _Tp> ::__value> 
# 498
struct __is_memcmp_ordered { 
# 500
static const bool __value = (((_Tp)(-1)) > ((_Tp)1)); 
# 501
}; 
# 503
template< class _Tp> 
# 504
struct __is_memcmp_ordered< _Tp, false>  { 
# 506
static const bool __value = false; 
# 507
}; 
# 510
template< class _Tp, class _Up, bool  = sizeof(_Tp) == sizeof(_Up)> 
# 511
struct __is_memcmp_ordered_with { 
# 513
static const bool __value = (__is_memcmp_ordered< _Tp> ::__value && __is_memcmp_ordered< _Up> ::__value); 
# 515
}; 
# 517
template< class _Tp, class _Up> 
# 518
struct __is_memcmp_ordered_with< _Tp, _Up, false>  { 
# 520
static const bool __value = false; 
# 521
}; 
# 533 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/cpp_type_traits.h" 3
template<> struct __is_memcmp_ordered_with< byte, byte, true>  { 
# 534
static constexpr inline bool __value = true; }; 
# 536
template< class _Tp, bool _SameSize> 
# 537
struct __is_memcmp_ordered_with< _Tp, byte, _SameSize>  { 
# 538
static constexpr inline bool __value = false; }; 
# 540
template< class _Up, bool _SameSize> 
# 541
struct __is_memcmp_ordered_with< byte, _Up, _SameSize>  { 
# 542
static constexpr inline bool __value = false; }; 
# 548
template< class _Tp> 
# 549
struct __is_move_iterator { 
# 551
enum { __value}; 
# 552
typedef __false_type __type; 
# 553
}; 
# 557
template< class _Iterator> inline _Iterator 
# 560
__miter_base(_Iterator __it) 
# 561
{ return __it; } 
# 564
}
# 565
}
# 37 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/ext/type_traits.h" 3
extern "C++" {
# 39
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 44
template< bool , class > 
# 45
struct __enable_if { 
# 46
}; 
# 48
template< class _Tp> 
# 49
struct __enable_if< true, _Tp>  { 
# 50
typedef _Tp __type; }; 
# 54
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 55
struct __conditional_type { 
# 56
typedef _Iftrue __type; }; 
# 58
template< class _Iftrue, class _Iffalse> 
# 59
struct __conditional_type< false, _Iftrue, _Iffalse>  { 
# 60
typedef _Iffalse __type; }; 
# 64
template< class _Tp> 
# 65
struct __add_unsigned { 
# 68
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 71
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 72
}; 
# 75
template<> struct __add_unsigned< char>  { 
# 76
typedef unsigned char __type; }; 
# 79
template<> struct __add_unsigned< signed char>  { 
# 80
typedef unsigned char __type; }; 
# 83
template<> struct __add_unsigned< short>  { 
# 84
typedef unsigned short __type; }; 
# 87
template<> struct __add_unsigned< int>  { 
# 88
typedef unsigned __type; }; 
# 91
template<> struct __add_unsigned< long>  { 
# 92
typedef unsigned long __type; }; 
# 95
template<> struct __add_unsigned< long long>  { 
# 96
typedef unsigned long long __type; }; 
# 100
template<> struct __add_unsigned< bool> ; 
# 103
template<> struct __add_unsigned< wchar_t> ; 
# 107
template< class _Tp> 
# 108
struct __remove_unsigned { 
# 111
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 114
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 115
}; 
# 118
template<> struct __remove_unsigned< char>  { 
# 119
typedef signed char __type; }; 
# 122
template<> struct __remove_unsigned< unsigned char>  { 
# 123
typedef signed char __type; }; 
# 126
template<> struct __remove_unsigned< unsigned short>  { 
# 127
typedef short __type; }; 
# 130
template<> struct __remove_unsigned< unsigned>  { 
# 131
typedef int __type; }; 
# 134
template<> struct __remove_unsigned< unsigned long>  { 
# 135
typedef long __type; }; 
# 138
template<> struct __remove_unsigned< unsigned long long>  { 
# 139
typedef long long __type; }; 
# 143
template<> struct __remove_unsigned< bool> ; 
# 146
template<> struct __remove_unsigned< wchar_t> ; 
# 150
template< class _Type> inline bool 
# 152
__is_null_pointer(_Type *__ptr) 
# 153
{ return __ptr == 0; } 
# 155
template< class _Type> inline bool 
# 157
__is_null_pointer(_Type) 
# 158
{ return false; } 
# 162
inline bool __is_null_pointer(std::nullptr_t) 
# 163
{ return true; } 
# 167
template< class _Tp, bool  = std::__is_integer< _Tp> ::__value> 
# 168
struct __promote { 
# 169
typedef double __type; }; 
# 174
template< class _Tp> 
# 175
struct __promote< _Tp, false>  { 
# 176
}; 
# 179
template<> struct __promote< long double>  { 
# 180
typedef long double __type; }; 
# 183
template<> struct __promote< double>  { 
# 184
typedef double __type; }; 
# 187
template<> struct __promote< float>  { 
# 188
typedef float __type; }; 
# 190
template< class _Tp, class _Up, class 
# 191
_Tp2 = typename __promote< _Tp> ::__type, class 
# 192
_Up2 = typename __promote< _Up> ::__type> 
# 193
struct __promote_2 { 
# 195
typedef __typeof__(_Tp2() + _Up2()) __type; 
# 196
}; 
# 198
template< class _Tp, class _Up, class _Vp, class 
# 199
_Tp2 = typename __promote< _Tp> ::__type, class 
# 200
_Up2 = typename __promote< _Up> ::__type, class 
# 201
_Vp2 = typename __promote< _Vp> ::__type> 
# 202
struct __promote_3 { 
# 204
typedef __typeof__((_Tp2() + _Up2()) + _Vp2()) __type; 
# 205
}; 
# 207
template< class _Tp, class _Up, class _Vp, class _Wp, class 
# 208
_Tp2 = typename __promote< _Tp> ::__type, class 
# 209
_Up2 = typename __promote< _Up> ::__type, class 
# 210
_Vp2 = typename __promote< _Vp> ::__type, class 
# 211
_Wp2 = typename __promote< _Wp> ::__type> 
# 212
struct __promote_4 { 
# 214
typedef __typeof__(((_Tp2() + _Up2()) + _Vp2()) + _Wp2()) __type; 
# 215
}; 
# 218
}
# 219
}
# 29 "/usr/include/math.h" 3
extern "C" {
# 28 "/usr/include/bits/mathdef.h" 3
typedef float float_t; 
# 29
typedef double double_t; 
# 54 "/usr/include/bits/mathcalls.h" 3
extern double acos(double __x) throw(); extern double __acos(double __x) throw(); 
# 56
extern double asin(double __x) throw(); extern double __asin(double __x) throw(); 
# 58
extern double atan(double __x) throw(); extern double __atan(double __x) throw(); 
# 60
extern double atan2(double __y, double __x) throw(); extern double __atan2(double __y, double __x) throw(); 
# 63
extern double cos(double __x) throw(); extern double __cos(double __x) throw(); 
# 65
extern double sin(double __x) throw(); extern double __sin(double __x) throw(); 
# 67
extern double tan(double __x) throw(); extern double __tan(double __x) throw(); 
# 72
extern double cosh(double __x) throw(); extern double __cosh(double __x) throw(); 
# 74
extern double sinh(double __x) throw(); extern double __sinh(double __x) throw(); 
# 76
extern double tanh(double __x) throw(); extern double __tanh(double __x) throw(); 
# 81
extern void sincos(double __x, double * __sinx, double * __cosx) throw(); extern void __sincos(double __x, double * __sinx, double * __cosx) throw(); 
# 88
extern double acosh(double __x) throw(); extern double __acosh(double __x) throw(); 
# 90
extern double asinh(double __x) throw(); extern double __asinh(double __x) throw(); 
# 92
extern double atanh(double __x) throw(); extern double __atanh(double __x) throw(); 
# 100
extern double exp(double __x) throw(); extern double __exp(double __x) throw(); 
# 103
extern double frexp(double __x, int * __exponent) throw(); extern double __frexp(double __x, int * __exponent) throw(); 
# 106
extern double ldexp(double __x, int __exponent) throw(); extern double __ldexp(double __x, int __exponent) throw(); 
# 109
extern double log(double __x) throw(); extern double __log(double __x) throw(); 
# 112
extern double log10(double __x) throw(); extern double __log10(double __x) throw(); 
# 115
extern double modf(double __x, double * __iptr) throw(); extern double __modf(double __x, double * __iptr) throw()
# 116
 __attribute((__nonnull__(2))); 
# 121
extern double exp10(double __x) throw(); extern double __exp10(double __x) throw(); 
# 123
extern double pow10(double __x) throw(); extern double __pow10(double __x) throw(); 
# 129
extern double expm1(double __x) throw(); extern double __expm1(double __x) throw(); 
# 132
extern double log1p(double __x) throw(); extern double __log1p(double __x) throw(); 
# 135
extern double logb(double __x) throw(); extern double __logb(double __x) throw(); 
# 142
extern double exp2(double __x) throw(); extern double __exp2(double __x) throw(); 
# 145
extern double log2(double __x) throw(); extern double __log2(double __x) throw(); 
# 154
extern double pow(double __x, double __y) throw(); extern double __pow(double __x, double __y) throw(); 
# 157
extern double sqrt(double __x) throw(); extern double __sqrt(double __x) throw(); 
# 163
extern double hypot(double __x, double __y) throw(); extern double __hypot(double __x, double __y) throw(); 
# 170
extern double cbrt(double __x) throw(); extern double __cbrt(double __x) throw(); 
# 179
extern double ceil(double __x) throw() __attribute((const)); extern double __ceil(double __x) throw() __attribute((const)); 
# 182
extern double fabs(double __x) throw() __attribute((const)); extern double __fabs(double __x) throw() __attribute((const)); 
# 185
extern double floor(double __x) throw() __attribute((const)); extern double __floor(double __x) throw() __attribute((const)); 
# 188
extern double fmod(double __x, double __y) throw(); extern double __fmod(double __x, double __y) throw(); 
# 193
extern int __isinf(double __value) throw() __attribute((const)); 
# 196
extern int __finite(double __value) throw() __attribute((const)); 
# 202
extern int isinf(double __value) throw() __attribute((const)); 
# 205
extern int finite(double __value) throw() __attribute((const)); 
# 208
extern double drem(double __x, double __y) throw(); extern double __drem(double __x, double __y) throw(); 
# 212
extern double significand(double __x) throw(); extern double __significand(double __x) throw(); 
# 218
extern double copysign(double __x, double __y) throw() __attribute((const)); extern double __copysign(double __x, double __y) throw() __attribute((const)); 
# 225
extern double nan(const char * __tagb) throw() __attribute((const)); extern double __nan(const char * __tagb) throw() __attribute((const)); 
# 231
extern int __isnan(double __value) throw() __attribute((const)); 
# 235
extern int isnan(double __value) throw() __attribute((const)); 
# 238
extern double j0(double) throw(); extern double __j0(double) throw(); 
# 239
extern double j1(double) throw(); extern double __j1(double) throw(); 
# 240
extern double jn(int, double) throw(); extern double __jn(int, double) throw(); 
# 241
extern double y0(double) throw(); extern double __y0(double) throw(); 
# 242
extern double y1(double) throw(); extern double __y1(double) throw(); 
# 243
extern double yn(int, double) throw(); extern double __yn(int, double) throw(); 
# 250
extern double erf(double) throw(); extern double __erf(double) throw(); 
# 251
extern double erfc(double) throw(); extern double __erfc(double) throw(); 
# 252
extern double lgamma(double) throw(); extern double __lgamma(double) throw(); 
# 259
extern double tgamma(double) throw(); extern double __tgamma(double) throw(); 
# 265
extern double gamma(double) throw(); extern double __gamma(double) throw(); 
# 272
extern double lgamma_r(double, int * __signgamp) throw(); extern double __lgamma_r(double, int * __signgamp) throw(); 
# 280
extern double rint(double __x) throw(); extern double __rint(double __x) throw(); 
# 283
extern double nextafter(double __x, double __y) throw() __attribute((const)); extern double __nextafter(double __x, double __y) throw() __attribute((const)); 
# 285
extern double nexttoward(double __x, long double __y) throw() __attribute((const)); extern double __nexttoward(double __x, long double __y) throw() __attribute((const)); 
# 289
extern double remainder(double __x, double __y) throw(); extern double __remainder(double __x, double __y) throw(); 
# 293
extern double scalbn(double __x, int __n) throw(); extern double __scalbn(double __x, int __n) throw(); 
# 297
extern int ilogb(double __x) throw(); extern int __ilogb(double __x) throw(); 
# 302
extern double scalbln(double __x, long __n) throw(); extern double __scalbln(double __x, long __n) throw(); 
# 306
extern double nearbyint(double __x) throw(); extern double __nearbyint(double __x) throw(); 
# 310
extern double round(double __x) throw() __attribute((const)); extern double __round(double __x) throw() __attribute((const)); 
# 314
extern double trunc(double __x) throw() __attribute((const)); extern double __trunc(double __x) throw() __attribute((const)); 
# 319
extern double remquo(double __x, double __y, int * __quo) throw(); extern double __remquo(double __x, double __y, int * __quo) throw(); 
# 326
extern long lrint(double __x) throw(); extern long __lrint(double __x) throw(); 
# 327
extern long long llrint(double __x) throw(); extern long long __llrint(double __x) throw(); 
# 331
extern long lround(double __x) throw(); extern long __lround(double __x) throw(); 
# 332
extern long long llround(double __x) throw(); extern long long __llround(double __x) throw(); 
# 336
extern double fdim(double __x, double __y) throw(); extern double __fdim(double __x, double __y) throw(); 
# 339
extern double fmax(double __x, double __y) throw() __attribute((const)); extern double __fmax(double __x, double __y) throw() __attribute((const)); 
# 342
extern double fmin(double __x, double __y) throw() __attribute((const)); extern double __fmin(double __x, double __y) throw() __attribute((const)); 
# 346
extern int __fpclassify(double __value) throw()
# 347
 __attribute((const)); 
# 350
extern int __signbit(double __value) throw()
# 351
 __attribute((const)); 
# 355
extern double fma(double __x, double __y, double __z) throw(); extern double __fma(double __x, double __y, double __z) throw(); 
# 364
extern double scalb(double __x, double __n) throw(); extern double __scalb(double __x, double __n) throw(); 
# 54 "/usr/include/bits/mathcalls.h" 3
extern float acosf(float __x) throw(); extern float __acosf(float __x) throw(); 
# 56
extern float asinf(float __x) throw(); extern float __asinf(float __x) throw(); 
# 58
extern float atanf(float __x) throw(); extern float __atanf(float __x) throw(); 
# 60
extern float atan2f(float __y, float __x) throw(); extern float __atan2f(float __y, float __x) throw(); 
# 63
extern float cosf(float __x) throw(); 
# 65
extern float sinf(float __x) throw(); 
# 67
extern float tanf(float __x) throw(); 
# 72
extern float coshf(float __x) throw(); extern float __coshf(float __x) throw(); 
# 74
extern float sinhf(float __x) throw(); extern float __sinhf(float __x) throw(); 
# 76
extern float tanhf(float __x) throw(); extern float __tanhf(float __x) throw(); 
# 81
extern void sincosf(float __x, float * __sinx, float * __cosx) throw(); 
# 88
extern float acoshf(float __x) throw(); extern float __acoshf(float __x) throw(); 
# 90
extern float asinhf(float __x) throw(); extern float __asinhf(float __x) throw(); 
# 92
extern float atanhf(float __x) throw(); extern float __atanhf(float __x) throw(); 
# 100
extern float expf(float __x) throw(); 
# 103
extern float frexpf(float __x, int * __exponent) throw(); extern float __frexpf(float __x, int * __exponent) throw(); 
# 106
extern float ldexpf(float __x, int __exponent) throw(); extern float __ldexpf(float __x, int __exponent) throw(); 
# 109
extern float logf(float __x) throw(); 
# 112
extern float log10f(float __x) throw(); 
# 115
extern float modff(float __x, float * __iptr) throw(); extern float __modff(float __x, float * __iptr) throw()
# 116
 __attribute((__nonnull__(2))); 
# 121
extern float exp10f(float __x) throw(); 
# 123
extern float pow10f(float __x) throw(); extern float __pow10f(float __x) throw(); 
# 129
extern float expm1f(float __x) throw(); extern float __expm1f(float __x) throw(); 
# 132
extern float log1pf(float __x) throw(); extern float __log1pf(float __x) throw(); 
# 135
extern float logbf(float __x) throw(); extern float __logbf(float __x) throw(); 
# 142
extern float exp2f(float __x) throw(); extern float __exp2f(float __x) throw(); 
# 145
extern float log2f(float __x) throw(); 
# 154
extern float powf(float __x, float __y) throw(); 
# 157
extern float sqrtf(float __x) throw(); extern float __sqrtf(float __x) throw(); 
# 163
extern float hypotf(float __x, float __y) throw(); extern float __hypotf(float __x, float __y) throw(); 
# 170
extern float cbrtf(float __x) throw(); extern float __cbrtf(float __x) throw(); 
# 179
extern float ceilf(float __x) throw() __attribute((const)); extern float __ceilf(float __x) throw() __attribute((const)); 
# 182
extern float fabsf(float __x) throw() __attribute((const)); extern float __fabsf(float __x) throw() __attribute((const)); 
# 185
extern float floorf(float __x) throw() __attribute((const)); extern float __floorf(float __x) throw() __attribute((const)); 
# 188
extern float fmodf(float __x, float __y) throw(); extern float __fmodf(float __x, float __y) throw(); 
# 193
extern int __isinff(float __value) throw() __attribute((const)); 
# 196
extern int __finitef(float __value) throw() __attribute((const)); 
# 202
extern int isinff(float __value) throw() __attribute((const)); 
# 205
extern int finitef(float __value) throw() __attribute((const)); 
# 208
extern float dremf(float __x, float __y) throw(); extern float __dremf(float __x, float __y) throw(); 
# 212
extern float significandf(float __x) throw(); extern float __significandf(float __x) throw(); 
# 218
extern float copysignf(float __x, float __y) throw() __attribute((const)); extern float __copysignf(float __x, float __y) throw() __attribute((const)); 
# 225
extern float nanf(const char * __tagb) throw() __attribute((const)); extern float __nanf(const char * __tagb) throw() __attribute((const)); 
# 231
extern int __isnanf(float __value) throw() __attribute((const)); 
# 235
extern int isnanf(float __value) throw() __attribute((const)); 
# 238
extern float j0f(float) throw(); extern float __j0f(float) throw(); 
# 239
extern float j1f(float) throw(); extern float __j1f(float) throw(); 
# 240
extern float jnf(int, float) throw(); extern float __jnf(int, float) throw(); 
# 241
extern float y0f(float) throw(); extern float __y0f(float) throw(); 
# 242
extern float y1f(float) throw(); extern float __y1f(float) throw(); 
# 243
extern float ynf(int, float) throw(); extern float __ynf(int, float) throw(); 
# 250
extern float erff(float) throw(); extern float __erff(float) throw(); 
# 251
extern float erfcf(float) throw(); extern float __erfcf(float) throw(); 
# 252
extern float lgammaf(float) throw(); extern float __lgammaf(float) throw(); 
# 259
extern float tgammaf(float) throw(); extern float __tgammaf(float) throw(); 
# 265
extern float gammaf(float) throw(); extern float __gammaf(float) throw(); 
# 272
extern float lgammaf_r(float, int * __signgamp) throw(); extern float __lgammaf_r(float, int * __signgamp) throw(); 
# 280
extern float rintf(float __x) throw(); extern float __rintf(float __x) throw(); 
# 283
extern float nextafterf(float __x, float __y) throw() __attribute((const)); extern float __nextafterf(float __x, float __y) throw() __attribute((const)); 
# 285
extern float nexttowardf(float __x, long double __y) throw() __attribute((const)); extern float __nexttowardf(float __x, long double __y) throw() __attribute((const)); 
# 289
extern float remainderf(float __x, float __y) throw(); extern float __remainderf(float __x, float __y) throw(); 
# 293
extern float scalbnf(float __x, int __n) throw(); extern float __scalbnf(float __x, int __n) throw(); 
# 297
extern int ilogbf(float __x) throw(); extern int __ilogbf(float __x) throw(); 
# 302
extern float scalblnf(float __x, long __n) throw(); extern float __scalblnf(float __x, long __n) throw(); 
# 306
extern float nearbyintf(float __x) throw(); extern float __nearbyintf(float __x) throw(); 
# 310
extern float roundf(float __x) throw() __attribute((const)); extern float __roundf(float __x) throw() __attribute((const)); 
# 314
extern float truncf(float __x) throw() __attribute((const)); extern float __truncf(float __x) throw() __attribute((const)); 
# 319
extern float remquof(float __x, float __y, int * __quo) throw(); extern float __remquof(float __x, float __y, int * __quo) throw(); 
# 326
extern long lrintf(float __x) throw(); extern long __lrintf(float __x) throw(); 
# 327
extern long long llrintf(float __x) throw(); extern long long __llrintf(float __x) throw(); 
# 331
extern long lroundf(float __x) throw(); extern long __lroundf(float __x) throw(); 
# 332
extern long long llroundf(float __x) throw(); extern long long __llroundf(float __x) throw(); 
# 336
extern float fdimf(float __x, float __y) throw(); extern float __fdimf(float __x, float __y) throw(); 
# 339
extern float fmaxf(float __x, float __y) throw() __attribute((const)); extern float __fmaxf(float __x, float __y) throw() __attribute((const)); 
# 342
extern float fminf(float __x, float __y) throw() __attribute((const)); extern float __fminf(float __x, float __y) throw() __attribute((const)); 
# 346
extern int __fpclassifyf(float __value) throw()
# 347
 __attribute((const)); 
# 350
extern int __signbitf(float __value) throw()
# 351
 __attribute((const)); 
# 355
extern float fmaf(float __x, float __y, float __z) throw(); extern float __fmaf(float __x, float __y, float __z) throw(); 
# 364
extern float scalbf(float __x, float __n) throw(); extern float __scalbf(float __x, float __n) throw(); 
# 54 "/usr/include/bits/mathcalls.h" 3
extern long double acosl(long double __x) throw(); extern long double __acosl(long double __x) throw(); 
# 56
extern long double asinl(long double __x) throw(); extern long double __asinl(long double __x) throw(); 
# 58
extern long double atanl(long double __x) throw(); extern long double __atanl(long double __x) throw(); 
# 60
extern long double atan2l(long double __y, long double __x) throw(); extern long double __atan2l(long double __y, long double __x) throw(); 
# 63
extern long double cosl(long double __x) throw(); extern long double __cosl(long double __x) throw(); 
# 65
extern long double sinl(long double __x) throw(); extern long double __sinl(long double __x) throw(); 
# 67
extern long double tanl(long double __x) throw(); extern long double __tanl(long double __x) throw(); 
# 72
extern long double coshl(long double __x) throw(); extern long double __coshl(long double __x) throw(); 
# 74
extern long double sinhl(long double __x) throw(); extern long double __sinhl(long double __x) throw(); 
# 76
extern long double tanhl(long double __x) throw(); extern long double __tanhl(long double __x) throw(); 
# 81
extern void sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); extern void __sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); 
# 88
extern long double acoshl(long double __x) throw(); extern long double __acoshl(long double __x) throw(); 
# 90
extern long double asinhl(long double __x) throw(); extern long double __asinhl(long double __x) throw(); 
# 92
extern long double atanhl(long double __x) throw(); extern long double __atanhl(long double __x) throw(); 
# 100
extern long double expl(long double __x) throw(); extern long double __expl(long double __x) throw(); 
# 103
extern long double frexpl(long double __x, int * __exponent) throw(); extern long double __frexpl(long double __x, int * __exponent) throw(); 
# 106
extern long double ldexpl(long double __x, int __exponent) throw(); extern long double __ldexpl(long double __x, int __exponent) throw(); 
# 109
extern long double logl(long double __x) throw(); extern long double __logl(long double __x) throw(); 
# 112
extern long double log10l(long double __x) throw(); extern long double __log10l(long double __x) throw(); 
# 115
extern long double modfl(long double __x, long double * __iptr) throw(); extern long double __modfl(long double __x, long double * __iptr) throw()
# 116
 __attribute((__nonnull__(2))); 
# 121
extern long double exp10l(long double __x) throw(); extern long double __exp10l(long double __x) throw(); 
# 123
extern long double pow10l(long double __x) throw(); extern long double __pow10l(long double __x) throw(); 
# 129
extern long double expm1l(long double __x) throw(); extern long double __expm1l(long double __x) throw(); 
# 132
extern long double log1pl(long double __x) throw(); extern long double __log1pl(long double __x) throw(); 
# 135
extern long double logbl(long double __x) throw(); extern long double __logbl(long double __x) throw(); 
# 142
extern long double exp2l(long double __x) throw(); extern long double __exp2l(long double __x) throw(); 
# 145
extern long double log2l(long double __x) throw(); extern long double __log2l(long double __x) throw(); 
# 154
extern long double powl(long double __x, long double __y) throw(); extern long double __powl(long double __x, long double __y) throw(); 
# 157
extern long double sqrtl(long double __x) throw(); extern long double __sqrtl(long double __x) throw(); 
# 163
extern long double hypotl(long double __x, long double __y) throw(); extern long double __hypotl(long double __x, long double __y) throw(); 
# 170
extern long double cbrtl(long double __x) throw(); extern long double __cbrtl(long double __x) throw(); 
# 179
extern long double ceill(long double __x) throw() __attribute((const)); extern long double __ceill(long double __x) throw() __attribute((const)); 
# 182
extern long double fabsl(long double __x) throw() __attribute((const)); extern long double __fabsl(long double __x) throw() __attribute((const)); 
# 185
extern long double floorl(long double __x) throw() __attribute((const)); extern long double __floorl(long double __x) throw() __attribute((const)); 
# 188
extern long double fmodl(long double __x, long double __y) throw(); extern long double __fmodl(long double __x, long double __y) throw(); 
# 193
extern int __isinfl(long double __value) throw() __attribute((const)); 
# 196
extern int __finitel(long double __value) throw() __attribute((const)); 
# 202
extern int isinfl(long double __value) throw() __attribute((const)); 
# 205
extern int finitel(long double __value) throw() __attribute((const)); 
# 208
extern long double dreml(long double __x, long double __y) throw(); extern long double __dreml(long double __x, long double __y) throw(); 
# 212
extern long double significandl(long double __x) throw(); extern long double __significandl(long double __x) throw(); 
# 218
extern long double copysignl(long double __x, long double __y) throw() __attribute((const)); extern long double __copysignl(long double __x, long double __y) throw() __attribute((const)); 
# 225
extern long double nanl(const char * __tagb) throw() __attribute((const)); extern long double __nanl(const char * __tagb) throw() __attribute((const)); 
# 231
extern int __isnanl(long double __value) throw() __attribute((const)); 
# 235
extern int isnanl(long double __value) throw() __attribute((const)); 
# 238
extern long double j0l(long double) throw(); extern long double __j0l(long double) throw(); 
# 239
extern long double j1l(long double) throw(); extern long double __j1l(long double) throw(); 
# 240
extern long double jnl(int, long double) throw(); extern long double __jnl(int, long double) throw(); 
# 241
extern long double y0l(long double) throw(); extern long double __y0l(long double) throw(); 
# 242
extern long double y1l(long double) throw(); extern long double __y1l(long double) throw(); 
# 243
extern long double ynl(int, long double) throw(); extern long double __ynl(int, long double) throw(); 
# 250
extern long double erfl(long double) throw(); extern long double __erfl(long double) throw(); 
# 251
extern long double erfcl(long double) throw(); extern long double __erfcl(long double) throw(); 
# 252
extern long double lgammal(long double) throw(); extern long double __lgammal(long double) throw(); 
# 259
extern long double tgammal(long double) throw(); extern long double __tgammal(long double) throw(); 
# 265
extern long double gammal(long double) throw(); extern long double __gammal(long double) throw(); 
# 272
extern long double lgammal_r(long double, int * __signgamp) throw(); extern long double __lgammal_r(long double, int * __signgamp) throw(); 
# 280
extern long double rintl(long double __x) throw(); extern long double __rintl(long double __x) throw(); 
# 283
extern long double nextafterl(long double __x, long double __y) throw() __attribute((const)); extern long double __nextafterl(long double __x, long double __y) throw() __attribute((const)); 
# 285
extern long double nexttowardl(long double __x, long double __y) throw() __attribute((const)); extern long double __nexttowardl(long double __x, long double __y) throw() __attribute((const)); 
# 289
extern long double remainderl(long double __x, long double __y) throw(); extern long double __remainderl(long double __x, long double __y) throw(); 
# 293
extern long double scalbnl(long double __x, int __n) throw(); extern long double __scalbnl(long double __x, int __n) throw(); 
# 297
extern int ilogbl(long double __x) throw(); extern int __ilogbl(long double __x) throw(); 
# 302
extern long double scalblnl(long double __x, long __n) throw(); extern long double __scalblnl(long double __x, long __n) throw(); 
# 306
extern long double nearbyintl(long double __x) throw(); extern long double __nearbyintl(long double __x) throw(); 
# 310
extern long double roundl(long double __x) throw() __attribute((const)); extern long double __roundl(long double __x) throw() __attribute((const)); 
# 314
extern long double truncl(long double __x) throw() __attribute((const)); extern long double __truncl(long double __x) throw() __attribute((const)); 
# 319
extern long double remquol(long double __x, long double __y, int * __quo) throw(); extern long double __remquol(long double __x, long double __y, int * __quo) throw(); 
# 326
extern long lrintl(long double __x) throw(); extern long __lrintl(long double __x) throw(); 
# 327
extern long long llrintl(long double __x) throw(); extern long long __llrintl(long double __x) throw(); 
# 331
extern long lroundl(long double __x) throw(); extern long __lroundl(long double __x) throw(); 
# 332
extern long long llroundl(long double __x) throw(); extern long long __llroundl(long double __x) throw(); 
# 336
extern long double fdiml(long double __x, long double __y) throw(); extern long double __fdiml(long double __x, long double __y) throw(); 
# 339
extern long double fmaxl(long double __x, long double __y) throw() __attribute((const)); extern long double __fmaxl(long double __x, long double __y) throw() __attribute((const)); 
# 342
extern long double fminl(long double __x, long double __y) throw() __attribute((const)); extern long double __fminl(long double __x, long double __y) throw() __attribute((const)); 
# 346
extern int __fpclassifyl(long double __value) throw()
# 347
 __attribute((const)); 
# 350
extern int __signbitl(long double __value) throw()
# 351
 __attribute((const)); 
# 355
extern long double fmal(long double __x, long double __y, long double __z) throw(); extern long double __fmal(long double __x, long double __y, long double __z) throw(); 
# 364
extern long double scalbl(long double __x, long double __n) throw(); extern long double __scalbl(long double __x, long double __n) throw(); 
# 149 "/usr/include/math.h" 3
extern int signgam; 
# 191 "/usr/include/math.h" 3
enum { 
# 192
FP_NAN, 
# 195
FP_INFINITE, 
# 198
FP_ZERO, 
# 201
FP_SUBNORMAL, 
# 204
FP_NORMAL
# 207
}; 
# 295 "/usr/include/math.h" 3
typedef 
# 289
enum { 
# 290
_IEEE_ = (-1), 
# 291
_SVID_ = 0, 
# 292
_XOPEN_, 
# 293
_POSIX_, 
# 294
_ISOC_
# 295
} _LIB_VERSION_TYPE; 
# 300
extern _LIB_VERSION_TYPE _LIB_VERSION; 
# 311 "/usr/include/math.h" 3
struct __exception { 
# 316
int type; 
# 317
char *name; 
# 318
double arg1; 
# 319
double arg2; 
# 320
double retval; 
# 321
}; 
# 324
extern int matherr(__exception * __exc) throw(); 
# 475 "/usr/include/math.h" 3
}
# 34 "/usr/include/stdlib.h" 3
extern "C" {
# 45 "/usr/include/bits/byteswap.h" 3
static inline unsigned __bswap_32(unsigned __bsx) 
# 46
{ 
# 47
return __builtin_bswap32(__bsx); 
# 48
} 
# 109 "/usr/include/bits/byteswap.h" 3
static inline __uint64_t __bswap_64(__uint64_t __bsx) 
# 110
{ 
# 111
return __builtin_bswap64(__bsx); 
# 112
} 
# 66 "/usr/include/bits/waitstatus.h" 3
union wait { 
# 68
int w_status; 
# 70
struct { 
# 72
unsigned __w_termsig:7; 
# 73
unsigned __w_coredump:1; 
# 74
unsigned __w_retcode:8; 
# 75
unsigned:16; 
# 83
} __wait_terminated; 
# 85
struct { 
# 87
unsigned __w_stopval:8; 
# 88
unsigned __w_stopsig:8; 
# 89
unsigned:16; 
# 96
} __wait_stopped; 
# 97
}; 
# 101 "/usr/include/stdlib.h" 3
typedef 
# 98
struct { 
# 99
int quot; 
# 100
int rem; 
# 101
} div_t; 
# 109
typedef 
# 106
struct { 
# 107
long quot; 
# 108
long rem; 
# 109
} ldiv_t; 
# 121
__extension__ typedef 
# 118
struct { 
# 119
long long quot; 
# 120
long long rem; 
# 121
} lldiv_t; 
# 139 "/usr/include/stdlib.h" 3
extern size_t __ctype_get_mb_cur_max() throw(); 
# 144
extern double atof(const char * __nptr) throw()
# 145
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 147
extern int atoi(const char * __nptr) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 150
extern long atol(const char * __nptr) throw()
# 151
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 157
__extension__ extern long long atoll(const char * __nptr) throw()
# 158
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 164
extern double strtod(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 166
 __attribute((__nonnull__(1))); 
# 172
extern float strtof(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 173
 __attribute((__nonnull__(1))); 
# 175
extern long double strtold(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 177
 __attribute((__nonnull__(1))); 
# 183
extern long strtol(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 185
 __attribute((__nonnull__(1))); 
# 187
extern unsigned long strtoul(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 189
 __attribute((__nonnull__(1))); 
# 195
__extension__ extern long long strtoq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 197
 __attribute((__nonnull__(1))); 
# 200
__extension__ extern unsigned long long strtouq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 202
 __attribute((__nonnull__(1))); 
# 209
__extension__ extern long long strtoll(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 211
 __attribute((__nonnull__(1))); 
# 214
__extension__ extern unsigned long long strtoull(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 216
 __attribute((__nonnull__(1))); 
# 239 "/usr/include/stdlib.h" 3
extern long strtol_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 241
 __attribute((__nonnull__(1, 4))); 
# 243
extern unsigned long strtoul_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 246
 __attribute((__nonnull__(1, 4))); 
# 249
__extension__ extern long long strtoll_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 252
 __attribute((__nonnull__(1, 4))); 
# 255
__extension__ extern unsigned long long strtoull_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 258
 __attribute((__nonnull__(1, 4))); 
# 260
extern double strtod_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 262
 __attribute((__nonnull__(1, 3))); 
# 264
extern float strtof_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 266
 __attribute((__nonnull__(1, 3))); 
# 268
extern long double strtold_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 271
 __attribute((__nonnull__(1, 3))); 
# 305 "/usr/include/stdlib.h" 3
extern char *l64a(long __n) throw(); 
# 308
extern long a64l(const char * __s) throw()
# 309
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 27 "/usr/include/sys/types.h" 3
extern "C" {
# 33
typedef __u_char u_char; 
# 34
typedef __u_short u_short; 
# 35
typedef __u_int u_int; 
# 36
typedef __u_long u_long; 
# 37
typedef __quad_t quad_t; 
# 38
typedef __u_quad_t u_quad_t; 
# 39
typedef __fsid_t fsid_t; 
# 44
typedef __loff_t loff_t; 
# 48
typedef __ino_t ino_t; 
# 55
typedef __ino64_t ino64_t; 
# 60
typedef __dev_t dev_t; 
# 65
typedef __gid_t gid_t; 
# 70
typedef __mode_t mode_t; 
# 75
typedef __nlink_t nlink_t; 
# 80
typedef __uid_t uid_t; 
# 86
typedef __off_t off_t; 
# 93
typedef __off64_t off64_t; 
# 104 "/usr/include/sys/types.h" 3
typedef __id_t id_t; 
# 109
typedef __ssize_t ssize_t; 
# 115
typedef __daddr_t daddr_t; 
# 116
typedef __caddr_t caddr_t; 
# 122
typedef __key_t key_t; 
# 136 "/usr/include/sys/types.h" 3
typedef __useconds_t useconds_t; 
# 140
typedef __suseconds_t suseconds_t; 
# 150 "/usr/include/sys/types.h" 3
typedef unsigned long ulong; 
# 151
typedef unsigned short ushort; 
# 152
typedef unsigned uint; 
# 194 "/usr/include/sys/types.h" 3
typedef signed char int8_t __attribute((__mode__(__QI__))); 
# 195
typedef short int16_t __attribute((__mode__(__HI__))); 
# 196
typedef int int32_t __attribute((__mode__(__SI__))); 
# 197
typedef long int64_t __attribute((__mode__(__DI__))); 
# 200
typedef unsigned char u_int8_t __attribute((__mode__(__QI__))); 
# 201
typedef unsigned short u_int16_t __attribute((__mode__(__HI__))); 
# 202
typedef unsigned u_int32_t __attribute((__mode__(__SI__))); 
# 203
typedef unsigned long u_int64_t __attribute((__mode__(__DI__))); 
# 205
typedef long register_t __attribute((__mode__(__word__))); 
# 23 "/usr/include/bits/sigset.h" 3
typedef int __sig_atomic_t; 
# 31
typedef 
# 29
struct { 
# 30
unsigned long __val[(1024) / ((8) * sizeof(unsigned long))]; 
# 31
} __sigset_t; 
# 37 "/usr/include/sys/select.h" 3
typedef __sigset_t sigset_t; 
# 54 "/usr/include/sys/select.h" 3
typedef long __fd_mask; 
# 75 "/usr/include/sys/select.h" 3
typedef 
# 65
struct { 
# 69
__fd_mask fds_bits[1024 / (8 * ((int)sizeof(__fd_mask)))]; 
# 75
} fd_set; 
# 82
typedef __fd_mask fd_mask; 
# 96 "/usr/include/sys/select.h" 3
extern "C" {
# 106 "/usr/include/sys/select.h" 3
extern int select(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, timeval *__restrict__ __timeout); 
# 118 "/usr/include/sys/select.h" 3
extern int pselect(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, const timespec *__restrict__ __timeout, const __sigset_t *__restrict__ __sigmask); 
# 131 "/usr/include/sys/select.h" 3
}
# 29 "/usr/include/sys/sysmacros.h" 3
extern "C" {
# 32
__extension__ extern unsigned gnu_dev_major(unsigned long long __dev) throw()
# 33
 __attribute((const)); 
# 35
__extension__ extern unsigned gnu_dev_minor(unsigned long long __dev) throw()
# 36
 __attribute((const)); 
# 38
__extension__ extern unsigned long long gnu_dev_makedev(unsigned __major, unsigned __minor) throw()
# 40
 __attribute((const)); 
# 63 "/usr/include/sys/sysmacros.h" 3
}
# 228 "/usr/include/sys/types.h" 3
typedef __blksize_t blksize_t; 
# 235
typedef __blkcnt_t blkcnt_t; 
# 239
typedef __fsblkcnt_t fsblkcnt_t; 
# 243
typedef __fsfilcnt_t fsfilcnt_t; 
# 262 "/usr/include/sys/types.h" 3
typedef __blkcnt64_t blkcnt64_t; 
# 263
typedef __fsblkcnt64_t fsblkcnt64_t; 
# 264
typedef __fsfilcnt64_t fsfilcnt64_t; 
# 60 "/usr/include/bits/pthreadtypes.h" 3
typedef unsigned long pthread_t; 
# 63
union pthread_attr_t { 
# 65
char __size[56]; 
# 66
long __align; 
# 67
}; 
# 69
typedef pthread_attr_t pthread_attr_t; 
# 79
typedef 
# 75
struct __pthread_internal_list { 
# 77
__pthread_internal_list *__prev; 
# 78
__pthread_internal_list *__next; 
# 79
} __pthread_list_t; 
# 128 "/usr/include/bits/pthreadtypes.h" 3
typedef 
# 91 "/usr/include/bits/pthreadtypes.h" 3
union { 
# 92
struct __pthread_mutex_s { 
# 94
int __lock; 
# 95
unsigned __count; 
# 96
int __owner; 
# 98
unsigned __nusers; 
# 102
int __kind; 
# 104
short __spins; 
# 105
short __elision; 
# 106
__pthread_list_t __list; 
# 125 "/usr/include/bits/pthreadtypes.h" 3
} __data; 
# 126
char __size[40]; 
# 127
long __align; 
# 128
} pthread_mutex_t; 
# 134
typedef 
# 131
union { 
# 132
char __size[4]; 
# 133
int __align; 
# 134
} pthread_mutexattr_t; 
# 154
typedef 
# 140
union { 
# 142
struct { 
# 143
int __lock; 
# 144
unsigned __futex; 
# 145
__extension__ unsigned long long __total_seq; 
# 146
__extension__ unsigned long long __wakeup_seq; 
# 147
__extension__ unsigned long long __woken_seq; 
# 148
void *__mutex; 
# 149
unsigned __nwaiters; 
# 150
unsigned __broadcast_seq; 
# 151
} __data; 
# 152
char __size[48]; 
# 153
__extension__ long long __align; 
# 154
} pthread_cond_t; 
# 160
typedef 
# 157
union { 
# 158
char __size[4]; 
# 159
int __align; 
# 160
} pthread_condattr_t; 
# 164
typedef unsigned pthread_key_t; 
# 168
typedef int pthread_once_t; 
# 214 "/usr/include/bits/pthreadtypes.h" 3
typedef 
# 175 "/usr/include/bits/pthreadtypes.h" 3
union { 
# 178
struct { 
# 179
int __lock; 
# 180
unsigned __nr_readers; 
# 181
unsigned __readers_wakeup; 
# 182
unsigned __writer_wakeup; 
# 183
unsigned __nr_readers_queued; 
# 184
unsigned __nr_writers_queued; 
# 185
int __writer; 
# 186
int __shared; 
# 187
unsigned long __pad1; 
# 188
unsigned long __pad2; 
# 191
unsigned __flags; 
# 193
} __data; 
# 212 "/usr/include/bits/pthreadtypes.h" 3
char __size[56]; 
# 213
long __align; 
# 214
} pthread_rwlock_t; 
# 220
typedef 
# 217
union { 
# 218
char __size[8]; 
# 219
long __align; 
# 220
} pthread_rwlockattr_t; 
# 226
typedef volatile int pthread_spinlock_t; 
# 235
typedef 
# 232
union { 
# 233
char __size[32]; 
# 234
long __align; 
# 235
} pthread_barrier_t; 
# 241
typedef 
# 238
union { 
# 239
char __size[4]; 
# 240
int __align; 
# 241
} pthread_barrierattr_t; 
# 273 "/usr/include/sys/types.h" 3
}
# 321 "/usr/include/stdlib.h" 3
extern long random() throw(); 
# 324
extern void srandom(unsigned __seed) throw(); 
# 330
extern char *initstate(unsigned __seed, char * __statebuf, size_t __statelen) throw()
# 331
 __attribute((__nonnull__(2))); 
# 335
extern char *setstate(char * __statebuf) throw() __attribute((__nonnull__(1))); 
# 343
struct random_data { 
# 345
int32_t *fptr; 
# 346
int32_t *rptr; 
# 347
int32_t *state; 
# 348
int rand_type; 
# 349
int rand_deg; 
# 350
int rand_sep; 
# 351
int32_t *end_ptr; 
# 352
}; 
# 354
extern int random_r(random_data *__restrict__ __buf, int32_t *__restrict__ __result) throw()
# 355
 __attribute((__nonnull__(1, 2))); 
# 357
extern int srandom_r(unsigned __seed, random_data * __buf) throw()
# 358
 __attribute((__nonnull__(2))); 
# 360
extern int initstate_r(unsigned __seed, char *__restrict__ __statebuf, size_t __statelen, random_data *__restrict__ __buf) throw()
# 363
 __attribute((__nonnull__(2, 4))); 
# 365
extern int setstate_r(char *__restrict__ __statebuf, random_data *__restrict__ __buf) throw()
# 367
 __attribute((__nonnull__(1, 2))); 
# 374
extern int rand() throw(); 
# 376
extern void srand(unsigned __seed) throw(); 
# 381
extern int rand_r(unsigned * __seed) throw(); 
# 389
extern double drand48() throw(); 
# 390
extern double erand48(unsigned short  __xsubi[3]) throw() __attribute((__nonnull__(1))); 
# 393
extern long lrand48() throw(); 
# 394
extern long nrand48(unsigned short  __xsubi[3]) throw()
# 395
 __attribute((__nonnull__(1))); 
# 398
extern long mrand48() throw(); 
# 399
extern long jrand48(unsigned short  __xsubi[3]) throw()
# 400
 __attribute((__nonnull__(1))); 
# 403
extern void srand48(long __seedval) throw(); 
# 404
extern unsigned short *seed48(unsigned short  __seed16v[3]) throw()
# 405
 __attribute((__nonnull__(1))); 
# 406
extern void lcong48(unsigned short  __param[7]) throw() __attribute((__nonnull__(1))); 
# 412
struct drand48_data { 
# 414
unsigned short __x[3]; 
# 415
unsigned short __old_x[3]; 
# 416
unsigned short __c; 
# 417
unsigned short __init; 
# 418
unsigned long long __a; 
# 419
}; 
# 422
extern int drand48_r(drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 423
 __attribute((__nonnull__(1, 2))); 
# 424
extern int erand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 426
 __attribute((__nonnull__(1, 2))); 
# 429
extern int lrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 431
 __attribute((__nonnull__(1, 2))); 
# 432
extern int nrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 435
 __attribute((__nonnull__(1, 2))); 
# 438
extern int mrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 440
 __attribute((__nonnull__(1, 2))); 
# 441
extern int jrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 444
 __attribute((__nonnull__(1, 2))); 
# 447
extern int srand48_r(long __seedval, drand48_data * __buffer) throw()
# 448
 __attribute((__nonnull__(2))); 
# 450
extern int seed48_r(unsigned short  __seed16v[3], drand48_data * __buffer) throw()
# 451
 __attribute((__nonnull__(1, 2))); 
# 453
extern int lcong48_r(unsigned short  __param[7], drand48_data * __buffer) throw()
# 455
 __attribute((__nonnull__(1, 2))); 
# 465
extern void *malloc(size_t __size) throw() __attribute((__malloc__)); 
# 467
extern void *calloc(size_t __nmemb, size_t __size) throw()
# 468
 __attribute((__malloc__)); 
# 479
extern void *realloc(void * __ptr, size_t __size) throw()
# 480
 __attribute((__warn_unused_result__)); 
# 482
extern void free(void * __ptr) throw(); 
# 487
extern void cfree(void * __ptr) throw(); 
# 26 "/usr/include/alloca.h" 3
extern "C" {
# 32
extern void *alloca(size_t __size) throw(); 
# 38
}
# 497 "/usr/include/stdlib.h" 3
extern void *valloc(size_t __size) throw() __attribute((__malloc__)); 
# 502
extern int posix_memalign(void ** __memptr, size_t __alignment, size_t __size) throw()
# 503
 __attribute((__nonnull__(1))); 
# 508
extern void *aligned_alloc(size_t __alignment, size_t __size) throw()
# 509
 __attribute((__malloc__, __alloc_size__(2))); 
# 514
extern void abort() throw() __attribute((__noreturn__)); 
# 518
extern int atexit(void (* __func)(void)) throw() __attribute((__nonnull__(1))); 
# 523
extern "C++" int at_quick_exit(void (* __func)(void)) throw() __asm__("at_quick_exit")
# 524
 __attribute((__nonnull__(1))); 
# 534
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg) throw()
# 535
 __attribute((__nonnull__(1))); 
# 542
extern void exit(int __status) throw() __attribute((__noreturn__)); 
# 548
extern void quick_exit(int __status) throw() __attribute((__noreturn__)); 
# 556
extern void _Exit(int __status) throw() __attribute((__noreturn__)); 
# 563
extern char *getenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 569
extern char *secure_getenv(const char * __name) throw()
# 570
 __attribute((__nonnull__(1))); 
# 577
extern int putenv(char * __string) throw() __attribute((__nonnull__(1))); 
# 583
extern int setenv(const char * __name, const char * __value, int __replace) throw()
# 584
 __attribute((__nonnull__(2))); 
# 587
extern int unsetenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 594
extern int clearenv() throw(); 
# 605 "/usr/include/stdlib.h" 3
extern char *mktemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 619 "/usr/include/stdlib.h" 3
extern int mkstemp(char * __template) __attribute((__nonnull__(1))); 
# 629 "/usr/include/stdlib.h" 3
extern int mkstemp64(char * __template) __attribute((__nonnull__(1))); 
# 641 "/usr/include/stdlib.h" 3
extern int mkstemps(char * __template, int __suffixlen) __attribute((__nonnull__(1))); 
# 651 "/usr/include/stdlib.h" 3
extern int mkstemps64(char * __template, int __suffixlen)
# 652
 __attribute((__nonnull__(1))); 
# 662 "/usr/include/stdlib.h" 3
extern char *mkdtemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 673 "/usr/include/stdlib.h" 3
extern int mkostemp(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 683 "/usr/include/stdlib.h" 3
extern int mkostemp64(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 693 "/usr/include/stdlib.h" 3
extern int mkostemps(char * __template, int __suffixlen, int __flags)
# 694
 __attribute((__nonnull__(1))); 
# 705 "/usr/include/stdlib.h" 3
extern int mkostemps64(char * __template, int __suffixlen, int __flags)
# 706
 __attribute((__nonnull__(1))); 
# 716
extern int system(const char * __command); 
# 723
extern char *canonicalize_file_name(const char * __name) throw()
# 724
 __attribute((__nonnull__(1))); 
# 733 "/usr/include/stdlib.h" 3
extern char *realpath(const char *__restrict__ __name, char *__restrict__ __resolved) throw(); 
# 741
typedef int (*__compar_fn_t)(const void *, const void *); 
# 744
typedef __compar_fn_t comparison_fn_t; 
# 748
typedef int (*__compar_d_fn_t)(const void *, const void *, void *); 
# 754
extern void *bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 756
 __attribute((__nonnull__(1, 2, 5))); 
# 760
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 761
 __attribute((__nonnull__(1, 4))); 
# 763
extern void qsort_r(void * __base, size_t __nmemb, size_t __size, __compar_d_fn_t __compar, void * __arg)
# 765
 __attribute((__nonnull__(1, 4))); 
# 770
extern int abs(int __x) throw() __attribute((const)); 
# 771
extern long labs(long __x) throw() __attribute((const)); 
# 775
__extension__ extern long long llabs(long long __x) throw()
# 776
 __attribute((const)); 
# 784
extern div_t div(int __numer, int __denom) throw()
# 785
 __attribute((const)); 
# 786
extern ldiv_t ldiv(long __numer, long __denom) throw()
# 787
 __attribute((const)); 
# 792
__extension__ extern lldiv_t lldiv(long long __numer, long long __denom) throw()
# 794
 __attribute((const)); 
# 807 "/usr/include/stdlib.h" 3
extern char *ecvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 808
 __attribute((__nonnull__(3, 4))); 
# 813
extern char *fcvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 814
 __attribute((__nonnull__(3, 4))); 
# 819
extern char *gcvt(double __value, int __ndigit, char * __buf) throw()
# 820
 __attribute((__nonnull__(3))); 
# 825
extern char *qecvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 827
 __attribute((__nonnull__(3, 4))); 
# 828
extern char *qfcvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 830
 __attribute((__nonnull__(3, 4))); 
# 831
extern char *qgcvt(long double __value, int __ndigit, char * __buf) throw()
# 832
 __attribute((__nonnull__(3))); 
# 837
extern int ecvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 839
 __attribute((__nonnull__(3, 4, 5))); 
# 840
extern int fcvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 842
 __attribute((__nonnull__(3, 4, 5))); 
# 844
extern int qecvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 847
 __attribute((__nonnull__(3, 4, 5))); 
# 848
extern int qfcvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 851
 __attribute((__nonnull__(3, 4, 5))); 
# 859
extern int mblen(const char * __s, size_t __n) throw(); 
# 862
extern int mbtowc(wchar_t *__restrict__ __pwc, const char *__restrict__ __s, size_t __n) throw(); 
# 866
extern int wctomb(char * __s, wchar_t __wchar) throw(); 
# 870
extern size_t mbstowcs(wchar_t *__restrict__ __pwcs, const char *__restrict__ __s, size_t __n) throw(); 
# 873
extern size_t wcstombs(char *__restrict__ __s, const wchar_t *__restrict__ __pwcs, size_t __n) throw(); 
# 884
extern int rpmatch(const char * __response) throw() __attribute((__nonnull__(1))); 
# 895 "/usr/include/stdlib.h" 3
extern int getsubopt(char **__restrict__ __optionp, char *const *__restrict__ __tokens, char **__restrict__ __valuep) throw()
# 898
 __attribute((__nonnull__(1, 2, 3))); 
# 904
extern void setkey(const char * __key) throw() __attribute((__nonnull__(1))); 
# 912
extern int posix_openpt(int __oflag); 
# 920
extern int grantpt(int __fd) throw(); 
# 924
extern int unlockpt(int __fd) throw(); 
# 929
extern char *ptsname(int __fd) throw(); 
# 936
extern int ptsname_r(int __fd, char * __buf, size_t __buflen) throw()
# 937
 __attribute((__nonnull__(2))); 
# 940
extern int getpt(); 
# 947
extern int getloadavg(double  __loadavg[], int __nelem) throw()
# 948
 __attribute((__nonnull__(1))); 
# 964 "/usr/include/stdlib.h" 3
}
# 46 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/std_abs.h" 3
extern "C++" {
# 48
namespace std __attribute((__visibility__("default"))) { 
# 52
using ::abs;
# 56
inline long abs(long __i) { return __builtin_labs(__i); } 
# 61
inline long long abs(long long __x) { return __builtin_llabs(__x); } 
# 71 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/std_abs.h" 3
constexpr double abs(double __x) 
# 72
{ return __builtin_fabs(__x); } 
# 75
constexpr float abs(float __x) 
# 76
{ return __builtin_fabsf(__x); } 
# 79
constexpr long double abs(long double __x) 
# 80
{ return __builtin_fabsl(__x); } 
# 85
constexpr __int128 abs(__int128 __x) { return (__x >= (0)) ? __x : (-__x); } 
# 103 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/std_abs.h" 3
constexpr __float128 abs(__float128 __x) 
# 104
{ return (__x < (0)) ? -__x : __x; } 
# 108
}
# 109
}
# 77 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/cmath" 3
extern "C++" {
# 79
namespace std __attribute((__visibility__("default"))) { 
# 83
using ::acos;
# 87
constexpr float acos(float __x) 
# 88
{ return __builtin_acosf(__x); } 
# 91
constexpr long double acos(long double __x) 
# 92
{ return __builtin_acosl(__x); } 
# 95
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 99
acos(_Tp __x) 
# 100
{ return __builtin_acos(__x); } 
# 102
using ::asin;
# 106
constexpr float asin(float __x) 
# 107
{ return __builtin_asinf(__x); } 
# 110
constexpr long double asin(long double __x) 
# 111
{ return __builtin_asinl(__x); } 
# 114
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 118
asin(_Tp __x) 
# 119
{ return __builtin_asin(__x); } 
# 121
using ::atan;
# 125
constexpr float atan(float __x) 
# 126
{ return __builtin_atanf(__x); } 
# 129
constexpr long double atan(long double __x) 
# 130
{ return __builtin_atanl(__x); } 
# 133
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 137
atan(_Tp __x) 
# 138
{ return __builtin_atan(__x); } 
# 140
using ::atan2;
# 144
constexpr float atan2(float __y, float __x) 
# 145
{ return __builtin_atan2f(__y, __x); } 
# 148
constexpr long double atan2(long double __y, long double __x) 
# 149
{ return __builtin_atan2l(__y, __x); } 
# 152
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 155
atan2(_Tp __y, _Up __x) 
# 156
{ 
# 157
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 158
return atan2((__type)__y, (__type)__x); 
# 159
} 
# 161
using ::ceil;
# 165
constexpr float ceil(float __x) 
# 166
{ return __builtin_ceilf(__x); } 
# 169
constexpr long double ceil(long double __x) 
# 170
{ return __builtin_ceill(__x); } 
# 173
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 177
ceil(_Tp __x) 
# 178
{ return __builtin_ceil(__x); } 
# 180
using ::cos;
# 184
constexpr float cos(float __x) 
# 185
{ return __builtin_cosf(__x); } 
# 188
constexpr long double cos(long double __x) 
# 189
{ return __builtin_cosl(__x); } 
# 192
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 196
cos(_Tp __x) 
# 197
{ return __builtin_cos(__x); } 
# 199
using ::cosh;
# 203
constexpr float cosh(float __x) 
# 204
{ return __builtin_coshf(__x); } 
# 207
constexpr long double cosh(long double __x) 
# 208
{ return __builtin_coshl(__x); } 
# 211
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 215
cosh(_Tp __x) 
# 216
{ return __builtin_cosh(__x); } 
# 218
using ::exp;
# 222
constexpr float exp(float __x) 
# 223
{ return __builtin_expf(__x); } 
# 226
constexpr long double exp(long double __x) 
# 227
{ return __builtin_expl(__x); } 
# 230
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 234
exp(_Tp __x) 
# 235
{ return __builtin_exp(__x); } 
# 237
using ::fabs;
# 241
constexpr float fabs(float __x) 
# 242
{ return __builtin_fabsf(__x); } 
# 245
constexpr long double fabs(long double __x) 
# 246
{ return __builtin_fabsl(__x); } 
# 249
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 253
fabs(_Tp __x) 
# 254
{ return __builtin_fabs(__x); } 
# 256
using ::floor;
# 260
constexpr float floor(float __x) 
# 261
{ return __builtin_floorf(__x); } 
# 264
constexpr long double floor(long double __x) 
# 265
{ return __builtin_floorl(__x); } 
# 268
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 272
floor(_Tp __x) 
# 273
{ return __builtin_floor(__x); } 
# 275
using ::fmod;
# 279
constexpr float fmod(float __x, float __y) 
# 280
{ return __builtin_fmodf(__x, __y); } 
# 283
constexpr long double fmod(long double __x, long double __y) 
# 284
{ return __builtin_fmodl(__x, __y); } 
# 287
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 290
fmod(_Tp __x, _Up __y) 
# 291
{ 
# 292
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 293
return fmod((__type)__x, (__type)__y); 
# 294
} 
# 296
using ::frexp;
# 300
inline float frexp(float __x, int *__exp) 
# 301
{ return __builtin_frexpf(__x, __exp); } 
# 304
inline long double frexp(long double __x, int *__exp) 
# 305
{ return __builtin_frexpl(__x, __exp); } 
# 308
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 312
frexp(_Tp __x, int *__exp) 
# 313
{ return __builtin_frexp(__x, __exp); } 
# 315
using ::ldexp;
# 319
constexpr float ldexp(float __x, int __exp) 
# 320
{ return __builtin_ldexpf(__x, __exp); } 
# 323
constexpr long double ldexp(long double __x, int __exp) 
# 324
{ return __builtin_ldexpl(__x, __exp); } 
# 327
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 331
ldexp(_Tp __x, int __exp) 
# 332
{ return __builtin_ldexp(__x, __exp); } 
# 334
using ::log;
# 338
constexpr float log(float __x) 
# 339
{ return __builtin_logf(__x); } 
# 342
constexpr long double log(long double __x) 
# 343
{ return __builtin_logl(__x); } 
# 346
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 350
log(_Tp __x) 
# 351
{ return __builtin_log(__x); } 
# 353
using ::log10;
# 357
constexpr float log10(float __x) 
# 358
{ return __builtin_log10f(__x); } 
# 361
constexpr long double log10(long double __x) 
# 362
{ return __builtin_log10l(__x); } 
# 365
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 369
log10(_Tp __x) 
# 370
{ return __builtin_log10(__x); } 
# 372
using ::modf;
# 376
inline float modf(float __x, float *__iptr) 
# 377
{ return __builtin_modff(__x, __iptr); } 
# 380
inline long double modf(long double __x, long double *__iptr) 
# 381
{ return __builtin_modfl(__x, __iptr); } 
# 384
using ::pow;
# 388
constexpr float pow(float __x, float __y) 
# 389
{ return __builtin_powf(__x, __y); } 
# 392
constexpr long double pow(long double __x, long double __y) 
# 393
{ return __builtin_powl(__x, __y); } 
# 412 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/cmath" 3
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 415
pow(_Tp __x, _Up __y) 
# 416
{ 
# 417
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 418
return pow((__type)__x, (__type)__y); 
# 419
} 
# 421
using ::sin;
# 425
constexpr float sin(float __x) 
# 426
{ return __builtin_sinf(__x); } 
# 429
constexpr long double sin(long double __x) 
# 430
{ return __builtin_sinl(__x); } 
# 433
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 437
sin(_Tp __x) 
# 438
{ return __builtin_sin(__x); } 
# 440
using ::sinh;
# 444
constexpr float sinh(float __x) 
# 445
{ return __builtin_sinhf(__x); } 
# 448
constexpr long double sinh(long double __x) 
# 449
{ return __builtin_sinhl(__x); } 
# 452
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 456
sinh(_Tp __x) 
# 457
{ return __builtin_sinh(__x); } 
# 459
using ::sqrt;
# 463
constexpr float sqrt(float __x) 
# 464
{ return __builtin_sqrtf(__x); } 
# 467
constexpr long double sqrt(long double __x) 
# 468
{ return __builtin_sqrtl(__x); } 
# 471
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 475
sqrt(_Tp __x) 
# 476
{ return __builtin_sqrt(__x); } 
# 478
using ::tan;
# 482
constexpr float tan(float __x) 
# 483
{ return __builtin_tanf(__x); } 
# 486
constexpr long double tan(long double __x) 
# 487
{ return __builtin_tanl(__x); } 
# 490
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 494
tan(_Tp __x) 
# 495
{ return __builtin_tan(__x); } 
# 497
using ::tanh;
# 501
constexpr float tanh(float __x) 
# 502
{ return __builtin_tanhf(__x); } 
# 505
constexpr long double tanh(long double __x) 
# 506
{ return __builtin_tanhl(__x); } 
# 509
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 513
tanh(_Tp __x) 
# 514
{ return __builtin_tanh(__x); } 
# 537 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/cmath" 3
constexpr int fpclassify(float __x) 
# 538
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 539
} 
# 542
constexpr int fpclassify(double __x) 
# 543
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 544
} 
# 547
constexpr int fpclassify(long double __x) 
# 548
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 549
} 
# 553
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 556
fpclassify(_Tp __x) 
# 557
{ return (__x != 0) ? 4 : 2; } 
# 562
constexpr bool isfinite(float __x) 
# 563
{ return __builtin_isfinite(__x); } 
# 566
constexpr bool isfinite(double __x) 
# 567
{ return __builtin_isfinite(__x); } 
# 570
constexpr bool isfinite(long double __x) 
# 571
{ return __builtin_isfinite(__x); } 
# 575
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 578
isfinite(_Tp __x) 
# 579
{ return true; } 
# 584
constexpr bool isinf(float __x) 
# 585
{ return __builtin_isinf(__x); } 
# 589
using ::isinf;
# 597
constexpr bool isinf(long double __x) 
# 598
{ return __builtin_isinf(__x); } 
# 602
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 605
isinf(_Tp __x) 
# 606
{ return false; } 
# 611
constexpr bool isnan(float __x) 
# 612
{ return __builtin_isnan(__x); } 
# 616
using ::isnan;
# 624
constexpr bool isnan(long double __x) 
# 625
{ return __builtin_isnan(__x); } 
# 629
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 632
isnan(_Tp __x) 
# 633
{ return false; } 
# 638
constexpr bool isnormal(float __x) 
# 639
{ return __builtin_isnormal(__x); } 
# 642
constexpr bool isnormal(double __x) 
# 643
{ return __builtin_isnormal(__x); } 
# 646
constexpr bool isnormal(long double __x) 
# 647
{ return __builtin_isnormal(__x); } 
# 651
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 654
isnormal(_Tp __x) 
# 655
{ return (__x != 0) ? true : false; } 
# 661
constexpr bool signbit(float __x) 
# 662
{ return __builtin_signbit(__x); } 
# 665
constexpr bool signbit(double __x) 
# 666
{ return __builtin_signbit(__x); } 
# 669
constexpr bool signbit(long double __x) 
# 670
{ return __builtin_signbit(__x); } 
# 674
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 677
signbit(_Tp __x) 
# 678
{ return (__x < 0) ? true : false; } 
# 683
constexpr bool isgreater(float __x, float __y) 
# 684
{ return __builtin_isgreater(__x, __y); } 
# 687
constexpr bool isgreater(double __x, double __y) 
# 688
{ return __builtin_isgreater(__x, __y); } 
# 691
constexpr bool isgreater(long double __x, long double __y) 
# 692
{ return __builtin_isgreater(__x, __y); } 
# 696
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 700
isgreater(_Tp __x, _Up __y) 
# 701
{ 
# 702
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 703
return __builtin_isgreater((__type)__x, (__type)__y); 
# 704
} 
# 709
constexpr bool isgreaterequal(float __x, float __y) 
# 710
{ return __builtin_isgreaterequal(__x, __y); } 
# 713
constexpr bool isgreaterequal(double __x, double __y) 
# 714
{ return __builtin_isgreaterequal(__x, __y); } 
# 717
constexpr bool isgreaterequal(long double __x, long double __y) 
# 718
{ return __builtin_isgreaterequal(__x, __y); } 
# 722
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 726
isgreaterequal(_Tp __x, _Up __y) 
# 727
{ 
# 728
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 729
return __builtin_isgreaterequal((__type)__x, (__type)__y); 
# 730
} 
# 735
constexpr bool isless(float __x, float __y) 
# 736
{ return __builtin_isless(__x, __y); } 
# 739
constexpr bool isless(double __x, double __y) 
# 740
{ return __builtin_isless(__x, __y); } 
# 743
constexpr bool isless(long double __x, long double __y) 
# 744
{ return __builtin_isless(__x, __y); } 
# 748
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 752
isless(_Tp __x, _Up __y) 
# 753
{ 
# 754
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 755
return __builtin_isless((__type)__x, (__type)__y); 
# 756
} 
# 761
constexpr bool islessequal(float __x, float __y) 
# 762
{ return __builtin_islessequal(__x, __y); } 
# 765
constexpr bool islessequal(double __x, double __y) 
# 766
{ return __builtin_islessequal(__x, __y); } 
# 769
constexpr bool islessequal(long double __x, long double __y) 
# 770
{ return __builtin_islessequal(__x, __y); } 
# 774
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 778
islessequal(_Tp __x, _Up __y) 
# 779
{ 
# 780
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 781
return __builtin_islessequal((__type)__x, (__type)__y); 
# 782
} 
# 787
constexpr bool islessgreater(float __x, float __y) 
# 788
{ return __builtin_islessgreater(__x, __y); } 
# 791
constexpr bool islessgreater(double __x, double __y) 
# 792
{ return __builtin_islessgreater(__x, __y); } 
# 795
constexpr bool islessgreater(long double __x, long double __y) 
# 796
{ return __builtin_islessgreater(__x, __y); } 
# 800
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 804
islessgreater(_Tp __x, _Up __y) 
# 805
{ 
# 806
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 807
return __builtin_islessgreater((__type)__x, (__type)__y); 
# 808
} 
# 813
constexpr bool isunordered(float __x, float __y) 
# 814
{ return __builtin_isunordered(__x, __y); } 
# 817
constexpr bool isunordered(double __x, double __y) 
# 818
{ return __builtin_isunordered(__x, __y); } 
# 821
constexpr bool isunordered(long double __x, long double __y) 
# 822
{ return __builtin_isunordered(__x, __y); } 
# 826
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 830
isunordered(_Tp __x, _Up __y) 
# 831
{ 
# 832
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 833
return __builtin_isunordered((__type)__x, (__type)__y); 
# 834
} 
# 1065 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/cmath" 3
using ::double_t;
# 1066
using ::float_t;
# 1069
using ::acosh;
# 1070
using ::acoshf;
# 1071
using ::acoshl;
# 1073
using ::asinh;
# 1074
using ::asinhf;
# 1075
using ::asinhl;
# 1077
using ::atanh;
# 1078
using ::atanhf;
# 1079
using ::atanhl;
# 1081
using ::cbrt;
# 1082
using ::cbrtf;
# 1083
using ::cbrtl;
# 1085
using ::copysign;
# 1086
using ::copysignf;
# 1087
using ::copysignl;
# 1089
using ::erf;
# 1090
using ::erff;
# 1091
using ::erfl;
# 1093
using ::erfc;
# 1094
using ::erfcf;
# 1095
using ::erfcl;
# 1097
using ::exp2;
# 1098
using ::exp2f;
# 1099
using ::exp2l;
# 1101
using ::expm1;
# 1102
using ::expm1f;
# 1103
using ::expm1l;
# 1105
using ::fdim;
# 1106
using ::fdimf;
# 1107
using ::fdiml;
# 1109
using ::fma;
# 1110
using ::fmaf;
# 1111
using ::fmal;
# 1113
using ::fmax;
# 1114
using ::fmaxf;
# 1115
using ::fmaxl;
# 1117
using ::fmin;
# 1118
using ::fminf;
# 1119
using ::fminl;
# 1121
using ::hypot;
# 1122
using ::hypotf;
# 1123
using ::hypotl;
# 1125
using ::ilogb;
# 1126
using ::ilogbf;
# 1127
using ::ilogbl;
# 1129
using ::lgamma;
# 1130
using ::lgammaf;
# 1131
using ::lgammal;
# 1134
using ::llrint;
# 1135
using ::llrintf;
# 1136
using ::llrintl;
# 1138
using ::llround;
# 1139
using ::llroundf;
# 1140
using ::llroundl;
# 1143
using ::log1p;
# 1144
using ::log1pf;
# 1145
using ::log1pl;
# 1147
using ::log2;
# 1148
using ::log2f;
# 1149
using ::log2l;
# 1151
using ::logb;
# 1152
using ::logbf;
# 1153
using ::logbl;
# 1155
using ::lrint;
# 1156
using ::lrintf;
# 1157
using ::lrintl;
# 1159
using ::lround;
# 1160
using ::lroundf;
# 1161
using ::lroundl;
# 1163
using ::nan;
# 1164
using ::nanf;
# 1165
using ::nanl;
# 1167
using ::nearbyint;
# 1168
using ::nearbyintf;
# 1169
using ::nearbyintl;
# 1171
using ::nextafter;
# 1172
using ::nextafterf;
# 1173
using ::nextafterl;
# 1175
using ::nexttoward;
# 1176
using ::nexttowardf;
# 1177
using ::nexttowardl;
# 1179
using ::remainder;
# 1180
using ::remainderf;
# 1181
using ::remainderl;
# 1183
using ::remquo;
# 1184
using ::remquof;
# 1185
using ::remquol;
# 1187
using ::rint;
# 1188
using ::rintf;
# 1189
using ::rintl;
# 1191
using ::round;
# 1192
using ::roundf;
# 1193
using ::roundl;
# 1195
using ::scalbln;
# 1196
using ::scalblnf;
# 1197
using ::scalblnl;
# 1199
using ::scalbn;
# 1200
using ::scalbnf;
# 1201
using ::scalbnl;
# 1203
using ::tgamma;
# 1204
using ::tgammaf;
# 1205
using ::tgammal;
# 1207
using ::trunc;
# 1208
using ::truncf;
# 1209
using ::truncl;
# 1214
constexpr float acosh(float __x) 
# 1215
{ return __builtin_acoshf(__x); } 
# 1218
constexpr long double acosh(long double __x) 
# 1219
{ return __builtin_acoshl(__x); } 
# 1223
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1226
acosh(_Tp __x) 
# 1227
{ return __builtin_acosh(__x); } 
# 1232
constexpr float asinh(float __x) 
# 1233
{ return __builtin_asinhf(__x); } 
# 1236
constexpr long double asinh(long double __x) 
# 1237
{ return __builtin_asinhl(__x); } 
# 1241
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1244
asinh(_Tp __x) 
# 1245
{ return __builtin_asinh(__x); } 
# 1250
constexpr float atanh(float __x) 
# 1251
{ return __builtin_atanhf(__x); } 
# 1254
constexpr long double atanh(long double __x) 
# 1255
{ return __builtin_atanhl(__x); } 
# 1259
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1262
atanh(_Tp __x) 
# 1263
{ return __builtin_atanh(__x); } 
# 1268
constexpr float cbrt(float __x) 
# 1269
{ return __builtin_cbrtf(__x); } 
# 1272
constexpr long double cbrt(long double __x) 
# 1273
{ return __builtin_cbrtl(__x); } 
# 1277
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1280
cbrt(_Tp __x) 
# 1281
{ return __builtin_cbrt(__x); } 
# 1286
constexpr float copysign(float __x, float __y) 
# 1287
{ return __builtin_copysignf(__x, __y); } 
# 1290
constexpr long double copysign(long double __x, long double __y) 
# 1291
{ return __builtin_copysignl(__x, __y); } 
# 1295
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1297
copysign(_Tp __x, _Up __y) 
# 1298
{ 
# 1299
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1300
return copysign((__type)__x, (__type)__y); 
# 1301
} 
# 1306
constexpr float erf(float __x) 
# 1307
{ return __builtin_erff(__x); } 
# 1310
constexpr long double erf(long double __x) 
# 1311
{ return __builtin_erfl(__x); } 
# 1315
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1318
erf(_Tp __x) 
# 1319
{ return __builtin_erf(__x); } 
# 1324
constexpr float erfc(float __x) 
# 1325
{ return __builtin_erfcf(__x); } 
# 1328
constexpr long double erfc(long double __x) 
# 1329
{ return __builtin_erfcl(__x); } 
# 1333
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1336
erfc(_Tp __x) 
# 1337
{ return __builtin_erfc(__x); } 
# 1342
constexpr float exp2(float __x) 
# 1343
{ return __builtin_exp2f(__x); } 
# 1346
constexpr long double exp2(long double __x) 
# 1347
{ return __builtin_exp2l(__x); } 
# 1351
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1354
exp2(_Tp __x) 
# 1355
{ return __builtin_exp2(__x); } 
# 1360
constexpr float expm1(float __x) 
# 1361
{ return __builtin_expm1f(__x); } 
# 1364
constexpr long double expm1(long double __x) 
# 1365
{ return __builtin_expm1l(__x); } 
# 1369
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1372
expm1(_Tp __x) 
# 1373
{ return __builtin_expm1(__x); } 
# 1378
constexpr float fdim(float __x, float __y) 
# 1379
{ return __builtin_fdimf(__x, __y); } 
# 1382
constexpr long double fdim(long double __x, long double __y) 
# 1383
{ return __builtin_fdiml(__x, __y); } 
# 1387
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1389
fdim(_Tp __x, _Up __y) 
# 1390
{ 
# 1391
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1392
return fdim((__type)__x, (__type)__y); 
# 1393
} 
# 1398
constexpr float fma(float __x, float __y, float __z) 
# 1399
{ return __builtin_fmaf(__x, __y, __z); } 
# 1402
constexpr long double fma(long double __x, long double __y, long double __z) 
# 1403
{ return __builtin_fmal(__x, __y, __z); } 
# 1407
template< class _Tp, class _Up, class _Vp> constexpr typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type 
# 1409
fma(_Tp __x, _Up __y, _Vp __z) 
# 1410
{ 
# 1411
typedef typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type __type; 
# 1412
return fma((__type)__x, (__type)__y, (__type)__z); 
# 1413
} 
# 1418
constexpr float fmax(float __x, float __y) 
# 1419
{ return __builtin_fmaxf(__x, __y); } 
# 1422
constexpr long double fmax(long double __x, long double __y) 
# 1423
{ return __builtin_fmaxl(__x, __y); } 
# 1427
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1429
fmax(_Tp __x, _Up __y) 
# 1430
{ 
# 1431
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1432
return fmax((__type)__x, (__type)__y); 
# 1433
} 
# 1438
constexpr float fmin(float __x, float __y) 
# 1439
{ return __builtin_fminf(__x, __y); } 
# 1442
constexpr long double fmin(long double __x, long double __y) 
# 1443
{ return __builtin_fminl(__x, __y); } 
# 1447
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1449
fmin(_Tp __x, _Up __y) 
# 1450
{ 
# 1451
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1452
return fmin((__type)__x, (__type)__y); 
# 1453
} 
# 1458
constexpr float hypot(float __x, float __y) 
# 1459
{ return __builtin_hypotf(__x, __y); } 
# 1462
constexpr long double hypot(long double __x, long double __y) 
# 1463
{ return __builtin_hypotl(__x, __y); } 
# 1467
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1469
hypot(_Tp __x, _Up __y) 
# 1470
{ 
# 1471
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1472
return hypot((__type)__x, (__type)__y); 
# 1473
} 
# 1478
constexpr int ilogb(float __x) 
# 1479
{ return __builtin_ilogbf(__x); } 
# 1482
constexpr int ilogb(long double __x) 
# 1483
{ return __builtin_ilogbl(__x); } 
# 1487
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 1491
ilogb(_Tp __x) 
# 1492
{ return __builtin_ilogb(__x); } 
# 1497
constexpr float lgamma(float __x) 
# 1498
{ return __builtin_lgammaf(__x); } 
# 1501
constexpr long double lgamma(long double __x) 
# 1502
{ return __builtin_lgammal(__x); } 
# 1506
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1509
lgamma(_Tp __x) 
# 1510
{ return __builtin_lgamma(__x); } 
# 1515
constexpr long long llrint(float __x) 
# 1516
{ return __builtin_llrintf(__x); } 
# 1519
constexpr long long llrint(long double __x) 
# 1520
{ return __builtin_llrintl(__x); } 
# 1524
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 1527
llrint(_Tp __x) 
# 1528
{ return __builtin_llrint(__x); } 
# 1533
constexpr long long llround(float __x) 
# 1534
{ return __builtin_llroundf(__x); } 
# 1537
constexpr long long llround(long double __x) 
# 1538
{ return __builtin_llroundl(__x); } 
# 1542
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 1545
llround(_Tp __x) 
# 1546
{ return __builtin_llround(__x); } 
# 1551
constexpr float log1p(float __x) 
# 1552
{ return __builtin_log1pf(__x); } 
# 1555
constexpr long double log1p(long double __x) 
# 1556
{ return __builtin_log1pl(__x); } 
# 1560
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1563
log1p(_Tp __x) 
# 1564
{ return __builtin_log1p(__x); } 
# 1570
constexpr float log2(float __x) 
# 1571
{ return __builtin_log2f(__x); } 
# 1574
constexpr long double log2(long double __x) 
# 1575
{ return __builtin_log2l(__x); } 
# 1579
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1582
log2(_Tp __x) 
# 1583
{ return __builtin_log2(__x); } 
# 1588
constexpr float logb(float __x) 
# 1589
{ return __builtin_logbf(__x); } 
# 1592
constexpr long double logb(long double __x) 
# 1593
{ return __builtin_logbl(__x); } 
# 1597
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1600
logb(_Tp __x) 
# 1601
{ return __builtin_logb(__x); } 
# 1606
constexpr long lrint(float __x) 
# 1607
{ return __builtin_lrintf(__x); } 
# 1610
constexpr long lrint(long double __x) 
# 1611
{ return __builtin_lrintl(__x); } 
# 1615
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 1618
lrint(_Tp __x) 
# 1619
{ return __builtin_lrint(__x); } 
# 1624
constexpr long lround(float __x) 
# 1625
{ return __builtin_lroundf(__x); } 
# 1628
constexpr long lround(long double __x) 
# 1629
{ return __builtin_lroundl(__x); } 
# 1633
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 1636
lround(_Tp __x) 
# 1637
{ return __builtin_lround(__x); } 
# 1642
constexpr float nearbyint(float __x) 
# 1643
{ return __builtin_nearbyintf(__x); } 
# 1646
constexpr long double nearbyint(long double __x) 
# 1647
{ return __builtin_nearbyintl(__x); } 
# 1651
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1654
nearbyint(_Tp __x) 
# 1655
{ return __builtin_nearbyint(__x); } 
# 1660
constexpr float nextafter(float __x, float __y) 
# 1661
{ return __builtin_nextafterf(__x, __y); } 
# 1664
constexpr long double nextafter(long double __x, long double __y) 
# 1665
{ return __builtin_nextafterl(__x, __y); } 
# 1669
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1671
nextafter(_Tp __x, _Up __y) 
# 1672
{ 
# 1673
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1674
return nextafter((__type)__x, (__type)__y); 
# 1675
} 
# 1680
constexpr float nexttoward(float __x, long double __y) 
# 1681
{ return __builtin_nexttowardf(__x, __y); } 
# 1684
constexpr long double nexttoward(long double __x, long double __y) 
# 1685
{ return __builtin_nexttowardl(__x, __y); } 
# 1689
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1692
nexttoward(_Tp __x, long double __y) 
# 1693
{ return __builtin_nexttoward(__x, __y); } 
# 1698
constexpr float remainder(float __x, float __y) 
# 1699
{ return __builtin_remainderf(__x, __y); } 
# 1702
constexpr long double remainder(long double __x, long double __y) 
# 1703
{ return __builtin_remainderl(__x, __y); } 
# 1707
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1709
remainder(_Tp __x, _Up __y) 
# 1710
{ 
# 1711
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1712
return remainder((__type)__x, (__type)__y); 
# 1713
} 
# 1718
inline float remquo(float __x, float __y, int *__pquo) 
# 1719
{ return __builtin_remquof(__x, __y, __pquo); } 
# 1722
inline long double remquo(long double __x, long double __y, int *__pquo) 
# 1723
{ return __builtin_remquol(__x, __y, __pquo); } 
# 1727
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1729
remquo(_Tp __x, _Up __y, int *__pquo) 
# 1730
{ 
# 1731
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1732
return remquo((__type)__x, (__type)__y, __pquo); 
# 1733
} 
# 1738
constexpr float rint(float __x) 
# 1739
{ return __builtin_rintf(__x); } 
# 1742
constexpr long double rint(long double __x) 
# 1743
{ return __builtin_rintl(__x); } 
# 1747
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1750
rint(_Tp __x) 
# 1751
{ return __builtin_rint(__x); } 
# 1756
constexpr float round(float __x) 
# 1757
{ return __builtin_roundf(__x); } 
# 1760
constexpr long double round(long double __x) 
# 1761
{ return __builtin_roundl(__x); } 
# 1765
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1768
round(_Tp __x) 
# 1769
{ return __builtin_round(__x); } 
# 1774
constexpr float scalbln(float __x, long __ex) 
# 1775
{ return __builtin_scalblnf(__x, __ex); } 
# 1778
constexpr long double scalbln(long double __x, long __ex) 
# 1779
{ return __builtin_scalblnl(__x, __ex); } 
# 1783
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1786
scalbln(_Tp __x, long __ex) 
# 1787
{ return __builtin_scalbln(__x, __ex); } 
# 1792
constexpr float scalbn(float __x, int __ex) 
# 1793
{ return __builtin_scalbnf(__x, __ex); } 
# 1796
constexpr long double scalbn(long double __x, int __ex) 
# 1797
{ return __builtin_scalbnl(__x, __ex); } 
# 1801
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1804
scalbn(_Tp __x, int __ex) 
# 1805
{ return __builtin_scalbn(__x, __ex); } 
# 1810
constexpr float tgamma(float __x) 
# 1811
{ return __builtin_tgammaf(__x); } 
# 1814
constexpr long double tgamma(long double __x) 
# 1815
{ return __builtin_tgammal(__x); } 
# 1819
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1822
tgamma(_Tp __x) 
# 1823
{ return __builtin_tgamma(__x); } 
# 1828
constexpr float trunc(float __x) 
# 1829
{ return __builtin_truncf(__x); } 
# 1832
constexpr long double trunc(long double __x) 
# 1833
{ return __builtin_truncl(__x); } 
# 1837
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1840
trunc(_Tp __x) 
# 1841
{ return __builtin_trunc(__x); } 
# 1852 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/cmath" 3
template< class _Tp> inline _Tp 
# 1854
__hypot3(_Tp __x, _Tp __y, _Tp __z) 
# 1855
{ 
# 1856
__x = std::abs(__x); 
# 1857
__y = std::abs(__y); 
# 1858
__z = std::abs(__z); 
# 1859
if (_Tp __a = (__x < __y) ? (__y < __z) ? __z : __y : ((__x < __z) ? __z : __x)) { 
# 1860
return __a * std::sqrt((((__x / __a) * (__x / __a)) + ((__y / __a) * (__y / __a))) + ((__z / __a) * (__z / __a))); } else { 
# 1864
return {}; }  
# 1865
} 
# 1868
inline float hypot(float __x, float __y, float __z) 
# 1869
{ return std::__hypot3< float> (__x, __y, __z); } 
# 1872
inline double hypot(double __x, double __y, double __z) 
# 1873
{ return std::__hypot3< double> (__x, __y, __z); } 
# 1876
inline long double hypot(long double __x, long double __y, long double __z) 
# 1877
{ return std::__hypot3< long double> (__x, __y, __z); } 
# 1879
template< class _Tp, class _Up, class _Vp> typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type 
# 1881
hypot(_Tp __x, _Up __y, _Vp __z) 
# 1882
{ 
# 1883
using __type = typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type; 
# 1884
return std::__hypot3< typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type> (__x, __y, __z); 
# 1885
} 
# 1924 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/cmath" 3
}
# 33 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
#pragma GCC visibility push ( default )
# 42 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/functexcept.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 48
void __throw_bad_exception() __attribute((__noreturn__)); 
# 52
void __throw_bad_alloc() __attribute((__noreturn__)); 
# 55
void __throw_bad_array_new_length() __attribute((__noreturn__)); 
# 59
void __throw_bad_cast() __attribute((__noreturn__)); 
# 62
void __throw_bad_typeid() __attribute((__noreturn__)); 
# 66
void __throw_logic_error(const char *) __attribute((__noreturn__)); 
# 69
void __throw_domain_error(const char *) __attribute((__noreturn__)); 
# 72
void __throw_invalid_argument(const char *) __attribute((__noreturn__)); 
# 75
void __throw_length_error(const char *) __attribute((__noreturn__)); 
# 78
void __throw_out_of_range(const char *) __attribute((__noreturn__)); 
# 81
void __throw_out_of_range_fmt(const char *, ...) __attribute((__noreturn__))
# 82
 __attribute((__format__(__gnu_printf__, 1, 2))); 
# 85
void __throw_runtime_error(const char *) __attribute((__noreturn__)); 
# 88
void __throw_range_error(const char *) __attribute((__noreturn__)); 
# 91
void __throw_overflow_error(const char *) __attribute((__noreturn__)); 
# 94
void __throw_underflow_error(const char *) __attribute((__noreturn__)); 
# 98
void __throw_ios_failure(const char *) __attribute((__noreturn__)); 
# 101
void __throw_ios_failure(const char *, int) __attribute((__noreturn__)); 
# 105
void __throw_system_error(int) __attribute((__noreturn__)); 
# 109
void __throw_future_error(int) __attribute((__noreturn__)); 
# 113
void __throw_bad_function_call() __attribute((__noreturn__)); 
# 116
}
# 37 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/ext/numeric_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 50 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/ext/numeric_traits.h" 3
template< class _Tp> 
# 51
struct __is_integer_nonstrict : public std::__is_integer< _Tp>  { 
# 54
using std::__is_integer< _Tp> ::__value;
# 57
enum { __width = (__value) ? sizeof(_Tp) * (8) : (0)}; 
# 58
}; 
# 60
template< class _Value> 
# 61
struct __numeric_traits_integer { 
# 64
static_assert((__is_integer_nonstrict< _Value> ::__value), "invalid specialization");
# 70
static const bool __is_signed = (((_Value)(-1)) < 0); 
# 71
static const int __digits = (__is_integer_nonstrict< _Value> ::__width - __is_signed); 
# 75
static const _Value __max = (__is_signed ? (((((_Value)1) << (__digits - 1)) - 1) << 1) + 1 : (~((_Value)0))); 
# 78
static const _Value __min = (__is_signed ? (-__max) - 1 : ((_Value)0)); 
# 79
}; 
# 81
template< class _Value> const _Value __numeric_traits_integer< _Value> ::__min; 
# 84
template< class _Value> const _Value __numeric_traits_integer< _Value> ::__max; 
# 87
template< class _Value> const bool __numeric_traits_integer< _Value> ::__is_signed; 
# 90
template< class _Value> const int __numeric_traits_integer< _Value> ::__digits; 
# 135 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/ext/numeric_traits.h" 3
template< class _Tp> using __int_traits = __numeric_traits_integer< _Tp> ; 
# 155 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/ext/numeric_traits.h" 3
template< class _Value> 
# 156
struct __numeric_traits_floating { 
# 159
static const int __max_digits10 = ((2) + ((((std::__are_same< _Value, float> ::__value) ? 24 : ((std::__are_same< _Value, double> ::__value) ? 53 : 64)) * 643L) / (2136))); 
# 162
static const bool __is_signed = true; 
# 163
static const int __digits10 = ((std::__are_same< _Value, float> ::__value) ? 6 : ((std::__are_same< _Value, double> ::__value) ? 15 : 18)); 
# 164
static const int __max_exponent10 = ((std::__are_same< _Value, float> ::__value) ? 38 : ((std::__are_same< _Value, double> ::__value) ? 308 : 4932)); 
# 165
}; 
# 167
template< class _Value> const int __numeric_traits_floating< _Value> ::__max_digits10; 
# 170
template< class _Value> const bool __numeric_traits_floating< _Value> ::__is_signed; 
# 173
template< class _Value> const int __numeric_traits_floating< _Value> ::__digits10; 
# 176
template< class _Value> const int __numeric_traits_floating< _Value> ::__max_exponent10; 
# 184
template< class _Value> 
# 185
struct __numeric_traits : public __numeric_traits_integer< _Value>  { 
# 187
}; 
# 190
template<> struct __numeric_traits< float>  : public __numeric_traits_floating< float>  { 
# 192
}; 
# 195
template<> struct __numeric_traits< double>  : public __numeric_traits_floating< double>  { 
# 197
}; 
# 200
template<> struct __numeric_traits< long double>  : public __numeric_traits_floating< long double>  { 
# 202
}; 
# 237 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/ext/numeric_traits.h" 3
}
# 38 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/move.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 47
template< class _Tp> constexpr _Tp *
# 49
__addressof(_Tp &__r) noexcept 
# 50
{ return __builtin_addressof(__r); } 
# 55
}
# 40 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
namespace std __attribute((__visibility__("default"))) { 
# 56 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class _Tp, _Tp __v> 
# 57
struct integral_constant { 
# 59
static constexpr inline _Tp value = (__v); 
# 60
typedef _Tp value_type; 
# 61
typedef integral_constant type; 
# 62
constexpr operator value_type() const noexcept { return value; } 
# 67
constexpr value_type operator()() const noexcept { return value; } 
# 69
}; 
# 71
template< class _Tp, _Tp __v> constexpr inline _Tp integral_constant< _Tp, __v> ::value; 
# 75
typedef integral_constant< bool, true>  true_type; 
# 78
typedef integral_constant< bool, false>  false_type; 
# 80
template< bool __v> using __bool_constant = integral_constant< bool, __v> ; 
# 85
template< bool __v> using bool_constant = integral_constant< bool, __v> ; 
# 91
template< bool , class , class > struct conditional; 
# 94
template< class _Type> 
# 95
struct __type_identity { 
# 96
using type = _Type; }; 
# 98
template< class _Tp> using __type_identity_t = typename __type_identity< _Tp> ::type; 
# 101
template< class ...> struct __or_; 
# 105
template<> struct __or_< >  : public false_type { 
# 107
}; 
# 109
template< class _B1> 
# 110
struct __or_< _B1>  : public _B1 { 
# 112
}; 
# 114
template< class _B1, class _B2> 
# 115
struct __or_< _B1, _B2>  : public conditional< _B1::value, _B1, _B2> ::type { 
# 117
}; 
# 119
template< class _B1, class _B2, class _B3, class ..._Bn> 
# 120
struct __or_< _B1, _B2, _B3, _Bn...>  : public conditional< _B1::value, _B1, std::__or_< _B2, _B3, _Bn...> > ::type { 
# 122
}; 
# 124
template< class ...> struct __and_; 
# 128
template<> struct __and_< >  : public true_type { 
# 130
}; 
# 132
template< class _B1> 
# 133
struct __and_< _B1>  : public _B1 { 
# 135
}; 
# 137
template< class _B1, class _B2> 
# 138
struct __and_< _B1, _B2>  : public conditional< _B1::value, _B2, _B1> ::type { 
# 140
}; 
# 142
template< class _B1, class _B2, class _B3, class ..._Bn> 
# 143
struct __and_< _B1, _B2, _B3, _Bn...>  : public conditional< _B1::value, std::__and_< _B2, _B3, _Bn...> , _B1> ::type { 
# 145
}; 
# 147
template< class _Pp> 
# 148
struct __not_ : public __bool_constant< !((bool)_Pp::value)>  { 
# 150
}; 
# 154
template< class ..._Bn> constexpr bool 
# 155
__or_v = (__or_< _Bn...> ::value); 
# 156
template< class ..._Bn> constexpr bool 
# 157
__and_v = (__and_< _Bn...> ::value); 
# 161
template< class ..._Bn> 
# 162
struct conjunction : public __and_< _Bn...>  { 
# 164
}; 
# 166
template< class ..._Bn> 
# 167
struct disjunction : public __or_< _Bn...>  { 
# 169
}; 
# 171
template< class _Pp> 
# 172
struct negation : public __not_< _Pp>  { 
# 174
}; 
# 176
template< class ..._Bn> constexpr bool 
# 177
conjunction_v = (conjunction< _Bn...> ::value); 
# 179
template< class ..._Bn> constexpr bool 
# 180
disjunction_v = (disjunction< _Bn...> ::value); 
# 182
template< class _Pp> constexpr bool 
# 183
negation_v = (negation< _Pp> ::value); 
# 188
template< class > struct is_reference; 
# 190
template< class > struct is_function; 
# 192
template< class > struct is_void; 
# 194
template< class > struct __is_array_unknown_bounds; 
# 200
template< class _Tp, size_t  = sizeof(_Tp)> constexpr true_type 
# 201
__is_complete_or_unbounded(__type_identity< _Tp> ) 
# 202
{ return {}; } 
# 204
template< class _TypeIdentity, class 
# 205
_NestedType = typename _TypeIdentity::type> constexpr typename __or_< is_reference< _NestedType> , is_function< _NestedType> , is_void< _NestedType> , __is_array_unknown_bounds< _NestedType> > ::type 
# 211
__is_complete_or_unbounded(_TypeIdentity) 
# 212
{ return {}; } 
# 219
template< class _Tp> 
# 220
struct __success_type { 
# 221
typedef _Tp type; }; 
# 223
struct __failure_type { 
# 224
}; 
# 226
template< class > struct remove_cv; 
# 230
template< class _Tp> using __remove_cv_t = typename remove_cv< _Tp> ::type; 
# 233
template< class > struct is_const; 
# 238
template< class > 
# 239
struct __is_void_helper : public false_type { 
# 240
}; 
# 243
template<> struct __is_void_helper< void>  : public true_type { 
# 244
}; 
# 247
template< class _Tp> 
# 248
struct is_void : public __is_void_helper< __remove_cv_t< _Tp> > ::type { 
# 250
}; 
# 252
template< class > 
# 253
struct __is_integral_helper : public false_type { 
# 254
}; 
# 257
template<> struct __is_integral_helper< bool>  : public true_type { 
# 258
}; 
# 261
template<> struct __is_integral_helper< char>  : public true_type { 
# 262
}; 
# 265
template<> struct __is_integral_helper< signed char>  : public true_type { 
# 266
}; 
# 269
template<> struct __is_integral_helper< unsigned char>  : public true_type { 
# 270
}; 
# 277
template<> struct __is_integral_helper< wchar_t>  : public true_type { 
# 278
}; 
# 288 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template<> struct __is_integral_helper< char16_t>  : public true_type { 
# 289
}; 
# 292
template<> struct __is_integral_helper< char32_t>  : public true_type { 
# 293
}; 
# 296
template<> struct __is_integral_helper< short>  : public true_type { 
# 297
}; 
# 300
template<> struct __is_integral_helper< unsigned short>  : public true_type { 
# 301
}; 
# 304
template<> struct __is_integral_helper< int>  : public true_type { 
# 305
}; 
# 308
template<> struct __is_integral_helper< unsigned>  : public true_type { 
# 309
}; 
# 312
template<> struct __is_integral_helper< long>  : public true_type { 
# 313
}; 
# 316
template<> struct __is_integral_helper< unsigned long>  : public true_type { 
# 317
}; 
# 320
template<> struct __is_integral_helper< long long>  : public true_type { 
# 321
}; 
# 324
template<> struct __is_integral_helper< unsigned long long>  : public true_type { 
# 325
}; 
# 331
template<> struct __is_integral_helper< __int128>  : public true_type { 
# 332
}; 
# 335
template<> struct __is_integral_helper< unsigned __int128>  : public true_type { 
# 336
}; 
# 367 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class _Tp> 
# 368
struct is_integral : public __is_integral_helper< __remove_cv_t< _Tp> > ::type { 
# 370
}; 
# 372
template< class > 
# 373
struct __is_floating_point_helper : public false_type { 
# 374
}; 
# 377
template<> struct __is_floating_point_helper< float>  : public true_type { 
# 378
}; 
# 381
template<> struct __is_floating_point_helper< double>  : public true_type { 
# 382
}; 
# 385
template<> struct __is_floating_point_helper< long double>  : public true_type { 
# 386
}; 
# 390
template<> struct __is_floating_point_helper< __float128>  : public true_type { 
# 391
}; 
# 395
template< class _Tp> 
# 396
struct is_floating_point : public __is_floating_point_helper< __remove_cv_t< _Tp> > ::type { 
# 398
}; 
# 401
template< class > 
# 402
struct is_array : public false_type { 
# 403
}; 
# 405
template< class _Tp, size_t _Size> 
# 406
struct is_array< _Tp [_Size]>  : public true_type { 
# 407
}; 
# 409
template< class _Tp> 
# 410
struct is_array< _Tp []>  : public true_type { 
# 411
}; 
# 413
template< class > 
# 414
struct __is_pointer_helper : public false_type { 
# 415
}; 
# 417
template< class _Tp> 
# 418
struct __is_pointer_helper< _Tp *>  : public true_type { 
# 419
}; 
# 422
template< class _Tp> 
# 423
struct is_pointer : public __is_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 425
}; 
# 428
template< class > 
# 429
struct is_lvalue_reference : public false_type { 
# 430
}; 
# 432
template< class _Tp> 
# 433
struct is_lvalue_reference< _Tp &>  : public true_type { 
# 434
}; 
# 437
template< class > 
# 438
struct is_rvalue_reference : public false_type { 
# 439
}; 
# 441
template< class _Tp> 
# 442
struct is_rvalue_reference< _Tp &&>  : public true_type { 
# 443
}; 
# 445
template< class > 
# 446
struct __is_member_object_pointer_helper : public false_type { 
# 447
}; 
# 449
template< class _Tp, class _Cp> 
# 450
struct __is_member_object_pointer_helper< _Tp (_Cp::*)>  : public __not_< is_function< _Tp> > ::type { 
# 451
}; 
# 454
template< class _Tp> 
# 455
struct is_member_object_pointer : public __is_member_object_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 457
}; 
# 459
template< class > 
# 460
struct __is_member_function_pointer_helper : public false_type { 
# 461
}; 
# 463
template< class _Tp, class _Cp> 
# 464
struct __is_member_function_pointer_helper< _Tp (_Cp::*)>  : public is_function< _Tp> ::type { 
# 465
}; 
# 468
template< class _Tp> 
# 469
struct is_member_function_pointer : public __is_member_function_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 471
}; 
# 474
template< class _Tp> 
# 475
struct is_enum : public integral_constant< bool, __is_enum(_Tp)>  { 
# 477
}; 
# 480
template< class _Tp> 
# 481
struct is_union : public integral_constant< bool, __is_union(_Tp)>  { 
# 483
}; 
# 486
template< class _Tp> 
# 487
struct is_class : public integral_constant< bool, __is_class(_Tp)>  { 
# 489
}; 
# 492
template< class _Tp> 
# 493
struct is_function : public __bool_constant< !is_const< const _Tp> ::value>  { 
# 494
}; 
# 496
template< class _Tp> 
# 497
struct is_function< _Tp &>  : public false_type { 
# 498
}; 
# 500
template< class _Tp> 
# 501
struct is_function< _Tp &&>  : public false_type { 
# 502
}; 
# 506
template< class > 
# 507
struct __is_null_pointer_helper : public false_type { 
# 508
}; 
# 511
template<> struct __is_null_pointer_helper< nullptr_t>  : public true_type { 
# 512
}; 
# 515
template< class _Tp> 
# 516
struct is_null_pointer : public __is_null_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 518
}; 
# 521
template< class _Tp> 
# 522
struct __is_nullptr_t : public is_null_pointer< _Tp>  { 
# 524
} __attribute((__deprecated__("use \'std::is_null_pointer\' instead"))); 
# 529
template< class _Tp> 
# 530
struct is_reference : public __or_< is_lvalue_reference< _Tp> , is_rvalue_reference< _Tp> > ::type { 
# 533
}; 
# 536
template< class _Tp> 
# 537
struct is_arithmetic : public __or_< is_integral< _Tp> , is_floating_point< _Tp> > ::type { 
# 539
}; 
# 542
template< class _Tp> 
# 543
struct is_fundamental : public __or_< is_arithmetic< _Tp> , is_void< _Tp> , is_null_pointer< _Tp> > ::type { 
# 546
}; 
# 549
template< class _Tp> 
# 550
struct is_object : public __not_< __or_< is_function< _Tp> , is_reference< _Tp> , is_void< _Tp> > > ::type { 
# 553
}; 
# 555
template< class > struct is_member_pointer; 
# 559
template< class _Tp> 
# 560
struct is_scalar : public __or_< is_arithmetic< _Tp> , is_enum< _Tp> , is_pointer< _Tp> , is_member_pointer< _Tp> , is_null_pointer< _Tp> > ::type { 
# 563
}; 
# 566
template< class _Tp> 
# 567
struct is_compound : public __not_< is_fundamental< _Tp> > ::type { 
# 568
}; 
# 570
template< class _Tp> 
# 571
struct __is_member_pointer_helper : public false_type { 
# 572
}; 
# 574
template< class _Tp, class _Cp> 
# 575
struct __is_member_pointer_helper< _Tp (_Cp::*)>  : public true_type { 
# 576
}; 
# 579
template< class _Tp> 
# 580
struct is_member_pointer : public __is_member_pointer_helper< __remove_cv_t< _Tp> > ::type { 
# 582
}; 
# 584
template< class , class > struct is_same; 
# 587
template< class _Tp, class ..._Types> using __is_one_of = __or_< is_same< _Tp, _Types> ...> ; 
# 591
template< class _Tp> using __is_signed_integer = __is_one_of< __remove_cv_t< _Tp> , signed char, signed short, signed int, signed long, signed long long, signed __int128> ; 
# 610 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class _Tp> using __is_unsigned_integer = __is_one_of< __remove_cv_t< _Tp> , unsigned char, unsigned short, unsigned, unsigned long, unsigned long long, unsigned __int128> ; 
# 629 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class _Tp> using __is_standard_integer = __or_< __is_signed_integer< _Tp> , __is_unsigned_integer< _Tp> > ; 
# 634
template< class ...> using __void_t = void; 
# 638
template< class _Tp, class  = void> 
# 639
struct __is_referenceable : public false_type { 
# 641
}; 
# 643
template< class _Tp> 
# 644
struct __is_referenceable< _Tp, __void_t< _Tp &> >  : public true_type { 
# 646
}; 
# 651
template< class > 
# 652
struct is_const : public false_type { 
# 653
}; 
# 655
template< class _Tp> 
# 656
struct is_const< const _Tp>  : public true_type { 
# 657
}; 
# 660
template< class > 
# 661
struct is_volatile : public false_type { 
# 662
}; 
# 664
template< class _Tp> 
# 665
struct is_volatile< volatile _Tp>  : public true_type { 
# 666
}; 
# 669
template< class _Tp> 
# 670
struct is_trivial : public integral_constant< bool, __is_trivial(_Tp)>  { 
# 673
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 675
}; 
# 678
template< class _Tp> 
# 679
struct is_trivially_copyable : public integral_constant< bool, __is_trivially_copyable(_Tp)>  { 
# 682
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 684
}; 
# 687
template< class _Tp> 
# 688
struct is_standard_layout : public integral_constant< bool, __is_standard_layout(_Tp)>  { 
# 691
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 693
}; 
# 697
template< class _Tp> 
# 700
struct is_pod : public integral_constant< bool, __is_pod(_Tp)>  { 
# 703
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 705
}; 
# 708
template< class _Tp> 
# 711
struct
# 710
 [[__deprecated__]] is_literal_type : public integral_constant< bool, __is_literal_type(_Tp)>  { 
# 714
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 716
}; 
# 719
template< class _Tp> 
# 720
struct is_empty : public integral_constant< bool, __is_empty(_Tp)>  { 
# 722
}; 
# 725
template< class _Tp> 
# 726
struct is_polymorphic : public integral_constant< bool, __is_polymorphic(_Tp)>  { 
# 728
}; 
# 733
template< class _Tp> 
# 734
struct is_final : public integral_constant< bool, __is_final(_Tp)>  { 
# 736
}; 
# 740
template< class _Tp> 
# 741
struct is_abstract : public integral_constant< bool, __is_abstract(_Tp)>  { 
# 743
}; 
# 745
template< class _Tp, bool 
# 746
 = is_arithmetic< _Tp> ::value> 
# 747
struct __is_signed_helper : public false_type { 
# 748
}; 
# 750
template< class _Tp> 
# 751
struct __is_signed_helper< _Tp, true>  : public integral_constant< bool, ((_Tp)(-1)) < ((_Tp)0)>  { 
# 753
}; 
# 756
template< class _Tp> 
# 757
struct is_signed : public __is_signed_helper< _Tp> ::type { 
# 759
}; 
# 762
template< class _Tp> 
# 763
struct is_unsigned : public __and_< is_arithmetic< _Tp> , __not_< is_signed< _Tp> > >  { 
# 765
}; 
# 775 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class _Tp, class _Up = _Tp &&> _Up __declval(int); 
# 779
template< class _Tp> _Tp __declval(long); 
# 783
template< class _Tp> auto declval() noexcept->__decltype((__declval< _Tp> (0))); 
# 786
template< class , unsigned  = 0U> struct extent; 
# 789
template< class > struct remove_all_extents; 
# 792
template< class _Tp> 
# 793
struct __is_array_known_bounds : public integral_constant< bool, (extent< _Tp> ::value > 0)>  { 
# 795
}; 
# 797
template< class _Tp> 
# 798
struct __is_array_unknown_bounds : public __and_< is_array< _Tp> , __not_< extent< _Tp> > >  { 
# 800
}; 
# 807
struct __do_is_destructible_impl { 
# 809
template< class _Tp, class  = __decltype((declval< _Tp &> ().~_Tp()))> static true_type __test(int); 
# 812
template< class > static false_type __test(...); 
# 814
}; 
# 816
template< class _Tp> 
# 817
struct __is_destructible_impl : public __do_is_destructible_impl { 
# 820
typedef __decltype((__test< _Tp> (0))) type; 
# 821
}; 
# 823
template< class _Tp, bool 
# 824
 = __or_< is_void< _Tp> , __is_array_unknown_bounds< _Tp> , is_function< _Tp> > ::value, bool 
# 827
 = __or_< is_reference< _Tp> , is_scalar< _Tp> > ::value> struct __is_destructible_safe; 
# 830
template< class _Tp> 
# 831
struct __is_destructible_safe< _Tp, false, false>  : public __is_destructible_impl< typename remove_all_extents< _Tp> ::type> ::type { 
# 834
}; 
# 836
template< class _Tp> 
# 837
struct __is_destructible_safe< _Tp, true, false>  : public false_type { 
# 838
}; 
# 840
template< class _Tp> 
# 841
struct __is_destructible_safe< _Tp, false, true>  : public true_type { 
# 842
}; 
# 845
template< class _Tp> 
# 846
struct is_destructible : public __is_destructible_safe< _Tp> ::type { 
# 849
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 851
}; 
# 857
struct __do_is_nt_destructible_impl { 
# 859
template< class _Tp> static __bool_constant< noexcept(declval< _Tp &> ().~_Tp())>  __test(int); 
# 863
template< class > static false_type __test(...); 
# 865
}; 
# 867
template< class _Tp> 
# 868
struct __is_nt_destructible_impl : public __do_is_nt_destructible_impl { 
# 871
typedef __decltype((__test< _Tp> (0))) type; 
# 872
}; 
# 874
template< class _Tp, bool 
# 875
 = __or_< is_void< _Tp> , __is_array_unknown_bounds< _Tp> , is_function< _Tp> > ::value, bool 
# 878
 = __or_< is_reference< _Tp> , is_scalar< _Tp> > ::value> struct __is_nt_destructible_safe; 
# 881
template< class _Tp> 
# 882
struct __is_nt_destructible_safe< _Tp, false, false>  : public __is_nt_destructible_impl< typename remove_all_extents< _Tp> ::type> ::type { 
# 885
}; 
# 887
template< class _Tp> 
# 888
struct __is_nt_destructible_safe< _Tp, true, false>  : public false_type { 
# 889
}; 
# 891
template< class _Tp> 
# 892
struct __is_nt_destructible_safe< _Tp, false, true>  : public true_type { 
# 893
}; 
# 896
template< class _Tp> 
# 897
struct is_nothrow_destructible : public __is_nt_destructible_safe< _Tp> ::type { 
# 900
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 902
}; 
# 904
template< class _Tp, class ..._Args> 
# 905
struct __is_constructible_impl : public __bool_constant< __is_constructible(_Tp, _Args...)>  { 
# 907
}; 
# 910
template< class _Tp, class ..._Args> 
# 911
struct is_constructible : public __is_constructible_impl< _Tp, _Args...>  { 
# 914
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 916
}; 
# 919
template< class _Tp> 
# 920
struct is_default_constructible : public __is_constructible_impl< _Tp> ::type { 
# 923
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 925
}; 
# 927
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_copy_constructible_impl; 
# 930
template< class _Tp> 
# 931
struct __is_copy_constructible_impl< _Tp, false>  : public false_type { 
# 932
}; 
# 934
template< class _Tp> 
# 935
struct __is_copy_constructible_impl< _Tp, true>  : public __is_constructible_impl< _Tp, const _Tp &>  { 
# 937
}; 
# 940
template< class _Tp> 
# 941
struct is_copy_constructible : public __is_copy_constructible_impl< _Tp>  { 
# 944
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 946
}; 
# 948
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_move_constructible_impl; 
# 951
template< class _Tp> 
# 952
struct __is_move_constructible_impl< _Tp, false>  : public false_type { 
# 953
}; 
# 955
template< class _Tp> 
# 956
struct __is_move_constructible_impl< _Tp, true>  : public __is_constructible_impl< _Tp, _Tp &&>  { 
# 958
}; 
# 961
template< class _Tp> 
# 962
struct is_move_constructible : public __is_move_constructible_impl< _Tp>  { 
# 965
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 967
}; 
# 969
template< class _Tp, class ..._Args> using __is_nothrow_constructible_impl = __bool_constant< __is_nothrow_constructible(_Tp, _Args...)> ; 
# 974
template< class _Tp, class ..._Args> 
# 975
struct is_nothrow_constructible : public integral_constant< bool, __is_nothrow_constructible(_Tp, _Args...)> ::type { 
# 978
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 980
}; 
# 983
template< class _Tp> 
# 984
struct is_nothrow_default_constructible : public __bool_constant< __is_nothrow_constructible(_Tp)>  { 
# 987
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 989
}; 
# 992
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nothrow_copy_constructible_impl; 
# 995
template< class _Tp> 
# 996
struct __is_nothrow_copy_constructible_impl< _Tp, false>  : public false_type { 
# 997
}; 
# 999
template< class _Tp> 
# 1000
struct __is_nothrow_copy_constructible_impl< _Tp, true>  : public __is_nothrow_constructible_impl< _Tp, const _Tp &>  { 
# 1002
}; 
# 1005
template< class _Tp> 
# 1006
struct is_nothrow_copy_constructible : public __is_nothrow_copy_constructible_impl< _Tp> ::type { 
# 1009
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1011
}; 
# 1013
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nothrow_move_constructible_impl; 
# 1016
template< class _Tp> 
# 1017
struct __is_nothrow_move_constructible_impl< _Tp, false>  : public false_type { 
# 1018
}; 
# 1020
template< class _Tp> 
# 1021
struct __is_nothrow_move_constructible_impl< _Tp, true>  : public __is_nothrow_constructible_impl< _Tp, _Tp &&>  { 
# 1023
}; 
# 1026
template< class _Tp> 
# 1027
struct is_nothrow_move_constructible : public __is_nothrow_move_constructible_impl< _Tp> ::type { 
# 1030
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1032
}; 
# 1035
template< class _Tp, class _Up> 
# 1036
struct is_assignable : public __bool_constant< __is_assignable(_Tp, _Up)>  { 
# 1039
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1041
}; 
# 1043
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_copy_assignable_impl; 
# 1046
template< class _Tp> 
# 1047
struct __is_copy_assignable_impl< _Tp, false>  : public false_type { 
# 1048
}; 
# 1050
template< class _Tp> 
# 1051
struct __is_copy_assignable_impl< _Tp, true>  : public __bool_constant< __is_assignable(_Tp &, const _Tp &)>  { 
# 1053
}; 
# 1056
template< class _Tp> 
# 1057
struct is_copy_assignable : public __is_copy_assignable_impl< _Tp> ::type { 
# 1060
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1062
}; 
# 1064
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_move_assignable_impl; 
# 1067
template< class _Tp> 
# 1068
struct __is_move_assignable_impl< _Tp, false>  : public false_type { 
# 1069
}; 
# 1071
template< class _Tp> 
# 1072
struct __is_move_assignable_impl< _Tp, true>  : public __bool_constant< __is_assignable(_Tp &, _Tp &&)>  { 
# 1074
}; 
# 1077
template< class _Tp> 
# 1078
struct is_move_assignable : public __is_move_assignable_impl< _Tp> ::type { 
# 1081
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1083
}; 
# 1085
template< class _Tp, class _Up> using __is_nothrow_assignable_impl = __bool_constant< __is_nothrow_assignable(_Tp, _Up)> ; 
# 1090
template< class _Tp, class _Up> 
# 1091
struct is_nothrow_assignable : public __is_nothrow_assignable_impl< _Tp, _Up>  { 
# 1094
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1096
}; 
# 1098
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nt_copy_assignable_impl; 
# 1101
template< class _Tp> 
# 1102
struct __is_nt_copy_assignable_impl< _Tp, false>  : public false_type { 
# 1103
}; 
# 1105
template< class _Tp> 
# 1106
struct __is_nt_copy_assignable_impl< _Tp, true>  : public __is_nothrow_assignable_impl< _Tp &, const _Tp &>  { 
# 1108
}; 
# 1111
template< class _Tp> 
# 1112
struct is_nothrow_copy_assignable : public __is_nt_copy_assignable_impl< _Tp>  { 
# 1115
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1117
}; 
# 1119
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_nt_move_assignable_impl; 
# 1122
template< class _Tp> 
# 1123
struct __is_nt_move_assignable_impl< _Tp, false>  : public false_type { 
# 1124
}; 
# 1126
template< class _Tp> 
# 1127
struct __is_nt_move_assignable_impl< _Tp, true>  : public __is_nothrow_assignable_impl< _Tp &, _Tp &&>  { 
# 1129
}; 
# 1132
template< class _Tp> 
# 1133
struct is_nothrow_move_assignable : public __is_nt_move_assignable_impl< _Tp>  { 
# 1136
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1138
}; 
# 1141
template< class _Tp, class ..._Args> 
# 1142
struct is_trivially_constructible : public __bool_constant< __is_trivially_constructible(_Tp, _Args...)>  { 
# 1145
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1147
}; 
# 1150
template< class _Tp> 
# 1151
struct is_trivially_default_constructible : public __bool_constant< __is_trivially_constructible(_Tp)>  { 
# 1154
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1156
}; 
# 1158
struct __do_is_implicitly_default_constructible_impl { 
# 1160
template< class _Tp> static void __helper(const _Tp &); 
# 1163
template< class _Tp> static true_type __test(const _Tp &, __decltype((__helper< const _Tp &> ({}))) * = 0); 
# 1167
static false_type __test(...); 
# 1168
}; 
# 1170
template< class _Tp> 
# 1171
struct __is_implicitly_default_constructible_impl : public __do_is_implicitly_default_constructible_impl { 
# 1174
typedef __decltype((__test(declval< _Tp> ()))) type; 
# 1175
}; 
# 1177
template< class _Tp> 
# 1178
struct __is_implicitly_default_constructible_safe : public __is_implicitly_default_constructible_impl< _Tp> ::type { 
# 1180
}; 
# 1182
template< class _Tp> 
# 1183
struct __is_implicitly_default_constructible : public __and_< __is_constructible_impl< _Tp> , __is_implicitly_default_constructible_safe< _Tp> >  { 
# 1186
}; 
# 1188
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_copy_constructible_impl; 
# 1191
template< class _Tp> 
# 1192
struct __is_trivially_copy_constructible_impl< _Tp, false>  : public false_type { 
# 1193
}; 
# 1195
template< class _Tp> 
# 1196
struct __is_trivially_copy_constructible_impl< _Tp, true>  : public __and_< __is_copy_constructible_impl< _Tp> , integral_constant< bool, __is_trivially_constructible(_Tp, const _Tp &)> >  { 
# 1200
}; 
# 1203
template< class _Tp> 
# 1204
struct is_trivially_copy_constructible : public __is_trivially_copy_constructible_impl< _Tp>  { 
# 1207
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1209
}; 
# 1211
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_move_constructible_impl; 
# 1214
template< class _Tp> 
# 1215
struct __is_trivially_move_constructible_impl< _Tp, false>  : public false_type { 
# 1216
}; 
# 1218
template< class _Tp> 
# 1219
struct __is_trivially_move_constructible_impl< _Tp, true>  : public __and_< __is_move_constructible_impl< _Tp> , integral_constant< bool, __is_trivially_constructible(_Tp, _Tp &&)> >  { 
# 1223
}; 
# 1226
template< class _Tp> 
# 1227
struct is_trivially_move_constructible : public __is_trivially_move_constructible_impl< _Tp>  { 
# 1230
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1232
}; 
# 1235
template< class _Tp, class _Up> 
# 1236
struct is_trivially_assignable : public __bool_constant< __is_trivially_assignable(_Tp, _Up)>  { 
# 1239
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1241
}; 
# 1243
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_copy_assignable_impl; 
# 1246
template< class _Tp> 
# 1247
struct __is_trivially_copy_assignable_impl< _Tp, false>  : public false_type { 
# 1248
}; 
# 1250
template< class _Tp> 
# 1251
struct __is_trivially_copy_assignable_impl< _Tp, true>  : public __bool_constant< __is_trivially_assignable(_Tp &, const _Tp &)>  { 
# 1253
}; 
# 1256
template< class _Tp> 
# 1257
struct is_trivially_copy_assignable : public __is_trivially_copy_assignable_impl< _Tp>  { 
# 1260
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1262
}; 
# 1264
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> struct __is_trivially_move_assignable_impl; 
# 1267
template< class _Tp> 
# 1268
struct __is_trivially_move_assignable_impl< _Tp, false>  : public false_type { 
# 1269
}; 
# 1271
template< class _Tp> 
# 1272
struct __is_trivially_move_assignable_impl< _Tp, true>  : public __bool_constant< __is_trivially_assignable(_Tp &, _Tp &&)>  { 
# 1274
}; 
# 1277
template< class _Tp> 
# 1278
struct is_trivially_move_assignable : public __is_trivially_move_assignable_impl< _Tp>  { 
# 1281
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1283
}; 
# 1286
template< class _Tp> 
# 1287
struct is_trivially_destructible : public __and_< __is_destructible_safe< _Tp> , __bool_constant< __has_trivial_destructor(_Tp)> >  { 
# 1291
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1293
}; 
# 1297
template< class _Tp> 
# 1298
struct has_virtual_destructor : public integral_constant< bool, __has_virtual_destructor(_Tp)>  { 
# 1301
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1303
}; 
# 1309
template< class _Tp> 
# 1310
struct alignment_of : public integral_constant< unsigned long, __alignof__(_Tp)>  { 
# 1313
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 1315
}; 
# 1318
template< class > 
# 1319
struct rank : public integral_constant< unsigned long, 0UL>  { 
# 1320
}; 
# 1322
template< class _Tp, size_t _Size> 
# 1323
struct rank< _Tp [_Size]>  : public integral_constant< unsigned long, 1 + std::rank< _Tp> ::value>  { 
# 1324
}; 
# 1326
template< class _Tp> 
# 1327
struct rank< _Tp []>  : public integral_constant< unsigned long, 1 + std::rank< _Tp> ::value>  { 
# 1328
}; 
# 1331
template< class , unsigned _Uint> 
# 1332
struct extent : public integral_constant< unsigned long, 0UL>  { 
# 1333
}; 
# 1335
template< class _Tp, unsigned _Uint, size_t _Size> 
# 1336
struct extent< _Tp [_Size], _Uint>  : public integral_constant< unsigned long, (_Uint == (0)) ? _Size : std::extent< _Tp, _Uint - (1)> ::value>  { 
# 1340
}; 
# 1342
template< class _Tp, unsigned _Uint> 
# 1343
struct extent< _Tp [], _Uint>  : public integral_constant< unsigned long, (_Uint == (0)) ? 0 : std::extent< _Tp, _Uint - (1)> ::value>  { 
# 1347
}; 
# 1353
template< class _Tp, class _Up> 
# 1354
struct is_same : public integral_constant< bool, __is_same(_Tp, _Up)>  { 
# 1360
}; 
# 1370 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class _Base, class _Derived> 
# 1371
struct is_base_of : public integral_constant< bool, __is_base_of(_Base, _Derived)>  { 
# 1373
}; 
# 1375
template< class _From, class _To, bool 
# 1376
 = __or_< is_void< _From> , is_function< _To> , is_array< _To> > ::value> 
# 1378
struct __is_convertible_helper { 
# 1380
typedef typename is_void< _To> ::type type; 
# 1381
}; 
# 1383
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
template< class _From, class _To> 
# 1386
class __is_convertible_helper< _From, _To, false>  { 
# 1388
template< class _To1> static void __test_aux(_To1) noexcept; 
# 1391
template< class _From1, class _To1, class 
# 1392
 = __decltype((__test_aux< _To1> (std::declval< _From1> ())))> static true_type 
# 1391
__test(int); 
# 1396
template< class , class > static false_type __test(...); 
# 1401
public: typedef __decltype((__test< _From, _To> (0))) type; 
# 1402
}; 
#pragma GCC diagnostic pop
# 1406
template< class _From, class _To> 
# 1407
struct is_convertible : public __is_convertible_helper< _From, _To> ::type { 
# 1409
}; 
# 1412
template< class _ToElementType, class _FromElementType> using __is_array_convertible = is_convertible< _FromElementType (*)[], _ToElementType (*)[]> ; 
# 1416
template< class _From, class _To, bool 
# 1417
 = __or_< is_void< _From> , is_function< _To> , is_array< _To> > ::value> 
# 1419
struct __is_nt_convertible_helper : public is_void< _To>  { 
# 1421
}; 
# 1423
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
template< class _From, class _To> 
# 1426
class __is_nt_convertible_helper< _From, _To, false>  { 
# 1428
template< class _To1> static void __test_aux(_To1) noexcept; 
# 1431
template< class _From1, class _To1> static __bool_constant< noexcept(__test_aux< _To1> (std::declval< _From1> ()))>  __test(int); 
# 1436
template< class , class > static false_type __test(...); 
# 1441
public: using type = __decltype((__test< _From, _To> (0))); 
# 1442
}; 
#pragma GCC diagnostic pop
# 1446
template< class _From, class _To> 
# 1447
struct __is_nothrow_convertible : public __is_nt_convertible_helper< _From, _To> ::type { 
# 1449
}; 
# 1468 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class _Tp> 
# 1469
struct remove_const { 
# 1470
typedef _Tp type; }; 
# 1472
template< class _Tp> 
# 1473
struct remove_const< const _Tp>  { 
# 1474
typedef _Tp type; }; 
# 1477
template< class _Tp> 
# 1478
struct remove_volatile { 
# 1479
typedef _Tp type; }; 
# 1481
template< class _Tp> 
# 1482
struct remove_volatile< volatile _Tp>  { 
# 1483
typedef _Tp type; }; 
# 1486
template< class _Tp> 
# 1487
struct remove_cv { 
# 1488
using type = _Tp; }; 
# 1490
template< class _Tp> 
# 1491
struct remove_cv< const _Tp>  { 
# 1492
using type = _Tp; }; 
# 1494
template< class _Tp> 
# 1495
struct remove_cv< volatile _Tp>  { 
# 1496
using type = _Tp; }; 
# 1498
template< class _Tp> 
# 1499
struct remove_cv< const volatile _Tp>  { 
# 1500
using type = _Tp; }; 
# 1503
template< class _Tp> 
# 1504
struct add_const { 
# 1505
typedef const _Tp type; }; 
# 1508
template< class _Tp> 
# 1509
struct add_volatile { 
# 1510
typedef volatile _Tp type; }; 
# 1513
template< class _Tp> 
# 1514
struct add_cv { 
# 1517
typedef typename add_const< typename add_volatile< _Tp> ::type> ::type type; 
# 1518
}; 
# 1525
template< class _Tp> using remove_const_t = typename remove_const< _Tp> ::type; 
# 1529
template< class _Tp> using remove_volatile_t = typename remove_volatile< _Tp> ::type; 
# 1533
template< class _Tp> using remove_cv_t = typename remove_cv< _Tp> ::type; 
# 1537
template< class _Tp> using add_const_t = typename add_const< _Tp> ::type; 
# 1541
template< class _Tp> using add_volatile_t = typename add_volatile< _Tp> ::type; 
# 1545
template< class _Tp> using add_cv_t = typename add_cv< _Tp> ::type; 
# 1552
template< class _Tp> 
# 1553
struct remove_reference { 
# 1554
typedef _Tp type; }; 
# 1556
template< class _Tp> 
# 1557
struct remove_reference< _Tp &>  { 
# 1558
typedef _Tp type; }; 
# 1560
template< class _Tp> 
# 1561
struct remove_reference< _Tp &&>  { 
# 1562
typedef _Tp type; }; 
# 1564
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> 
# 1565
struct __add_lvalue_reference_helper { 
# 1566
typedef _Tp type; }; 
# 1568
template< class _Tp> 
# 1569
struct __add_lvalue_reference_helper< _Tp, true>  { 
# 1570
typedef _Tp &type; }; 
# 1573
template< class _Tp> 
# 1574
struct add_lvalue_reference : public __add_lvalue_reference_helper< _Tp>  { 
# 1576
}; 
# 1578
template< class _Tp, bool  = __is_referenceable< _Tp> ::value> 
# 1579
struct __add_rvalue_reference_helper { 
# 1580
typedef _Tp type; }; 
# 1582
template< class _Tp> 
# 1583
struct __add_rvalue_reference_helper< _Tp, true>  { 
# 1584
typedef _Tp &&type; }; 
# 1587
template< class _Tp> 
# 1588
struct add_rvalue_reference : public __add_rvalue_reference_helper< _Tp>  { 
# 1590
}; 
# 1594
template< class _Tp> using remove_reference_t = typename remove_reference< _Tp> ::type; 
# 1598
template< class _Tp> using add_lvalue_reference_t = typename add_lvalue_reference< _Tp> ::type; 
# 1602
template< class _Tp> using add_rvalue_reference_t = typename add_rvalue_reference< _Tp> ::type; 
# 1609
template< class _Unqualified, bool _IsConst, bool _IsVol> struct __cv_selector; 
# 1612
template< class _Unqualified> 
# 1613
struct __cv_selector< _Unqualified, false, false>  { 
# 1614
typedef _Unqualified __type; }; 
# 1616
template< class _Unqualified> 
# 1617
struct __cv_selector< _Unqualified, false, true>  { 
# 1618
typedef volatile _Unqualified __type; }; 
# 1620
template< class _Unqualified> 
# 1621
struct __cv_selector< _Unqualified, true, false>  { 
# 1622
typedef const _Unqualified __type; }; 
# 1624
template< class _Unqualified> 
# 1625
struct __cv_selector< _Unqualified, true, true>  { 
# 1626
typedef const volatile _Unqualified __type; }; 
# 1628
template< class _Qualified, class _Unqualified, bool 
# 1629
_IsConst = is_const< _Qualified> ::value, bool 
# 1630
_IsVol = is_volatile< _Qualified> ::value> 
# 1631
class __match_cv_qualifiers { 
# 1633
typedef __cv_selector< _Unqualified, _IsConst, _IsVol>  __match; 
# 1636
public: typedef typename __cv_selector< _Unqualified, _IsConst, _IsVol> ::__type __type; 
# 1637
}; 
# 1640
template< class _Tp> 
# 1641
struct __make_unsigned { 
# 1642
typedef _Tp __type; }; 
# 1645
template<> struct __make_unsigned< char>  { 
# 1646
typedef unsigned char __type; }; 
# 1649
template<> struct __make_unsigned< signed char>  { 
# 1650
typedef unsigned char __type; }; 
# 1653
template<> struct __make_unsigned< short>  { 
# 1654
typedef unsigned short __type; }; 
# 1657
template<> struct __make_unsigned< int>  { 
# 1658
typedef unsigned __type; }; 
# 1661
template<> struct __make_unsigned< long>  { 
# 1662
typedef unsigned long __type; }; 
# 1665
template<> struct __make_unsigned< long long>  { 
# 1666
typedef unsigned long long __type; }; 
# 1670
template<> struct __make_unsigned< __int128>  { 
# 1671
typedef unsigned __int128 __type; }; 
# 1690 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class _Tp, bool 
# 1691
_IsInt = is_integral< _Tp> ::value, bool 
# 1692
_IsEnum = is_enum< _Tp> ::value> class __make_unsigned_selector; 
# 1695
template< class _Tp> 
# 1696
class __make_unsigned_selector< _Tp, true, false>  { 
# 1698
using __unsigned_type = typename __make_unsigned< __remove_cv_t< _Tp> > ::__type; 
# 1702
public: using __type = typename __match_cv_qualifiers< _Tp, __unsigned_type> ::__type; 
# 1704
}; 
# 1706
class __make_unsigned_selector_base { 
# 1709
protected: template< class ...> struct _List { }; 
# 1711
template< class _Tp, class ..._Up> 
# 1712
struct _List< _Tp, _Up...>  : public __make_unsigned_selector_base::_List< _Up...>  { 
# 1713
static constexpr inline std::size_t __size = sizeof(_Tp); }; 
# 1715
template< size_t _Sz, class _Tp, bool  = _Sz <= _Tp::__size> struct __select; 
# 1718
template< size_t _Sz, class _Uint, class ..._UInts> 
# 1719
struct __select< _Sz, _List< _Uint, _UInts...> , true>  { 
# 1720
using __type = _Uint; }; 
# 1722
template< size_t _Sz, class _Uint, class ..._UInts> 
# 1723
struct __select< _Sz, _List< _Uint, _UInts...> , false>  : public __make_unsigned_selector_base::__select< _Sz, _List< _UInts...> >  { 
# 1725
}; 
# 1726
}; 
# 1729
template< class _Tp> 
# 1730
class __make_unsigned_selector< _Tp, false, true>  : private __make_unsigned_selector_base { 
# 1734
using _UInts = _List< unsigned char, unsigned short, unsigned, unsigned long, unsigned long long> ; 
# 1737
using __unsigned_type = typename __select< sizeof(_Tp), _List< unsigned char, unsigned short, unsigned, unsigned long, unsigned long long> > ::__type; 
# 1740
public: using __type = typename __match_cv_qualifiers< _Tp, __unsigned_type> ::__type; 
# 1742
}; 
# 1750
template<> struct __make_unsigned< wchar_t>  { 
# 1752
using __type = __make_unsigned_selector< wchar_t, false, true> ::__type; 
# 1754
}; 
# 1767 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template<> struct __make_unsigned< char16_t>  { 
# 1769
using __type = __make_unsigned_selector< char16_t, false, true> ::__type; 
# 1771
}; 
# 1774
template<> struct __make_unsigned< char32_t>  { 
# 1776
using __type = __make_unsigned_selector< char32_t, false, true> ::__type; 
# 1778
}; 
# 1784
template< class _Tp> 
# 1785
struct make_unsigned { 
# 1786
typedef typename __make_unsigned_selector< _Tp> ::__type type; }; 
# 1790
template<> struct make_unsigned< bool> ; 
# 1794
template< class _Tp> 
# 1795
struct __make_signed { 
# 1796
typedef _Tp __type; }; 
# 1799
template<> struct __make_signed< char>  { 
# 1800
typedef signed char __type; }; 
# 1803
template<> struct __make_signed< unsigned char>  { 
# 1804
typedef signed char __type; }; 
# 1807
template<> struct __make_signed< unsigned short>  { 
# 1808
typedef signed short __type; }; 
# 1811
template<> struct __make_signed< unsigned>  { 
# 1812
typedef signed int __type; }; 
# 1815
template<> struct __make_signed< unsigned long>  { 
# 1816
typedef signed long __type; }; 
# 1819
template<> struct __make_signed< unsigned long long>  { 
# 1820
typedef signed long long __type; }; 
# 1824
template<> struct __make_signed< unsigned __int128>  { 
# 1825
typedef __int128 __type; }; 
# 1844 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class _Tp, bool 
# 1845
_IsInt = is_integral< _Tp> ::value, bool 
# 1846
_IsEnum = is_enum< _Tp> ::value> class __make_signed_selector; 
# 1849
template< class _Tp> 
# 1850
class __make_signed_selector< _Tp, true, false>  { 
# 1852
using __signed_type = typename __make_signed< __remove_cv_t< _Tp> > ::__type; 
# 1856
public: using __type = typename __match_cv_qualifiers< _Tp, __signed_type> ::__type; 
# 1858
}; 
# 1861
template< class _Tp> 
# 1862
class __make_signed_selector< _Tp, false, true>  { 
# 1864
typedef typename __make_unsigned_selector< _Tp> ::__type __unsigned_type; 
# 1867
public: typedef typename std::__make_signed_selector< __unsigned_type> ::__type __type; 
# 1868
}; 
# 1876
template<> struct __make_signed< wchar_t>  { 
# 1878
using __type = __make_signed_selector< wchar_t, false, true> ::__type; 
# 1880
}; 
# 1893 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template<> struct __make_signed< char16_t>  { 
# 1895
using __type = __make_signed_selector< char16_t, false, true> ::__type; 
# 1897
}; 
# 1900
template<> struct __make_signed< char32_t>  { 
# 1902
using __type = __make_signed_selector< char32_t, false, true> ::__type; 
# 1904
}; 
# 1910
template< class _Tp> 
# 1911
struct make_signed { 
# 1912
typedef typename __make_signed_selector< _Tp> ::__type type; }; 
# 1916
template<> struct make_signed< bool> ; 
# 1920
template< class _Tp> using make_signed_t = typename make_signed< _Tp> ::type; 
# 1924
template< class _Tp> using make_unsigned_t = typename make_unsigned< _Tp> ::type; 
# 1931
template< class _Tp> 
# 1932
struct remove_extent { 
# 1933
typedef _Tp type; }; 
# 1935
template< class _Tp, size_t _Size> 
# 1936
struct remove_extent< _Tp [_Size]>  { 
# 1937
typedef _Tp type; }; 
# 1939
template< class _Tp> 
# 1940
struct remove_extent< _Tp []>  { 
# 1941
typedef _Tp type; }; 
# 1944
template< class _Tp> 
# 1945
struct remove_all_extents { 
# 1946
typedef _Tp type; }; 
# 1948
template< class _Tp, size_t _Size> 
# 1949
struct remove_all_extents< _Tp [_Size]>  { 
# 1950
typedef typename std::remove_all_extents< _Tp> ::type type; }; 
# 1952
template< class _Tp> 
# 1953
struct remove_all_extents< _Tp []>  { 
# 1954
typedef typename std::remove_all_extents< _Tp> ::type type; }; 
# 1958
template< class _Tp> using remove_extent_t = typename remove_extent< _Tp> ::type; 
# 1962
template< class _Tp> using remove_all_extents_t = typename remove_all_extents< _Tp> ::type; 
# 1968
template< class _Tp, class > 
# 1969
struct __remove_pointer_helper { 
# 1970
typedef _Tp type; }; 
# 1972
template< class _Tp, class _Up> 
# 1973
struct __remove_pointer_helper< _Tp, _Up *>  { 
# 1974
typedef _Up type; }; 
# 1977
template< class _Tp> 
# 1978
struct remove_pointer : public __remove_pointer_helper< _Tp, __remove_cv_t< _Tp> >  { 
# 1980
}; 
# 1983
template< class _Tp, bool  = __or_< __is_referenceable< _Tp> , is_void< _Tp> > ::value> 
# 1985
struct __add_pointer_helper { 
# 1986
typedef _Tp type; }; 
# 1988
template< class _Tp> 
# 1989
struct __add_pointer_helper< _Tp, true>  { 
# 1990
typedef typename remove_reference< _Tp> ::type *type; }; 
# 1992
template< class _Tp> 
# 1993
struct add_pointer : public __add_pointer_helper< _Tp>  { 
# 1995
}; 
# 1999
template< class _Tp> using remove_pointer_t = typename remove_pointer< _Tp> ::type; 
# 2003
template< class _Tp> using add_pointer_t = typename add_pointer< _Tp> ::type; 
# 2007
template< size_t _Len> 
# 2008
struct __aligned_storage_msa { 
# 2010
union __type { 
# 2012
unsigned char __data[_Len]; 
# 2013
struct __attribute((__aligned__)) { } __align; 
# 2014
}; 
# 2015
}; 
# 2027 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< size_t _Len, size_t _Align = __alignof__(typename __aligned_storage_msa< _Len> ::__type)> 
# 2029
struct aligned_storage { 
# 2031
union type { 
# 2033
unsigned char __data[_Len]; 
# 2034
struct __attribute((__aligned__(_Align))) { } __align; 
# 2035
}; 
# 2036
}; 
# 2038
template< class ..._Types> 
# 2039
struct __strictest_alignment { 
# 2041
static const size_t _S_alignment = (0); 
# 2042
static const size_t _S_size = (0); 
# 2043
}; 
# 2045
template< class _Tp, class ..._Types> 
# 2046
struct __strictest_alignment< _Tp, _Types...>  { 
# 2048
static const size_t _S_alignment = ((__alignof__(_Tp) > __strictest_alignment< _Types...> ::_S_alignment) ? __alignof__(_Tp) : __strictest_alignment< _Types...> ::_S_alignment); 
# 2051
static const size_t _S_size = ((sizeof(_Tp) > __strictest_alignment< _Types...> ::_S_size) ? sizeof(_Tp) : __strictest_alignment< _Types...> ::_S_size); 
# 2054
}; 
# 2066 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< size_t _Len, class ..._Types> 
# 2067
struct aligned_union { 
# 2070
static_assert((sizeof...(_Types) != (0)), "At least one type is required");
# 2072
private: using __strictest = __strictest_alignment< _Types...> ; 
# 2073
static const size_t _S_len = ((_Len > __strictest::_S_size) ? _Len : __strictest::_S_size); 
# 2077
public: static const size_t alignment_value = (__strictest::_S_alignment); 
# 2079
typedef typename aligned_storage< _S_len, alignment_value> ::type type; 
# 2080
}; 
# 2082
template< size_t _Len, class ..._Types> const size_t aligned_union< _Len, _Types...> ::alignment_value; 
# 2087
template< class _Up, bool 
# 2088
_IsArray = is_array< _Up> ::value, bool 
# 2089
_IsFunction = is_function< _Up> ::value> struct __decay_selector; 
# 2093
template< class _Up> 
# 2094
struct __decay_selector< _Up, false, false>  { 
# 2095
typedef __remove_cv_t< _Up>  __type; }; 
# 2097
template< class _Up> 
# 2098
struct __decay_selector< _Up, true, false>  { 
# 2099
typedef typename remove_extent< _Up> ::type *__type; }; 
# 2101
template< class _Up> 
# 2102
struct __decay_selector< _Up, false, true>  { 
# 2103
typedef typename add_pointer< _Up> ::type __type; }; 
# 2106
template< class _Tp> 
# 2107
class decay { 
# 2109
typedef typename remove_reference< _Tp> ::type __remove_type; 
# 2112
public: typedef typename __decay_selector< __remove_type> ::__type type; 
# 2113
}; 
# 2116
template< class _Tp> using __decay_t = typename decay< _Tp> ::type; 
# 2119
template< class _Tp> class reference_wrapper; 
# 2123
template< class _Tp> 
# 2124
struct __strip_reference_wrapper { 
# 2126
typedef _Tp __type; 
# 2127
}; 
# 2129
template< class _Tp> 
# 2130
struct __strip_reference_wrapper< reference_wrapper< _Tp> >  { 
# 2132
typedef _Tp &__type; 
# 2133
}; 
# 2135
template< class _Tp> using __decay_and_strip = __strip_reference_wrapper< __decay_t< _Tp> > ; 
# 2141
template< bool , class _Tp = void> 
# 2142
struct enable_if { 
# 2143
}; 
# 2146
template< class _Tp> 
# 2147
struct enable_if< true, _Tp>  { 
# 2148
typedef _Tp type; }; 
# 2151
template< bool _Cond, class _Tp = void> using __enable_if_t = typename enable_if< _Cond, _Tp> ::type; 
# 2154
template< class ..._Cond> using _Require = __enable_if_t< __and_< _Cond...> ::value> ; 
# 2159
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 2160
struct conditional { 
# 2161
typedef _Iftrue type; }; 
# 2164
template< class _Iftrue, class _Iffalse> 
# 2165
struct conditional< false, _Iftrue, _Iffalse>  { 
# 2166
typedef _Iffalse type; }; 
# 2169
template< class _Tp> using __remove_cvref_t = typename remove_cv< typename remove_reference< _Tp> ::type> ::type; 
# 2174
template< class ..._Tp> struct common_type; 
# 2179
struct __do_common_type_impl { 
# 2181
template< class _Tp, class _Up> using __cond_t = __decltype((true ? std::declval< _Tp> () : std::declval< _Up> ())); 
# 2187
template< class _Tp, class _Up> static __success_type< __decay_t< __cond_t< _Tp, _Up> > >  _S_test(int); 
# 2199 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class , class > static __failure_type _S_test_2(...); 
# 2203
template< class _Tp, class _Up> static __decltype((_S_test_2< _Tp, _Up> (0))) _S_test(...); 
# 2206
}; 
# 2210
template<> struct common_type< >  { 
# 2211
}; 
# 2214
template< class _Tp0> 
# 2215
struct common_type< _Tp0>  : public std::common_type< _Tp0, _Tp0>  { 
# 2217
}; 
# 2220
template< class _Tp1, class _Tp2, class 
# 2221
_Dp1 = __decay_t< _Tp1> , class _Dp2 = __decay_t< _Tp2> > 
# 2222
struct __common_type_impl { 
# 2226
using type = common_type< _Dp1, _Dp2> ; 
# 2227
}; 
# 2229
template< class _Tp1, class _Tp2> 
# 2230
struct __common_type_impl< _Tp1, _Tp2, _Tp1, _Tp2>  : private __do_common_type_impl { 
# 2235
using type = __decltype((_S_test< _Tp1, _Tp2> (0))); 
# 2236
}; 
# 2239
template< class _Tp1, class _Tp2> 
# 2240
struct common_type< _Tp1, _Tp2>  : public __common_type_impl< _Tp1, _Tp2> ::type { 
# 2242
}; 
# 2244
template< class ...> 
# 2245
struct __common_type_pack { 
# 2246
}; 
# 2248
template< class , class , class  = void> struct __common_type_fold; 
# 2252
template< class _Tp1, class _Tp2, class ..._Rp> 
# 2253
struct common_type< _Tp1, _Tp2, _Rp...>  : public __common_type_fold< std::common_type< _Tp1, _Tp2> , __common_type_pack< _Rp...> >  { 
# 2256
}; 
# 2261
template< class _CTp, class ..._Rp> 
# 2262
struct __common_type_fold< _CTp, __common_type_pack< _Rp...> , __void_t< typename _CTp::type> >  : public common_type< typename _CTp::type, _Rp...>  { 
# 2265
}; 
# 2268
template< class _CTp, class _Rp> 
# 2269
struct __common_type_fold< _CTp, _Rp, void>  { 
# 2270
}; 
# 2272
template< class _Tp, bool  = is_enum< _Tp> ::value> 
# 2273
struct __underlying_type_impl { 
# 2275
using type = __underlying_type(_Tp); 
# 2276
}; 
# 2278
template< class _Tp> 
# 2279
struct __underlying_type_impl< _Tp, false>  { 
# 2280
}; 
# 2283
template< class _Tp> 
# 2284
struct underlying_type : public __underlying_type_impl< _Tp>  { 
# 2286
}; 
# 2288
template< class _Tp> 
# 2289
struct __declval_protector { 
# 2291
static const bool __stop = false; 
# 2292
}; 
# 2294
template< class _Tp> auto 
# 2295
declval() noexcept->__decltype((__declval< _Tp> (0))) 
# 2296
{ 
# 2297
static_assert((__declval_protector< _Tp> ::__stop), "declval() must not be used!");
# 2299
return __declval< _Tp> (0); 
# 2300
} 
# 2303
template< class _Signature> struct result_of; 
# 2310
struct __invoke_memfun_ref { }; 
# 2311
struct __invoke_memfun_deref { }; 
# 2312
struct __invoke_memobj_ref { }; 
# 2313
struct __invoke_memobj_deref { }; 
# 2314
struct __invoke_other { }; 
# 2317
template< class _Tp, class _Tag> 
# 2318
struct __result_of_success : public __success_type< _Tp>  { 
# 2319
using __invoke_type = _Tag; }; 
# 2322
struct __result_of_memfun_ref_impl { 
# 2324
template< class _Fp, class _Tp1, class ..._Args> static __result_of_success< __decltype(((std::declval< _Tp1> ().*std::declval< _Fp> ())(std::declval< _Args> ()...))), __invoke_memfun_ref>  _S_test(int); 
# 2329
template< class ...> static __failure_type _S_test(...); 
# 2331
}; 
# 2333
template< class _MemPtr, class _Arg, class ..._Args> 
# 2334
struct __result_of_memfun_ref : private __result_of_memfun_ref_impl { 
# 2337
typedef __decltype((_S_test< _MemPtr, _Arg, _Args...> (0))) type; 
# 2338
}; 
# 2341
struct __result_of_memfun_deref_impl { 
# 2343
template< class _Fp, class _Tp1, class ..._Args> static __result_of_success< __decltype((((*std::declval< _Tp1> ()).*std::declval< _Fp> ())(std::declval< _Args> ()...))), __invoke_memfun_deref>  _S_test(int); 
# 2348
template< class ...> static __failure_type _S_test(...); 
# 2350
}; 
# 2352
template< class _MemPtr, class _Arg, class ..._Args> 
# 2353
struct __result_of_memfun_deref : private __result_of_memfun_deref_impl { 
# 2356
typedef __decltype((_S_test< _MemPtr, _Arg, _Args...> (0))) type; 
# 2357
}; 
# 2360
struct __result_of_memobj_ref_impl { 
# 2362
template< class _Fp, class _Tp1> static __result_of_success< __decltype((std::declval< _Tp1> ().*std::declval< _Fp> ())), __invoke_memobj_ref>  _S_test(int); 
# 2367
template< class , class > static __failure_type _S_test(...); 
# 2369
}; 
# 2371
template< class _MemPtr, class _Arg> 
# 2372
struct __result_of_memobj_ref : private __result_of_memobj_ref_impl { 
# 2375
typedef __decltype((_S_test< _MemPtr, _Arg> (0))) type; 
# 2376
}; 
# 2379
struct __result_of_memobj_deref_impl { 
# 2381
template< class _Fp, class _Tp1> static __result_of_success< __decltype(((*std::declval< _Tp1> ()).*std::declval< _Fp> ())), __invoke_memobj_deref>  _S_test(int); 
# 2386
template< class , class > static __failure_type _S_test(...); 
# 2388
}; 
# 2390
template< class _MemPtr, class _Arg> 
# 2391
struct __result_of_memobj_deref : private __result_of_memobj_deref_impl { 
# 2394
typedef __decltype((_S_test< _MemPtr, _Arg> (0))) type; 
# 2395
}; 
# 2397
template< class _MemPtr, class _Arg> struct __result_of_memobj; 
# 2400
template< class _Res, class _Class, class _Arg> 
# 2401
struct __result_of_memobj< _Res (_Class::*), _Arg>  { 
# 2403
typedef __remove_cvref_t< _Arg>  _Argval; 
# 2404
typedef _Res (_Class::*_MemPtr); 
# 2409
typedef typename conditional< __or_< is_same< _Argval, _Class> , is_base_of< _Class, _Argval> > ::value, __result_of_memobj_ref< _MemPtr, _Arg> , __result_of_memobj_deref< _MemPtr, _Arg> > ::type::type type; 
# 2410
}; 
# 2412
template< class _MemPtr, class _Arg, class ..._Args> struct __result_of_memfun; 
# 2415
template< class _Res, class _Class, class _Arg, class ..._Args> 
# 2416
struct __result_of_memfun< _Res (_Class::*), _Arg, _Args...>  { 
# 2418
typedef typename remove_reference< _Arg> ::type _Argval; 
# 2419
typedef _Res (_Class::*_MemPtr); 
# 2423
typedef typename conditional< is_base_of< _Class, _Argval> ::value, __result_of_memfun_ref< _MemPtr, _Arg, _Args...> , __result_of_memfun_deref< _MemPtr, _Arg, _Args...> > ::type::type type; 
# 2424
}; 
# 2431
template< class _Tp, class _Up = __remove_cvref_t< _Tp> > 
# 2432
struct __inv_unwrap { 
# 2434
using type = _Tp; 
# 2435
}; 
# 2437
template< class _Tp, class _Up> 
# 2438
struct __inv_unwrap< _Tp, reference_wrapper< _Up> >  { 
# 2440
using type = _Up &; 
# 2441
}; 
# 2443
template< bool , bool , class _Functor, class ..._ArgTypes> 
# 2444
struct __result_of_impl { 
# 2446
typedef __failure_type type; 
# 2447
}; 
# 2449
template< class _MemPtr, class _Arg> 
# 2450
struct __result_of_impl< true, false, _MemPtr, _Arg>  : public __result_of_memobj< __decay_t< _MemPtr> , typename __inv_unwrap< _Arg> ::type>  { 
# 2453
}; 
# 2455
template< class _MemPtr, class _Arg, class ..._Args> 
# 2456
struct __result_of_impl< false, true, _MemPtr, _Arg, _Args...>  : public __result_of_memfun< __decay_t< _MemPtr> , typename __inv_unwrap< _Arg> ::type, _Args...>  { 
# 2459
}; 
# 2462
struct __result_of_other_impl { 
# 2464
template< class _Fn, class ..._Args> static __result_of_success< __decltype((std::declval< _Fn> ()(std::declval< _Args> ()...))), __invoke_other>  _S_test(int); 
# 2469
template< class ...> static __failure_type _S_test(...); 
# 2471
}; 
# 2473
template< class _Functor, class ..._ArgTypes> 
# 2474
struct __result_of_impl< false, false, _Functor, _ArgTypes...>  : private __result_of_other_impl { 
# 2477
typedef __decltype((_S_test< _Functor, _ArgTypes...> (0))) type; 
# 2478
}; 
# 2481
template< class _Functor, class ..._ArgTypes> 
# 2482
struct __invoke_result : public __result_of_impl< is_member_object_pointer< typename remove_reference< _Functor> ::type> ::value, is_member_function_pointer< typename remove_reference< _Functor> ::type> ::value, _Functor, _ArgTypes...> ::type { 
# 2492
}; 
# 2494
template< class _Functor, class ..._ArgTypes> 
# 2495
struct result_of< _Functor (_ArgTypes ...)>  : public __invoke_result< _Functor, _ArgTypes...>  { 
# 2497
}; 
# 2501
template< size_t _Len, size_t _Align = __alignof__(typename __aligned_storage_msa< _Len> ::__type)> using aligned_storage_t = typename aligned_storage< _Len, _Align> ::type; 
# 2505
template< size_t _Len, class ..._Types> using aligned_union_t = typename aligned_union< _Len, _Types...> ::type; 
# 2509
template< class _Tp> using decay_t = typename decay< _Tp> ::type; 
# 2513
template< bool _Cond, class _Tp = void> using enable_if_t = typename enable_if< _Cond, _Tp> ::type; 
# 2517
template< bool _Cond, class _Iftrue, class _Iffalse> using conditional_t = typename conditional< _Cond, _Iftrue, _Iffalse> ::type; 
# 2521
template< class ..._Tp> using common_type_t = typename common_type< _Tp...> ::type; 
# 2525
template< class _Tp> using underlying_type_t = typename underlying_type< _Tp> ::type; 
# 2529
template< class _Tp> using result_of_t = typename result_of< _Tp> ::type; 
# 2536
template< class ...> using void_t = void; 
# 2540
template< class _Default, class _AlwaysVoid, 
# 2541
template< class ...>  class _Op, class ..._Args> 
# 2542
struct __detector { 
# 2544
using value_t = false_type; 
# 2545
using type = _Default; 
# 2546
}; 
# 2549
template< class _Default, template< class ...>  class _Op, class ...
# 2550
_Args> 
# 2551
struct __detector< _Default, __void_t< _Op< _Args...> > , _Op, _Args...>  { 
# 2553
using value_t = true_type; 
# 2554
using type = _Op< _Args...> ; 
# 2555
}; 
# 2558
template< class _Default, template< class ...>  class _Op, class ...
# 2559
_Args> using __detected_or = __detector< _Default, void, _Op, _Args...> ; 
# 2563
template< class _Default, template< class ...>  class _Op, class ...
# 2564
_Args> using __detected_or_t = typename __detector< _Default, void, _Op, _Args...> ::type; 
# 2584 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
template< class _Tp> struct __is_swappable; 
# 2587
template< class _Tp> struct __is_nothrow_swappable; 
# 2590
template< class ..._Elements> class tuple; 
# 2593
template< class > 
# 2594
struct __is_tuple_like_impl : public false_type { 
# 2595
}; 
# 2597
template< class ..._Tps> 
# 2598
struct __is_tuple_like_impl< tuple< _Tps...> >  : public true_type { 
# 2599
}; 
# 2602
template< class _Tp> 
# 2603
struct __is_tuple_like : public __is_tuple_like_impl< __remove_cvref_t< _Tp> > ::type { 
# 2605
}; 
# 2607
template< class _Tp> inline _Require< __not_< __is_tuple_like< _Tp> > , is_move_constructible< _Tp> , is_move_assignable< _Tp> >  swap(_Tp &, _Tp &) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_move_assignable< _Tp> > ::value); 
# 2617
template< class _Tp, size_t _Nm> inline __enable_if_t< __is_swappable< _Tp> ::value>  swap(_Tp (& __a)[_Nm], _Tp (& __b)[_Nm]) noexcept(__is_nothrow_swappable< _Tp> ::value); 
# 2624
namespace __swappable_details { 
# 2625
using std::swap;
# 2627
struct __do_is_swappable_impl { 
# 2629
template< class _Tp, class 
# 2630
 = __decltype((swap(std::declval< _Tp &> (), std::declval< _Tp &> ())))> static true_type 
# 2629
__test(int); 
# 2633
template< class > static false_type __test(...); 
# 2635
}; 
# 2637
struct __do_is_nothrow_swappable_impl { 
# 2639
template< class _Tp> static __bool_constant< noexcept(swap(std::declval< _Tp &> (), std::declval< _Tp &> ()))>  __test(int); 
# 2644
template< class > static false_type __test(...); 
# 2646
}; 
# 2648
}
# 2650
template< class _Tp> 
# 2651
struct __is_swappable_impl : public __swappable_details::__do_is_swappable_impl { 
# 2654
typedef __decltype((__test< _Tp> (0))) type; 
# 2655
}; 
# 2657
template< class _Tp> 
# 2658
struct __is_nothrow_swappable_impl : public __swappable_details::__do_is_nothrow_swappable_impl { 
# 2661
typedef __decltype((__test< _Tp> (0))) type; 
# 2662
}; 
# 2664
template< class _Tp> 
# 2665
struct __is_swappable : public __is_swappable_impl< _Tp> ::type { 
# 2667
}; 
# 2669
template< class _Tp> 
# 2670
struct __is_nothrow_swappable : public __is_nothrow_swappable_impl< _Tp> ::type { 
# 2672
}; 
# 2679
template< class _Tp> 
# 2680
struct is_swappable : public __is_swappable_impl< _Tp> ::type { 
# 2683
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 2685
}; 
# 2688
template< class _Tp> 
# 2689
struct is_nothrow_swappable : public __is_nothrow_swappable_impl< _Tp> ::type { 
# 2692
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 2694
}; 
# 2698
template< class _Tp> constexpr bool 
# 2699
is_swappable_v = (is_swappable< _Tp> ::value); 
# 2703
template< class _Tp> constexpr bool 
# 2704
is_nothrow_swappable_v = (is_nothrow_swappable< _Tp> ::value); 
# 2708
namespace __swappable_with_details { 
# 2709
using std::swap;
# 2711
struct __do_is_swappable_with_impl { 
# 2713
template< class _Tp, class _Up, class 
# 2714
 = __decltype((swap(std::declval< _Tp> (), std::declval< _Up> ()))), class 
# 2716
 = __decltype((swap(std::declval< _Up> (), std::declval< _Tp> ())))> static true_type 
# 2713
__test(int); 
# 2719
template< class , class > static false_type __test(...); 
# 2721
}; 
# 2723
struct __do_is_nothrow_swappable_with_impl { 
# 2725
template< class _Tp, class _Up> static __bool_constant< noexcept(swap(std::declval< _Tp> (), std::declval< _Up> ())) && noexcept(swap(std::declval< _Up> (), std::declval< _Tp> ()))>  __test(int); 
# 2732
template< class , class > static false_type __test(...); 
# 2734
}; 
# 2736
}
# 2738
template< class _Tp, class _Up> 
# 2739
struct __is_swappable_with_impl : public __swappable_with_details::__do_is_swappable_with_impl { 
# 2742
typedef __decltype((__test< _Tp, _Up> (0))) type; 
# 2743
}; 
# 2746
template< class _Tp> 
# 2747
struct __is_swappable_with_impl< _Tp &, _Tp &>  : public __swappable_details::__do_is_swappable_impl { 
# 2750
typedef __decltype((__test< _Tp &> (0))) type; 
# 2751
}; 
# 2753
template< class _Tp, class _Up> 
# 2754
struct __is_nothrow_swappable_with_impl : public __swappable_with_details::__do_is_nothrow_swappable_with_impl { 
# 2757
typedef __decltype((__test< _Tp, _Up> (0))) type; 
# 2758
}; 
# 2761
template< class _Tp> 
# 2762
struct __is_nothrow_swappable_with_impl< _Tp &, _Tp &>  : public __swappable_details::__do_is_nothrow_swappable_impl { 
# 2765
typedef __decltype((__test< _Tp &> (0))) type; 
# 2766
}; 
# 2769
template< class _Tp, class _Up> 
# 2770
struct is_swappable_with : public __is_swappable_with_impl< _Tp, _Up> ::type { 
# 2773
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "first template argument must be a complete class or an unbounded array");
# 2775
static_assert((std::__is_complete_or_unbounded(__type_identity< _Up> {})), "second template argument must be a complete class or an unbounded array");
# 2777
}; 
# 2780
template< class _Tp, class _Up> 
# 2781
struct is_nothrow_swappable_with : public __is_nothrow_swappable_with_impl< _Tp, _Up> ::type { 
# 2784
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "first template argument must be a complete class or an unbounded array");
# 2786
static_assert((std::__is_complete_or_unbounded(__type_identity< _Up> {})), "second template argument must be a complete class or an unbounded array");
# 2788
}; 
# 2792
template< class _Tp, class _Up> constexpr bool 
# 2793
is_swappable_with_v = (is_swappable_with< _Tp, _Up> ::value); 
# 2797
template< class _Tp, class _Up> constexpr bool 
# 2798
is_nothrow_swappable_with_v = (is_nothrow_swappable_with< _Tp, _Up> ::value); 
# 2807
template< class _Result, class _Ret, bool 
# 2808
 = is_void< _Ret> ::value, class  = void> 
# 2809
struct __is_invocable_impl : public false_type { }; 
# 2812
template< class _Result, class _Ret> 
# 2813
struct __is_invocable_impl< _Result, _Ret, true, __void_t< typename _Result::type> >  : public true_type { 
# 2817
}; 
# 2819
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
# 2822
template< class _Result, class _Ret> 
# 2823
struct __is_invocable_impl< _Result, _Ret, false, __void_t< typename _Result::type> >  { 
# 2830
private: static typename _Result::type _S_get(); 
# 2832
template< class _Tp> static void _S_conv(_Tp); 
# 2836
template< class _Tp, class  = __decltype((_S_conv< _Tp> ((_S_get)())))> static true_type _S_test(int); 
# 2840
template< class _Tp> static false_type _S_test(...); 
# 2845
public: using type = __decltype((_S_test< _Ret> (1))); 
# 2846
}; 
#pragma GCC diagnostic pop
# 2849
template< class _Fn, class ..._ArgTypes> 
# 2850
struct __is_invocable : public __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , void> ::type { 
# 2852
}; 
# 2854
template< class _Fn, class _Tp, class ..._Args> constexpr bool 
# 2855
__call_is_nt(__invoke_memfun_ref) 
# 2856
{ 
# 2857
using _Up = typename __inv_unwrap< _Tp> ::type; 
# 2858
return noexcept((std::declval< typename __inv_unwrap< _Tp> ::type> ().*std::declval< _Fn> ())(std::declval< _Args> ()...)); 
# 2860
} 
# 2862
template< class _Fn, class _Tp, class ..._Args> constexpr bool 
# 2863
__call_is_nt(__invoke_memfun_deref) 
# 2864
{ 
# 2865
return noexcept(((*std::declval< _Tp> ()).*std::declval< _Fn> ())(std::declval< _Args> ()...)); 
# 2867
} 
# 2869
template< class _Fn, class _Tp> constexpr bool 
# 2870
__call_is_nt(__invoke_memobj_ref) 
# 2871
{ 
# 2872
using _Up = typename __inv_unwrap< _Tp> ::type; 
# 2873
return noexcept((std::declval< typename __inv_unwrap< _Tp> ::type> ().*std::declval< _Fn> ())); 
# 2874
} 
# 2876
template< class _Fn, class _Tp> constexpr bool 
# 2877
__call_is_nt(__invoke_memobj_deref) 
# 2878
{ 
# 2879
return noexcept(((*std::declval< _Tp> ()).*std::declval< _Fn> ())); 
# 2880
} 
# 2882
template< class _Fn, class ..._Args> constexpr bool 
# 2883
__call_is_nt(__invoke_other) 
# 2884
{ 
# 2885
return noexcept(std::declval< _Fn> ()(std::declval< _Args> ()...)); 
# 2886
} 
# 2888
template< class _Result, class _Fn, class ..._Args> 
# 2889
struct __call_is_nothrow : public __bool_constant< std::__call_is_nt< _Fn, _Args...> (typename _Result::__invoke_type{})>  { 
# 2893
}; 
# 2895
template< class _Fn, class ..._Args> using __call_is_nothrow_ = __call_is_nothrow< __invoke_result< _Fn, _Args...> , _Fn, _Args...> ; 
# 2900
template< class _Fn, class ..._Args> 
# 2901
struct __is_nothrow_invocable : public __and_< __is_invocable< _Fn, _Args...> , __call_is_nothrow_< _Fn, _Args...> > ::type { 
# 2904
}; 
# 2906
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
struct __nonesuchbase { }; 
# 2909
struct __nonesuch : private __nonesuchbase { 
# 2910
~__nonesuch() = delete;
# 2911
__nonesuch(const __nonesuch &) = delete;
# 2912
void operator=(const __nonesuch &) = delete;
# 2913
}; 
#pragma GCC diagnostic pop
# 2920
template< class _Functor, class ..._ArgTypes> 
# 2921
struct invoke_result : public __invoke_result< _Functor, _ArgTypes...>  { 
# 2924
static_assert((std::__is_complete_or_unbounded(__type_identity< _Functor> {})), "_Functor must be a complete class or an unbounded array");
# 2926
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 2929
}; 
# 2932
template< class _Fn, class ..._Args> using invoke_result_t = typename invoke_result< _Fn, _Args...> ::type; 
# 2936
template< class _Fn, class ..._ArgTypes> 
# 2937
struct is_invocable : public __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , void> ::type { 
# 2940
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 2942
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 2945
}; 
# 2948
template< class _Ret, class _Fn, class ..._ArgTypes> 
# 2949
struct is_invocable_r : public __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , _Ret> ::type { 
# 2952
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 2954
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 2957
static_assert((std::__is_complete_or_unbounded(__type_identity< _Ret> {})), "_Ret must be a complete class or an unbounded array");
# 2959
}; 
# 2962
template< class _Fn, class ..._ArgTypes> 
# 2963
struct is_nothrow_invocable : public __and_< __is_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , void> , __call_is_nothrow_< _Fn, _ArgTypes...> > ::type { 
# 2967
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 2969
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 2972
}; 
# 2974
template< class _Result, class _Ret, class  = void> 
# 2975
struct __is_nt_invocable_impl : public false_type { }; 
# 2977
template< class _Result, class _Ret> 
# 2978
struct __is_nt_invocable_impl< _Result, _Ret, __void_t< typename _Result::type> >  : public __or_< is_void< _Ret> , __is_nothrow_convertible< typename _Result::type, _Ret> >  { 
# 2982
}; 
# 2985
template< class _Ret, class _Fn, class ..._ArgTypes> 
# 2986
struct is_nothrow_invocable_r : public __and_< __is_nt_invocable_impl< __invoke_result< _Fn, _ArgTypes...> , _Ret> , __call_is_nothrow_< _Fn, _ArgTypes...> > ::type { 
# 2990
static_assert((std::__is_complete_or_unbounded(__type_identity< _Fn> {})), "_Fn must be a complete class or an unbounded array");
# 2992
static_assert(((std::__is_complete_or_unbounded(__type_identity< _ArgTypes> {}) && ... )), "each argument type must be a complete class or an unbounded array");
# 2995
static_assert((std::__is_complete_or_unbounded(__type_identity< _Ret> {})), "_Ret must be a complete class or an unbounded array");
# 2997
}; 
# 3000
template< class _Fn, class ..._Args> constexpr bool 
# 3001
is_invocable_v = (is_invocable< _Fn, _Args...> ::value); 
# 3004
template< class _Fn, class ..._Args> constexpr bool 
# 3005
is_nothrow_invocable_v = (is_nothrow_invocable< _Fn, _Args...> ::value); 
# 3009
template< class _Ret, class _Fn, class ..._Args> constexpr bool 
# 3010
is_invocable_r_v = (is_invocable_r< _Ret, _Fn, _Args...> ::value); 
# 3014
template< class _Ret, class _Fn, class ..._Args> constexpr bool 
# 3015
is_nothrow_invocable_r_v = (is_nothrow_invocable_r< _Ret, _Fn, _Args...> ::value); 
# 3021
template< class _Tp> constexpr bool 
# 3022
is_void_v = (is_void< _Tp> ::value); 
# 3023
template< class _Tp> constexpr bool 
# 3024
is_null_pointer_v = (is_null_pointer< _Tp> ::value); 
# 3025
template< class _Tp> constexpr bool 
# 3026
is_integral_v = (is_integral< _Tp> ::value); 
# 3027
template< class _Tp> constexpr bool 
# 3028
is_floating_point_v = (is_floating_point< _Tp> ::value); 
# 3029
template< class _Tp> constexpr bool 
# 3030
is_array_v = (is_array< _Tp> ::value); 
# 3031
template< class _Tp> constexpr bool 
# 3032
is_pointer_v = (is_pointer< _Tp> ::value); 
# 3033
template< class _Tp> constexpr bool 
# 3034
is_lvalue_reference_v = (is_lvalue_reference< _Tp> ::value); 
# 3036
template< class _Tp> constexpr bool 
# 3037
is_rvalue_reference_v = (is_rvalue_reference< _Tp> ::value); 
# 3039
template< class _Tp> constexpr bool 
# 3040
is_member_object_pointer_v = (is_member_object_pointer< _Tp> ::value); 
# 3042
template< class _Tp> constexpr bool 
# 3043
is_member_function_pointer_v = (is_member_function_pointer< _Tp> ::value); 
# 3045
template< class _Tp> constexpr bool 
# 3046
is_enum_v = (is_enum< _Tp> ::value); 
# 3047
template< class _Tp> constexpr bool 
# 3048
is_union_v = (is_union< _Tp> ::value); 
# 3049
template< class _Tp> constexpr bool 
# 3050
is_class_v = (is_class< _Tp> ::value); 
# 3051
template< class _Tp> constexpr bool 
# 3052
is_function_v = (is_function< _Tp> ::value); 
# 3053
template< class _Tp> constexpr bool 
# 3054
is_reference_v = (is_reference< _Tp> ::value); 
# 3055
template< class _Tp> constexpr bool 
# 3056
is_arithmetic_v = (is_arithmetic< _Tp> ::value); 
# 3057
template< class _Tp> constexpr bool 
# 3058
is_fundamental_v = (is_fundamental< _Tp> ::value); 
# 3059
template< class _Tp> constexpr bool 
# 3060
is_object_v = (is_object< _Tp> ::value); 
# 3061
template< class _Tp> constexpr bool 
# 3062
is_scalar_v = (is_scalar< _Tp> ::value); 
# 3063
template< class _Tp> constexpr bool 
# 3064
is_compound_v = (is_compound< _Tp> ::value); 
# 3065
template< class _Tp> constexpr bool 
# 3066
is_member_pointer_v = (is_member_pointer< _Tp> ::value); 
# 3067
template< class _Tp> constexpr bool 
# 3068
is_const_v = (is_const< _Tp> ::value); 
# 3069
template< class _Tp> constexpr bool 
# 3070
is_volatile_v = (is_volatile< _Tp> ::value); 
# 3071
template< class _Tp> constexpr bool 
# 3072
is_trivial_v = (is_trivial< _Tp> ::value); 
# 3073
template< class _Tp> constexpr bool 
# 3074
is_trivially_copyable_v = (is_trivially_copyable< _Tp> ::value); 
# 3076
template< class _Tp> constexpr bool 
# 3077
is_standard_layout_v = (is_standard_layout< _Tp> ::value); 
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template< class _Tp> constexpr bool 
# 3082
is_pod_v = (is_pod< _Tp> ::value); 
# 3083
template< class _Tp> 
# 3084
[[__deprecated__]] constexpr bool 
# 3085
is_literal_type_v = (is_literal_type< _Tp> ::value); 
#pragma GCC diagnostic pop
template< class _Tp> constexpr bool 
# 3088
is_empty_v = (is_empty< _Tp> ::value); 
# 3089
template< class _Tp> constexpr bool 
# 3090
is_polymorphic_v = (is_polymorphic< _Tp> ::value); 
# 3091
template< class _Tp> constexpr bool 
# 3092
is_abstract_v = (is_abstract< _Tp> ::value); 
# 3093
template< class _Tp> constexpr bool 
# 3094
is_final_v = (is_final< _Tp> ::value); 
# 3095
template< class _Tp> constexpr bool 
# 3096
is_signed_v = (is_signed< _Tp> ::value); 
# 3097
template< class _Tp> constexpr bool 
# 3098
is_unsigned_v = (is_unsigned< _Tp> ::value); 
# 3099
template< class _Tp, class ..._Args> constexpr bool 
# 3100
is_constructible_v = (is_constructible< _Tp, _Args...> ::value); 
# 3102
template< class _Tp> constexpr bool 
# 3103
is_default_constructible_v = (is_default_constructible< _Tp> ::value); 
# 3105
template< class _Tp> constexpr bool 
# 3106
is_copy_constructible_v = (is_copy_constructible< _Tp> ::value); 
# 3108
template< class _Tp> constexpr bool 
# 3109
is_move_constructible_v = (is_move_constructible< _Tp> ::value); 
# 3111
template< class _Tp, class _Up> constexpr bool 
# 3112
is_assignable_v = (is_assignable< _Tp, _Up> ::value); 
# 3113
template< class _Tp> constexpr bool 
# 3114
is_copy_assignable_v = (is_copy_assignable< _Tp> ::value); 
# 3115
template< class _Tp> constexpr bool 
# 3116
is_move_assignable_v = (is_move_assignable< _Tp> ::value); 
# 3117
template< class _Tp> constexpr bool 
# 3118
is_destructible_v = (is_destructible< _Tp> ::value); 
# 3119
template< class _Tp, class ..._Args> constexpr bool 
# 3120
is_trivially_constructible_v = (is_trivially_constructible< _Tp, _Args...> ::value); 
# 3122
template< class _Tp> constexpr bool 
# 3123
is_trivially_default_constructible_v = (is_trivially_default_constructible< _Tp> ::value); 
# 3125
template< class _Tp> constexpr bool 
# 3126
is_trivially_copy_constructible_v = (is_trivially_copy_constructible< _Tp> ::value); 
# 3128
template< class _Tp> constexpr bool 
# 3129
is_trivially_move_constructible_v = (is_trivially_move_constructible< _Tp> ::value); 
# 3131
template< class _Tp, class _Up> constexpr bool 
# 3132
is_trivially_assignable_v = (is_trivially_assignable< _Tp, _Up> ::value); 
# 3134
template< class _Tp> constexpr bool 
# 3135
is_trivially_copy_assignable_v = (is_trivially_copy_assignable< _Tp> ::value); 
# 3137
template< class _Tp> constexpr bool 
# 3138
is_trivially_move_assignable_v = (is_trivially_move_assignable< _Tp> ::value); 
# 3140
template< class _Tp> constexpr bool 
# 3141
is_trivially_destructible_v = (is_trivially_destructible< _Tp> ::value); 
# 3143
template< class _Tp, class ..._Args> constexpr bool 
# 3144
is_nothrow_constructible_v = (is_nothrow_constructible< _Tp, _Args...> ::value); 
# 3146
template< class _Tp> constexpr bool 
# 3147
is_nothrow_default_constructible_v = (is_nothrow_default_constructible< _Tp> ::value); 
# 3149
template< class _Tp> constexpr bool 
# 3150
is_nothrow_copy_constructible_v = (is_nothrow_copy_constructible< _Tp> ::value); 
# 3152
template< class _Tp> constexpr bool 
# 3153
is_nothrow_move_constructible_v = (is_nothrow_move_constructible< _Tp> ::value); 
# 3155
template< class _Tp, class _Up> constexpr bool 
# 3156
is_nothrow_assignable_v = (is_nothrow_assignable< _Tp, _Up> ::value); 
# 3158
template< class _Tp> constexpr bool 
# 3159
is_nothrow_copy_assignable_v = (is_nothrow_copy_assignable< _Tp> ::value); 
# 3161
template< class _Tp> constexpr bool 
# 3162
is_nothrow_move_assignable_v = (is_nothrow_move_assignable< _Tp> ::value); 
# 3164
template< class _Tp> constexpr bool 
# 3165
is_nothrow_destructible_v = (is_nothrow_destructible< _Tp> ::value); 
# 3167
template< class _Tp> constexpr bool 
# 3168
has_virtual_destructor_v = (has_virtual_destructor< _Tp> ::value); 
# 3170
template< class _Tp> constexpr size_t 
# 3171
alignment_of_v = (alignment_of< _Tp> ::value); 
# 3172
template< class _Tp> constexpr size_t 
# 3173
rank_v = (rank< _Tp> ::value); 
# 3174
template< class _Tp, unsigned _Idx = 0U> constexpr size_t 
# 3175
extent_v = (extent< _Tp, _Idx> ::value); 
# 3177
template< class _Tp, class _Up> constexpr bool 
# 3178
is_same_v = __is_same(_Tp, _Up); 
# 3183
template< class _Base, class _Derived> constexpr bool 
# 3184
is_base_of_v = (is_base_of< _Base, _Derived> ::value); 
# 3185
template< class _From, class _To> constexpr bool 
# 3186
is_convertible_v = (is_convertible< _From, _To> ::value); 
# 3191
template< class _Tp> 
# 3192
struct has_unique_object_representations : public bool_constant< __has_unique_object_representations(remove_cv_t< remove_all_extents_t< _Tp> > )>  { 
# 3197
static_assert((std::__is_complete_or_unbounded(__type_identity< _Tp> {})), "template argument must be a complete class or an unbounded array");
# 3199
}; 
# 3201
template< class _Tp> constexpr bool 
# 3202
has_unique_object_representations_v = (has_unique_object_representations< _Tp> ::value); 
# 3209
template< class _Tp> 
# 3210
struct is_aggregate : public bool_constant< __is_aggregate(remove_cv_t< _Tp> )>  { 
# 3212
}; 
# 3215
template< class _Tp> constexpr bool 
# 3216
is_aggregate_v = (is_aggregate< _Tp> ::value); 
# 3480 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/type_traits" 3
}
# 59 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/move.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 74 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/move.h" 3
template< class _Tp> 
# 75
[[__nodiscard__]] constexpr _Tp &&
# 77
forward(typename remove_reference< _Tp> ::type &__t) noexcept 
# 78
{ return static_cast< _Tp &&>(__t); } 
# 86
template< class _Tp> 
# 87
[[__nodiscard__]] constexpr _Tp &&
# 89
forward(typename remove_reference< _Tp> ::type &&__t) noexcept 
# 90
{ 
# 91
static_assert((!std::is_lvalue_reference< _Tp> ::value), "template argument substituting _Tp must not be an lvalue reference type");
# 93
return static_cast< _Tp &&>(__t); 
# 94
} 
# 101
template< class _Tp> 
# 102
[[__nodiscard__]] constexpr typename remove_reference< _Tp> ::type &&
# 104
move(_Tp &&__t) noexcept 
# 105
{ return static_cast< typename remove_reference< _Tp> ::type &&>(__t); } 
# 108
template< class _Tp> 
# 109
struct __move_if_noexcept_cond : public __and_< __not_< is_nothrow_move_constructible< _Tp> > , is_copy_constructible< _Tp> > ::type { 
# 111
}; 
# 121 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/move.h" 3
template< class _Tp> 
# 122
[[__nodiscard__]] constexpr typename conditional< __move_if_noexcept_cond< _Tp> ::value, const _Tp &, _Tp &&> ::type 
# 125
move_if_noexcept(_Tp &__x) noexcept 
# 126
{ return std::move(__x); } 
# 142 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/move.h" 3
template< class _Tp> 
# 143
[[__nodiscard__]] constexpr _Tp *
# 145
addressof(_Tp &__r) noexcept 
# 146
{ return std::__addressof(__r); } 
# 150
template < typename _Tp >
    const _Tp * addressof ( const _Tp && ) = delete;
# 154
template< class _Tp, class _Up = _Tp> inline _Tp 
# 157
__exchange(_Tp &__obj, _Up &&__new_val) 
# 158
{ 
# 159
_Tp __old_val = std::move(__obj); 
# 160
__obj = std::forward< _Up> (__new_val); 
# 161
return __old_val; 
# 162
} 
# 186 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/move.h" 3
template< class _Tp> inline typename enable_if< __and_< __not_< __is_tuple_like< _Tp> > , is_move_constructible< _Tp> , is_move_assignable< _Tp> > ::value> ::type 
# 196
swap(_Tp &__a, _Tp &__b) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_move_assignable< _Tp> > ::value) 
# 199
{ 
# 204
_Tp __tmp = std::move(__a); 
# 205
__a = std::move(__b); 
# 206
__b = std::move(__tmp); 
# 207
} 
# 212
template< class _Tp, size_t _Nm> inline typename enable_if< __is_swappable< _Tp> ::value> ::type 
# 220
swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm]) noexcept(__is_nothrow_swappable< _Tp> ::value) 
# 222
{ 
# 223
for (size_t __n = (0); __n < _Nm; ++__n) { 
# 224
swap(__a[__n], __b[__n]); }  
# 225
} 
# 229
}
# 69 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_pair.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 80 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_pair.h" 3
struct piecewise_construct_t { explicit piecewise_construct_t() = default;}; 
# 83
constexpr inline piecewise_construct_t piecewise_construct = piecewise_construct_t(); 
# 89
template< class ...> class tuple; 
# 92
template< size_t ...> struct _Index_tuple; 
# 100
template< bool , class _T1, class _T2> 
# 101
struct _PCC { 
# 103
template< class _U1, class _U2> static constexpr bool 
# 104
_ConstructiblePair() 
# 105
{ 
# 106
return __and_< is_constructible< _T1, const _U1 &> , is_constructible< _T2, const _U2 &> > ::value; 
# 108
} 
# 110
template< class _U1, class _U2> static constexpr bool 
# 111
_ImplicitlyConvertiblePair() 
# 112
{ 
# 113
return __and_< is_convertible< const _U1 &, _T1> , is_convertible< const _U2 &, _T2> > ::value; 
# 115
} 
# 117
template< class _U1, class _U2> static constexpr bool 
# 118
_MoveConstructiblePair() 
# 119
{ 
# 120
return __and_< is_constructible< _T1, _U1 &&> , is_constructible< _T2, _U2 &&> > ::value; 
# 122
} 
# 124
template< class _U1, class _U2> static constexpr bool 
# 125
_ImplicitlyMoveConvertiblePair() 
# 126
{ 
# 127
return __and_< is_convertible< _U1 &&, _T1> , is_convertible< _U2 &&, _T2> > ::value; 
# 129
} 
# 131
template< bool __implicit, class _U1, class _U2> static constexpr bool 
# 132
_CopyMovePair() 
# 133
{ 
# 134
using __do_converts = __and_< is_convertible< const _U1 &, _T1> , is_convertible< _U2 &&, _T2> > ; 
# 136
using __converts = typename conditional< __implicit, __and_< is_convertible< const _U1 &, _T1> , is_convertible< _U2 &&, _T2> > , __not_< __and_< is_convertible< const _U1 &, _T1> , is_convertible< _U2 &&, _T2> > > > ::type; 
# 139
return __and_< is_constructible< _T1, const _U1 &> , is_constructible< _T2, _U2 &&> , typename conditional< __implicit, __and_< is_convertible< const _U1 &, _T1> , is_convertible< _U2 &&, _T2> > , __not_< __and_< is_convertible< const _U1 &, _T1> , is_convertible< _U2 &&, _T2> > > > ::type> ::value; 
# 143
} 
# 145
template< bool __implicit, class _U1, class _U2> static constexpr bool 
# 146
_MoveCopyPair() 
# 147
{ 
# 148
using __do_converts = __and_< is_convertible< _U1 &&, _T1> , is_convertible< const _U2 &, _T2> > ; 
# 150
using __converts = typename conditional< __implicit, __and_< is_convertible< _U1 &&, _T1> , is_convertible< const _U2 &, _T2> > , __not_< __and_< is_convertible< _U1 &&, _T1> , is_convertible< const _U2 &, _T2> > > > ::type; 
# 153
return __and_< is_constructible< _T1, _U1 &&> , is_constructible< _T2, const _U2 &&> , typename conditional< __implicit, __and_< is_convertible< _U1 &&, _T1> , is_convertible< const _U2 &, _T2> > , __not_< __and_< is_convertible< _U1 &&, _T1> , is_convertible< const _U2 &, _T2> > > > ::type> ::value; 
# 157
} 
# 158
}; 
# 160
template< class _T1, class _T2> 
# 161
struct _PCC< false, _T1, _T2>  { 
# 163
template< class _U1, class _U2> static constexpr bool 
# 164
_ConstructiblePair() 
# 165
{ 
# 166
return false; 
# 167
} 
# 169
template< class _U1, class _U2> static constexpr bool 
# 170
_ImplicitlyConvertiblePair() 
# 171
{ 
# 172
return false; 
# 173
} 
# 175
template< class _U1, class _U2> static constexpr bool 
# 176
_MoveConstructiblePair() 
# 177
{ 
# 178
return false; 
# 179
} 
# 181
template< class _U1, class _U2> static constexpr bool 
# 182
_ImplicitlyMoveConvertiblePair() 
# 183
{ 
# 184
return false; 
# 185
} 
# 186
}; 
# 189
template< class _U1, class _U2> class __pair_base { 
# 192
template< class _T1, class _T2> friend struct pair; 
# 193
__pair_base() = default;
# 194
~__pair_base() = default;
# 195
__pair_base(const __pair_base &) = default;
# 196
__pair_base &operator=(const __pair_base &) = delete;
# 198
}; 
# 210 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_pair.h" 3
template< class _T1, class _T2> 
# 211
struct pair : private __pair_base< _T1, _T2>  { 
# 214
typedef _T1 first_type; 
# 215
typedef _T2 second_type; 
# 217
_T1 first; 
# 218
_T2 second; 
# 225
template< class _U1 = _T1, class 
# 226
_U2 = _T2, typename enable_if< __and_< __is_implicitly_default_constructible< _U1> , __is_implicitly_default_constructible< _U2> > ::value, bool> ::type 
# 230
 = true> constexpr 
# 232
pair() : first(), second() 
# 233
{ } 
# 236
template< class _U1 = _T1, class 
# 237
_U2 = _T2, typename enable_if< __and_< is_default_constructible< _U1> , is_default_constructible< _U2> , __not_< __and_< __is_implicitly_default_constructible< _U1> , __is_implicitly_default_constructible< _U2> > > > ::value, bool> ::type 
# 244
 = false> constexpr explicit 
# 245
pair() : first(), second() 
# 246
{ } 
# 256 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_pair.h" 3
using _PCCP = _PCC< true, _T1, _T2> ; 
# 260
template< class _U1 = _T1, class _U2 = _T2, typename enable_if< _PCC< true, _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && _PCC< true, _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> (), bool> ::type 
# 265
 = true> constexpr 
# 266
pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) 
# 267
{ } 
# 270
template< class _U1 = _T1, class _U2 = _T2, typename enable_if< _PCC< true, _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && (!_PCC< true, _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> ()), bool> ::type 
# 275
 = false> constexpr explicit 
# 276
pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) 
# 277
{ } 
# 288 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_pair.h" 3
template< class _U1, class _U2> using _PCCFP = _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ; 
# 294
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> (), bool> ::type 
# 299
 = true> constexpr 
# 300
pair(const std::pair< _U1, _U2>  &__p) : first((__p.first)), second((__p.second)) 
# 301
{ } 
# 303
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ConstructiblePair< _U1, _U2> () && (!_PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyConvertiblePair< _U1, _U2> ()), bool> ::type 
# 308
 = false> constexpr explicit 
# 309
pair(const std::pair< _U1, _U2>  &__p) : first((__p.first)), second((__p.second)) 
# 310
{ } 
# 314
constexpr pair(const pair &) = default;
# 315
constexpr pair(pair &&) = default;
# 318
template< class _U1, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveCopyPair< true, _U1, _T2> (), bool> ::type 
# 321
 = true> constexpr 
# 322
pair(_U1 &&__x, const _T2 &__y) : first(std::forward< _U1> (__x)), second(__y) 
# 323
{ } 
# 325
template< class _U1, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveCopyPair< false, _U1, _T2> (), bool> ::type 
# 328
 = false> constexpr explicit 
# 329
pair(_U1 &&__x, const _T2 &__y) : first(std::forward< _U1> (__x)), second(__y) 
# 330
{ } 
# 332
template< class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _CopyMovePair< true, _T1, _U2> (), bool> ::type 
# 335
 = true> constexpr 
# 336
pair(const _T1 &__x, _U2 &&__y) : first(__x), second(std::forward< _U2> (__y)) 
# 337
{ } 
# 339
template< class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _CopyMovePair< false, _T1, _U2> (), bool> ::type 
# 342
 = false> explicit 
# 343
pair(const _T1 &__x, _U2 &&__y) : first(__x), second(std::forward< _U2> (__y)) 
# 344
{ } 
# 346
template< class _U1, class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && _PCC< true, _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> (), bool> ::type 
# 351
 = true> constexpr 
# 352
pair(_U1 &&__x, _U2 &&__y) : first(std::forward< _U1> (__x)), second(std::forward< _U2> (__y)) 
# 353
{ } 
# 355
template< class _U1, class _U2, typename enable_if< _PCC< true, _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && (!_PCC< true, _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> ()), bool> ::type 
# 360
 = false> constexpr explicit 
# 361
pair(_U1 &&__x, _U2 &&__y) : first(std::forward< _U1> (__x)), second(std::forward< _U2> (__y)) 
# 362
{ } 
# 365
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> (), bool> ::type 
# 370
 = true> constexpr 
# 371
pair(std::pair< _U1, _U2>  &&__p) : first(std::forward< _U1> ((__p.first))), second(std::forward< _U2> ((__p.second))) 
# 373
{ } 
# 375
template< class _U1, class _U2, typename enable_if< _PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _MoveConstructiblePair< _U1, _U2> () && (!_PCC< (!is_same< _T1, _U1> ::value) || (!is_same< _T2, _U2> ::value), _T1, _T2> ::template _ImplicitlyMoveConvertiblePair< _U1, _U2> ()), bool> ::type 
# 380
 = false> constexpr explicit 
# 381
pair(std::pair< _U1, _U2>  &&__p) : first(std::forward< _U1> ((__p.first))), second(std::forward< _U2> ((__p.second))) 
# 383
{ } 
# 385
template< class ..._Args1, class ..._Args2> pair(std::piecewise_construct_t, tuple< _Args1...> , tuple< _Args2...> ); 
# 390
pair &operator=(typename conditional< __and_< is_copy_assignable< _T1> , is_copy_assignable< _T2> > ::value, const pair &, const std::__nonesuch &> ::type 
# 393
__p) 
# 394
{ 
# 395
(first) = (__p.first); 
# 396
(second) = (__p.second); 
# 397
return *this; 
# 398
} 
# 401
pair &operator=(typename conditional< __and_< is_move_assignable< _T1> , is_move_assignable< _T2> > ::value, pair &&, std::__nonesuch &&> ::type 
# 404
__p) noexcept(__and_< is_nothrow_move_assignable< _T1> , is_nothrow_move_assignable< _T2> > ::value) 
# 407
{ 
# 408
(first) = std::forward< first_type> ((__p.first)); 
# 409
(second) = std::forward< second_type> ((__p.second)); 
# 410
return *this; 
# 411
} 
# 413
template< class _U1, class _U2> typename enable_if< __and_< is_assignable< _T1 &, const _U1 &> , is_assignable< _T2 &, const _U2 &> > ::value, pair &> ::type 
# 418
operator=(const std::pair< _U1, _U2>  &__p) 
# 419
{ 
# 420
(first) = (__p.first); 
# 421
(second) = (__p.second); 
# 422
return *this; 
# 423
} 
# 425
template< class _U1, class _U2> typename enable_if< __and_< is_assignable< _T1 &, _U1 &&> , is_assignable< _T2 &, _U2 &&> > ::value, pair &> ::type 
# 430
operator=(std::pair< _U1, _U2>  &&__p) 
# 431
{ 
# 432
(first) = std::forward< _U1> ((__p.first)); 
# 433
(second) = std::forward< _U2> ((__p.second)); 
# 434
return *this; 
# 435
} 
# 439
void swap(pair &__p) noexcept(__and_< __is_nothrow_swappable< _T1> , __is_nothrow_swappable< _T2> > ::value) 
# 442
{ 
# 443
using std::swap;
# 444
swap(first, __p.first); 
# 445
swap(second, __p.second); 
# 446
} 
# 449
private: template< class ..._Args1, std::size_t ..._Indexes1, class ...
# 450
_Args2, std::size_t ..._Indexes2> 
# 449
pair(tuple< _Args1...>  &, tuple< _Args2...>  &, _Index_tuple< _Indexes1...> , _Index_tuple< _Indexes2...> ); 
# 455
}; 
# 460
template< class _T1, class _T2> pair(_T1, _T2)->pair< _T1, _T2> ; 
# 464
template< class _T1, class _T2> constexpr bool 
# 466
operator==(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 467
{ return ((__x.first) == (__y.first)) && ((__x.second) == (__y.second)); } 
# 487 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_pair.h" 3
template< class _T1, class _T2> constexpr bool 
# 489
operator<(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 490
{ return ((__x.first) < (__y.first)) || ((!((__y.first) < (__x.first))) && ((__x.second) < (__y.second))); 
# 491
} 
# 494
template< class _T1, class _T2> constexpr bool 
# 496
operator!=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 497
{ return !(__x == __y); } 
# 500
template< class _T1, class _T2> constexpr bool 
# 502
operator>(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 503
{ return __y < __x; } 
# 506
template< class _T1, class _T2> constexpr bool 
# 508
operator<=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 509
{ return !(__y < __x); } 
# 512
template< class _T1, class _T2> constexpr bool 
# 514
operator>=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 515
{ return !(__x < __y); } 
# 524 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_pair.h" 3
template< class _T1, class _T2> inline typename enable_if< __and_< __is_swappable< _T1> , __is_swappable< _T2> > ::value> ::type 
# 533
swap(pair< _T1, _T2>  &__x, pair< _T1, _T2>  &__y) noexcept(noexcept(__x.swap(__y))) 
# 535
{ __x.swap(__y); } 
# 538
template < typename _T1, typename _T2 >
    typename enable_if < ! __and_ < __is_swappable < _T1 >,
          __is_swappable < _T2 > > :: value > :: type
    swap ( pair < _T1, _T2 > &, pair < _T1, _T2 > & ) = delete;
# 564 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_pair.h" 3
template< class _T1, class _T2> constexpr pair< typename __strip_reference_wrapper< __decay_t< _T1> > ::__type, typename __strip_reference_wrapper< __decay_t< _T2> > ::__type>  
# 567
make_pair(_T1 &&__x, _T2 &&__y) 
# 568
{ 
# 569
typedef typename __strip_reference_wrapper< __decay_t< _T1> > ::__type __ds_type1; 
# 570
typedef typename __strip_reference_wrapper< __decay_t< _T2> > ::__type __ds_type2; 
# 571
typedef pair< typename __strip_reference_wrapper< __decay_t< _T1> > ::__type, typename __strip_reference_wrapper< __decay_t< _T2> > ::__type>  __pair_type; 
# 572
return __pair_type(std::forward< _T1> (__x), std::forward< _T2> (__y)); 
# 573
} 
# 584 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_pair.h" 3
}
# 74 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator_base_types.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 93 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator_base_types.h" 3
struct input_iterator_tag { }; 
# 96
struct output_iterator_tag { }; 
# 99
struct forward_iterator_tag : public input_iterator_tag { }; 
# 103
struct bidirectional_iterator_tag : public forward_iterator_tag { }; 
# 107
struct random_access_iterator_tag : public bidirectional_iterator_tag { }; 
# 125 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator_base_types.h" 3
template< class _Category, class _Tp, class _Distance = ptrdiff_t, class 
# 126
_Pointer = _Tp *, class _Reference = _Tp &> 
# 127
struct iterator { 
# 130
typedef _Category iterator_category; 
# 132
typedef _Tp value_type; 
# 134
typedef _Distance difference_type; 
# 136
typedef _Pointer pointer; 
# 138
typedef _Reference reference; 
# 139
}; 
# 149 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator_base_types.h" 3
template< class _Iterator> struct iterator_traits; 
# 155
template< class _Iterator, class  = __void_t< > > 
# 156
struct __iterator_traits { }; 
# 160
template< class _Iterator> 
# 161
struct __iterator_traits< _Iterator, __void_t< typename _Iterator::iterator_category, typename _Iterator::value_type, typename _Iterator::difference_type, typename _Iterator::pointer, typename _Iterator::reference> >  { 
# 168
typedef typename _Iterator::iterator_category iterator_category; 
# 169
typedef typename _Iterator::value_type value_type; 
# 170
typedef typename _Iterator::difference_type difference_type; 
# 171
typedef typename _Iterator::pointer pointer; 
# 172
typedef typename _Iterator::reference reference; 
# 173
}; 
# 176
template< class _Iterator> 
# 177
struct iterator_traits : public __iterator_traits< _Iterator>  { 
# 178
}; 
# 209 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator_base_types.h" 3
template< class _Tp> 
# 210
struct iterator_traits< _Tp *>  { 
# 212
typedef random_access_iterator_tag iterator_category; 
# 213
typedef _Tp value_type; 
# 214
typedef ptrdiff_t difference_type; 
# 215
typedef _Tp *pointer; 
# 216
typedef _Tp &reference; 
# 217
}; 
# 220
template< class _Tp> 
# 221
struct iterator_traits< const _Tp *>  { 
# 223
typedef random_access_iterator_tag iterator_category; 
# 224
typedef _Tp value_type; 
# 225
typedef ptrdiff_t difference_type; 
# 226
typedef const _Tp *pointer; 
# 227
typedef const _Tp &reference; 
# 228
}; 
# 235
template< class _Iter> constexpr typename iterator_traits< _Iter> ::iterator_category 
# 238
__iterator_category(const _Iter &) 
# 239
{ return typename iterator_traits< _Iter> ::iterator_category(); } 
# 244
template< class _Iter> using __iterator_category_t = typename iterator_traits< _Iter> ::iterator_category; 
# 248
template< class _InIter> using _RequireInputIter = __enable_if_t< is_convertible< __iterator_category_t< _InIter> , input_iterator_tag> ::value> ; 
# 253
template< class _It, class 
# 254
_Cat = __iterator_category_t< _It> > 
# 255
struct __is_random_access_iter : public is_base_of< random_access_iterator_tag, _Cat>  { 
# 258
typedef is_base_of< std::random_access_iterator_tag, _Cat>  _Base; 
# 259
enum { __value = is_base_of< std::random_access_iterator_tag, _Cat> ::value}; 
# 260
}; 
# 269
}
# 67 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator_base_funcs.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 73
template< class > struct _List_iterator; 
# 74
template< class > struct _List_const_iterator; 
# 77
template< class _InputIterator> constexpr typename iterator_traits< _InputIterator> ::difference_type 
# 80
__distance(_InputIterator __first, _InputIterator __last, input_iterator_tag) 
# 82
{ 
# 86
typename iterator_traits< _InputIterator> ::difference_type __n = (0); 
# 87
while (__first != __last) 
# 88
{ 
# 89
++__first; 
# 90
++__n; 
# 91
}  
# 92
return __n; 
# 93
} 
# 95
template< class _RandomAccessIterator> constexpr typename iterator_traits< _RandomAccessIterator> ::difference_type 
# 98
__distance(_RandomAccessIterator __first, _RandomAccessIterator __last, random_access_iterator_tag) 
# 100
{ 
# 104
return __last - __first; 
# 105
} 
# 109
template< class _Tp> ptrdiff_t __distance(_List_iterator< _Tp> , _List_iterator< _Tp> , input_iterator_tag); 
# 115
template< class _Tp> ptrdiff_t __distance(_List_const_iterator< _Tp> , _List_const_iterator< _Tp> , input_iterator_tag); 
# 135 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator_base_funcs.h" 3
template< class _InputIterator> constexpr typename iterator_traits< _InputIterator> ::difference_type 
# 138
distance(_InputIterator __first, _InputIterator __last) 
# 139
{ 
# 141
return std::__distance(__first, __last, std::__iterator_category(__first)); 
# 143
} 
# 145
template< class _InputIterator, class _Distance> constexpr void 
# 147
__advance(_InputIterator &__i, _Distance __n, input_iterator_tag) 
# 148
{ 
# 151
do { if (__builtin_is_constant_evaluated()) { void __failed_assertion(); if (!((bool)(__n >= 0))) { __failed_assertion(); }  } else { ; }  } while (false); 
# 152
while (__n--) { 
# 153
++__i; }  
# 154
} 
# 156
template< class _BidirectionalIterator, class _Distance> constexpr void 
# 158
__advance(_BidirectionalIterator &__i, _Distance __n, bidirectional_iterator_tag) 
# 160
{ 
# 164
if (__n > 0) { 
# 165
while (__n--) { 
# 166
++__i; }  } else { 
# 168
while (__n++) { 
# 169
--__i; }  }  
# 170
} 
# 172
template< class _RandomAccessIterator, class _Distance> constexpr void 
# 174
__advance(_RandomAccessIterator &__i, _Distance __n, random_access_iterator_tag) 
# 176
{ 
# 180
if (__builtin_constant_p(__n) && (__n == 1)) { 
# 181
++__i; } else { 
# 182
if (__builtin_constant_p(__n) && (__n == (-1))) { 
# 183
--__i; } else { 
# 185
__i += __n; }  }  
# 186
} 
# 200 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator_base_funcs.h" 3
template< class _InputIterator, class _Distance> constexpr void 
# 202
advance(_InputIterator &__i, _Distance __n) 
# 203
{ 
# 205
typename iterator_traits< _InputIterator> ::difference_type __d = __n; 
# 206
std::__advance(__i, __d, std::__iterator_category(__i)); 
# 207
} 
# 211
template< class _InputIterator> constexpr _InputIterator 
# 213
next(_InputIterator __x, typename iterator_traits< _InputIterator> ::difference_type 
# 214
__n = 1) 
# 215
{ 
# 218
std::advance(__x, __n); 
# 219
return __x; 
# 220
} 
# 222
template< class _BidirectionalIterator> constexpr _BidirectionalIterator 
# 224
prev(_BidirectionalIterator __x, typename iterator_traits< _BidirectionalIterator> ::difference_type 
# 225
__n = 1) 
# 226
{ 
# 230
std::advance(__x, -__n); 
# 231
return __x; 
# 232
} 
# 237
}
# 42 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/ptr_traits.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 46
class __undefined; 
# 49
template< class _Tp> 
# 50
struct __get_first_arg { 
# 51
using type = __undefined; }; 
# 53
template< template< class , class ...>  class _Template, class _Tp, class ...
# 54
_Types> 
# 55
struct __get_first_arg< _Template< _Tp, _Types...> >  { 
# 56
using type = _Tp; }; 
# 58
template< class _Tp> using __get_first_arg_t = typename __get_first_arg< _Tp> ::type; 
# 62
template< class _Tp, class _Up> 
# 63
struct __replace_first_arg { 
# 64
}; 
# 66
template< template< class , class ...>  class _Template, class _Up, class 
# 67
_Tp, class ..._Types> 
# 68
struct __replace_first_arg< _Template< _Tp, _Types...> , _Up>  { 
# 69
using type = _Template< _Up, _Types...> ; }; 
# 71
template< class _Tp, class _Up> using __replace_first_arg_t = typename __replace_first_arg< _Tp, _Up> ::type; 
# 74
template< class _Tp> using __make_not_void = typename conditional< is_void< _Tp> ::value, __undefined, _Tp> ::type; 
# 82
template< class _Ptr> 
# 83
struct pointer_traits { 
# 87
private: 
# 86
template< class _Tp> using __element_type = typename _Tp::element_type; 
# 89
template< class _Tp> using __difference_type = typename _Tp::difference_type; 
# 92
template< class _Tp, class _Up, class  = void> 
# 93
struct __rebind : public __replace_first_arg< _Tp, _Up>  { }; 
# 95
template< class _Tp, class _Up> 
# 96
struct __rebind< _Tp, _Up, __void_t< typename _Tp::template rebind< _Up> > >  { 
# 97
using type = typename _Tp::template rebind< _Up> ; }; 
# 101
public: using pointer = _Ptr; 
# 104
using element_type = __detected_or_t< __get_first_arg_t< _Ptr> , __element_type, _Ptr> ; 
# 108
using difference_type = __detected_or_t< ptrdiff_t, __difference_type, _Ptr> ; 
# 112
template< class _Up> using rebind = typename __rebind< _Ptr, _Up> ::type; 
# 116
static _Ptr pointer_to(__make_not_void< element_type>  &__e) 
# 117
{ return _Ptr::pointer_to(__e); } 
# 119
static_assert((!is_same< element_type, __undefined> ::value), "pointer type defines element_type or is like SomePointer<T, Args>");
# 121
}; 
# 127
template< class _Tp> 
# 128
struct pointer_traits< _Tp *>  { 
# 131
typedef _Tp *pointer; 
# 133
typedef _Tp element_type; 
# 135
typedef ptrdiff_t difference_type; 
# 137
template< class _Up> using rebind = _Up *; 
# 146
static pointer pointer_to(__make_not_void< element_type>  &__r) noexcept 
# 147
{ return std::addressof(__r); } 
# 148
}; 
# 151
template< class _Ptr, class _Tp> using __ptr_rebind = typename pointer_traits< _Ptr> ::template rebind< _Tp> ; 
# 154
template< class _Tp> constexpr _Tp *
# 156
__to_address(_Tp *__ptr) noexcept 
# 157
{ 
# 158
static_assert((!std::is_function< _Tp> ::value), "not a function pointer");
# 159
return __ptr; 
# 160
} 
# 163
template< class _Ptr> constexpr typename pointer_traits< _Ptr> ::element_type *
# 165
__to_address(const _Ptr &__ptr) 
# 166
{ return std::__to_address(__ptr.operator->()); } 
# 211 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/ptr_traits.h" 3
}
# 87 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 126 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Iterator> 
# 127
class reverse_iterator : public iterator< typename iterator_traits< _Iterator> ::iterator_category, typename iterator_traits< _Iterator> ::value_type, typename iterator_traits< _Iterator> ::difference_type, typename iterator_traits< _Iterator> ::pointer, typename iterator_traits< _Iterator> ::reference>  { 
# 134
template< class _Iter> friend class reverse_iterator; 
# 146 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
protected: _Iterator current; 
# 148
typedef iterator_traits< _Iterator>  __traits_type; 
# 151
public: typedef _Iterator iterator_type; 
# 152
typedef typename iterator_traits< _Iterator> ::difference_type difference_type; 
# 153
typedef typename iterator_traits< _Iterator> ::pointer pointer; 
# 154
typedef typename iterator_traits< _Iterator> ::reference reference; 
# 174 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
constexpr reverse_iterator() : current() { } 
# 180
constexpr explicit reverse_iterator(iterator_type __x) : current(__x) { } 
# 186
constexpr reverse_iterator(const reverse_iterator &__x) : current(__x.current) 
# 187
{ } 
# 190
reverse_iterator &operator=(const reverse_iterator &) = default;
# 197
template< class _Iter> constexpr 
# 202
reverse_iterator(const reverse_iterator< _Iter>  &__x) : current((__x.current)) 
# 203
{ } 
# 206
template< class _Iter> constexpr reverse_iterator &
# 213
operator=(const reverse_iterator< _Iter>  &__x) 
# 214
{ 
# 215
(current) = (__x.current); 
# 216
return *this; 
# 217
} 
# 224
constexpr iterator_type base() const 
# 225
{ return current; } 
# 238 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
constexpr reference operator*() const 
# 239
{ 
# 240
_Iterator __tmp = current; 
# 241
return *(--__tmp); 
# 242
} 
# 250
constexpr pointer operator->() const 
# 255
{ 
# 258
_Iterator __tmp = current; 
# 259
--__tmp; 
# 260
return _S_to_pointer(__tmp); 
# 261
} 
# 269
constexpr reverse_iterator &operator++() 
# 270
{ 
# 271
--(current); 
# 272
return *this; 
# 273
} 
# 281
constexpr reverse_iterator operator++(int) 
# 282
{ 
# 283
reverse_iterator __tmp = *this; 
# 284
--(current); 
# 285
return __tmp; 
# 286
} 
# 294
constexpr reverse_iterator &operator--() 
# 295
{ 
# 296
++(current); 
# 297
return *this; 
# 298
} 
# 306
constexpr reverse_iterator operator--(int) 
# 307
{ 
# 308
reverse_iterator __tmp = *this; 
# 309
++(current); 
# 310
return __tmp; 
# 311
} 
# 319
constexpr reverse_iterator operator+(difference_type __n) const 
# 320
{ return ((reverse_iterator)((current) - __n)); } 
# 329
constexpr reverse_iterator &operator+=(difference_type __n) 
# 330
{ 
# 331
(current) -= __n; 
# 332
return *this; 
# 333
} 
# 341
constexpr reverse_iterator operator-(difference_type __n) const 
# 342
{ return ((reverse_iterator)((current) + __n)); } 
# 351
constexpr reverse_iterator &operator-=(difference_type __n) 
# 352
{ 
# 353
(current) += __n; 
# 354
return *this; 
# 355
} 
# 363
constexpr reference operator[](difference_type __n) const 
# 364
{ return *((*this) + __n); } 
# 394 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
private: 
# 392
template< class _Tp> static constexpr _Tp *
# 394
_S_to_pointer(_Tp *__p) 
# 395
{ return __p; } 
# 397
template< class _Tp> static constexpr pointer 
# 399
_S_to_pointer(_Tp __t) 
# 400
{ return __t.operator->(); } 
# 401
}; 
# 414 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Iterator> constexpr bool 
# 416
operator==(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 417
__y) 
# 418
{ return __x.base() == __y.base(); } 
# 420
template< class _Iterator> constexpr bool 
# 422
operator<(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 423
__y) 
# 424
{ return __y.base() < __x.base(); } 
# 426
template< class _Iterator> constexpr bool 
# 428
operator!=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 429
__y) 
# 430
{ return !(__x == __y); } 
# 432
template< class _Iterator> constexpr bool 
# 434
operator>(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 435
__y) 
# 436
{ return __y < __x; } 
# 438
template< class _Iterator> constexpr bool 
# 440
operator<=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 441
__y) 
# 442
{ return !(__y < __x); } 
# 444
template< class _Iterator> constexpr bool 
# 446
operator>=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 447
__y) 
# 448
{ return !(__x < __y); } 
# 453
template< class _IteratorL, class _IteratorR> constexpr bool 
# 455
operator==(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 456
__y) 
# 457
{ return __x.base() == __y.base(); } 
# 459
template< class _IteratorL, class _IteratorR> constexpr bool 
# 461
operator<(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 462
__y) 
# 463
{ return __x.base() > __y.base(); } 
# 465
template< class _IteratorL, class _IteratorR> constexpr bool 
# 467
operator!=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 468
__y) 
# 469
{ return __x.base() != __y.base(); } 
# 471
template< class _IteratorL, class _IteratorR> constexpr bool 
# 473
operator>(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 474
__y) 
# 475
{ return __x.base() < __y.base(); } 
# 477
template< class _IteratorL, class _IteratorR> constexpr bool 
# 479
operator<=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 480
__y) 
# 481
{ return __x.base() >= __y.base(); } 
# 483
template< class _IteratorL, class _IteratorR> constexpr bool 
# 485
operator>=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 486
__y) 
# 487
{ return __x.base() <= __y.base(); } 
# 555 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _IteratorL, class _IteratorR> constexpr auto 
# 557
operator-(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 558
__y)->__decltype((__y.base() - __x.base())) 
# 560
{ return __y.base() - __x.base(); } 
# 563
template< class _Iterator> constexpr reverse_iterator< _Iterator>  
# 565
operator+(typename reverse_iterator< _Iterator> ::difference_type __n, const reverse_iterator< _Iterator>  &
# 566
__x) 
# 567
{ return ((reverse_iterator< _Iterator> )(__x.base() - __n)); } 
# 571
template< class _Iterator> constexpr reverse_iterator< _Iterator>  
# 573
__make_reverse_iterator(_Iterator __i) 
# 574
{ return ((reverse_iterator< _Iterator> )(__i)); } 
# 582
template< class _Iterator> constexpr reverse_iterator< _Iterator>  
# 584
make_reverse_iterator(_Iterator __i) 
# 585
{ return ((reverse_iterator< _Iterator> )(__i)); } 
# 596 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Iterator> auto 
# 599
__niter_base(reverse_iterator< _Iterator>  __it)->__decltype((__make_reverse_iterator(__niter_base(__it.base())))) 
# 601
{ return __make_reverse_iterator(__niter_base(__it.base())); } 
# 603
template< class _Iterator> 
# 604
struct __is_move_iterator< reverse_iterator< _Iterator> >  : public std::__is_move_iterator< _Iterator>  { 
# 606
}; 
# 608
template< class _Iterator> auto 
# 611
__miter_base(reverse_iterator< _Iterator>  __it)->__decltype((__make_reverse_iterator(__miter_base(__it.base())))) 
# 613
{ return __make_reverse_iterator(__miter_base(__it.base())); } 
# 627 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Container> 
# 628
class back_insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 632
protected: _Container *container; 
# 636
public: typedef _Container container_type; 
# 645
explicit back_insert_iterator(_Container &__x) : container(std::__addressof(__x)) 
# 646
{ } 
# 669 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
back_insert_iterator &operator=(const typename _Container::value_type &__value) 
# 670
{ 
# 671
(container)->push_back(__value); 
# 672
return *this; 
# 673
} 
# 677
back_insert_iterator &operator=(typename _Container::value_type &&__value) 
# 678
{ 
# 679
(container)->push_back(std::move(__value)); 
# 680
return *this; 
# 681
} 
# 687
back_insert_iterator &operator*() 
# 688
{ return *this; } 
# 693
back_insert_iterator &operator++() 
# 694
{ return *this; } 
# 699
back_insert_iterator operator++(int) 
# 700
{ return *this; } 
# 701
}; 
# 714 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Container> inline back_insert_iterator< _Container>  
# 717
back_inserter(_Container &__x) 
# 718
{ return ((back_insert_iterator< _Container> )(__x)); } 
# 730 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Container> 
# 731
class front_insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 735
protected: _Container *container; 
# 739
public: typedef _Container container_type; 
# 748
explicit front_insert_iterator(_Container &__x) : container(std::__addressof(__x)) 
# 749
{ } 
# 772 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
front_insert_iterator &operator=(const typename _Container::value_type &__value) 
# 773
{ 
# 774
(container)->push_front(__value); 
# 775
return *this; 
# 776
} 
# 780
front_insert_iterator &operator=(typename _Container::value_type &&__value) 
# 781
{ 
# 782
(container)->push_front(std::move(__value)); 
# 783
return *this; 
# 784
} 
# 790
front_insert_iterator &operator*() 
# 791
{ return *this; } 
# 796
front_insert_iterator &operator++() 
# 797
{ return *this; } 
# 802
front_insert_iterator operator++(int) 
# 803
{ return *this; } 
# 804
}; 
# 817 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Container> inline front_insert_iterator< _Container>  
# 820
front_inserter(_Container &__x) 
# 821
{ return ((front_insert_iterator< _Container> )(__x)); } 
# 837 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Container> 
# 838
class insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 848
typedef typename _Container::iterator _Iter; 
# 851
protected: _Container *container; 
# 852
_Iter iter; 
# 857
public: typedef _Container container_type; 
# 870 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
insert_iterator(_Container &__x, _Iter __i) : container(std::__addressof(__x)), iter(__i) 
# 871
{ } 
# 907 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
insert_iterator &operator=(const typename _Container::value_type &__value) 
# 908
{ 
# 909
(iter) = (container)->insert(iter, __value); 
# 910
++(iter); 
# 911
return *this; 
# 912
} 
# 916
insert_iterator &operator=(typename _Container::value_type &&__value) 
# 917
{ 
# 918
(iter) = (container)->insert(iter, std::move(__value)); 
# 919
++(iter); 
# 920
return *this; 
# 921
} 
# 927
insert_iterator &operator*() 
# 928
{ return *this; } 
# 933
insert_iterator &operator++() 
# 934
{ return *this; } 
# 939
insert_iterator &operator++(int) 
# 940
{ return *this; } 
# 941
}; 
# 961 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Container> inline insert_iterator< _Container>  
# 963
inserter(_Container &__x, typename _Container::iterator __i) 
# 964
{ return insert_iterator< _Container> (__x, __i); } 
# 970
}
# 972
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 983 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Iterator, class _Container> 
# 984
class __normal_iterator { 
# 987
protected: _Iterator _M_current; 
# 989
typedef std::iterator_traits< _Iterator>  __traits_type; 
# 992
public: typedef _Iterator iterator_type; 
# 993
typedef typename std::iterator_traits< _Iterator> ::iterator_category iterator_category; 
# 994
typedef typename std::iterator_traits< _Iterator> ::value_type value_type; 
# 995
typedef typename std::iterator_traits< _Iterator> ::difference_type difference_type; 
# 996
typedef typename std::iterator_traits< _Iterator> ::reference reference; 
# 997
typedef typename std::iterator_traits< _Iterator> ::pointer pointer; 
# 1003
constexpr __normal_iterator() noexcept : _M_current(_Iterator()) 
# 1004
{ } 
# 1007
explicit __normal_iterator(const _Iterator &__i) noexcept : _M_current(__i) 
# 1008
{ } 
# 1011
template< class _Iter> 
# 1013
__normal_iterator(const __normal_iterator< _Iter, typename __enable_if< std::__are_same< _Iter, typename _Container::pointer> ::__value, _Container> ::__type>  &
# 1016
__i) noexcept : _M_current(__i.base()) 
# 1017
{ } 
# 1022
reference operator*() const noexcept 
# 1023
{ return *(_M_current); } 
# 1027
pointer operator->() const noexcept 
# 1028
{ return _M_current; } 
# 1032
__normal_iterator &operator++() noexcept 
# 1033
{ 
# 1034
++(_M_current); 
# 1035
return *this; 
# 1036
} 
# 1040
__normal_iterator operator++(int) noexcept 
# 1041
{ return ((__normal_iterator)((_M_current)++)); } 
# 1046
__normal_iterator &operator--() noexcept 
# 1047
{ 
# 1048
--(_M_current); 
# 1049
return *this; 
# 1050
} 
# 1054
__normal_iterator operator--(int) noexcept 
# 1055
{ return ((__normal_iterator)((_M_current)--)); } 
# 1060
reference operator[](difference_type __n) const noexcept 
# 1061
{ return (_M_current)[__n]; } 
# 1065
__normal_iterator &operator+=(difference_type __n) noexcept 
# 1066
{ (_M_current) += __n; return *this; } 
# 1070
__normal_iterator operator+(difference_type __n) const noexcept 
# 1071
{ return ((__normal_iterator)((_M_current) + __n)); } 
# 1075
__normal_iterator &operator-=(difference_type __n) noexcept 
# 1076
{ (_M_current) -= __n; return *this; } 
# 1080
__normal_iterator operator-(difference_type __n) const noexcept 
# 1081
{ return ((__normal_iterator)((_M_current) - __n)); } 
# 1085
const _Iterator &base() const noexcept 
# 1086
{ return _M_current; } 
# 1087
}; 
# 1115 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 1118
operator==(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1119
__rhs) noexcept 
# 1121
{ return __lhs.base() == __rhs.base(); } 
# 1123
template< class _Iterator, class _Container> inline bool 
# 1126
operator==(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1127
__rhs) noexcept 
# 1129
{ return __lhs.base() == __rhs.base(); } 
# 1131
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 1134
operator!=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1135
__rhs) noexcept 
# 1137
{ return __lhs.base() != __rhs.base(); } 
# 1139
template< class _Iterator, class _Container> inline bool 
# 1142
operator!=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1143
__rhs) noexcept 
# 1145
{ return __lhs.base() != __rhs.base(); } 
# 1148
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 1150
operator<(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1151
__rhs) noexcept 
# 1153
{ return __lhs.base() < __rhs.base(); } 
# 1155
template< class _Iterator, class _Container> inline bool 
# 1158
operator<(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1159
__rhs) noexcept 
# 1161
{ return __lhs.base() < __rhs.base(); } 
# 1163
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 1165
operator>(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1166
__rhs) noexcept 
# 1168
{ return __lhs.base() > __rhs.base(); } 
# 1170
template< class _Iterator, class _Container> inline bool 
# 1173
operator>(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1174
__rhs) noexcept 
# 1176
{ return __lhs.base() > __rhs.base(); } 
# 1178
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 1180
operator<=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1181
__rhs) noexcept 
# 1183
{ return __lhs.base() <= __rhs.base(); } 
# 1185
template< class _Iterator, class _Container> inline bool 
# 1188
operator<=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1189
__rhs) noexcept 
# 1191
{ return __lhs.base() <= __rhs.base(); } 
# 1193
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 1195
operator>=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1196
__rhs) noexcept 
# 1198
{ return __lhs.base() >= __rhs.base(); } 
# 1200
template< class _Iterator, class _Container> inline bool 
# 1203
operator>=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1204
__rhs) noexcept 
# 1206
{ return __lhs.base() >= __rhs.base(); } 
# 1213
template< class _IteratorL, class _IteratorR, class _Container> inline auto 
# 1218
operator-(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 1219
__rhs) noexcept->__decltype((__lhs.base() - __rhs.base())) 
# 1226
{ return __lhs.base() - __rhs.base(); } 
# 1228
template< class _Iterator, class _Container> inline typename __normal_iterator< _Iterator, _Container> ::difference_type 
# 1231
operator-(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 1232
__rhs) noexcept 
# 1234
{ return __lhs.base() - __rhs.base(); } 
# 1236
template< class _Iterator, class _Container> inline __normal_iterator< _Iterator, _Container>  
# 1239
operator+(typename __normal_iterator< _Iterator, _Container> ::difference_type 
# 1240
__n, const __normal_iterator< _Iterator, _Container>  &__i) noexcept 
# 1242
{ return ((__normal_iterator< _Iterator, _Container> )(__i.base() + __n)); } 
# 1245
}
# 1247
namespace std __attribute((__visibility__("default"))) { 
# 1251
template< class _Iterator, class _Container> _Iterator 
# 1254
__niter_base(__gnu_cxx::__normal_iterator< _Iterator, _Container>  __it) noexcept(std::is_nothrow_copy_constructible< _Iterator> ::value) 
# 1256
{ return __it.base(); } 
# 1305 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
namespace __detail { 
# 1321 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
}
# 1332 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Iterator> 
# 1333
class move_iterator { 
# 1338
_Iterator _M_current; 
# 1340
using __traits_type = iterator_traits< _Iterator> ; 
# 1342
using __base_ref = typename iterator_traits< _Iterator> ::reference; 
# 1345
template< class _Iter2> friend class move_iterator; 
# 1357 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
public: using iterator_type = _Iterator; 
# 1367 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
typedef typename iterator_traits< _Iterator> ::iterator_category iterator_category; 
# 1368
typedef typename iterator_traits< _Iterator> ::value_type value_type; 
# 1369
typedef typename iterator_traits< _Iterator> ::difference_type difference_type; 
# 1371
typedef _Iterator pointer; 
# 1376
typedef typename conditional< is_reference< __base_ref> ::value, typename remove_reference< __base_ref> ::type &&, __base_ref> ::type reference; 
# 1380
constexpr move_iterator() : _M_current() 
# 1381
{ } 
# 1384
constexpr explicit move_iterator(iterator_type __i) : _M_current(std::move(__i)) 
# 1385
{ } 
# 1387
template< class _Iter> constexpr 
# 1392
move_iterator(const move_iterator< _Iter>  &__i) : _M_current((__i._M_current)) 
# 1393
{ } 
# 1395
template< class _Iter> constexpr move_iterator &
# 1401
operator=(const move_iterator< _Iter>  &__i) 
# 1402
{ 
# 1403
(_M_current) = (__i._M_current); 
# 1404
return *this; 
# 1405
} 
# 1409
constexpr iterator_type base() const 
# 1410
{ return _M_current; } 
# 1425 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
constexpr reference operator*() const 
# 1429
{ return static_cast< reference>(*(_M_current)); } 
# 1433
constexpr pointer operator->() const 
# 1434
{ return _M_current; } 
# 1437
constexpr move_iterator &operator++() 
# 1438
{ 
# 1439
++(_M_current); 
# 1440
return *this; 
# 1441
} 
# 1444
constexpr move_iterator operator++(int) 
# 1445
{ 
# 1446
move_iterator __tmp = *this; 
# 1447
++(_M_current); 
# 1448
return __tmp; 
# 1449
} 
# 1458
constexpr move_iterator &operator--() 
# 1459
{ 
# 1460
--(_M_current); 
# 1461
return *this; 
# 1462
} 
# 1465
constexpr move_iterator operator--(int) 
# 1466
{ 
# 1467
move_iterator __tmp = *this; 
# 1468
--(_M_current); 
# 1469
return __tmp; 
# 1470
} 
# 1473
constexpr move_iterator operator+(difference_type __n) const 
# 1474
{ return ((move_iterator)((_M_current) + __n)); } 
# 1477
constexpr move_iterator &operator+=(difference_type __n) 
# 1478
{ 
# 1479
(_M_current) += __n; 
# 1480
return *this; 
# 1481
} 
# 1484
constexpr move_iterator operator-(difference_type __n) const 
# 1485
{ return ((move_iterator)((_M_current) - __n)); } 
# 1488
constexpr move_iterator &operator-=(difference_type __n) 
# 1489
{ 
# 1490
(_M_current) -= __n; 
# 1491
return *this; 
# 1492
} 
# 1495
constexpr reference operator[](difference_type __n) const 
# 1499
{ return std::move((_M_current)[__n]); } 
# 1529 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
}; 
# 1531
template< class _IteratorL, class _IteratorR> constexpr bool 
# 1533
operator==(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1534
__y) 
# 1538
{ return __x.base() == __y.base(); } 
# 1548 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _IteratorL, class _IteratorR> constexpr bool 
# 1550
operator!=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1551
__y) 
# 1552
{ return !(__x == __y); } 
# 1555
template< class _IteratorL, class _IteratorR> constexpr bool 
# 1557
operator<(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1558
__y) 
# 1562
{ return __x.base() < __y.base(); } 
# 1564
template< class _IteratorL, class _IteratorR> constexpr bool 
# 1566
operator<=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1567
__y) 
# 1571
{ return !(__y < __x); } 
# 1573
template< class _IteratorL, class _IteratorR> constexpr bool 
# 1575
operator>(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1576
__y) 
# 1580
{ return __y < __x; } 
# 1582
template< class _IteratorL, class _IteratorR> constexpr bool 
# 1584
operator>=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1585
__y) 
# 1589
{ return !(__x < __y); } 
# 1599 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Iterator> constexpr bool 
# 1601
operator==(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1602
__y) 
# 1603
{ return __x.base() == __y.base(); } 
# 1605
template< class _Iterator> constexpr bool 
# 1607
operator!=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1608
__y) 
# 1609
{ return !(__x == __y); } 
# 1611
template< class _Iterator> constexpr bool 
# 1613
operator<(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1614
__y) 
# 1615
{ return __x.base() < __y.base(); } 
# 1617
template< class _Iterator> constexpr bool 
# 1619
operator<=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1620
__y) 
# 1621
{ return !(__y < __x); } 
# 1623
template< class _Iterator> constexpr bool 
# 1625
operator>(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1626
__y) 
# 1627
{ return __y < __x; } 
# 1629
template< class _Iterator> constexpr bool 
# 1631
operator>=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1632
__y) 
# 1633
{ return !(__x < __y); } 
# 1637
template< class _IteratorL, class _IteratorR> constexpr auto 
# 1639
operator-(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1640
__y)->__decltype((__x.base() - __y.base())) 
# 1642
{ return __x.base() - __y.base(); } 
# 1644
template< class _Iterator> constexpr move_iterator< _Iterator>  
# 1646
operator+(typename move_iterator< _Iterator> ::difference_type __n, const move_iterator< _Iterator>  &
# 1647
__x) 
# 1648
{ return __x + __n; } 
# 1650
template< class _Iterator> constexpr move_iterator< _Iterator>  
# 1652
make_move_iterator(_Iterator __i) 
# 1653
{ return ((move_iterator< _Iterator> )(std::move(__i))); } 
# 1655
template< class _Iterator, class _ReturnType = typename conditional< __move_if_noexcept_cond< typename iterator_traits< _Iterator> ::value_type> ::value, _Iterator, move_iterator< _Iterator> > ::type> constexpr _ReturnType 
# 1660
__make_move_if_noexcept_iterator(_Iterator __i) 
# 1661
{ return (_ReturnType)__i; } 
# 1665
template< class _Tp, class _ReturnType = typename conditional< __move_if_noexcept_cond< _Tp> ::value, const _Tp *, move_iterator< _Tp *> > ::type> constexpr _ReturnType 
# 1669
__make_move_if_noexcept_iterator(_Tp *__i) 
# 1670
{ return (_ReturnType)__i; } 
# 2340 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _Iterator> auto 
# 2342
__niter_base(move_iterator< _Iterator>  __it)->__decltype((make_move_iterator(__niter_base(__it.base())))) 
# 2344
{ return make_move_iterator(__niter_base(__it.base())); } 
# 2346
template< class _Iterator> 
# 2347
struct __is_move_iterator< move_iterator< _Iterator> >  { 
# 2349
enum { __value = 1}; 
# 2350
typedef __true_type __type; 
# 2351
}; 
# 2353
template< class _Iterator> auto 
# 2355
__miter_base(move_iterator< _Iterator>  __it)->__decltype((__miter_base(__it.base()))) 
# 2357
{ return __miter_base(__it.base()); } 
# 2370 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_iterator.h" 3
template< class _InputIterator> using __iter_key_t = remove_const_t< typename iterator_traits< _InputIterator> ::value_type::first_type> ; 
# 2374
template< class _InputIterator> using __iter_val_t = typename iterator_traits< _InputIterator> ::value_type::second_type; 
# 2378
template< class _T1, class _T2> struct pair; 
# 2381
template< class _InputIterator> using __iter_to_alloc_t = pair< add_const_t< __iter_key_t< _InputIterator> > , __iter_val_t< _InputIterator> > ; 
# 2388
}
# 48 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/debug/debug.h" 3
namespace std { 
# 50
namespace __debug { }
# 51
}
# 56
namespace __gnu_debug { 
# 58
using namespace std::__debug;
# 60
template< class _Ite, class _Seq, class _Cat> struct _Safe_iterator; 
# 62
}
# 35 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/predefined_ops.h" 3
namespace __gnu_cxx { 
# 37
namespace __ops { 
# 39
struct _Iter_less_iter { 
# 41
template< class _Iterator1, class _Iterator2> constexpr bool 
# 44
operator()(_Iterator1 __it1, _Iterator2 __it2) const 
# 45
{ return (*__it1) < (*__it2); } 
# 46
}; 
# 50
constexpr _Iter_less_iter __iter_less_iter() 
# 51
{ return _Iter_less_iter(); } 
# 53
struct _Iter_less_val { 
# 56
constexpr _Iter_less_val() = default;
# 63
explicit _Iter_less_val(_Iter_less_iter) { } 
# 65
template< class _Iterator, class _Value> bool 
# 68
operator()(_Iterator __it, _Value &__val) const 
# 69
{ return (*__it) < __val; } 
# 70
}; 
# 74
inline _Iter_less_val __iter_less_val() 
# 75
{ return _Iter_less_val(); } 
# 79
inline _Iter_less_val __iter_comp_val(_Iter_less_iter) 
# 80
{ return _Iter_less_val(); } 
# 82
struct _Val_less_iter { 
# 85
constexpr _Val_less_iter() = default;
# 92
explicit _Val_less_iter(_Iter_less_iter) { } 
# 94
template< class _Value, class _Iterator> bool 
# 97
operator()(_Value &__val, _Iterator __it) const 
# 98
{ return __val < (*__it); } 
# 99
}; 
# 103
inline _Val_less_iter __val_less_iter() 
# 104
{ return _Val_less_iter(); } 
# 108
inline _Val_less_iter __val_comp_iter(_Iter_less_iter) 
# 109
{ return _Val_less_iter(); } 
# 111
struct _Iter_equal_to_iter { 
# 113
template< class _Iterator1, class _Iterator2> bool 
# 116
operator()(_Iterator1 __it1, _Iterator2 __it2) const 
# 117
{ return (*__it1) == (*__it2); } 
# 118
}; 
# 122
inline _Iter_equal_to_iter __iter_equal_to_iter() 
# 123
{ return _Iter_equal_to_iter(); } 
# 125
struct _Iter_equal_to_val { 
# 127
template< class _Iterator, class _Value> bool 
# 130
operator()(_Iterator __it, _Value &__val) const 
# 131
{ return (*__it) == __val; } 
# 132
}; 
# 136
inline _Iter_equal_to_val __iter_equal_to_val() 
# 137
{ return _Iter_equal_to_val(); } 
# 141
inline _Iter_equal_to_val __iter_comp_val(_Iter_equal_to_iter) 
# 142
{ return _Iter_equal_to_val(); } 
# 144
template< class _Compare> 
# 145
struct _Iter_comp_iter { 
# 147
_Compare _M_comp; 
# 150
constexpr explicit _Iter_comp_iter(_Compare __comp) : _M_comp(std::move(__comp)) 
# 152
{ } 
# 154
template< class _Iterator1, class _Iterator2> constexpr bool 
# 157
operator()(_Iterator1 __it1, _Iterator2 __it2) 
# 158
{ return (bool)(_M_comp)(*__it1, *__it2); } 
# 159
}; 
# 161
template< class _Compare> constexpr _Iter_comp_iter< _Compare>  
# 164
__iter_comp_iter(_Compare __comp) 
# 165
{ return ((_Iter_comp_iter< _Compare> )(std::move(__comp))); } 
# 167
template< class _Compare> 
# 168
struct _Iter_comp_val { 
# 170
_Compare _M_comp; 
# 174
explicit _Iter_comp_val(_Compare __comp) : _M_comp(std::move(__comp)) 
# 176
{ } 
# 180
explicit _Iter_comp_val(const _Iter_comp_iter< _Compare>  &__comp) : _M_comp((__comp._M_comp)) 
# 182
{ } 
# 187
explicit _Iter_comp_val(_Iter_comp_iter< _Compare>  &&__comp) : _M_comp(std::move((__comp._M_comp))) 
# 189
{ } 
# 192
template< class _Iterator, class _Value> bool 
# 195
operator()(_Iterator __it, _Value &__val) 
# 196
{ return (bool)(_M_comp)(*__it, __val); } 
# 197
}; 
# 199
template< class _Compare> inline _Iter_comp_val< _Compare>  
# 202
__iter_comp_val(_Compare __comp) 
# 203
{ return ((_Iter_comp_val< _Compare> )(std::move(__comp))); } 
# 205
template< class _Compare> inline _Iter_comp_val< _Compare>  
# 208
__iter_comp_val(_Iter_comp_iter< _Compare>  __comp) 
# 209
{ return ((_Iter_comp_val< _Compare> )(std::move(__comp))); } 
# 211
template< class _Compare> 
# 212
struct _Val_comp_iter { 
# 214
_Compare _M_comp; 
# 218
explicit _Val_comp_iter(_Compare __comp) : _M_comp(std::move(__comp)) 
# 220
{ } 
# 224
explicit _Val_comp_iter(const _Iter_comp_iter< _Compare>  &__comp) : _M_comp((__comp._M_comp)) 
# 226
{ } 
# 231
explicit _Val_comp_iter(_Iter_comp_iter< _Compare>  &&__comp) : _M_comp(std::move((__comp._M_comp))) 
# 233
{ } 
# 236
template< class _Value, class _Iterator> bool 
# 239
operator()(_Value &__val, _Iterator __it) 
# 240
{ return (bool)(_M_comp)(__val, *__it); } 
# 241
}; 
# 243
template< class _Compare> inline _Val_comp_iter< _Compare>  
# 246
__val_comp_iter(_Compare __comp) 
# 247
{ return ((_Val_comp_iter< _Compare> )(std::move(__comp))); } 
# 249
template< class _Compare> inline _Val_comp_iter< _Compare>  
# 252
__val_comp_iter(_Iter_comp_iter< _Compare>  __comp) 
# 253
{ return ((_Val_comp_iter< _Compare> )(std::move(__comp))); } 
# 255
template< class _Value> 
# 256
struct _Iter_equals_val { 
# 258
_Value &_M_value; 
# 262
explicit _Iter_equals_val(_Value &__value) : _M_value(__value) 
# 264
{ } 
# 266
template< class _Iterator> bool 
# 269
operator()(_Iterator __it) 
# 270
{ return (*__it) == (_M_value); } 
# 271
}; 
# 273
template< class _Value> inline _Iter_equals_val< _Value>  
# 276
__iter_equals_val(_Value &__val) 
# 277
{ return ((_Iter_equals_val< _Value> )(__val)); } 
# 279
template< class _Iterator1> 
# 280
struct _Iter_equals_iter { 
# 282
_Iterator1 _M_it1; 
# 286
explicit _Iter_equals_iter(_Iterator1 __it1) : _M_it1(__it1) 
# 288
{ } 
# 290
template< class _Iterator2> bool 
# 293
operator()(_Iterator2 __it2) 
# 294
{ return (*__it2) == (*(_M_it1)); } 
# 295
}; 
# 297
template< class _Iterator> inline _Iter_equals_iter< _Iterator>  
# 300
__iter_comp_iter(_Iter_equal_to_iter, _Iterator __it) 
# 301
{ return ((_Iter_equals_iter< _Iterator> )(__it)); } 
# 303
template< class _Predicate> 
# 304
struct _Iter_pred { 
# 306
_Predicate _M_pred; 
# 310
explicit _Iter_pred(_Predicate __pred) : _M_pred(std::move(__pred)) 
# 312
{ } 
# 314
template< class _Iterator> bool 
# 317
operator()(_Iterator __it) 
# 318
{ return (bool)(_M_pred)(*__it); } 
# 319
}; 
# 321
template< class _Predicate> inline _Iter_pred< _Predicate>  
# 324
__pred_iter(_Predicate __pred) 
# 325
{ return ((_Iter_pred< _Predicate> )(std::move(__pred))); } 
# 327
template< class _Compare, class _Value> 
# 328
struct _Iter_comp_to_val { 
# 330
_Compare _M_comp; 
# 331
_Value &_M_value; 
# 334
_Iter_comp_to_val(_Compare __comp, _Value &__value) : _M_comp(std::move(__comp)), _M_value(__value) 
# 336
{ } 
# 338
template< class _Iterator> bool 
# 341
operator()(_Iterator __it) 
# 342
{ return (bool)(_M_comp)(*__it, _M_value); } 
# 343
}; 
# 345
template< class _Compare, class _Value> _Iter_comp_to_val< _Compare, _Value>  
# 348
__iter_comp_val(_Compare __comp, _Value &__val) 
# 349
{ 
# 350
return _Iter_comp_to_val< _Compare, _Value> (std::move(__comp), __val); 
# 351
} 
# 353
template< class _Compare, class _Iterator1> 
# 354
struct _Iter_comp_to_iter { 
# 356
_Compare _M_comp; 
# 357
_Iterator1 _M_it1; 
# 360
_Iter_comp_to_iter(_Compare __comp, _Iterator1 __it1) : _M_comp(std::move(__comp)), _M_it1(__it1) 
# 362
{ } 
# 364
template< class _Iterator2> bool 
# 367
operator()(_Iterator2 __it2) 
# 368
{ return (bool)(_M_comp)(*__it2, *(_M_it1)); } 
# 369
}; 
# 371
template< class _Compare, class _Iterator> inline _Iter_comp_to_iter< _Compare, _Iterator>  
# 374
__iter_comp_iter(_Iter_comp_iter< _Compare>  __comp, _Iterator __it) 
# 375
{ 
# 376
return _Iter_comp_to_iter< _Compare, _Iterator> (std::move((__comp._M_comp)), __it); 
# 378
} 
# 380
template< class _Predicate> 
# 381
struct _Iter_negate { 
# 383
_Predicate _M_pred; 
# 387
explicit _Iter_negate(_Predicate __pred) : _M_pred(std::move(__pred)) 
# 389
{ } 
# 391
template< class _Iterator> bool 
# 394
operator()(_Iterator __it) 
# 395
{ return !((bool)(_M_pred)(*__it)); } 
# 396
}; 
# 398
template< class _Predicate> inline _Iter_negate< _Predicate>  
# 401
__negate(_Iter_pred< _Predicate>  __pred) 
# 402
{ return ((_Iter_negate< _Predicate> )(std::move((__pred._M_pred)))); } 
# 404
}
# 405
}
# 79 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 87
template< class _Tp, class _Up> constexpr int 
# 90
__memcmp(const _Tp *__first1, const _Up *__first2, size_t __num) 
# 91
{ 
# 93
static_assert((sizeof(_Tp) == sizeof(_Up)), "can be compared with memcmp");
# 105 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
return __builtin_memcmp(__first1, __first2, sizeof(_Tp) * __num); 
# 106
} 
# 149 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _ForwardIterator1, class _ForwardIterator2> inline void 
# 152
iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b) 
# 153
{ 
# 182 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
swap(*__a, *__b); 
# 184
} 
# 198 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _ForwardIterator1, class _ForwardIterator2> _ForwardIterator2 
# 201
swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 
# 202
__first2) 
# 203
{ 
# 209
; 
# 211
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 212
std::iter_swap(__first1, __first2); }  
# 213
return __first2; 
# 214
} 
# 227 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _Tp> constexpr const _Tp &
# 230
min(const _Tp &__a, const _Tp &__b) 
# 231
{ 
# 235
if (__b < __a) { 
# 236
return __b; }  
# 237
return __a; 
# 238
} 
# 251 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _Tp> constexpr const _Tp &
# 254
max(const _Tp &__a, const _Tp &__b) 
# 255
{ 
# 259
if (__a < __b) { 
# 260
return __b; }  
# 261
return __a; 
# 262
} 
# 275 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _Tp, class _Compare> constexpr const _Tp &
# 278
min(const _Tp &__a, const _Tp &__b, _Compare __comp) 
# 279
{ 
# 281
if (__comp(__b, __a)) { 
# 282
return __b; }  
# 283
return __a; 
# 284
} 
# 297 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _Tp, class _Compare> constexpr const _Tp &
# 300
max(const _Tp &__a, const _Tp &__b, _Compare __comp) 
# 301
{ 
# 303
if (__comp(__a, __b)) { 
# 304
return __b; }  
# 305
return __a; 
# 306
} 
# 310
template< class _Iterator> inline _Iterator 
# 313
__niter_base(_Iterator __it) noexcept(std::is_nothrow_copy_constructible< _Iterator> ::value) 
# 315
{ return __it; } 
# 317
template< class _Ite, class _Seq> _Ite __niter_base(const __gnu_debug::_Safe_iterator< _Ite, _Seq, random_access_iterator_tag>  &); 
# 325
template< class _From, class _To> inline _From 
# 328
__niter_wrap(_From __from, _To __res) 
# 329
{ return __from + (__res - std::__niter_base(__from)); } 
# 332
template< class _Iterator> inline _Iterator 
# 335
__niter_wrap(const _Iterator &, _Iterator __res) 
# 336
{ return __res; } 
# 344
template< bool _IsMove, bool _IsSimple, class _Category> 
# 345
struct __copy_move { 
# 347
template< class _II, class _OI> static _OI 
# 350
__copy_m(_II __first, _II __last, _OI __result) 
# 351
{ 
# 352
for (; __first != __last; (++__result), ((void)(++__first))) { 
# 353
(*__result) = (*__first); }  
# 354
return __result; 
# 355
} 
# 356
}; 
# 359
template< class _Category> 
# 360
struct __copy_move< true, false, _Category>  { 
# 362
template< class _II, class _OI> static _OI 
# 365
__copy_m(_II __first, _II __last, _OI __result) 
# 366
{ 
# 367
for (; __first != __last; (++__result), ((void)(++__first))) { 
# 368
(*__result) = std::move(*__first); }  
# 369
return __result; 
# 370
} 
# 371
}; 
# 375
template<> struct __copy_move< false, false, random_access_iterator_tag>  { 
# 377
template< class _II, class _OI> static _OI 
# 380
__copy_m(_II __first, _II __last, _OI __result) 
# 381
{ 
# 382
typedef typename iterator_traits< _II> ::difference_type _Distance; 
# 383
for (_Distance __n = __last - __first; __n > 0; --__n) 
# 384
{ 
# 385
(*__result) = (*__first); 
# 386
++__first; 
# 387
++__result; 
# 388
}  
# 389
return __result; 
# 390
} 
# 391
}; 
# 395
template<> struct __copy_move< true, false, random_access_iterator_tag>  { 
# 397
template< class _II, class _OI> static _OI 
# 400
__copy_m(_II __first, _II __last, _OI __result) 
# 401
{ 
# 402
typedef typename iterator_traits< _II> ::difference_type _Distance; 
# 403
for (_Distance __n = __last - __first; __n > 0; --__n) 
# 404
{ 
# 405
(*__result) = std::move(*__first); 
# 406
++__first; 
# 407
++__result; 
# 408
}  
# 409
return __result; 
# 410
} 
# 411
}; 
# 414
template< bool _IsMove> 
# 415
struct __copy_move< _IsMove, true, random_access_iterator_tag>  { 
# 417
template< class _Tp> static _Tp *
# 420
__copy_m(const _Tp *__first, const _Tp *__last, _Tp *__result) 
# 421
{ 
# 423
using __assignable = conditional< _IsMove, is_move_assignable< _Tp> , is_copy_assignable< _Tp> > ; 
# 427
static_assert((__assignable::type::value), "type is not assignable");
# 429
const ptrdiff_t _Num = __last - __first; 
# 430
if (_Num) { 
# 431
__builtin_memmove(__result, __first, sizeof(_Tp) * _Num); }  
# 432
return __result + _Num; 
# 433
} 
# 434
}; 
# 438
template< class _Tp, class _Ref, class _Ptr> struct _Deque_iterator; 
# 441
struct _Bit_iterator; 
# 447
template< class _CharT> struct char_traits; 
# 450
template< class _CharT, class _Traits> class istreambuf_iterator; 
# 453
template< class _CharT, class _Traits> class ostreambuf_iterator; 
# 456
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type __copy_move_a2(_CharT *, _CharT *, ostreambuf_iterator< _CharT, char_traits< _CharT> > ); 
# 462
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type __copy_move_a2(const _CharT *, const _CharT *, ostreambuf_iterator< _CharT, char_traits< _CharT> > ); 
# 468
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _CharT *> ::__type __copy_move_a2(istreambuf_iterator< _CharT, char_traits< _CharT> > , istreambuf_iterator< _CharT, char_traits< _CharT> > , _CharT *); 
# 474
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _Deque_iterator< _CharT, _CharT &, _CharT *> > ::__type __copy_move_a2(istreambuf_iterator< _CharT, char_traits< _CharT> > , istreambuf_iterator< _CharT, char_traits< _CharT> > , _Deque_iterator< _CharT, _CharT &, _CharT *> ); 
# 483
template< bool _IsMove, class _II, class _OI> inline _OI 
# 486
__copy_move_a2(_II __first, _II __last, _OI __result) 
# 487
{ 
# 488
typedef typename iterator_traits< _II> ::iterator_category _Category; 
# 494
return std::__copy_move< _IsMove, __memcpyable< _OI, _II> ::__value, typename iterator_traits< _II> ::iterator_category> ::__copy_m(__first, __last, __result); 
# 496
} 
# 498
template< bool _IsMove, class 
# 499
_Tp, class _Ref, class _Ptr, class _OI> _OI 
# 498
__copy_move_a1(_Deque_iterator< _Tp, _Ref, _Ptr> , _Deque_iterator< _Tp, _Ref, _Ptr> , _OI); 
# 505
template< bool _IsMove, class 
# 506
_ITp, class _IRef, class _IPtr, class _OTp> _Deque_iterator< _OTp, _OTp &, _OTp *>  
# 505
__copy_move_a1(_Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _OTp, _OTp &, _OTp *> ); 
# 512
template< bool _IsMove, class _II, class _Tp> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, _Deque_iterator< _Tp, _Tp &, _Tp *> > ::__type __copy_move_a1(_II, _II, _Deque_iterator< _Tp, _Tp &, _Tp *> ); 
# 518
template< bool _IsMove, class _II, class _OI> inline _OI 
# 521
__copy_move_a1(_II __first, _II __last, _OI __result) 
# 522
{ return std::__copy_move_a2< _IsMove> (__first, __last, __result); } 
# 524
template< bool _IsMove, class _II, class _OI> inline _OI 
# 527
__copy_move_a(_II __first, _II __last, _OI __result) 
# 528
{ 
# 529
return std::__niter_wrap(__result, std::__copy_move_a1< _IsMove> (std::__niter_base(__first), std::__niter_base(__last), std::__niter_base(__result))); 
# 533
} 
# 535
template< bool _IsMove, class 
# 536
_Ite, class _Seq, class _Cat, class _OI> _OI 
# 535
__copy_move_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, _OI); 
# 542
template< bool _IsMove, class 
# 543
_II, class _Ite, class _Seq, class _Cat> __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  
# 542
__copy_move_a(_II, _II, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &); 
# 548
template< bool _IsMove, class 
# 549
_IIte, class _ISeq, class _ICat, class 
# 550
_OIte, class _OSeq, class _OCat> __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  
# 548
__copy_move_a(const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  &); 
# 556
template< class _InputIterator, class _Size, class _OutputIterator> _OutputIterator 
# 559
__copy_n_a(_InputIterator __first, _Size __n, _OutputIterator __result, bool) 
# 561
{ 
# 562
if (__n > 0) 
# 563
{ 
# 564
while (true) 
# 565
{ 
# 566
(*__result) = (*__first); 
# 567
++__result; 
# 568
if ((--__n) > 0) { 
# 569
++__first; } else { 
# 571
break; }  
# 572
}  
# 573
}  
# 574
return __result; 
# 575
} 
# 577
template< class _CharT, class _Size> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _CharT *> ::__type __copy_n_a(istreambuf_iterator< _CharT, char_traits< _CharT> > , _Size, _CharT *, bool); 
# 583
template< class _CharT, class _Size> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _Deque_iterator< _CharT, _CharT &, _CharT *> > ::__type __copy_n_a(istreambuf_iterator< _CharT, char_traits< _CharT> > , _Size, _Deque_iterator< _CharT, _CharT &, _CharT *> , bool); 
# 608 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _II, class _OI> inline _OI 
# 611
copy(_II __first, _II __last, _OI __result) 
# 612
{ 
# 617
; 
# 619
return std::__copy_move_a< __is_move_iterator< _II> ::__value> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 621
} 
# 641 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _II, class _OI> inline _OI 
# 644
move(_II __first, _II __last, _OI __result) 
# 645
{ 
# 650
; 
# 652
return std::__copy_move_a< true> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 654
} 
# 661
template< bool _IsMove, bool _IsSimple, class _Category> 
# 662
struct __copy_move_backward { 
# 664
template< class _BI1, class _BI2> static _BI2 
# 667
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 668
{ 
# 669
while (__first != __last) { 
# 670
(*(--__result)) = (*(--__last)); }  
# 671
return __result; 
# 672
} 
# 673
}; 
# 676
template< class _Category> 
# 677
struct __copy_move_backward< true, false, _Category>  { 
# 679
template< class _BI1, class _BI2> static _BI2 
# 682
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 683
{ 
# 684
while (__first != __last) { 
# 685
(*(--__result)) = std::move(*(--__last)); }  
# 686
return __result; 
# 687
} 
# 688
}; 
# 692
template<> struct __copy_move_backward< false, false, random_access_iterator_tag>  { 
# 694
template< class _BI1, class _BI2> static _BI2 
# 697
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 698
{ 
# 700
typename iterator_traits< _BI1> ::difference_type __n = __last - __first; 
# 701
for (; __n > 0; --__n) { 
# 702
(*(--__result)) = (*(--__last)); }  
# 703
return __result; 
# 704
} 
# 705
}; 
# 709
template<> struct __copy_move_backward< true, false, random_access_iterator_tag>  { 
# 711
template< class _BI1, class _BI2> static _BI2 
# 714
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 715
{ 
# 717
typename iterator_traits< _BI1> ::difference_type __n = __last - __first; 
# 718
for (; __n > 0; --__n) { 
# 719
(*(--__result)) = std::move(*(--__last)); }  
# 720
return __result; 
# 721
} 
# 722
}; 
# 725
template< bool _IsMove> 
# 726
struct __copy_move_backward< _IsMove, true, random_access_iterator_tag>  { 
# 728
template< class _Tp> static _Tp *
# 731
__copy_move_b(const _Tp *__first, const _Tp *__last, _Tp *__result) 
# 732
{ 
# 734
using __assignable = conditional< _IsMove, is_move_assignable< _Tp> , is_copy_assignable< _Tp> > ; 
# 738
static_assert((__assignable::type::value), "type is not assignable");
# 740
const ptrdiff_t _Num = __last - __first; 
# 741
if (_Num) { 
# 742
__builtin_memmove(__result - _Num, __first, sizeof(_Tp) * _Num); }  
# 743
return __result - _Num; 
# 744
} 
# 745
}; 
# 747
template< bool _IsMove, class _BI1, class _BI2> inline _BI2 
# 750
__copy_move_backward_a2(_BI1 __first, _BI1 __last, _BI2 __result) 
# 751
{ 
# 752
typedef typename iterator_traits< _BI1> ::iterator_category _Category; 
# 758
return std::__copy_move_backward< _IsMove, __memcpyable< _BI2, _BI1> ::__value, typename iterator_traits< _BI1> ::iterator_category> ::__copy_move_b(__first, __last, __result); 
# 763
} 
# 765
template< bool _IsMove, class _BI1, class _BI2> inline _BI2 
# 768
__copy_move_backward_a1(_BI1 __first, _BI1 __last, _BI2 __result) 
# 769
{ return std::__copy_move_backward_a2< _IsMove> (__first, __last, __result); } 
# 771
template< bool _IsMove, class 
# 772
_Tp, class _Ref, class _Ptr, class _OI> _OI 
# 771
__copy_move_backward_a1(_Deque_iterator< _Tp, _Ref, _Ptr> , _Deque_iterator< _Tp, _Ref, _Ptr> , _OI); 
# 778
template< bool _IsMove, class 
# 779
_ITp, class _IRef, class _IPtr, class _OTp> _Deque_iterator< _OTp, _OTp &, _OTp *>  
# 778
__copy_move_backward_a1(_Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _ITp, _IRef, _IPtr> , _Deque_iterator< _OTp, _OTp &, _OTp *> ); 
# 786
template< bool _IsMove, class _II, class _Tp> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, _Deque_iterator< _Tp, _Tp &, _Tp *> > ::__type __copy_move_backward_a1(_II, _II, _Deque_iterator< _Tp, _Tp &, _Tp *> ); 
# 793
template< bool _IsMove, class _II, class _OI> inline _OI 
# 796
__copy_move_backward_a(_II __first, _II __last, _OI __result) 
# 797
{ 
# 798
return std::__niter_wrap(__result, std::__copy_move_backward_a1< _IsMove> (std::__niter_base(__first), std::__niter_base(__last), std::__niter_base(__result))); 
# 802
} 
# 804
template< bool _IsMove, class 
# 805
_Ite, class _Seq, class _Cat, class _OI> _OI 
# 804
__copy_move_backward_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, _OI); 
# 812
template< bool _IsMove, class 
# 813
_II, class _Ite, class _Seq, class _Cat> __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  
# 812
__copy_move_backward_a(_II, _II, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &); 
# 818
template< bool _IsMove, class 
# 819
_IIte, class _ISeq, class _ICat, class 
# 820
_OIte, class _OSeq, class _OCat> __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  
# 818
__copy_move_backward_a(const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _IIte, _ISeq, _ICat>  &, const __gnu_debug::_Safe_iterator< _OIte, _OSeq, _OCat>  &); 
# 845 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _BI1, class _BI2> inline _BI2 
# 848
copy_backward(_BI1 __first, _BI1 __last, _BI2 __result) 
# 849
{ 
# 856
; 
# 858
return std::__copy_move_backward_a< __is_move_iterator< _BI1> ::__value> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 860
} 
# 881 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _BI1, class _BI2> inline _BI2 
# 884
move_backward(_BI1 __first, _BI1 __last, _BI2 __result) 
# 885
{ 
# 892
; 
# 894
return std::__copy_move_backward_a< true> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 897
} 
# 904
template< class _ForwardIterator, class _Tp> inline typename __gnu_cxx::__enable_if< !__is_scalar< _Tp> ::__value, void> ::__type 
# 908
__fill_a1(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 909
__value) 
# 910
{ 
# 911
for (; __first != __last; ++__first) { 
# 912
(*__first) = __value; }  
# 913
} 
# 915
template< class _ForwardIterator, class _Tp> inline typename __gnu_cxx::__enable_if< __is_scalar< _Tp> ::__value, void> ::__type 
# 919
__fill_a1(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 920
__value) 
# 921
{ 
# 922
const _Tp __tmp = __value; 
# 923
for (; __first != __last; ++__first) { 
# 924
(*__first) = __tmp; }  
# 925
} 
# 928
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_byte< _Tp> ::__value, void> ::__type 
# 932
__fill_a1(_Tp *__first, _Tp *__last, const _Tp &__c) 
# 933
{ 
# 934
const _Tp __tmp = __c; 
# 943 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
if (const size_t __len = __last - __first) { 
# 944
__builtin_memset(__first, static_cast< unsigned char>(__tmp), __len); }  
# 945
} 
# 947
template< class _Ite, class _Cont, class _Tp> inline void 
# 950
__fill_a1(__gnu_cxx::__normal_iterator< _Ite, _Cont>  __first, __gnu_cxx::__normal_iterator< _Ite, _Cont>  
# 951
__last, const _Tp &
# 952
__value) 
# 953
{ std::__fill_a1(__first.base(), __last.base(), __value); } 
# 955
template< class _Tp, class _VTp> void __fill_a1(const _Deque_iterator< _Tp, _Tp &, _Tp *>  &, const _Deque_iterator< _Tp, _Tp &, _Tp *>  &, const _VTp &); 
# 962
void __fill_a1(_Bit_iterator, _Bit_iterator, const bool &); 
# 965
template< class _FIte, class _Tp> inline void 
# 968
__fill_a(_FIte __first, _FIte __last, const _Tp &__value) 
# 969
{ std::__fill_a1(__first, __last, __value); } 
# 971
template< class _Ite, class _Seq, class _Cat, class _Tp> void __fill_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  &, const _Tp &); 
# 989 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _ForwardIterator, class _Tp> inline void 
# 992
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp &__value) 
# 993
{ 
# 997
; 
# 999
std::__fill_a(__first, __last, __value); 
# 1000
} 
# 1004
constexpr int __size_to_integer(int __n) { return __n; } 
# 1006
constexpr unsigned __size_to_integer(unsigned __n) { return __n; } 
# 1008
constexpr long __size_to_integer(long __n) { return __n; } 
# 1010
constexpr unsigned long __size_to_integer(unsigned long __n) { return __n; } 
# 1012
constexpr long long __size_to_integer(long long __n) { return __n; } 
# 1014
constexpr unsigned long long __size_to_integer(unsigned long long __n) { return __n; } 
# 1018
constexpr __int128 __size_to_integer(__int128 __n) { return __n; } 
# 1020
constexpr unsigned __int128 __size_to_integer(unsigned __int128 __n) { return __n; } 
# 1042 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
constexpr long long __size_to_integer(float __n) { return (long long)__n; } 
# 1044
constexpr long long __size_to_integer(double __n) { return (long long)__n; } 
# 1046
constexpr long long __size_to_integer(long double __n) { return (long long)__n; } 
# 1049
constexpr long long __size_to_integer(__float128 __n) { return (long long)__n; } 
# 1052
template< class _OutputIterator, class _Size, class _Tp> inline typename __gnu_cxx::__enable_if< !__is_scalar< _Tp> ::__value, _OutputIterator> ::__type 
# 1056
__fill_n_a1(_OutputIterator __first, _Size __n, const _Tp &__value) 
# 1057
{ 
# 1058
for (; __n > 0; (--__n), ((void)(++__first))) { 
# 1059
(*__first) = __value; }  
# 1060
return __first; 
# 1061
} 
# 1063
template< class _OutputIterator, class _Size, class _Tp> inline typename __gnu_cxx::__enable_if< __is_scalar< _Tp> ::__value, _OutputIterator> ::__type 
# 1067
__fill_n_a1(_OutputIterator __first, _Size __n, const _Tp &__value) 
# 1068
{ 
# 1069
const _Tp __tmp = __value; 
# 1070
for (; __n > 0; (--__n), ((void)(++__first))) { 
# 1071
(*__first) = __tmp; }  
# 1072
return __first; 
# 1073
} 
# 1075
template< class _Ite, class _Seq, class _Cat, class _Size, class 
# 1076
_Tp> __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  
# 1075
__fill_n_a(const __gnu_debug::_Safe_iterator< _Ite, _Seq, _Cat>  & __first, _Size __n, const _Tp & __value, input_iterator_tag); 
# 1082
template< class _OutputIterator, class _Size, class _Tp> inline _OutputIterator 
# 1085
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value, output_iterator_tag) 
# 1087
{ 
# 1089
static_assert((is_integral< _Size> {}), "fill_n must pass integral size");
# 1091
return __fill_n_a1(__first, __n, __value); 
# 1092
} 
# 1094
template< class _OutputIterator, class _Size, class _Tp> inline _OutputIterator 
# 1097
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value, input_iterator_tag) 
# 1099
{ 
# 1101
static_assert((is_integral< _Size> {}), "fill_n must pass integral size");
# 1103
return __fill_n_a1(__first, __n, __value); 
# 1104
} 
# 1106
template< class _OutputIterator, class _Size, class _Tp> inline _OutputIterator 
# 1109
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value, random_access_iterator_tag) 
# 1111
{ 
# 1113
static_assert((is_integral< _Size> {}), "fill_n must pass integral size");
# 1115
if (__n <= 0) { 
# 1116
return __first; }  
# 1118
; 
# 1120
std::__fill_a(__first, __first + __n, __value); 
# 1121
return __first + __n; 
# 1122
} 
# 1141 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _OI, class _Size, class _Tp> inline _OI 
# 1144
fill_n(_OI __first, _Size __n, const _Tp &__value) 
# 1145
{ 
# 1149
return std::__fill_n_a(__first, std::__size_to_integer(__n), __value, std::__iterator_category(__first)); 
# 1151
} 
# 1153
template< bool _BoolType> 
# 1154
struct __equal { 
# 1156
template< class _II1, class _II2> static bool 
# 1159
equal(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1160
{ 
# 1161
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 1162
if (!((*__first1) == (*__first2))) { 
# 1163
return false; }  }  
# 1164
return true; 
# 1165
} 
# 1166
}; 
# 1169
template<> struct __equal< true>  { 
# 1171
template< class _Tp> static bool 
# 1174
equal(const _Tp *__first1, const _Tp *__last1, const _Tp *__first2) 
# 1175
{ 
# 1176
if (const size_t __len = __last1 - __first1) { 
# 1177
return !std::__memcmp(__first1, __first2, __len); }  
# 1178
return true; 
# 1179
} 
# 1180
}; 
# 1182
template< class _Tp, class _Ref, class _Ptr, class _II> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, bool> ::__type __equal_aux1(_Deque_iterator< _Tp, _Ref, _Ptr> , _Deque_iterator< _Tp, _Ref, _Ptr> , _II); 
# 1189
template< class _Tp1, class _Ref1, class _Ptr1, class 
# 1190
_Tp2, class _Ref2, class _Ptr2> bool 
# 1189
__equal_aux1(_Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> ); 
# 1196
template< class _II, class _Tp, class _Ref, class _Ptr> typename __gnu_cxx::__enable_if< __is_random_access_iter< _II> ::__value, bool> ::__type __equal_aux1(_II, _II, _Deque_iterator< _Tp, _Ref, _Ptr> ); 
# 1202
template< class _II1, class _II2> inline bool 
# 1205
__equal_aux1(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1206
{ 
# 1207
typedef typename iterator_traits< _II1> ::value_type _ValueType1; 
# 1208
const bool __simple = ((__is_integer< typename iterator_traits< _II1> ::value_type> ::__value || __is_pointer< typename iterator_traits< _II1> ::value_type> ::__value) && __memcmpable< _II1, _II2> ::__value); 
# 1211
return std::__equal< __simple> ::equal(__first1, __last1, __first2); 
# 1212
} 
# 1214
template< class _II1, class _II2> inline bool 
# 1217
__equal_aux(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1218
{ 
# 1219
return std::__equal_aux1(std::__niter_base(__first1), std::__niter_base(__last1), std::__niter_base(__first2)); 
# 1222
} 
# 1224
template< class _II1, class _Seq1, class _Cat1, class _II2> bool __equal_aux(const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, _II2); 
# 1230
template< class _II1, class _II2, class _Seq2, class _Cat2> bool __equal_aux(_II1, _II1, const __gnu_debug::_Safe_iterator< _II2, _Seq2, _Cat2>  &); 
# 1235
template< class _II1, class _Seq1, class _Cat1, class 
# 1236
_II2, class _Seq2, class _Cat2> bool 
# 1235
__equal_aux(const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _II1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _II2, _Seq2, _Cat2>  &); 
# 1242
template< class , class > 
# 1243
struct __lc_rai { 
# 1245
template< class _II1, class _II2> static _II1 
# 1248
__newlast1(_II1, _II1 __last1, _II2, _II2) 
# 1249
{ return __last1; } 
# 1251
template< class _II> static bool 
# 1254
__cnd2(_II __first, _II __last) 
# 1255
{ return __first != __last; } 
# 1256
}; 
# 1259
template<> struct __lc_rai< random_access_iterator_tag, random_access_iterator_tag>  { 
# 1261
template< class _RAI1, class _RAI2> static _RAI1 
# 1264
__newlast1(_RAI1 __first1, _RAI1 __last1, _RAI2 
# 1265
__first2, _RAI2 __last2) 
# 1266
{ 
# 1268
const typename iterator_traits< _RAI1> ::difference_type __diff1 = __last1 - __first1; 
# 1270
const typename iterator_traits< _RAI2> ::difference_type __diff2 = __last2 - __first2; 
# 1271
return (__diff2 < __diff1) ? __first1 + __diff2 : __last1; 
# 1272
} 
# 1274
template< class _RAI> static bool 
# 1276
__cnd2(_RAI, _RAI) 
# 1277
{ return true; } 
# 1278
}; 
# 1280
template< class _II1, class _II2, class _Compare> bool 
# 1283
__lexicographical_compare_impl(_II1 __first1, _II1 __last1, _II2 
# 1284
__first2, _II2 __last2, _Compare 
# 1285
__comp) 
# 1286
{ 
# 1287
typedef typename iterator_traits< _II1> ::iterator_category _Category1; 
# 1288
typedef typename iterator_traits< _II2> ::iterator_category _Category2; 
# 1289
typedef __lc_rai< typename iterator_traits< _II1> ::iterator_category, typename iterator_traits< _II2> ::iterator_category>  __rai_type; 
# 1291
__last1 = __rai_type::__newlast1(__first1, __last1, __first2, __last2); 
# 1292
for (; (__first1 != __last1) && __rai_type::__cnd2(__first2, __last2); (++__first1), ((void)(++__first2))) 
# 1294
{ 
# 1295
if (__comp(__first1, __first2)) { 
# 1296
return true; }  
# 1297
if (__comp(__first2, __first1)) { 
# 1298
return false; }  
# 1299
}  
# 1300
return (__first1 == __last1) && (__first2 != __last2); 
# 1301
} 
# 1303
template< bool _BoolType> 
# 1304
struct __lexicographical_compare { 
# 1306
template< class _II1, class _II2> static bool 
# 1309
__lc(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1310
{ 
# 1311
using __gnu_cxx::__ops::__iter_less_iter;
# 1312
return std::__lexicographical_compare_impl(__first1, __last1, __first2, __last2, __iter_less_iter()); 
# 1315
} 
# 1317
template< class _II1, class _II2> static int 
# 1320
__3way(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1321
{ 
# 1322
while (__first1 != __last1) 
# 1323
{ 
# 1324
if (__first2 == __last2) { 
# 1325
return +1; }  
# 1326
if ((*__first1) < (*__first2)) { 
# 1327
return -1; }  
# 1328
if ((*__first2) < (*__first1)) { 
# 1329
return +1; }  
# 1330
++__first1; 
# 1331
++__first2; 
# 1332
}  
# 1333
return ((int)(__first2 == __last2)) - 1; 
# 1334
} 
# 1335
}; 
# 1338
template<> struct __lexicographical_compare< true>  { 
# 1340
template< class _Tp, class _Up> static bool 
# 1343
__lc(const _Tp *__first1, const _Tp *__last1, const _Up *
# 1344
__first2, const _Up *__last2) 
# 1345
{ return __3way(__first1, __last1, __first2, __last2) < 0; } 
# 1347
template< class _Tp, class _Up> static ptrdiff_t 
# 1350
__3way(const _Tp *__first1, const _Tp *__last1, const _Up *
# 1351
__first2, const _Up *__last2) 
# 1352
{ 
# 1353
const size_t __len1 = __last1 - __first1; 
# 1354
const size_t __len2 = __last2 - __first2; 
# 1355
if (const size_t __len = std::min(__len1, __len2)) { 
# 1356
if (int __result = std::__memcmp(__first1, __first2, __len)) { 
# 1357
return __result; }  }  
# 1358
return (ptrdiff_t)(__len1 - __len2); 
# 1359
} 
# 1360
}; 
# 1362
template< class _II1, class _II2> inline bool 
# 1365
__lexicographical_compare_aux1(_II1 __first1, _II1 __last1, _II2 
# 1366
__first2, _II2 __last2) 
# 1367
{ 
# 1368
typedef typename iterator_traits< _II1> ::value_type _ValueType1; 
# 1369
typedef typename iterator_traits< _II2> ::value_type _ValueType2; 
# 1370
const bool __simple = (__is_memcmp_ordered_with< typename iterator_traits< _II1> ::value_type, typename iterator_traits< _II2> ::value_type> ::__value && __is_pointer< _II1> ::__value && __is_pointer< _II2> ::__value); 
# 1383
return std::__lexicographical_compare< __simple> ::__lc(__first1, __last1, __first2, __last2); 
# 1385
} 
# 1387
template< class _Tp1, class _Ref1, class _Ptr1, class 
# 1388
_Tp2> bool 
# 1387
__lexicographical_compare_aux1(_Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Tp2 *, _Tp2 *); 
# 1395
template< class _Tp1, class 
# 1396
_Tp2, class _Ref2, class _Ptr2> bool 
# 1395
__lexicographical_compare_aux1(_Tp1 *, _Tp1 *, _Deque_iterator< _Tp2, _Ref2, _Ptr2> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> ); 
# 1402
template< class _Tp1, class _Ref1, class _Ptr1, class 
# 1403
_Tp2, class _Ref2, class _Ptr2> bool 
# 1402
__lexicographical_compare_aux1(_Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp1, _Ref1, _Ptr1> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> , _Deque_iterator< _Tp2, _Ref2, _Ptr2> ); 
# 1411
template< class _II1, class _II2> inline bool 
# 1414
__lexicographical_compare_aux(_II1 __first1, _II1 __last1, _II2 
# 1415
__first2, _II2 __last2) 
# 1416
{ 
# 1417
return std::__lexicographical_compare_aux1(std::__niter_base(__first1), std::__niter_base(__last1), std::__niter_base(__first2), std::__niter_base(__last2)); 
# 1421
} 
# 1423
template< class _Iter1, class _Seq1, class _Cat1, class 
# 1424
_II2> bool 
# 1423
__lexicographical_compare_aux(const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, _II2, _II2); 
# 1431
template< class _II1, class 
# 1432
_Iter2, class _Seq2, class _Cat2> bool 
# 1431
__lexicographical_compare_aux(_II1, _II1, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &); 
# 1439
template< class _Iter1, class _Seq1, class _Cat1, class 
# 1440
_Iter2, class _Seq2, class _Cat2> bool 
# 1439
__lexicographical_compare_aux(const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _Iter1, _Seq1, _Cat1>  &, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &, const __gnu_debug::_Safe_iterator< _Iter2, _Seq2, _Cat2>  &); 
# 1448
template< class _ForwardIterator, class _Tp, class _Compare> _ForwardIterator 
# 1451
__lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 1452
__val, _Compare __comp) 
# 1453
{ 
# 1455
typedef typename iterator_traits< _ForwardIterator> ::difference_type _DistanceType; 
# 1457
_DistanceType __len = std::distance(__first, __last); 
# 1459
while (__len > 0) 
# 1460
{ 
# 1461
_DistanceType __half = __len >> 1; 
# 1462
_ForwardIterator __middle = __first; 
# 1463
std::advance(__middle, __half); 
# 1464
if (__comp(__middle, __val)) 
# 1465
{ 
# 1466
__first = __middle; 
# 1467
++__first; 
# 1468
__len = ((__len - __half) - 1); 
# 1469
} else { 
# 1471
__len = __half; }  
# 1472
}  
# 1473
return __first; 
# 1474
} 
# 1487 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _ForwardIterator, class _Tp> inline _ForwardIterator 
# 1490
lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 1491
__val) 
# 1492
{ 
# 1497
; 
# 1499
return std::__lower_bound(__first, __last, __val, __gnu_cxx::__ops::__iter_less_val()); 
# 1501
} 
# 1506
constexpr int __lg(int __n) 
# 1507
{ return ((((int)sizeof(int)) * 8) - 1) - __builtin_clz(__n); } 
# 1510
constexpr unsigned __lg(unsigned __n) 
# 1511
{ return ((((int)sizeof(int)) * 8) - 1) - __builtin_clz(__n); } 
# 1514
constexpr long __lg(long __n) 
# 1515
{ return ((((int)sizeof(long)) * 8) - 1) - __builtin_clzl(__n); } 
# 1518
constexpr unsigned long __lg(unsigned long __n) 
# 1519
{ return ((((int)sizeof(long)) * 8) - 1) - __builtin_clzl(__n); } 
# 1522
constexpr long long __lg(long long __n) 
# 1523
{ return ((((int)sizeof(long long)) * 8) - 1) - __builtin_clzll(__n); } 
# 1526
constexpr unsigned long long __lg(unsigned long long __n) 
# 1527
{ return ((((int)sizeof(long long)) * 8) - 1) - __builtin_clzll(__n); } 
# 1543 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _II1, class _II2> inline bool 
# 1546
equal(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1547
{ 
# 1554
; 
# 1556
return std::__equal_aux(__first1, __last1, __first2); 
# 1557
} 
# 1574 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _IIter1, class _IIter2, class _BinaryPredicate> inline bool 
# 1577
equal(_IIter1 __first1, _IIter1 __last1, _IIter2 
# 1578
__first2, _BinaryPredicate __binary_pred) 
# 1579
{ 
# 1583
; 
# 1585
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 1586
if (!((bool)__binary_pred(*__first1, *__first2))) { 
# 1587
return false; }  }  
# 1588
return true; 
# 1589
} 
# 1593
template< class _II1, class _II2> inline bool 
# 1596
__equal4(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1597
{ 
# 1598
using _RATag = random_access_iterator_tag; 
# 1599
using _Cat1 = typename iterator_traits< _II1> ::iterator_category; 
# 1600
using _Cat2 = typename iterator_traits< _II2> ::iterator_category; 
# 1601
using _RAIters = __and_< is_same< typename iterator_traits< _II1> ::iterator_category, random_access_iterator_tag> , is_same< typename iterator_traits< _II2> ::iterator_category, random_access_iterator_tag> > ; 
# 1602
if (_RAIters()) 
# 1603
{ 
# 1604
auto __d1 = std::distance(__first1, __last1); 
# 1605
auto __d2 = std::distance(__first2, __last2); 
# 1606
if (__d1 != __d2) { 
# 1607
return false; }  
# 1608
return std::equal(__first1, __last1, __first2); 
# 1609
}  
# 1611
for (; (__first1 != __last1) && (__first2 != __last2); (++__first1), ((void)(++__first2))) { 
# 1613
if (!((*__first1) == (*__first2))) { 
# 1614
return false; }  }  
# 1615
return (__first1 == __last1) && (__first2 == __last2); 
# 1616
} 
# 1619
template< class _II1, class _II2, class _BinaryPredicate> inline bool 
# 1622
__equal4(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2, _BinaryPredicate 
# 1623
__binary_pred) 
# 1624
{ 
# 1625
using _RATag = random_access_iterator_tag; 
# 1626
using _Cat1 = typename iterator_traits< _II1> ::iterator_category; 
# 1627
using _Cat2 = typename iterator_traits< _II2> ::iterator_category; 
# 1628
using _RAIters = __and_< is_same< typename iterator_traits< _II1> ::iterator_category, random_access_iterator_tag> , is_same< typename iterator_traits< _II2> ::iterator_category, random_access_iterator_tag> > ; 
# 1629
if (_RAIters()) 
# 1630
{ 
# 1631
auto __d1 = std::distance(__first1, __last1); 
# 1632
auto __d2 = std::distance(__first2, __last2); 
# 1633
if (__d1 != __d2) { 
# 1634
return false; }  
# 1635
return std::equal(__first1, __last1, __first2, __binary_pred); 
# 1637
}  
# 1639
for (; (__first1 != __last1) && (__first2 != __last2); (++__first1), ((void)(++__first2))) { 
# 1641
if (!((bool)__binary_pred(*__first1, *__first2))) { 
# 1642
return false; }  }  
# 1643
return (__first1 == __last1) && (__first2 == __last2); 
# 1644
} 
# 1664 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _II1, class _II2> inline bool 
# 1667
equal(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 1668
{ 
# 1675
; 
# 1676
; 
# 1678
return std::__equal4(__first1, __last1, __first2, __last2); 
# 1679
} 
# 1697 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _IIter1, class _IIter2, class _BinaryPredicate> inline bool 
# 1700
equal(_IIter1 __first1, _IIter1 __last1, _IIter2 
# 1701
__first2, _IIter2 __last2, _BinaryPredicate __binary_pred) 
# 1702
{ 
# 1706
; 
# 1707
; 
# 1709
return std::__equal4(__first1, __last1, __first2, __last2, __binary_pred); 
# 1711
} 
# 1729 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _II1, class _II2> inline bool 
# 1732
lexicographical_compare(_II1 __first1, _II1 __last1, _II2 
# 1733
__first2, _II2 __last2) 
# 1734
{ 
# 1744
; 
# 1745
; 
# 1747
return std::__lexicographical_compare_aux(__first1, __last1, __first2, __last2); 
# 1749
} 
# 1764 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _II1, class _II2, class _Compare> inline bool 
# 1767
lexicographical_compare(_II1 __first1, _II1 __last1, _II2 
# 1768
__first2, _II2 __last2, _Compare __comp) 
# 1769
{ 
# 1773
; 
# 1774
; 
# 1776
return std::__lexicographical_compare_impl(__first1, __last1, __first2, __last2, __gnu_cxx::__ops::__iter_comp_iter(__comp)); 
# 1779
} 
# 1877 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2, class 
# 1878
_BinaryPredicate> pair< _InputIterator1, _InputIterator2>  
# 1881
__mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1882
__first2, _BinaryPredicate __binary_pred) 
# 1883
{ 
# 1884
while ((__first1 != __last1) && __binary_pred(__first1, __first2)) 
# 1885
{ 
# 1886
++__first1; 
# 1887
++__first2; 
# 1888
}  
# 1889
return pair< _InputIterator1, _InputIterator2> (__first1, __first2); 
# 1890
} 
# 1905 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2> inline pair< _InputIterator1, _InputIterator2>  
# 1908
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1909
__first2) 
# 1910
{ 
# 1917
; 
# 1919
return std::__mismatch(__first1, __last1, __first2, __gnu_cxx::__ops::__iter_equal_to_iter()); 
# 1921
} 
# 1939 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2, class 
# 1940
_BinaryPredicate> inline pair< _InputIterator1, _InputIterator2>  
# 1943
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1944
__first2, _BinaryPredicate __binary_pred) 
# 1945
{ 
# 1949
; 
# 1951
return std::__mismatch(__first1, __last1, __first2, __gnu_cxx::__ops::__iter_comp_iter(__binary_pred)); 
# 1953
} 
# 1957
template< class _InputIterator1, class _InputIterator2, class 
# 1958
_BinaryPredicate> pair< _InputIterator1, _InputIterator2>  
# 1961
__mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1962
__first2, _InputIterator2 __last2, _BinaryPredicate 
# 1963
__binary_pred) 
# 1964
{ 
# 1965
while ((__first1 != __last1) && (__first2 != __last2) && __binary_pred(__first1, __first2)) 
# 1967
{ 
# 1968
++__first1; 
# 1969
++__first2; 
# 1970
}  
# 1971
return pair< _InputIterator1, _InputIterator2> (__first1, __first2); 
# 1972
} 
# 1988 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2> inline pair< _InputIterator1, _InputIterator2>  
# 1991
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1992
__first2, _InputIterator2 __last2) 
# 1993
{ 
# 2000
; 
# 2001
; 
# 2003
return std::__mismatch(__first1, __last1, __first2, __last2, __gnu_cxx::__ops::__iter_equal_to_iter()); 
# 2005
} 
# 2024 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _InputIterator1, class _InputIterator2, class 
# 2025
_BinaryPredicate> inline pair< _InputIterator1, _InputIterator2>  
# 2028
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 2029
__first2, _InputIterator2 __last2, _BinaryPredicate 
# 2030
__binary_pred) 
# 2031
{ 
# 2035
; 
# 2036
; 
# 2038
return std::__mismatch(__first1, __last1, __first2, __last2, __gnu_cxx::__ops::__iter_comp_iter(__binary_pred)); 
# 2040
} 
# 2046
template< class _InputIterator, class _Predicate> inline _InputIterator 
# 2049
__find_if(_InputIterator __first, _InputIterator __last, _Predicate 
# 2050
__pred, input_iterator_tag) 
# 2051
{ 
# 2052
while ((__first != __last) && (!__pred(__first))) { 
# 2053
++__first; }  
# 2054
return __first; 
# 2055
} 
# 2058
template< class _RandomAccessIterator, class _Predicate> _RandomAccessIterator 
# 2061
__find_if(_RandomAccessIterator __first, _RandomAccessIterator __last, _Predicate 
# 2062
__pred, random_access_iterator_tag) 
# 2063
{ 
# 2065
typename iterator_traits< _RandomAccessIterator> ::difference_type __trip_count = (__last - __first) >> 2; 
# 2067
for (; __trip_count > 0; --__trip_count) 
# 2068
{ 
# 2069
if (__pred(__first)) { 
# 2070
return __first; }  
# 2071
++__first; 
# 2073
if (__pred(__first)) { 
# 2074
return __first; }  
# 2075
++__first; 
# 2077
if (__pred(__first)) { 
# 2078
return __first; }  
# 2079
++__first; 
# 2081
if (__pred(__first)) { 
# 2082
return __first; }  
# 2083
++__first; 
# 2084
}  
# 2086
switch (__last - __first) 
# 2087
{ 
# 2088
case 3:  
# 2089
if (__pred(__first)) { 
# 2090
return __first; }  
# 2091
++__first; 
# 2093
case 2:  
# 2094
if (__pred(__first)) { 
# 2095
return __first; }  
# 2096
++__first; 
# 2098
case 1:  
# 2099
if (__pred(__first)) { 
# 2100
return __first; }  
# 2101
++__first; 
# 2103
case 0:  
# 2104
default:  
# 2105
return __last; 
# 2106
}  
# 2107
} 
# 2109
template< class _Iterator, class _Predicate> inline _Iterator 
# 2112
__find_if(_Iterator __first, _Iterator __last, _Predicate __pred) 
# 2113
{ 
# 2114
return __find_if(__first, __last, __pred, std::__iterator_category(__first)); 
# 2116
} 
# 2118
template< class _InputIterator, class _Predicate> typename iterator_traits< _InputIterator> ::difference_type 
# 2121
__count_if(_InputIterator __first, _InputIterator __last, _Predicate __pred) 
# 2122
{ 
# 2123
typename iterator_traits< _InputIterator> ::difference_type __n = (0); 
# 2124
for (; __first != __last; ++__first) { 
# 2125
if (__pred(__first)) { 
# 2126
++__n; }  }  
# 2127
return __n; 
# 2128
} 
# 2131
template< class _ForwardIterator1, class _ForwardIterator2, class 
# 2132
_BinaryPredicate> bool 
# 2135
__is_permutation(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 
# 2136
__first2, _BinaryPredicate __pred) 
# 2137
{ 
# 2140
for (; __first1 != __last1; (++__first1), ((void)(++__first2))) { 
# 2141
if (!__pred(__first1, __first2)) { 
# 2142
break; }  }  
# 2144
if (__first1 == __last1) { 
# 2145
return true; }  
# 2149
_ForwardIterator2 __last2 = __first2; 
# 2150
std::advance(__last2, std::distance(__first1, __last1)); 
# 2151
for (_ForwardIterator1 __scan = __first1; __scan != __last1; ++__scan) 
# 2152
{ 
# 2153
if (__scan != std::__find_if(__first1, __scan, __gnu_cxx::__ops::__iter_comp_iter(__pred, __scan))) { 
# 2155
continue; }  
# 2157
auto __matches = std::__count_if(__first2, __last2, __gnu_cxx::__ops::__iter_comp_iter(__pred, __scan)); 
# 2160
if ((0 == __matches) || (std::__count_if(__scan, __last1, __gnu_cxx::__ops::__iter_comp_iter(__pred, __scan)) != __matches)) { 
# 2164
return false; }  
# 2165
}   
# 2166
return true; 
# 2167
} 
# 2181 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/stl_algobase.h" 3
template< class _ForwardIterator1, class _ForwardIterator2> inline bool 
# 2184
is_permutation(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 
# 2185
__first2) 
# 2186
{ 
# 2193
; 
# 2195
return std::__is_permutation(__first1, __last1, __first2, __gnu_cxx::__ops::__iter_equal_to_iter()); 
# 2197
} 
# 2201
}
# 158 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/limits" 3
namespace std __attribute((__visibility__("default"))) { 
# 167
enum float_round_style { 
# 169
round_indeterminate = (-1), 
# 170
round_toward_zero = 0, 
# 171
round_to_nearest, 
# 172
round_toward_infinity, 
# 173
round_toward_neg_infinity
# 174
}; 
# 182
enum float_denorm_style { 
# 185
denorm_indeterminate = (-1), 
# 187
denorm_absent = 0, 
# 189
denorm_present
# 190
}; 
# 202 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/limits" 3
struct __numeric_limits_base { 
# 206
static constexpr inline bool is_specialized = false; 
# 211
static constexpr inline int digits = 0; 
# 214
static constexpr inline int digits10 = 0; 
# 219
static constexpr inline int max_digits10 = 0; 
# 223
static constexpr inline bool is_signed = false; 
# 226
static constexpr inline bool is_integer = false; 
# 231
static constexpr inline bool is_exact = false; 
# 235
static constexpr inline int radix = 0; 
# 239
static constexpr inline int min_exponent = 0; 
# 243
static constexpr inline int min_exponent10 = 0; 
# 248
static constexpr inline int max_exponent = 0; 
# 252
static constexpr inline int max_exponent10 = 0; 
# 255
static constexpr inline bool has_infinity = false; 
# 259
static constexpr inline bool has_quiet_NaN = false; 
# 263
static constexpr inline bool has_signaling_NaN = false; 
# 266
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 270
static constexpr inline bool has_denorm_loss = false; 
# 274
static constexpr inline bool is_iec559 = false; 
# 279
static constexpr inline bool is_bounded = false; 
# 288 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/limits" 3
static constexpr inline bool is_modulo = false; 
# 291
static constexpr inline bool traps = false; 
# 294
static constexpr inline bool tinyness_before = false; 
# 299
static constexpr inline float_round_style round_style = round_toward_zero; 
# 301
}; 
# 311 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/limits" 3
template< class _Tp> 
# 312
struct numeric_limits : public __numeric_limits_base { 
# 317
static constexpr _Tp min() noexcept { return _Tp(); } 
# 321
static constexpr _Tp max() noexcept { return _Tp(); } 
# 327
static constexpr _Tp lowest() noexcept { return _Tp(); } 
# 333
static constexpr _Tp epsilon() noexcept { return _Tp(); } 
# 337
static constexpr _Tp round_error() noexcept { return _Tp(); } 
# 341
static constexpr _Tp infinity() noexcept { return _Tp(); } 
# 346
static constexpr _Tp quiet_NaN() noexcept { return _Tp(); } 
# 351
static constexpr _Tp signaling_NaN() noexcept { return _Tp(); } 
# 357
static constexpr _Tp denorm_min() noexcept { return _Tp(); } 
# 358
}; 
# 363
template< class _Tp> 
# 364
struct numeric_limits< const _Tp>  : public std::numeric_limits< _Tp>  { 
# 365
}; 
# 367
template< class _Tp> 
# 368
struct numeric_limits< volatile _Tp>  : public std::numeric_limits< _Tp>  { 
# 369
}; 
# 371
template< class _Tp> 
# 372
struct numeric_limits< const volatile _Tp>  : public std::numeric_limits< _Tp>  { 
# 373
}; 
# 384 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/limits" 3
template<> struct numeric_limits< bool>  { 
# 386
static constexpr inline bool is_specialized = true; 
# 389
static constexpr bool min() noexcept { return false; } 
# 392
static constexpr bool max() noexcept { return true; } 
# 396
static constexpr bool lowest() noexcept { return min(); } 
# 398
static constexpr inline int digits = 1; 
# 399
static constexpr inline int digits10 = 0; 
# 401
static constexpr inline int max_digits10 = 0; 
# 403
static constexpr inline bool is_signed = false; 
# 404
static constexpr inline bool is_integer = true; 
# 405
static constexpr inline bool is_exact = true; 
# 406
static constexpr inline int radix = 2; 
# 409
static constexpr bool epsilon() noexcept { return false; } 
# 412
static constexpr bool round_error() noexcept { return false; } 
# 414
static constexpr inline int min_exponent = 0; 
# 415
static constexpr inline int min_exponent10 = 0; 
# 416
static constexpr inline int max_exponent = 0; 
# 417
static constexpr inline int max_exponent10 = 0; 
# 419
static constexpr inline bool has_infinity = false; 
# 420
static constexpr inline bool has_quiet_NaN = false; 
# 421
static constexpr inline bool has_signaling_NaN = false; 
# 422
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 424
static constexpr inline bool has_denorm_loss = false; 
# 427
static constexpr bool infinity() noexcept { return false; } 
# 430
static constexpr bool quiet_NaN() noexcept { return false; } 
# 433
static constexpr bool signaling_NaN() noexcept { return false; } 
# 436
static constexpr bool denorm_min() noexcept { return false; } 
# 438
static constexpr inline bool is_iec559 = false; 
# 439
static constexpr inline bool is_bounded = true; 
# 440
static constexpr inline bool is_modulo = false; 
# 445
static constexpr inline bool traps = true; 
# 446
static constexpr inline bool tinyness_before = false; 
# 447
static constexpr inline float_round_style round_style = round_toward_zero; 
# 449
}; 
# 453
template<> struct numeric_limits< char>  { 
# 455
static constexpr inline bool is_specialized = true; 
# 458
static constexpr char min() noexcept { return ((((char)(-1)) < 0) ? (-((((char)(-1)) < 0) ? (((((char)1) << (((sizeof(char) * (8)) - (((char)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char)0)))) - 1 : ((char)0)); } 
# 461
static constexpr char max() noexcept { return ((((char)(-1)) < 0) ? (((((char)1) << (((sizeof(char) * (8)) - (((char)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char)0))); } 
# 465
static constexpr char lowest() noexcept { return min(); } 
# 468
static constexpr inline int digits = ((sizeof(char) * (8)) - (((char)(-1)) < 0)); 
# 469
static constexpr inline int digits10 = ((((sizeof(char) * (8)) - (((char)(-1)) < 0)) * (643L)) / (2136)); 
# 471
static constexpr inline int max_digits10 = 0; 
# 473
static constexpr inline bool is_signed = (((char)(-1)) < 0); 
# 474
static constexpr inline bool is_integer = true; 
# 475
static constexpr inline bool is_exact = true; 
# 476
static constexpr inline int radix = 2; 
# 479
static constexpr char epsilon() noexcept { return 0; } 
# 482
static constexpr char round_error() noexcept { return 0; } 
# 484
static constexpr inline int min_exponent = 0; 
# 485
static constexpr inline int min_exponent10 = 0; 
# 486
static constexpr inline int max_exponent = 0; 
# 487
static constexpr inline int max_exponent10 = 0; 
# 489
static constexpr inline bool has_infinity = false; 
# 490
static constexpr inline bool has_quiet_NaN = false; 
# 491
static constexpr inline bool has_signaling_NaN = false; 
# 492
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 494
static constexpr inline bool has_denorm_loss = false; 
# 497
static constexpr char infinity() noexcept { return ((char)0); } 
# 500
static constexpr char quiet_NaN() noexcept { return ((char)0); } 
# 503
static constexpr char signaling_NaN() noexcept { return ((char)0); } 
# 506
static constexpr char denorm_min() noexcept { return static_cast< char>(0); } 
# 508
static constexpr inline bool is_iec559 = false; 
# 509
static constexpr inline bool is_bounded = true; 
# 510
static constexpr inline bool is_modulo = (!is_signed); 
# 512
static constexpr inline bool traps = true; 
# 513
static constexpr inline bool tinyness_before = false; 
# 514
static constexpr inline float_round_style round_style = round_toward_zero; 
# 516
}; 
# 520
template<> struct numeric_limits< signed char>  { 
# 522
static constexpr inline bool is_specialized = true; 
# 525
static constexpr signed char min() noexcept { return (-127) - 1; } 
# 528
static constexpr signed char max() noexcept { return 127; } 
# 532
static constexpr signed char lowest() noexcept { return min(); } 
# 535
static constexpr inline int digits = ((sizeof(signed char) * (8)) - (((signed char)(-1)) < 0)); 
# 536
static constexpr inline int digits10 = ((((sizeof(signed char) * (8)) - (((signed char)(-1)) < 0)) * (643L)) / (2136)); 
# 539
static constexpr inline int max_digits10 = 0; 
# 541
static constexpr inline bool is_signed = true; 
# 542
static constexpr inline bool is_integer = true; 
# 543
static constexpr inline bool is_exact = true; 
# 544
static constexpr inline int radix = 2; 
# 547
static constexpr signed char epsilon() noexcept { return 0; } 
# 550
static constexpr signed char round_error() noexcept { return 0; } 
# 552
static constexpr inline int min_exponent = 0; 
# 553
static constexpr inline int min_exponent10 = 0; 
# 554
static constexpr inline int max_exponent = 0; 
# 555
static constexpr inline int max_exponent10 = 0; 
# 557
static constexpr inline bool has_infinity = false; 
# 558
static constexpr inline bool has_quiet_NaN = false; 
# 559
static constexpr inline bool has_signaling_NaN = false; 
# 560
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 562
static constexpr inline bool has_denorm_loss = false; 
# 565
static constexpr signed char infinity() noexcept { return static_cast< signed char>(0); } 
# 568
static constexpr signed char quiet_NaN() noexcept { return static_cast< signed char>(0); } 
# 571
static constexpr signed char signaling_NaN() noexcept 
# 572
{ return static_cast< signed char>(0); } 
# 575
static constexpr signed char denorm_min() noexcept 
# 576
{ return static_cast< signed char>(0); } 
# 578
static constexpr inline bool is_iec559 = false; 
# 579
static constexpr inline bool is_bounded = true; 
# 580
static constexpr inline bool is_modulo = false; 
# 582
static constexpr inline bool traps = true; 
# 583
static constexpr inline bool tinyness_before = false; 
# 584
static constexpr inline float_round_style round_style = round_toward_zero; 
# 586
}; 
# 590
template<> struct numeric_limits< unsigned char>  { 
# 592
static constexpr inline bool is_specialized = true; 
# 595
static constexpr unsigned char min() noexcept { return 0; } 
# 598
static constexpr unsigned char max() noexcept { return ((127) * 2U) + (1); } 
# 602
static constexpr unsigned char lowest() noexcept { return min(); } 
# 605
static constexpr inline int digits = ((sizeof(unsigned char) * (8)) - (((unsigned char)(-1)) < 0)); 
# 607
static constexpr inline int digits10 = ((((sizeof(unsigned char) * (8)) - (((unsigned char)(-1)) < 0)) * (643L)) / (2136)); 
# 610
static constexpr inline int max_digits10 = 0; 
# 612
static constexpr inline bool is_signed = false; 
# 613
static constexpr inline bool is_integer = true; 
# 614
static constexpr inline bool is_exact = true; 
# 615
static constexpr inline int radix = 2; 
# 618
static constexpr unsigned char epsilon() noexcept { return 0; } 
# 621
static constexpr unsigned char round_error() noexcept { return 0; } 
# 623
static constexpr inline int min_exponent = 0; 
# 624
static constexpr inline int min_exponent10 = 0; 
# 625
static constexpr inline int max_exponent = 0; 
# 626
static constexpr inline int max_exponent10 = 0; 
# 628
static constexpr inline bool has_infinity = false; 
# 629
static constexpr inline bool has_quiet_NaN = false; 
# 630
static constexpr inline bool has_signaling_NaN = false; 
# 631
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 633
static constexpr inline bool has_denorm_loss = false; 
# 636
static constexpr unsigned char infinity() noexcept 
# 637
{ return static_cast< unsigned char>(0); } 
# 640
static constexpr unsigned char quiet_NaN() noexcept 
# 641
{ return static_cast< unsigned char>(0); } 
# 644
static constexpr unsigned char signaling_NaN() noexcept 
# 645
{ return static_cast< unsigned char>(0); } 
# 648
static constexpr unsigned char denorm_min() noexcept 
# 649
{ return static_cast< unsigned char>(0); } 
# 651
static constexpr inline bool is_iec559 = false; 
# 652
static constexpr inline bool is_bounded = true; 
# 653
static constexpr inline bool is_modulo = true; 
# 655
static constexpr inline bool traps = true; 
# 656
static constexpr inline bool tinyness_before = false; 
# 657
static constexpr inline float_round_style round_style = round_toward_zero; 
# 659
}; 
# 663
template<> struct numeric_limits< wchar_t>  { 
# 665
static constexpr inline bool is_specialized = true; 
# 668
static constexpr wchar_t min() noexcept { return ((((wchar_t)(-1)) < 0) ? (-((((wchar_t)(-1)) < 0) ? (((((wchar_t)1) << (((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((wchar_t)0)))) - 1 : ((wchar_t)0)); } 
# 671
static constexpr wchar_t max() noexcept { return ((((wchar_t)(-1)) < 0) ? (((((wchar_t)1) << (((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((wchar_t)0))); } 
# 675
static constexpr wchar_t lowest() noexcept { return min(); } 
# 678
static constexpr inline int digits = ((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)); 
# 679
static constexpr inline int digits10 = ((((sizeof(wchar_t) * (8)) - (((wchar_t)(-1)) < 0)) * (643L)) / (2136)); 
# 682
static constexpr inline int max_digits10 = 0; 
# 684
static constexpr inline bool is_signed = (((wchar_t)(-1)) < 0); 
# 685
static constexpr inline bool is_integer = true; 
# 686
static constexpr inline bool is_exact = true; 
# 687
static constexpr inline int radix = 2; 
# 690
static constexpr wchar_t epsilon() noexcept { return 0; } 
# 693
static constexpr wchar_t round_error() noexcept { return 0; } 
# 695
static constexpr inline int min_exponent = 0; 
# 696
static constexpr inline int min_exponent10 = 0; 
# 697
static constexpr inline int max_exponent = 0; 
# 698
static constexpr inline int max_exponent10 = 0; 
# 700
static constexpr inline bool has_infinity = false; 
# 701
static constexpr inline bool has_quiet_NaN = false; 
# 702
static constexpr inline bool has_signaling_NaN = false; 
# 703
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 705
static constexpr inline bool has_denorm_loss = false; 
# 708
static constexpr wchar_t infinity() noexcept { return ((wchar_t)0); } 
# 711
static constexpr wchar_t quiet_NaN() noexcept { return ((wchar_t)0); } 
# 714
static constexpr wchar_t signaling_NaN() noexcept { return ((wchar_t)0); } 
# 717
static constexpr wchar_t denorm_min() noexcept { return ((wchar_t)0); } 
# 719
static constexpr inline bool is_iec559 = false; 
# 720
static constexpr inline bool is_bounded = true; 
# 721
static constexpr inline bool is_modulo = (!is_signed); 
# 723
static constexpr inline bool traps = true; 
# 724
static constexpr inline bool tinyness_before = false; 
# 725
static constexpr inline float_round_style round_style = round_toward_zero; 
# 727
}; 
# 797 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/limits" 3
template<> struct numeric_limits< char16_t>  { 
# 799
static constexpr inline bool is_specialized = true; 
# 802
static constexpr char16_t min() noexcept { return ((((char16_t)(-1)) < 0) ? (-((((char16_t)(-1)) < 0) ? (((((char16_t)1) << (((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char16_t)0)))) - 1 : ((char16_t)0)); } 
# 805
static constexpr char16_t max() noexcept { return ((((char16_t)(-1)) < 0) ? (((((char16_t)1) << (((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((char16_t)0))); } 
# 808
static constexpr char16_t lowest() noexcept { return min(); } 
# 810
static constexpr inline int digits = ((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)); 
# 811
static constexpr inline int digits10 = ((((sizeof(char16_t) * (8)) - (((char16_t)(-1)) < 0)) * (643L)) / (2136)); 
# 812
static constexpr inline int max_digits10 = 0; 
# 813
static constexpr inline bool is_signed = (((char16_t)(-1)) < 0); 
# 814
static constexpr inline bool is_integer = true; 
# 815
static constexpr inline bool is_exact = true; 
# 816
static constexpr inline int radix = 2; 
# 819
static constexpr char16_t epsilon() noexcept { return 0; } 
# 822
static constexpr char16_t round_error() noexcept { return 0; } 
# 824
static constexpr inline int min_exponent = 0; 
# 825
static constexpr inline int min_exponent10 = 0; 
# 826
static constexpr inline int max_exponent = 0; 
# 827
static constexpr inline int max_exponent10 = 0; 
# 829
static constexpr inline bool has_infinity = false; 
# 830
static constexpr inline bool has_quiet_NaN = false; 
# 831
static constexpr inline bool has_signaling_NaN = false; 
# 832
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 833
static constexpr inline bool has_denorm_loss = false; 
# 836
static constexpr char16_t infinity() noexcept { return ((char16_t)0); } 
# 839
static constexpr char16_t quiet_NaN() noexcept { return ((char16_t)0); } 
# 842
static constexpr char16_t signaling_NaN() noexcept { return ((char16_t)0); } 
# 845
static constexpr char16_t denorm_min() noexcept { return ((char16_t)0); } 
# 847
static constexpr inline bool is_iec559 = false; 
# 848
static constexpr inline bool is_bounded = true; 
# 849
static constexpr inline bool is_modulo = (!is_signed); 
# 851
static constexpr inline bool traps = true; 
# 852
static constexpr inline bool tinyness_before = false; 
# 853
static constexpr inline float_round_style round_style = round_toward_zero; 
# 854
}; 
# 858
template<> struct numeric_limits< char32_t>  { 
# 860
static constexpr inline bool is_specialized = true; 
# 863
static constexpr char32_t min() noexcept { return ((((char32_t)(-1)) < (0)) ? (-((((char32_t)(-1)) < (0)) ? (((((char32_t)1) << (((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))) - (1))) - (1)) << 1) + (1) : (~((char32_t)0)))) - (1) : ((char32_t)0)); } 
# 866
static constexpr char32_t max() noexcept { return ((((char32_t)(-1)) < (0)) ? (((((char32_t)1) << (((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))) - (1))) - (1)) << 1) + (1) : (~((char32_t)0))); } 
# 869
static constexpr char32_t lowest() noexcept { return min(); } 
# 871
static constexpr inline int digits = ((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))); 
# 872
static constexpr inline int digits10 = ((((sizeof(char32_t) * (8)) - (((char32_t)(-1)) < (0))) * (643L)) / (2136)); 
# 873
static constexpr inline int max_digits10 = 0; 
# 874
static constexpr inline bool is_signed = (((char32_t)(-1)) < (0)); 
# 875
static constexpr inline bool is_integer = true; 
# 876
static constexpr inline bool is_exact = true; 
# 877
static constexpr inline int radix = 2; 
# 880
static constexpr char32_t epsilon() noexcept { return 0; } 
# 883
static constexpr char32_t round_error() noexcept { return 0; } 
# 885
static constexpr inline int min_exponent = 0; 
# 886
static constexpr inline int min_exponent10 = 0; 
# 887
static constexpr inline int max_exponent = 0; 
# 888
static constexpr inline int max_exponent10 = 0; 
# 890
static constexpr inline bool has_infinity = false; 
# 891
static constexpr inline bool has_quiet_NaN = false; 
# 892
static constexpr inline bool has_signaling_NaN = false; 
# 893
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 894
static constexpr inline bool has_denorm_loss = false; 
# 897
static constexpr char32_t infinity() noexcept { return ((char32_t)0); } 
# 900
static constexpr char32_t quiet_NaN() noexcept { return ((char32_t)0); } 
# 903
static constexpr char32_t signaling_NaN() noexcept { return ((char32_t)0); } 
# 906
static constexpr char32_t denorm_min() noexcept { return ((char32_t)0); } 
# 908
static constexpr inline bool is_iec559 = false; 
# 909
static constexpr inline bool is_bounded = true; 
# 910
static constexpr inline bool is_modulo = (!is_signed); 
# 912
static constexpr inline bool traps = true; 
# 913
static constexpr inline bool tinyness_before = false; 
# 914
static constexpr inline float_round_style round_style = round_toward_zero; 
# 915
}; 
# 920
template<> struct numeric_limits< short>  { 
# 922
static constexpr inline bool is_specialized = true; 
# 925
static constexpr short min() noexcept { return (-32767) - 1; } 
# 928
static constexpr short max() noexcept { return 32767; } 
# 932
static constexpr short lowest() noexcept { return min(); } 
# 935
static constexpr inline int digits = ((sizeof(short) * (8)) - (((short)(-1)) < 0)); 
# 936
static constexpr inline int digits10 = ((((sizeof(short) * (8)) - (((short)(-1)) < 0)) * (643L)) / (2136)); 
# 938
static constexpr inline int max_digits10 = 0; 
# 940
static constexpr inline bool is_signed = true; 
# 941
static constexpr inline bool is_integer = true; 
# 942
static constexpr inline bool is_exact = true; 
# 943
static constexpr inline int radix = 2; 
# 946
static constexpr short epsilon() noexcept { return 0; } 
# 949
static constexpr short round_error() noexcept { return 0; } 
# 951
static constexpr inline int min_exponent = 0; 
# 952
static constexpr inline int min_exponent10 = 0; 
# 953
static constexpr inline int max_exponent = 0; 
# 954
static constexpr inline int max_exponent10 = 0; 
# 956
static constexpr inline bool has_infinity = false; 
# 957
static constexpr inline bool has_quiet_NaN = false; 
# 958
static constexpr inline bool has_signaling_NaN = false; 
# 959
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 961
static constexpr inline bool has_denorm_loss = false; 
# 964
static constexpr short infinity() noexcept { return ((short)0); } 
# 967
static constexpr short quiet_NaN() noexcept { return ((short)0); } 
# 970
static constexpr short signaling_NaN() noexcept { return ((short)0); } 
# 973
static constexpr short denorm_min() noexcept { return ((short)0); } 
# 975
static constexpr inline bool is_iec559 = false; 
# 976
static constexpr inline bool is_bounded = true; 
# 977
static constexpr inline bool is_modulo = false; 
# 979
static constexpr inline bool traps = true; 
# 980
static constexpr inline bool tinyness_before = false; 
# 981
static constexpr inline float_round_style round_style = round_toward_zero; 
# 983
}; 
# 987
template<> struct numeric_limits< unsigned short>  { 
# 989
static constexpr inline bool is_specialized = true; 
# 992
static constexpr unsigned short min() noexcept { return 0; } 
# 995
static constexpr unsigned short max() noexcept { return ((32767) * 2U) + (1); } 
# 999
static constexpr unsigned short lowest() noexcept { return min(); } 
# 1002
static constexpr inline int digits = ((sizeof(unsigned short) * (8)) - (((unsigned short)(-1)) < 0)); 
# 1004
static constexpr inline int digits10 = ((((sizeof(unsigned short) * (8)) - (((unsigned short)(-1)) < 0)) * (643L)) / (2136)); 
# 1007
static constexpr inline int max_digits10 = 0; 
# 1009
static constexpr inline bool is_signed = false; 
# 1010
static constexpr inline bool is_integer = true; 
# 1011
static constexpr inline bool is_exact = true; 
# 1012
static constexpr inline int radix = 2; 
# 1015
static constexpr unsigned short epsilon() noexcept { return 0; } 
# 1018
static constexpr unsigned short round_error() noexcept { return 0; } 
# 1020
static constexpr inline int min_exponent = 0; 
# 1021
static constexpr inline int min_exponent10 = 0; 
# 1022
static constexpr inline int max_exponent = 0; 
# 1023
static constexpr inline int max_exponent10 = 0; 
# 1025
static constexpr inline bool has_infinity = false; 
# 1026
static constexpr inline bool has_quiet_NaN = false; 
# 1027
static constexpr inline bool has_signaling_NaN = false; 
# 1028
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1030
static constexpr inline bool has_denorm_loss = false; 
# 1033
static constexpr unsigned short infinity() noexcept 
# 1034
{ return static_cast< unsigned short>(0); } 
# 1037
static constexpr unsigned short quiet_NaN() noexcept 
# 1038
{ return static_cast< unsigned short>(0); } 
# 1041
static constexpr unsigned short signaling_NaN() noexcept 
# 1042
{ return static_cast< unsigned short>(0); } 
# 1045
static constexpr unsigned short denorm_min() noexcept 
# 1046
{ return static_cast< unsigned short>(0); } 
# 1048
static constexpr inline bool is_iec559 = false; 
# 1049
static constexpr inline bool is_bounded = true; 
# 1050
static constexpr inline bool is_modulo = true; 
# 1052
static constexpr inline bool traps = true; 
# 1053
static constexpr inline bool tinyness_before = false; 
# 1054
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1056
}; 
# 1060
template<> struct numeric_limits< int>  { 
# 1062
static constexpr inline bool is_specialized = true; 
# 1065
static constexpr int min() noexcept { return (-2147483647) - 1; } 
# 1068
static constexpr int max() noexcept { return 2147483647; } 
# 1072
static constexpr int lowest() noexcept { return min(); } 
# 1075
static constexpr inline int digits = ((sizeof(int) * (8)) - (((int)(-1)) < 0)); 
# 1076
static constexpr inline int digits10 = ((((sizeof(int) * (8)) - (((int)(-1)) < 0)) * (643L)) / (2136)); 
# 1078
static constexpr inline int max_digits10 = 0; 
# 1080
static constexpr inline bool is_signed = true; 
# 1081
static constexpr inline bool is_integer = true; 
# 1082
static constexpr inline bool is_exact = true; 
# 1083
static constexpr inline int radix = 2; 
# 1086
static constexpr int epsilon() noexcept { return 0; } 
# 1089
static constexpr int round_error() noexcept { return 0; } 
# 1091
static constexpr inline int min_exponent = 0; 
# 1092
static constexpr inline int min_exponent10 = 0; 
# 1093
static constexpr inline int max_exponent = 0; 
# 1094
static constexpr inline int max_exponent10 = 0; 
# 1096
static constexpr inline bool has_infinity = false; 
# 1097
static constexpr inline bool has_quiet_NaN = false; 
# 1098
static constexpr inline bool has_signaling_NaN = false; 
# 1099
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1101
static constexpr inline bool has_denorm_loss = false; 
# 1104
static constexpr int infinity() noexcept { return static_cast< int>(0); } 
# 1107
static constexpr int quiet_NaN() noexcept { return static_cast< int>(0); } 
# 1110
static constexpr int signaling_NaN() noexcept { return static_cast< int>(0); } 
# 1113
static constexpr int denorm_min() noexcept { return static_cast< int>(0); } 
# 1115
static constexpr inline bool is_iec559 = false; 
# 1116
static constexpr inline bool is_bounded = true; 
# 1117
static constexpr inline bool is_modulo = false; 
# 1119
static constexpr inline bool traps = true; 
# 1120
static constexpr inline bool tinyness_before = false; 
# 1121
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1123
}; 
# 1127
template<> struct numeric_limits< unsigned>  { 
# 1129
static constexpr inline bool is_specialized = true; 
# 1132
static constexpr unsigned min() noexcept { return 0; } 
# 1135
static constexpr unsigned max() noexcept { return ((2147483647) * 2U) + (1); } 
# 1139
static constexpr unsigned lowest() noexcept { return min(); } 
# 1142
static constexpr inline int digits = ((sizeof(unsigned) * (8)) - (((unsigned)(-1)) < (0))); 
# 1144
static constexpr inline int digits10 = ((((sizeof(unsigned) * (8)) - (((unsigned)(-1)) < (0))) * (643L)) / (2136)); 
# 1147
static constexpr inline int max_digits10 = 0; 
# 1149
static constexpr inline bool is_signed = false; 
# 1150
static constexpr inline bool is_integer = true; 
# 1151
static constexpr inline bool is_exact = true; 
# 1152
static constexpr inline int radix = 2; 
# 1155
static constexpr unsigned epsilon() noexcept { return 0; } 
# 1158
static constexpr unsigned round_error() noexcept { return 0; } 
# 1160
static constexpr inline int min_exponent = 0; 
# 1161
static constexpr inline int min_exponent10 = 0; 
# 1162
static constexpr inline int max_exponent = 0; 
# 1163
static constexpr inline int max_exponent10 = 0; 
# 1165
static constexpr inline bool has_infinity = false; 
# 1166
static constexpr inline bool has_quiet_NaN = false; 
# 1167
static constexpr inline bool has_signaling_NaN = false; 
# 1168
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1170
static constexpr inline bool has_denorm_loss = false; 
# 1173
static constexpr unsigned infinity() noexcept { return static_cast< unsigned>(0); } 
# 1176
static constexpr unsigned quiet_NaN() noexcept 
# 1177
{ return static_cast< unsigned>(0); } 
# 1180
static constexpr unsigned signaling_NaN() noexcept 
# 1181
{ return static_cast< unsigned>(0); } 
# 1184
static constexpr unsigned denorm_min() noexcept 
# 1185
{ return static_cast< unsigned>(0); } 
# 1187
static constexpr inline bool is_iec559 = false; 
# 1188
static constexpr inline bool is_bounded = true; 
# 1189
static constexpr inline bool is_modulo = true; 
# 1191
static constexpr inline bool traps = true; 
# 1192
static constexpr inline bool tinyness_before = false; 
# 1193
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1195
}; 
# 1199
template<> struct numeric_limits< long>  { 
# 1201
static constexpr inline bool is_specialized = true; 
# 1204
static constexpr long min() noexcept { return (-9223372036854775807L) - (1); } 
# 1207
static constexpr long max() noexcept { return 9223372036854775807L; } 
# 1211
static constexpr long lowest() noexcept { return min(); } 
# 1214
static constexpr inline int digits = ((sizeof(long) * (8)) - (((long)(-1)) < (0))); 
# 1215
static constexpr inline int digits10 = ((((sizeof(long) * (8)) - (((long)(-1)) < (0))) * (643L)) / (2136)); 
# 1217
static constexpr inline int max_digits10 = 0; 
# 1219
static constexpr inline bool is_signed = true; 
# 1220
static constexpr inline bool is_integer = true; 
# 1221
static constexpr inline bool is_exact = true; 
# 1222
static constexpr inline int radix = 2; 
# 1225
static constexpr long epsilon() noexcept { return 0; } 
# 1228
static constexpr long round_error() noexcept { return 0; } 
# 1230
static constexpr inline int min_exponent = 0; 
# 1231
static constexpr inline int min_exponent10 = 0; 
# 1232
static constexpr inline int max_exponent = 0; 
# 1233
static constexpr inline int max_exponent10 = 0; 
# 1235
static constexpr inline bool has_infinity = false; 
# 1236
static constexpr inline bool has_quiet_NaN = false; 
# 1237
static constexpr inline bool has_signaling_NaN = false; 
# 1238
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1240
static constexpr inline bool has_denorm_loss = false; 
# 1243
static constexpr long infinity() noexcept { return static_cast< long>(0); } 
# 1246
static constexpr long quiet_NaN() noexcept { return static_cast< long>(0); } 
# 1249
static constexpr long signaling_NaN() noexcept { return static_cast< long>(0); } 
# 1252
static constexpr long denorm_min() noexcept { return static_cast< long>(0); } 
# 1254
static constexpr inline bool is_iec559 = false; 
# 1255
static constexpr inline bool is_bounded = true; 
# 1256
static constexpr inline bool is_modulo = false; 
# 1258
static constexpr inline bool traps = true; 
# 1259
static constexpr inline bool tinyness_before = false; 
# 1260
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1262
}; 
# 1266
template<> struct numeric_limits< unsigned long>  { 
# 1268
static constexpr inline bool is_specialized = true; 
# 1271
static constexpr unsigned long min() noexcept { return 0; } 
# 1274
static constexpr unsigned long max() noexcept { return ((9223372036854775807L) * 2UL) + (1); } 
# 1278
static constexpr unsigned long lowest() noexcept { return min(); } 
# 1281
static constexpr inline int digits = ((sizeof(unsigned long) * (8)) - (((unsigned long)(-1)) < (0))); 
# 1283
static constexpr inline int digits10 = ((((sizeof(unsigned long) * (8)) - (((unsigned long)(-1)) < (0))) * (643L)) / (2136)); 
# 1286
static constexpr inline int max_digits10 = 0; 
# 1288
static constexpr inline bool is_signed = false; 
# 1289
static constexpr inline bool is_integer = true; 
# 1290
static constexpr inline bool is_exact = true; 
# 1291
static constexpr inline int radix = 2; 
# 1294
static constexpr unsigned long epsilon() noexcept { return 0; } 
# 1297
static constexpr unsigned long round_error() noexcept { return 0; } 
# 1299
static constexpr inline int min_exponent = 0; 
# 1300
static constexpr inline int min_exponent10 = 0; 
# 1301
static constexpr inline int max_exponent = 0; 
# 1302
static constexpr inline int max_exponent10 = 0; 
# 1304
static constexpr inline bool has_infinity = false; 
# 1305
static constexpr inline bool has_quiet_NaN = false; 
# 1306
static constexpr inline bool has_signaling_NaN = false; 
# 1307
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1309
static constexpr inline bool has_denorm_loss = false; 
# 1312
static constexpr unsigned long infinity() noexcept 
# 1313
{ return static_cast< unsigned long>(0); } 
# 1316
static constexpr unsigned long quiet_NaN() noexcept 
# 1317
{ return static_cast< unsigned long>(0); } 
# 1320
static constexpr unsigned long signaling_NaN() noexcept 
# 1321
{ return static_cast< unsigned long>(0); } 
# 1324
static constexpr unsigned long denorm_min() noexcept 
# 1325
{ return static_cast< unsigned long>(0); } 
# 1327
static constexpr inline bool is_iec559 = false; 
# 1328
static constexpr inline bool is_bounded = true; 
# 1329
static constexpr inline bool is_modulo = true; 
# 1331
static constexpr inline bool traps = true; 
# 1332
static constexpr inline bool tinyness_before = false; 
# 1333
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1335
}; 
# 1339
template<> struct numeric_limits< long long>  { 
# 1341
static constexpr inline bool is_specialized = true; 
# 1344
static constexpr long long min() noexcept { return (-9223372036854775807LL) - (1); } 
# 1347
static constexpr long long max() noexcept { return 9223372036854775807LL; } 
# 1351
static constexpr long long lowest() noexcept { return min(); } 
# 1354
static constexpr inline int digits = ((sizeof(long long) * (8)) - (((long long)(-1)) < (0))); 
# 1356
static constexpr inline int digits10 = ((((sizeof(long long) * (8)) - (((long long)(-1)) < (0))) * (643L)) / (2136)); 
# 1359
static constexpr inline int max_digits10 = 0; 
# 1361
static constexpr inline bool is_signed = true; 
# 1362
static constexpr inline bool is_integer = true; 
# 1363
static constexpr inline bool is_exact = true; 
# 1364
static constexpr inline int radix = 2; 
# 1367
static constexpr long long epsilon() noexcept { return 0; } 
# 1370
static constexpr long long round_error() noexcept { return 0; } 
# 1372
static constexpr inline int min_exponent = 0; 
# 1373
static constexpr inline int min_exponent10 = 0; 
# 1374
static constexpr inline int max_exponent = 0; 
# 1375
static constexpr inline int max_exponent10 = 0; 
# 1377
static constexpr inline bool has_infinity = false; 
# 1378
static constexpr inline bool has_quiet_NaN = false; 
# 1379
static constexpr inline bool has_signaling_NaN = false; 
# 1380
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1382
static constexpr inline bool has_denorm_loss = false; 
# 1385
static constexpr long long infinity() noexcept { return static_cast< long long>(0); } 
# 1388
static constexpr long long quiet_NaN() noexcept { return static_cast< long long>(0); } 
# 1391
static constexpr long long signaling_NaN() noexcept 
# 1392
{ return static_cast< long long>(0); } 
# 1395
static constexpr long long denorm_min() noexcept { return static_cast< long long>(0); } 
# 1397
static constexpr inline bool is_iec559 = false; 
# 1398
static constexpr inline bool is_bounded = true; 
# 1399
static constexpr inline bool is_modulo = false; 
# 1401
static constexpr inline bool traps = true; 
# 1402
static constexpr inline bool tinyness_before = false; 
# 1403
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1405
}; 
# 1409
template<> struct numeric_limits< unsigned long long>  { 
# 1411
static constexpr inline bool is_specialized = true; 
# 1414
static constexpr unsigned long long min() noexcept { return 0; } 
# 1417
static constexpr unsigned long long max() noexcept { return ((9223372036854775807LL) * 2ULL) + (1); } 
# 1421
static constexpr unsigned long long lowest() noexcept { return min(); } 
# 1424
static constexpr inline int digits = ((sizeof(unsigned long long) * (8)) - (((unsigned long long)(-1)) < (0))); 
# 1426
static constexpr inline int digits10 = ((((sizeof(unsigned long long) * (8)) - (((unsigned long long)(-1)) < (0))) * (643L)) / (2136)); 
# 1429
static constexpr inline int max_digits10 = 0; 
# 1431
static constexpr inline bool is_signed = false; 
# 1432
static constexpr inline bool is_integer = true; 
# 1433
static constexpr inline bool is_exact = true; 
# 1434
static constexpr inline int radix = 2; 
# 1437
static constexpr unsigned long long epsilon() noexcept { return 0; } 
# 1440
static constexpr unsigned long long round_error() noexcept { return 0; } 
# 1442
static constexpr inline int min_exponent = 0; 
# 1443
static constexpr inline int min_exponent10 = 0; 
# 1444
static constexpr inline int max_exponent = 0; 
# 1445
static constexpr inline int max_exponent10 = 0; 
# 1447
static constexpr inline bool has_infinity = false; 
# 1448
static constexpr inline bool has_quiet_NaN = false; 
# 1449
static constexpr inline bool has_signaling_NaN = false; 
# 1450
static constexpr inline float_denorm_style has_denorm = denorm_absent; 
# 1452
static constexpr inline bool has_denorm_loss = false; 
# 1455
static constexpr unsigned long long infinity() noexcept 
# 1456
{ return static_cast< unsigned long long>(0); } 
# 1459
static constexpr unsigned long long quiet_NaN() noexcept 
# 1460
{ return static_cast< unsigned long long>(0); } 
# 1463
static constexpr unsigned long long signaling_NaN() noexcept 
# 1464
{ return static_cast< unsigned long long>(0); } 
# 1467
static constexpr unsigned long long denorm_min() noexcept 
# 1468
{ return static_cast< unsigned long long>(0); } 
# 1470
static constexpr inline bool is_iec559 = false; 
# 1471
static constexpr inline bool is_bounded = true; 
# 1472
static constexpr inline bool is_modulo = true; 
# 1474
static constexpr inline bool traps = true; 
# 1475
static constexpr inline bool tinyness_before = false; 
# 1476
static constexpr inline float_round_style round_style = round_toward_zero; 
# 1478
}; 
# 1635 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/limits" 3
template<> struct numeric_limits< __int128>  { static constexpr inline bool is_specialized = true; static constexpr __int128 min() noexcept { return ((((__int128)(-1)) < (0)) ? (-((((__int128)(-1)) < (0)) ? (((((__int128)1) << ((128 - (((__int128)(-1)) < (0))) - 1)) - (1)) << 1) + (1) : (~((__int128)0)))) - (1) : ((__int128)0)); } static constexpr __int128 max() noexcept { return ((((__int128)(-1)) < (0)) ? (((((__int128)1) << ((128 - (((__int128)(-1)) < (0))) - 1)) - (1)) << 1) + (1) : (~((__int128)0))); } static constexpr inline int digits = (128 - 1); static constexpr inline int digits10 = (((128 - 1) * 643L) / (2136)); static constexpr inline bool is_signed = true; static constexpr inline bool is_integer = true; static constexpr inline bool is_exact = true; static constexpr inline int radix = 2; static constexpr __int128 epsilon() noexcept { return 0; } static constexpr __int128 round_error() noexcept { return 0; } static constexpr __int128 lowest() noexcept { return min(); } static constexpr inline int max_digits10 = 0; static constexpr inline int min_exponent = 0; static constexpr inline int min_exponent10 = 0; static constexpr inline int max_exponent = 0; static constexpr inline int max_exponent10 = 0; static constexpr inline bool has_infinity = false; static constexpr inline bool has_quiet_NaN = false; static constexpr inline bool has_signaling_NaN = false; static constexpr inline float_denorm_style has_denorm = denorm_absent; static constexpr inline bool has_denorm_loss = false; static constexpr __int128 infinity() noexcept { return static_cast< __int128>(0); } static constexpr __int128 quiet_NaN() noexcept { return static_cast< __int128>(0); } static constexpr __int128 signaling_NaN() noexcept { return static_cast< __int128>(0); } static constexpr __int128 denorm_min() noexcept { return static_cast< __int128>(0); } static constexpr inline bool is_iec559 = false; static constexpr inline bool is_bounded = true; static constexpr inline bool is_modulo = false; static constexpr inline bool traps = true; static constexpr inline bool tinyness_before = false; static constexpr inline float_round_style round_style = round_toward_zero; }; template<> struct numeric_limits< unsigned __int128>  { static constexpr inline bool is_specialized = true; static constexpr unsigned __int128 min() noexcept { return 0; } static constexpr unsigned __int128 max() noexcept { return ((((unsigned __int128)(-1)) < (0)) ? (((((unsigned __int128)1) << ((128 - (((unsigned __int128)(-1)) < (0))) - 1)) - (1)) << 1) + (1) : (~((unsigned __int128)0))); } static constexpr unsigned __int128 lowest() noexcept { return min(); } static constexpr inline int max_digits10 = 0; static constexpr inline int digits = 128; static constexpr inline int digits10 = (((128) * 643L) / (2136)); static constexpr inline bool is_signed = false; static constexpr inline bool is_integer = true; static constexpr inline bool is_exact = true; static constexpr inline int radix = 2; static constexpr unsigned __int128 epsilon() noexcept { return 0; } static constexpr unsigned __int128 round_error() noexcept { return 0; } static constexpr inline int min_exponent = 0; static constexpr inline int min_exponent10 = 0; static constexpr inline int max_exponent = 0; static constexpr inline int max_exponent10 = 0; static constexpr inline bool has_infinity = false; static constexpr inline bool has_quiet_NaN = false; static constexpr inline bool has_signaling_NaN = false; static constexpr inline float_denorm_style has_denorm = denorm_absent; static constexpr inline bool has_denorm_loss = false; static constexpr unsigned __int128 infinity() noexcept { return static_cast< unsigned __int128>(0); } static constexpr unsigned __int128 quiet_NaN() noexcept { return static_cast< unsigned __int128>(0); } static constexpr unsigned __int128 signaling_NaN() noexcept { return static_cast< unsigned __int128>(0); } static constexpr unsigned __int128 denorm_min() noexcept { return static_cast< unsigned __int128>(0); } static constexpr inline bool is_iec559 = false; static constexpr inline bool is_bounded = true; static constexpr inline bool is_modulo = true; static constexpr inline bool traps = true; static constexpr inline bool tinyness_before = false; static constexpr inline float_round_style round_style = round_toward_zero; }; 
# 1668 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/limits" 3
template<> struct numeric_limits< float>  { 
# 1670
static constexpr inline bool is_specialized = true; 
# 1673
static constexpr float min() noexcept { return (1.1754944E-38F); } 
# 1676
static constexpr float max() noexcept { return (3.4028235E38F); } 
# 1680
static constexpr float lowest() noexcept { return -(3.4028235E38F); } 
# 1683
static constexpr inline int digits = 24; 
# 1684
static constexpr inline int digits10 = 6; 
# 1686
static constexpr inline int max_digits10 = ((2) + (((24) * 643L) / (2136))); 
# 1689
static constexpr inline bool is_signed = true; 
# 1690
static constexpr inline bool is_integer = false; 
# 1691
static constexpr inline bool is_exact = false; 
# 1692
static constexpr inline int radix = 2; 
# 1695
static constexpr float epsilon() noexcept { return (1.1920929E-7F); } 
# 1698
static constexpr float round_error() noexcept { return (0.5F); } 
# 1700
static constexpr inline int min_exponent = (-125); 
# 1701
static constexpr inline int min_exponent10 = (-37); 
# 1702
static constexpr inline int max_exponent = 128; 
# 1703
static constexpr inline int max_exponent10 = 38; 
# 1705
static constexpr inline bool has_infinity = (1); 
# 1706
static constexpr inline bool has_quiet_NaN = (1); 
# 1707
static constexpr inline bool has_signaling_NaN = has_quiet_NaN; 
# 1708
static constexpr inline float_denorm_style has_denorm = (((bool)1) ? denorm_present : denorm_absent); 
# 1710
static constexpr inline bool has_denorm_loss = false; 
# 1714
static constexpr float infinity() noexcept { return __builtin_huge_valf(); } 
# 1717
static constexpr float quiet_NaN() noexcept { return __builtin_nanf(""); } 
# 1720
static constexpr float signaling_NaN() noexcept { return __builtin_nansf(""); } 
# 1723
static constexpr float denorm_min() noexcept { return (1.4E-45F); } 
# 1725
static constexpr inline bool is_iec559 = (has_infinity && has_quiet_NaN && (has_denorm == (denorm_present))); 
# 1727
static constexpr inline bool is_bounded = true; 
# 1728
static constexpr inline bool is_modulo = false; 
# 1730
static constexpr inline bool traps = false; 
# 1731
static constexpr inline bool tinyness_before = false; 
# 1733
static constexpr inline float_round_style round_style = round_to_nearest; 
# 1735
}; 
# 1743
template<> struct numeric_limits< double>  { 
# 1745
static constexpr inline bool is_specialized = true; 
# 1748
static constexpr double min() noexcept { return (double)(2.2250738585072013831E-308L); } 
# 1751
static constexpr double max() noexcept { return (double)(1.7976931348623157081E308L); } 
# 1755
static constexpr double lowest() noexcept { return -((double)(1.7976931348623157081E308L)); } 
# 1758
static constexpr inline int digits = 53; 
# 1759
static constexpr inline int digits10 = 15; 
# 1761
static constexpr inline int max_digits10 = ((2) + (((53) * 643L) / (2136))); 
# 1764
static constexpr inline bool is_signed = true; 
# 1765
static constexpr inline bool is_integer = false; 
# 1766
static constexpr inline bool is_exact = false; 
# 1767
static constexpr inline int radix = 2; 
# 1770
static constexpr double epsilon() noexcept { return (double)(2.2204460492503130808E-16L); } 
# 1773
static constexpr double round_error() noexcept { return (0.5); } 
# 1775
static constexpr inline int min_exponent = (-1021); 
# 1776
static constexpr inline int min_exponent10 = (-307); 
# 1777
static constexpr inline int max_exponent = 1024; 
# 1778
static constexpr inline int max_exponent10 = 308; 
# 1780
static constexpr inline bool has_infinity = (1); 
# 1781
static constexpr inline bool has_quiet_NaN = (1); 
# 1782
static constexpr inline bool has_signaling_NaN = has_quiet_NaN; 
# 1783
static constexpr inline float_denorm_style has_denorm = (((bool)1) ? denorm_present : denorm_absent); 
# 1785
static constexpr inline bool has_denorm_loss = false; 
# 1789
static constexpr double infinity() noexcept { return __builtin_huge_val(); } 
# 1792
static constexpr double quiet_NaN() noexcept { return __builtin_nan(""); } 
# 1795
static constexpr double signaling_NaN() noexcept { return __builtin_nans(""); } 
# 1798
static constexpr double denorm_min() noexcept { return (double)(4.940656458412465442E-324L); } 
# 1800
static constexpr inline bool is_iec559 = (has_infinity && has_quiet_NaN && (has_denorm == (denorm_present))); 
# 1802
static constexpr inline bool is_bounded = true; 
# 1803
static constexpr inline bool is_modulo = false; 
# 1805
static constexpr inline bool traps = false; 
# 1806
static constexpr inline bool tinyness_before = false; 
# 1808
static constexpr inline float_round_style round_style = round_to_nearest; 
# 1810
}; 
# 1818
template<> struct numeric_limits< long double>  { 
# 1820
static constexpr inline bool is_specialized = true; 
# 1823
static constexpr long double min() noexcept { return (3.3621031431120935063E-4932L); } 
# 1826
static constexpr long double max() noexcept { return (1.189731495357231765E4932L); } 
# 1830
static constexpr long double lowest() noexcept { return -(1.189731495357231765E4932L); } 
# 1833
static constexpr inline int digits = 64; 
# 1834
static constexpr inline int digits10 = 18; 
# 1836
static constexpr inline int max_digits10 = ((2) + (((64) * 643L) / (2136))); 
# 1839
static constexpr inline bool is_signed = true; 
# 1840
static constexpr inline bool is_integer = false; 
# 1841
static constexpr inline bool is_exact = false; 
# 1842
static constexpr inline int radix = 2; 
# 1845
static constexpr long double epsilon() noexcept { return (1.084202172485504434E-19L); } 
# 1848
static constexpr long double round_error() noexcept { return (0.5L); } 
# 1850
static constexpr inline int min_exponent = (-16381); 
# 1851
static constexpr inline int min_exponent10 = (-4931); 
# 1852
static constexpr inline int max_exponent = 16384; 
# 1853
static constexpr inline int max_exponent10 = 4932; 
# 1855
static constexpr inline bool has_infinity = (1); 
# 1856
static constexpr inline bool has_quiet_NaN = (1); 
# 1857
static constexpr inline bool has_signaling_NaN = has_quiet_NaN; 
# 1858
static constexpr inline float_denorm_style has_denorm = (((bool)1) ? denorm_present : denorm_absent); 
# 1860
static constexpr inline bool has_denorm_loss = false; 
# 1864
static constexpr long double infinity() noexcept { return __builtin_huge_vall(); } 
# 1867
static constexpr long double quiet_NaN() noexcept { return __builtin_nanl(""); } 
# 1870
static constexpr long double signaling_NaN() noexcept { return __builtin_nansl(""); } 
# 1873
static constexpr long double denorm_min() noexcept { return (3.6E-4951L); } 
# 1875
static constexpr inline bool is_iec559 = (has_infinity && has_quiet_NaN && (has_denorm == (denorm_present))); 
# 1877
static constexpr inline bool is_bounded = true; 
# 1878
static constexpr inline bool is_modulo = false; 
# 1880
static constexpr inline bool traps = false; 
# 1881
static constexpr inline bool tinyness_before = false; 
# 1883
static constexpr inline float_round_style round_style = round_to_nearest; 
# 1885
}; 
# 1892
}
# 39 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/special_function_util.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 50 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/special_function_util.h" 3
namespace __detail { 
# 55
template< class _Tp> 
# 56
struct __floating_point_constant { 
# 58
static const _Tp __value; 
# 59
}; 
# 63
template< class _Tp> 
# 64
struct __numeric_constants { 
# 67
static _Tp __pi() throw() 
# 68
{ return static_cast< _Tp>((3.1415926535897932385L)); } 
# 70
static _Tp __pi_2() throw() 
# 71
{ return static_cast< _Tp>((1.5707963267948966193L)); } 
# 73
static _Tp __pi_3() throw() 
# 74
{ return static_cast< _Tp>((1.0471975511965977461L)); } 
# 76
static _Tp __pi_4() throw() 
# 77
{ return static_cast< _Tp>((0.78539816339744830963L)); } 
# 79
static _Tp __1_pi() throw() 
# 80
{ return static_cast< _Tp>((0.31830988618379067154L)); } 
# 82
static _Tp __2_sqrtpi() throw() 
# 83
{ return static_cast< _Tp>((1.1283791670955125738L)); } 
# 85
static _Tp __sqrt2() throw() 
# 86
{ return static_cast< _Tp>((1.4142135623730950488L)); } 
# 88
static _Tp __sqrt3() throw() 
# 89
{ return static_cast< _Tp>((1.7320508075688772936L)); } 
# 91
static _Tp __sqrtpio2() throw() 
# 92
{ return static_cast< _Tp>((1.2533141373155002512L)); } 
# 94
static _Tp __sqrt1_2() throw() 
# 95
{ return static_cast< _Tp>((0.7071067811865475244L)); } 
# 97
static _Tp __lnpi() throw() 
# 98
{ return static_cast< _Tp>((1.1447298858494001742L)); } 
# 100
static _Tp __gamma_e() throw() 
# 101
{ return static_cast< _Tp>((0.5772156649015328606L)); } 
# 103
static _Tp __euler() throw() 
# 104
{ return static_cast< _Tp>((2.7182818284590452354L)); } 
# 105
}; 
# 114 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/special_function_util.h" 3
template< class _Tp> inline bool 
# 115
__isnan(_Tp __x) 
# 116
{ return std::isnan(__x); } 
# 133 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/special_function_util.h" 3
}
# 139
}
# 51 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
namespace __detail { 
# 76 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 78
__bernoulli_series(unsigned __n) 
# 79
{ 
# 81
static const _Tp __num[28] = {((_Tp)1UL), ((-((_Tp)1UL)) / ((_Tp)2UL)), (((_Tp)1UL) / ((_Tp)6UL)), ((_Tp)0UL), ((-((_Tp)1UL)) / ((_Tp)30UL)), ((_Tp)0UL), (((_Tp)1UL) / ((_Tp)42UL)), ((_Tp)0UL), ((-((_Tp)1UL)) / ((_Tp)30UL)), ((_Tp)0UL), (((_Tp)5UL) / ((_Tp)66UL)), ((_Tp)0UL), ((-((_Tp)691UL)) / ((_Tp)2730UL)), ((_Tp)0UL), (((_Tp)7UL) / ((_Tp)6UL)), ((_Tp)0UL), ((-((_Tp)3617UL)) / ((_Tp)510UL)), ((_Tp)0UL), (((_Tp)43867UL) / ((_Tp)798UL)), ((_Tp)0UL), ((-((_Tp)174611)) / ((_Tp)330UL)), ((_Tp)0UL), (((_Tp)854513UL) / ((_Tp)138UL)), ((_Tp)0UL), ((-((_Tp)236364091UL)) / ((_Tp)2730UL)), ((_Tp)0UL), (((_Tp)8553103UL) / ((_Tp)6UL)), ((_Tp)0UL)}; 
# 98
if (__n == (0)) { 
# 99
return (_Tp)1; }  
# 101
if (__n == (1)) { 
# 102
return (-((_Tp)1)) / ((_Tp)2); }  
# 105
if ((__n % (2)) == (1)) { 
# 106
return (_Tp)0; }  
# 109
if (__n < (28)) { 
# 110
return __num[__n]; }  
# 113
_Tp __fact = ((_Tp)1); 
# 114
if (((__n / (2)) % (2)) == (0)) { 
# 115
__fact *= ((_Tp)(-1)); }  
# 116
for (unsigned __k = (1); __k <= __n; ++__k) { 
# 117
__fact *= (__k / (((_Tp)2) * __numeric_constants< _Tp> ::__pi())); }  
# 118
__fact *= ((_Tp)2); 
# 120
_Tp __sum = ((_Tp)0); 
# 121
for (unsigned __i = (1); __i < (1000); ++__i) 
# 122
{ 
# 123
_Tp __term = std::pow((_Tp)__i, -((_Tp)__n)); 
# 124
if (__term < std::numeric_limits< _Tp> ::epsilon()) { 
# 125
break; }  
# 126
__sum += __term; 
# 127
}  
# 129
return __fact * __sum; 
# 130
} 
# 139 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> inline _Tp 
# 141
__bernoulli(int __n) 
# 142
{ return __bernoulli_series< _Tp> (__n); } 
# 153 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 155
__log_gamma_bernoulli(_Tp __x) 
# 156
{ 
# 157
_Tp __lg = (((__x - ((_Tp)(0.5L))) * std::log(__x)) - __x) + (((_Tp)(0.5L)) * std::log(((_Tp)2) * __numeric_constants< _Tp> ::__pi())); 
# 161
const _Tp __xx = __x * __x; 
# 162
_Tp __help = ((_Tp)1) / __x; 
# 163
for (unsigned __i = (1); __i < (20); ++__i) 
# 164
{ 
# 165
const _Tp __2i = (_Tp)((2) * __i); 
# 166
__help /= ((__2i * (__2i - ((_Tp)1))) * __xx); 
# 167
__lg += (__bernoulli< _Tp> ((2) * __i) * __help); 
# 168
}  
# 170
return __lg; 
# 171
} 
# 181 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 183
__log_gamma_lanczos(_Tp __x) 
# 184
{ 
# 185
const _Tp __xm1 = __x - ((_Tp)1); 
# 187
static const _Tp __lanczos_cheb_7[9] = {((_Tp)(0.99999999999980993226L)), ((_Tp)(676.52036812188509857L)), ((_Tp)(-(1259.1392167224028704L))), ((_Tp)(771.32342877765307887L)), ((_Tp)(-(176.61502916214059906L))), ((_Tp)(12.507343278686904814L)), ((_Tp)(-(0.1385710952657201169L))), ((_Tp)(9.9843695780195708595E-6L)), ((_Tp)(1.5056327351493115584E-7L))}; 
# 199
static const _Tp __LOGROOT2PI = ((_Tp)(0.9189385332046727418L)); 
# 202
_Tp __sum = (__lanczos_cheb_7[0]); 
# 203
for (unsigned __k = (1); __k < (9); ++__k) { 
# 204
__sum += ((__lanczos_cheb_7[__k]) / (__xm1 + __k)); }  
# 206
const _Tp __term1 = (__xm1 + ((_Tp)(0.5L))) * std::log((__xm1 + ((_Tp)(7.5L))) / __numeric_constants< _Tp> ::__euler()); 
# 209
const _Tp __term2 = __LOGROOT2PI + std::log(__sum); 
# 210
const _Tp __result = __term1 + (__term2 - ((_Tp)7)); 
# 212
return __result; 
# 213
} 
# 225 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 227
__log_gamma(_Tp __x) 
# 228
{ 
# 229
if (__x > ((_Tp)(0.5L))) { 
# 230
return __log_gamma_lanczos(__x); } else 
# 232
{ 
# 233
const _Tp __sin_fact = std::abs(std::sin(__numeric_constants< _Tp> ::__pi() * __x)); 
# 235
if (__sin_fact == ((_Tp)0)) { 
# 236
std::__throw_domain_error("Argument is nonpositive integer in __log_gamma"); }  
# 238
return (__numeric_constants< _Tp> ::__lnpi() - std::log(__sin_fact)) - __log_gamma_lanczos(((_Tp)1) - __x); 
# 241
}  
# 242
} 
# 252 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 254
__log_gamma_sign(_Tp __x) 
# 255
{ 
# 256
if (__x > ((_Tp)0)) { 
# 257
return (_Tp)1; } else 
# 259
{ 
# 260
const _Tp __sin_fact = std::sin(__numeric_constants< _Tp> ::__pi() * __x); 
# 262
if (__sin_fact > ((_Tp)0)) { 
# 263
return 1; } else { 
# 264
if (__sin_fact < ((_Tp)0)) { 
# 265
return -((_Tp)1); } else { 
# 267
return (_Tp)0; }  }  
# 268
}  
# 269
} 
# 283 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 285
__log_bincoef(unsigned __n, unsigned __k) 
# 286
{ 
# 288
static const _Tp __max_bincoeff = (std::numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 292
_Tp __coeff = (std::lgamma((_Tp)((1) + __n)) - std::lgamma((_Tp)((1) + __k))) - std::lgamma((_Tp)(((1) + __n) - __k)); 
# 300
} 
# 314 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 316
__bincoef(unsigned __n, unsigned __k) 
# 317
{ 
# 319
static const _Tp __max_bincoeff = (std::numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 323
const _Tp __log_coeff = __log_bincoef< _Tp> (__n, __k); 
# 324
if (__log_coeff > __max_bincoeff) { 
# 325
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 327
return std::exp(__log_coeff); }  
# 328
} 
# 337 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> inline _Tp 
# 339
__gamma(_Tp __x) 
# 340
{ return std::exp(__log_gamma(__x)); } 
# 356 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 358
__psi_series(_Tp __x) 
# 359
{ 
# 360
_Tp __sum = (-__numeric_constants< _Tp> ::__gamma_e()) - (((_Tp)1) / __x); 
# 361
const unsigned __max_iter = (100000); 
# 362
for (unsigned __k = (1); __k < __max_iter; ++__k) 
# 363
{ 
# 364
const _Tp __term = __x / (__k * (__k + __x)); 
# 365
__sum += __term; 
# 366
if (std::abs(__term / __sum) < std::numeric_limits< _Tp> ::epsilon()) { 
# 367
break; }  
# 368
}  
# 369
return __sum; 
# 370
} 
# 386 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 388
__psi_asymp(_Tp __x) 
# 389
{ 
# 390
_Tp __sum = std::log(__x) - (((_Tp)(0.5L)) / __x); 
# 391
const _Tp __xx = __x * __x; 
# 392
_Tp __xp = __xx; 
# 393
const unsigned __max_iter = (100); 
# 394
for (unsigned __k = (1); __k < __max_iter; ++__k) 
# 395
{ 
# 396
const _Tp __term = __bernoulli< _Tp> ((2) * __k) / (((2) * __k) * __xp); 
# 397
__sum -= __term; 
# 398
if (std::abs(__term / __sum) < std::numeric_limits< _Tp> ::epsilon()) { 
# 399
break; }  
# 400
__xp *= __xx; 
# 401
}  
# 402
return __sum; 
# 403
} 
# 417 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 419
__psi(_Tp __x) 
# 420
{ 
# 421
const int __n = static_cast< int>(__x + (0.5L)); 
# 422
const _Tp __eps = ((_Tp)4) * std::numeric_limits< _Tp> ::epsilon(); 
# 423
if ((__n <= 0) && (std::abs(__x - ((_Tp)__n)) < __eps)) { 
# 424
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 425
if (__x < ((_Tp)0)) 
# 426
{ 
# 427
const _Tp __pi = __numeric_constants< _Tp> ::__pi(); 
# 428
return __psi(((_Tp)1) - __x) - ((__pi * std::cos(__pi * __x)) / std::sin(__pi * __x)); 
# 430
} else { 
# 431
if (__x > ((_Tp)100)) { 
# 432
return __psi_asymp(__x); } else { 
# 434
return __psi_series(__x); }  }  }  
# 435
} 
# 446 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/gamma.tcc" 3
template< class _Tp> _Tp 
# 448
__psi(unsigned __n, _Tp __x) 
# 449
{ 
# 450
if (__x <= ((_Tp)0)) { 
# 451
std::__throw_domain_error("Argument out of range in __psi"); } else { 
# 453
if (__n == (0)) { 
# 454
return __psi(__x); } else 
# 456
{ 
# 457
const _Tp __hzeta = __hurwitz_zeta((_Tp)(__n + (1)), __x); 
# 459
const _Tp __ln_nfact = std::lgamma((_Tp)(__n + (1))); 
# 463
_Tp __result = std::exp(__ln_nfact) * __hzeta; 
# 464
if ((__n % (2)) == (1)) { 
# 465
__result = (-__result); }  
# 466
return __result; 
# 467
}  }  
# 468
} 
# 469
}
# 476
}
# 55 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 71 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
namespace __detail { 
# 98 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
template< class _Tp> void 
# 100
__gamma_temme(_Tp __mu, _Tp &
# 101
__gam1, _Tp &__gam2, _Tp &__gampl, _Tp &__gammi) 
# 102
{ 
# 104
__gampl = (((_Tp)1) / std::tgamma(((_Tp)1) + __mu)); 
# 105
__gammi = (((_Tp)1) / std::tgamma(((_Tp)1) - __mu)); 
# 111
if (std::abs(__mu) < std::numeric_limits< _Tp> ::epsilon()) { 
# 112
__gam1 = (-((_Tp)__numeric_constants< _Tp> ::__gamma_e())); } else { 
# 114
__gam1 = ((__gammi - __gampl) / (((_Tp)2) * __mu)); }  
# 116
__gam2 = ((__gammi + __gampl) / ((_Tp)2)); 
# 119
} 
# 136 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
template< class _Tp> void 
# 138
__bessel_jn(_Tp __nu, _Tp __x, _Tp &
# 139
__Jnu, _Tp &__Nnu, _Tp &__Jpnu, _Tp &__Npnu) 
# 140
{ 
# 141
if (__x == ((_Tp)0)) 
# 142
{ 
# 143
if (__nu == ((_Tp)0)) 
# 144
{ 
# 145
__Jnu = ((_Tp)1); 
# 146
__Jpnu = ((_Tp)0); 
# 147
} else { 
# 148
if (__nu == ((_Tp)1)) 
# 149
{ 
# 150
__Jnu = ((_Tp)0); 
# 151
__Jpnu = ((_Tp)(0.5L)); 
# 152
} else 
# 154
{ 
# 155
__Jnu = ((_Tp)0); 
# 156
__Jpnu = ((_Tp)0); 
# 157
}  }  
# 158
__Nnu = (-std::numeric_limits< _Tp> ::infinity()); 
# 159
__Npnu = std::numeric_limits< _Tp> ::infinity(); 
# 160
return; 
# 161
}  
# 163
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 168
const _Tp __fp_min = std::sqrt(std::numeric_limits< _Tp> ::min()); 
# 169
const int __max_iter = 15000; 
# 170
const _Tp __x_min = ((_Tp)2); 
# 172
const int __nl = (__x < __x_min) ? static_cast< int>(__nu + ((_Tp)(0.5L))) : std::max(0, static_cast< int>((__nu - __x) + ((_Tp)(1.5L)))); 
# 176
const _Tp __mu = __nu - __nl; 
# 177
const _Tp __mu2 = __mu * __mu; 
# 178
const _Tp __xi = ((_Tp)1) / __x; 
# 179
const _Tp __xi2 = ((_Tp)2) * __xi; 
# 180
_Tp __w = __xi2 / __numeric_constants< _Tp> ::__pi(); 
# 181
int __isign = 1; 
# 182
_Tp __h = __nu * __xi; 
# 183
if (__h < __fp_min) { 
# 184
__h = __fp_min; }  
# 185
_Tp __b = __xi2 * __nu; 
# 186
_Tp __d = ((_Tp)0); 
# 187
_Tp __c = __h; 
# 188
int __i; 
# 189
for (__i = 1; __i <= __max_iter; ++__i) 
# 190
{ 
# 191
__b += __xi2; 
# 192
__d = (__b - __d); 
# 193
if (std::abs(__d) < __fp_min) { 
# 194
__d = __fp_min; }  
# 195
__c = (__b - (((_Tp)1) / __c)); 
# 196
if (std::abs(__c) < __fp_min) { 
# 197
__c = __fp_min; }  
# 198
__d = (((_Tp)1) / __d); 
# 199
const _Tp __del = __c * __d; 
# 200
__h *= __del; 
# 201
if (__d < ((_Tp)0)) { 
# 202
__isign = (-__isign); }  
# 203
if (std::abs(__del - ((_Tp)1)) < __eps) { 
# 204
break; }  
# 205
}  
# 206
if (__i > __max_iter) { 
# 207
std::__throw_runtime_error("Argument x too large in __bessel_jn; try asymptotic expansion."); }  
# 209
_Tp __Jnul = __isign * __fp_min; 
# 210
_Tp __Jpnul = __h * __Jnul; 
# 211
_Tp __Jnul1 = __Jnul; 
# 212
_Tp __Jpnu1 = __Jpnul; 
# 213
_Tp __fact = __nu * __xi; 
# 214
for (int __l = __nl; __l >= 1; --__l) 
# 215
{ 
# 216
const _Tp __Jnutemp = (__fact * __Jnul) + __Jpnul; 
# 217
__fact -= __xi; 
# 218
__Jpnul = ((__fact * __Jnutemp) - __Jnul); 
# 219
__Jnul = __Jnutemp; 
# 220
}  
# 221
if (__Jnul == ((_Tp)0)) { 
# 222
__Jnul = __eps; }  
# 223
_Tp __f = __Jpnul / __Jnul; 
# 224
_Tp __Nmu, __Nnu1, __Npmu, __Jmu; 
# 225
if (__x < __x_min) 
# 226
{ 
# 227
const _Tp __x2 = __x / ((_Tp)2); 
# 228
const _Tp __pimu = __numeric_constants< _Tp> ::__pi() * __mu; 
# 229
_Tp __fact = (std::abs(__pimu) < __eps) ? (_Tp)1 : (__pimu / std::sin(__pimu)); 
# 231
_Tp __d = (-std::log(__x2)); 
# 232
_Tp __e = __mu * __d; 
# 233
_Tp __fact2 = (std::abs(__e) < __eps) ? (_Tp)1 : (std::sinh(__e) / __e); 
# 235
_Tp __gam1, __gam2, __gampl, __gammi; 
# 236
__gamma_temme(__mu, __gam1, __gam2, __gampl, __gammi); 
# 237
_Tp __ff = ((((_Tp)2) / __numeric_constants< _Tp> ::__pi()) * __fact) * ((__gam1 * std::cosh(__e)) + ((__gam2 * __fact2) * __d)); 
# 239
__e = std::exp(__e); 
# 240
_Tp __p = __e / (__numeric_constants< _Tp> ::__pi() * __gampl); 
# 241
_Tp __q = ((_Tp)1) / ((__e * __numeric_constants< _Tp> ::__pi()) * __gammi); 
# 242
const _Tp __pimu2 = __pimu / ((_Tp)2); 
# 243
_Tp __fact3 = (std::abs(__pimu2) < __eps) ? (_Tp)1 : (std::sin(__pimu2) / __pimu2); 
# 245
_Tp __r = ((__numeric_constants< _Tp> ::__pi() * __pimu2) * __fact3) * __fact3; 
# 246
_Tp __c = ((_Tp)1); 
# 247
__d = ((-__x2) * __x2); 
# 248
_Tp __sum = __ff + (__r * __q); 
# 249
_Tp __sum1 = __p; 
# 250
for (__i = 1; __i <= __max_iter; ++__i) 
# 251
{ 
# 252
__ff = ((((__i * __ff) + __p) + __q) / ((__i * __i) - __mu2)); 
# 253
__c *= (__d / ((_Tp)__i)); 
# 254
__p /= (((_Tp)__i) - __mu); 
# 255
__q /= (((_Tp)__i) + __mu); 
# 256
const _Tp __del = __c * (__ff + (__r * __q)); 
# 257
__sum += __del; 
# 258
const _Tp __del1 = (__c * __p) - (__i * __del); 
# 259
__sum1 += __del1; 
# 260
if (std::abs(__del) < (__eps * (((_Tp)1) + std::abs(__sum)))) { 
# 261
break; }  
# 262
}  
# 263
if (__i > __max_iter) { 
# 264
std::__throw_runtime_error("Bessel y series failed to converge in __bessel_jn."); }  
# 266
__Nmu = (-__sum); 
# 267
__Nnu1 = ((-__sum1) * __xi2); 
# 268
__Npmu = (((__mu * __xi) * __Nmu) - __Nnu1); 
# 269
__Jmu = (__w / (__Npmu - (__f * __Nmu))); 
# 270
} else 
# 272
{ 
# 273
_Tp __a = ((_Tp)(0.25L)) - __mu2; 
# 274
_Tp __q = ((_Tp)1); 
# 275
_Tp __p = ((-__xi) / ((_Tp)2)); 
# 276
_Tp __br = ((_Tp)2) * __x; 
# 277
_Tp __bi = ((_Tp)2); 
# 278
_Tp __fact = (__a * __xi) / ((__p * __p) + (__q * __q)); 
# 279
_Tp __cr = __br + (__q * __fact); 
# 280
_Tp __ci = __bi + (__p * __fact); 
# 281
_Tp __den = (__br * __br) + (__bi * __bi); 
# 282
_Tp __dr = __br / __den; 
# 283
_Tp __di = (-__bi) / __den; 
# 284
_Tp __dlr = (__cr * __dr) - (__ci * __di); 
# 285
_Tp __dli = (__cr * __di) + (__ci * __dr); 
# 286
_Tp __temp = (__p * __dlr) - (__q * __dli); 
# 287
__q = ((__p * __dli) + (__q * __dlr)); 
# 288
__p = __temp; 
# 289
int __i; 
# 290
for (__i = 2; __i <= __max_iter; ++__i) 
# 291
{ 
# 292
__a += ((_Tp)(2 * (__i - 1))); 
# 293
__bi += ((_Tp)2); 
# 294
__dr = ((__a * __dr) + __br); 
# 295
__di = ((__a * __di) + __bi); 
# 296
if ((std::abs(__dr) + std::abs(__di)) < __fp_min) { 
# 297
__dr = __fp_min; }  
# 298
__fact = (__a / ((__cr * __cr) + (__ci * __ci))); 
# 299
__cr = (__br + (__cr * __fact)); 
# 300
__ci = (__bi - (__ci * __fact)); 
# 301
if ((std::abs(__cr) + std::abs(__ci)) < __fp_min) { 
# 302
__cr = __fp_min; }  
# 303
__den = ((__dr * __dr) + (__di * __di)); 
# 304
__dr /= __den; 
# 305
__di /= (-__den); 
# 306
__dlr = ((__cr * __dr) - (__ci * __di)); 
# 307
__dli = ((__cr * __di) + (__ci * __dr)); 
# 308
__temp = ((__p * __dlr) - (__q * __dli)); 
# 309
__q = ((__p * __dli) + (__q * __dlr)); 
# 310
__p = __temp; 
# 311
if ((std::abs(__dlr - ((_Tp)1)) + std::abs(__dli)) < __eps) { 
# 312
break; }  
# 313
}  
# 314
if (__i > __max_iter) { 
# 315
std::__throw_runtime_error("Lentz\'s method failed in __bessel_jn."); }  
# 317
const _Tp __gam = (__p - __f) / __q; 
# 318
__Jmu = std::sqrt(__w / (((__p - __f) * __gam) + __q)); 
# 320
__Jmu = std::copysign(__Jmu, __Jnul); 
# 325
__Nmu = (__gam * __Jmu); 
# 326
__Npmu = ((__p + (__q / __gam)) * __Nmu); 
# 327
__Nnu1 = (((__mu * __xi) * __Nmu) - __Npmu); 
# 328
}  
# 329
__fact = (__Jmu / __Jnul); 
# 330
__Jnu = (__fact * __Jnul1); 
# 331
__Jpnu = (__fact * __Jpnu1); 
# 332
for (__i = 1; __i <= __nl; ++__i) 
# 333
{ 
# 334
const _Tp __Nnutemp = (((__mu + __i) * __xi2) * __Nnu1) - __Nmu; 
# 335
__Nmu = __Nnu1; 
# 336
__Nnu1 = __Nnutemp; 
# 337
}  
# 338
__Nnu = __Nmu; 
# 339
__Npnu = (((__nu * __xi) * __Nmu) - __Nnu1); 
# 342
} 
# 361 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
template< class _Tp> void 
# 363
__cyl_bessel_jn_asymp(_Tp __nu, _Tp __x, _Tp &__Jnu, _Tp &__Nnu) 
# 364
{ 
# 365
const _Tp __mu = (((_Tp)4) * __nu) * __nu; 
# 366
const _Tp __8x = ((_Tp)8) * __x; 
# 368
_Tp __P = ((_Tp)0); 
# 369
_Tp __Q = ((_Tp)0); 
# 371
_Tp __k = ((_Tp)0); 
# 372
_Tp __term = ((_Tp)1); 
# 374
int __epsP = 0; 
# 375
int __epsQ = 0; 
# 377
_Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 379
do 
# 380
{ 
# 381
__term *= ((__k == 0) ? (_Tp)1 : ((-(__mu - (((2 * __k) - 1) * ((2 * __k) - 1)))) / (__k * __8x))); 
# 385
__epsP = (std::abs(__term) < (__eps * std::abs(__P))); 
# 386
__P += __term; 
# 388
__k++; 
# 390
__term *= ((__mu - (((2 * __k) - 1) * ((2 * __k) - 1))) / (__k * __8x)); 
# 391
__epsQ = (std::abs(__term) < (__eps * std::abs(__Q))); 
# 392
__Q += __term; 
# 394
if (__epsP && __epsQ && (__k > (__nu / (2.0)))) { 
# 395
break; }  
# 397
__k++; 
# 398
} 
# 399
while (__k < 1000); 
# 401
const _Tp __chi = __x - ((__nu + ((_Tp)(0.5L))) * __numeric_constants< _Tp> ::__pi_2()); 
# 404
const _Tp __c = std::cos(__chi); 
# 405
const _Tp __s = std::sin(__chi); 
# 407
const _Tp __coef = std::sqrt(((_Tp)2) / (__numeric_constants< _Tp> ::__pi() * __x)); 
# 410
__Jnu = (__coef * ((__c * __P) - (__s * __Q))); 
# 411
__Nnu = (__coef * ((__s * __P) + (__c * __Q))); 
# 414
} 
# 444 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
template< class _Tp> _Tp 
# 446
__cyl_bessel_ij_series(_Tp __nu, _Tp __x, _Tp __sgn, unsigned 
# 447
__max_iter) 
# 448
{ 
# 449
if (__x == ((_Tp)0)) { 
# 450
return (__nu == ((_Tp)0)) ? (_Tp)1 : ((_Tp)0); }  
# 452
const _Tp __x2 = __x / ((_Tp)2); 
# 453
_Tp __fact = __nu * std::log(__x2); 
# 455
__fact -= std::lgamma(__nu + ((_Tp)1)); 
# 459
__fact = std::exp(__fact); 
# 460
const _Tp __xx4 = (__sgn * __x2) * __x2; 
# 461
_Tp __Jn = ((_Tp)1); 
# 462
_Tp __term = ((_Tp)1); 
# 464
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 465
{ 
# 466
__term *= (__xx4 / (((_Tp)__i) * (__nu + ((_Tp)__i)))); 
# 467
__Jn += __term; 
# 468
if (std::abs(__term / __Jn) < std::numeric_limits< _Tp> ::epsilon()) { 
# 469
break; }  
# 470
}  
# 472
return __fact * __Jn; 
# 473
} 
# 490 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
template< class _Tp> _Tp 
# 492
__cyl_bessel_j(_Tp __nu, _Tp __x) 
# 493
{ 
# 494
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 495
std::__throw_domain_error("Bad argument in __cyl_bessel_j."); } else { 
# 497
if (__isnan(__nu) || __isnan(__x)) { 
# 498
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 499
if ((__x * __x) < (((_Tp)10) * (__nu + ((_Tp)1)))) { 
# 500
return __cyl_bessel_ij_series(__nu, __x, -((_Tp)1), 200); } else { 
# 501
if (__x > ((_Tp)1000)) 
# 502
{ 
# 503
_Tp __J_nu, __N_nu; 
# 504
__cyl_bessel_jn_asymp(__nu, __x, __J_nu, __N_nu); 
# 505
return __J_nu; 
# 506
} else 
# 508
{ 
# 509
_Tp __J_nu, __N_nu, __Jp_nu, __Np_nu; 
# 510
__bessel_jn(__nu, __x, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 511
return __J_nu; 
# 512
}  }  }  }  
# 513
} 
# 532 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
template< class _Tp> _Tp 
# 534
__cyl_neumann_n(_Tp __nu, _Tp __x) 
# 535
{ 
# 536
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 537
std::__throw_domain_error("Bad argument in __cyl_neumann_n."); } else { 
# 539
if (__isnan(__nu) || __isnan(__x)) { 
# 540
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 541
if (__x > ((_Tp)1000)) 
# 542
{ 
# 543
_Tp __J_nu, __N_nu; 
# 544
__cyl_bessel_jn_asymp(__nu, __x, __J_nu, __N_nu); 
# 545
return __N_nu; 
# 546
} else 
# 548
{ 
# 549
_Tp __J_nu, __N_nu, __Jp_nu, __Np_nu; 
# 550
__bessel_jn(__nu, __x, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 551
return __N_nu; 
# 552
}  }  }  
# 553
} 
# 569 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
template< class _Tp> void 
# 571
__sph_bessel_jn(unsigned __n, _Tp __x, _Tp &
# 572
__j_n, _Tp &__n_n, _Tp &__jp_n, _Tp &__np_n) 
# 573
{ 
# 574
const _Tp __nu = ((_Tp)__n) + ((_Tp)(0.5L)); 
# 576
_Tp __J_nu, __N_nu, __Jp_nu, __Np_nu; 
# 577
__bessel_jn(__nu, __x, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 579
const _Tp __factor = __numeric_constants< _Tp> ::__sqrtpio2() / std::sqrt(__x); 
# 582
__j_n = (__factor * __J_nu); 
# 583
__n_n = (__factor * __N_nu); 
# 584
__jp_n = ((__factor * __Jp_nu) - (__j_n / (((_Tp)2) * __x))); 
# 585
__np_n = ((__factor * __Np_nu) - (__n_n / (((_Tp)2) * __x))); 
# 588
} 
# 604 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
template< class _Tp> _Tp 
# 606
__sph_bessel(unsigned __n, _Tp __x) 
# 607
{ 
# 608
if (__x < ((_Tp)0)) { 
# 609
std::__throw_domain_error("Bad argument in __sph_bessel."); } else { 
# 611
if (__isnan(__x)) { 
# 612
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 613
if (__x == ((_Tp)0)) 
# 614
{ 
# 615
if (__n == (0)) { 
# 616
return (_Tp)1; } else { 
# 618
return (_Tp)0; }  
# 619
} else 
# 621
{ 
# 622
_Tp __j_n, __n_n, __jp_n, __np_n; 
# 623
__sph_bessel_jn(__n, __x, __j_n, __n_n, __jp_n, __np_n); 
# 624
return __j_n; 
# 625
}  }  }  
# 626
} 
# 642 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/bessel_function.tcc" 3
template< class _Tp> _Tp 
# 644
__sph_neumann(unsigned __n, _Tp __x) 
# 645
{ 
# 646
if (__x < ((_Tp)0)) { 
# 647
std::__throw_domain_error("Bad argument in __sph_neumann."); } else { 
# 649
if (__isnan(__x)) { 
# 650
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 651
if (__x == ((_Tp)0)) { 
# 652
return -std::numeric_limits< _Tp> ::infinity(); } else 
# 654
{ 
# 655
_Tp __j_n, __n_n, __jp_n, __np_n; 
# 656
__sph_bessel_jn(__n, __x, __j_n, __n_n, __jp_n, __np_n); 
# 657
return __n_n; 
# 658
}  }  }  
# 659
} 
# 660
}
# 667
}
# 49 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/beta_function.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/beta_function.tcc" 3
namespace __detail { 
# 79 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/beta_function.tcc" 3
template< class _Tp> _Tp 
# 81
__beta_gamma(_Tp __x, _Tp __y) 
# 82
{ 
# 84
_Tp __bet; 
# 86
if (__x > __y) 
# 87
{ 
# 88
__bet = (std::tgamma(__x) / std::tgamma(__x + __y)); 
# 90
__bet *= std::tgamma(__y); 
# 91
} else 
# 93
{ 
# 94
__bet = (std::tgamma(__y) / std::tgamma(__x + __y)); 
# 96
__bet *= std::tgamma(__x); 
# 97
}  
# 111 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/beta_function.tcc" 3
return __bet; 
# 112
} 
# 127 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/beta_function.tcc" 3
template< class _Tp> _Tp 
# 129
__beta_lgamma(_Tp __x, _Tp __y) 
# 130
{ 
# 132
_Tp __bet = (std::lgamma(__x) + std::lgamma(__y)) - std::lgamma(__x + __y); 
# 140
__bet = std::exp(__bet); 
# 141
return __bet; 
# 142
} 
# 158 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/beta_function.tcc" 3
template< class _Tp> _Tp 
# 160
__beta_product(_Tp __x, _Tp __y) 
# 161
{ 
# 163
_Tp __bet = (__x + __y) / (__x * __y); 
# 165
unsigned __max_iter = (1000000); 
# 166
for (unsigned __k = (1); __k < __max_iter; ++__k) 
# 167
{ 
# 168
_Tp __term = (((_Tp)1) + ((__x + __y) / __k)) / ((((_Tp)1) + (__x / __k)) * (((_Tp)1) + (__y / __k))); 
# 170
__bet *= __term; 
# 171
}  
# 173
return __bet; 
# 174
} 
# 189 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/beta_function.tcc" 3
template< class _Tp> inline _Tp 
# 191
__beta(_Tp __x, _Tp __y) 
# 192
{ 
# 193
if (__isnan(__x) || __isnan(__y)) { 
# 194
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 196
return __beta_lgamma(__x, __y); }  
# 197
} 
# 198
}
# 205
}
# 45 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 59 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
namespace __detail { 
# 76 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 78
__ellint_rf(_Tp __x, _Tp __y, _Tp __z) 
# 79
{ 
# 80
const _Tp __min = std::numeric_limits< _Tp> ::min(); 
# 81
const _Tp __lolim = ((_Tp)5) * __min; 
# 83
if (((__x < ((_Tp)0)) || (__y < ((_Tp)0))) || (__z < ((_Tp)0))) { 
# 84
std::__throw_domain_error("Argument less than zero in __ellint_rf."); } else { 
# 86
if ((((__x + __y) < __lolim) || ((__x + __z) < __lolim)) || ((__y + __z) < __lolim)) { 
# 88
std::__throw_domain_error("Argument too small in __ellint_rf"); } else 
# 90
{ 
# 91
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 92
const _Tp __c1 = (((_Tp)1) / ((_Tp)24)); 
# 93
const _Tp __c2 = (((_Tp)1) / ((_Tp)10)); 
# 94
const _Tp __c3 = (((_Tp)3) / ((_Tp)44)); 
# 95
const _Tp __c4 = (((_Tp)1) / ((_Tp)14)); 
# 97
_Tp __xn = __x; 
# 98
_Tp __yn = __y; 
# 99
_Tp __zn = __z; 
# 101
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 102
const _Tp __errtol = std::pow(__eps, ((_Tp)1) / ((_Tp)6)); 
# 103
_Tp __mu; 
# 104
_Tp __xndev, __yndev, __zndev; 
# 106
const unsigned __max_iter = (100); 
# 107
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 108
{ 
# 109
__mu = (((__xn + __yn) + __zn) / ((_Tp)3)); 
# 110
__xndev = (2 - ((__mu + __xn) / __mu)); 
# 111
__yndev = (2 - ((__mu + __yn) / __mu)); 
# 112
__zndev = (2 - ((__mu + __zn) / __mu)); 
# 113
_Tp __epsilon = std::max(std::abs(__xndev), std::abs(__yndev)); 
# 114
__epsilon = std::max(__epsilon, std::abs(__zndev)); 
# 115
if (__epsilon < __errtol) { 
# 116
break; }  
# 117
const _Tp __xnroot = std::sqrt(__xn); 
# 118
const _Tp __ynroot = std::sqrt(__yn); 
# 119
const _Tp __znroot = std::sqrt(__zn); 
# 120
const _Tp __lambda = (__xnroot * (__ynroot + __znroot)) + (__ynroot * __znroot); 
# 122
__xn = (__c0 * (__xn + __lambda)); 
# 123
__yn = (__c0 * (__yn + __lambda)); 
# 124
__zn = (__c0 * (__zn + __lambda)); 
# 125
}  
# 127
const _Tp __e2 = (__xndev * __yndev) - (__zndev * __zndev); 
# 128
const _Tp __e3 = (__xndev * __yndev) * __zndev; 
# 129
const _Tp __s = (((_Tp)1) + ((((__c1 * __e2) - __c2) - (__c3 * __e3)) * __e2)) + (__c4 * __e3); 
# 132
return __s / std::sqrt(__mu); 
# 133
}  }  
# 134
} 
# 153 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 155
__comp_ellint_1_series(_Tp __k) 
# 156
{ 
# 158
const _Tp __kk = __k * __k; 
# 160
_Tp __term = __kk / ((_Tp)4); 
# 161
_Tp __sum = ((_Tp)1) + __term; 
# 163
const unsigned __max_iter = (1000); 
# 164
for (unsigned __i = (2); __i < __max_iter; ++__i) 
# 165
{ 
# 166
__term *= (((((2) * __i) - (1)) * __kk) / ((2) * __i)); 
# 167
if (__term < std::numeric_limits< _Tp> ::epsilon()) { 
# 168
break; }  
# 169
__sum += __term; 
# 170
}  
# 172
return __numeric_constants< _Tp> ::__pi_2() * __sum; 
# 173
} 
# 191 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 193
__comp_ellint_1(_Tp __k) 
# 194
{ 
# 196
if (__isnan(__k)) { 
# 197
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 198
if (std::abs(__k) >= ((_Tp)1)) { 
# 199
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 201
return __ellint_rf((_Tp)0, ((_Tp)1) - (__k * __k), (_Tp)1); }  }  
# 202
} 
# 219 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 221
__ellint_1(_Tp __k, _Tp __phi) 
# 222
{ 
# 224
if (__isnan(__k) || __isnan(__phi)) { 
# 225
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 226
if (std::abs(__k) > ((_Tp)1)) { 
# 227
std::__throw_domain_error("Bad argument in __ellint_1."); } else 
# 229
{ 
# 231
const int __n = std::floor((__phi / __numeric_constants< _Tp> ::__pi()) + ((_Tp)(0.5L))); 
# 233
const _Tp __phi_red = __phi - (__n * __numeric_constants< _Tp> ::__pi()); 
# 236
const _Tp __s = std::sin(__phi_red); 
# 237
const _Tp __c = std::cos(__phi_red); 
# 239
const _Tp __F = __s * __ellint_rf(__c * __c, ((_Tp)1) - (((__k * __k) * __s) * __s), (_Tp)1); 
# 243
if (__n == 0) { 
# 244
return __F; } else { 
# 246
return __F + ((((_Tp)2) * __n) * __comp_ellint_1(__k)); }  
# 247
}  }  
# 248
} 
# 266 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 268
__comp_ellint_2_series(_Tp __k) 
# 269
{ 
# 271
const _Tp __kk = __k * __k; 
# 273
_Tp __term = __kk; 
# 274
_Tp __sum = __term; 
# 276
const unsigned __max_iter = (1000); 
# 277
for (unsigned __i = (2); __i < __max_iter; ++__i) 
# 278
{ 
# 279
const _Tp __i2m = ((2) * __i) - (1); 
# 280
const _Tp __i2 = (2) * __i; 
# 281
__term *= (((__i2m * __i2m) * __kk) / (__i2 * __i2)); 
# 282
if (__term < std::numeric_limits< _Tp> ::epsilon()) { 
# 283
break; }  
# 284
__sum += (__term / __i2m); 
# 285
}  
# 287
return __numeric_constants< _Tp> ::__pi_2() * (((_Tp)1) - __sum); 
# 288
} 
# 314 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 316
__ellint_rd(_Tp __x, _Tp __y, _Tp __z) 
# 317
{ 
# 318
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 319
const _Tp __errtol = std::pow(__eps / ((_Tp)8), ((_Tp)1) / ((_Tp)6)); 
# 320
const _Tp __max = std::numeric_limits< _Tp> ::max(); 
# 321
const _Tp __lolim = (((_Tp)2) / std::pow(__max, ((_Tp)2) / ((_Tp)3))); 
# 323
if ((__x < ((_Tp)0)) || (__y < ((_Tp)0))) { 
# 324
std::__throw_domain_error("Argument less than zero in __ellint_rd."); } else { 
# 326
if (((__x + __y) < __lolim) || (__z < __lolim)) { 
# 327
std::__throw_domain_error("Argument too small in __ellint_rd."); } else 
# 330
{ 
# 331
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 332
const _Tp __c1 = (((_Tp)3) / ((_Tp)14)); 
# 333
const _Tp __c2 = (((_Tp)1) / ((_Tp)6)); 
# 334
const _Tp __c3 = (((_Tp)9) / ((_Tp)22)); 
# 335
const _Tp __c4 = (((_Tp)3) / ((_Tp)26)); 
# 337
_Tp __xn = __x; 
# 338
_Tp __yn = __y; 
# 339
_Tp __zn = __z; 
# 340
_Tp __sigma = ((_Tp)0); 
# 341
_Tp __power4 = ((_Tp)1); 
# 343
_Tp __mu; 
# 344
_Tp __xndev, __yndev, __zndev; 
# 346
const unsigned __max_iter = (100); 
# 347
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 348
{ 
# 349
__mu = (((__xn + __yn) + (((_Tp)3) * __zn)) / ((_Tp)5)); 
# 350
__xndev = ((__mu - __xn) / __mu); 
# 351
__yndev = ((__mu - __yn) / __mu); 
# 352
__zndev = ((__mu - __zn) / __mu); 
# 353
_Tp __epsilon = std::max(std::abs(__xndev), std::abs(__yndev)); 
# 354
__epsilon = std::max(__epsilon, std::abs(__zndev)); 
# 355
if (__epsilon < __errtol) { 
# 356
break; }  
# 357
_Tp __xnroot = std::sqrt(__xn); 
# 358
_Tp __ynroot = std::sqrt(__yn); 
# 359
_Tp __znroot = std::sqrt(__zn); 
# 360
_Tp __lambda = (__xnroot * (__ynroot + __znroot)) + (__ynroot * __znroot); 
# 362
__sigma += (__power4 / (__znroot * (__zn + __lambda))); 
# 363
__power4 *= __c0; 
# 364
__xn = (__c0 * (__xn + __lambda)); 
# 365
__yn = (__c0 * (__yn + __lambda)); 
# 366
__zn = (__c0 * (__zn + __lambda)); 
# 367
}  
# 369
_Tp __ea = __xndev * __yndev; 
# 370
_Tp __eb = __zndev * __zndev; 
# 371
_Tp __ec = __ea - __eb; 
# 372
_Tp __ed = __ea - (((_Tp)6) * __eb); 
# 373
_Tp __ef = (__ed + __ec) + __ec; 
# 374
_Tp __s1 = __ed * (((-__c1) + ((__c3 * __ed) / ((_Tp)3))) - ((((((_Tp)3) * __c4) * __zndev) * __ef) / ((_Tp)2))); 
# 377
_Tp __s2 = __zndev * ((__c2 * __ef) + (__zndev * ((((-__c3) * __ec) - (__zndev * __c4)) - __ea))); 
# 381
return (((_Tp)3) * __sigma) + ((__power4 * ((((_Tp)1) + __s1) + __s2)) / (__mu * std::sqrt(__mu))); 
# 383
}  }  
# 384
} 
# 399 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 401
__comp_ellint_2(_Tp __k) 
# 402
{ 
# 404
if (__isnan(__k)) { 
# 405
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 406
if (std::abs(__k) == 1) { 
# 407
return (_Tp)1; } else { 
# 408
if (std::abs(__k) > ((_Tp)1)) { 
# 409
std::__throw_domain_error("Bad argument in __comp_ellint_2."); } else 
# 411
{ 
# 412
const _Tp __kk = __k * __k; 
# 414
return __ellint_rf((_Tp)0, ((_Tp)1) - __kk, (_Tp)1) - ((__kk * __ellint_rd((_Tp)0, ((_Tp)1) - __kk, (_Tp)1)) / ((_Tp)3)); 
# 416
}  }  }  
# 417
} 
# 433 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 435
__ellint_2(_Tp __k, _Tp __phi) 
# 436
{ 
# 438
if (__isnan(__k) || __isnan(__phi)) { 
# 439
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 440
if (std::abs(__k) > ((_Tp)1)) { 
# 441
std::__throw_domain_error("Bad argument in __ellint_2."); } else 
# 443
{ 
# 445
const int __n = std::floor((__phi / __numeric_constants< _Tp> ::__pi()) + ((_Tp)(0.5L))); 
# 447
const _Tp __phi_red = __phi - (__n * __numeric_constants< _Tp> ::__pi()); 
# 450
const _Tp __kk = __k * __k; 
# 451
const _Tp __s = std::sin(__phi_red); 
# 452
const _Tp __ss = __s * __s; 
# 453
const _Tp __sss = __ss * __s; 
# 454
const _Tp __c = std::cos(__phi_red); 
# 455
const _Tp __cc = __c * __c; 
# 457
const _Tp __E = (__s * __ellint_rf(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1)) - (((__kk * __sss) * __ellint_rd(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1)) / ((_Tp)3)); 
# 463
if (__n == 0) { 
# 464
return __E; } else { 
# 466
return __E + ((((_Tp)2) * __n) * __comp_ellint_2(__k)); }  
# 467
}  }  
# 468
} 
# 492 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 494
__ellint_rc(_Tp __x, _Tp __y) 
# 495
{ 
# 496
const _Tp __min = std::numeric_limits< _Tp> ::min(); 
# 497
const _Tp __lolim = ((_Tp)5) * __min; 
# 499
if (((__x < ((_Tp)0)) || (__y < ((_Tp)0))) || ((__x + __y) < __lolim)) { 
# 500
std::__throw_domain_error("Argument less than zero in __ellint_rc."); } else 
# 503
{ 
# 504
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 505
const _Tp __c1 = (((_Tp)1) / ((_Tp)7)); 
# 506
const _Tp __c2 = (((_Tp)9) / ((_Tp)22)); 
# 507
const _Tp __c3 = (((_Tp)3) / ((_Tp)10)); 
# 508
const _Tp __c4 = (((_Tp)3) / ((_Tp)8)); 
# 510
_Tp __xn = __x; 
# 511
_Tp __yn = __y; 
# 513
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 514
const _Tp __errtol = std::pow(__eps / ((_Tp)30), ((_Tp)1) / ((_Tp)6)); 
# 515
_Tp __mu; 
# 516
_Tp __sn; 
# 518
const unsigned __max_iter = (100); 
# 519
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 520
{ 
# 521
__mu = ((__xn + (((_Tp)2) * __yn)) / ((_Tp)3)); 
# 522
__sn = (((__yn + __mu) / __mu) - ((_Tp)2)); 
# 523
if (std::abs(__sn) < __errtol) { 
# 524
break; }  
# 525
const _Tp __lambda = ((((_Tp)2) * std::sqrt(__xn)) * std::sqrt(__yn)) + __yn; 
# 527
__xn = (__c0 * (__xn + __lambda)); 
# 528
__yn = (__c0 * (__yn + __lambda)); 
# 529
}  
# 531
_Tp __s = (__sn * __sn) * (__c3 + (__sn * (__c1 + (__sn * (__c4 + (__sn * __c2)))))); 
# 534
return (((_Tp)1) + __s) / std::sqrt(__mu); 
# 535
}  
# 536
} 
# 561 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 563
__ellint_rj(_Tp __x, _Tp __y, _Tp __z, _Tp __p) 
# 564
{ 
# 565
const _Tp __min = std::numeric_limits< _Tp> ::min(); 
# 566
const _Tp __lolim = std::pow(((_Tp)5) * __min, ((_Tp)1) / ((_Tp)3)); 
# 568
if (((__x < ((_Tp)0)) || (__y < ((_Tp)0))) || (__z < ((_Tp)0))) { 
# 569
std::__throw_domain_error("Argument less than zero in __ellint_rj."); } else { 
# 571
if (((((__x + __y) < __lolim) || ((__x + __z) < __lolim)) || ((__y + __z) < __lolim)) || (__p < __lolim)) { 
# 573
std::__throw_domain_error("Argument too small in __ellint_rj"); } else 
# 576
{ 
# 577
const _Tp __c0 = (((_Tp)1) / ((_Tp)4)); 
# 578
const _Tp __c1 = (((_Tp)3) / ((_Tp)14)); 
# 579
const _Tp __c2 = (((_Tp)1) / ((_Tp)3)); 
# 580
const _Tp __c3 = (((_Tp)3) / ((_Tp)22)); 
# 581
const _Tp __c4 = (((_Tp)3) / ((_Tp)26)); 
# 583
_Tp __xn = __x; 
# 584
_Tp __yn = __y; 
# 585
_Tp __zn = __z; 
# 586
_Tp __pn = __p; 
# 587
_Tp __sigma = ((_Tp)0); 
# 588
_Tp __power4 = ((_Tp)1); 
# 590
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 591
const _Tp __errtol = std::pow(__eps / ((_Tp)8), ((_Tp)1) / ((_Tp)6)); 
# 593
_Tp __mu; 
# 594
_Tp __xndev, __yndev, __zndev, __pndev; 
# 596
const unsigned __max_iter = (100); 
# 597
for (unsigned __iter = (0); __iter < __max_iter; ++__iter) 
# 598
{ 
# 599
__mu = ((((__xn + __yn) + __zn) + (((_Tp)2) * __pn)) / ((_Tp)5)); 
# 600
__xndev = ((__mu - __xn) / __mu); 
# 601
__yndev = ((__mu - __yn) / __mu); 
# 602
__zndev = ((__mu - __zn) / __mu); 
# 603
__pndev = ((__mu - __pn) / __mu); 
# 604
_Tp __epsilon = std::max(std::abs(__xndev), std::abs(__yndev)); 
# 605
__epsilon = std::max(__epsilon, std::abs(__zndev)); 
# 606
__epsilon = std::max(__epsilon, std::abs(__pndev)); 
# 607
if (__epsilon < __errtol) { 
# 608
break; }  
# 609
const _Tp __xnroot = std::sqrt(__xn); 
# 610
const _Tp __ynroot = std::sqrt(__yn); 
# 611
const _Tp __znroot = std::sqrt(__zn); 
# 612
const _Tp __lambda = (__xnroot * (__ynroot + __znroot)) + (__ynroot * __znroot); 
# 614
const _Tp __alpha1 = (__pn * ((__xnroot + __ynroot) + __znroot)) + ((__xnroot * __ynroot) * __znroot); 
# 616
const _Tp __alpha2 = __alpha1 * __alpha1; 
# 617
const _Tp __beta = (__pn * (__pn + __lambda)) * (__pn + __lambda); 
# 619
__sigma += (__power4 * __ellint_rc(__alpha2, __beta)); 
# 620
__power4 *= __c0; 
# 621
__xn = (__c0 * (__xn + __lambda)); 
# 622
__yn = (__c0 * (__yn + __lambda)); 
# 623
__zn = (__c0 * (__zn + __lambda)); 
# 624
__pn = (__c0 * (__pn + __lambda)); 
# 625
}  
# 627
_Tp __ea = (__xndev * (__yndev + __zndev)) + (__yndev * __zndev); 
# 628
_Tp __eb = (__xndev * __yndev) * __zndev; 
# 629
_Tp __ec = __pndev * __pndev; 
# 630
_Tp __e2 = __ea - (((_Tp)3) * __ec); 
# 631
_Tp __e3 = __eb + ((((_Tp)2) * __pndev) * (__ea - __ec)); 
# 632
_Tp __s1 = ((_Tp)1) + (__e2 * (((-__c1) + (((((_Tp)3) * __c3) * __e2) / ((_Tp)4))) - (((((_Tp)3) * __c4) * __e3) / ((_Tp)2)))); 
# 634
_Tp __s2 = __eb * ((__c2 / ((_Tp)2)) + (__pndev * (((-__c3) - __c3) + (__pndev * __c4)))); 
# 636
_Tp __s3 = ((__pndev * __ea) * (__c2 - (__pndev * __c3))) - ((__c2 * __pndev) * __ec); 
# 639
return (((_Tp)3) * __sigma) + ((__power4 * ((__s1 + __s2) + __s3)) / (__mu * std::sqrt(__mu))); 
# 641
}  }  
# 642
} 
# 661 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 663
__comp_ellint_3(_Tp __k, _Tp __nu) 
# 664
{ 
# 666
if (__isnan(__k) || __isnan(__nu)) { 
# 667
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 668
if (__nu == ((_Tp)1)) { 
# 669
return std::numeric_limits< _Tp> ::infinity(); } else { 
# 670
if (std::abs(__k) > ((_Tp)1)) { 
# 671
std::__throw_domain_error("Bad argument in __comp_ellint_3."); } else 
# 673
{ 
# 674
const _Tp __kk = __k * __k; 
# 676
return __ellint_rf((_Tp)0, ((_Tp)1) - __kk, (_Tp)1) + ((__nu * __ellint_rj((_Tp)0, ((_Tp)1) - __kk, (_Tp)1, ((_Tp)1) - __nu)) / ((_Tp)3)); 
# 680
}  }  }  
# 681
} 
# 701 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/ell_integral.tcc" 3
template< class _Tp> _Tp 
# 703
__ellint_3(_Tp __k, _Tp __nu, _Tp __phi) 
# 704
{ 
# 706
if ((__isnan(__k) || __isnan(__nu)) || __isnan(__phi)) { 
# 707
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 708
if (std::abs(__k) > ((_Tp)1)) { 
# 709
std::__throw_domain_error("Bad argument in __ellint_3."); } else 
# 711
{ 
# 713
const int __n = std::floor((__phi / __numeric_constants< _Tp> ::__pi()) + ((_Tp)(0.5L))); 
# 715
const _Tp __phi_red = __phi - (__n * __numeric_constants< _Tp> ::__pi()); 
# 718
const _Tp __kk = __k * __k; 
# 719
const _Tp __s = std::sin(__phi_red); 
# 720
const _Tp __ss = __s * __s; 
# 721
const _Tp __sss = __ss * __s; 
# 722
const _Tp __c = std::cos(__phi_red); 
# 723
const _Tp __cc = __c * __c; 
# 725
const _Tp __Pi = (__s * __ellint_rf(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1)) + (((__nu * __sss) * __ellint_rj(__cc, ((_Tp)1) - (__kk * __ss), (_Tp)1, ((_Tp)1) - (__nu * __ss))) / ((_Tp)3)); 
# 731
if (__n == 0) { 
# 732
return __Pi; } else { 
# 734
return __Pi + ((((_Tp)2) * __n) * __comp_ellint_3(__k, __nu)); }  
# 735
}  }  
# 736
} 
# 737
}
# 743
}
# 50 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 64 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
namespace __detail { 
# 66
template< class _Tp> _Tp __expint_E1(_Tp); 
# 81 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 83
__expint_E1_series(_Tp __x) 
# 84
{ 
# 85
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 86
_Tp __term = ((_Tp)1); 
# 87
_Tp __esum = ((_Tp)0); 
# 88
_Tp __osum = ((_Tp)0); 
# 89
const unsigned __max_iter = (1000); 
# 90
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 91
{ 
# 92
__term *= ((-__x) / __i); 
# 93
if (std::abs(__term) < __eps) { 
# 94
break; }  
# 95
if (__term >= ((_Tp)0)) { 
# 96
__esum += (__term / __i); } else { 
# 98
__osum += (__term / __i); }  
# 99
}  
# 101
return (((-__esum) - __osum) - __numeric_constants< _Tp> ::__gamma_e()) - std::log(__x); 
# 103
} 
# 118 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 120
__expint_E1_asymp(_Tp __x) 
# 121
{ 
# 122
_Tp __term = ((_Tp)1); 
# 123
_Tp __esum = ((_Tp)1); 
# 124
_Tp __osum = ((_Tp)0); 
# 125
const unsigned __max_iter = (1000); 
# 126
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 127
{ 
# 128
_Tp __prev = __term; 
# 129
__term *= ((-__i) / __x); 
# 130
if (std::abs(__term) > std::abs(__prev)) { 
# 131
break; }  
# 132
if (__term >= ((_Tp)0)) { 
# 133
__esum += __term; } else { 
# 135
__osum += __term; }  
# 136
}  
# 138
return (std::exp(-__x) * (__esum + __osum)) / __x; 
# 139
} 
# 155 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 157
__expint_En_series(unsigned __n, _Tp __x) 
# 158
{ 
# 159
const unsigned __max_iter = (1000); 
# 160
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 161
const int __nm1 = __n - (1); 
# 162
_Tp __ans = (__nm1 != 0) ? ((_Tp)1) / __nm1 : ((-std::log(__x)) - __numeric_constants< _Tp> ::__gamma_e()); 
# 165
_Tp __fact = ((_Tp)1); 
# 166
for (int __i = 1; __i <= __max_iter; ++__i) 
# 167
{ 
# 168
__fact *= ((-__x) / ((_Tp)__i)); 
# 169
_Tp __del; 
# 170
if (__i != __nm1) { 
# 171
__del = ((-__fact) / ((_Tp)(__i - __nm1))); } else 
# 173
{ 
# 174
_Tp __psi = (-__numeric_constants< _Tp> ::gamma_e()); 
# 175
for (int __ii = 1; __ii <= __nm1; ++__ii) { 
# 176
__psi += (((_Tp)1) / ((_Tp)__ii)); }  
# 177
__del = (__fact * (__psi - std::log(__x))); 
# 178
}  
# 179
__ans += __del; 
# 180
if (std::abs(__del) < (__eps * std::abs(__ans))) { 
# 181
return __ans; }  
# 182
}  
# 183
std::__throw_runtime_error("Series summation failed in __expint_En_series."); 
# 185
} 
# 201 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 203
__expint_En_cont_frac(unsigned __n, _Tp __x) 
# 204
{ 
# 205
const unsigned __max_iter = (1000); 
# 206
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 207
const _Tp __fp_min = std::numeric_limits< _Tp> ::min(); 
# 208
const int __nm1 = __n - (1); 
# 209
_Tp __b = __x + ((_Tp)__n); 
# 210
_Tp __c = ((_Tp)1) / __fp_min; 
# 211
_Tp __d = ((_Tp)1) / __b; 
# 212
_Tp __h = __d; 
# 213
for (unsigned __i = (1); __i <= __max_iter; ++__i) 
# 214
{ 
# 215
_Tp __a = (-((_Tp)(__i * (__nm1 + __i)))); 
# 216
__b += ((_Tp)2); 
# 217
__d = (((_Tp)1) / ((__a * __d) + __b)); 
# 218
__c = (__b + (__a / __c)); 
# 219
const _Tp __del = __c * __d; 
# 220
__h *= __del; 
# 221
if (std::abs(__del - ((_Tp)1)) < __eps) 
# 222
{ 
# 223
const _Tp __ans = __h * std::exp(-__x); 
# 224
return __ans; 
# 225
}  
# 226
}  
# 227
std::__throw_runtime_error("Continued fraction failed in __expint_En_cont_frac."); 
# 229
} 
# 246 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 248
__expint_En_recursion(unsigned __n, _Tp __x) 
# 249
{ 
# 250
_Tp __En; 
# 251
_Tp __E1 = __expint_E1(__x); 
# 252
if (__x < ((_Tp)__n)) 
# 253
{ 
# 255
__En = __E1; 
# 256
for (unsigned __j = (2); __j < __n; ++__j) { 
# 257
__En = ((std::exp(-__x) - (__x * __En)) / ((_Tp)(__j - (1)))); }  
# 258
} else 
# 260
{ 
# 262
__En = ((_Tp)1); 
# 263
const int __N = __n + (20); 
# 264
_Tp __save = ((_Tp)0); 
# 265
for (int __j = __N; __j > 0; --__j) 
# 266
{ 
# 267
__En = ((std::exp(-__x) - (__j * __En)) / __x); 
# 268
if (__j == __n) { 
# 269
__save = __En; }  
# 270
}  
# 271
_Tp __norm = __En / __E1; 
# 272
__En /= __norm; 
# 273
}  
# 275
return __En; 
# 276
} 
# 290 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 292
__expint_Ei_series(_Tp __x) 
# 293
{ 
# 294
_Tp __term = ((_Tp)1); 
# 295
_Tp __sum = ((_Tp)0); 
# 296
const unsigned __max_iter = (1000); 
# 297
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 298
{ 
# 299
__term *= (__x / __i); 
# 300
__sum += (__term / __i); 
# 301
if (__term < (std::numeric_limits< _Tp> ::epsilon() * __sum)) { 
# 302
break; }  
# 303
}  
# 305
return (__numeric_constants< _Tp> ::__gamma_e() + __sum) + std::log(__x); 
# 306
} 
# 321 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 323
__expint_Ei_asymp(_Tp __x) 
# 324
{ 
# 325
_Tp __term = ((_Tp)1); 
# 326
_Tp __sum = ((_Tp)1); 
# 327
const unsigned __max_iter = (1000); 
# 328
for (unsigned __i = (1); __i < __max_iter; ++__i) 
# 329
{ 
# 330
_Tp __prev = __term; 
# 331
__term *= (__i / __x); 
# 332
if (__term < std::numeric_limits< _Tp> ::epsilon()) { 
# 333
break; }  
# 334
if (__term >= __prev) { 
# 335
break; }  
# 336
__sum += __term; 
# 337
}  
# 339
return (std::exp(__x) * __sum) / __x; 
# 340
} 
# 354 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 356
__expint_Ei(_Tp __x) 
# 357
{ 
# 358
if (__x < ((_Tp)0)) { 
# 359
return -__expint_E1(-__x); } else { 
# 360
if (__x < (-std::log(std::numeric_limits< _Tp> ::epsilon()))) { 
# 361
return __expint_Ei_series(__x); } else { 
# 363
return __expint_Ei_asymp(__x); }  }  
# 364
} 
# 378 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 380
__expint_E1(_Tp __x) 
# 381
{ 
# 382
if (__x < ((_Tp)0)) { 
# 383
return -__expint_Ei(-__x); } else { 
# 384
if (__x < ((_Tp)1)) { 
# 385
return __expint_E1_series(__x); } else { 
# 386
if (__x < ((_Tp)100)) { 
# 387
return __expint_En_cont_frac(1, __x); } else { 
# 389
return __expint_E1_asymp(__x); }  }  }  
# 390
} 
# 408 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 410
__expint_asymp(unsigned __n, _Tp __x) 
# 411
{ 
# 412
_Tp __term = ((_Tp)1); 
# 413
_Tp __sum = ((_Tp)1); 
# 414
for (unsigned __i = (1); __i <= __n; ++__i) 
# 415
{ 
# 416
_Tp __prev = __term; 
# 417
__term *= ((-((__n - __i) + (1))) / __x); 
# 418
if (std::abs(__term) > std::abs(__prev)) { 
# 419
break; }  
# 420
__sum += __term; 
# 421
}  
# 423
return (std::exp(-__x) * __sum) / __x; 
# 424
} 
# 442 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 444
__expint_large_n(unsigned __n, _Tp __x) 
# 445
{ 
# 446
const _Tp __xpn = __x + __n; 
# 447
const _Tp __xpn2 = __xpn * __xpn; 
# 448
_Tp __term = ((_Tp)1); 
# 449
_Tp __sum = ((_Tp)1); 
# 450
for (unsigned __i = (1); __i <= __n; ++__i) 
# 451
{ 
# 452
_Tp __prev = __term; 
# 453
__term *= ((__n - (((2) * (__i - (1))) * __x)) / __xpn2); 
# 454
if (std::abs(__term) < std::numeric_limits< _Tp> ::epsilon()) { 
# 455
break; }  
# 456
__sum += __term; 
# 457
}  
# 459
return (std::exp(-__x) * __sum) / __xpn; 
# 460
} 
# 476 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> _Tp 
# 478
__expint(unsigned __n, _Tp __x) 
# 479
{ 
# 481
if (__isnan(__x)) { 
# 482
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 483
if ((__n <= (1)) && (__x == ((_Tp)0))) { 
# 484
return std::numeric_limits< _Tp> ::infinity(); } else 
# 486
{ 
# 487
_Tp __E0 = std::exp(__x) / __x; 
# 488
if (__n == (0)) { 
# 489
return __E0; }  
# 491
_Tp __E1 = __expint_E1(__x); 
# 492
if (__n == (1)) { 
# 493
return __E1; }  
# 495
if (__x == ((_Tp)0)) { 
# 496
return ((_Tp)1) / (static_cast< _Tp>(__n - (1))); }  
# 498
_Tp __En = __expint_En_recursion(__n, __x); 
# 500
return __En; 
# 501
}  }  
# 502
} 
# 516 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/exp_integral.tcc" 3
template< class _Tp> inline _Tp 
# 518
__expint(_Tp __x) 
# 519
{ 
# 520
if (__isnan(__x)) { 
# 521
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 523
return __expint_Ei(__x); }  
# 524
} 
# 525
}
# 531
}
# 44 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/hypergeometric.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 60 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/hypergeometric.tcc" 3
namespace __detail { 
# 83 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 85
__conf_hyperg_series(_Tp __a, _Tp __c, _Tp __x) 
# 86
{ 
# 87
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 89
_Tp __term = ((_Tp)1); 
# 90
_Tp __Fac = ((_Tp)1); 
# 91
const unsigned __max_iter = (100000); 
# 92
unsigned __i; 
# 93
for (__i = (0); __i < __max_iter; ++__i) 
# 94
{ 
# 95
__term *= (((__a + ((_Tp)__i)) * __x) / ((__c + ((_Tp)__i)) * ((_Tp)((1) + __i)))); 
# 97
if (std::abs(__term) < __eps) 
# 98
{ 
# 99
break; 
# 100
}  
# 101
__Fac += __term; 
# 102
}  
# 103
if (__i == __max_iter) { 
# 104
std::__throw_runtime_error("Series failed to converge in __conf_hyperg_series."); }  
# 107
return __Fac; 
# 108
} 
# 120 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 122
__conf_hyperg_luke(_Tp __a, _Tp __c, _Tp __xin) 
# 123
{ 
# 124
const _Tp __big = std::pow(std::numeric_limits< _Tp> ::max(), (_Tp)(0.16L)); 
# 125
const int __nmax = 20000; 
# 126
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 127
const _Tp __x = (-__xin); 
# 128
const _Tp __x3 = (__x * __x) * __x; 
# 129
const _Tp __t0 = __a / __c; 
# 130
const _Tp __t1 = (__a + ((_Tp)1)) / (((_Tp)2) * __c); 
# 131
const _Tp __t2 = (__a + ((_Tp)2)) / (((_Tp)2) * (__c + ((_Tp)1))); 
# 132
_Tp __F = ((_Tp)1); 
# 133
_Tp __prec; 
# 135
_Tp __Bnm3 = ((_Tp)1); 
# 136
_Tp __Bnm2 = ((_Tp)1) + (__t1 * __x); 
# 137
_Tp __Bnm1 = ((_Tp)1) + ((__t2 * __x) * (((_Tp)1) + ((__t1 / ((_Tp)3)) * __x))); 
# 139
_Tp __Anm3 = ((_Tp)1); 
# 140
_Tp __Anm2 = __Bnm2 - (__t0 * __x); 
# 141
_Tp __Anm1 = (__Bnm1 - ((__t0 * (((_Tp)1) + (__t2 * __x))) * __x)) + ((((__t0 * __t1) * (__c / (__c + ((_Tp)1)))) * __x) * __x); 
# 144
int __n = 3; 
# 145
while (1) 
# 146
{ 
# 147
_Tp __npam1 = ((_Tp)(__n - 1)) + __a; 
# 148
_Tp __npcm1 = ((_Tp)(__n - 1)) + __c; 
# 149
_Tp __npam2 = ((_Tp)(__n - 2)) + __a; 
# 150
_Tp __npcm2 = ((_Tp)(__n - 2)) + __c; 
# 151
_Tp __tnm1 = (_Tp)((2 * __n) - 1); 
# 152
_Tp __tnm3 = (_Tp)((2 * __n) - 3); 
# 153
_Tp __tnm5 = (_Tp)((2 * __n) - 5); 
# 154
_Tp __F1 = (((_Tp)(__n - 2)) - __a) / ((((_Tp)2) * __tnm3) * __npcm1); 
# 155
_Tp __F2 = ((((_Tp)__n) + __a) * __npam1) / ((((((_Tp)4) * __tnm1) * __tnm3) * __npcm2) * __npcm1); 
# 157
_Tp __F3 = (((-__npam2) * __npam1) * (((_Tp)(__n - 2)) - __a)) / ((((((((_Tp)8) * __tnm3) * __tnm3) * __tnm5) * (((_Tp)(__n - 3)) + __c)) * __npcm2) * __npcm1); 
# 160
_Tp __E = ((-__npam1) * (((_Tp)(__n - 1)) - __c)) / (((((_Tp)2) * __tnm3) * __npcm2) * __npcm1); 
# 163
_Tp __An = (((((_Tp)1) + (__F1 * __x)) * __Anm1) + (((__E + (__F2 * __x)) * __x) * __Anm2)) + ((__F3 * __x3) * __Anm3); 
# 165
_Tp __Bn = (((((_Tp)1) + (__F1 * __x)) * __Bnm1) + (((__E + (__F2 * __x)) * __x) * __Bnm2)) + ((__F3 * __x3) * __Bnm3); 
# 167
_Tp __r = __An / __Bn; 
# 169
__prec = std::abs((__F - __r) / __F); 
# 170
__F = __r; 
# 172
if ((__prec < __eps) || (__n > __nmax)) { 
# 173
break; }  
# 175
if ((std::abs(__An) > __big) || (std::abs(__Bn) > __big)) 
# 176
{ 
# 177
__An /= __big; 
# 178
__Bn /= __big; 
# 179
__Anm1 /= __big; 
# 180
__Bnm1 /= __big; 
# 181
__Anm2 /= __big; 
# 182
__Bnm2 /= __big; 
# 183
__Anm3 /= __big; 
# 184
__Bnm3 /= __big; 
# 185
} else { 
# 186
if ((std::abs(__An) < (((_Tp)1) / __big)) || (std::abs(__Bn) < (((_Tp)1) / __big))) 
# 188
{ 
# 189
__An *= __big; 
# 190
__Bn *= __big; 
# 191
__Anm1 *= __big; 
# 192
__Bnm1 *= __big; 
# 193
__Anm2 *= __big; 
# 194
__Bnm2 *= __big; 
# 195
__Anm3 *= __big; 
# 196
__Bnm3 *= __big; 
# 197
}  }  
# 199
++__n; 
# 200
__Bnm3 = __Bnm2; 
# 201
__Bnm2 = __Bnm1; 
# 202
__Bnm1 = __Bn; 
# 203
__Anm3 = __Anm2; 
# 204
__Anm2 = __Anm1; 
# 205
__Anm1 = __An; 
# 206
}  
# 208
if (__n >= __nmax) { 
# 209
std::__throw_runtime_error("Iteration failed to converge in __conf_hyperg_luke."); }  
# 212
return __F; 
# 213
} 
# 227 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 229
__conf_hyperg(_Tp __a, _Tp __c, _Tp __x) 
# 230
{ 
# 232
const _Tp __c_nint = std::nearbyint(__c); 
# 236
if ((__isnan(__a) || __isnan(__c)) || __isnan(__x)) { 
# 237
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 238
if ((__c_nint == __c) && (__c_nint <= 0)) { 
# 239
return std::numeric_limits< _Tp> ::infinity(); } else { 
# 240
if (__a == ((_Tp)0)) { 
# 241
return (_Tp)1; } else { 
# 242
if (__c == __a) { 
# 243
return std::exp(__x); } else { 
# 244
if (__x < ((_Tp)0)) { 
# 245
return __conf_hyperg_luke(__a, __c, __x); } else { 
# 247
return __conf_hyperg_series(__a, __c, __x); }  }  }  }  }  
# 248
} 
# 271 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 273
__hyperg_series(_Tp __a, _Tp __b, _Tp __c, _Tp __x) 
# 274
{ 
# 275
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 277
_Tp __term = ((_Tp)1); 
# 278
_Tp __Fabc = ((_Tp)1); 
# 279
const unsigned __max_iter = (100000); 
# 280
unsigned __i; 
# 281
for (__i = (0); __i < __max_iter; ++__i) 
# 282
{ 
# 283
__term *= ((((__a + ((_Tp)__i)) * (__b + ((_Tp)__i))) * __x) / ((__c + ((_Tp)__i)) * ((_Tp)((1) + __i)))); 
# 285
if (std::abs(__term) < __eps) 
# 286
{ 
# 287
break; 
# 288
}  
# 289
__Fabc += __term; 
# 290
}  
# 291
if (__i == __max_iter) { 
# 292
std::__throw_runtime_error("Series failed to converge in __hyperg_series."); }  
# 295
return __Fabc; 
# 296
} 
# 304
template< class _Tp> _Tp 
# 306
__hyperg_luke(_Tp __a, _Tp __b, _Tp __c, _Tp __xin) 
# 307
{ 
# 308
const _Tp __big = std::pow(std::numeric_limits< _Tp> ::max(), (_Tp)(0.16L)); 
# 309
const int __nmax = 20000; 
# 310
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 311
const _Tp __x = (-__xin); 
# 312
const _Tp __x3 = (__x * __x) * __x; 
# 313
const _Tp __t0 = (__a * __b) / __c; 
# 314
const _Tp __t1 = ((__a + ((_Tp)1)) * (__b + ((_Tp)1))) / (((_Tp)2) * __c); 
# 315
const _Tp __t2 = ((__a + ((_Tp)2)) * (__b + ((_Tp)2))) / (((_Tp)2) * (__c + ((_Tp)1))); 
# 318
_Tp __F = ((_Tp)1); 
# 320
_Tp __Bnm3 = ((_Tp)1); 
# 321
_Tp __Bnm2 = ((_Tp)1) + (__t1 * __x); 
# 322
_Tp __Bnm1 = ((_Tp)1) + ((__t2 * __x) * (((_Tp)1) + ((__t1 / ((_Tp)3)) * __x))); 
# 324
_Tp __Anm3 = ((_Tp)1); 
# 325
_Tp __Anm2 = __Bnm2 - (__t0 * __x); 
# 326
_Tp __Anm1 = (__Bnm1 - ((__t0 * (((_Tp)1) + (__t2 * __x))) * __x)) + ((((__t0 * __t1) * (__c / (__c + ((_Tp)1)))) * __x) * __x); 
# 329
int __n = 3; 
# 330
while (1) 
# 331
{ 
# 332
const _Tp __npam1 = ((_Tp)(__n - 1)) + __a; 
# 333
const _Tp __npbm1 = ((_Tp)(__n - 1)) + __b; 
# 334
const _Tp __npcm1 = ((_Tp)(__n - 1)) + __c; 
# 335
const _Tp __npam2 = ((_Tp)(__n - 2)) + __a; 
# 336
const _Tp __npbm2 = ((_Tp)(__n - 2)) + __b; 
# 337
const _Tp __npcm2 = ((_Tp)(__n - 2)) + __c; 
# 338
const _Tp __tnm1 = (_Tp)((2 * __n) - 1); 
# 339
const _Tp __tnm3 = (_Tp)((2 * __n) - 3); 
# 340
const _Tp __tnm5 = (_Tp)((2 * __n) - 5); 
# 341
const _Tp __n2 = __n * __n; 
# 342
const _Tp __F1 = (((((((_Tp)3) * __n2) + (((__a + __b) - ((_Tp)6)) * __n)) + ((_Tp)2)) - (__a * __b)) - (((_Tp)2) * (__a + __b))) / ((((_Tp)2) * __tnm3) * __npcm1); 
# 345
const _Tp __F2 = (((-((((((_Tp)3) * __n2) - (((__a + __b) + ((_Tp)6)) * __n)) + ((_Tp)2)) - (__a * __b))) * __npam1) * __npbm1) / ((((((_Tp)4) * __tnm1) * __tnm3) * __npcm2) * __npcm1); 
# 348
const _Tp __F3 = (((((__npam2 * __npam1) * __npbm2) * __npbm1) * (((_Tp)(__n - 2)) - __a)) * (((_Tp)(__n - 2)) - __b)) / ((((((((_Tp)8) * __tnm3) * __tnm3) * __tnm5) * (((_Tp)(__n - 3)) + __c)) * __npcm2) * __npcm1); 
# 352
const _Tp __E = (((-__npam1) * __npbm1) * (((_Tp)(__n - 1)) - __c)) / (((((_Tp)2) * __tnm3) * __npcm2) * __npcm1); 
# 355
_Tp __An = (((((_Tp)1) + (__F1 * __x)) * __Anm1) + (((__E + (__F2 * __x)) * __x) * __Anm2)) + ((__F3 * __x3) * __Anm3); 
# 357
_Tp __Bn = (((((_Tp)1) + (__F1 * __x)) * __Bnm1) + (((__E + (__F2 * __x)) * __x) * __Bnm2)) + ((__F3 * __x3) * __Bnm3); 
# 359
const _Tp __r = __An / __Bn; 
# 361
const _Tp __prec = std::abs((__F - __r) / __F); 
# 362
__F = __r; 
# 364
if ((__prec < __eps) || (__n > __nmax)) { 
# 365
break; }  
# 367
if ((std::abs(__An) > __big) || (std::abs(__Bn) > __big)) 
# 368
{ 
# 369
__An /= __big; 
# 370
__Bn /= __big; 
# 371
__Anm1 /= __big; 
# 372
__Bnm1 /= __big; 
# 373
__Anm2 /= __big; 
# 374
__Bnm2 /= __big; 
# 375
__Anm3 /= __big; 
# 376
__Bnm3 /= __big; 
# 377
} else { 
# 378
if ((std::abs(__An) < (((_Tp)1) / __big)) || (std::abs(__Bn) < (((_Tp)1) / __big))) 
# 380
{ 
# 381
__An *= __big; 
# 382
__Bn *= __big; 
# 383
__Anm1 *= __big; 
# 384
__Bnm1 *= __big; 
# 385
__Anm2 *= __big; 
# 386
__Bnm2 *= __big; 
# 387
__Anm3 *= __big; 
# 388
__Bnm3 *= __big; 
# 389
}  }  
# 391
++__n; 
# 392
__Bnm3 = __Bnm2; 
# 393
__Bnm2 = __Bnm1; 
# 394
__Bnm1 = __Bn; 
# 395
__Anm3 = __Anm2; 
# 396
__Anm2 = __Anm1; 
# 397
__Anm1 = __An; 
# 398
}  
# 400
if (__n >= __nmax) { 
# 401
std::__throw_runtime_error("Iteration failed to converge in __hyperg_luke."); }  
# 404
return __F; 
# 405
} 
# 438 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 440
__hyperg_reflect(_Tp __a, _Tp __b, _Tp __c, _Tp __x) 
# 441
{ 
# 442
const _Tp __d = (__c - __a) - __b; 
# 443
const int __intd = std::floor(__d + ((_Tp)(0.5L))); 
# 444
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 445
const _Tp __toler = ((_Tp)1000) * __eps; 
# 446
const _Tp __log_max = std::log(std::numeric_limits< _Tp> ::max()); 
# 447
const bool __d_integer = std::abs(__d - __intd) < __toler; 
# 449
if (__d_integer) 
# 450
{ 
# 451
const _Tp __ln_omx = std::log(((_Tp)1) - __x); 
# 452
const _Tp __ad = std::abs(__d); 
# 453
_Tp __F1, __F2; 
# 455
_Tp __d1, __d2; 
# 456
if (__d >= ((_Tp)0)) 
# 457
{ 
# 458
__d1 = __d; 
# 459
__d2 = ((_Tp)0); 
# 460
} else 
# 462
{ 
# 463
__d1 = ((_Tp)0); 
# 464
__d2 = __d; 
# 465
}  
# 467
const _Tp __lng_c = __log_gamma(__c); 
# 470
if (__ad < __eps) 
# 471
{ 
# 473
__F1 = ((_Tp)0); 
# 474
} else 
# 476
{ 
# 478
bool __ok_d1 = true; 
# 479
_Tp __lng_ad, __lng_ad1, __lng_bd1; 
# 480
try 
# 481
{ 
# 482
__lng_ad = __log_gamma(__ad); 
# 483
__lng_ad1 = __log_gamma(__a + __d1); 
# 484
__lng_bd1 = __log_gamma(__b + __d1); 
# 485
} 
# 486
catch (...) 
# 487
{ 
# 488
__ok_d1 = false; 
# 489
}  
# 491
if (__ok_d1) 
# 492
{ 
# 496
_Tp __sum1 = ((_Tp)1); 
# 497
_Tp __term = ((_Tp)1); 
# 498
_Tp __ln_pre1 = (((__lng_ad + __lng_c) + (__d2 * __ln_omx)) - __lng_ad1) - __lng_bd1; 
# 503
for (int __i = 1; __i < __ad; ++__i) 
# 504
{ 
# 505
const int __j = __i - 1; 
# 506
__term *= ((((((__a + __d2) + __j) * ((__b + __d2) + __j)) / ((((_Tp)1) + __d2) + __j)) / __i) * (((_Tp)1) - __x)); 
# 508
__sum1 += __term; 
# 509
}  
# 511
if (__ln_pre1 > __log_max) { 
# 512
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_luke."); } else { 
# 515
__F1 = (std::exp(__ln_pre1) * __sum1); }  
# 516
} else 
# 518
{ 
# 521
__F1 = ((_Tp)0); 
# 522
}  
# 523
}  
# 526
bool __ok_d2 = true; 
# 527
_Tp __lng_ad2, __lng_bd2; 
# 528
try 
# 529
{ 
# 530
__lng_ad2 = __log_gamma(__a + __d2); 
# 531
__lng_bd2 = __log_gamma(__b + __d2); 
# 532
} 
# 533
catch (...) 
# 534
{ 
# 535
__ok_d2 = false; 
# 536
}  
# 538
if (__ok_d2) 
# 539
{ 
# 542
const int __maxiter = 2000; 
# 543
const _Tp __psi_1 = (-__numeric_constants< _Tp> ::__gamma_e()); 
# 544
const _Tp __psi_1pd = __psi(((_Tp)1) + __ad); 
# 545
const _Tp __psi_apd1 = __psi(__a + __d1); 
# 546
const _Tp __psi_bpd1 = __psi(__b + __d1); 
# 548
_Tp __psi_term = (((__psi_1 + __psi_1pd) - __psi_apd1) - __psi_bpd1) - __ln_omx; 
# 550
_Tp __fact = ((_Tp)1); 
# 551
_Tp __sum2 = __psi_term; 
# 552
_Tp __ln_pre2 = ((__lng_c + (__d1 * __ln_omx)) - __lng_ad2) - __lng_bd2; 
# 556
int __j; 
# 557
for (__j = 1; __j < __maxiter; ++__j) 
# 558
{ 
# 561
const _Tp __term1 = (((_Tp)1) / ((_Tp)__j)) + (((_Tp)1) / (__ad + __j)); 
# 563
const _Tp __term2 = (((_Tp)1) / ((__a + __d1) + ((_Tp)(__j - 1)))) + (((_Tp)1) / ((__b + __d1) + ((_Tp)(__j - 1)))); 
# 565
__psi_term += (__term1 - __term2); 
# 566
__fact *= (((((__a + __d1) + ((_Tp)(__j - 1))) * ((__b + __d1) + ((_Tp)(__j - 1)))) / ((__ad + __j) * __j)) * (((_Tp)1) - __x)); 
# 569
const _Tp __delta = __fact * __psi_term; 
# 570
__sum2 += __delta; 
# 571
if (std::abs(__delta) < (__eps * std::abs(__sum2))) { 
# 572
break; }  
# 573
}  
# 574
if (__j == __maxiter) { 
# 575
std::__throw_runtime_error("Sum F2 failed to converge in __hyperg_reflect"); }  
# 578
if (__sum2 == ((_Tp)0)) { 
# 579
__F2 = ((_Tp)0); } else { 
# 581
__F2 = (std::exp(__ln_pre2) * __sum2); }  
# 582
} else 
# 584
{ 
# 587
__F2 = ((_Tp)0); 
# 588
}  
# 590
const _Tp __sgn_2 = (((__intd % 2) == 1) ? -((_Tp)1) : ((_Tp)1)); 
# 591
const _Tp __F = __F1 + (__sgn_2 * __F2); 
# 593
return __F; 
# 594
} else 
# 596
{ 
# 601
bool __ok1 = true; 
# 602
_Tp __sgn_g1ca = ((_Tp)0), __ln_g1ca = ((_Tp)0); 
# 603
_Tp __sgn_g1cb = ((_Tp)0), __ln_g1cb = ((_Tp)0); 
# 604
try 
# 605
{ 
# 606
__sgn_g1ca = __log_gamma_sign(__c - __a); 
# 607
__ln_g1ca = __log_gamma(__c - __a); 
# 608
__sgn_g1cb = __log_gamma_sign(__c - __b); 
# 609
__ln_g1cb = __log_gamma(__c - __b); 
# 610
} 
# 611
catch (...) 
# 612
{ 
# 613
__ok1 = false; 
# 614
}  
# 616
bool __ok2 = true; 
# 617
_Tp __sgn_g2a = ((_Tp)0), __ln_g2a = ((_Tp)0); 
# 618
_Tp __sgn_g2b = ((_Tp)0), __ln_g2b = ((_Tp)0); 
# 619
try 
# 620
{ 
# 621
__sgn_g2a = __log_gamma_sign(__a); 
# 622
__ln_g2a = __log_gamma(__a); 
# 623
__sgn_g2b = __log_gamma_sign(__b); 
# 624
__ln_g2b = __log_gamma(__b); 
# 625
} 
# 626
catch (...) 
# 627
{ 
# 628
__ok2 = false; 
# 629
}  
# 631
const _Tp __sgn_gc = __log_gamma_sign(__c); 
# 632
const _Tp __ln_gc = __log_gamma(__c); 
# 633
const _Tp __sgn_gd = __log_gamma_sign(__d); 
# 634
const _Tp __ln_gd = __log_gamma(__d); 
# 635
const _Tp __sgn_gmd = __log_gamma_sign(-__d); 
# 636
const _Tp __ln_gmd = __log_gamma(-__d); 
# 638
const _Tp __sgn1 = ((__sgn_gc * __sgn_gd) * __sgn_g1ca) * __sgn_g1cb; 
# 639
const _Tp __sgn2 = ((__sgn_gc * __sgn_gmd) * __sgn_g2a) * __sgn_g2b; 
# 641
_Tp __pre1, __pre2; 
# 642
if (__ok1 && __ok2) 
# 643
{ 
# 644
_Tp __ln_pre1 = ((__ln_gc + __ln_gd) - __ln_g1ca) - __ln_g1cb; 
# 645
_Tp __ln_pre2 = (((__ln_gc + __ln_gmd) - __ln_g2a) - __ln_g2b) + (__d * std::log(((_Tp)1) - __x)); 
# 647
if ((__ln_pre1 < __log_max) && (__ln_pre2 < __log_max)) 
# 648
{ 
# 649
__pre1 = std::exp(__ln_pre1); 
# 650
__pre2 = std::exp(__ln_pre2); 
# 651
__pre1 *= __sgn1; 
# 652
__pre2 *= __sgn2; 
# 653
} else 
# 655
{ 
# 656
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_reflect"); 
# 658
}  
# 659
} else { 
# 660
if (__ok1 && (!__ok2)) 
# 661
{ 
# 662
_Tp __ln_pre1 = ((__ln_gc + __ln_gd) - __ln_g1ca) - __ln_g1cb; 
# 663
if (__ln_pre1 < __log_max) 
# 664
{ 
# 665
__pre1 = std::exp(__ln_pre1); 
# 666
__pre1 *= __sgn1; 
# 667
__pre2 = ((_Tp)0); 
# 668
} else 
# 670
{ 
# 671
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_reflect"); 
# 673
}  
# 674
} else { 
# 675
if ((!__ok1) && __ok2) 
# 676
{ 
# 677
_Tp __ln_pre2 = (((__ln_gc + __ln_gmd) - __ln_g2a) - __ln_g2b) + (__d * std::log(((_Tp)1) - __x)); 
# 679
if (__ln_pre2 < __log_max) 
# 680
{ 
# 681
__pre1 = ((_Tp)0); 
# 682
__pre2 = std::exp(__ln_pre2); 
# 683
__pre2 *= __sgn2; 
# 684
} else 
# 686
{ 
# 687
std::__throw_runtime_error("Overflow of gamma functions in __hyperg_reflect"); 
# 689
}  
# 690
} else 
# 692
{ 
# 693
__pre1 = ((_Tp)0); 
# 694
__pre2 = ((_Tp)0); 
# 695
std::__throw_runtime_error("Underflow of gamma functions in __hyperg_reflect"); 
# 697
}  }  }  
# 699
const _Tp __F1 = __hyperg_series(__a, __b, ((_Tp)1) - __d, ((_Tp)1) - __x); 
# 701
const _Tp __F2 = __hyperg_series(__c - __a, __c - __b, ((_Tp)1) + __d, ((_Tp)1) - __x); 
# 704
const _Tp __F = (__pre1 * __F1) + (__pre2 * __F2); 
# 706
return __F; 
# 707
}  
# 708
} 
# 728 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/hypergeometric.tcc" 3
template< class _Tp> _Tp 
# 730
__hyperg(_Tp __a, _Tp __b, _Tp __c, _Tp __x) 
# 731
{ 
# 733
const _Tp __a_nint = std::nearbyint(__a); 
# 734
const _Tp __b_nint = std::nearbyint(__b); 
# 735
const _Tp __c_nint = std::nearbyint(__c); 
# 741
const _Tp __toler = ((_Tp)1000) * std::numeric_limits< _Tp> ::epsilon(); 
# 742
if (std::abs(__x) >= ((_Tp)1)) { 
# 743
std::__throw_domain_error("Argument outside unit circle in __hyperg."); } else { 
# 745
if (((__isnan(__a) || __isnan(__b)) || __isnan(__c)) || __isnan(__x)) { 
# 747
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 748
if ((__c_nint == __c) && (__c_nint <= ((_Tp)0))) { 
# 749
return std::numeric_limits< _Tp> ::infinity(); } else { 
# 750
if ((std::abs(__c - __b) < __toler) || (std::abs(__c - __a) < __toler)) { 
# 751
return std::pow(((_Tp)1) - __x, (__c - __a) - __b); } else { 
# 752
if ((__a >= ((_Tp)0)) && (__b >= ((_Tp)0)) && (__c >= ((_Tp)0)) && (__x >= ((_Tp)0)) && (__x < ((_Tp)(0.995L)))) { 
# 754
return __hyperg_series(__a, __b, __c, __x); } else { 
# 755
if ((std::abs(__a) < ((_Tp)10)) && (std::abs(__b) < ((_Tp)10))) 
# 756
{ 
# 759
if ((__a < ((_Tp)0)) && (std::abs(__a - __a_nint) < __toler)) { 
# 760
return __hyperg_series(__a_nint, __b, __c, __x); } else { 
# 761
if ((__b < ((_Tp)0)) && (std::abs(__b - __b_nint) < __toler)) { 
# 762
return __hyperg_series(__a, __b_nint, __c, __x); } else { 
# 763
if (__x < (-((_Tp)(0.25L)))) { 
# 764
return __hyperg_luke(__a, __b, __c, __x); } else { 
# 765
if (__x < ((_Tp)(0.5L))) { 
# 766
return __hyperg_series(__a, __b, __c, __x); } else { 
# 768
if (std::abs(__c) > ((_Tp)10)) { 
# 769
return __hyperg_series(__a, __b, __c, __x); } else { 
# 771
return __hyperg_reflect(__a, __b, __c, __x); }  }  }  }  }  
# 772
} else { 
# 774
return __hyperg_luke(__a, __b, __c, __x); }  }  }  }  }  }  
# 775
} 
# 776
}
# 783
}
# 49 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/legendre_function.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/legendre_function.tcc" 3
namespace __detail { 
# 80 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/legendre_function.tcc" 3
template< class _Tp> _Tp 
# 82
__poly_legendre_p(unsigned __l, _Tp __x) 
# 83
{ 
# 85
if (__isnan(__x)) { 
# 86
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 87
if (__x == (+((_Tp)1))) { 
# 88
return +((_Tp)1); } else { 
# 89
if (__x == (-((_Tp)1))) { 
# 90
return (((__l % (2)) == (1)) ? -((_Tp)1) : (+((_Tp)1))); } else 
# 92
{ 
# 93
_Tp __p_lm2 = ((_Tp)1); 
# 94
if (__l == (0)) { 
# 95
return __p_lm2; }  
# 97
_Tp __p_lm1 = __x; 
# 98
if (__l == (1)) { 
# 99
return __p_lm1; }  
# 101
_Tp __p_l = (0); 
# 102
for (unsigned __ll = (2); __ll <= __l; ++__ll) 
# 103
{ 
# 106
__p_l = ((((((_Tp)2) * __x) * __p_lm1) - __p_lm2) - (((__x * __p_lm1) - __p_lm2) / ((_Tp)__ll))); 
# 108
__p_lm2 = __p_lm1; 
# 109
__p_lm1 = __p_l; 
# 110
}  
# 112
return __p_l; 
# 113
}  }  }  
# 114
} 
# 136 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/legendre_function.tcc" 3
template< class _Tp> _Tp 
# 138
__assoc_legendre_p(unsigned __l, unsigned __m, _Tp __x, _Tp 
# 139
__phase = (_Tp)(+1)) 
# 140
{ 
# 142
if (__m > __l) { 
# 143
return (_Tp)0; } else { 
# 144
if (__isnan(__x)) { 
# 145
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 146
if (__m == (0)) { 
# 147
return __poly_legendre_p(__l, __x); } else 
# 149
{ 
# 150
_Tp __p_mm = ((_Tp)1); 
# 151
if (__m > (0)) 
# 152
{ 
# 155
_Tp __root = std::sqrt(((_Tp)1) - __x) * std::sqrt(((_Tp)1) + __x); 
# 156
_Tp __fact = ((_Tp)1); 
# 157
for (unsigned __i = (1); __i <= __m; ++__i) 
# 158
{ 
# 159
__p_mm *= ((__phase * __fact) * __root); 
# 160
__fact += ((_Tp)2); 
# 161
}  
# 162
}  
# 163
if (__l == __m) { 
# 164
return __p_mm; }  
# 166
_Tp __p_mp1m = (((_Tp)(((2) * __m) + (1))) * __x) * __p_mm; 
# 167
if (__l == (__m + (1))) { 
# 168
return __p_mp1m; }  
# 170
_Tp __p_lm2m = __p_mm; 
# 171
_Tp __P_lm1m = __p_mp1m; 
# 172
_Tp __p_lm = ((_Tp)0); 
# 173
for (unsigned __j = __m + (2); __j <= __l; ++__j) 
# 174
{ 
# 175
__p_lm = ((((((_Tp)(((2) * __j) - (1))) * __x) * __P_lm1m) - (((_Tp)((__j + __m) - (1))) * __p_lm2m)) / ((_Tp)(__j - __m))); 
# 177
__p_lm2m = __P_lm1m; 
# 178
__P_lm1m = __p_lm; 
# 179
}  
# 181
return __p_lm; 
# 182
}  }  }  
# 183
} 
# 214 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/legendre_function.tcc" 3
template< class _Tp> _Tp 
# 216
__sph_legendre(unsigned __l, unsigned __m, _Tp __theta) 
# 217
{ 
# 218
if (__isnan(__theta)) { 
# 219
return std::numeric_limits< _Tp> ::quiet_NaN(); }  
# 221
const _Tp __x = std::cos(__theta); 
# 223
if (__m > __l) { 
# 224
return (_Tp)0; } else { 
# 225
if (__m == (0)) 
# 226
{ 
# 227
_Tp __P = __poly_legendre_p(__l, __x); 
# 228
_Tp __fact = std::sqrt(((_Tp)(((2) * __l) + (1))) / (((_Tp)4) * __numeric_constants< _Tp> ::__pi())); 
# 230
__P *= __fact; 
# 231
return __P; 
# 232
} else { 
# 233
if ((__x == ((_Tp)1)) || (__x == (-((_Tp)1)))) 
# 234
{ 
# 236
return (_Tp)0; 
# 237
} else 
# 239
{ 
# 245
const _Tp __sgn = ((__m % (2)) == (1)) ? -((_Tp)1) : ((_Tp)1); 
# 246
const _Tp __y_mp1m_factor = __x * std::sqrt((_Tp)(((2) * __m) + (3))); 
# 248
const _Tp __lncirc = std::log1p((-__x) * __x); 
# 254
const _Tp __lnpoch = std::lgamma((_Tp)(__m + ((_Tp)(0.5L)))) - std::lgamma((_Tp)__m); 
# 260
const _Tp __lnpre_val = ((-((_Tp)(0.25L))) * __numeric_constants< _Tp> ::__lnpi()) + (((_Tp)(0.5L)) * (__lnpoch + (__m * __lncirc))); 
# 263
const _Tp __sr = std::sqrt((((_Tp)2) + (((_Tp)1) / __m)) / (((_Tp)4) * __numeric_constants< _Tp> ::__pi())); 
# 265
_Tp __y_mm = (__sgn * __sr) * std::exp(__lnpre_val); 
# 266
_Tp __y_mp1m = __y_mp1m_factor * __y_mm; 
# 268
if (__l == __m) { 
# 269
return __y_mm; } else { 
# 270
if (__l == (__m + (1))) { 
# 271
return __y_mp1m; } else 
# 273
{ 
# 274
_Tp __y_lm = ((_Tp)0); 
# 277
for (unsigned __ll = __m + (2); __ll <= __l; ++__ll) 
# 278
{ 
# 279
const _Tp __rat1 = ((_Tp)(__ll - __m)) / ((_Tp)(__ll + __m)); 
# 280
const _Tp __rat2 = ((_Tp)((__ll - __m) - (1))) / ((_Tp)((__ll + __m) - (1))); 
# 281
const _Tp __fact1 = std::sqrt((__rat1 * ((_Tp)(((2) * __ll) + (1)))) * ((_Tp)(((2) * __ll) - (1)))); 
# 283
const _Tp __fact2 = std::sqrt(((__rat1 * __rat2) * ((_Tp)(((2) * __ll) + (1)))) / ((_Tp)(((2) * __ll) - (3)))); 
# 285
__y_lm = ((((__x * __y_mp1m) * __fact1) - ((((__ll + __m) - (1)) * __y_mm) * __fact2)) / ((_Tp)(__ll - __m))); 
# 287
__y_mm = __y_mp1m; 
# 288
__y_mp1m = __y_lm; 
# 289
}  
# 291
return __y_lm; 
# 292
}  }  
# 293
}  }  }  
# 294
} 
# 295
}
# 302
}
# 51 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/modified_bessel_func.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 65 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/modified_bessel_func.tcc" 3
namespace __detail { 
# 83 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/modified_bessel_func.tcc" 3
template< class _Tp> void 
# 85
__bessel_ik(_Tp __nu, _Tp __x, _Tp &
# 86
__Inu, _Tp &__Knu, _Tp &__Ipnu, _Tp &__Kpnu) 
# 87
{ 
# 88
if (__x == ((_Tp)0)) 
# 89
{ 
# 90
if (__nu == ((_Tp)0)) 
# 91
{ 
# 92
__Inu = ((_Tp)1); 
# 93
__Ipnu = ((_Tp)0); 
# 94
} else { 
# 95
if (__nu == ((_Tp)1)) 
# 96
{ 
# 97
__Inu = ((_Tp)0); 
# 98
__Ipnu = ((_Tp)(0.5L)); 
# 99
} else 
# 101
{ 
# 102
__Inu = ((_Tp)0); 
# 103
__Ipnu = ((_Tp)0); 
# 104
}  }  
# 105
__Knu = std::numeric_limits< _Tp> ::infinity(); 
# 106
__Kpnu = (-std::numeric_limits< _Tp> ::infinity()); 
# 107
return; 
# 108
}  
# 110
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 111
const _Tp __fp_min = ((_Tp)10) * std::numeric_limits< _Tp> ::epsilon(); 
# 112
const int __max_iter = 15000; 
# 113
const _Tp __x_min = ((_Tp)2); 
# 115
const int __nl = static_cast< int>(__nu + ((_Tp)(0.5L))); 
# 117
const _Tp __mu = __nu - __nl; 
# 118
const _Tp __mu2 = __mu * __mu; 
# 119
const _Tp __xi = ((_Tp)1) / __x; 
# 120
const _Tp __xi2 = ((_Tp)2) * __xi; 
# 121
_Tp __h = __nu * __xi; 
# 122
if (__h < __fp_min) { 
# 123
__h = __fp_min; }  
# 124
_Tp __b = __xi2 * __nu; 
# 125
_Tp __d = ((_Tp)0); 
# 126
_Tp __c = __h; 
# 127
int __i; 
# 128
for (__i = 1; __i <= __max_iter; ++__i) 
# 129
{ 
# 130
__b += __xi2; 
# 131
__d = (((_Tp)1) / (__b + __d)); 
# 132
__c = (__b + (((_Tp)1) / __c)); 
# 133
const _Tp __del = __c * __d; 
# 134
__h *= __del; 
# 135
if (std::abs(__del - ((_Tp)1)) < __eps) { 
# 136
break; }  
# 137
}  
# 138
if (__i > __max_iter) { 
# 139
std::__throw_runtime_error("Argument x too large in __bessel_ik; try asymptotic expansion."); }  
# 142
_Tp __Inul = __fp_min; 
# 143
_Tp __Ipnul = __h * __Inul; 
# 144
_Tp __Inul1 = __Inul; 
# 145
_Tp __Ipnu1 = __Ipnul; 
# 146
_Tp __fact = __nu * __xi; 
# 147
for (int __l = __nl; __l >= 1; --__l) 
# 148
{ 
# 149
const _Tp __Inutemp = (__fact * __Inul) + __Ipnul; 
# 150
__fact -= __xi; 
# 151
__Ipnul = ((__fact * __Inutemp) + __Inul); 
# 152
__Inul = __Inutemp; 
# 153
}  
# 154
_Tp __f = __Ipnul / __Inul; 
# 155
_Tp __Kmu, __Knu1; 
# 156
if (__x < __x_min) 
# 157
{ 
# 158
const _Tp __x2 = __x / ((_Tp)2); 
# 159
const _Tp __pimu = __numeric_constants< _Tp> ::__pi() * __mu; 
# 160
const _Tp __fact = (std::abs(__pimu) < __eps) ? (_Tp)1 : (__pimu / std::sin(__pimu)); 
# 162
_Tp __d = (-std::log(__x2)); 
# 163
_Tp __e = __mu * __d; 
# 164
const _Tp __fact2 = (std::abs(__e) < __eps) ? (_Tp)1 : (std::sinh(__e) / __e); 
# 166
_Tp __gam1, __gam2, __gampl, __gammi; 
# 167
__gamma_temme(__mu, __gam1, __gam2, __gampl, __gammi); 
# 168
_Tp __ff = __fact * ((__gam1 * std::cosh(__e)) + ((__gam2 * __fact2) * __d)); 
# 170
_Tp __sum = __ff; 
# 171
__e = std::exp(__e); 
# 172
_Tp __p = __e / (((_Tp)2) * __gampl); 
# 173
_Tp __q = ((_Tp)1) / ((((_Tp)2) * __e) * __gammi); 
# 174
_Tp __c = ((_Tp)1); 
# 175
__d = (__x2 * __x2); 
# 176
_Tp __sum1 = __p; 
# 177
int __i; 
# 178
for (__i = 1; __i <= __max_iter; ++__i) 
# 179
{ 
# 180
__ff = ((((__i * __ff) + __p) + __q) / ((__i * __i) - __mu2)); 
# 181
__c *= (__d / __i); 
# 182
__p /= (__i - __mu); 
# 183
__q /= (__i + __mu); 
# 184
const _Tp __del = __c * __ff; 
# 185
__sum += __del; 
# 186
const _Tp __del1 = __c * (__p - (__i * __ff)); 
# 187
__sum1 += __del1; 
# 188
if (std::abs(__del) < (__eps * std::abs(__sum))) { 
# 189
break; }  
# 190
}  
# 191
if (__i > __max_iter) { 
# 192
std::__throw_runtime_error("Bessel k series failed to converge in __bessel_ik."); }  
# 194
__Kmu = __sum; 
# 195
__Knu1 = (__sum1 * __xi2); 
# 196
} else 
# 198
{ 
# 199
_Tp __b = ((_Tp)2) * (((_Tp)1) + __x); 
# 200
_Tp __d = ((_Tp)1) / __b; 
# 201
_Tp __delh = __d; 
# 202
_Tp __h = __delh; 
# 203
_Tp __q1 = ((_Tp)0); 
# 204
_Tp __q2 = ((_Tp)1); 
# 205
_Tp __a1 = ((_Tp)(0.25L)) - __mu2; 
# 206
_Tp __q = __c = __a1; 
# 207
_Tp __a = (-__a1); 
# 208
_Tp __s = ((_Tp)1) + (__q * __delh); 
# 209
int __i; 
# 210
for (__i = 2; __i <= __max_iter; ++__i) 
# 211
{ 
# 212
__a -= (2 * (__i - 1)); 
# 213
__c = (((-__a) * __c) / __i); 
# 214
const _Tp __qnew = (__q1 - (__b * __q2)) / __a; 
# 215
__q1 = __q2; 
# 216
__q2 = __qnew; 
# 217
__q += (__c * __qnew); 
# 218
__b += ((_Tp)2); 
# 219
__d = (((_Tp)1) / (__b + (__a * __d))); 
# 220
__delh = (((__b * __d) - ((_Tp)1)) * __delh); 
# 221
__h += __delh; 
# 222
const _Tp __dels = __q * __delh; 
# 223
__s += __dels; 
# 224
if (std::abs(__dels / __s) < __eps) { 
# 225
break; }  
# 226
}  
# 227
if (__i > __max_iter) { 
# 228
std::__throw_runtime_error("Steed\'s method failed in __bessel_ik."); }  
# 230
__h = (__a1 * __h); 
# 231
__Kmu = ((std::sqrt(__numeric_constants< _Tp> ::__pi() / (((_Tp)2) * __x)) * std::exp(-__x)) / __s); 
# 233
__Knu1 = ((__Kmu * (((__mu + __x) + ((_Tp)(0.5L))) - __h)) * __xi); 
# 234
}  
# 236
_Tp __Kpmu = ((__mu * __xi) * __Kmu) - __Knu1; 
# 237
_Tp __Inumu = __xi / ((__f * __Kmu) - __Kpmu); 
# 238
__Inu = ((__Inumu * __Inul1) / __Inul); 
# 239
__Ipnu = ((__Inumu * __Ipnu1) / __Inul); 
# 240
for (__i = 1; __i <= __nl; ++__i) 
# 241
{ 
# 242
const _Tp __Knutemp = (((__mu + __i) * __xi2) * __Knu1) + __Kmu; 
# 243
__Kmu = __Knu1; 
# 244
__Knu1 = __Knutemp; 
# 245
}  
# 246
__Knu = __Kmu; 
# 247
__Kpnu = (((__nu * __xi) * __Kmu) - __Knu1); 
# 250
} 
# 267 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/modified_bessel_func.tcc" 3
template< class _Tp> _Tp 
# 269
__cyl_bessel_i(_Tp __nu, _Tp __x) 
# 270
{ 
# 271
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 272
std::__throw_domain_error("Bad argument in __cyl_bessel_i."); } else { 
# 274
if (__isnan(__nu) || __isnan(__x)) { 
# 275
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 276
if ((__x * __x) < (((_Tp)10) * (__nu + ((_Tp)1)))) { 
# 277
return __cyl_bessel_ij_series(__nu, __x, +((_Tp)1), 200); } else 
# 279
{ 
# 280
_Tp __I_nu, __K_nu, __Ip_nu, __Kp_nu; 
# 281
__bessel_ik(__nu, __x, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 282
return __I_nu; 
# 283
}  }  }  
# 284
} 
# 303 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/modified_bessel_func.tcc" 3
template< class _Tp> _Tp 
# 305
__cyl_bessel_k(_Tp __nu, _Tp __x) 
# 306
{ 
# 307
if ((__nu < ((_Tp)0)) || (__x < ((_Tp)0))) { 
# 308
std::__throw_domain_error("Bad argument in __cyl_bessel_k."); } else { 
# 310
if (__isnan(__nu) || __isnan(__x)) { 
# 311
return std::numeric_limits< _Tp> ::quiet_NaN(); } else 
# 313
{ 
# 314
_Tp __I_nu, __K_nu, __Ip_nu, __Kp_nu; 
# 315
__bessel_ik(__nu, __x, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 316
return __K_nu; 
# 317
}  }  
# 318
} 
# 337 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/modified_bessel_func.tcc" 3
template< class _Tp> void 
# 339
__sph_bessel_ik(unsigned __n, _Tp __x, _Tp &
# 340
__i_n, _Tp &__k_n, _Tp &__ip_n, _Tp &__kp_n) 
# 341
{ 
# 342
const _Tp __nu = ((_Tp)__n) + ((_Tp)(0.5L)); 
# 344
_Tp __I_nu, __Ip_nu, __K_nu, __Kp_nu; 
# 345
__bessel_ik(__nu, __x, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 347
const _Tp __factor = __numeric_constants< _Tp> ::__sqrtpio2() / std::sqrt(__x); 
# 350
__i_n = (__factor * __I_nu); 
# 351
__k_n = (__factor * __K_nu); 
# 352
__ip_n = ((__factor * __Ip_nu) - (__i_n / (((_Tp)2) * __x))); 
# 353
__kp_n = ((__factor * __Kp_nu) - (__k_n / (((_Tp)2) * __x))); 
# 356
} 
# 373 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/modified_bessel_func.tcc" 3
template< class _Tp> void 
# 375
__airy(_Tp __x, _Tp &__Ai, _Tp &__Bi, _Tp &__Aip, _Tp &__Bip) 
# 376
{ 
# 377
const _Tp __absx = std::abs(__x); 
# 378
const _Tp __rootx = std::sqrt(__absx); 
# 379
const _Tp __z = ((((_Tp)2) * __absx) * __rootx) / ((_Tp)3); 
# 380
const _Tp _S_inf = std::numeric_limits< _Tp> ::infinity(); 
# 382
if (__isnan(__x)) { 
# 383
__Bip = (__Aip = (__Bi = (__Ai = std::numeric_limits< _Tp> ::quiet_NaN()))); } else { 
# 384
if (__z == _S_inf) 
# 385
{ 
# 386
__Aip = (__Ai = ((_Tp)0)); 
# 387
__Bip = (__Bi = _S_inf); 
# 388
} else { 
# 389
if (__z == (-_S_inf)) { 
# 390
__Bip = (__Aip = (__Bi = (__Ai = ((_Tp)0)))); } else { 
# 391
if (__x > ((_Tp)0)) 
# 392
{ 
# 393
_Tp __I_nu, __Ip_nu, __K_nu, __Kp_nu; 
# 395
__bessel_ik(((_Tp)1) / ((_Tp)3), __z, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 396
__Ai = ((__rootx * __K_nu) / (__numeric_constants< _Tp> ::__sqrt3() * __numeric_constants< _Tp> ::__pi())); 
# 399
__Bi = (__rootx * ((__K_nu / __numeric_constants< _Tp> ::__pi()) + ((((_Tp)2) * __I_nu) / __numeric_constants< _Tp> ::__sqrt3()))); 
# 402
__bessel_ik(((_Tp)2) / ((_Tp)3), __z, __I_nu, __K_nu, __Ip_nu, __Kp_nu); 
# 403
__Aip = (((-__x) * __K_nu) / (__numeric_constants< _Tp> ::__sqrt3() * __numeric_constants< _Tp> ::__pi())); 
# 406
__Bip = (__x * ((__K_nu / __numeric_constants< _Tp> ::__pi()) + ((((_Tp)2) * __I_nu) / __numeric_constants< _Tp> ::__sqrt3()))); 
# 409
} else { 
# 410
if (__x < ((_Tp)0)) 
# 411
{ 
# 412
_Tp __J_nu, __Jp_nu, __N_nu, __Np_nu; 
# 414
__bessel_jn(((_Tp)1) / ((_Tp)3), __z, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 415
__Ai = ((__rootx * (__J_nu - (__N_nu / __numeric_constants< _Tp> ::__sqrt3()))) / ((_Tp)2)); 
# 417
__Bi = (((-__rootx) * (__N_nu + (__J_nu / __numeric_constants< _Tp> ::__sqrt3()))) / ((_Tp)2)); 
# 420
__bessel_jn(((_Tp)2) / ((_Tp)3), __z, __J_nu, __N_nu, __Jp_nu, __Np_nu); 
# 421
__Aip = ((__absx * ((__N_nu / __numeric_constants< _Tp> ::__sqrt3()) + __J_nu)) / ((_Tp)2)); 
# 423
__Bip = ((__absx * ((__J_nu / __numeric_constants< _Tp> ::__sqrt3()) - __N_nu)) / ((_Tp)2)); 
# 425
} else 
# 427
{ 
# 431
__Ai = ((_Tp)(0.35502805388781723926L)); 
# 432
__Bi = (__Ai * __numeric_constants< _Tp> ::__sqrt3()); 
# 437
__Aip = (-((_Tp)(0.2588194037928067984L))); 
# 438
__Bip = ((-__Aip) * __numeric_constants< _Tp> ::__sqrt3()); 
# 439
}  }  }  }  }  
# 442
} 
# 443
}
# 449
}
# 42 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_hermite.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 56 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_hermite.tcc" 3
namespace __detail { 
# 72 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_hermite.tcc" 3
template< class _Tp> _Tp 
# 74
__poly_hermite_recursion(unsigned __n, _Tp __x) 
# 75
{ 
# 77
_Tp __H_0 = (1); 
# 78
if (__n == (0)) { 
# 79
return __H_0; }  
# 82
_Tp __H_1 = 2 * __x; 
# 83
if (__n == (1)) { 
# 84
return __H_1; }  
# 87
_Tp __H_n, __H_nm1, __H_nm2; 
# 88
unsigned __i; 
# 89
for (((__H_nm2 = __H_0), (__H_nm1 = __H_1)), (__i = (2)); __i <= __n; ++__i) 
# 90
{ 
# 91
__H_n = (2 * ((__x * __H_nm1) - ((__i - (1)) * __H_nm2))); 
# 92
__H_nm2 = __H_nm1; 
# 93
__H_nm1 = __H_n; 
# 94
}  
# 96
return __H_n; 
# 97
} 
# 114 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_hermite.tcc" 3
template< class _Tp> inline _Tp 
# 116
__poly_hermite(unsigned __n, _Tp __x) 
# 117
{ 
# 118
if (__isnan(__x)) { 
# 119
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 121
return __poly_hermite_recursion(__n, __x); }  
# 122
} 
# 123
}
# 129
}
# 44 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_laguerre.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 60 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_laguerre.tcc" 3
namespace __detail { 
# 75 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_laguerre.tcc" 3
template< class _Tpa, class _Tp> _Tp 
# 77
__poly_laguerre_large_n(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 78
{ 
# 79
const _Tp __a = (-((_Tp)__n)); 
# 80
const _Tp __b = ((_Tp)__alpha1) + ((_Tp)1); 
# 81
const _Tp __eta = (((_Tp)2) * __b) - (((_Tp)4) * __a); 
# 82
const _Tp __cos2th = __x / __eta; 
# 83
const _Tp __sin2th = ((_Tp)1) - __cos2th; 
# 84
const _Tp __th = std::acos(std::sqrt(__cos2th)); 
# 85
const _Tp __pre_h = ((((__numeric_constants< _Tp> ::__pi_2() * __numeric_constants< _Tp> ::__pi_2()) * __eta) * __eta) * __cos2th) * __sin2th; 
# 90
const _Tp __lg_b = std::lgamma(((_Tp)__n) + __b); 
# 91
const _Tp __lnfact = std::lgamma((_Tp)(__n + (1))); 
# 97
_Tp __pre_term1 = (((_Tp)(0.5L)) * (((_Tp)1) - __b)) * std::log((((_Tp)(0.25L)) * __x) * __eta); 
# 99
_Tp __pre_term2 = ((_Tp)(0.25L)) * std::log(__pre_h); 
# 100
_Tp __lnpre = (((__lg_b - __lnfact) + (((_Tp)(0.5L)) * __x)) + __pre_term1) - __pre_term2; 
# 102
_Tp __ser_term1 = std::sin(__a * __numeric_constants< _Tp> ::__pi()); 
# 103
_Tp __ser_term2 = std::sin(((((_Tp)(0.25L)) * __eta) * ((((_Tp)2) * __th) - std::sin(((_Tp)2) * __th))) + __numeric_constants< _Tp> ::__pi_4()); 
# 107
_Tp __ser = __ser_term1 + __ser_term2; 
# 109
return std::exp(__lnpre) * __ser; 
# 110
} 
# 129 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_laguerre.tcc" 3
template< class _Tpa, class _Tp> _Tp 
# 131
__poly_laguerre_hyperg(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 132
{ 
# 133
const _Tp __b = ((_Tp)__alpha1) + ((_Tp)1); 
# 134
const _Tp __mx = (-__x); 
# 135
const _Tp __tc_sgn = (__x < ((_Tp)0)) ? (_Tp)1 : (((__n % (2)) == (1)) ? -((_Tp)1) : ((_Tp)1)); 
# 138
_Tp __tc = ((_Tp)1); 
# 139
const _Tp __ax = std::abs(__x); 
# 140
for (unsigned __k = (1); __k <= __n; ++__k) { 
# 141
__tc *= (__ax / __k); }  
# 143
_Tp __term = __tc * __tc_sgn; 
# 144
_Tp __sum = __term; 
# 145
for (int __k = ((int)__n) - 1; __k >= 0; --__k) 
# 146
{ 
# 147
__term *= ((((__b + ((_Tp)__k)) / ((_Tp)(((int)__n) - __k))) * ((_Tp)(__k + 1))) / __mx); 
# 149
__sum += __term; 
# 150
}  
# 152
return __sum; 
# 153
} 
# 185 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_laguerre.tcc" 3
template< class _Tpa, class _Tp> _Tp 
# 187
__poly_laguerre_recursion(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 188
{ 
# 190
_Tp __l_0 = ((_Tp)1); 
# 191
if (__n == (0)) { 
# 192
return __l_0; }  
# 195
_Tp __l_1 = (((-__x) + ((_Tp)1)) + ((_Tp)__alpha1)); 
# 196
if (__n == (1)) { 
# 197
return __l_1; }  
# 200
_Tp __l_n2 = __l_0; 
# 201
_Tp __l_n1 = __l_1; 
# 202
_Tp __l_n = ((_Tp)0); 
# 203
for (unsigned __nn = (2); __nn <= __n; ++__nn) 
# 204
{ 
# 205
__l_n = (((((((_Tp)(((2) * __nn) - (1))) + ((_Tp)__alpha1)) - __x) * __l_n1) / ((_Tp)__nn)) - (((((_Tp)(__nn - (1))) + ((_Tp)__alpha1)) * __l_n2) / ((_Tp)__nn))); 
# 208
__l_n2 = __l_n1; 
# 209
__l_n1 = __l_n; 
# 210
}  
# 212
return __l_n; 
# 213
} 
# 244 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_laguerre.tcc" 3
template< class _Tpa, class _Tp> _Tp 
# 246
__poly_laguerre(unsigned __n, _Tpa __alpha1, _Tp __x) 
# 247
{ 
# 248
if (__x < ((_Tp)0)) { 
# 249
std::__throw_domain_error("Negative argument in __poly_laguerre."); } else { 
# 252
if (__isnan(__x)) { 
# 253
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 254
if (__n == (0)) { 
# 255
return (_Tp)1; } else { 
# 256
if (__n == (1)) { 
# 257
return (((_Tp)1) + ((_Tp)__alpha1)) - __x; } else { 
# 258
if (__x == ((_Tp)0)) 
# 259
{ 
# 260
_Tp __prod = ((_Tp)__alpha1) + ((_Tp)1); 
# 261
for (unsigned __k = (2); __k <= __n; ++__k) { 
# 262
__prod *= ((((_Tp)__alpha1) + ((_Tp)__k)) / ((_Tp)__k)); }  
# 263
return __prod; 
# 264
} else { 
# 265
if ((__n > (10000000)) && (((_Tp)__alpha1) > (-((_Tp)1))) && (__x < ((((_Tp)2) * (((_Tp)__alpha1) + ((_Tp)1))) + ((_Tp)((4) * __n))))) { 
# 267
return __poly_laguerre_large_n(__n, __alpha1, __x); } else { 
# 268
if ((((_Tp)__alpha1) >= ((_Tp)0)) || ((__x > ((_Tp)0)) && (((_Tp)__alpha1) < (-((_Tp)(__n + (1))))))) { 
# 270
return __poly_laguerre_recursion(__n, __alpha1, __x); } else { 
# 272
return __poly_laguerre_hyperg(__n, __alpha1, __x); }  }  }  }  }  }  }  
# 273
} 
# 296 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_laguerre.tcc" 3
template< class _Tp> inline _Tp 
# 298
__assoc_laguerre(unsigned __n, unsigned __m, _Tp __x) 
# 299
{ return __poly_laguerre< unsigned, _Tp> (__n, __m, __x); } 
# 316 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/poly_laguerre.tcc" 3
template< class _Tp> inline _Tp 
# 318
__laguerre(unsigned __n, _Tp __x) 
# 319
{ return __poly_laguerre< unsigned, _Tp> (__n, 0, __x); } 
# 320
}
# 327
}
# 47 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/riemann_zeta.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 63 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/riemann_zeta.tcc" 3
namespace __detail { 
# 78 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 80
__riemann_zeta_sum(_Tp __s) 
# 81
{ 
# 83
if (__s < ((_Tp)1)) { 
# 84
std::__throw_domain_error("Bad argument in zeta sum."); }  
# 86
const unsigned max_iter = (10000); 
# 87
_Tp __zeta = ((_Tp)0); 
# 88
for (unsigned __k = (1); __k < max_iter; ++__k) 
# 89
{ 
# 90
_Tp __term = std::pow(static_cast< _Tp>(__k), -__s); 
# 91
if (__term < std::numeric_limits< _Tp> ::epsilon()) 
# 92
{ 
# 93
break; 
# 94
}  
# 95
__zeta += __term; 
# 96
}  
# 98
return __zeta; 
# 99
} 
# 115 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 117
__riemann_zeta_alt(_Tp __s) 
# 118
{ 
# 119
_Tp __sgn = ((_Tp)1); 
# 120
_Tp __zeta = ((_Tp)0); 
# 121
for (unsigned __i = (1); __i < (10000000); ++__i) 
# 122
{ 
# 123
_Tp __term = __sgn / std::pow(__i, __s); 
# 124
if (std::abs(__term) < std::numeric_limits< _Tp> ::epsilon()) { 
# 125
break; }  
# 126
__zeta += __term; 
# 127
__sgn *= ((_Tp)(-1)); 
# 128
}  
# 129
__zeta /= (((_Tp)1) - std::pow((_Tp)2, ((_Tp)1) - __s)); 
# 131
return __zeta; 
# 132
} 
# 157 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 159
__riemann_zeta_glob(_Tp __s) 
# 160
{ 
# 161
_Tp __zeta = ((_Tp)0); 
# 163
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 165
const _Tp __max_bincoeff = (std::numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 170
if (__s < ((_Tp)0)) 
# 171
{ 
# 173
if (std::fmod(__s, (_Tp)2) == ((_Tp)0)) { 
# 174
return (_Tp)0; } else 
# 177
{ 
# 178
_Tp __zeta = __riemann_zeta_glob(((_Tp)1) - __s); 
# 179
__zeta *= (((std::pow(((_Tp)2) * __numeric_constants< _Tp> ::__pi(), __s) * std::sin(__numeric_constants< _Tp> ::__pi_2() * __s)) * std::exp(std::lgamma(((_Tp)1) - __s))) / __numeric_constants< _Tp> ::__pi()); 
# 188
return __zeta; 
# 189
}  
# 190
}  
# 192
_Tp __num = ((_Tp)(0.5L)); 
# 193
const unsigned __maxit = (10000); 
# 194
for (unsigned __i = (0); __i < __maxit; ++__i) 
# 195
{ 
# 196
bool __punt = false; 
# 197
_Tp __sgn = ((_Tp)1); 
# 198
_Tp __term = ((_Tp)0); 
# 199
for (unsigned __j = (0); __j <= __i; ++__j) 
# 200
{ 
# 202
_Tp __bincoeff = (std::lgamma((_Tp)((1) + __i)) - std::lgamma((_Tp)((1) + __j))) - std::lgamma((_Tp)(((1) + __i) - __j)); 
# 210
if (__bincoeff > __max_bincoeff) 
# 211
{ 
# 213
__punt = true; 
# 214
break; 
# 215
}  
# 216
__bincoeff = std::exp(__bincoeff); 
# 217
__term += ((__sgn * __bincoeff) * std::pow((_Tp)((1) + __j), -__s)); 
# 218
__sgn *= ((_Tp)(-1)); 
# 219
}  
# 220
if (__punt) { 
# 221
break; }  
# 222
__term *= __num; 
# 223
__zeta += __term; 
# 224
if (std::abs(__term / __zeta) < __eps) { 
# 225
break; }  
# 226
__num *= ((_Tp)(0.5L)); 
# 227
}  
# 229
__zeta /= (((_Tp)1) - std::pow((_Tp)2, ((_Tp)1) - __s)); 
# 231
return __zeta; 
# 232
} 
# 252 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 254
__riemann_zeta_product(_Tp __s) 
# 255
{ 
# 256
static const _Tp __prime[] = {((_Tp)2), ((_Tp)3), ((_Tp)5), ((_Tp)7), ((_Tp)11), ((_Tp)13), ((_Tp)17), ((_Tp)19), ((_Tp)23), ((_Tp)29), ((_Tp)31), ((_Tp)37), ((_Tp)41), ((_Tp)43), ((_Tp)47), ((_Tp)53), ((_Tp)59), ((_Tp)61), ((_Tp)67), ((_Tp)71), ((_Tp)73), ((_Tp)79), ((_Tp)83), ((_Tp)89), ((_Tp)97), ((_Tp)101), ((_Tp)103), ((_Tp)107), ((_Tp)109)}; 
# 262
static const unsigned __num_primes = (sizeof(__prime) / sizeof(_Tp)); 
# 264
_Tp __zeta = ((_Tp)1); 
# 265
for (unsigned __i = (0); __i < __num_primes; ++__i) 
# 266
{ 
# 267
const _Tp __fact = ((_Tp)1) - std::pow(__prime[__i], -__s); 
# 268
__zeta *= __fact; 
# 269
if ((((_Tp)1) - __fact) < std::numeric_limits< _Tp> ::epsilon()) { 
# 270
break; }  
# 271
}  
# 273
__zeta = (((_Tp)1) / __zeta); 
# 275
return __zeta; 
# 276
} 
# 293 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 295
__riemann_zeta(_Tp __s) 
# 296
{ 
# 297
if (__isnan(__s)) { 
# 298
return std::numeric_limits< _Tp> ::quiet_NaN(); } else { 
# 299
if (__s == ((_Tp)1)) { 
# 300
return std::numeric_limits< _Tp> ::infinity(); } else { 
# 301
if (__s < (-((_Tp)19))) 
# 302
{ 
# 303
_Tp __zeta = __riemann_zeta_product(((_Tp)1) - __s); 
# 304
__zeta *= (((std::pow(((_Tp)2) * __numeric_constants< _Tp> ::__pi(), __s) * std::sin(__numeric_constants< _Tp> ::__pi_2() * __s)) * std::exp(std::lgamma(((_Tp)1) - __s))) / __numeric_constants< _Tp> ::__pi()); 
# 312
return __zeta; 
# 313
} else { 
# 314
if (__s < ((_Tp)20)) 
# 315
{ 
# 317
bool __glob = true; 
# 318
if (__glob) { 
# 319
return __riemann_zeta_glob(__s); } else 
# 321
{ 
# 322
if (__s > ((_Tp)1)) { 
# 323
return __riemann_zeta_sum(__s); } else 
# 325
{ 
# 326
_Tp __zeta = ((std::pow(((_Tp)2) * __numeric_constants< _Tp> ::__pi(), __s) * std::sin(__numeric_constants< _Tp> ::__pi_2() * __s)) * std::tgamma(((_Tp)1) - __s)) * __riemann_zeta_sum(((_Tp)1) - __s); 
# 335
return __zeta; 
# 336
}  
# 337
}  
# 338
} else { 
# 340
return __riemann_zeta_product(__s); }  }  }  }  
# 341
} 
# 365 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/riemann_zeta.tcc" 3
template< class _Tp> _Tp 
# 367
__hurwitz_zeta_glob(_Tp __a, _Tp __s) 
# 368
{ 
# 369
_Tp __zeta = ((_Tp)0); 
# 371
const _Tp __eps = std::numeric_limits< _Tp> ::epsilon(); 
# 373
const _Tp __max_bincoeff = (std::numeric_limits< _Tp> ::max_exponent10 * std::log((_Tp)10)) - ((_Tp)1); 
# 376
const unsigned __maxit = (10000); 
# 377
for (unsigned __i = (0); __i < __maxit; ++__i) 
# 378
{ 
# 379
bool __punt = false; 
# 380
_Tp __sgn = ((_Tp)1); 
# 381
_Tp __term = ((_Tp)0); 
# 382
for (unsigned __j = (0); __j <= __i; ++__j) 
# 383
{ 
# 385
_Tp __bincoeff = (std::lgamma((_Tp)((1) + __i)) - std::lgamma((_Tp)((1) + __j))) - std::lgamma((_Tp)(((1) + __i) - __j)); 
# 393
if (__bincoeff > __max_bincoeff) 
# 394
{ 
# 396
__punt = true; 
# 397
break; 
# 398
}  
# 399
__bincoeff = std::exp(__bincoeff); 
# 400
__term += ((__sgn * __bincoeff) * std::pow((_Tp)(__a + __j), -__s)); 
# 401
__sgn *= ((_Tp)(-1)); 
# 402
}  
# 403
if (__punt) { 
# 404
break; }  
# 405
__term /= ((_Tp)(__i + (1))); 
# 406
if (std::abs(__term / __zeta) < __eps) { 
# 407
break; }  
# 408
__zeta += __term; 
# 409
}  
# 411
__zeta /= (__s - ((_Tp)1)); 
# 413
return __zeta; 
# 414
} 
# 430 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/tr1/riemann_zeta.tcc" 3
template< class _Tp> inline _Tp 
# 432
__hurwitz_zeta(_Tp __a, _Tp __s) 
# 433
{ return __hurwitz_zeta_glob(__a, __s); } 
# 434
}
# 441
}
# 61 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 206 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float assoc_laguerref(unsigned __n, unsigned __m, float __x) 
# 207
{ return __detail::__assoc_laguerre< float> (__n, __m, __x); } 
# 216
inline long double assoc_laguerrel(unsigned __n, unsigned __m, long double __x) 
# 217
{ return __detail::__assoc_laguerre< long double> (__n, __m, __x); } 
# 250 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 252
assoc_laguerre(unsigned __n, unsigned __m, _Tp __x) 
# 253
{ 
# 254
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 255
return __detail::__assoc_laguerre< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __m, __x); 
# 256
} 
# 267 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float assoc_legendref(unsigned __l, unsigned __m, float __x) 
# 268
{ return __detail::__assoc_legendre_p< float> (__l, __m, __x); } 
# 276
inline long double assoc_legendrel(unsigned __l, unsigned __m, long double __x) 
# 277
{ return __detail::__assoc_legendre_p< long double> (__l, __m, __x); } 
# 296 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 298
assoc_legendre(unsigned __l, unsigned __m, _Tp __x) 
# 299
{ 
# 300
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 301
return __detail::__assoc_legendre_p< typename __gnu_cxx::__promote< _Tp> ::__type> (__l, __m, __x); 
# 302
} 
# 312 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float betaf(float __a, float __b) 
# 313
{ return __detail::__beta< float> (__a, __b); } 
# 322
inline long double betal(long double __a, long double __b) 
# 323
{ return __detail::__beta< long double> (__a, __b); } 
# 341 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tpa, class _Tpb> inline typename __gnu_cxx::__promote_2< _Tpa, _Tpb> ::__type 
# 343
beta(_Tpa __a, _Tpb __b) 
# 344
{ 
# 345
typedef typename __gnu_cxx::__promote_2< _Tpa, _Tpb> ::__type __type; 
# 346
return __detail::__beta< typename __gnu_cxx::__promote_2< _Tpa, _Tpb> ::__type> (__a, __b); 
# 347
} 
# 358 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float comp_ellint_1f(float __k) 
# 359
{ return __detail::__comp_ellint_1< float> (__k); } 
# 368
inline long double comp_ellint_1l(long double __k) 
# 369
{ return __detail::__comp_ellint_1< long double> (__k); } 
# 389 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 391
comp_ellint_1(_Tp __k) 
# 392
{ 
# 393
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 394
return __detail::__comp_ellint_1< typename __gnu_cxx::__promote< _Tp> ::__type> (__k); 
# 395
} 
# 406 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float comp_ellint_2f(float __k) 
# 407
{ return __detail::__comp_ellint_2< float> (__k); } 
# 416
inline long double comp_ellint_2l(long double __k) 
# 417
{ return __detail::__comp_ellint_2< long double> (__k); } 
# 436 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 438
comp_ellint_2(_Tp __k) 
# 439
{ 
# 440
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 441
return __detail::__comp_ellint_2< typename __gnu_cxx::__promote< _Tp> ::__type> (__k); 
# 442
} 
# 453 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float comp_ellint_3f(float __k, float __nu) 
# 454
{ return __detail::__comp_ellint_3< float> (__k, __nu); } 
# 463
inline long double comp_ellint_3l(long double __k, long double __nu) 
# 464
{ return __detail::__comp_ellint_3< long double> (__k, __nu); } 
# 487 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp, class _Tpn> inline typename __gnu_cxx::__promote_2< _Tp, _Tpn> ::__type 
# 489
comp_ellint_3(_Tp __k, _Tpn __nu) 
# 490
{ 
# 491
typedef typename __gnu_cxx::__promote_2< _Tp, _Tpn> ::__type __type; 
# 492
return __detail::__comp_ellint_3< typename __gnu_cxx::__promote_2< _Tp, _Tpn> ::__type> (__k, __nu); 
# 493
} 
# 504 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float cyl_bessel_if(float __nu, float __x) 
# 505
{ return __detail::__cyl_bessel_i< float> (__nu, __x); } 
# 514
inline long double cyl_bessel_il(long double __nu, long double __x) 
# 515
{ return __detail::__cyl_bessel_i< long double> (__nu, __x); } 
# 533 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 535
cyl_bessel_i(_Tpnu __nu, _Tp __x) 
# 536
{ 
# 537
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 538
return __detail::__cyl_bessel_i< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 539
} 
# 550 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float cyl_bessel_jf(float __nu, float __x) 
# 551
{ return __detail::__cyl_bessel_j< float> (__nu, __x); } 
# 560
inline long double cyl_bessel_jl(long double __nu, long double __x) 
# 561
{ return __detail::__cyl_bessel_j< long double> (__nu, __x); } 
# 579 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 581
cyl_bessel_j(_Tpnu __nu, _Tp __x) 
# 582
{ 
# 583
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 584
return __detail::__cyl_bessel_j< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 585
} 
# 596 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float cyl_bessel_kf(float __nu, float __x) 
# 597
{ return __detail::__cyl_bessel_k< float> (__nu, __x); } 
# 606
inline long double cyl_bessel_kl(long double __nu, long double __x) 
# 607
{ return __detail::__cyl_bessel_k< long double> (__nu, __x); } 
# 631 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 633
cyl_bessel_k(_Tpnu __nu, _Tp __x) 
# 634
{ 
# 635
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 636
return __detail::__cyl_bessel_k< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 637
} 
# 648 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float cyl_neumannf(float __nu, float __x) 
# 649
{ return __detail::__cyl_neumann_n< float> (__nu, __x); } 
# 658
inline long double cyl_neumannl(long double __nu, long double __x) 
# 659
{ return __detail::__cyl_neumann_n< long double> (__nu, __x); } 
# 679 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tpnu, class _Tp> inline typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type 
# 681
cyl_neumann(_Tpnu __nu, _Tp __x) 
# 682
{ 
# 683
typedef typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type __type; 
# 684
return __detail::__cyl_neumann_n< typename __gnu_cxx::__promote_2< _Tpnu, _Tp> ::__type> (__nu, __x); 
# 685
} 
# 696 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float ellint_1f(float __k, float __phi) 
# 697
{ return __detail::__ellint_1< float> (__k, __phi); } 
# 706
inline long double ellint_1l(long double __k, long double __phi) 
# 707
{ return __detail::__ellint_1< long double> (__k, __phi); } 
# 727 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp, class _Tpp> inline typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type 
# 729
ellint_1(_Tp __k, _Tpp __phi) 
# 730
{ 
# 731
typedef typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type __type; 
# 732
return __detail::__ellint_1< typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type> (__k, __phi); 
# 733
} 
# 744 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float ellint_2f(float __k, float __phi) 
# 745
{ return __detail::__ellint_2< float> (__k, __phi); } 
# 754
inline long double ellint_2l(long double __k, long double __phi) 
# 755
{ return __detail::__ellint_2< long double> (__k, __phi); } 
# 775 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp, class _Tpp> inline typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type 
# 777
ellint_2(_Tp __k, _Tpp __phi) 
# 778
{ 
# 779
typedef typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type __type; 
# 780
return __detail::__ellint_2< typename __gnu_cxx::__promote_2< _Tp, _Tpp> ::__type> (__k, __phi); 
# 781
} 
# 792 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float ellint_3f(float __k, float __nu, float __phi) 
# 793
{ return __detail::__ellint_3< float> (__k, __nu, __phi); } 
# 802
inline long double ellint_3l(long double __k, long double __nu, long double __phi) 
# 803
{ return __detail::__ellint_3< long double> (__k, __nu, __phi); } 
# 828 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp, class _Tpn, class _Tpp> inline typename __gnu_cxx::__promote_3< _Tp, _Tpn, _Tpp> ::__type 
# 830
ellint_3(_Tp __k, _Tpn __nu, _Tpp __phi) 
# 831
{ 
# 832
typedef typename __gnu_cxx::__promote_3< _Tp, _Tpn, _Tpp> ::__type __type; 
# 833
return __detail::__ellint_3< typename __gnu_cxx::__promote_3< _Tp, _Tpn, _Tpp> ::__type> (__k, __nu, __phi); 
# 834
} 
# 844 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float expintf(float __x) 
# 845
{ return __detail::__expint< float> (__x); } 
# 854
inline long double expintl(long double __x) 
# 855
{ return __detail::__expint< long double> (__x); } 
# 868 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 870
expint(_Tp __x) 
# 871
{ 
# 872
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 873
return __detail::__expint< typename __gnu_cxx::__promote< _Tp> ::__type> (__x); 
# 874
} 
# 885 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float hermitef(unsigned __n, float __x) 
# 886
{ return __detail::__poly_hermite< float> (__n, __x); } 
# 895
inline long double hermitel(unsigned __n, long double __x) 
# 896
{ return __detail::__poly_hermite< long double> (__n, __x); } 
# 916 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 918
hermite(unsigned __n, _Tp __x) 
# 919
{ 
# 920
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 921
return __detail::__poly_hermite< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 922
} 
# 933 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float laguerref(unsigned __n, float __x) 
# 934
{ return __detail::__laguerre< float> (__n, __x); } 
# 943
inline long double laguerrel(unsigned __n, long double __x) 
# 944
{ return __detail::__laguerre< long double> (__n, __x); } 
# 960 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 962
laguerre(unsigned __n, _Tp __x) 
# 963
{ 
# 964
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 965
return __detail::__laguerre< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 966
} 
# 977 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float legendref(unsigned __l, float __x) 
# 978
{ return __detail::__poly_legendre_p< float> (__l, __x); } 
# 987
inline long double legendrel(unsigned __l, long double __x) 
# 988
{ return __detail::__poly_legendre_p< long double> (__l, __x); } 
# 1005 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1007
legendre(unsigned __l, _Tp __x) 
# 1008
{ 
# 1009
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1010
return __detail::__poly_legendre_p< typename __gnu_cxx::__promote< _Tp> ::__type> (__l, __x); 
# 1011
} 
# 1022 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float riemann_zetaf(float __s) 
# 1023
{ return __detail::__riemann_zeta< float> (__s); } 
# 1032
inline long double riemann_zetal(long double __s) 
# 1033
{ return __detail::__riemann_zeta< long double> (__s); } 
# 1056 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1058
riemann_zeta(_Tp __s) 
# 1059
{ 
# 1060
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1061
return __detail::__riemann_zeta< typename __gnu_cxx::__promote< _Tp> ::__type> (__s); 
# 1062
} 
# 1073 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float sph_besself(unsigned __n, float __x) 
# 1074
{ return __detail::__sph_bessel< float> (__n, __x); } 
# 1083
inline long double sph_bessell(unsigned __n, long double __x) 
# 1084
{ return __detail::__sph_bessel< long double> (__n, __x); } 
# 1100 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1102
sph_bessel(unsigned __n, _Tp __x) 
# 1103
{ 
# 1104
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1105
return __detail::__sph_bessel< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 1106
} 
# 1117 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float sph_legendref(unsigned __l, unsigned __m, float __theta) 
# 1118
{ return __detail::__sph_legendre< float> (__l, __m, __theta); } 
# 1128 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline long double sph_legendrel(unsigned __l, unsigned __m, long double __theta) 
# 1129
{ return __detail::__sph_legendre< long double> (__l, __m, __theta); } 
# 1147 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1149
sph_legendre(unsigned __l, unsigned __m, _Tp __theta) 
# 1150
{ 
# 1151
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1152
return __detail::__sph_legendre< typename __gnu_cxx::__promote< _Tp> ::__type> (__l, __m, __theta); 
# 1153
} 
# 1164 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float sph_neumannf(unsigned __n, float __x) 
# 1165
{ return __detail::__sph_neumann< float> (__n, __x); } 
# 1174
inline long double sph_neumannl(unsigned __n, long double __x) 
# 1175
{ return __detail::__sph_neumann< long double> (__n, __x); } 
# 1191 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tp> inline typename __gnu_cxx::__promote< _Tp> ::__type 
# 1193
sph_neumann(unsigned __n, _Tp __x) 
# 1194
{ 
# 1195
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 1196
return __detail::__sph_neumann< typename __gnu_cxx::__promote< _Tp> ::__type> (__n, __x); 
# 1197
} 
# 1202
}
# 1205
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 1219 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float airy_aif(float __x) 
# 1220
{ 
# 1221
float __Ai, __Bi, __Aip, __Bip; 
# 1222
std::__detail::__airy< float> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1223
return __Ai; 
# 1224
} 
# 1230
inline long double airy_ail(long double __x) 
# 1231
{ 
# 1232
long double __Ai, __Bi, __Aip, __Bip; 
# 1233
std::__detail::__airy< long double> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1234
return __Ai; 
# 1235
} 
# 1240
template< class _Tp> inline typename __promote< _Tp> ::__type 
# 1242
airy_ai(_Tp __x) 
# 1243
{ 
# 1244
typedef typename __promote< _Tp> ::__type __type; 
# 1245
__type __Ai, __Bi, __Aip, __Bip; 
# 1246
std::__detail::__airy< typename __promote< _Tp> ::__type> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1247
return __Ai; 
# 1248
} 
# 1254
inline float airy_bif(float __x) 
# 1255
{ 
# 1256
float __Ai, __Bi, __Aip, __Bip; 
# 1257
std::__detail::__airy< float> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1258
return __Bi; 
# 1259
} 
# 1265
inline long double airy_bil(long double __x) 
# 1266
{ 
# 1267
long double __Ai, __Bi, __Aip, __Bip; 
# 1268
std::__detail::__airy< long double> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1269
return __Bi; 
# 1270
} 
# 1275
template< class _Tp> inline typename __promote< _Tp> ::__type 
# 1277
airy_bi(_Tp __x) 
# 1278
{ 
# 1279
typedef typename __promote< _Tp> ::__type __type; 
# 1280
__type __Ai, __Bi, __Aip, __Bip; 
# 1281
std::__detail::__airy< typename __promote< _Tp> ::__type> (__x, __Ai, __Bi, __Aip, __Bip); 
# 1282
return __Bi; 
# 1283
} 
# 1295 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float conf_hypergf(float __a, float __c, float __x) 
# 1296
{ return std::__detail::__conf_hyperg< float> (__a, __c, __x); } 
# 1306 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline long double conf_hypergl(long double __a, long double __c, long double __x) 
# 1307
{ return std::__detail::__conf_hyperg< long double> (__a, __c, __x); } 
# 1325 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tpa, class _Tpc, class _Tp> inline typename __promote_3< _Tpa, _Tpc, _Tp> ::__type 
# 1327
conf_hyperg(_Tpa __a, _Tpc __c, _Tp __x) 
# 1328
{ 
# 1329
typedef typename __promote_3< _Tpa, _Tpc, _Tp> ::__type __type; 
# 1330
return std::__detail::__conf_hyperg< typename __promote_3< _Tpa, _Tpc, _Tp> ::__type> (__a, __c, __x); 
# 1331
} 
# 1343 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline float hypergf(float __a, float __b, float __c, float __x) 
# 1344
{ return std::__detail::__hyperg< float> (__a, __b, __c, __x); } 
# 1354 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
inline long double hypergl(long double __a, long double __b, long double __c, long double __x) 
# 1355
{ return std::__detail::__hyperg< long double> (__a, __b, __c, __x); } 
# 1374 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/bits/specfun.h" 3
template< class _Tpa, class _Tpb, class _Tpc, class _Tp> inline typename __promote_4< _Tpa, _Tpb, _Tpc, _Tp> ::__type 
# 1376
hyperg(_Tpa __a, _Tpb __b, _Tpc __c, _Tp __x) 
# 1377
{ 
# 1379
typedef typename __promote_4< _Tpa, _Tpb, _Tpc, _Tp> ::__type __type; 
# 1380
return std::__detail::__hyperg< typename __promote_4< _Tpa, _Tpb, _Tpc, _Tp> ::__type> (__a, __b, __c, __x); 
# 1381
} 
# 1385
}
# 1388
#pragma GCC visibility pop
# 1930 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/cmath" 3
}
# 38 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/math.h" 3
using std::abs;
# 39
using std::acos;
# 40
using std::asin;
# 41
using std::atan;
# 42
using std::atan2;
# 43
using std::cos;
# 44
using std::sin;
# 45
using std::tan;
# 46
using std::cosh;
# 47
using std::sinh;
# 48
using std::tanh;
# 49
using std::exp;
# 50
using std::frexp;
# 51
using std::ldexp;
# 52
using std::log;
# 53
using std::log10;
# 54
using std::modf;
# 55
using std::pow;
# 56
using std::sqrt;
# 57
using std::ceil;
# 58
using std::fabs;
# 59
using std::floor;
# 60
using std::fmod;
# 63
using std::fpclassify;
# 64
using std::isfinite;
# 65
using std::isinf;
# 66
using std::isnan;
# 67
using std::isnormal;
# 68
using std::signbit;
# 69
using std::isgreater;
# 70
using std::isgreaterequal;
# 71
using std::isless;
# 72
using std::islessequal;
# 73
using std::islessgreater;
# 74
using std::isunordered;
# 78
using std::acosh;
# 79
using std::asinh;
# 80
using std::atanh;
# 81
using std::cbrt;
# 82
using std::copysign;
# 83
using std::erf;
# 84
using std::erfc;
# 85
using std::exp2;
# 86
using std::expm1;
# 87
using std::fdim;
# 88
using std::fma;
# 89
using std::fmax;
# 90
using std::fmin;
# 91
using std::hypot;
# 92
using std::ilogb;
# 93
using std::lgamma;
# 94
using std::llrint;
# 95
using std::llround;
# 96
using std::log1p;
# 97
using std::log2;
# 98
using std::logb;
# 99
using std::lrint;
# 100
using std::lround;
# 101
using std::nearbyint;
# 102
using std::nextafter;
# 103
using std::nexttoward;
# 104
using std::remainder;
# 105
using std::remquo;
# 106
using std::rint;
# 107
using std::round;
# 108
using std::scalbln;
# 109
using std::scalbn;
# 110
using std::tgamma;
# 111
using std::trunc;
# 121 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/cstdlib" 3
extern "C++" {
# 123
namespace std __attribute((__visibility__("default"))) { 
# 127
using ::div_t;
# 128
using ::ldiv_t;
# 130
using ::abort;
# 132
using ::aligned_alloc;
# 134
using ::atexit;
# 137
using ::at_quick_exit;
# 140
using ::atof;
# 141
using ::atoi;
# 142
using ::atol;
# 143
using ::bsearch;
# 144
using ::calloc;
# 145
using ::div;
# 146
using ::exit;
# 147
using ::free;
# 148
using ::getenv;
# 149
using ::labs;
# 150
using ::ldiv;
# 151
using ::malloc;
# 153
using ::mblen;
# 154
using ::mbstowcs;
# 155
using ::mbtowc;
# 157
using ::qsort;
# 160
using ::quick_exit;
# 163
using ::rand;
# 164
using ::realloc;
# 165
using ::srand;
# 166
using ::strtod;
# 167
using ::strtol;
# 168
using ::strtoul;
# 169
using ::system;
# 171
using ::wcstombs;
# 172
using ::wctomb;
# 177
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); } 
# 182
}
# 195 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/cstdlib" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 200
using ::lldiv_t;
# 206
using ::_Exit;
# 210
using ::llabs;
# 213
inline lldiv_t div(long long __n, long long __d) 
# 214
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; } 
# 216
using ::lldiv;
# 227 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/cstdlib" 3
using ::atoll;
# 228
using ::strtoll;
# 229
using ::strtoull;
# 231
using ::strtof;
# 232
using ::strtold;
# 235
}
# 237
namespace std { 
# 240
using __gnu_cxx::lldiv_t;
# 242
using __gnu_cxx::_Exit;
# 244
using __gnu_cxx::llabs;
# 245
using __gnu_cxx::div;
# 246
using __gnu_cxx::lldiv;
# 248
using __gnu_cxx::atoll;
# 249
using __gnu_cxx::strtof;
# 250
using __gnu_cxx::strtoll;
# 251
using __gnu_cxx::strtoull;
# 252
using __gnu_cxx::strtold;
# 253
}
# 257
}
# 38 "/pawsey/centos7.6/devel/gcc/4.8.5/gcc/11.1.0/include/c++/11.1.0/stdlib.h" 3
using std::abort;
# 39
using std::atexit;
# 40
using std::exit;
# 43
using std::at_quick_exit;
# 46
using std::quick_exit;
# 54
using std::abs;
# 55
using std::atof;
# 56
using std::atoi;
# 57
using std::atol;
# 58
using std::bsearch;
# 59
using std::calloc;
# 60
using std::div;
# 61
using std::free;
# 62
using std::getenv;
# 63
using std::labs;
# 64
using std::ldiv;
# 65
using std::malloc;
# 67
using std::mblen;
# 68
using std::mbstowcs;
# 69
using std::mbtowc;
# 71
using std::qsort;
# 72
using std::rand;
# 73
using std::realloc;
# 74
using std::srand;
# 75
using std::strtod;
# 76
using std::strtol;
# 77
using std::strtoul;
# 78
using std::system;
# 80
using std::wcstombs;
# 81
using std::wctomb;
# 9029 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 9030
constexpr bool signbit(float x); 
# 9031
constexpr bool signbit(double x); 
# 9032
constexpr bool signbit(long double x); 
# 9033
constexpr bool isfinite(float x); 
# 9034
constexpr bool isfinite(double x); 
# 9035
constexpr bool isfinite(long double x); 
# 9036
constexpr bool isnan(float x); 
# 9039
extern "C" int isnan(double x) throw(); 
# 9043
constexpr bool isnan(long double x); 
# 9044
constexpr bool isinf(float x); 
# 9047
extern "C" int isinf(double x) throw(); 
# 9051
constexpr bool isinf(long double x); 
# 9052
}
# 9205 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 9207
template< class T> extern T __pow_helper(T, int); 
# 9208
template< class T> extern T __cmath_power(T, unsigned); 
# 9209
}
# 9211
using std::abs;
# 9212
using std::fabs;
# 9213
using std::ceil;
# 9214
using std::floor;
# 9215
using std::sqrt;
# 9217
using std::pow;
# 9219
using std::log;
# 9220
using std::log10;
# 9221
using std::fmod;
# 9222
using std::modf;
# 9223
using std::exp;
# 9224
using std::frexp;
# 9225
using std::ldexp;
# 9226
using std::asin;
# 9227
using std::sin;
# 9228
using std::sinh;
# 9229
using std::acos;
# 9230
using std::cos;
# 9231
using std::cosh;
# 9232
using std::atan;
# 9233
using std::atan2;
# 9234
using std::tan;
# 9235
using std::tanh;
# 9606 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 9615 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline long long abs(long long); 
# 9625 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline long abs(long); 
# 9626
extern constexpr float abs(float); 
# 9627
extern constexpr double abs(double); 
# 9628
extern constexpr float fabs(float); 
# 9629
extern constexpr float ceil(float); 
# 9630
extern constexpr float floor(float); 
# 9631
extern constexpr float sqrt(float); 
# 9632
extern constexpr float pow(float, float); 
# 9637
template< class _Tp, class _Up> extern constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type pow(_Tp, _Up); 
# 9647
extern constexpr float log(float); 
# 9648
extern constexpr float log10(float); 
# 9649
extern constexpr float fmod(float, float); 
# 9650
extern inline float modf(float, float *); 
# 9651
extern constexpr float exp(float); 
# 9652
extern inline float frexp(float, int *); 
# 9653
extern constexpr float ldexp(float, int); 
# 9654
extern constexpr float asin(float); 
# 9655
extern constexpr float sin(float); 
# 9656
extern constexpr float sinh(float); 
# 9657
extern constexpr float acos(float); 
# 9658
extern constexpr float cos(float); 
# 9659
extern constexpr float cosh(float); 
# 9660
extern constexpr float atan(float); 
# 9661
extern constexpr float atan2(float, float); 
# 9662
extern constexpr float tan(float); 
# 9663
extern constexpr float tanh(float); 
# 9742 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 9848 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 9849
constexpr float logb(float a); 
# 9850
constexpr int ilogb(float a); 
# 9851
constexpr float scalbn(float a, int b); 
# 9852
constexpr float scalbln(float a, long b); 
# 9853
constexpr float exp2(float a); 
# 9854
constexpr float expm1(float a); 
# 9855
constexpr float log2(float a); 
# 9856
constexpr float log1p(float a); 
# 9857
constexpr float acosh(float a); 
# 9858
constexpr float asinh(float a); 
# 9859
constexpr float atanh(float a); 
# 9860
constexpr float hypot(float a, float b); 
# 9861
constexpr float cbrt(float a); 
# 9862
constexpr float erf(float a); 
# 9863
constexpr float erfc(float a); 
# 9864
constexpr float lgamma(float a); 
# 9865
constexpr float tgamma(float a); 
# 9866
constexpr float copysign(float a, float b); 
# 9867
constexpr float nextafter(float a, float b); 
# 9868
constexpr float remainder(float a, float b); 
# 9869
inline float remquo(float a, float b, int * quo); 
# 9870
constexpr float round(float a); 
# 9871
constexpr long lround(float a); 
# 9872
constexpr long long llround(float a); 
# 9873
constexpr float trunc(float a); 
# 9874
constexpr float rint(float a); 
# 9875
constexpr long lrint(float a); 
# 9876
constexpr long long llrint(float a); 
# 9877
constexpr float nearbyint(float a); 
# 9878
constexpr float fdim(float a, float b); 
# 9879
constexpr float fma(float a, float b, float c); 
# 9880
constexpr float fmax(float a, float b); 
# 9881
constexpr float fmin(float a, float b); 
# 9882
}
# 9987 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline float exp10(const float a); 
# 9989
static inline float rsqrt(const float a); 
# 9991
static inline float rcbrt(const float a); 
# 9993
static inline float sinpi(const float a); 
# 9995
static inline float cospi(const float a); 
# 9997
static inline void sincospi(const float a, float *const sptr, float *const cptr); 
# 9999
static inline void sincos(const float a, float *const sptr, float *const cptr); 
# 10001
static inline float j0(const float a); 
# 10003
static inline float j1(const float a); 
# 10005
static inline float jn(const int n, const float a); 
# 10007
static inline float y0(const float a); 
# 10009
static inline float y1(const float a); 
# 10011
static inline float yn(const int n, const float a); 
# 10013
__attribute__((unused)) static inline float cyl_bessel_i0(const float a); 
# 10015
__attribute__((unused)) static inline float cyl_bessel_i1(const float a); 
# 10017
static inline float erfinv(const float a); 
# 10019
static inline float erfcinv(const float a); 
# 10021
static inline float normcdfinv(const float a); 
# 10023
static inline float normcdf(const float a); 
# 10025
static inline float erfcx(const float a); 
# 10027
static inline double copysign(const double a, const float b); 
# 10029
static inline double copysign(const float a, const double b); 
# 10037
static inline unsigned min(const unsigned a, const unsigned b); 
# 10045
static inline unsigned min(const int a, const unsigned b); 
# 10053
static inline unsigned min(const unsigned a, const int b); 
# 10061
static inline long min(const long a, const long b); 
# 10069
static inline unsigned long min(const unsigned long a, const unsigned long b); 
# 10077
static inline unsigned long min(const long a, const unsigned long b); 
# 10085
static inline unsigned long min(const unsigned long a, const long b); 
# 10093
static inline long long min(const long long a, const long long b); 
# 10101
static inline unsigned long long min(const unsigned long long a, const unsigned long long b); 
# 10109
static inline unsigned long long min(const long long a, const unsigned long long b); 
# 10117
static inline unsigned long long min(const unsigned long long a, const long long b); 
# 10128 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline float min(const float a, const float b); 
# 10139 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline double min(const double a, const double b); 
# 10149 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline double min(const float a, const double b); 
# 10159 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline double min(const double a, const float b); 
# 10167
static inline unsigned max(const unsigned a, const unsigned b); 
# 10175
static inline unsigned max(const int a, const unsigned b); 
# 10183
static inline unsigned max(const unsigned a, const int b); 
# 10191
static inline long max(const long a, const long b); 
# 10199
static inline unsigned long max(const unsigned long a, const unsigned long b); 
# 10207
static inline unsigned long max(const long a, const unsigned long b); 
# 10215
static inline unsigned long max(const unsigned long a, const long b); 
# 10223
static inline long long max(const long long a, const long long b); 
# 10231
static inline unsigned long long max(const unsigned long long a, const unsigned long long b); 
# 10239
static inline unsigned long long max(const long long a, const unsigned long long b); 
# 10247
static inline unsigned long long max(const unsigned long long a, const long long b); 
# 10258 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline float max(const float a, const float b); 
# 10269 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline double max(const double a, const double b); 
# 10279 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline double max(const float a, const double b); 
# 10289 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline double max(const double a, const float b); 
# 10300 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern "C" {
# 10301
__attribute__((unused)) inline void *__nv_aligned_device_malloc(size_t size, size_t align) 
# 10302
{int volatile ___ = 1;(void)size;(void)align;
# 10305
::exit(___);}
#if 0
# 10302
{ 
# 10303
__attribute__((unused)) void *__nv_aligned_device_malloc_impl(size_t, size_t); 
# 10304
return __nv_aligned_device_malloc_impl(size, align); 
# 10305
} 
#endif
# 10306 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 758 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
static inline float exp10(const float a) 
# 759
{ 
# 760
return exp10f(a); 
# 761
} 
# 763
static inline float rsqrt(const float a) 
# 764
{ 
# 765
return rsqrtf(a); 
# 766
} 
# 768
static inline float rcbrt(const float a) 
# 769
{ 
# 770
return rcbrtf(a); 
# 771
} 
# 773
static inline float sinpi(const float a) 
# 774
{ 
# 775
return sinpif(a); 
# 776
} 
# 778
static inline float cospi(const float a) 
# 779
{ 
# 780
return cospif(a); 
# 781
} 
# 783
static inline void sincospi(const float a, float *const sptr, float *const cptr) 
# 784
{ 
# 785
sincospif(a, sptr, cptr); 
# 786
} 
# 788
static inline void sincos(const float a, float *const sptr, float *const cptr) 
# 789
{ 
# 790
sincosf(a, sptr, cptr); 
# 791
} 
# 793
static inline float j0(const float a) 
# 794
{ 
# 795
return j0f(a); 
# 796
} 
# 798
static inline float j1(const float a) 
# 799
{ 
# 800
return j1f(a); 
# 801
} 
# 803
static inline float jn(const int n, const float a) 
# 804
{ 
# 805
return jnf(n, a); 
# 806
} 
# 808
static inline float y0(const float a) 
# 809
{ 
# 810
return y0f(a); 
# 811
} 
# 813
static inline float y1(const float a) 
# 814
{ 
# 815
return y1f(a); 
# 816
} 
# 818
static inline float yn(const int n, const float a) 
# 819
{ 
# 820
return ynf(n, a); 
# 821
} 
# 823
__attribute__((unused)) static inline float cyl_bessel_i0(const float a) 
# 824
{int volatile ___ = 1;(void)a;
# 826
::exit(___);}
#if 0
# 824
{ 
# 825
return cyl_bessel_i0f(a); 
# 826
} 
#endif
# 828 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
__attribute__((unused)) static inline float cyl_bessel_i1(const float a) 
# 829
{int volatile ___ = 1;(void)a;
# 831
::exit(___);}
#if 0
# 829
{ 
# 830
return cyl_bessel_i1f(a); 
# 831
} 
#endif
# 833 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
static inline float erfinv(const float a) 
# 834
{ 
# 835
return erfinvf(a); 
# 836
} 
# 838
static inline float erfcinv(const float a) 
# 839
{ 
# 840
return erfcinvf(a); 
# 841
} 
# 843
static inline float normcdfinv(const float a) 
# 844
{ 
# 845
return normcdfinvf(a); 
# 846
} 
# 848
static inline float normcdf(const float a) 
# 849
{ 
# 850
return normcdff(a); 
# 851
} 
# 853
static inline float erfcx(const float a) 
# 854
{ 
# 855
return erfcxf(a); 
# 856
} 
# 858
static inline double copysign(const double a, const float b) 
# 859
{ 
# 860
return copysign(a, static_cast< double>(b)); 
# 861
} 
# 863
static inline double copysign(const float a, const double b) 
# 864
{ 
# 865
return copysign(static_cast< double>(a), b); 
# 866
} 
# 868
static inline unsigned min(const unsigned a, const unsigned b) 
# 869
{ 
# 870
return umin(a, b); 
# 871
} 
# 873
static inline unsigned min(const int a, const unsigned b) 
# 874
{ 
# 875
return umin(static_cast< unsigned>(a), b); 
# 876
} 
# 878
static inline unsigned min(const unsigned a, const int b) 
# 879
{ 
# 880
return umin(a, static_cast< unsigned>(b)); 
# 881
} 
# 883
static inline long min(const long a, const long b) 
# 884
{ 
# 885
long retval; 
# 891
if (sizeof(long) == sizeof(int)) { 
# 895
retval = (static_cast< long>(min(static_cast< int>(a), static_cast< int>(b)))); 
# 896
} else { 
# 897
retval = (static_cast< long>(llmin(static_cast< long long>(a), static_cast< long long>(b)))); 
# 898
}  
# 899
return retval; 
# 900
} 
# 902
static inline unsigned long min(const unsigned long a, const unsigned long b) 
# 903
{ 
# 904
unsigned long retval; 
# 908
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 912
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 913
} else { 
# 914
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 915
}  
# 916
return retval; 
# 917
} 
# 919
static inline unsigned long min(const long a, const unsigned long b) 
# 920
{ 
# 921
unsigned long retval; 
# 925
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 929
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 930
} else { 
# 931
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 932
}  
# 933
return retval; 
# 934
} 
# 936
static inline unsigned long min(const unsigned long a, const long b) 
# 937
{ 
# 938
unsigned long retval; 
# 942
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 946
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 947
} else { 
# 948
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 949
}  
# 950
return retval; 
# 951
} 
# 953
static inline long long min(const long long a, const long long b) 
# 954
{ 
# 955
return llmin(a, b); 
# 956
} 
# 958
static inline unsigned long long min(const unsigned long long a, const unsigned long long b) 
# 959
{ 
# 960
return ullmin(a, b); 
# 961
} 
# 963
static inline unsigned long long min(const long long a, const unsigned long long b) 
# 964
{ 
# 965
return ullmin(static_cast< unsigned long long>(a), b); 
# 966
} 
# 968
static inline unsigned long long min(const unsigned long long a, const long long b) 
# 969
{ 
# 970
return ullmin(a, static_cast< unsigned long long>(b)); 
# 971
} 
# 973
static inline float min(const float a, const float b) 
# 974
{ 
# 975
return fminf(a, b); 
# 976
} 
# 978
static inline double min(const double a, const double b) 
# 979
{ 
# 980
return fmin(a, b); 
# 981
} 
# 983
static inline double min(const float a, const double b) 
# 984
{ 
# 985
return fmin(static_cast< double>(a), b); 
# 986
} 
# 988
static inline double min(const double a, const float b) 
# 989
{ 
# 990
return fmin(a, static_cast< double>(b)); 
# 991
} 
# 993
static inline unsigned max(const unsigned a, const unsigned b) 
# 994
{ 
# 995
return umax(a, b); 
# 996
} 
# 998
static inline unsigned max(const int a, const unsigned b) 
# 999
{ 
# 1000
return umax(static_cast< unsigned>(a), b); 
# 1001
} 
# 1003
static inline unsigned max(const unsigned a, const int b) 
# 1004
{ 
# 1005
return umax(a, static_cast< unsigned>(b)); 
# 1006
} 
# 1008
static inline long max(const long a, const long b) 
# 1009
{ 
# 1010
long retval; 
# 1015
if (sizeof(long) == sizeof(int)) { 
# 1019
retval = (static_cast< long>(max(static_cast< int>(a), static_cast< int>(b)))); 
# 1020
} else { 
# 1021
retval = (static_cast< long>(llmax(static_cast< long long>(a), static_cast< long long>(b)))); 
# 1022
}  
# 1023
return retval; 
# 1024
} 
# 1026
static inline unsigned long max(const unsigned long a, const unsigned long b) 
# 1027
{ 
# 1028
unsigned long retval; 
# 1032
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1036
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1037
} else { 
# 1038
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1039
}  
# 1040
return retval; 
# 1041
} 
# 1043
static inline unsigned long max(const long a, const unsigned long b) 
# 1044
{ 
# 1045
unsigned long retval; 
# 1049
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1053
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1054
} else { 
# 1055
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1056
}  
# 1057
return retval; 
# 1058
} 
# 1060
static inline unsigned long max(const unsigned long a, const long b) 
# 1061
{ 
# 1062
unsigned long retval; 
# 1066
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1070
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
# 1071
} else { 
# 1072
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned long long>(a), static_cast< unsigned long long>(b)))); 
# 1073
}  
# 1074
return retval; 
# 1075
} 
# 1077
static inline long long max(const long long a, const long long b) 
# 1078
{ 
# 1079
return llmax(a, b); 
# 1080
} 
# 1082
static inline unsigned long long max(const unsigned long long a, const unsigned long long b) 
# 1083
{ 
# 1084
return ullmax(a, b); 
# 1085
} 
# 1087
static inline unsigned long long max(const long long a, const unsigned long long b) 
# 1088
{ 
# 1089
return ullmax(static_cast< unsigned long long>(a), b); 
# 1090
} 
# 1092
static inline unsigned long long max(const unsigned long long a, const long long b) 
# 1093
{ 
# 1094
return ullmax(a, static_cast< unsigned long long>(b)); 
# 1095
} 
# 1097
static inline float max(const float a, const float b) 
# 1098
{ 
# 1099
return fmaxf(a, b); 
# 1100
} 
# 1102
static inline double max(const double a, const double b) 
# 1103
{ 
# 1104
return fmax(a, b); 
# 1105
} 
# 1107
static inline double max(const float a, const double b) 
# 1108
{ 
# 1109
return fmax(static_cast< double>(a), b); 
# 1110
} 
# 1112
static inline double max(const double a, const float b) 
# 1113
{ 
# 1114
return fmax(a, static_cast< double>(b)); 
# 1115
} 
# 1126 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
inline int min(const int a, const int b) 
# 1127
{ 
# 1128
return (a < b) ? a : b; 
# 1129
} 
# 1131
inline unsigned umin(const unsigned a, const unsigned b) 
# 1132
{ 
# 1133
return (a < b) ? a : b; 
# 1134
} 
# 1136
inline long long llmin(const long long a, const long long b) 
# 1137
{ 
# 1138
return (a < b) ? a : b; 
# 1139
} 
# 1141
inline unsigned long long ullmin(const unsigned long long a, const unsigned long long 
# 1142
b) 
# 1143
{ 
# 1144
return (a < b) ? a : b; 
# 1145
} 
# 1147
inline int max(const int a, const int b) 
# 1148
{ 
# 1149
return (a > b) ? a : b; 
# 1150
} 
# 1152
inline unsigned umax(const unsigned a, const unsigned b) 
# 1153
{ 
# 1154
return (a > b) ? a : b; 
# 1155
} 
# 1157
inline long long llmax(const long long a, const long long b) 
# 1158
{ 
# 1159
return (a > b) ? a : b; 
# 1160
} 
# 1162
inline unsigned long long ullmax(const unsigned long long a, const unsigned long long 
# 1163
b) 
# 1164
{ 
# 1165
return (a > b) ? a : b; 
# 1166
} 
# 74 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_surface_types.h"
template< class T, int dim = 1> 
# 75
struct surface : public surfaceReference { 
# 78
surface() 
# 79
{ 
# 80
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 81
} 
# 83
surface(cudaChannelFormatDesc desc) 
# 84
{ 
# 85
(channelDesc) = desc; 
# 86
} 
# 88
}; 
# 90
template< int dim> 
# 91
struct surface< void, dim>  : public surfaceReference { 
# 94
surface() 
# 95
{ 
# 96
(channelDesc) = cudaCreateChannelDesc< void> (); 
# 97
} 
# 99
}; 
# 74 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_texture_types.h"
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
# 75
struct texture : public textureReference { 
# 78
texture(int norm = 0, cudaTextureFilterMode 
# 79
fMode = cudaFilterModePoint, cudaTextureAddressMode 
# 80
aMode = cudaAddressModeClamp) 
# 81
{ 
# 82
(normalized) = norm; 
# 83
(filterMode) = fMode; 
# 84
((addressMode)[0]) = aMode; 
# 85
((addressMode)[1]) = aMode; 
# 86
((addressMode)[2]) = aMode; 
# 87
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 88
(sRGB) = 0; 
# 89
} 
# 91
texture(int norm, cudaTextureFilterMode 
# 92
fMode, cudaTextureAddressMode 
# 93
aMode, cudaChannelFormatDesc 
# 94
desc) 
# 95
{ 
# 96
(normalized) = norm; 
# 97
(filterMode) = fMode; 
# 98
((addressMode)[0]) = aMode; 
# 99
((addressMode)[1]) = aMode; 
# 100
((addressMode)[2]) = aMode; 
# 101
(channelDesc) = desc; 
# 102
(sRGB) = 0; 
# 103
} 
# 105
}; 
# 89 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
extern "C" {
# 3215 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
}
# 3223
__attribute__((unused)) static inline int mulhi(const int a, const int b); 
# 3225
__attribute__((unused)) static inline unsigned mulhi(const unsigned a, const unsigned b); 
# 3227
__attribute__((unused)) static inline unsigned mulhi(const int a, const unsigned b); 
# 3229
__attribute__((unused)) static inline unsigned mulhi(const unsigned a, const int b); 
# 3231
__attribute__((unused)) static inline long long mul64hi(const long long a, const long long b); 
# 3233
__attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const unsigned long long b); 
# 3235
__attribute__((unused)) static inline unsigned long long mul64hi(const long long a, const unsigned long long b); 
# 3237
__attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const long long b); 
# 3239
__attribute__((unused)) static inline int float_as_int(const float a); 
# 3241
__attribute__((unused)) static inline float int_as_float(const int a); 
# 3243
__attribute__((unused)) static inline unsigned float_as_uint(const float a); 
# 3245
__attribute__((unused)) static inline float uint_as_float(const unsigned a); 
# 3247
__attribute__((unused)) static inline float saturate(const float a); 
# 3249
__attribute__((unused)) static inline int mul24(const int a, const int b); 
# 3251
__attribute__((unused)) static inline unsigned umul24(const unsigned a, const unsigned b); 
# 3253
__attribute__((unused)) static inline int float2int(const float a, const cudaRoundMode mode = cudaRoundZero); 
# 3255
__attribute__((unused)) static inline unsigned float2uint(const float a, const cudaRoundMode mode = cudaRoundZero); 
# 3257
__attribute__((unused)) static inline float int2float(const int a, const cudaRoundMode mode = cudaRoundNearest); 
# 3259
__attribute__((unused)) static inline float uint2float(const unsigned a, const cudaRoundMode mode = cudaRoundNearest); 
# 90 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int mulhi(const int a, const int b) 
# 91
{int volatile ___ = 1;(void)a;(void)b;
# 93
::exit(___);}
#if 0
# 91
{ 
# 92
return __mulhi(a, b); 
# 93
} 
#endif
# 95 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(const unsigned a, const unsigned b) 
# 96
{int volatile ___ = 1;(void)a;(void)b;
# 98
::exit(___);}
#if 0
# 96
{ 
# 97
return __umulhi(a, b); 
# 98
} 
#endif
# 100 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(const int a, const unsigned b) 
# 101
{int volatile ___ = 1;(void)a;(void)b;
# 103
::exit(___);}
#if 0
# 101
{ 
# 102
return __umulhi(static_cast< unsigned>(a), b); 
# 103
} 
#endif
# 105 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(const unsigned a, const int b) 
# 106
{int volatile ___ = 1;(void)a;(void)b;
# 108
::exit(___);}
#if 0
# 106
{ 
# 107
return __umulhi(a, static_cast< unsigned>(b)); 
# 108
} 
#endif
# 110 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline long long mul64hi(const long long a, const long long b) 
# 111
{int volatile ___ = 1;(void)a;(void)b;
# 113
::exit(___);}
#if 0
# 111
{ 
# 112
return __mul64hi(a, b); 
# 113
} 
#endif
# 115 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const unsigned long long b) 
# 116
{int volatile ___ = 1;(void)a;(void)b;
# 118
::exit(___);}
#if 0
# 116
{ 
# 117
return __umul64hi(a, b); 
# 118
} 
#endif
# 120 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(const long long a, const unsigned long long b) 
# 121
{int volatile ___ = 1;(void)a;(void)b;
# 123
::exit(___);}
#if 0
# 121
{ 
# 122
return __umul64hi(static_cast< unsigned long long>(a), b); 
# 123
} 
#endif
# 125 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(const unsigned long long a, const long long b) 
# 126
{int volatile ___ = 1;(void)a;(void)b;
# 128
::exit(___);}
#if 0
# 126
{ 
# 127
return __umul64hi(a, static_cast< unsigned long long>(b)); 
# 128
} 
#endif
# 130 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int float_as_int(const float a) 
# 131
{int volatile ___ = 1;(void)a;
# 133
::exit(___);}
#if 0
# 131
{ 
# 132
return __float_as_int(a); 
# 133
} 
#endif
# 135 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float int_as_float(const int a) 
# 136
{int volatile ___ = 1;(void)a;
# 138
::exit(___);}
#if 0
# 136
{ 
# 137
return __int_as_float(a); 
# 138
} 
#endif
# 140 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned float_as_uint(const float a) 
# 141
{int volatile ___ = 1;(void)a;
# 143
::exit(___);}
#if 0
# 141
{ 
# 142
return __float_as_uint(a); 
# 143
} 
#endif
# 145 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float uint_as_float(const unsigned a) 
# 146
{int volatile ___ = 1;(void)a;
# 148
::exit(___);}
#if 0
# 146
{ 
# 147
return __uint_as_float(a); 
# 148
} 
#endif
# 149 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float saturate(const float a) 
# 150
{int volatile ___ = 1;(void)a;
# 152
::exit(___);}
#if 0
# 150
{ 
# 151
return __saturatef(a); 
# 152
} 
#endif
# 154 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int mul24(const int a, const int b) 
# 155
{int volatile ___ = 1;(void)a;(void)b;
# 157
::exit(___);}
#if 0
# 155
{ 
# 156
return __mul24(a, b); 
# 157
} 
#endif
# 159 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned umul24(const unsigned a, const unsigned b) 
# 160
{int volatile ___ = 1;(void)a;(void)b;
# 162
::exit(___);}
#if 0
# 160
{ 
# 161
return __umul24(a, b); 
# 162
} 
#endif
# 164 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int float2int(const float a, const cudaRoundMode mode) 
# 165
{int volatile ___ = 1;(void)a;(void)mode;
# 170
::exit(___);}
#if 0
# 165
{ 
# 166
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
# 170
} 
#endif
# 172 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned float2uint(const float a, const cudaRoundMode mode) 
# 173
{int volatile ___ = 1;(void)a;(void)mode;
# 178
::exit(___);}
#if 0
# 173
{ 
# 174
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
# 178
} 
#endif
# 180 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float int2float(const int a, const cudaRoundMode mode) 
# 181
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 181
{ 
# 182
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
# 186
} 
#endif
# 188 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float uint2float(const unsigned a, const cudaRoundMode mode) 
# 189
{int volatile ___ = 1;(void)a;(void)mode;
# 194
::exit(___);}
#if 0
# 189
{ 
# 190
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
# 194
} 
#endif
# 106 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 106
{ } 
#endif
# 108 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 108
{ } 
#endif
# 110 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 120 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 120
{ } 
#endif
# 122 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 122
{ } 
#endif
# 124 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 126
{ } 
#endif
# 128 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 128
{ } 
#endif
# 130 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 130
{ } 
#endif
# 132 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 132
{ } 
#endif
# 134 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 134
{ } 
#endif
# 136 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 136
{ } 
#endif
# 138 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 138
{ } 
#endif
# 140 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 140
{ } 
#endif
# 142 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 142
{ } 
#endif
# 144 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 144
{ } 
#endif
# 146 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 146
{ } 
#endif
# 171 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
extern "C" {
# 180
}
# 189 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 189
{ } 
#endif
# 191 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 191
{ } 
#endif
# 193 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 193
{ } 
#endif
# 195 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute((deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 195
{ } 
#endif
# 197 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute((deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 197
{ } 
#endif
# 87 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern "C" {
# 1139 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
}
# 1147
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode); 
# 1149
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1151
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1153
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1155
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
# 1157
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
# 1159
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
# 1161
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
# 1163
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1165
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1167
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
# 1169
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 1171
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
# 93 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode) 
# 94
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;
# 99
::exit(___);}
#if 0
# 94
{ 
# 95
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
# 99
} 
#endif
# 101 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode) 
# 102
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 107
::exit(___);}
#if 0
# 102
{ 
# 103
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
# 107
} 
#endif
# 109 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode) 
# 110
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 115
::exit(___);}
#if 0
# 110
{ 
# 111
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
# 115
} 
#endif
# 117 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode) 
# 118
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 123
::exit(___);}
#if 0
# 118
{ 
# 119
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
# 123
} 
#endif
# 125 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode) 
# 126
{int volatile ___ = 1;(void)a;(void)mode;
# 131
::exit(___);}
#if 0
# 126
{ 
# 127
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
# 131
} 
#endif
# 133 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode) 
# 134
{int volatile ___ = 1;(void)a;(void)mode;
# 139
::exit(___);}
#if 0
# 134
{ 
# 135
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
# 139
} 
#endif
# 141 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode) 
# 142
{int volatile ___ = 1;(void)a;(void)mode;
# 147
::exit(___);}
#if 0
# 142
{ 
# 143
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
# 147
} 
#endif
# 149 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode) 
# 150
{int volatile ___ = 1;(void)a;(void)mode;
# 155
::exit(___);}
#if 0
# 150
{ 
# 151
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
# 155
} 
#endif
# 157 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode) 
# 158
{int volatile ___ = 1;(void)a;(void)mode;
# 163
::exit(___);}
#if 0
# 158
{ 
# 159
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
# 163
} 
#endif
# 165 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode) 
# 166
{int volatile ___ = 1;(void)a;(void)mode;
# 171
::exit(___);}
#if 0
# 166
{ 
# 167
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
# 171
} 
#endif
# 173 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode) 
# 174
{int volatile ___ = 1;(void)a;(void)mode;
# 176
::exit(___);}
#if 0
# 174
{ 
# 175
return (double)a; 
# 176
} 
#endif
# 178 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode) 
# 179
{int volatile ___ = 1;(void)a;(void)mode;
# 181
::exit(___);}
#if 0
# 179
{ 
# 180
return (double)a; 
# 181
} 
#endif
# 183 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode) 
# 184
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 184
{ 
# 185
return (double)a; 
# 186
} 
#endif
# 89 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 89
{ } 
#endif
# 100 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 102
{ } 
#endif
# 104 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 104
{ } 
#endif
# 106 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 106
{ } 
#endif
# 108 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 108
{ } 
#endif
# 110 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 303 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 303
{ } 
#endif
# 306 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 306
{ } 
#endif
# 309 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 309
{ } 
#endif
# 312 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 312
{ } 
#endif
# 315 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 315
{ } 
#endif
# 318 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 318
{ } 
#endif
# 321 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 321
{ } 
#endif
# 324 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 324
{ } 
#endif
# 327 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 327
{ } 
#endif
# 330 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 330
{ } 
#endif
# 333 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 333
{ } 
#endif
# 336 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 336
{ } 
#endif
# 339 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 339
{ } 
#endif
# 342 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 342
{ } 
#endif
# 345 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 345
{ } 
#endif
# 348 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 348
{ } 
#endif
# 351 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 351
{ } 
#endif
# 354 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 354
{ } 
#endif
# 357 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 357
{ } 
#endif
# 360 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 360
{ } 
#endif
# 363 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 363
{ } 
#endif
# 366 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 366
{ } 
#endif
# 369 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 369
{ } 
#endif
# 372 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 372
{ } 
#endif
# 375 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 375
{ } 
#endif
# 378 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 378
{ } 
#endif
# 381 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 381
{ } 
#endif
# 384 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 384
{ } 
#endif
# 387 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 387
{ } 
#endif
# 390 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 390
{ } 
#endif
# 393 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 393
{ } 
#endif
# 396 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 396
{ } 
#endif
# 399 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 399
{ } 
#endif
# 402 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 402
{ } 
#endif
# 405 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 405
{ } 
#endif
# 408 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 408
{ } 
#endif
# 411 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 411
{ } 
#endif
# 414 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 414
{ } 
#endif
# 417 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 417
{ } 
#endif
# 420 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 420
{ } 
#endif
# 423 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 423
{ } 
#endif
# 426 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 426
{ } 
#endif
# 429 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 429
{ } 
#endif
# 432 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 432
{ } 
#endif
# 435 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 435
{ } 
#endif
# 438 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
# 439
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 439
{ } 
#endif
# 442 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
# 443
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 443
{ } 
#endif
# 446 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_block(unsigned long long *address, unsigned long long 
# 447
compare, unsigned long long 
# 448
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 448
{ } 
#endif
# 451 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_system(unsigned long long *address, unsigned long long 
# 452
compare, unsigned long long 
# 453
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 453
{ } 
#endif
# 456 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 456
{ } 
#endif
# 459 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 459
{ } 
#endif
# 462 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 462
{ } 
#endif
# 465 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 465
{ } 
#endif
# 468 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 468
{ } 
#endif
# 471 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 471
{ } 
#endif
# 474 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 474
{ } 
#endif
# 477 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 477
{ } 
#endif
# 480 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 480
{ } 
#endif
# 483 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 483
{ } 
#endif
# 486 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 486
{ } 
#endif
# 489 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 489
{ } 
#endif
# 492 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 492
{ } 
#endif
# 495 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 495
{ } 
#endif
# 498 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 498
{ } 
#endif
# 501 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 501
{ } 
#endif
# 504 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 504
{ } 
#endif
# 507 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 507
{ } 
#endif
# 510 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 510
{ } 
#endif
# 513 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 513
{ } 
#endif
# 516 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 516
{ } 
#endif
# 519 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 519
{ } 
#endif
# 522 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 522
{ } 
#endif
# 525 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 525
{ } 
#endif
# 90 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern "C" {
# 1503 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
}
# 1510
__attribute((deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning)."))) __attribute__((unused)) static inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1510
{ } 
#endif
# 1512 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1512
{ } 
#endif
# 1514 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1514
{ } 
#endif
# 1516 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1516
{ } 
#endif
# 1521 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1521
{ } 
#endif
# 1522 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1522
{ } 
#endif
# 1523 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1523
{ } 
#endif
# 1524 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isLocal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1524
{ } 
#endif
# 1526 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_global(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1526
{ } 
#endif
# 1527 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_shared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1527
{ } 
#endif
# 1528 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1528
{ } 
#endif
# 1529 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline size_t __cvta_generic_to_local(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1529
{ } 
#endif
# 1531 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_global_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1531
{ } 
#endif
# 1532 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_shared_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1532
{ } 
#endif
# 1533 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1533
{ } 
#endif
# 1534 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline void *__cvta_local_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
# 1534
{ } 
#endif
# 102 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
# 110
{ } 
#endif
# 119 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 119
{ } 
#endif
# 120 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 120
{ } 
#endif
# 121 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 121
{ } 
#endif
# 122 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 122
{ } 
#endif
# 123 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 124
{ } 
#endif
# 125 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 125
{ } 
#endif
# 126 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 130
{ } 
#endif
# 133 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 137
{ } 
#endif
# 138 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 138
{ } 
#endif
# 139 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 144
{ } 
#endif
# 148 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl(unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long long __shfl(long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 149
{ } 
#endif
# 150 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_up(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 150
{ } 
#endif
# 151 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_up(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_down(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 152
{ } 
#endif
# 153 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_down(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_xor(long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 155
{ } 
#endif
# 156 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 156
{ } 
#endif
# 157 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 157
{ } 
#endif
# 158 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 158
{ } 
#endif
# 159 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 159
{ } 
#endif
# 162 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_sync(unsigned mask, long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_up_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_down_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 173
{ } 
#endif
# 177 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 184
{ } 
#endif
# 187 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 191
{ } 
#endif
# 192 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 192
{ } 
#endif
# 193 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 193
{ } 
#endif
# 194 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 194
{ } 
#endif
# 87 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 87
{ } 
#endif
# 88 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 88
{ } 
#endif
# 90 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 90
{ } 
#endif
# 91 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 101
{ } 
#endif
# 103 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 115
{ } 
#endif
# 116 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 116
{ } 
#endif
# 117 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 117
{ } 
#endif
# 118 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 118
{ } 
#endif
# 119 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 119
{ } 
#endif
# 123 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 137
{ } 
#endif
# 139 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 149
{ } 
#endif
# 151 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 152
{ } 
#endif
# 153 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 155
{ } 
#endif
# 159 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 159
{ } 
#endif
# 160 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 160
{ } 
#endif
# 162 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldca(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 173
{ } 
#endif
# 175 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldca(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 185
{ } 
#endif
# 187 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 191
{ } 
#endif
# 195 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 195
{ } 
#endif
# 196 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 196
{ } 
#endif
# 198 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 198
{ } 
#endif
# 199 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 199
{ } 
#endif
# 200 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 201
{ } 
#endif
# 202 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcs(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 202
{ } 
#endif
# 203 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 203
{ } 
#endif
# 204 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 205
{ } 
#endif
# 206 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 206
{ } 
#endif
# 207 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 207
{ } 
#endif
# 208 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 209
{ } 
#endif
# 211 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 211
{ } 
#endif
# 212 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 212
{ } 
#endif
# 213 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcs(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 214
{ } 
#endif
# 215 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 215
{ } 
#endif
# 216 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 216
{ } 
#endif
# 217 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 217
{ } 
#endif
# 218 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 218
{ } 
#endif
# 219 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 219
{ } 
#endif
# 220 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 220
{ } 
#endif
# 221 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 221
{ } 
#endif
# 223 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 223
{ } 
#endif
# 224 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 224
{ } 
#endif
# 225 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 225
{ } 
#endif
# 226 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 226
{ } 
#endif
# 227 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 227
{ } 
#endif
# 231 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldlu(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 231
{ } 
#endif
# 232 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldlu(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 232
{ } 
#endif
# 234 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldlu(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 234
{ } 
#endif
# 235 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldlu(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 235
{ } 
#endif
# 236 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldlu(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 236
{ } 
#endif
# 237 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldlu(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 237
{ } 
#endif
# 238 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldlu(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 238
{ } 
#endif
# 239 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldlu(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 239
{ } 
#endif
# 240 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldlu(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 240
{ } 
#endif
# 241 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldlu(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 241
{ } 
#endif
# 242 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldlu(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 242
{ } 
#endif
# 243 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldlu(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 243
{ } 
#endif
# 244 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldlu(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 244
{ } 
#endif
# 245 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldlu(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 245
{ } 
#endif
# 247 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldlu(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 247
{ } 
#endif
# 248 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldlu(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 248
{ } 
#endif
# 249 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldlu(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 249
{ } 
#endif
# 250 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldlu(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 250
{ } 
#endif
# 251 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldlu(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 251
{ } 
#endif
# 252 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldlu(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 252
{ } 
#endif
# 253 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldlu(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 253
{ } 
#endif
# 254 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldlu(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 254
{ } 
#endif
# 255 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldlu(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 255
{ } 
#endif
# 256 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldlu(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 256
{ } 
#endif
# 257 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldlu(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 257
{ } 
#endif
# 259 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldlu(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 259
{ } 
#endif
# 260 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldlu(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 260
{ } 
#endif
# 261 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldlu(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 261
{ } 
#endif
# 262 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldlu(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 262
{ } 
#endif
# 263 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldlu(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 263
{ } 
#endif
# 267 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcv(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 267
{ } 
#endif
# 268 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcv(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 268
{ } 
#endif
# 270 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcv(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 270
{ } 
#endif
# 271 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcv(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 271
{ } 
#endif
# 272 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcv(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 272
{ } 
#endif
# 273 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcv(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 273
{ } 
#endif
# 274 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcv(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 274
{ } 
#endif
# 275 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcv(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 275
{ } 
#endif
# 276 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcv(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 276
{ } 
#endif
# 277 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcv(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 277
{ } 
#endif
# 278 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcv(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 278
{ } 
#endif
# 279 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcv(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 279
{ } 
#endif
# 280 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcv(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 280
{ } 
#endif
# 281 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcv(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 281
{ } 
#endif
# 283 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcv(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 283
{ } 
#endif
# 284 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcv(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 284
{ } 
#endif
# 285 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcv(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 285
{ } 
#endif
# 286 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcv(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 286
{ } 
#endif
# 287 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcv(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 287
{ } 
#endif
# 288 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcv(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 288
{ } 
#endif
# 289 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcv(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 289
{ } 
#endif
# 290 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcv(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 290
{ } 
#endif
# 291 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcv(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 291
{ } 
#endif
# 292 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcv(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 292
{ } 
#endif
# 293 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcv(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 293
{ } 
#endif
# 295 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcv(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 295
{ } 
#endif
# 296 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcv(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 296
{ } 
#endif
# 297 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcv(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 297
{ } 
#endif
# 298 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcv(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 298
{ } 
#endif
# 299 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcv(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 299
{ } 
#endif
# 303 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 303
{ } 
#endif
# 304 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 304
{ } 
#endif
# 306 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 306
{ } 
#endif
# 307 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 307
{ } 
#endif
# 308 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 308
{ } 
#endif
# 309 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 309
{ } 
#endif
# 310 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 310
{ } 
#endif
# 311 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 311
{ } 
#endif
# 312 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 312
{ } 
#endif
# 313 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 313
{ } 
#endif
# 314 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 314
{ } 
#endif
# 315 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 315
{ } 
#endif
# 316 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 316
{ } 
#endif
# 317 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 317
{ } 
#endif
# 319 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 319
{ } 
#endif
# 320 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 320
{ } 
#endif
# 321 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 321
{ } 
#endif
# 322 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 322
{ } 
#endif
# 323 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 323
{ } 
#endif
# 324 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 324
{ } 
#endif
# 325 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 325
{ } 
#endif
# 326 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 326
{ } 
#endif
# 327 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 327
{ } 
#endif
# 328 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 328
{ } 
#endif
# 329 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 329
{ } 
#endif
# 331 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 331
{ } 
#endif
# 332 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 332
{ } 
#endif
# 333 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 333
{ } 
#endif
# 334 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 334
{ } 
#endif
# 335 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwb(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 335
{ } 
#endif
# 339 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 339
{ } 
#endif
# 340 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 340
{ } 
#endif
# 342 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 342
{ } 
#endif
# 343 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 343
{ } 
#endif
# 344 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 344
{ } 
#endif
# 345 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 345
{ } 
#endif
# 346 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 346
{ } 
#endif
# 347 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 347
{ } 
#endif
# 348 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 348
{ } 
#endif
# 349 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 349
{ } 
#endif
# 350 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 350
{ } 
#endif
# 351 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 351
{ } 
#endif
# 352 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 352
{ } 
#endif
# 353 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 353
{ } 
#endif
# 355 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 355
{ } 
#endif
# 356 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 356
{ } 
#endif
# 357 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 357
{ } 
#endif
# 358 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 358
{ } 
#endif
# 359 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 359
{ } 
#endif
# 360 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 360
{ } 
#endif
# 361 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 361
{ } 
#endif
# 362 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 362
{ } 
#endif
# 363 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 363
{ } 
#endif
# 364 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 364
{ } 
#endif
# 365 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 365
{ } 
#endif
# 367 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 367
{ } 
#endif
# 368 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 368
{ } 
#endif
# 369 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 369
{ } 
#endif
# 370 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 370
{ } 
#endif
# 371 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcg(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 371
{ } 
#endif
# 375 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 375
{ } 
#endif
# 376 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 376
{ } 
#endif
# 378 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 378
{ } 
#endif
# 379 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 379
{ } 
#endif
# 380 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 380
{ } 
#endif
# 381 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 381
{ } 
#endif
# 382 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 382
{ } 
#endif
# 383 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 383
{ } 
#endif
# 384 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 384
{ } 
#endif
# 385 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 385
{ } 
#endif
# 386 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 386
{ } 
#endif
# 387 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 387
{ } 
#endif
# 388 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 388
{ } 
#endif
# 389 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 389
{ } 
#endif
# 391 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 391
{ } 
#endif
# 392 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 392
{ } 
#endif
# 393 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 393
{ } 
#endif
# 394 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 394
{ } 
#endif
# 395 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 395
{ } 
#endif
# 396 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 396
{ } 
#endif
# 397 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 397
{ } 
#endif
# 398 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 398
{ } 
#endif
# 399 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 399
{ } 
#endif
# 400 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 400
{ } 
#endif
# 401 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 401
{ } 
#endif
# 403 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 403
{ } 
#endif
# 404 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 404
{ } 
#endif
# 405 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 405
{ } 
#endif
# 406 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 406
{ } 
#endif
# 407 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stcs(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 407
{ } 
#endif
# 411 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 411
{ } 
#endif
# 412 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 412
{ } 
#endif
# 414 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 414
{ } 
#endif
# 415 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 415
{ } 
#endif
# 416 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 416
{ } 
#endif
# 417 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 417
{ } 
#endif
# 418 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(long long *ptr, long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 418
{ } 
#endif
# 419 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 419
{ } 
#endif
# 420 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 420
{ } 
#endif
# 421 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 421
{ } 
#endif
# 422 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 422
{ } 
#endif
# 423 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 423
{ } 
#endif
# 424 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 424
{ } 
#endif
# 425 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 425
{ } 
#endif
# 427 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 427
{ } 
#endif
# 428 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 428
{ } 
#endif
# 429 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 429
{ } 
#endif
# 430 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(unsigned long long *ptr, unsigned long long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 430
{ } 
#endif
# 431 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 431
{ } 
#endif
# 432 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 432
{ } 
#endif
# 433 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 433
{ } 
#endif
# 434 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 434
{ } 
#endif
# 435 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 435
{ } 
#endif
# 436 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 436
{ } 
#endif
# 437 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 437
{ } 
#endif
# 439 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 439
{ } 
#endif
# 440 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 440
{ } 
#endif
# 441 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 441
{ } 
#endif
# 442 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 442
{ } 
#endif
# 443 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline void __stwt(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
# 443
{ } 
#endif
# 460 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 460
{ } 
#endif
# 472 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 472
{ } 
#endif
# 485 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 485
{ } 
#endif
# 497 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 497
{ } 
#endif
# 89 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 90
{ } 
#endif
# 92 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 93
{ } 
#endif
# 95 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 96
{ } 
#endif
# 98 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 99
{ } 
#endif
# 106 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 110
{ } 
#endif
# 93 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline void __nanosleep(unsigned ns) {int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
# 111
{ } 
#endif
# 113 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 113
{ } 
#endif
# 93 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_add_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_min_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_max_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 95
{ } 
#endif
# 97 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline int __reduce_add_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline int __reduce_min_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline int __reduce_max_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 101 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_and_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_or_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) static inline unsigned __reduce_xor_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 103
{ } 
#endif
# 106 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
extern "C" {
# 107
__attribute__((unused)) inline void *__nv_associate_access_property(const void *ptr, unsigned 
# 108
policy) {int volatile ___ = 1;(void)ptr;(void)policy;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_4(void *dst, const void *src, unsigned 
# 110
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_8(void *dst, const void *src, unsigned 
# 112
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
__attribute__((unused)) inline void __nv_memcpy_async_shared_global_16(void *dst, const void *src, unsigned 
# 114
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
# 114
{ } 
#endif
# 115 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
}
# 122 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 123
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 124
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;
# 128
::exit(___);}
#if 0
# 124
{ 
# 128
} 
#endif
# 130 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 131
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf1Dread(surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 132
{int volatile ___ = 1;(void)surf;(void)x;(void)mode;
# 138
::exit(___);}
#if 0
# 132
{ 
# 138
} 
#endif
# 140 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 141
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 142
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;
# 146
::exit(___);}
#if 0
# 142
{ 
# 146
} 
#endif
# 149 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 150
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 151
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 155
::exit(___);}
#if 0
# 151
{ 
# 155
} 
#endif
# 157 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 158
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf2Dread(surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 159
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;
# 165
::exit(___);}
#if 0
# 159
{ 
# 165
} 
#endif
# 167 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 168
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 169
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;
# 173
::exit(___);}
#if 0
# 169
{ 
# 173
} 
#endif
# 176 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 177
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 178
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 182
::exit(___);}
#if 0
# 178
{ 
# 182
} 
#endif
# 184 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 185
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf3Dread(surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 186
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 192
::exit(___);}
#if 0
# 186
{ 
# 192
} 
#endif
# 194 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 195
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 196
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 200
::exit(___);}
#if 0
# 196
{ 
# 200
} 
#endif
# 204 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 205
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 206
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 210
::exit(___);}
#if 0
# 206
{ 
# 210
} 
#endif
# 212 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 213
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf1DLayeredread(surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 214
{int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;
# 220
::exit(___);}
#if 0
# 214
{ 
# 220
} 
#endif
# 223 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 224
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 225
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;
# 229
::exit(___);}
#if 0
# 225
{ 
# 229
} 
#endif
# 232 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 233
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 234
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 238
::exit(___);}
#if 0
# 234
{ 
# 238
} 
#endif
# 240 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 241
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surf2DLayeredread(surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 242
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 248
::exit(___);}
#if 0
# 242
{ 
# 248
} 
#endif
# 251 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 252
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 253
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 257
::exit(___);}
#if 0
# 253
{ 
# 257
} 
#endif
# 260 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 261
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 262
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 266
::exit(___);}
#if 0
# 262
{ 
# 266
} 
#endif
# 268 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 269
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapread(surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 270
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 277
::exit(___);}
#if 0
# 270
{ 
# 277
} 
#endif
# 279 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 280
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 281
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 285
::exit(___);}
#if 0
# 281
{ 
# 285
} 
#endif
# 288 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 289
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 290
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 294
::exit(___);}
#if 0
# 290
{ 
# 294
} 
#endif
# 296 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 297
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapLayeredread(surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 298
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 304
::exit(___);}
#if 0
# 298
{ 
# 304
} 
#endif
# 306 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 307
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 308
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 312
::exit(___);}
#if 0
# 308
{ 
# 312
} 
#endif
# 315 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 316
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 317
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;
# 321
::exit(___);}
#if 0
# 317
{ 
# 321
} 
#endif
# 323 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 324
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 325
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;
# 329
::exit(___);}
#if 0
# 325
{ 
# 329
} 
#endif
# 333 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 334
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 335
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 339
::exit(___);}
#if 0
# 335
{ 
# 339
} 
#endif
# 341 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 342
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 343
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;
# 347
::exit(___);}
#if 0
# 343
{ 
# 347
} 
#endif
# 350 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 351
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 352
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 356
::exit(___);}
#if 0
# 352
{ 
# 356
} 
#endif
# 358 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 359
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 360
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 364
::exit(___);}
#if 0
# 360
{ 
# 364
} 
#endif
# 367 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 368
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 369
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 373
::exit(___);}
#if 0
# 369
{ 
# 373
} 
#endif
# 375 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 376
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 377
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;
# 381
::exit(___);}
#if 0
# 377
{ 
# 381
} 
#endif
# 384 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 385
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 386
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 390
::exit(___);}
#if 0
# 386
{ 
# 390
} 
#endif
# 392 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 393
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 394
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 398
::exit(___);}
#if 0
# 394
{ 
# 398
} 
#endif
# 401 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 402
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 403
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 407
::exit(___);}
#if 0
# 403
{ 
# 407
} 
#endif
# 409 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 410
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 411
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 415
::exit(___);}
#if 0
# 411
{ 
# 415
} 
#endif
# 419 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 420
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 421
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 425
::exit(___);}
#if 0
# 421
{ 
# 425
} 
#endif
# 427 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 428
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 429
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 433
::exit(___);}
#if 0
# 429
{ 
# 433
} 
#endif
# 72 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 73
struct __nv_tex_rmet_ret { }; 
# 75
template<> struct __nv_tex_rmet_ret< char>  { typedef char type; }; 
# 76
template<> struct __nv_tex_rmet_ret< signed char>  { typedef signed char type; }; 
# 77
template<> struct __nv_tex_rmet_ret< unsigned char>  { typedef unsigned char type; }; 
# 78
template<> struct __nv_tex_rmet_ret< char1>  { typedef char1 type; }; 
# 79
template<> struct __nv_tex_rmet_ret< uchar1>  { typedef uchar1 type; }; 
# 80
template<> struct __nv_tex_rmet_ret< char2>  { typedef char2 type; }; 
# 81
template<> struct __nv_tex_rmet_ret< uchar2>  { typedef uchar2 type; }; 
# 82
template<> struct __nv_tex_rmet_ret< char4>  { typedef char4 type; }; 
# 83
template<> struct __nv_tex_rmet_ret< uchar4>  { typedef uchar4 type; }; 
# 85
template<> struct __nv_tex_rmet_ret< short>  { typedef short type; }; 
# 86
template<> struct __nv_tex_rmet_ret< unsigned short>  { typedef unsigned short type; }; 
# 87
template<> struct __nv_tex_rmet_ret< short1>  { typedef short1 type; }; 
# 88
template<> struct __nv_tex_rmet_ret< ushort1>  { typedef ushort1 type; }; 
# 89
template<> struct __nv_tex_rmet_ret< short2>  { typedef short2 type; }; 
# 90
template<> struct __nv_tex_rmet_ret< ushort2>  { typedef ushort2 type; }; 
# 91
template<> struct __nv_tex_rmet_ret< short4>  { typedef short4 type; }; 
# 92
template<> struct __nv_tex_rmet_ret< ushort4>  { typedef ushort4 type; }; 
# 94
template<> struct __nv_tex_rmet_ret< int>  { typedef int type; }; 
# 95
template<> struct __nv_tex_rmet_ret< unsigned>  { typedef unsigned type; }; 
# 96
template<> struct __nv_tex_rmet_ret< int1>  { typedef int1 type; }; 
# 97
template<> struct __nv_tex_rmet_ret< uint1>  { typedef uint1 type; }; 
# 98
template<> struct __nv_tex_rmet_ret< int2>  { typedef int2 type; }; 
# 99
template<> struct __nv_tex_rmet_ret< uint2>  { typedef uint2 type; }; 
# 100
template<> struct __nv_tex_rmet_ret< int4>  { typedef int4 type; }; 
# 101
template<> struct __nv_tex_rmet_ret< uint4>  { typedef uint4 type; }; 
# 113 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template<> struct __nv_tex_rmet_ret< float>  { typedef float type; }; 
# 114
template<> struct __nv_tex_rmet_ret< float1>  { typedef float1 type; }; 
# 115
template<> struct __nv_tex_rmet_ret< float2>  { typedef float2 type; }; 
# 116
template<> struct __nv_tex_rmet_ret< float4>  { typedef float4 type; }; 
# 119
template< class T> struct __nv_tex_rmet_cast { typedef T *type; }; 
# 131 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 132
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeElementType>  t, int x) 
# 133
{int volatile ___ = 1;(void)t;(void)x;
# 139
::exit(___);}
#if 0
# 133
{ 
# 139
} 
#endif
# 141 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 142
struct __nv_tex_rmnf_ret { }; 
# 144
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
# 145
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
# 146
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
# 147
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
# 148
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
# 149
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
# 150
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
# 151
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
# 152
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
# 153
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
# 154
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
# 155
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
# 156
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
# 157
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
# 158
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 159
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
# 160
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 162
template< class T> 
# 163
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat>  t, int x) 
# 164
{int volatile ___ = 1;(void)t;(void)x;
# 171
::exit(___);}
#if 0
# 164
{ 
# 171
} 
#endif
# 174 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 175
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1D(texture< T, 1, cudaReadModeElementType>  t, float x) 
# 176
{int volatile ___ = 1;(void)t;(void)x;
# 182
::exit(___);}
#if 0
# 176
{ 
# 182
} 
#endif
# 184 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 185
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1D(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x) 
# 186
{int volatile ___ = 1;(void)t;(void)x;
# 193
::exit(___);}
#if 0
# 186
{ 
# 193
} 
#endif
# 197 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 198
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2D(texture< T, 2, cudaReadModeElementType>  t, float x, float y) 
# 199
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 206
::exit(___);}
#if 0
# 199
{ 
# 206
} 
#endif
# 208 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 209
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2D(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y) 
# 210
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 217
::exit(___);}
#if 0
# 210
{ 
# 217
} 
#endif
# 221 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 222
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeElementType>  t, float x, int layer) 
# 223
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 229
::exit(___);}
#if 0
# 223
{ 
# 229
} 
#endif
# 231 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 232
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer) 
# 233
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 240
::exit(___);}
#if 0
# 233
{ 
# 240
} 
#endif
# 244 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 245
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer) 
# 246
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 252
::exit(___);}
#if 0
# 246
{ 
# 252
} 
#endif
# 254 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 255
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer) 
# 256
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 263
::exit(___);}
#if 0
# 256
{ 
# 263
} 
#endif
# 266 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 267
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3D(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z) 
# 268
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 274
::exit(___);}
#if 0
# 268
{ 
# 274
} 
#endif
# 276 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 277
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3D(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 278
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 285
::exit(___);}
#if 0
# 278
{ 
# 285
} 
#endif
# 288 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 289
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z) 
# 290
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 296
::exit(___);}
#if 0
# 290
{ 
# 296
} 
#endif
# 298 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 299
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 300
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 307
::exit(___);}
#if 0
# 300
{ 
# 307
} 
#endif
# 310 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 311
struct __nv_tex2dgather_ret { }; 
# 312
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
# 313
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
# 314
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
# 315
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
# 316
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
# 317
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
# 318
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
# 319
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
# 320
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
# 321
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
# 322
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
# 324
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
# 325
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
# 326
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
# 327
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
# 328
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
# 329
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
# 330
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
# 331
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
# 332
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
# 333
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
# 335
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
# 336
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
# 337
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
# 338
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
# 339
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
# 340
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
# 341
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
# 342
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
# 343
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
# 344
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
# 346
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
# 347
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
# 348
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
# 349
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
# 350
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
# 352
template< class T> 
# 353
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeElementType>  t, float x, float y, int comp = 0) 
# 354
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 361
::exit(___);}
#if 0
# 354
{ 
# 361
} 
#endif
# 364 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> struct __nv_tex2dgather_rmnf_ret { }; 
# 365
template<> struct __nv_tex2dgather_rmnf_ret< char>  { typedef float4 type; }; 
# 366
template<> struct __nv_tex2dgather_rmnf_ret< signed char>  { typedef float4 type; }; 
# 367
template<> struct __nv_tex2dgather_rmnf_ret< unsigned char>  { typedef float4 type; }; 
# 368
template<> struct __nv_tex2dgather_rmnf_ret< char1>  { typedef float4 type; }; 
# 369
template<> struct __nv_tex2dgather_rmnf_ret< uchar1>  { typedef float4 type; }; 
# 370
template<> struct __nv_tex2dgather_rmnf_ret< char2>  { typedef float4 type; }; 
# 371
template<> struct __nv_tex2dgather_rmnf_ret< uchar2>  { typedef float4 type; }; 
# 372
template<> struct __nv_tex2dgather_rmnf_ret< char3>  { typedef float4 type; }; 
# 373
template<> struct __nv_tex2dgather_rmnf_ret< uchar3>  { typedef float4 type; }; 
# 374
template<> struct __nv_tex2dgather_rmnf_ret< char4>  { typedef float4 type; }; 
# 375
template<> struct __nv_tex2dgather_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 376
template<> struct __nv_tex2dgather_rmnf_ret< signed short>  { typedef float4 type; }; 
# 377
template<> struct __nv_tex2dgather_rmnf_ret< unsigned short>  { typedef float4 type; }; 
# 378
template<> struct __nv_tex2dgather_rmnf_ret< short1>  { typedef float4 type; }; 
# 379
template<> struct __nv_tex2dgather_rmnf_ret< ushort1>  { typedef float4 type; }; 
# 380
template<> struct __nv_tex2dgather_rmnf_ret< short2>  { typedef float4 type; }; 
# 381
template<> struct __nv_tex2dgather_rmnf_ret< ushort2>  { typedef float4 type; }; 
# 382
template<> struct __nv_tex2dgather_rmnf_ret< short3>  { typedef float4 type; }; 
# 383
template<> struct __nv_tex2dgather_rmnf_ret< ushort3>  { typedef float4 type; }; 
# 384
template<> struct __nv_tex2dgather_rmnf_ret< short4>  { typedef float4 type; }; 
# 385
template<> struct __nv_tex2dgather_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 387
template< class T> 
# 388
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_rmnf_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, int comp = 0) 
# 389
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 396
::exit(___);}
#if 0
# 389
{ 
# 396
} 
#endif
# 400 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 401
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeElementType>  t, float x, float level) 
# 402
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 408
::exit(___);}
#if 0
# 402
{ 
# 408
} 
#endif
# 410 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 411
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float level) 
# 412
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 419
::exit(___);}
#if 0
# 412
{ 
# 419
} 
#endif
# 422 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 423
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float level) 
# 424
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 430
::exit(___);}
#if 0
# 424
{ 
# 430
} 
#endif
# 432 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 433
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float level) 
# 434
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 441
::exit(___);}
#if 0
# 434
{ 
# 441
} 
#endif
# 444 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 445
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float level) 
# 446
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 452
::exit(___);}
#if 0
# 446
{ 
# 452
} 
#endif
# 454 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 455
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float level) 
# 456
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 463
::exit(___);}
#if 0
# 456
{ 
# 463
} 
#endif
# 466 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 467
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float level) 
# 468
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 474
::exit(___);}
#if 0
# 468
{ 
# 474
} 
#endif
# 476 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 477
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float level) 
# 478
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 485
::exit(___);}
#if 0
# 478
{ 
# 485
} 
#endif
# 488 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 489
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 490
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 496
::exit(___);}
#if 0
# 490
{ 
# 496
} 
#endif
# 498 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 499
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 500
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 507
::exit(___);}
#if 0
# 500
{ 
# 507
} 
#endif
# 510 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 511
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 512
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 518
::exit(___);}
#if 0
# 512
{ 
# 518
} 
#endif
# 520 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 521
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 522
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 529
::exit(___);}
#if 0
# 522
{ 
# 529
} 
#endif
# 533 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 534
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer) 
# 535
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 541
::exit(___);}
#if 0
# 535
{ 
# 541
} 
#endif
# 543 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 544
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer) 
# 545
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 552
::exit(___);}
#if 0
# 545
{ 
# 552
} 
#endif
# 556 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 557
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float level) 
# 558
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 564
::exit(___);}
#if 0
# 558
{ 
# 564
} 
#endif
# 566 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 567
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float level) 
# 568
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 575
::exit(___);}
#if 0
# 568
{ 
# 575
} 
#endif
# 579 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 580
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 581
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 587
::exit(___);}
#if 0
# 581
{ 
# 587
} 
#endif
# 589 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 590
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 591
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 598
::exit(___);}
#if 0
# 591
{ 
# 598
} 
#endif
# 602 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 603
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 604
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 610
::exit(___);}
#if 0
# 604
{ 
# 610
} 
#endif
# 612 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 613
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 614
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 621
::exit(___);}
#if 0
# 614
{ 
# 621
} 
#endif
# 625 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 626
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeElementType>  t, float x, float dPdx, float dPdy) 
# 627
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 633
::exit(___);}
#if 0
# 627
{ 
# 633
} 
#endif
# 635 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 636
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float dPdx, float dPdy) 
# 637
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 644
::exit(___);}
#if 0
# 637
{ 
# 644
} 
#endif
# 648 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 649
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 650
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 656
::exit(___);}
#if 0
# 650
{ 
# 656
} 
#endif
# 658 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 659
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 660
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 667
::exit(___);}
#if 0
# 660
{ 
# 667
} 
#endif
# 670 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 671
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float dPdx, float dPdy) 
# 672
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 678
::exit(___);}
#if 0
# 672
{ 
# 678
} 
#endif
# 680 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 681
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float dPdx, float dPdy) 
# 682
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 689
::exit(___);}
#if 0
# 682
{ 
# 689
} 
#endif
# 692 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 693
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 694
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 700
::exit(___);}
#if 0
# 694
{ 
# 700
} 
#endif
# 702 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 703
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 704
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 711
::exit(___);}
#if 0
# 704
{ 
# 711
} 
#endif
# 714 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 715
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 716
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 722
::exit(___);}
#if 0
# 716
{ 
# 722
} 
#endif
# 724 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 725
__attribute((deprecated)) __attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 726
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 733
::exit(___);}
#if 0
# 726
{ 
# 733
} 
#endif
# 64 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> struct __nv_itex_trait { }; 
# 65
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
# 66
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
# 67
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
# 68
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
# 69
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
# 70
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
# 71
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
# 72
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
# 73
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
# 74
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
# 75
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
# 76
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
# 77
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
# 78
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
# 79
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
# 80
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
# 81
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
# 82
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
# 83
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
# 84
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
# 85
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
# 86
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
# 87
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
# 88
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
# 89
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
# 100 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
# 101
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
# 102
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
# 103
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
# 107
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 108
tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x) 
# 109
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 113
::exit(___);}
#if 0
# 109
{ 
# 113
} 
#endif
# 115 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 116
tex1Dfetch(cudaTextureObject_t texObject, int x) 
# 117
{int volatile ___ = 1;(void)texObject;(void)x;
# 123
::exit(___);}
#if 0
# 117
{ 
# 123
} 
#endif
# 125 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 126
tex1D(T *ptr, cudaTextureObject_t obj, float x) 
# 127
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 131
::exit(___);}
#if 0
# 127
{ 
# 131
} 
#endif
# 134 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 135
tex1D(cudaTextureObject_t texObject, float x) 
# 136
{int volatile ___ = 1;(void)texObject;(void)x;
# 142
::exit(___);}
#if 0
# 136
{ 
# 142
} 
#endif
# 145 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 146
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y) 
# 147
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;
# 151
::exit(___);}
#if 0
# 147
{ 
# 151
} 
#endif
# 153 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 154
tex2D(cudaTextureObject_t texObject, float x, float y) 
# 155
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;
# 161
::exit(___);}
#if 0
# 155
{ 
# 161
} 
#endif
# 164 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 165
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y, bool *
# 166
isResident) 
# 167
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;
# 173
::exit(___);}
#if 0
# 167
{ 
# 173
} 
#endif
# 175 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 176
tex2D(cudaTextureObject_t texObject, float x, float y, bool *isResident) 
# 177
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)isResident;
# 183
::exit(___);}
#if 0
# 177
{ 
# 183
} 
#endif
# 188 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 189
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 190
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 194
::exit(___);}
#if 0
# 190
{ 
# 194
} 
#endif
# 196 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 197
tex3D(cudaTextureObject_t texObject, float x, float y, float z) 
# 198
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 204
::exit(___);}
#if 0
# 198
{ 
# 204
} 
#endif
# 207 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 208
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z, bool *
# 209
isResident) 
# 210
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)isResident;
# 216
::exit(___);}
#if 0
# 210
{ 
# 216
} 
#endif
# 218 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 219
tex3D(cudaTextureObject_t texObject, float x, float y, float z, bool *isResident) 
# 220
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)isResident;
# 226
::exit(___);}
#if 0
# 220
{ 
# 226
} 
#endif
# 230 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 231
tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer) 
# 232
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;
# 236
::exit(___);}
#if 0
# 232
{ 
# 236
} 
#endif
# 238 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 239
tex1DLayered(cudaTextureObject_t texObject, float x, int layer) 
# 240
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;
# 246
::exit(___);}
#if 0
# 240
{ 
# 246
} 
#endif
# 248 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 249
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer) 
# 250
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;
# 254
::exit(___);}
#if 0
# 250
{ 
# 254
} 
#endif
# 256 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 257
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer) 
# 258
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;
# 264
::exit(___);}
#if 0
# 258
{ 
# 264
} 
#endif
# 267 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 268
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, bool *isResident) 
# 269
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)isResident;
# 275
::exit(___);}
#if 0
# 269
{ 
# 275
} 
#endif
# 277 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 278
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer, bool *isResident) 
# 279
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)isResident;
# 285
::exit(___);}
#if 0
# 279
{ 
# 285
} 
#endif
# 289 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 290
texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 291
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 295
::exit(___);}
#if 0
# 291
{ 
# 295
} 
#endif
# 298 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 299
texCubemap(cudaTextureObject_t texObject, float x, float y, float z) 
# 300
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 306
::exit(___);}
#if 0
# 300
{ 
# 306
} 
#endif
# 309 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 310
texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer) 
# 311
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;
# 315
::exit(___);}
#if 0
# 311
{ 
# 315
} 
#endif
# 317 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 318
texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer) 
# 319
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;
# 325
::exit(___);}
#if 0
# 319
{ 
# 325
} 
#endif
# 327 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 328
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0) 
# 329
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;
# 333
::exit(___);}
#if 0
# 329
{ 
# 333
} 
#endif
# 335 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 336
tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0) 
# 337
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;
# 343
::exit(___);}
#if 0
# 337
{ 
# 343
} 
#endif
# 346 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 347
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, bool *isResident, int comp = 0) 
# 348
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;(void)comp;
# 354
::exit(___);}
#if 0
# 348
{ 
# 354
} 
#endif
# 356 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 357
tex2Dgather(cudaTextureObject_t to, float x, float y, bool *isResident, int comp = 0) 
# 358
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)isResident;(void)comp;
# 364
::exit(___);}
#if 0
# 358
{ 
# 364
} 
#endif
# 368 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 369
tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level) 
# 370
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;
# 374
::exit(___);}
#if 0
# 370
{ 
# 374
} 
#endif
# 376 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 377
tex1DLod(cudaTextureObject_t texObject, float x, float level) 
# 378
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;
# 384
::exit(___);}
#if 0
# 378
{ 
# 384
} 
#endif
# 387 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 388
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level) 
# 389
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;
# 393
::exit(___);}
#if 0
# 389
{ 
# 393
} 
#endif
# 395 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 396
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level) 
# 397
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;
# 403
::exit(___);}
#if 0
# 397
{ 
# 403
} 
#endif
# 407 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 408
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level, bool *isResident) 
# 409
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;(void)isResident;
# 415
::exit(___);}
#if 0
# 409
{ 
# 415
} 
#endif
# 417 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 418
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level, bool *isResident) 
# 419
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;(void)isResident;
# 425
::exit(___);}
#if 0
# 419
{ 
# 425
} 
#endif
# 430 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 431
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 432
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 436
::exit(___);}
#if 0
# 432
{ 
# 436
} 
#endif
# 438 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 439
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 440
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 446
::exit(___);}
#if 0
# 440
{ 
# 446
} 
#endif
# 449 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 450
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level, bool *isResident) 
# 451
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;(void)isResident;
# 457
::exit(___);}
#if 0
# 451
{ 
# 457
} 
#endif
# 459 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 460
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level, bool *isResident) 
# 461
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;(void)isResident;
# 467
::exit(___);}
#if 0
# 461
{ 
# 467
} 
#endif
# 472 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 473
tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level) 
# 474
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;
# 478
::exit(___);}
#if 0
# 474
{ 
# 478
} 
#endif
# 480 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 481
tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level) 
# 482
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;
# 488
::exit(___);}
#if 0
# 482
{ 
# 488
} 
#endif
# 491 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 492
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level) 
# 493
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;
# 497
::exit(___);}
#if 0
# 493
{ 
# 497
} 
#endif
# 499 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 500
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level) 
# 501
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;
# 507
::exit(___);}
#if 0
# 501
{ 
# 507
} 
#endif
# 510 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 511
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level, bool *isResident) 
# 512
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;(void)isResident;
# 518
::exit(___);}
#if 0
# 512
{ 
# 518
} 
#endif
# 520 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 521
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level, bool *isResident) 
# 522
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;(void)isResident;
# 528
::exit(___);}
#if 0
# 522
{ 
# 528
} 
#endif
# 531 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 532
texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 533
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 537
::exit(___);}
#if 0
# 533
{ 
# 537
} 
#endif
# 539 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 540
texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 541
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 547
::exit(___);}
#if 0
# 541
{ 
# 547
} 
#endif
# 550 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 551
texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 552
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 556
::exit(___);}
#if 0
# 552
{ 
# 556
} 
#endif
# 558 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 559
texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 560
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 566
::exit(___);}
#if 0
# 560
{ 
# 566
} 
#endif
# 568 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 569
texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level) 
# 570
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 574
::exit(___);}
#if 0
# 570
{ 
# 574
} 
#endif
# 576 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 577
texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
# 578
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 584
::exit(___);}
#if 0
# 578
{ 
# 584
} 
#endif
# 586 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 587
tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy) 
# 588
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;
# 592
::exit(___);}
#if 0
# 588
{ 
# 592
} 
#endif
# 594 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 595
tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
# 596
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;
# 602
::exit(___);}
#if 0
# 596
{ 
# 602
} 
#endif
# 605 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 606
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy) 
# 607
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 612
::exit(___);}
#if 0
# 607
{ 
# 612
} 
#endif
# 614 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 615
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy) 
# 616
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 622
::exit(___);}
#if 0
# 616
{ 
# 622
} 
#endif
# 625 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 626
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy, bool *isResident) 
# 627
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;
# 634
::exit(___);}
#if 0
# 627
{ 
# 634
} 
#endif
# 636 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 637
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy, bool *isResident) 
# 638
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;
# 644
::exit(___);}
#if 0
# 638
{ 
# 644
} 
#endif
# 648 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 649
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 650
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 654
::exit(___);}
#if 0
# 650
{ 
# 654
} 
#endif
# 656 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 657
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 658
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 664
::exit(___);}
#if 0
# 658
{ 
# 664
} 
#endif
# 667 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 668
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy, bool *isResident) 
# 669
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;
# 675
::exit(___);}
#if 0
# 669
{ 
# 675
} 
#endif
# 677 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 678
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy, bool *isResident) 
# 679
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;
# 685
::exit(___);}
#if 0
# 679
{ 
# 685
} 
#endif
# 690 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 691
tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) 
# 692
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 696
::exit(___);}
#if 0
# 692
{ 
# 696
} 
#endif
# 698 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 699
tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
# 700
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 706
::exit(___);}
#if 0
# 700
{ 
# 706
} 
#endif
# 709 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 710
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 711
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 715
::exit(___);}
#if 0
# 711
{ 
# 715
} 
#endif
# 717 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 718
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 719
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 725
::exit(___);}
#if 0
# 719
{ 
# 725
} 
#endif
# 728 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 729
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy, bool *isResident) 
# 730
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;
# 736
::exit(___);}
#if 0
# 730
{ 
# 736
} 
#endif
# 738 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 739
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy, bool *isResident) 
# 740
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;
# 746
::exit(___);}
#if 0
# 740
{ 
# 746
} 
#endif
# 750 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 751
texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 752
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 756
::exit(___);}
#if 0
# 752
{ 
# 756
} 
#endif
# 758 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 759
texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 760
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 766
::exit(___);}
#if 0
# 760
{ 
# 766
} 
#endif
# 59 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> struct __nv_isurf_trait { }; 
# 60
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
# 61
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
# 62
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
# 63
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
# 64
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
# 65
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
# 66
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
# 67
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
# 68
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
# 69
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
# 70
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
# 71
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
# 72
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
# 73
template<> struct __nv_isurf_trait< long long>  { typedef void type; }; 
# 74
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
# 75
template<> struct __nv_isurf_trait< unsigned long long>  { typedef void type; }; 
# 76
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
# 77
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
# 78
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
# 80
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
# 81
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
# 82
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
# 83
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
# 84
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
# 85
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
# 86
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
# 87
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
# 88
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
# 90
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
# 91
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
# 92
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
# 93
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
# 94
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
# 95
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
# 96
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
# 99
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 100
surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 101
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;
# 105
::exit(___);}
#if 0
# 101
{ 
# 105
} 
#endif
# 107 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 108
surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 109
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;
# 115
::exit(___);}
#if 0
# 109
{ 
# 115
} 
#endif
# 117 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 118
surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 119
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;
# 123
::exit(___);}
#if 0
# 119
{ 
# 123
} 
#endif
# 125 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 126
surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 127
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;
# 133
::exit(___);}
#if 0
# 127
{ 
# 133
} 
#endif
# 136 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 137
surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 138
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 142
::exit(___);}
#if 0
# 138
{ 
# 142
} 
#endif
# 144 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 145
surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 146
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;
# 152
::exit(___);}
#if 0
# 146
{ 
# 152
} 
#endif
# 154 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 155
surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 156
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;
# 160
::exit(___);}
#if 0
# 156
{ 
# 160
} 
#endif
# 162 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 163
surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 164
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;
# 170
::exit(___);}
#if 0
# 164
{ 
# 170
} 
#endif
# 172 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 173
surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 174
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 178
::exit(___);}
#if 0
# 174
{ 
# 178
} 
#endif
# 180 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 181
surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 182
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;
# 188
::exit(___);}
#if 0
# 182
{ 
# 188
} 
#endif
# 190 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 191
surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 192
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 196
::exit(___);}
#if 0
# 192
{ 
# 196
} 
#endif
# 198 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 199
surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 200
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;
# 206
::exit(___);}
#if 0
# 200
{ 
# 206
} 
#endif
# 208 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 209
surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 210
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 214
::exit(___);}
#if 0
# 210
{ 
# 214
} 
#endif
# 216 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 217
surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 218
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;
# 224
::exit(___);}
#if 0
# 218
{ 
# 224
} 
#endif
# 226 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 227
surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 228
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;
# 232
::exit(___);}
#if 0
# 228
{ 
# 232
} 
#endif
# 234 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 235
surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 236
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;
# 240
::exit(___);}
#if 0
# 236
{ 
# 240
} 
#endif
# 242 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 243
surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 244
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 248
::exit(___);}
#if 0
# 244
{ 
# 248
} 
#endif
# 250 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 251
surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 252
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;
# 256
::exit(___);}
#if 0
# 252
{ 
# 256
} 
#endif
# 258 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 259
surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 260
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 264
::exit(___);}
#if 0
# 260
{ 
# 264
} 
#endif
# 266 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 267
surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 268
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 272
::exit(___);}
#if 0
# 268
{ 
# 272
} 
#endif
# 274 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 275
surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 276
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 280
::exit(___);}
#if 0
# 276
{ 
# 280
} 
#endif
# 3295 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, CUstream_st * stream = 0); 
# 68 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/device_launch_parameters.h"
extern "C" {
# 71
extern const uint3 __device_builtin_variable_threadIdx; 
# 72
extern const uint3 __device_builtin_variable_blockIdx; 
# 73
extern const dim3 __device_builtin_variable_blockDim; 
# 74
extern const dim3 __device_builtin_variable_gridDim; 
# 75
extern const int __device_builtin_variable_warpSize; 
# 80
}
# 201 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 202
cudaLaunchKernel(const T *
# 203
func, dim3 
# 204
gridDim, dim3 
# 205
blockDim, void **
# 206
args, size_t 
# 207
sharedMem = 0, cudaStream_t 
# 208
stream = 0) 
# 210
{ 
# 211
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 212
} 
# 263 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 264
cudaLaunchCooperativeKernel(const T *
# 265
func, dim3 
# 266
gridDim, dim3 
# 267
blockDim, void **
# 268
args, size_t 
# 269
sharedMem = 0, cudaStream_t 
# 270
stream = 0) 
# 272
{ 
# 273
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 274
} 
# 307 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 308
event, unsigned 
# 309
flags) 
# 311
{ 
# 312
return ::cudaEventCreateWithFlags(event, flags); 
# 313
} 
# 372 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
static inline cudaError_t cudaMallocHost(void **
# 373
ptr, size_t 
# 374
size, unsigned 
# 375
flags) 
# 377
{ 
# 378
return ::cudaHostAlloc(ptr, size, flags); 
# 379
} 
# 381
template< class T> static inline cudaError_t 
# 382
cudaHostAlloc(T **
# 383
ptr, size_t 
# 384
size, unsigned 
# 385
flags) 
# 387
{ 
# 388
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
# 389
} 
# 391
template< class T> static inline cudaError_t 
# 392
cudaHostGetDevicePointer(T **
# 393
pDevice, void *
# 394
pHost, unsigned 
# 395
flags) 
# 397
{ 
# 398
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
# 399
} 
# 501 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 502
cudaMallocManaged(T **
# 503
devPtr, size_t 
# 504
size, unsigned 
# 505
flags = 1) 
# 507
{ 
# 508
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
# 509
} 
# 591 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 592
cudaStreamAttachMemAsync(cudaStream_t 
# 593
stream, T *
# 594
devPtr, size_t 
# 595
length = 0, unsigned 
# 596
flags = 4) 
# 598
{ 
# 599
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
# 600
} 
# 602
template< class T> inline cudaError_t 
# 603
cudaMalloc(T **
# 604
devPtr, size_t 
# 605
size) 
# 607
{ 
# 608
return ::cudaMalloc((void **)((void *)devPtr), size); 
# 609
} 
# 611
template< class T> static inline cudaError_t 
# 612
cudaMallocHost(T **
# 613
ptr, size_t 
# 614
size, unsigned 
# 615
flags = 0) 
# 617
{ 
# 618
return cudaMallocHost((void **)((void *)ptr), size, flags); 
# 619
} 
# 621
template< class T> static inline cudaError_t 
# 622
cudaMallocPitch(T **
# 623
devPtr, size_t *
# 624
pitch, size_t 
# 625
width, size_t 
# 626
height) 
# 628
{ 
# 629
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
# 630
} 
# 641 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
static inline cudaError_t cudaMallocAsync(void **
# 642
ptr, size_t 
# 643
size, cudaMemPool_t 
# 644
memPool, cudaStream_t 
# 645
stream) 
# 647
{ 
# 648
return ::cudaMallocFromPoolAsync(ptr, size, memPool, stream); 
# 649
} 
# 651
template< class T> static inline cudaError_t 
# 652
cudaMallocAsync(T **
# 653
ptr, size_t 
# 654
size, cudaMemPool_t 
# 655
memPool, cudaStream_t 
# 656
stream) 
# 658
{ 
# 659
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
# 660
} 
# 662
template< class T> static inline cudaError_t 
# 663
cudaMallocAsync(T **
# 664
ptr, size_t 
# 665
size, cudaStream_t 
# 666
stream) 
# 668
{ 
# 669
return ::cudaMallocAsync((void **)((void *)ptr), size, stream); 
# 670
} 
# 672
template< class T> static inline cudaError_t 
# 673
cudaMallocFromPoolAsync(T **
# 674
ptr, size_t 
# 675
size, cudaMemPool_t 
# 676
memPool, cudaStream_t 
# 677
stream) 
# 679
{ 
# 680
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
# 681
} 
# 720 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 721
cudaMemcpyToSymbol(const T &
# 722
symbol, const void *
# 723
src, size_t 
# 724
count, size_t 
# 725
offset = 0, cudaMemcpyKind 
# 726
kind = cudaMemcpyHostToDevice) 
# 728
{ 
# 729
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
# 730
} 
# 774 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 775
cudaMemcpyToSymbolAsync(const T &
# 776
symbol, const void *
# 777
src, size_t 
# 778
count, size_t 
# 779
offset = 0, cudaMemcpyKind 
# 780
kind = cudaMemcpyHostToDevice, cudaStream_t 
# 781
stream = 0) 
# 783
{ 
# 784
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
# 785
} 
# 822 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 823
cudaMemcpyFromSymbol(void *
# 824
dst, const T &
# 825
symbol, size_t 
# 826
count, size_t 
# 827
offset = 0, cudaMemcpyKind 
# 828
kind = cudaMemcpyDeviceToHost) 
# 830
{ 
# 831
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
# 832
} 
# 876 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 877
cudaMemcpyFromSymbolAsync(void *
# 878
dst, const T &
# 879
symbol, size_t 
# 880
count, size_t 
# 881
offset = 0, cudaMemcpyKind 
# 882
kind = cudaMemcpyDeviceToHost, cudaStream_t 
# 883
stream = 0) 
# 885
{ 
# 886
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
# 887
} 
# 945 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 946
cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *
# 947
pGraphNode, cudaGraph_t 
# 948
graph, const cudaGraphNode_t *
# 949
pDependencies, size_t 
# 950
numDependencies, const T &
# 951
symbol, const void *
# 952
src, size_t 
# 953
count, size_t 
# 954
offset, cudaMemcpyKind 
# 955
kind) 
# 956
{ 
# 957
return ::cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, (const void *)(&symbol), src, count, offset, kind); 
# 958
} 
# 1016 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1017
cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *
# 1018
pGraphNode, cudaGraph_t 
# 1019
graph, const cudaGraphNode_t *
# 1020
pDependencies, size_t 
# 1021
numDependencies, void *
# 1022
dst, const T &
# 1023
symbol, size_t 
# 1024
count, size_t 
# 1025
offset, cudaMemcpyKind 
# 1026
kind) 
# 1027
{ 
# 1028
return ::cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, (const void *)(&symbol), count, offset, kind); 
# 1029
} 
# 1067 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1068
cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t 
# 1069
node, const T &
# 1070
symbol, const void *
# 1071
src, size_t 
# 1072
count, size_t 
# 1073
offset, cudaMemcpyKind 
# 1074
kind) 
# 1075
{ 
# 1076
return ::cudaGraphMemcpyNodeSetParamsToSymbol(node, (const void *)(&symbol), src, count, offset, kind); 
# 1077
} 
# 1115 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1116
cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t 
# 1117
node, void *
# 1118
dst, const T &
# 1119
symbol, size_t 
# 1120
count, size_t 
# 1121
offset, cudaMemcpyKind 
# 1122
kind) 
# 1123
{ 
# 1124
return ::cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, (const void *)(&symbol), count, offset, kind); 
# 1125
} 
# 1173 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1174
cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t 
# 1175
hGraphExec, cudaGraphNode_t 
# 1176
node, const T &
# 1177
symbol, const void *
# 1178
src, size_t 
# 1179
count, size_t 
# 1180
offset, cudaMemcpyKind 
# 1181
kind) 
# 1182
{ 
# 1183
return ::cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, (const void *)(&symbol), src, count, offset, kind); 
# 1184
} 
# 1232 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1233
cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t 
# 1234
hGraphExec, cudaGraphNode_t 
# 1235
node, void *
# 1236
dst, const T &
# 1237
symbol, size_t 
# 1238
count, size_t 
# 1239
offset, cudaMemcpyKind 
# 1240
kind) 
# 1241
{ 
# 1242
return ::cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, (const void *)(&symbol), count, offset, kind); 
# 1243
} 
# 1271 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1272
cudaUserObjectCreate(cudaUserObject_t *
# 1273
object_out, T *
# 1274
objectToWrap, unsigned 
# 1275
initialRefcount, unsigned 
# 1276
flags) 
# 1277
{ 
# 1278
return ::cudaUserObjectCreate(object_out, objectToWrap, [](void *
# 1281
vpObj) { delete (reinterpret_cast< T *>(vpObj)); } , initialRefcount, flags); 
# 1284
} 
# 1286
template< class T> static inline cudaError_t 
# 1287
cudaUserObjectCreate(cudaUserObject_t *
# 1288
object_out, T *
# 1289
objectToWrap, unsigned 
# 1290
initialRefcount, cudaUserObjectFlags 
# 1291
flags) 
# 1292
{ 
# 1293
return cudaUserObjectCreate(object_out, objectToWrap, initialRefcount, (unsigned)flags); 
# 1294
} 
# 1321 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1322
cudaGetSymbolAddress(void **
# 1323
devPtr, const T &
# 1324
symbol) 
# 1326
{ 
# 1327
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
# 1328
} 
# 1353 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1354
cudaGetSymbolSize(size_t *
# 1355
size, const T &
# 1356
symbol) 
# 1358
{ 
# 1359
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
# 1360
} 
# 1397 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1398
__attribute((deprecated)) static inline cudaError_t cudaBindTexture(size_t *
# 1399
offset, const texture< T, dim, readMode>  &
# 1400
tex, const void *
# 1401
devPtr, const cudaChannelFormatDesc &
# 1402
desc, size_t 
# 1403
size = ((2147483647) * 2U) + 1U) 
# 1405 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
{ 
# 1406
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
# 1407
} 
# 1443 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1444
__attribute((deprecated)) static inline cudaError_t cudaBindTexture(size_t *
# 1445
offset, const texture< T, dim, readMode>  &
# 1446
tex, const void *
# 1447
devPtr, size_t 
# 1448
size = ((2147483647) * 2U) + 1U) 
# 1450 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
{ 
# 1451
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
# 1452
} 
# 1500 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1501
__attribute((deprecated)) static inline cudaError_t cudaBindTexture2D(size_t *
# 1502
offset, const texture< T, dim, readMode>  &
# 1503
tex, const void *
# 1504
devPtr, const cudaChannelFormatDesc &
# 1505
desc, size_t 
# 1506
width, size_t 
# 1507
height, size_t 
# 1508
pitch) 
# 1510
{ 
# 1511
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
# 1512
} 
# 1559 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1560
__attribute((deprecated)) static inline cudaError_t cudaBindTexture2D(size_t *
# 1561
offset, const texture< T, dim, readMode>  &
# 1562
tex, const void *
# 1563
devPtr, size_t 
# 1564
width, size_t 
# 1565
height, size_t 
# 1566
pitch) 
# 1568
{ 
# 1569
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
# 1570
} 
# 1602 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1603
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1604
tex, cudaArray_const_t 
# 1605
array, const cudaChannelFormatDesc &
# 1606
desc) 
# 1608
{ 
# 1609
return ::cudaBindTextureToArray(&tex, array, &desc); 
# 1610
} 
# 1641 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1642
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1643
tex, cudaArray_const_t 
# 1644
array) 
# 1646
{ 
# 1647
cudaChannelFormatDesc desc; 
# 1648
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 1650
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
# 1651
} 
# 1683 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1684
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1685
tex, cudaMipmappedArray_const_t 
# 1686
mipmappedArray, const cudaChannelFormatDesc &
# 1687
desc) 
# 1689
{ 
# 1690
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
# 1691
} 
# 1722 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1723
__attribute((deprecated)) static inline cudaError_t cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1724
tex, cudaMipmappedArray_const_t 
# 1725
mipmappedArray) 
# 1727
{ 
# 1728
cudaChannelFormatDesc desc; 
# 1729
cudaArray_t levelArray; 
# 1730
cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
# 1732
if (err != (cudaSuccess)) { 
# 1733
return err; 
# 1734
}  
# 1735
err = ::cudaGetChannelDesc(&desc, levelArray); 
# 1737
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
# 1738
} 
# 1765 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1766
__attribute((deprecated)) static inline cudaError_t cudaUnbindTexture(const texture< T, dim, readMode>  &
# 1767
tex) 
# 1769
{ 
# 1770
return ::cudaUnbindTexture(&tex); 
# 1771
} 
# 1801 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> 
# 1802
__attribute((deprecated)) static inline cudaError_t cudaGetTextureAlignmentOffset(size_t *
# 1803
offset, const texture< T, dim, readMode>  &
# 1804
tex) 
# 1806
{ 
# 1807
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
# 1808
} 
# 1853 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1854
cudaFuncSetCacheConfig(T *
# 1855
func, cudaFuncCache 
# 1856
cacheConfig) 
# 1858
{ 
# 1859
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
# 1860
} 
# 1862
template< class T> static inline cudaError_t 
# 1863
cudaFuncSetSharedMemConfig(T *
# 1864
func, cudaSharedMemConfig 
# 1865
config) 
# 1867
{ 
# 1868
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
# 1869
} 
# 1901 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1902
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
# 1903
numBlocks, T 
# 1904
func, int 
# 1905
blockSize, size_t 
# 1906
dynamicSMemSize) 
# 1907
{ 
# 1908
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
# 1909
} 
# 1953 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1954
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
# 1955
numBlocks, T 
# 1956
func, int 
# 1957
blockSize, size_t 
# 1958
dynamicSMemSize, unsigned 
# 1959
flags) 
# 1960
{ 
# 1961
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
# 1962
} 
# 1967
class __cudaOccupancyB2DHelper { 
# 1968
size_t n; 
# 1970
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
# 1971
size_t operator()(int) 
# 1972
{ 
# 1973
return n; 
# 1974
} 
# 1975
}; 
# 2023 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class UnaryFunction, class T> static inline cudaError_t 
# 2024
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
# 2025
minGridSize, int *
# 2026
blockSize, T 
# 2027
func, UnaryFunction 
# 2028
blockSizeToDynamicSMemSize, int 
# 2029
blockSizeLimit = 0, unsigned 
# 2030
flags = 0) 
# 2031
{ 
# 2032
cudaError_t status; 
# 2035
int device; 
# 2036
cudaFuncAttributes attr; 
# 2039
int maxThreadsPerMultiProcessor; 
# 2040
int warpSize; 
# 2041
int devMaxThreadsPerBlock; 
# 2042
int multiProcessorCount; 
# 2043
int funcMaxThreadsPerBlock; 
# 2044
int occupancyLimit; 
# 2045
int granularity; 
# 2048
int maxBlockSize = 0; 
# 2049
int numBlocks = 0; 
# 2050
int maxOccupancy = 0; 
# 2053
int blockSizeToTryAligned; 
# 2054
int blockSizeToTry; 
# 2055
int blockSizeLimitAligned; 
# 2056
int occupancyInBlocks; 
# 2057
int occupancyInThreads; 
# 2058
size_t dynamicSMemSize; 
# 2064
if (((!minGridSize) || (!blockSize)) || (!func)) { 
# 2065
return cudaErrorInvalidValue; 
# 2066
}  
# 2072
status = ::cudaGetDevice(&device); 
# 2073
if (status != (cudaSuccess)) { 
# 2074
return status; 
# 2075
}  
# 2077
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
# 2081
if (status != (cudaSuccess)) { 
# 2082
return status; 
# 2083
}  
# 2085
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
# 2089
if (status != (cudaSuccess)) { 
# 2090
return status; 
# 2091
}  
# 2093
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
# 2097
if (status != (cudaSuccess)) { 
# 2098
return status; 
# 2099
}  
# 2101
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
# 2105
if (status != (cudaSuccess)) { 
# 2106
return status; 
# 2107
}  
# 2109
status = cudaFuncGetAttributes(&attr, func); 
# 2110
if (status != (cudaSuccess)) { 
# 2111
return status; 
# 2112
}  
# 2114
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
# 2120
occupancyLimit = maxThreadsPerMultiProcessor; 
# 2121
granularity = warpSize; 
# 2123
if (blockSizeLimit == 0) { 
# 2124
blockSizeLimit = devMaxThreadsPerBlock; 
# 2125
}  
# 2127
if (devMaxThreadsPerBlock < blockSizeLimit) { 
# 2128
blockSizeLimit = devMaxThreadsPerBlock; 
# 2129
}  
# 2131
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
# 2132
blockSizeLimit = funcMaxThreadsPerBlock; 
# 2133
}  
# 2135
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
# 2137
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
# 2141
if (blockSizeLimit < blockSizeToTryAligned) { 
# 2142
blockSizeToTry = blockSizeLimit; 
# 2143
} else { 
# 2144
blockSizeToTry = blockSizeToTryAligned; 
# 2145
}  
# 2147
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
# 2149
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
# 2156
if (status != (cudaSuccess)) { 
# 2157
return status; 
# 2158
}  
# 2160
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
# 2162
if (occupancyInThreads > maxOccupancy) { 
# 2163
maxBlockSize = blockSizeToTry; 
# 2164
numBlocks = occupancyInBlocks; 
# 2165
maxOccupancy = occupancyInThreads; 
# 2166
}  
# 2170
if (occupancyLimit == maxOccupancy) { 
# 2171
break; 
# 2172
}  
# 2173
}  
# 2181
(*minGridSize) = (numBlocks * multiProcessorCount); 
# 2182
(*blockSize) = maxBlockSize; 
# 2184
return status; 
# 2185
} 
# 2219 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class UnaryFunction, class T> static inline cudaError_t 
# 2220
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
# 2221
minGridSize, int *
# 2222
blockSize, T 
# 2223
func, UnaryFunction 
# 2224
blockSizeToDynamicSMemSize, int 
# 2225
blockSizeLimit = 0) 
# 2226
{ 
# 2227
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
# 2228
} 
# 2265 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 2266
cudaOccupancyMaxPotentialBlockSize(int *
# 2267
minGridSize, int *
# 2268
blockSize, T 
# 2269
func, size_t 
# 2270
dynamicSMemSize = 0, int 
# 2271
blockSizeLimit = 0) 
# 2272
{ 
# 2273
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
# 2274
} 
# 2303 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 2304
cudaOccupancyAvailableDynamicSMemPerBlock(size_t *
# 2305
dynamicSmemSize, T 
# 2306
func, int 
# 2307
numBlocks, int 
# 2308
blockSize) 
# 2309
{ 
# 2310
return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (const void *)func, numBlocks, blockSize); 
# 2311
} 
# 2362 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 2363
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
# 2364
minGridSize, int *
# 2365
blockSize, T 
# 2366
func, size_t 
# 2367
dynamicSMemSize = 0, int 
# 2368
blockSizeLimit = 0, unsigned 
# 2369
flags = 0) 
# 2370
{ 
# 2371
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
# 2372
} 
# 2405 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 2406
cudaFuncGetAttributes(cudaFuncAttributes *
# 2407
attr, T *
# 2408
entry) 
# 2410
{ 
# 2411
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
# 2412
} 
# 2450 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 2451
cudaFuncSetAttribute(T *
# 2452
entry, cudaFuncAttribute 
# 2453
attr, int 
# 2454
value) 
# 2456
{ 
# 2457
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
# 2458
} 
# 2482 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim> 
# 2483
__attribute((deprecated)) static inline cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
# 2484
surf, cudaArray_const_t 
# 2485
array, const cudaChannelFormatDesc &
# 2486
desc) 
# 2488
{ 
# 2489
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
# 2490
} 
# 2513 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim> 
# 2514
__attribute((deprecated)) static inline cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
# 2515
surf, cudaArray_const_t 
# 2516
array) 
# 2518
{ 
# 2519
cudaChannelFormatDesc desc; 
# 2520
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 2522
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
# 2523
} 
# 2534 "/pawsey/centos7.6/devel/binary/cuda/11.4.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
#pragma GCC diagnostic pop
# 64 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
# 66
const char *info_simulate = ("INFO:simulate[GNU]"); 
# 305 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((11 / 10000000) % 10)), (('0') + ((11 / 1000000) % 10)), (('0') + ((11 / 100000) % 10)), (('0') + ((11 / 10000) % 10)), (('0') + ((11 / 1000) % 10)), (('0') + ((11 / 100) % 10)), (('0') + ((11 / 10) % 10)), (('0') + (11 % 10)), '.', (('0') + ((4 / 10000000) % 10)), (('0') + ((4 / 1000000) % 10)), (('0') + ((4 / 100000) % 10)), (('0') + ((4 / 10000) % 10)), (('0') + ((4 / 1000) % 10)), (('0') + ((4 / 100) % 10)), (('0') + ((4 / 10) % 10)), (('0') + (4 % 10)), '.', (('0') + ((120 / 10000000) % 10)), (('0') + ((120 / 1000000) % 10)), (('0') + ((120 / 100000) % 10)), (('0') + ((120 / 10000) % 10)), (('0') + ((120 / 1000) % 10)), (('0') + ((120 / 100) % 10)), (('0') + ((120 / 10) % 10)), (('0') + (120 % 10)), ']', '\000'}; 
# 332 "CMakeCUDACompilerId.cu"
const char info_simulate_version[] = {'I', 'N', 'F', 'O', ':', 's', 'i', 'm', 'u', 'l', 'a', 't', 'e', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((11 / 10000000) % 10)), (('0') + ((11 / 1000000) % 10)), (('0') + ((11 / 100000) % 10)), (('0') + ((11 / 10000) % 10)), (('0') + ((11 / 1000) % 10)), (('0') + ((11 / 100) % 10)), (('0') + ((11 / 10) % 10)), (('0') + (11 % 10)), '.', (('0') + ((1 / 10000000) % 10)), (('0') + ((1 / 1000000) % 10)), (('0') + ((1 / 100000) % 10)), (('0') + ((1 / 10000) % 10)), (('0') + ((1 / 1000) % 10)), (('0') + ((1 / 100) % 10)), (('0') + ((1 / 10) % 10)), (('0') + (1 % 10)), ']', '\000'}; 
# 352
const char *info_platform = ("INFO:platform[Linux]"); 
# 353
const char *info_arch = ("INFO:arch[]"); 
# 358
const char *info_language_dialect_default = ("INFO:dialect_default[17]"); 
# 374
int main(int argc, char *argv[]) 
# 375
{ 
# 376
int require = 0; 
# 377
require += (info_compiler[argc]); 
# 378
require += (info_platform[argc]); 
# 380
require += (info_version[argc]); 
# 383
require += (info_simulate[argc]); 
# 386
require += (info_simulate_version[argc]); 
# 388
require += (info_language_dialect_default[argc]); 
# 389
(void)argv; 
# 390
return require; 
# 391
} 

# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__27_CMakeCUDACompilerId_cpp1_ii_bd57c623
#ifdef _NV_ANON_NAMESPACE
#endif
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
