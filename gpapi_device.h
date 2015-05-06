#pragma once

#ifdef __CUDACC__
#define BARRIER_LOCAL_MEM_FENCE
#define SHARED __shared__
#define BARRIER(X) __syncthreads()
#define CONSTANT __constant__

#define DEVICE __device__
#define GLOBAL "extern C" __global__
#define NEEDS_PRIM_GENERATORS
#define GLOBAL_CONST(type) const type * __restrict
#define NO_INLINE __noinline__
#define INLINE __forceinline__
int getGlobalId() { return blockIdx.x*blockDim.x + threadIdx.x; }
#elif defined __OPENCL_VERSION__
#define SHARED local
#define BARRIER_LOCAL_MEM_FENCE CLK_LOCAL_MEM_FENCE
#define BARRIER(X) barrier(X)
#define CONSTANT __constant

#define DEVICE __device
#define GLOBAL __global
#define GLOBAL_CONST(type) __global const type * restrict
#define NO_INLINE
#define INLINE

int getGlobalId() { return get_global_id(0); }
#else //x86
#define BARRIER_LOCAL_MEM_FENCE
#define SHARED
#define BARRIER(X)
#define CONSTANT static const

#define DEVICE
#define GLOBAL
#define NEEDS_PRIM_TYPES
#define GLOBAL_CONST(type) const type *
#define NO_INLINE
#define INLINE inline

INLINE float max(float a, float b) { return fmax(a, b); }
INLINE float min(float a, float b) { return fmin(a, b); }

struct PerThreadData {
    int id;
};

static PerThreadData threadData[128];

int getGlobalId() { return 0; }
#endif
#define SYNC_THREADS BARRIER(BARRIER_LOCAL_MEM_FENCE)

#ifdef NEEDS_PRIM_TYPES
struct float4 {
    float x, y, z, w;
    float4(){}
    float4(float x, float y, float z, float w):x(x), y(y), z(z), w(w){}
    float4 operator+(const float4& rhs) {
        return float4(x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w);
    }
    float4 operator-(const float4& rhs) {
        return float4(x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w);
    }
    float4 operator*(const float4& rhs) {
        return float4(x * rhs.x, y * rhs.y, z * rhs.z, w * rhs.w);
    }
    float4 operator/(const float4& rhs) {
        return float4(x / rhs.x, y / rhs.y, z / rhs.z, w / rhs.w);
    }
};

struct int4 {
    int x,y,z,w;
    int4(){}
    int4(int x, int y, int z, int w):x(x),y(y),z(z),w(w){}
    int4 operator+(const int4& rhs){
        return int4(x+rhs.x, y+rhs.y, z+rhs.z, w+rhs.w);
    }
    int4 operator*(const int4& rhs){
        return int4(x*rhs.x, y*rhs.y, z*rhs.z, w*rhs.w);
    }
    int4 operator/(const int4& rhs){
        return int4(x/rhs.x, y/rhs.y, z/rhs.z, w/rhs.w);
    }
    int4 operator-(const int4& rhs){
        return int4(x-rhs.x, y-rhs.y, z-rhs.z, w-rhs.w);
    }
};

INLINE DEVICE float dot( float4 a, float4 b ) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

INLINE DEVICE float length( float4 v ) {
    return sqrtf( v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w );
}

INLINE DEVICE float4 normalize( float4 v ) {
    float l = 1.0f / sqrtf( dot(v,v) );
    return float4( v.x*l, v.y*l, v.z*l, v.w*l );
}

INLINE DEVICE float4 cross( float4 a, float4 b ) {
    return float4(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x, 0.0f );
}

INLINE DEVICE float clamp(float f, float a, float b) {
    return max(a, min(f, b));
}

INLINE DEVICE float4 operator*(const float4 &a, const float b ) {
    return float4( a.x*b, a.y*b, a.z*b, a.w*b );
}

INLINE DEVICE float4 operator/(const float4 &a, const float b ) {
    return float4( a.x/b, a.y/b, a.z/b, a.w/b );
}

INLINE DEVICE float4 operator*(const float b, const float4 &a) {
    return float4( a.x*b, a.y*b, a.z*b, a.w*b );
}

INLINE DEVICE float4 operator+(const float4 &a, const float4 &b ) {
    return float4( a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w );
}

INLINE DEVICE float4 operator-(const float4 &a, const float4 &b ) {
    return float4( a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w );
}

INLINE DEVICE float4 operator*(const float4 &a, const float4 &b ) {
    return float4( a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w );
}

INLINE DEVICE float4 operator-(const float4 &b) {
    return float4( -b.x, -b.y, -b.z, -b.w );
}

INLINE DEVICE float4 min( const float4 &a, const float4 &b ) {
    return float4( min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w) );
}

INLINE DEVICE float4 max( const float4 &a, const float4 &b ) {
    return float4( max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w) );
}

INLINE DEVICE float4& operator*=(float4 &a, const float4 &b ) {
    a.x*=b.x;
    a.y*=b.y;
    a.z*=b.z;
    a.w*=b.w;
    return a;
}

INLINE DEVICE float4& operator*=(float4 &a, const float &b ) {
    a.x*=b;
    a.y*=b;
    a.z*=b;
    a.w*=b;
    return a;
}

INLINE DEVICE float4& operator+=(float4 &a, const float4 &b ) {
    a.x+=b.x;
    a.y+=b.y;
    a.z+=b.z;
    a.w+=b.w;
    return a;
}

INLINE DEVICE float4& operator-=(float4 &a, const float4 &b ) {
    a.x-=b.x;
    a.y-=b.y;
    a.z-=b.z;
    a.w-=b.w;
    return a;
}

INLINE DEVICE float4& operator/=(float4 &a, const float &b ) {
    a.x/=b;
    a.y/=b;
    a.z/=b;
    a.w/=b;
    return a;
}
#endif

#ifdef NEEDS_PRIM_GENERATORS
#define float4 make_float4
#define float2 make_float2
#define int4 make_int4
#define int2 make_int2
#else
#define make_float4 float4
#define make_float2 float2
#define make_int4 int4
#define make_int2 int2
#endif
