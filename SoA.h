//
//  SoA.h
//  GPAPI
//
//  Created by savage309 on 6.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef GPAPI_SoA_h
#define GPAPI_SoA_h

#include <cmath>
#include <memory>
#include "common.h"
#include <iostream>


namespace SoA {
    
    
#define SOA_TEST_SIZE 50000000

    
    struct Particle
    {
        Particle() {
            randomize();
        }
        float x;
        float y;
        float z;
        float w;
        
        float vx;
        float vy;
        float vz;
        float vw;
        
        float a,b,c,d,e,f,g,h,i,j,k,l;
        
        void randomize() {
            x = randomFloat();
            y = randomFloat();
            z = randomFloat();
            w = randomFloat();
            vx = randomFloat();
            vy = randomFloat();
            vz = randomFloat();
            vw = randomFloat();
        }
    };
    
    struct ParticleSystem_AoS
    {
        Particle * particles;
        int count;
        
        void update(float dt)
        {
            for (int i=0; i<count; i++)
            {
                Particle& p = particles[i];
                p.x += p.vx * dt;
                p.y += p.vy * dt;
                p.z += p.vz * dt;
                p.w += p.vw * dt;
            }
        }
    };
    //
    struct ParticleSystem_SoA
    {
        ParticleSystem_SoA(int count):count(count) {
            x = new float[count];
            y = new float[count];
            z = new float[count];
            w = new float[count];
            
            vx = new float[count];
            vy = new float[count];
            vz = new float[count];
            vw = new float[count];
        }
        
        float * x;
        float * y;
        float * z;
        float * w;
        
        float * vx;
        float * vy;
        float * vz;
        float * vw;
        float* a; float* b; float* c; float* d;
        float* e; float* f; float* g; float* h;
        float* i; float* j; float* k; float* l;
        
        int count;
        
        void randomize() {
            struct {
                void operator()(float* ptr, int count) {
                    for (int i = 0; i < count; ++i) {
                        ptr[i] = randomFloat();
                    }
                }
                
            } randomizeArr;
            
            randomizeArr(x, count);
            randomizeArr(y, count);
            randomizeArr(z, count);
            randomizeArr(w, count);
            randomizeArr(vx, count);
            randomizeArr(vy, count);
            randomizeArr(vz, count);
            randomizeArr(vw, count);
            
        }
        
        void update(float dt)
        {
            for (int i=0; i<count; i++)
            {
                x[i] += vx[i] * dt;
                y[i] += vy[i] * dt;
                z[i] += vz[i] * dt;
                w[i] += vw[i] * dt;
            }
        }
    };
    
    
    void AoS() {
        Particle* particles = new Particle[SOA_TEST_SIZE];
        ParticleSystem_AoS aos;
        aos.particles = particles;
        aos.count = SOA_TEST_SIZE;
        auto t0 = getTime();
        for (float f = 0.f; f < 1.f; f += .1f) {
            aos.update(f);
        }
        auto t1 = getTime();
        std::cout << "time AoS " << diffclock(t1, t0) << std::endl;

    }
    
    void SoA() {
      
        ParticleSystem_SoA soa(SOA_TEST_SIZE);
        
        soa.randomize();
        
        auto t0 = getTime();
        for (float f = 0.f; f < 1.f; f += .1f) {
            soa.update(f);
        }
        auto t1 = getTime();
        
        std::cout << "time SoA " << diffclock(t1, t0) << std::endl;
    
    }
    
    void test() {
        std::cout << "Testing SoA vs AoS ..." << std::endl;
        AoS();
        SoA();
        std::cout << "\n **** \n\n";
    }
    
    /*
    struct Vector {
        Vector(){}
        Vector(float x, float y, float z):x(x), y(y), z(z){}
        float x, y, z;
        float length() const {
            return sqrt(x*x + y*y + z*z);
        }
        Vector crossProduct(const Vector& rhs) const {
            return Vector(y * rhs.z - z * rhs.y,
                          x * rhs.z - z * rhs.x,
                          x * rhs.y - y * rhs.x);
        }
        float dotProduct(const Vector& rhs) const {
            return x*x + y*y + z*z;
        }
        Vector operator-(const Vector& rhs) const {
            return Vector(x - rhs.x, y - rhs.y, z - rhs.z);
        }
        Vector operator+(const Vector& rhs) const {
            return Vector(x + rhs.x, y + rhs.y, z + rhs.z);
        }
        Vector operator*(float f) const {
            return Vector(x * f, y * f, z *f);
        }
    };

    
    struct FaceIndex {
        int f[3];
    };
    
    typedef Vector Point;
    
    struct Mesh {
        Point* vertexes;
        size_t numVertexes;
        FaceIndex* faces;
        size_t numFaces;
    };
    
    struct MeshSoA {
        float* x;
        float* y;
        float* z;
        float* w;
        int* faces;
    };
    
    struct Ray {
        Point origin;
        Vector direction;
        void randomize() {
            origin.x = randomFloat();
            origin.y = randomFloat();
            origin.z = randomFloat();
            direction.x = randomFloat();
            direction.y = randomFloat();
            direction.z = randomFloat();
        }
    };
    
    #define EPSILON 1e-6f
    
    bool rayTriangleIntersect(const Point &orig, const Vector &dir,
                              const Point &v0, const Point &v1, const Point &v2,
                              float &t)
    {
        // compute plane's normal
        Vector v0v1 = v1 - v0;
        Vector v0v2 = v2 - v0;
        // no need to normalize
        Vector N = v0v1.crossProduct(v0v2); // N
        // Step 1: finding P
        
        // check if ray and plane are parallel ?
        float NdotRayDirection = N.dotProduct(dir);
        if (fabs(NdotRayDirection) < EPSILON) // almost 0
            return false; // they are parallel so they don't intersect !
        
        // compute d parameter using equation 2
        float d = N.dotProduct(v0);
        
        // compute t (equation 3)
        t = (N.dotProduct(orig) + d) / NdotRayDirection;
        // check if the triangle is in behind the ray
        if (t < 0) return false; // the triangle is behind
        
        // compute the intersection point using equation 1
        Vector P = orig + dir * t;
        
        // Step 2: inside-outside test
        Vector C; // vector perpendicular to triangle's plane
        
        // edge 0
        Vector edge0 = v1 - v0;
        Vector vp0 = P - v0;
        C = edge0.crossProduct(vp0);
        if (N.dotProduct(C) < 0) return false; // P is on the right side
        
        // edge 1
        Vector edge1 = v2 - v1;
        Vector vp1 = P - v1;
        C = edge1.crossProduct(vp1);
        if (N.dotProduct(C) < 0)  return false; // P is on the right side
        
        // edge 2
        Vector edge2 = v0 - v2; 
        Vector vp2 = P - v2; 
        C = edge2.crossProduct(vp2); 
        if (N.dotProduct(C) < 0) return false; // P is on the right side; 
        
        return true; // this ray hits the triangle 
    }
    
    Mesh* newMesh() {
        Mesh* result = new Mesh;
        
        result->numFaces = randomInt(10000, 20000);
        result->numVertexes = result->numFaces * 3;
        for (int i = 0; i < result->numFaces; ++i) {
            result->faces[i].f[0] = randomInt(0, (int)result->numVertexes);
            result->faces[i].f[1] = randomInt(0, (int)result->numVertexes);
            result->faces[i].f[2] = randomInt(0, (int)result->numVertexes);
        }
        
        for (int i = 0; i < result->numVertexes; ++i) {
            result->vertexes[i].x = randomFloat();
            result->vertexes[i].y = randomFloat();
            result->vertexes[i].z = randomFloat();
        }
        
        return result;
    }
    
    void test0() {
        const int NUM_MESHES = 10;
        const int NUM_RAYS = 100000;
        Mesh** meshes = new Mesh*[NUM_MESHES];
        
        for (int i = 0; i < NUM_MESHES; ++i) {
            meshes[i] = newMesh();
            
            Mesh& mesh = *meshes[i];
            
            for (int r = 0; r < NUM_RAYS; ++r) {
                
                Ray ray;
                ray.randomize();
                
                for (int t = 0; t < mesh.numFaces; ++t) {
                    FaceIndex& index = mesh.faces[i];
                    Point& vertex0 = mesh.vertexes[index.f[0]];
                    Point& vertex1 = mesh.vertexes[index.f[1]];
                    Point& vertex2 = mesh.vertexes[index.f[2]];
                    float distance;
                    rayTriangleIntersect(ray.origin, ray.direction, vertex0, vertex1, vertex2, distance);
                }
            }
        }
        
        for (int i = 0; i < NUM_MESHES; ++i) {
            delete meshes[i];
        }
        delete[] meshes;
    }*/
    
} //namespace SoA

#endif
