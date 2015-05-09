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
    
    size_t getTestSize() {
        return 50000000;
    }
    
    struct Particle {
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
    
    struct ParticleSystem_AoS {
        Particle * particles;
        size_t count;
        
        void update(float dt) {
            for (int i=0; i<count; i++) {
                Particle& p = particles[i];
                p.x += p.vx * dt;
                p.y += p.vy * dt;
                p.z += p.vz * dt;
                p.w += p.vw * dt;
            }
        }
    };
    //
    struct ParticleSystem_SoA {
        ParticleSystem_SoA(size_t count):count(count) {
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
        
        size_t count;
        
        void randomize() {
            struct {
                void operator()(float* ptr, size_t count) {
                    for (size_t i = 0; i < count; ++i) {
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
        
        void update(float dt) {
            for (size_t i = 0; i < count; i++) {
                x[i] += vx[i] * dt;
                y[i] += vy[i] * dt;
                z[i] += vz[i] * dt;
                w[i] += vw[i] * dt;
            }
        }
    };
    
    
    void AoS() {
        Particle* particles = new Particle[getTestSize()];
        ParticleSystem_AoS aos;
        aos.particles = particles;
        aos.count = getTestSize();
        auto t0 = getTime();
        for (float f = 0.f; f < 1.f; f += .1f) {
            aos.update(f);
        }
        auto t1 = getTime();
        std::cout << "time AoS " << diffclock(t1, t0) << std::endl;

    }
    
    void SoA() {
      
        ParticleSystem_SoA soa(getTestSize());
        
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
    
} //namespace SoA

#endif
