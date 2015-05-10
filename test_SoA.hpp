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
        return 20000000;
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
        std::unique_ptr<Particle[]> particles;
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
            x.reset(new float[count]);
            y.reset(new float[count]);
            z.reset(new float[count]);
            w.reset(new float[count]);
            
            vx.reset(new float[count]);
            vy.reset(new float[count]);
            vz.reset(new float[count]);
            vw.reset(new float[count]);
        }
        
        std::unique_ptr<float[]> x;
        std::unique_ptr<float[]> y;
        std::unique_ptr<float[]> z;
        std::unique_ptr<float[]> w;
        
        std::unique_ptr<float[]> vx;
        std::unique_ptr<float[]> vy;
        std::unique_ptr<float[]> vz;
        std::unique_ptr<float[]> vw;
        std::unique_ptr<float[]> a; std::unique_ptr<float[]> b; std::unique_ptr<float[]> c; std::unique_ptr<float[]>  d;
        std::unique_ptr<float[]> e; std::unique_ptr<float[]> f; std::unique_ptr<float[]> g; std::unique_ptr<float[]>  h;
        std::unique_ptr<float[]> i; std::unique_ptr<float[]> j; std::unique_ptr<float[]> k; std::unique_ptr<float[]>  l;
        
        size_t count;
        
        void randomize() {
            std::for_each(x.get(), x.get() + count, [](float) { return randomFloat(); } );
            std::for_each(y.get(), y.get() + count, [](float) { return randomFloat(); } );
            std::for_each(z.get(), z.get() + count, [](float) { return randomFloat(); } );
            std::for_each(w.get(), w.get() + count, [](float) { return randomFloat(); } );
            std::for_each(vx.get(), vx.get() + count, [](float) { return randomFloat(); } );
            std::for_each(vy.get(), vy.get() + count, [](float) { return randomFloat(); } );
            std::for_each(vz.get(), vz.get() + count, [](float) { return randomFloat(); } );
            std::for_each(vw.get(), vw.get() + count, [](float) { return randomFloat(); } );

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
        std::unique_ptr<Particle[]> particles(new Particle[getTestSize()]);
        ParticleSystem_AoS aos;
        aos.particles = std::move(particles);
        aos.count = getTestSize();
        auto t0 = getTime();
        for (float f = 0.f; f < 1.f; f += .1f) {
            aos.update(f);
        }
        auto t1 = getTime();
        std::cout << '\t' << "time AoS " << diffclock(t1, t0) << std::endl;

    }
    
    void SoA() {
      
        ParticleSystem_SoA soa(getTestSize());
        
        soa.randomize();
        
        auto t0 = getTime();
        for (float f = 0.f; f < 1.f; f += .1f) {
            soa.update(f);
        }
        auto t1 = getTime();
        
        std::cout << '\t' << "time SoA " << diffclock(t1, t0) << std::endl;
    
    }
    
    void test() {
        std::cout << "Testing SoA vs AoS ..." << std::endl;
        AoS();
        SoA();
        std::cout << "\n **** \n\n";
    }
    
} //namespace SoA

#endif
