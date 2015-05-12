//
//  test_image.h
//  GPAPI
//
//  Created by savage309 on 12.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef GPAPI_test_image_h
#define GPAPI_test_image_h
#include "simd/simd.h"

namespace Image {
    using std::pow;
    embree::ssef exp(const embree::ssef& s) {
        embree::ssef res;
        res.m128 = exp(s.m128);
        return res;
    }
    embree::ssef pow(embree::ssef s, float pow) {
        v4sf logps = (log_ps(s.m128));
        embree::ssef ss;
        ss.m128 = logps;
        return exp(pow*ss);
    }
    struct Float {
        enum { size = 1 };
        float f;
        Float() {}
        Float(float f):f(f){}
        Float operator/(Float rhs) {
            return Float(f/rhs.f);
        }
        Float operator/(float rhs) {
            return Float(f/rhs);
        }
        Float& operator*=(float rhs) {
            f *= rhs;
            return *this;
        }
        operator float() const {
            return f;
        }
    };
    
    
    template<typename T>
    T decode(T color) {
        color=pow(color/0.5f, 2.2f)*0.5f;
        color=pow((color+0.055f)/1.055f, 2.4f);
        return color;
    }
    
    template<typename T>
    T correctHDRIColor(T color, int colorSpace, float gamma, float colorMult) {
        color *= colorMult;
        color = pow(color, gamma);
        color = decode(color);
        return color;
    }
    
    template <typename T>
    T apply(int flags, float gammaColor, int colorSpace, T res, T blendColor, float colorMult, float blend) {
        res=correctHDRIColor(res, colorSpace, gammaColor, colorMult);
        res=res*blend + res*((1.0f-blend)*blend);
        return res;
    }
    
    template <typename T>
    T randomColor();
    
    template <>
    Float randomColor<Float>() {
        Float res;
        res.f = randomFloat();
        return res;
    }
    
    template <>
    embree::ssef randomColor<embree::ssef>() {
        embree::ssef f;
        return f;
    }
    
    template <typename T>
    struct Image {
        int width, height, colorSpace, flags;
        float gamma, colorMult;
        std::unique_ptr<T[]> colors;
        Image(int width, int height):width(width), height(height) {
            colors.reset(new T[width * height / (embree::avxf::size/T::size)]);
            gamma = randomFloat();
            colorMult = randomFloat();
            colorSpace = randomInt(0, 2);
            flags = randomInt(0, 1);
            for (int i = 0; i < width; ++i) {
                for (int j = 0; j < height; ++j) {
                    colors[i + width * j] = randomColor<T>();
                }
            }
        }
        
    };
    
    void test() {
        Image<Float> img(1024, 768);
        
        Float blendColor = randomColor<Float>();
        float blend = randomFloat();
        
        for (int i = 0; i < img.width; ++i) {
            for (int j = 0; j < img.height; ++j) {
                apply(img.flags, img.gamma, img.colorSpace, img.colors[i + img.width*j], blendColor, img.colorMult, blend);
            }
        }
        
        for (int j = 0; j < img.width; ++j) {
            for (int i = 0; i < img.height; ++i) {
                apply(img.flags, img.gamma, img.colorSpace, img.colors[i + img.width*j], blendColor, img.colorMult, blend);
            }
        }
        
        Image<embree::ssef> img2(1024, 768);
        embree::ssef blendColor2;// = randomColor<Float>();

        for (int i = 0; i < img2.width; ++i) {
            for (int j = 0; j < img2.height; ++j) {
                apply(img2.flags, img2.gamma, img2.colorSpace, img2.colors[i + img2.width*j], blendColor2, img2.colorMult, blend);
            }
        }
    }
}

#endif
