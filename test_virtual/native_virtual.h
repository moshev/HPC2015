//
//  native_virtual.h
//  GPAPI
//
//  Created by savage309 on 9.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef __GPAPI__native_virtual__
#define __GPAPI__native_virtual__

#include <stdio.h>
namespace Virtual {
    class NBase {
    public:
        virtual int get() const = 0;
        virtual void set(int val) = 0;
    };
    
    class NDerived : public NBase {
        int i;
    public:
        int get() const override {
            return i;
        }
        void set(int val) override {
            i = val;
        }
    };
    
    class NDerived2 : public NBase {
        float f;
    public:
        int get() const override {
            return f;
        }
        void set(int val) override {
            f = val * 42.196f;
        }
    };
}
#endif /* defined(__GPAPI__native_virtual__) */
