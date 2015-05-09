//
//  handmade_virtual.cpp
//  GPAPI
//
//  Created by savage309 on 9.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#include "handmade_virtual.h"

//custom vtable that uses 8 bits for vtable ptr, in stead of 64 bits
//the vtable ptr is first in the mem layout, aways there and aways giving us that +1 cache miss
//this vtable implementation is slimmer in terms of memory and as fast as c++ vtable (this two facts = (probably) faster code)
//it is ugly, but that should not scares you
//there may be some (macro) redudancy, but this is just a PoC
//credits : a.alexandrescu, going native 2013.

#include <iostream>
namespace Virtual {
//every base class has :
//1. tag - every hierarchy class writes its id there
//2. static vtable

//every hierarchy class has :
//1. static ID


void init/*vtable*/() {
    //init the vtable for "Base" class
    REGISTER_METHOD1(METHOD_SET,
                     INTERFACE(Base),
                     CLASS(Base),
                     RETURN(void),
                     NAME(set),
                     PARAM0(int)
                     );
    REGISTER_METHOD0(METHOD_GET,
                     INTERFACE(Base),
                     CLASS(Base),
                     RETURN(int),
                     NAME(get)
                     );
    // ***
    //init the vtable for "Derived" class
    REGISTER_METHOD0(METHOD_GET,
                     INTERFACE(Base),
                     CLASS(Derived),
                     RETURN(int),
                     NAME(get)
                     );
    REGISTER_METHOD1(METHOD_SET,
                     INTERFACE(Base),
                     CLASS(Derived),
                     RETURN(void),
                     NAME(set),
                     PARAM0(int)
                     );
    // ***
    //init the vtable for "Derived2" class
    REGISTER_METHOD0(METHOD_GET,
                     INTERFACE(Base),
                     CLASS(Derived2),
                     RETURN(int),
                     NAME(get)
                     );
    REGISTER_METHOD1(METHOD_SET,
                     INTERFACE(Base),
                     CLASS(Derived2),
                     RETURN(void),
                     NAME(set),
                     PARAM0(int));
}

Base::FP Base::vtbl[(CLASS_LAST+1) * METHOD_LAST];

int test(int argc, const char* argv[]) {
    init();
    
    Base* der = new Derived2();
    
    der->set(196);
    der->get();
    
    return 0;
}
}