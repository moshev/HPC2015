//
//  pointer_aliash.cpp
//  GPAPI
//
//  Created by savage309 on 6.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//
#include "test_pointer_alias_impl.h"
namespace PointerAlias {
    struct A{};
    struct B{};
    inline constexpr size_t getTestSize() { return 65000000;//65m since js can't process more (at least node.js can't) 300000000;;};
    }
    void test();
}