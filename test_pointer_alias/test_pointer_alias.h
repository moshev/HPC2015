#pragma once

//
#include "test_pointer_alias_impl.h"
namespace PointerAlias {
    struct A{};
    struct B{};
    inline constexpr size_t getTestSize() { return 100000000;//65m since js can't process more (at least node.js can't) 300000000;;};
    }
    void test();
}