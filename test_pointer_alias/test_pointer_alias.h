#pragma once

//
#include "test_pointer_alias_impl.h"
namespace PointerAlias {
    struct A{};
    struct B{};
    inline constexpr size_t getTestSize() { return 100000000;    }
    void test();
}