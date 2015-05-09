#include <iostream>
#include <cmath>
#include <memory>

#include "diffclock.h"

#include "test_cache_miss.hpp"
#include "test_pointer_alias/test_pointer_alias.h"
#include "test_SoA.hpp"
#include "test_inline/test_inline.h"
#include "test_data_oriented_design.hpp"
#include "test_float_double.hpp"

#include "test_virtual/test_virtual.h"


int main(int argc, const char * argv[]) {
    std::cout << "Starting tests ...\n" << std::endl;
    Virtual::test();
    FloatDouble::test();
    DataOrientedDesign::test();
    SoA::test();
    Inline::test();
    PointerAlias::test();
    CacheMiss::test();
    std::cout << "Tests completed."  << std::endl;
    return 0;
}


