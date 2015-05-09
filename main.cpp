#include <iostream>
#include <cmath>
#include <memory>

#include "diffclock.h"

#include "test_cache_miss.hpp"
#include "test_pointer_alias/test_pointer_alias.h"
#include "test_SoA.hpp"
#include "test_inline/test_inline.h"
#include "test_data_oriented_design.hpp"

int main(int argc, const char * argv[]) {
    srand (0);
    std::cout << "Starting tests ..." << std::endl;
    DataOrientedDesign::test();
    SoA::test();
    Inline::test();
    PointerAlias::test();
    CacheMiss::test();
    std::cout << "Tests completed."  << std::endl;
    return 0;
}


