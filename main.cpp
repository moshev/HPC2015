#include <iostream>
#include <cmath>
#include <memory>

#include <algorithm>

#include "common.h"
#include "diffclock.h"

#include "test_cache_miss.hpp"
#include "test_pointer_alias/test_pointer_alias.h"
#include "test_SoA.hpp"
#include "test_inline/test_inline.h"
#include "test_data_oriented_design.hpp"
#include "test_float_double.hpp"
#include "test_virtual/test_virtual.h"
#include "test_ilp.h"
#include "test_threads.h"
#include "test_image.h"
#include "test_factoriel.hpp"

benchpress::registration* benchpress::registration::d_this;

using namespace std;
int main(int argc, const char * argv[]) {
    std::cout << "Starting tests ...\n" << std::endl;
    
    
    auto t0 = getTime();
    
    /* #0 Instruction level parallelism*/
    ILP::test();
    std::cout << "\n **** \n\n";

    /* #1 Cache misses */
    Image::test();
    std::cout << "\n **** \n\n";
    
    Virtual::test();
    std::cout << "\n **** \n\n";
    
    CacheMiss::test();
    std::cout << "\n **** \n\n";
    
    /* #2 Pointer alliasing */
    PointerAlias::test();
    std::cout << "\n **** \n\n";
    
    /* #3 Compiler related */
    Inline::test();
    std::cout << "\n **** \n\n";
    
    Factoriel::test();
    std::cout << "\n **** \n\n";
    
    FloatDouble::test();
    std::cout << "\n **** \n\n";
    
    /* Data oriented design */
    
    DataOrientedDesign::test();
    std::cout << "\n **** \n\n";

    SoA::test();
    std::cout << "\n **** \n\n";
    
    Threads::test();
    std::cout << "\n **** \n\n";
    auto t1 = getTime();
    
    std::cout << "Tests completed, time " << diffclock(t1, t0) << "s" << std::endl;
    
    return 0;
}
#if 0

for (i=0; i<NRUNS; i++)
    for (j=0; j<size; j++)
        array[j] = 2.3*array[j]+1.2;

for (b=0; b<size/l1size; b++) {
    blockstart = 0;
    for (i=0; i<NRUNS; i++) {
        for (j=0; j<l1size; j++)
            array[blockstart+j] = 2.3*array[blockstart+j]+1.2;
    }
    blockstart += l1size;
}

///TLB
#define INDEX(i,j,m,n) i+j*m
array = (double*) malloc(m*n*sizeof(double));
/* traversal #1 */
for (j=0; j<n; j++)
    for (i=0; i<m; i++)
        array[INDEX(i,j,m,n)] = array[INDEX(i,j,m,n)]+1;
/* traversal #2 */
for (i=0; i<m; i++)
    for (j=0; j<n; j++)
        array[INDEX(i,j,m,n)] = array[INDEX(i,j,m,n)]+1;

#endif
//shared memory & shared nothing
//concurrency & parallelism
//subroutines & couroutines
//test_and_set
//amdahls law
//car-passenger observation
//coffman conditions
//foster methodology
//flynn taxonomy
//conditional variables / monitors
//spin lock, read-write lock, mutex, concurrent collections
//roofline model

