#if __cplusplus >= 201103L
#   define CPP11
#endif

#if __cplusplus >= 201103L && (defined __clang__)
#   define UNROLL _Pragma("unroll")
#else
#   define UNROLL
#endif

#include <ctime>
#include <ratio>
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;

float derCorput(unsigned n)
{
    float sum = 0;
    
    UNROLL
    for(int k = 0; k < sizeof(unsigned) * 8; ++k)
    {
        sum +=  ((n >> k) & 1) ? (1.0f / (2 << k )): 0.0f;
    }
    return sum;
}

int main(int argc, const char * argv[]) {
    for (auto i = 0; i < 2<<19; ++i)    
        printf("%f, ", derCorput(i));
            
    
    
    return 0;
}
