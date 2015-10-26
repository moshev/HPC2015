#include <stdio.h> 
#include <cmath> 
#include <iomanip> 

#include <bitset> 

using namespace std; 

float* p; 

float* generateArray() 
{ 
    float* array = new float[32]; 
    for (int i = 0; i < 32; ++i) 
    { 
        array[i] = pow(0.5, i + 1); 
    } 
     
    return array; 
} 

float derCorput(unsigned int number) 
{     
    float sum = 0; 
    static bool b = false;
    if (!b) {
        b = true;
        p = generateArray();
    }
    for (int i = 31; i >= 0; --i) 
    { 
        int bit = ((number & (1 << i)) >> i); 
        if (bit) 
        { 
            sum += p[i]; 
        } 
    } 
     
    return sum; 
} 

int main() 
{ 
    p = generateArray(); 

    for (int i = 0; i < 20; ++i) 
    { 
        printf("%f ", derCorput(i));
    } 
     
   return 0; 
} 