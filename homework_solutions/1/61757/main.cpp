#include <iostream>
#include <bitset>
#include <cmath>

float derCorput(unsigned n){

    std::bitset<32> bit_representation_n = std::bitset<32>(n);
    int important_bits = std::log2(n) + 1;
    float derCorput_number = 0;

    for(int i=0; i<important_bits; i++){

        derCorput_number += pow(2, -(i+1)) * bit_representation_n[i];
    }

    return derCorput_number;
}
