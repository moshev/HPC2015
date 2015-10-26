#include <iostream>
#include <stdio.h>
using namespace std;

float memoize[32] = {0,};

int mem() {
  
    for(unsigned int i = 0; i < 32; ++i) {
        memoize[i] = 1.f / (unsigned int)(1 << i);
    }
    
    
    
    return 0;
}


float getFraction(unsigned n, unsigned base) {
    static bool b = false;
    if (!b) {
        mem();
        b = true;
    }
	float fraction = 0;
	int index = 1;
	char bit;
	while(n > 0) {
        bit = n % base;
        n /= base;

        if(bit) {
            fraction += memoize[index];
        }
        ++index;
	}

	return fraction;
}

float derCorput(unsigned n) {
    if(n == 0) {
        return 0.f;
    }
	return getFraction(n, 2);
}


