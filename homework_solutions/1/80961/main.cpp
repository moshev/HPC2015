float derCorput(unsigned n) {
    
    float sum = 0;
    for(int bit = 1, pow2 = 0; bit <= n; bit <<= 1, pow2++) {
        if(n & bit) {
            sum += 1.0 / (1 << (pow2+1));
        }
    }

    return sum;
}
