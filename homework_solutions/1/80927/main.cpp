inline float derCorput(unsigned n) {
    unsigned denumerator = 2;
    float dc = 0;
    float dc2 = 0;
    float dc3 = 0;
    float dc4 = 0;
    float dc5 = 0;

    while (n) {
        dc += float(n & 1) / denumerator;
        dc2 += float((n >> 1) & 1) / (denumerator << 1);
        dc3 += float((n >> 2) & 1) / (denumerator << 2);
        dc4 += float((n >> 3) & 1) / (denumerator << 3);
        dc5 += float((n >> 4) & 1) / (denumerator << 4);
        denumerator <<= 5;
        n >>= 5;
    }

    return dc + dc2 + dc3 + dc4 + dc5;
}
