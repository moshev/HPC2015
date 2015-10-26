
float derCorput(unsigned m)
{
#ifdef _MSC_VER
    m = _byteswap_ulong(m);
#else
    m = __builtin_bswap32(m);
#endif
    m = ((m & 0x0f0f0f0f) << 4) | ((m & 0xf0f0f0f0) >> 4);
    m = ((m & 0x33333333) << 2) | ((m & 0xcccccccc) >> 2);
    m = ((m & 0x55555555) << 1) | ((m & 0xaaaaaaaa) >> 1);
    
    return float(m) / float(0x100000000LL);
}
