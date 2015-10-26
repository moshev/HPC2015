
static const unsigned char BitReverseTable256[256] = 
{
#   define R2(n)    n,     n + 2*64,     n + 1*64,     n + 3*64
#   define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#   define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
    R6(0), R6(2), R6(1), R6(3)
};

static const char LogTable256[256] = 
{
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
};

float derCorput(unsigned n)
{
	if( n == 0 )
		return 0.0f;
	
	// start with 00000(10110)
	// https://graphics.stanford.edu/~seander/bithacks.html#BitReverseTable
	unsigned reversedN;
	
	unsigned char* p = (unsigned char*)&n;
	unsigned char* q = (unsigned char*)&reversedN;

	q[3] = BitReverseTable256[p[0]];
	q[2] = BitReverseTable256[p[1]];
	q[1] = BitReverseTable256[p[2]];
	q[0] = BitReverseTable256[p[3]];
	// we get (01101)00000
	
	// find the number of digits used
	// https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogLookup
	unsigned log2ofN;
	unsigned int t, tt;

	if( (tt = n >> 16) )
		log2ofN = (t = tt >> 8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
	else 
		log2ofN = (t = n >> 8) ? 8 + LogTable256[t] : LogTable256[n];
	
	unsigned digitsUsed = log2ofN + 1;
	
	// shift back into position
	reversedN >>= (8 * sizeof(unsigned) - digitsUsed);
	// we get 00000(01101)
	
	// convert to decimal float
	float f = reversedN / float(2 << (digitsUsed-1));
	
	return f < 0.0f ? -f : f;
}
int main(){}
