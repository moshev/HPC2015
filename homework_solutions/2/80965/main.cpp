#include "immintrin.h"
#include <cstdlib>

void reverse (char* bytes, int numChunks)
{
	char* res;
	res=(char*)malloc(64);
	for(int n=0;n<numChunks;++n)
	{__m256i a = _mm256_set_epi8(bytes[0+64*n], bytes[1+64*n], bytes[2+64*n],bytes[3+64*n],bytes[4+64*n],bytes[5+64*n],bytes[6+64*n],bytes[7+64*n],bytes[8+64*n],bytes[9+64*n],bytes[10+64*n],bytes[11+64*n],bytes[12+64*n],bytes[13+64*n],bytes[14+64*n],bytes[15+64*n],bytes[16+64*n],bytes[17+64*n],bytes[18+64*n],bytes[19+64*n],bytes[20+64*n],bytes[21+64*n],bytes[22+64*n],bytes[23+64*n],bytes[24+64*n],bytes[25+64*n],bytes[26+64*n],bytes[27+64*n],bytes[28+64*n],bytes[29+64*n],bytes[30+64*n],bytes[31+64*n]);
	__m256i b = _mm256_set_epi8(bytes[32+64*n],bytes[33+64*n],bytes[34+64*n],bytes[35+64*n],bytes[36+64*n],bytes[37+64*n],bytes[38+64*n],bytes[39+64*n],bytes[40+64*n],bytes[41+64*n],bytes[42+64*n],bytes[43+64*n],bytes[44+64*n],bytes[45+64*n],bytes[46+64*n],bytes[47+64*n],bytes[48],bytes[49+64*n],bytes[50+64*n],bytes[51+64*n],bytes[52+64*n],bytes[53+64*n],bytes[54+64*n],bytes[55+64*n],bytes[56+64*n],bytes[57+64*n],bytes[58+64*n],bytes[59+64*n],bytes[60+64*n],bytes[61+64*n],bytes[62+64*n],bytes[63+64*n]);
	//Когато имаш идеи, но нямаш време...
	char* bt;
	bt = (char*)malloc(32);
	bt = (char*) &b;
	int i=0;
	while(i!=32)
	{
		bytes[i+64*n]=bt[i];
		i++;
	}

	char* s;
	s = (char*)malloc(32);
	s = (char*) &a;
	i=0;
	while(i!=32)
	{
		bytes[32+64*n+i]=s[i];
		i++;
	}
	}
}
