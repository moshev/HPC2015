#include <cstdlib>
#include <cstdio>

//needs x64 :D
float derCorput(unsigned n) {
	union {
		unsigned int val;
		unsigned char pieces[4];
	} rot;
	rot.val = n;

	//black magic start here
	
	for(int i=0; i<4; ++i)
		rot.pieces[i]=(unsigned char)(((rot.pieces[i] * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32);

	unsigned char temp = rot.pieces[0];
	rot.pieces[0] = rot.pieces[3];
	rot.pieces[3] = temp;
	temp = rot.pieces[1];
	rot.pieces[1] = rot.pieces[2];
	rot.pieces[2] = temp;

	union {
		double result;
		unsigned long long bit64;
	} gen;
	gen.result = 1.0;
	gen.bit64 |= 0x000FFFFFFFFFFFFFULL & (static_cast<unsigned long long>(rot.val) << 20); 

	return (float)(gen.result - 1.0);
	//black magic ends here
}

//int main() {
//	for (auto i = 0; i < 20; ++i)
//		printf("%f, ", derCorput(i));
//	system("pause");
//	return 0;
//}