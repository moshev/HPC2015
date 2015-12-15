//#include <cstdio>

void reverse(char* bytes, int numChunks) {
	int part = 0;

	for(int j=0;j<numChunks; ++j) {	
		for (int i = 0; i < 32; ++i) {

			int left = i + part;
			int right = part + 64 - i - 1;
			int tmp = bytes[left];

			bytes[left] = bytes[right];
			bytes[right] = tmp;
		}

		part += 64;
	}
}

int main() {
	const int len = 64*16099;
	char bytes[len];
	for (int i = 0; i < len; ++i)
	   bytes[i] = i;

	reverse((char*)bytes, (len/64));
	//printf("%i %i %i %i\n", bytes[0], bytes[1], bytes[len - 1], bytes[len - 2]);

	return 0;
}