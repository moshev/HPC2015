#include <iostream>
using namespace std;

float derCorput(unsigned n)
{
	float result = 0.0;
	float exponent = 1;
	while (n>0)
	{
		result = result + ((n % 2) / (exponent*=2));
		n /= 2;
	}
	return result;
	
}

int main()
{
	for (int i = 0; i < 20; ++i)
    cout<<derCorput(i)<<" ,";
	cout<<endl;
	return 0;
}