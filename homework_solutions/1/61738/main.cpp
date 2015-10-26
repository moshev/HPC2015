#include <stdio.h>
#include <string>
#include <bitset>

//converts the passed string (which is number in binary)
//to real number, which is less than 1
float Convert(std::string& str)
{
	float sum = 0;

	int size = str.size();
	for (int i = 0; i < size; i++)
	{
		if ( str[i] == '1')
		{
			int power = i + 1;
			//the power is infact negative, that's why
			//divide 1 by the result
			sum += 1.0 / (1 << power);
		}
	}

	return sum;
}

//////////////////////////////////////////////////////////////////

void ReverseString(std::string& str)
{
	int size = str.size();
	for (int i = 0; i < size / 2; i++)
	{
		char temp = str[i];
		str[i] = str[size - i - 1];
		str[size - i - 1] = temp;
	}
}

//////////////////////////////////////////////////////////////////

float derCorput(unsigned i)
{
	//convert passed number to binary
	std::string b = std::bitset<32>(i).to_string();

	//reverse the binary string
	ReverseString(b);

	//convert the binary string to number smaller than 1
	return Convert(b);
}

//////////////////////////////////////////////////////////////////

int main()
{
	for (auto i = 0; i < 100; ++i)
		printf("%f, ", derCorput(i));

	return 0;
}


