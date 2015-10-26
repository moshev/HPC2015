#include <iostream>


//Using the simplified explanation for the way a member of the sequence is calculated,
//given here:   http://rosettacode.org/wiki/Van_der_Corput_sequence
float derCorput(unsigned n)
{
	float currentMultiplier = 0.5f;
	float nthMember = 0.0f;

	while(n != 0)
	{
		nthMember += currentMultiplier * (n % 2);
		currentMultiplier *= 0.5;
        n /= 2 ;
	}

	return nthMember;
}


int main()
{
	return 0;
}
