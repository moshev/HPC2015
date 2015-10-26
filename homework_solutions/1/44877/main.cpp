//
//  VanDerCorputSeqence.cpp
//  Test
//
//  Created by Iliyan Kafedzhiev on 10/17/15.
//  Copyright © 2015 Iliyan Kafedzhiev. All rights reserved.
//

#include <cmath>
#include <iostream>
#include <stdio.h>

double derCorput(int n, int base = 2)
{
	double vdc = 0, denom = 1;
	
	while (n)
	{
		vdc += fmod(n, base) / (denom = denom * base);
		n = n / base;
	}

	return vdc;
}

int main()
{
	for (int n = 0; n < 20; ++n)
	{
		printf("%f, ", derCorput(n));
	}
	  
	return 0;
}
