const int BASE = 2;

float derCorput(unsigned n)
{
	float result = 0.0f;
	int i = 1;
	while (n)
	{
		result += static_cast<float>(n%BASE) / (1 << i); // d_k(n)*( b^(-k-1) )
		n /= BASE;
		++i;
	}

	return result;
}