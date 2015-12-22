#include<string>
#include<iostream>
#include<vector>
#include<sstream>
#include<set>
#include<algorithm>

constexpr float G = 0.0000000000667f;

#pragma warning( disable : 4996)

struct Vector3D
{
	float x, y, z;

	Vector3D() :x(0), y(0), z(0)
	{}

	Vector3D(float _x, float _y, float _z):x(_x),y(_y),z(_z)
	{}

	Vector3D operator-(const Vector3D& other) const
	{
		Vector3D temp;
		temp.x = other.x - x;
		temp.y = other.y - y;
		temp.z = other.z - z;

		return temp;
	}

	float operator*(const Vector3D& other) const
	{
		return x*other.x+y*other.y + z*other.z;
	}

	Vector3D operator+=(const Vector3D& other)
	{
		x += other.x;
		y += other.y;
		z += other.z;

		return *this;
	}

	Vector3D operator+(const Vector3D& other) const
	{
		return Vector3D(x + other.x, y + other.y, z + other.z);
	}

	Vector3D operator*(float scalar) const
	{
		Vector3D temp;
		temp.x = scalar*x;
		temp.y = scalar*y;
		temp.z = scalar*z;

		return temp;
	}

	Vector3D operator/(float scalar) const
	{
		Vector3D temp;
		temp.x = x/ scalar;
		temp.y = y/ scalar;
		temp.z = z/ scalar;

		return temp;
	}

	float Length() const
	{
		return sqrt(x*x + y*y + z*z);
	}

	Vector3D Normalize() const 
	{
		Vector3D temp(x,y,z);
		float length = temp.Length();
		temp.x /= length;
		temp.y /= length;
		temp.z /= length;

		return temp;
	}

	float Distance(const Vector3D& vec) const
	{
		return sqrt((x-vec.x)*(x - vec.x) + (y - vec.y)*(y - vec.y) + (z - vec.z)*(z - vec.z));
	}

	friend Vector3D operator*(float scalar, const Vector3D& vec);
};

Vector3D operator*(float scalar,const Vector3D& vec)
{
	Vector3D temp;
	temp.x = scalar*vec.x;
	temp.y = scalar*vec.y;
	temp.z = scalar*vec.z;

	return temp;
}

Vector3D parseVector(const std::string& str)
{
	Vector3D temp;

	std::istringstream iss(str);
	std::string s;
	getline(iss, s, ' ');
	temp.x = std::stof(s);
	getline(iss, s, ' ');
	temp.y = std::stof(s);
	getline(iss, s, ' ');
	temp.z = std::stof(s);

	return temp;
}

struct Destruction
{
	float time;
	int destructor;
	int destructed;
};

struct Collision
{
	float time;
	int body1;
	int body2; 
};

struct BodyPosition
{
	int body;
	Vector3D position;
};

struct BodyMass
{
	int body;
	float mass;
};

struct BodySize
{
	int body;
	float size;
};

struct BodySpeed
{
	int body;
	Vector3D speed;
};

struct BodyAcceleration
{
	int body;
	Vector3D acceleration;
};

struct BodyFuel
{
	int body;
	float fuel;
};

struct BodyConsumption
{
	int body;
	float consumption;
};

struct BodyRange
{
	int body;
	float range;
};

struct BoundingSphere
{
	Vector3D pos;
	float radius;
};

enum class Type
{
	Planet,
	Asteroid,
	DeathStar,
	XWing,
	Unknown
};

struct BodyType
{
	int body;
	Type type;
};

struct BodyBoundingSphere
{
	int body;
	BoundingSphere boundingSphere;
};

bool Collide(BoundingSphere& sphere1, BoundingSphere& sphere2)
{
	float distance = sphere1.pos.Distance(sphere2.pos);

	if (distance > (sphere1.radius + sphere2.radius))
	{
		return false;
	}

	return true;
}

bool IsInRange(BoundingSphere& sphere1, float range, BoundingSphere& sphere2)
{
	float distance = sphere1.pos.Distance(sphere2.pos);

	if (distance > (sphere1.radius + sphere2.radius + range))
	{
		return false;
	}

	return true;
}

Type GetType(const std::string& type)
{
	if( !type.compare("Planet") )
	{
		return Type::Planet;
	}
	else if(!type.compare("Asteroid"))
	{
		return Type::Asteroid;
	}
	else if(!type.compare("Death Star"))
	{
		return Type::DeathStar;
	}
	else if(!type.compare("X-Wing"))
	{
		return Type::XWing;
	}
	else
	{
		return Type::Unknown;
	}
}

std::string GetType(Type& type)
{
	if (type == Type::Planet)
	{
		return "Planet";
	}

	if (type == Type::Asteroid)
	{
		return "Asteroid";
	}
	
	if (type == Type::Planet)
	{
		return "Planet";
	}
	
	if (type == Type::DeathStar)
	{
		return "Death Star";
	}
	
	if(type == Type::XWing)
	{
		return "X-Wing";
	}

	return "Unknown";
}

std::vector<std::string> ParseLine(std::string line,std::string delimeters)
{
	std::vector<std::string> tokens;

	std::size_t prev = 0, pos;
	while ((pos = line.find_first_of(delimeters, prev)) != std::string::npos)
	{
		if (pos > prev)
		{
			tokens.push_back(line.substr(prev, pos - prev));
		}
		prev = pos + 1;
	}
	if (prev < line.length())
	{
		tokens.push_back(line.substr(prev, std::string::npos));
	}

	tokens.erase(remove_if(tokens.begin(), tokens.end(),[](std::string& str1)
														{
															return str1.empty() || str1 == " ";
														}), tokens.end());

	return tokens;
}

struct Result
{
	Destruction* destructions;
	Collision* collisions;
	BodyPosition* positions;
	BodyType* types;
	BodySpeed* speeds;
	BodyMass* masses;
	BodySize* sizes;
	BodyAcceleration* acceletations;
	BodyConsumption* consumptions;
	BodyFuel* fuels;
	BodyRange* ranges;
	BodyBoundingSphere* boundingSpheres;

	int destructions_count;
	int collisions_count;
	int positions_count;
};

Result* res = new Result;

void Universe_Initialize(const char* file)
{
	FILE* fp = fopen(file, "r");
	
	std::vector<std::string> data;

	char line[256];
	//store all the lines from the file
	while (fgets(line, sizeof(line), fp))
	{
		data.push_back(line);
	}

	//remove \n in the lines
	for(auto& line1: data)
	{
		auto pos = line1.find("\n");
		if (pos != std::string::npos )
		{
			line1 = line1.substr(0, pos);
		}
	}

	int amountOfObjects = std::stoi(data[0]);

	res->collisions		 = new Collision[amountOfObjects / 2];
	res->destructions	 = new Destruction[amountOfObjects - 1];

	res->positions		 = new BodyPosition[amountOfObjects];
	res->speeds			 = new BodySpeed[amountOfObjects];
	res->masses			 = new BodyMass[amountOfObjects];
	res->types			 = new BodyType[amountOfObjects];
	res->sizes			 = new BodySize[amountOfObjects];
	res->acceletations	 = new BodyAcceleration[amountOfObjects];
	res->consumptions	 = new BodyConsumption[amountOfObjects];
	res->fuels			 = new BodyFuel[amountOfObjects];
	res->ranges			 = new BodyRange[amountOfObjects];
	res->boundingSpheres = new BodyBoundingSphere[amountOfObjects];

	for (int i = 1; i < data.size(); i++)
	{
		std::vector<std::string> tokens = ParseLine(data[i],"<>[]");

		res->positions_count = amountOfObjects;
		res->collisions_count = 0;
		res->destructions_count = 0;

		int id = std::stoi(tokens[0]);
		res->positions[i - 1].body = id;
		res->speeds[i - 1].body    = id;
		res->masses[i - 1].body    = id;
		res->types[i - 1].body     = id;
		res->boundingSpheres[i - 1].body = id;

		Type t = GetType(tokens[1]);
		res->types[i - 1].type			= t;
		res->masses[i - 1].mass			= std::stof(tokens[2]);
		res->sizes[i - 1].size			= std::stof(tokens[3]);
		res->positions[i - 1].position  = parseVector(tokens[4]);
		res->speeds[i - 1].speed		= parseVector(tokens[5]);

		if( t == Type::XWing || t == Type::DeathStar )
		{
			res->acceletations[i - 1].body = id;
			res->consumptions[i - 1].body = id;
			res->fuels[i - 1].body = id;
			res->ranges[i - 1].body = id;

			res->acceletations[i - 1].acceleration  = parseVector(tokens[6]);
			res->fuels[i - 1].fuel					= std::stof(tokens[7]);
			res->consumptions[i - 1].consumption    = std::stof(tokens[8]);
			res->ranges[i - 1].range				= std::stof(tokens[9]);
		}
		else
		{
			res->acceletations[i - 1].body = 0;
			res->consumptions[i - 1].body = 0;
			res->fuels[i - 1].body = 0;
			res->ranges[i - 1].body = 0;
		}

		//set the data for the bounding sphere
		res->boundingSpheres[i - 1].boundingSphere.pos    = res->positions[i - 1].position;
		res->boundingSpheres[i - 1].boundingSphere.radius = res->sizes[i - 1].size;
	}
	
	fclose(fp);
}

void ObjectDestroyed(Result* result, Type destructorType, int objIndex1, int objIndex2, float t,std::set<int>& destroyedIndices)
{
	int destructorIndex;
	int destructedIndex;

	if (result->types[objIndex1].type == destructorType)
	{
		destructorIndex = objIndex1;
		destructedIndex = objIndex2;
	}
	else
	{
		destructorIndex = objIndex2;
		destructedIndex = objIndex1;
	}
	int index = result->destructions_count++;
	result->destructions[index].destructor = result->types[destructorIndex].body;
	result->destructions[index].destructed = result->types[destructedIndex].body;
	result->destructions[index].time = t;

	std::cout << GetType(result->types[destructorIndex].type) << " id:" << result->types[destructorIndex].body << " destroyed "
		<< GetType(result->types[destructedIndex].type) << " id:" << result->types[destructedIndex].body << std::endl;

	destroyedIndices.insert(destructedIndex);
}

void SwapData(Result* result, int index1, int index2)
{
	std::swap(result->positions[index1], result->positions[index2]);
	std::swap(result->boundingSpheres[index1], result->boundingSpheres[index2]);
	std::swap(result->types[index1], result->types[index2]);
	std::swap(result->sizes[index1], result->sizes[index2]);
	std::swap(result->masses[index1], result->masses[index2]);
	std::swap(result->speeds[index1], result->speeds[index2]);
	std::swap(result->acceletations[index1], result->acceletations[index2]);
	std::swap(result->consumptions[index1], result->consumptions[index2]);
	std::swap(result->fuels[index1], result->fuels[index2]);
	std::swap(result->ranges[index1], result->ranges[index2]);
}

void UpdatePositions(Result* result,int aliveObjectsAmount, float delta)
{
	for (int i = 0; i < aliveObjectsAmount; ++i)
	{
		Vector3D F_i;

		for (int j = 0; j < aliveObjectsAmount; ++j)
		{
			if (i != j)
			{
				Vector3D r_ij = result->positions[j].position - result->positions[i].position;
				Vector3D n_ij = r_ij.Normalize();

				Vector3D F_ij = ((G * result->masses[i].mass * result->masses[j].mass) / (r_ij * r_ij)) * n_ij;

				F_i += F_ij;
			}
		}

		Vector3D a = F_i / result->masses[i].mass;

		Vector3D newPosition = result->speeds[i].speed * delta + (a * delta * delta) / 2;

		result->positions[i].position += newPosition;
		result->boundingSpheres[i].boundingSphere.pos = result->positions[i].position;
	}
}

void UpdateCollisionsAndDestructions(Result* result, int aliveObjectsAmount, float t)
{
	//store the unique indices of the collided objects
	std::set<int> collidedIndices;

	//store the unique indices of the destroyed objects
	std::set<int> destroyedIndices;

	//check for collisions, death-starts, etc.
	for (int i = 0; i < aliveObjectsAmount; ++i)
	{
		for (int j = 0; j < aliveObjectsAmount; ++j)
		{
			if (i != j &&
				std::find(collidedIndices.begin(), collidedIndices.end(), i) == collidedIndices.end() &&
				std::find(collidedIndices.begin(), collidedIndices.end(), j) == collidedIndices.end() &&
				std::find(destroyedIndices.begin(), destroyedIndices.end(), i) == destroyedIndices.end() &&
				std::find(destroyedIndices.begin(), destroyedIndices.end(), j) == destroyedIndices.end())
			{
				BoundingSphere& BSphere1 = result->boundingSpheres[i].boundingSphere;
				BoundingSphere& BSphere2 = result->boundingSpheres[j].boundingSphere;
				Type& Type1 = result->types[i].type;
				Type& Type2 = result->types[j].type;

				//check if planet or asteroid is in the range of death star
				if (Type1 == Type::DeathStar && (Type2 == Type::Planet || Type2 == Type::Asteroid) &&
					IsInRange(BSphere1, result->ranges[i].range, BSphere2) ||
					(Type1 == Type::Planet || Type1 == Type::Asteroid) && Type2 == Type::DeathStar &&
					IsInRange(BSphere2, result->ranges[j].range, BSphere1))
				{
					ObjectDestroyed(result, Type::DeathStar, i, j, t, destroyedIndices);
				}

				//check if asteroid or death star is in the range of x-wing
				if (Type1 == Type::XWing && (Type2 == Type::DeathStar || Type2 == Type::Asteroid) &&
					IsInRange(BSphere1, result->ranges[i].range, BSphere2) ||
					(Type1 == Type::DeathStar || Type1 == Type::Asteroid) && Type2 == Type::XWing &&
					IsInRange(BSphere2, result->ranges[j].range, BSphere1))
				{
					//i or j might have been destroyed in the previous check
					if (std::find(destroyedIndices.begin(), destroyedIndices.end(), i) == destroyedIndices.end() &&
						std::find(destroyedIndices.begin(), destroyedIndices.end(), j) == destroyedIndices.end())
					{
						ObjectDestroyed(result, Type::XWing, i, j, t, destroyedIndices);
					}
				}

				//check if 2 objects collide
				if (Collide(BSphere1, BSphere2))
				{
					int index = result->collisions_count++;
					result->collisions[index].body1 = result->boundingSpheres[i].body;
					result->collisions[index].body2 = result->boundingSpheres[j].body;
					result->collisions[index].time = t;
					std::cout << "Collision between " << GetType(result->types[i].type) << " id:" << result->types[i].body <<
						" and " << GetType(result->types[j].type) << " id:" << result->types[j].body << std::endl;

					collidedIndices.insert(i);
					collidedIndices.insert(j);
				}
			}
		}
	}

	//swap collided objects with alive objects
	int collidedCounter = 0;
	for (auto it = collidedIndices.rbegin(); it != collidedIndices.rend(); ++it)
	{
		int aliveIndex = aliveObjectsAmount - collidedCounter - 1;
		int currentIndex = (*it);

		SwapData(result, currentIndex, aliveIndex);
		collidedCounter++;
	}

	//swap destroyed objects with alive objects
	int destroyedCounter = 0;
	for (auto it = destroyedIndices.rbegin(); it != destroyedIndices.rend(); ++it)
	{
		int aliveIndex = aliveObjectsAmount - collidedCounter - destroyedCounter - 1;
		int currentIndex = (*it);

		SwapData(result, currentIndex, aliveIndex);
		destroyedCounter++;
	}
}

void Universe_Run(float time, float delta, Result* result)
{
	for (float t = 0; t < time; t+=delta)
	{
		//if we have 1 collision between 2 objects, 2 objects died
		int aliveObjectsAmount = result->positions_count - result->collisions_count*2 - result->destructions_count;

		UpdatePositions(result, aliveObjectsAmount, delta);
		UpdateCollisionsAndDestructions(result, aliveObjectsAmount, t);
	}
}

int main()
{
	Universe_Initialize("data.txt");

	auto delta = 0.001f;
	auto time = 100.0f;

	Universe_Run(time, delta, res);

	return 0;
}