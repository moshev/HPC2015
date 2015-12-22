#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

using namespace std;

#define G (6.674 * 1e-17) //gravitational constant in N*km2/kg (not in N*m2/kg)

//Vector3D
struct Vector3D
{
    float x, y, z;

	Vector3D() : x(0.0), y(0.0), z(0.0) {};
	Vector3D(float xCoord, float yCoord, float zCoord) : x(xCoord), y(yCoord), z(zCoord) {}

	Vector3D operator + (const Vector3D& other) const;
    Vector3D operator - (const Vector3D& other) const;
    Vector3D& operator += (const Vector3D& other);
    Vector3D& operator -= (const Vector3D& other);


	float Length() const;
    Vector3D Normalized() const; //returns the corresponding normalized vector
    Vector3D Opposite() const; //returns the opposite vector
    float ScalarProduct(const Vector3D& other) const; //returns the scalar product of this vector and the argument vector
    Vector3D Scaled(float scalar) const; // returns the vector, scaled by scalar
};


Vector3D Vector3D::operator + (const Vector3D& other) const
{
	return Vector3D(this->x + other.x , this->y + other.y, this->z + other.z);
}


Vector3D Vector3D::operator - (const Vector3D& other) const
{
	return Vector3D(this->x - other.x , this->y - other.y, this->z - other.z);
}


Vector3D& Vector3D::operator += (const Vector3D& other)
{
	this->x += other.x;
	this->y += other.y;
	this->z += other.z;

	return *this;
}


Vector3D& Vector3D::operator -= (const Vector3D& other)
{
	this->x -= other.x;
	this->y -= other.y;
	this->z -= other.z;

	return *this;
}


float Vector3D::Length() const
{
    return sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
}


Vector3D Vector3D::Normalized() const
{
	float length = this->Length();

	return (length == 0.0 ? *this : Vector3D(this->x / length, this->y / length, this->z / length));
}


Vector3D Vector3D::Opposite() const
{
	return Vector3D(-this->x, -this->y, -this->z);
}


float Vector3D::ScalarProduct(const Vector3D& other) const
{
	return (this->x * other.x) + (this->y * other.y) + (this->z * other.z);
}


Vector3D Vector3D::Scaled(float scalar) const
{
	return Vector3D(scalar * this->x, scalar * this->y, scalar * this->z);
}


//calculates the distance between vectors a and b
float CalculateDistance(const Vector3D& a, const Vector3D& b)
{
    return (b - a).Length();
}


//Destruction
struct Destruction
{
    float time; // timestamp of the destruction
    int destructor; // id of the Death Star or X-Wing
    int destructed; // id of the destroyed body
};


//Collision
struct Collision
{
  float time; // timestamp of the collision
  int body1; // id of the first collided body
  int body2; // id of the second collided body
};


//BodyPosition
struct BodyPosition
{
  int body;
  Vector3D position;
};


//Result
struct Result
{
    Destruction* destructions;
    Collision* collisions;
    BodyPosition* positions;

    int destructions_count;
    int collisions_count;
    int positions_count;
};


//type
enum type {NONE, PLANET, ASTEROID, DEATH_STAR, X_WING};


//Universe
 struct Universe{
	vector<int> id; //bodies' ids
	vector<type> types; //each body's index
	vector<float> mass; //each body's mass
	vector<float> size; //each body's size, I assume it's the diameter(not the radius) of the sphere that the body is
	vector<Vector3D> position; //current position of each body
	vector<Vector3D> speed; //current speed for each body
	vector<Vector3D> acceleration; //default acceleration of each body, it's (0.0, 0.0, 0.0) Planets and Asteroids
	vector<float> fuel; //current amount of fuel for Death Star and X-wing, is 0.0 by default and after that too;
	vector<float> consumption; //consumption of Death Stars' and X-Wings' engines, it's 0.0 by default for Planets and Asteroids
	vector<float> range; //range of Death Stars' and X-Wings' weapons, it's 0.0 for Planets and Asteroids

	int bodiesCount; //current  number of bodies in the universe

	Universe() {};

	void Resize(int n); //resizes each field of the structure (each vector) to the given size n and respectively bodiesCount becomes equal to n too

} currentUniverse;


void Universe::Resize(int n)
{
	this->bodiesCount = n;
	this->id.resize(this->bodiesCount);
    this->types.resize(this->bodiesCount);
    this->mass.resize(this->bodiesCount);
    this->size.resize(this->bodiesCount);
    this->position.resize(this->bodiesCount);
    this->speed.resize(this->bodiesCount);
    this->acceleration.resize(this->bodiesCount);
    this->fuel.resize(this->bodiesCount);
    this->consumption.resize(this->bodiesCount);
    this->range.resize(this->bodiesCount);
}


//Initializing and running the universe


//tells if the two bodies collide (is the distance between their positions is smaller than the sum of their radiuses)
bool Collide(int bodyIndex1, int bodyIndex2)
{
    return (((currentUniverse.size[bodyIndex1] * 0.5) + (currentUniverse.size[bodyIndex2] * 0.5)) >=
			CalculateDistance(currentUniverse.position[bodyIndex1], currentUniverse.position[bodyIndex2]));
}


//returns the sum of body's radius and it's weapon's range, which makes body's range
float WeaponBodyRange(int weaponBodyIndex)
{
	return (currentUniverse.size[weaponBodyIndex] * 0.5) + currentUniverse.range[weaponBodyIndex];
}


//tells if the second body will be destroyed by the first body
bool IsInDeadlyRangeOf(int weaponBodyIndex, int otherBodyIndex)
{
	type body1 = currentUniverse.types[weaponBodyIndex], body2 = currentUniverse.types[otherBodyIndex];

	if((body1 == DEATH_STAR && (body2 == DEATH_STAR || body2 == X_WING)) || (body1 == X_WING && (body2 == PLANET || body2 == X_WING))) return false;

	float distanceBetweenBodies = CalculateDistance(currentUniverse.position[weaponBodyIndex],
													currentUniverse.position[otherBodyIndex]);

	return (WeaponBodyRange(weaponBodyIndex) >= distanceBetweenBodies);
}

//finds and writes collisions and destructions of bodies at given moment
void FindCollisionsAndDestructions(vector<int>& indexesOfBodiesToRemove, Result* result, float timestamp)
{
	for(int i = 0; i < currentUniverse.bodiesCount; ++i)
	{
		if(find(indexesOfBodiesToRemove.begin(), indexesOfBodiesToRemove.end(), i) !=
		   indexesOfBodiesToRemove.end()) continue; //indexes of bodies already dead shouldn't be looked at

		for(int j = i + 1; j < currentUniverse.bodiesCount; ++j)
		{
			if(find(indexesOfBodiesToRemove.begin(), indexesOfBodiesToRemove.end(), j) !=
			   indexesOfBodiesToRemove.end()) continue; //indexes of bodies already dead shouldn't be looked at

            if(Collide(i, j))
			{
				indexesOfBodiesToRemove.push_back(i);
				indexesOfBodiesToRemove.push_back(j);

				result->collisions[result->collisions_count].time = timestamp;
				result->collisions[result->collisions_count].body1 = currentUniverse.id[i];
				result->collisions[result->collisions_count].body2 = currentUniverse.id[j];
				++(result->collisions_count);

				break;//the body with current index i does not exist anymore
					  //so it's wrong to check anything more about it, that's why we
					  //continue with the next value/next index
			}

			else if(currentUniverse.types[i] == DEATH_STAR || currentUniverse.types[i] == X_WING)
			{
                if(IsInDeadlyRangeOf(i, j))
				{
					indexesOfBodiesToRemove.push_back(j);

					result->destructions[result->destructions_count].time = timestamp;
					result->destructions[result->destructions_count].destructor = currentUniverse.id[i];
					result->destructions[result->destructions_count].destructed = currentUniverse.id[j];
					++(result->destructions_count);
				}
			}

			else if(currentUniverse.types[j] == DEATH_STAR || currentUniverse.types[j] == X_WING)
			{
                if(IsInDeadlyRangeOf(j, i))
				{
					indexesOfBodiesToRemove.push_back(i);

					result->destructions[result->destructions_count].time = timestamp;
					result->destructions[result->destructions_count].destructor = currentUniverse.id[j];
					result->destructions[result->destructions_count].destructed = currentUniverse.id[i];
					++(result->destructions_count);

					break; //the body with current index i does not exist anymore
						   //so it's wrong to check anything more about it, that's why we
						   //continue with the next value/next index
				}
			}
		}
	}
}

//helper for the next function
bool Greater(int a, int b)
{
	return a > b;
}

//removes records for the bodies whose indexes are in the given vector and which were destroyed an certain moment
void RemoveDeadBodiesRecords(vector<int>& indexes)
{
	int indexesCount = indexes.size();

    if(indexesCount == 0) return;

	sort(indexes.begin(), indexes.end(), Greater); //this is done in order to remove elements from Universe's vectors
												   //in a way that every index in 'indexes' is still valid
												   //after each previous erasing of an element from each vector;
												   //this way we will remove elements from the back to the front
												   //of each vector

    for(int i = 0; i < indexesCount; ++i)
	{
		currentUniverse.id.erase(currentUniverse.id.begin() + indexes[i]);
		currentUniverse.types.erase(currentUniverse.types.begin() + indexes[i]);
		currentUniverse.mass.erase(currentUniverse.mass.begin() + indexes[i]);
		currentUniverse.size.erase(currentUniverse.size.begin() + indexes[i]);
		currentUniverse.position.erase(currentUniverse.position.begin() + indexes[i]);
		currentUniverse.speed.erase(currentUniverse.speed.begin() + indexes[i]);
		currentUniverse.acceleration.erase(currentUniverse.acceleration.begin() + indexes[i]);
		currentUniverse.fuel.erase(currentUniverse.fuel.begin() + indexes[i]);
		currentUniverse.consumption.erase(currentUniverse.consumption.begin() + indexes[i]);
		currentUniverse.range.erase(currentUniverse.range.begin() + indexes[i]);

		--currentUniverse.bodiesCount;
	}

	indexes.resize(0);
}


//calculates the force, applied on the body at the given index, using the given formulas
Vector3D forceOnBody(int bodyIndex)
{
    Vector3D force(0.0, 0.0, 0.0);

    for(int i = 0; i < currentUniverse.bodiesCount; ++i)
	{
		if(i != bodyIndex)
		{
			Vector3D currentDistanceVector = currentUniverse.position[i] - currentUniverse.position[bodyIndex];

			force += currentDistanceVector.Normalized().Scaled( (G * currentUniverse.mass[bodyIndex] *
					 currentUniverse.mass[i]) * (1.0 / (currentDistanceVector.Length() * currentDistanceVector.Length())));
		}

		return force;
	}
}


void Universe_Initialize(const char* file)
{
    fstream description;
    description.open(file, ios::in);

    if(!description) return;

    int N;
    description >> N;

    currentUniverse.Resize(N);

	string typeStr;
    for(int i = 0; i < N; ++i)
	{
		description >> currentUniverse.id[i];
		description >> typeStr;

		if(typeStr == "Planet") currentUniverse.types[i] = PLANET;
		if(typeStr == "Asteroid") currentUniverse.types[i] = ASTEROID;
		if(typeStr == "Death")
		{
			currentUniverse.types[i] = DEATH_STAR;
			description >> typeStr;//here the word "Star" is read in order to easily continue the reading pattern
		}
		if(typeStr == "X-Wing") currentUniverse.types[i] = X_WING;
		typeStr = "";

		description >> currentUniverse.mass[i] >> currentUniverse.size[i] >>
		currentUniverse.position[i].x >> currentUniverse.position[i].y >>
		currentUniverse.position[i].z >> currentUniverse.speed[i].x >>
		currentUniverse.speed[i].y >> currentUniverse.speed[i].z;

		if(currentUniverse.types[i] == DEATH_STAR || currentUniverse.types[i] == X_WING)
		{
			description >> currentUniverse.acceleration[i].x >> currentUniverse.acceleration[i].y >>
			currentUniverse.acceleration[i].z >> currentUniverse.fuel[i] >>
			currentUniverse.consumption[i] >> currentUniverse.range[i];
		}
	}
}


void Universe_Run(float time, float delta, Result* result)
{
	float currentTime = 0.0;

	vector<int> currentIndexesOfDeadBodies;

	while(currentTime < time)
	{
		currentTime += delta;

		for(int i = 0; i < currentUniverse.bodiesCount; ++i)
		{
			Vector3D a  = forceOnBody(i).Scaled(1.0 / currentUniverse.mass[i]) +
						  currentUniverse.acceleration[i]; //currentUniverse.acceleration[i] is (0.0, 0.0, 0.0)
														   //if the body is of type Planet or Asteroid

			Vector3D path = currentUniverse.speed[i].Scaled(delta) + (a.Scaled(delta*delta)).Scaled(0.5);
            currentUniverse.position[i] += path;

            currentUniverse.speed[i] = a + currentUniverse.speed[i].Scaled(delta);
		}

		FindCollisionsAndDestructions(currentIndexesOfDeadBodies, result, currentTime);

		RemoveDeadBodiesRecords(currentIndexesOfDeadBodies);

		//calculate the new amounts of fuel for Death Stars and X-Wings
		for(int i = 0; i < currentUniverse.bodiesCount; ++i)
		{
            if(currentUniverse.fuel[i] != 0.0 &&
			  (currentUniverse.types[i] == DEATH_STAR || currentUniverse.types[i] == X_WING))
			{
                float fuelConsumed = delta * currentUniverse.consumption[i];

				if(fuelConsumed >= currentUniverse.fuel[i])
				{
					currentUniverse.fuel[i] = 0.0;
					currentUniverse.acceleration[i] = Vector3D(0.0, 0.0, 0.0); //if body's fuel's over,its default
					//acceleration proven by it's engine, becomes the zero vector
				}

				else currentUniverse.fuel[i] -= fuelConsumed;
			}
		}
	}

	for(int i = 0; i < currentUniverse.bodiesCount; ++i)
	{
		result->positions[result->positions_count].body = currentUniverse.id[i];
		result->positions[result->positions_count].position = currentUniverse.position[i];
		++(result->positions_count);
	}
}


int main()
{
	 return 0;
}
