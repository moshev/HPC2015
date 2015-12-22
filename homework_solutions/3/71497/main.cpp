#include <stdio.h>
#include <fstream>

///
///	Не успях да го довърша цялото :/ ,
/// Но ми се ще да разбера дали съм на прав път
///

const int TYPE_SIZE = 64;

const double G = 6.674e-11;

struct Vector3D
{
	float x, y, z;
};

struct Destruction
{
	float time; // timestamp of the destruction
	int destructor; // id of the Death Star or X-Wing
	int destructed; // id of the destroyed body
};

struct Collision
{
	float time; // timestamp of the collision
	int body1; // id of the first collided body
	int body2; // id of the second collided body
};

struct BodyPosition
{
	int body;
	Vector3D position;
};

struct Result
{
	Destruction* destructions;
	Collision* collisions;
	BodyPosition* positions;

	int destructions_count;
	int collisions_count;
	int positions_count;
};

struct Body
{
	int id;
	char type[TYPE_SIZE];
	float mass;
	float size;
	Vector3D position;
	Vector3D speed;

	Vector3D acceleration;
	float fuel;
	float range;
	float consumption;

	float force;
};

struct Universe
{
	int bodiesCount;
	Body* bodies;
};

Universe universe{ 0 };

void substractVector(const Vector3D& lhs, const Vector3D& rhs, Vector3D& result)
{
	result = lhs;
	result.x -= rhs.x;
	result.y -= rhs.y;
	result.z -= rhs.z;
}

float vectorLength(const Vector3D& target)
{
	return sqrt(target.x * target.x + target.y * target.y + target.z * target.z);
}

void normalize(const Vector3D& target, Vector3D& result)
{
	float length = vectorLength(target);

	result = target;
	result.x /= length;
	result.y /= length;
	result.z /= length;
}

float calculateForce(const Body& lhs, const Body& rhs)
{
	Vector3D rIJ{ 0 };
	substractVector(rhs.position, lhs.position, rIJ);
	Vector3D nIJ{ 0 };
	normalize(rIJ, nIJ);

	float rIJLength = vectorLength(rIJ);
	float nIJLength = vectorLength(nIJ);

	return ((G * lhs.mass * rhs.mass) / (rIJLength * rIJLength)) * nIJLength;
}

void updatePosition(Body& body, float delta)
{
	float a = body.force / body.mass;

	body.position.x += body.speed.x * delta + (a * delta * delta) / 2;
	body.position.y += body.speed.y * delta + (a * delta * delta) / 2;
	body.position.z += body.speed.z * delta + (a * delta * delta) / 2;
}

void updateFuel(Body& body, float delta)
{
	if (body.fuel <= 0)
		body.fuel -= body.consumption * delta;
}

void addForces(float delta)
{
	for (int i = 0; i < universe.bodiesCount; ++i)
	{
		universe.bodies[i].force = 0;
		for (int j = 0; j < universe.bodiesCount; ++j)
		{
			if (i != j)
				universe.bodies[i].force += calculateForce(universe.bodies[i], universe.bodies[j]);
		}
	}

	for (int k = 0; k < universe.bodiesCount; ++k)
	{
		updatePosition(universe.bodies[k], delta);
		updateFuel(universe.bodies[k], delta);
	}
}

bool isCollision(const Body& lhs, const Body& rhs)
{
	float deltaX = (lhs.position.x - rhs.position.x) * (lhs.position.x - rhs.position.x);
	float deltaY = (lhs.position.y - rhs.position.y) * (lhs.position.y - rhs.position.y);

	// is Body.size the diameter?
	float radii = (lhs.size / 2 + rhs.size / 2);
	radii *= radii;

	return (deltaX + deltaY) <= radii;
}

void checkForCollisions(Result* result, float delta)
{
	for (int i = 0; i < universe.bodiesCount; ++i)
	{
		for (int j = 0; j < universe.bodiesCount; ++j)
		{
			if (isCollision(universe.bodies[i], universe.bodies[j]))
			{
				result->collisions[result->collisions_count].body1 = universe.bodies[i].id;
				result->collisions[result->collisions_count].body2 = universe.bodies[j].id;
				result->collisions[result->collisions_count].time = delta;
				++result->collisions_count;
			}
		}
	}
}

void Universe_Run(float time, float delta, Result* result)
{
	for (float i = 0; i < time; i += delta)
	{
		addForces(delta);
		checkForCollisions(result, delta);
	}

	delete[] universe.bodies;
}

void Universe_Initialize(const char* file)
{
	std::ifstream openHandle(file, std::ios::in);
	if (!openHandle)
		throw std::exception("Couldn't open universe file.");

	openHandle >> universe.bodiesCount;
	universe.bodies = new Body[universe.bodiesCount]{ 0 };

	for (int i = 0; i < universe.bodiesCount; ++i)
	{
		openHandle >> universe.bodies[i].id;
		openHandle >> universe.bodies[i].type;
		openHandle >> universe.bodies[i].mass;
		openHandle >> universe.bodies[i].size;

		openHandle >> universe.bodies[i].position.x;
		openHandle >> universe.bodies[i].position.y;
		openHandle >> universe.bodies[i].position.z;

		openHandle >> universe.bodies[i].speed.x;
		openHandle >> universe.bodies[i].speed.y;
		openHandle >> universe.bodies[i].speed.z;

		if (strcmp(universe.bodies[i].type, "Planet") == 0 || strcmp(universe.bodies[i].type, "Asteroid") == 0)
			continue;

		openHandle >> universe.bodies[i].acceleration.x;
		openHandle >> universe.bodies[i].acceleration.y;
		openHandle >> universe.bodies[i].acceleration.z;

		openHandle >> universe.bodies[i].fuel;
		openHandle >> universe.bodies[i].consumption;
		openHandle >> universe.bodies[i].range;
	}

	openHandle.close();
	if (!openHandle)
		throw std::exception("Could not close file handle.");
}


int main()
{
	//Result universeResult{ 0 };
	//universeResult.collisions = new Collision[1000];
	//universeResult.destructions = new Destruction[10];
	//universeResult.positions = new BodyPosition[10];

	//Universe_Initialize("test_data.txt");
	//Universe_Run(10, 0.1, &universeResult);

	//printf("%i\n", universeResult.collisions_count);

	return 0;
}
