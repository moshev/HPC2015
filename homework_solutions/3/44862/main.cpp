#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

struct Vector3D
{
	Vector3D()
	{
		x = 0;
		y = 0;
		z = 0;
	}
	Vector3D(float x, float y, float z):
		x(x), y(y), z(z) {}

	inline Vector3D& operator+=(const Vector3D& rhs)
	{
		x += rhs.x;
		y += rhs.y;
		z += rhs.z;
		return *this;
	}

	inline Vector3D operator-(const Vector3D& rhs) const
	{
		return Vector3D(x - rhs.x, y - rhs.y, z - rhs.z);
	}

	inline Vector3D operator*(float value) const
	{
		return Vector3D(x * value, y * value, z * value);
	}

	inline float operator*(const Vector3D& rhs) const
	{
		return x * rhs.x + y * rhs.y + z * rhs.z;
	}

	inline Vector3D operator/(float value) const
	{
		return Vector3D(x / value, y / value, z / value);
	}

	inline Vector3D operator+(const Vector3D& rhs) const
	{
		Vector3D result;
		result += rhs;
		return result;
	}

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

enum BodyType
{
    Planet,
    Asteroid,
    DeathStar,
    X_Wing,
};

typedef std::vector<Vector3D> Forces;
typedef std::vector<Vector3D> Velocities;
typedef std::vector<Vector3D> Accelerations;
typedef std::vector<Vector3D> Positions;
typedef std::vector<Vector3D> Paths;

typedef std::vector<float> Sizes;
typedef std::vector<float> Masses;
typedef std::vector<float> Ranges;
typedef std::vector<float> Consumptions;
typedef std::vector<float> Fuels;

typedef std::vector<BodyType> Types;

typedef std::vector<int> IDs;

struct BodiesCharacteristics
{
	int count;
	IDs ids;
	Types types;
	Masses masses;
	Sizes sizes; // diameters
	Positions positions;
	Velocities velocities;
	Accelerations accelerations;

	Forces forces;
	Paths paths;

	Accelerations defaultAcc;
	Fuels fuels;
	Consumptions consumptions;
	Ranges ranges;
};

BodiesCharacteristics bodiesChars;

std::vector<Collision> collisions;
std::vector<Destruction> destructions;

/*
 * Memory is reserved only once for these two vectors
 * with the initial size of the universe and they
 * are reused every time when checking for
 * collisions and destructions
 * */
std::vector<int> toDelete;
std::vector<char> isDestroyed;

const float G = 6.67408 * 1e-14; // multiplied by 10 ^ (-3) to convert to km^3 * kg^(-1) * s^(-2)

inline float length(const Vector3D& vec)
{
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

inline Vector3D normalize(const Vector3D& vec)
{
	float len = length(vec);
	return Vector3D(vec.x / len, vec.y / len, vec.z / len);
}

inline float distance(const Vector3D& body1Pos, const Vector3D& body2Pos)
{
	return length(body1Pos - body2Pos);
}

inline Vector3D nullVector()
{
	return Vector3D(0, 0, 0);
}

inline Vector3D forceIJ(const Vector3D& vec, float masI, float masJ)
{
	Vector3D normalized = normalize(vec);
	float dotProduct = vec * vec;

	return normalized * ((G * masI * masJ) / dotProduct);
}

void calculateForces()
{
	std::fill(bodiesChars.forces.begin(), bodiesChars.forces.end(), Vector3D(0, 0, 0));

	/*
	 * Calculating only the force IJ for every i and j: i < j
	 * force JI is -1 * force IJ
	 * */
	for(int i = 0; i < bodiesChars.count; ++i) {
		for(int j = i + 1; j < bodiesChars.count; ++j) {
			Vector3D bodiesVector = bodiesChars.positions[j] - bodiesChars.positions[i];
			Vector3D force = forceIJ(bodiesVector, bodiesChars.masses[i], bodiesChars.masses[j]);

			bodiesChars.forces[i] += force;
			bodiesChars.forces[j] += force * (-1);
		}
	}
}

void calculateAccelerations(float delta)
{
	for(int i = 0; i < bodiesChars.count; ++i) {
		bodiesChars.accelerations[i] = bodiesChars.defaultAcc[i];
		bodiesChars.accelerations[i] += bodiesChars.forces[i] / bodiesChars.masses[i];
	}
}

void calculateVelocitiesPathsPositions(float delta)
{
	for(int i = 0; i < bodiesChars.count; ++i) {
		//velocities
		bodiesChars.velocities[i] += bodiesChars.accelerations[i] * delta;

		//paths
		const Vector3D& velocity = bodiesChars.velocities[i];
		const Vector3D& acceleration = bodiesChars.accelerations[i];
		bodiesChars.paths[i] = velocity * delta + (acceleration * delta * delta) * 0.5f;

		//positions
		bodiesChars.positions[i] += bodiesChars.paths[i];
	}
}

void updateDestroyersChars(float delta)
{
	for(int i = 0; i < bodiesChars.count; ++i) {
		float remainingFuel = bodiesChars.fuels[i] - bodiesChars.consumptions[i] * delta;
		if(remainingFuel > 0) {
			bodiesChars.fuels[i] = remainingFuel;
		} else {
			bodiesChars.fuels[i] = 0;
			bodiesChars.defaultAcc[i] = nullVector();
		}
	}
}

inline bool collide(const Vector3D& body1Pos, const Vector3D& body2Pos, float size1, float size2)
{
	return distance(body1Pos, body2Pos) <= (size1 + size2) * 0.5f;
}

inline bool destroys(const Vector3D& body1Pos, const Vector3D& body2Pos,
		float body1Size, float body2Size, float destroyerRange)
{
	return distance(body1Pos, body2Pos) <= (body1Size + body2Size + destroyerRange) * 0.5f;
}

inline Collision gatherCollisionInfo(int body1Index, int body2Index)
{
	Collision collision;
	collision.body1 = bodiesChars.ids[body1Index];
	collision.body2 = bodiesChars.ids[body2Index];

	return collision;
}

inline Destruction gatherDestructionInfo(int destroyerIndex, int destructedIndex)
{
	Destruction destruction;
	destruction.destructor = bodiesChars.ids[destroyerIndex];
	destruction.destructed = bodiesChars.ids[destructedIndex];

	return destruction;
}

inline void addCollision(Collision collision, float time)
{
	collision.time = time;
	collisions.push_back(collision);
}

inline void addDestruction(Destruction destruction, float time)
{
	destruction.time = time;
	destructions.push_back(destruction);
}

inline bool isDestroyer(int i)
{
	return bodiesChars.types[i] == BodyType::DeathStar || bodiesChars.types[i] == BodyType::X_Wing;
}

bool canDestroy(int i, int j)
{
	if(bodiesChars.types[i] == BodyType::DeathStar) {
		return bodiesChars.types[j] == BodyType::Planet || bodiesChars.types[j] == BodyType::Asteroid;
	} else if(bodiesChars.types[i] == BodyType::X_Wing){
		return bodiesChars.types[j] == BodyType::Asteroid || bodiesChars.types[j] == BodyType::DeathStar;
	}
	return false;
}

int checkCollisionsDestructions(float time)
{
	std::fill(isDestroyed.begin(), isDestroyed.begin() + bodiesChars.count, 0);
	int destroyedCount = 0;

	for(int i = 0; i < bodiesChars.count; ++i) {
		if(isDestroyed[i]) {
			continue;
		}
		const Vector3D& body1Pos = bodiesChars.positions[i];
		float body1Size = bodiesChars.sizes[i];
		float body1Range = bodiesChars.ranges[i];

		for(int j = i + 1; j < bodiesChars.count; ++j) {
			if(isDestroyed[j]) {
				continue;
			}
			const Vector3D& body2Pos = bodiesChars.positions[j];
			float body2Size = bodiesChars.sizes[j];
			float body2Range = bodiesChars.ranges[j];

			if(collide(body1Pos, body2Pos, body1Size, body2Size)) {
				addCollision(gatherCollisionInfo(i, j), time);
				toDelete[destroyedCount++] = i;
				toDelete[destroyedCount++] = j;
				isDestroyed[i] = 1;
				isDestroyed[j] = 1;
				break;
			}
			if(isDestroyer(i) || isDestroyer(j)) {
				if(canDestroy(i, j) && destroys(body1Pos, body2Pos, body1Size, body2Size, body1Range)) {
					addDestruction(gatherDestructionInfo(i, j), time);
					toDelete[destroyedCount++] = j;
					isDestroyed[j] = 1;
					break;
				}
				if(canDestroy(j, i) && destroys(body1Pos, body2Pos, body1Size, body2Size, body2Range)) {
					addDestruction(gatherDestructionInfo(j, i), time);
					toDelete[destroyedCount++] = i;
					isDestroyed[i] = 1;
					break;
				}
			}
		}
	}
	return destroyedCount;
}

void fillResult(Result* result)
{
	if(!result) {
		return;
	}
	result->collisions_count = collisions.size();
	result->destructions_count = destructions.size();
	result->positions_count = bodiesChars.count;

	for(int i = 0; i < result->collisions_count; ++i) {
		result->collisions[i] = collisions[i];
	}
	for(int i = 0; i < result->destructions_count; ++i) {
		result->destructions[i] = destructions[i];
	}
	for(int i = 0; i < result->positions_count; ++i) {
		result->positions[i].body =  bodiesChars.ids[i];
		result->positions[i].position = bodiesChars.positions[i];
	}
}

template <typename T>
inline void deleteBody(std::vector<T>& elements, int index)
{
	elements[index] =  elements.back();
	elements.pop_back();
}

void deleteBodies(int destroyedCount)
{
	auto cmpfuncion = [](int l, int r) {
		return r < l;
	};

	/*
	 * Sort the indices in decreasing order for efficient delete
	 * Otherwise delete won't be correct
	 * */
	std::sort(toDelete.begin(), toDelete.begin() + destroyedCount, cmpfuncion);

	for(int i = 0; i < destroyedCount; ++i) {
		deleteBody(bodiesChars.accelerations, toDelete[i]);
		deleteBody(bodiesChars.consumptions, toDelete[i]);
		deleteBody(bodiesChars.defaultAcc, toDelete[i]);
		deleteBody(bodiesChars.forces, toDelete[i]);
		deleteBody(bodiesChars.fuels, toDelete[i]);
		deleteBody(bodiesChars.ids, toDelete[i]);
		deleteBody(bodiesChars.masses, toDelete[i]);
		deleteBody(bodiesChars.paths, toDelete[i]);
		deleteBody(bodiesChars.positions, toDelete[i]);
		deleteBody(bodiesChars.ranges, toDelete[i]);
		deleteBody(bodiesChars.sizes, toDelete[i]);
		deleteBody(bodiesChars.types, toDelete[i]);
		deleteBody(bodiesChars.velocities, toDelete[i]);
	}
	bodiesChars.count -= destroyedCount;
}

void resizeUniverse(int size)
{
	bodiesChars.accelerations.resize(size);
	bodiesChars.consumptions.resize(size);
	bodiesChars.defaultAcc.resize(size);
	bodiesChars.forces.resize(size);
	bodiesChars.fuels.resize(size);
	bodiesChars.ids.resize(size);
	bodiesChars.masses.resize(size);
	bodiesChars.paths.resize(size);
	bodiesChars.positions.resize(size);
	bodiesChars.ranges.resize(size);
	bodiesChars.sizes.resize(size);
	bodiesChars.types.resize(size);
	bodiesChars.velocities.resize(size);
}

void Universe_Initialize(const char* file)
{
    std::ifstream universe(file);
    if(!universe) {
        return;
    }

    int id;
    int type;
    float mass, size, fuel, consumption, range;
    Vector3D position, velocity, acceleration, defaultAcceleration;

    universe >> bodiesChars.count;
    resizeUniverse(bodiesChars.count);

    for(int i = 0; i < bodiesChars.count; ++i) {
    	universe >> id;
    	bodiesChars.ids[i] = id;

    	universe >> type;
    	bodiesChars.types[i] = static_cast<BodyType>(type);

    	universe >> mass;
    	bodiesChars.masses[i] = type;

    	universe >> size;
    	bodiesChars.sizes[i] = size;

    	universe >> position.x >> position.y >> position.z;
    	bodiesChars.positions[i] = position;

    	universe >> velocity.x >> velocity.y >> velocity.z;
    	bodiesChars.velocities[i] = velocity;

    	bodiesChars.defaultAcc[i] = nullVector();
		bodiesChars.fuels[i] = 0;
		bodiesChars.consumptions[i] = 0;
		bodiesChars.ranges[i] = 0;


    	if(type == BodyType::DeathStar || type == BodyType::X_Wing) {
    		universe >> defaultAcceleration.x >> defaultAcceleration.y >> defaultAcceleration.z;
    		bodiesChars.defaultAcc[i] = defaultAcceleration;

    		universe >> fuel;
    		bodiesChars.fuels[i] = fuel;

    		universe >> consumption;
    		bodiesChars.consumptions[i] = consumption;

    		universe >> range;
    		bodiesChars.ranges[i] = range;
    	}
    }

    universe.close();

    toDelete.reserve(bodiesChars.count);
    isDestroyed.reserve(bodiesChars.count);
}

void Universe_Run(float time, float delta, Result* result)
{
	for(float t = 0; t <= time; t += delta) {
		int destroyedCount = checkCollisionsDestructions(time);
		deleteBodies(destroyedCount);
		calculateForces();
		calculateAccelerations(delta);
		calculateVelocitiesPathsPositions(delta);
		updateDestroyersChars(delta);
	}
	fillResult(result);
}

int main()
{
	return 0;
}
