#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include <stdio.h>
#include <Simd\avx.h>
#include <sys\alloc.h>
#include <chrono>
using namespace std;
const double G = 6.674e-11;
enum BodyTypes {
	Planet = 0,
	Asteroid = 1,
	Death_Star = 2,
	X_Wing = 3
};
struct Vector3D
{
	float x, y, z;
};

struct VectorSimd
{
	void Load(Vector3D& vec)
	{
		float v[4] = { vec.x, vec.y, vec.z, 0 };
		data.load(v);
	}
	embree::ssef data;
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

struct BaseObjectsInfo
{
	unsigned planetsCount;
	unsigned asteroidCount;
	unsigned deathStarsCount;
	unsigned xWingsCount;
	unsigned plainBodies;
	unsigned xWingIndex;
	unsigned all;


	embree::avxf* positions;
	float* masses;
	float* sizes;
	embree::avxf* speeds;
	BodyTypes* bodyTypes;
}baseObjectsInfo[2];

int currentObject = 0;

struct AdditionalObjectsInfo
{
	embree::avxf* accelerations;
	float* fuelAmounts;
	float* consumptionAmounts;
	float* ranges;
}additionalObjectsInfo;

struct Result
{
	Destruction* destructions;
	Collision* collisions;
	BodyPosition* positions;

	int destructions_count;
	int collisions_count;
	int positions_count;
};

//float Parse_Float(const string& str)
//{
//	return atof(str.substr(1, str.length() - 1).c_str());
//}
//
//void Parse_Vector(const string& str, Vector3D& res)
//{
//	stringstream ss(str.substr(1, str.length() - 1).c_str());
//	ss >> res.x >> res.y >> res.z;
//}

struct ParsedObject
{
	Vector3D position;
	float mass;
	float size;
	Vector3D speed;
};

struct AdvancedObject : public ParsedObject
{
	Vector3D acceleration;
	float fuelAmount;
	float consumptionAmount;
	float range;
};

void* AlignedAlloc(unsigned long long size)
{
	size = size + (64 - size % 64);
#ifdef _MSC_VER
	return _aligned_malloc(size, 8);
#elif defined __GNUC__
	void* res;
	posix_memalign(&res, 8, size);
#endif
}
void AlignedFree(void * ptr)
{
#ifdef _MSC_VER
	return _aligned_free(ptr);
#elif defined __GNUC__
	free(ptr);
#endif
}

void allocAligned(void* originalPtr, void* alignedPtr, int size)
{
	originalPtr = malloc(size + 31);
	alignedPtr = (void*)((unsigned long long)originalPtr & 0xFFFFFFFFFFFFFFE0L);
}

void Parse_File(const char* file)
{
	ifstream fileStream(file);
	int n;
	fileStream >> n;
	string line;
	vector<ParsedObject> planets;
	vector<ParsedObject> asteroids;
	vector<AdvancedObject> deathStars;
	vector<AdvancedObject> xWings;

	while (std::getline(fileStream, line))
	{
		if (line == "") continue;
		std::istringstream iss(line);
		//<id> <Type> <Mass> <Size> <Position> <Speed> [<Acceleration> <Fuel> <Consumption> <Range>]
		//string idStr, typeStr, massStr, sizeString, positionStr, speedStr;
		string typeStr;
		int id;
		BodyTypes bodyType;
		float mass, size;
		Vector3D position, speed;

		iss >> id >> typeStr;
		if (typeStr == "Death")
		{
			string tmp;
			iss >> tmp;
			typeStr = typeStr + " " + tmp;
		}
		iss >> mass >> size >> position.x >> position.y >> position.z >> speed.x >> speed.y >> speed.z;
		position.x *= 1000.0;
		position.y *= 1000.0;
		position.z *= 1000.0;
		speed.x / 3.6f;
		speed.y / 3.6f;
		speed.z / 3.6f;
		//id = atoi(idStr.substr(1, idStr.length() - 1).c_str());

		if (typeStr == "Planet")
			bodyType = BodyTypes::Planet;
		else if (typeStr == "Asteroid")
			bodyType = BodyTypes::Asteroid;
		else if (typeStr == "Death Star")
			bodyType = BodyTypes::Death_Star;
		else if (typeStr == "X-Wing")
			bodyType = BodyTypes::X_Wing;

		if (bodyType == BodyTypes::Death_Star || bodyType == BodyTypes::X_Wing)
		{
			Vector3D acceleration;
			float fuel, consumption, range;
			iss >> acceleration.x >> acceleration.y >> acceleration.z;
			acceleration.x *= 7.716049382716E-5;
			acceleration.y *= 7.716049382716E-5;
			acceleration.z *= 7.716049382716E-5;
			iss >> fuel >> consumption >> range;
			AdvancedObject advObj;
			advObj.acceleration = acceleration;
			advObj.consumptionAmount = consumption;
			advObj.fuelAmount = fuel;
			advObj.mass = mass;
			advObj.position = position;
			advObj.range = range * 1000.0;
			advObj.size = size * 1000.0f;
			advObj.speed = speed;
			if (bodyType == BodyTypes::Death_Star)
			{
				deathStars.push_back(advObj);
			}
			else
			{
				xWings.push_back(advObj);
			}
		}
		else
		{
			ParsedObject parsedObject;
			parsedObject.mass = mass;
			parsedObject.position = position;
			parsedObject.size = size;
			parsedObject.speed = speed;
			if (bodyType == BodyTypes::Planet)
			{
				planets.push_back(parsedObject);
			}
			else
			{
				asteroids.push_back(parsedObject);
			}
		}
	}

	baseObjectsInfo[0].asteroidCount = asteroids.size();
	baseObjectsInfo[0].deathStarsCount = deathStars.size();
	baseObjectsInfo[0].planetsCount = planets.size();
	baseObjectsInfo[0].xWingsCount = xWings.size();
	baseObjectsInfo[1] = baseObjectsInfo[0];
	baseObjectsInfo[0].positions = (embree::avxf*)AlignedAlloc(n * sizeof(embree::avxf));

	baseObjectsInfo[0].speeds = (embree::avxf*)AlignedAlloc(n * sizeof(embree::avxf));
	baseObjectsInfo[0].masses = (float*)AlignedAlloc(n * sizeof(float));
	baseObjectsInfo[0].sizes = (float*)AlignedAlloc(n * sizeof(float));
	baseObjectsInfo[0].bodyTypes = new BodyTypes[n];

	baseObjectsInfo[1].positions = (embree::avxf*)AlignedAlloc(n * sizeof(embree::avxf));
	baseObjectsInfo[1].speeds = (embree::avxf*)AlignedAlloc(n * sizeof(embree::avxf));
	baseObjectsInfo[1].masses = baseObjectsInfo[0].masses;
	baseObjectsInfo[1].sizes = baseObjectsInfo[0].sizes;
	baseObjectsInfo[1].bodyTypes = baseObjectsInfo[0].bodyTypes;

	additionalObjectsInfo.accelerations = (embree::avxf*)AlignedAlloc((deathStars.size() + xWings.size()) * sizeof(embree::avxf));
	additionalObjectsInfo.consumptionAmounts = (float*)AlignedAlloc((deathStars.size() + xWings.size()) * sizeof(float));
	additionalObjectsInfo.fuelAmounts = (float*)AlignedAlloc((deathStars.size() + xWings.size()) * sizeof(float));
	additionalObjectsInfo.ranges = (float*)AlignedAlloc((deathStars.size() + xWings.size()) * sizeof(float));

	float pos[8], speeds[8], acc[8];
	int planetsCount = planets.size();
	int insertedCount = 0;
	int processedCount = 0;
	for (size_t i = 0; i < planetsCount; i++)
	{
		int offset = (processedCount % 2) * 4;
		pos[offset + 0] = planets[i].position.x;
		pos[offset + 1] = planets[i].position.y;
		pos[offset + 2] = planets[i].position.z;
		pos[offset + 3] = 0;

		speeds[offset + 0] = planets[i].speed.x;
		speeds[offset + 1] = planets[i].speed.y;
		speeds[offset + 2] = planets[i].speed.z;
		speeds[offset + 3] = 0;

		if (processedCount % 2 == 1)
		{
			baseObjectsInfo[0].speeds[i / 2] = embree::avxf::load(speeds);
			baseObjectsInfo[0].positions[i / 2] = embree::avxf::load(pos);
			insertedCount++;
		}

		baseObjectsInfo[0].masses[i] = planets[i].mass;
		baseObjectsInfo[0].sizes[i] = planets[i].size;
		baseObjectsInfo[0].bodyTypes[i] = BodyTypes::Planet;
		processedCount++;
	}

	int skip = planets.size();
	for (size_t i = 0; i < asteroids.size(); i++)
	{
		int offset = (processedCount % 2) * 4;
		pos[offset + 0] = asteroids[i].position.x;
		pos[offset + 1] = asteroids[i].position.y;
		pos[offset + 2] = asteroids[i].position.z;
		pos[offset + 3] = 0;
		speeds[offset + 0] = asteroids[i].speed.x;
		speeds[offset + 1] = asteroids[i].speed.y;
		speeds[offset + 2] = asteroids[i].speed.z;
		speeds[offset + 3] = 0;
		if (processedCount % 2 == 1)
		{
			baseObjectsInfo[0].speeds[(skip + i) / 2] = embree::avxf::load(speeds);
			baseObjectsInfo[0].positions[(skip + i) / 2] = embree::avxf::load(pos);
			insertedCount++;
		}
		baseObjectsInfo[0].masses[skip + i] = asteroids[i].mass;
		baseObjectsInfo[0].sizes[skip + i] = asteroids[i].size;
		baseObjectsInfo[0].bodyTypes[i] = BodyTypes::Asteroid;
		processedCount++;
	}
	skip += asteroids.size();
	int complexObjectsProcessedCount = 0;
	int complexObjectsInserted = 0;
	for (size_t i = 0; i < deathStars.size(); i++)
	{
		int offset = (processedCount % 2) * 4;
		pos[offset + 0] = deathStars[i].position.x;
		pos[offset + 1] = deathStars[i].position.y;
		pos[offset + 2] = deathStars[i].position.z;
		pos[offset + 3] = 0;
		speeds[offset + 0] = deathStars[i].speed.x;
		speeds[offset + 1] = deathStars[i].speed.y;
		speeds[offset + 2] = deathStars[i].speed.z;
		speeds[offset + 3] = 0;

		int complexObjectOffset = (complexObjectsProcessedCount % 2);
		acc[offset + 0] = deathStars[i].acceleration.x;
		acc[offset + 1] = deathStars[i].acceleration.y;
		acc[offset + 2] = deathStars[i].acceleration.z;
		acc[offset + 3] = 0;

		if (processedCount % 2 == 1 == 1)
		{
			baseObjectsInfo[0].speeds[(skip + i) / 2] = embree::avxf::load(speeds);
			baseObjectsInfo[0].positions[(skip + i) / 2] = embree::avxf::load(pos);
			insertedCount++;
		}
		if (complexObjectsProcessedCount % 2 == 1)
		{
			additionalObjectsInfo.accelerations[complexObjectsInserted] = embree::avxf::load(acc);
			complexObjectsInserted++;
		}
		baseObjectsInfo[0].masses[skip + i] = deathStars[i].mass;
		baseObjectsInfo[0].sizes[skip + i] = deathStars[i].size;
		additionalObjectsInfo.consumptionAmounts[i] = deathStars[i].consumptionAmount;
		additionalObjectsInfo.fuelAmounts[i] = deathStars[i].fuelAmount;
		additionalObjectsInfo.ranges[i] = deathStars[i].range;
		baseObjectsInfo[0].bodyTypes[i] = BodyTypes::Death_Star;
		processedCount++;
		complexObjectsProcessedCount++;
	}

	skip += deathStars.size();
	for (size_t i = 0; i < xWings.size(); i++)
	{
		int offset = (processedCount % 2) * 4;
		pos[offset + 0] = xWings[i].position.x;
		pos[offset + 1] = xWings[i].position.y;
		pos[offset + 2] = xWings[i].position.z;
		pos[offset + 3] = 0;
		speeds[offset + 0] = xWings[i].speed.x;
		speeds[offset + 1] = xWings[i].speed.y;
		speeds[offset + 2] = xWings[i].speed.z;
		speeds[offset + 3] = 0;
		if (processedCount % 2 == 1)
		{
			baseObjectsInfo[0].speeds[(skip + i) / 2] = embree::avxf::load(speeds);
			baseObjectsInfo[0].positions[(skip + i) / 2] = embree::avxf::load(pos);
			insertedCount++;
		}

		int complexObjectOffset = (complexObjectsProcessedCount % 2);
		acc[offset + 0] = deathStars[i].acceleration.x;
		acc[offset + 1] = deathStars[i].acceleration.y;
		acc[offset + 2] = deathStars[i].acceleration.z;
		acc[offset + 3] = 0;

		if (complexObjectsProcessedCount % 2 == 1)
		{
			additionalObjectsInfo.accelerations[complexObjectsInserted] = embree::avxf::load(acc);
			complexObjectsInserted++;
		}
		baseObjectsInfo[0].masses[skip + i] = xWings[i].mass;
		baseObjectsInfo[0].sizes[skip + i] = xWings[i].size;
		additionalObjectsInfo.consumptionAmounts[deathStars.size() + i] = xWings[i].consumptionAmount;
		additionalObjectsInfo.fuelAmounts[deathStars.size() + i] = xWings[i].fuelAmount;
		additionalObjectsInfo.ranges[deathStars.size() + i] = xWings[i].range;
		baseObjectsInfo[0].bodyTypes[i] = BodyTypes::X_Wing;
		processedCount++;
		complexObjectsProcessedCount++;
		complexObjectsInserted++;
	}
	if (processedCount % 2 == 1)
	{
		pos[4] = 0;
		pos[5] = 0;
		pos[6] = 0;
		pos[7] = 0;
		speeds[4] = 0;
		speeds[5] = 0;
		speeds[6] = 0;
		speeds[7] = 0;
		baseObjectsInfo[0].sizes[n] = -1;
		baseObjectsInfo[0].masses[n] = 0;
		baseObjectsInfo[0].speeds[n / 2] = embree::avxf::load(speeds);
		baseObjectsInfo[0].positions[n / 2] = embree::avxf::load(pos);
		insertedCount++;
	}

	baseObjectsInfo[0].plainBodies = planets.size() + asteroids.size();
	baseObjectsInfo[0].xWingIndex = baseObjectsInfo[0].plainBodies + deathStars.size();
	baseObjectsInfo[0].all = baseObjectsInfo[0].xWingIndex + xWings.size();
	baseObjectsInfo[1].plainBodies = baseObjectsInfo[0].plainBodies;
	baseObjectsInfo[1].xWingIndex = baseObjectsInfo[0].xWingIndex;
	baseObjectsInfo[1].all = baseObjectsInfo[0].all;
}

void Universe_Initialize(const char* file)
{
	try
	{
		Parse_File(file);
	}
	catch (const std::exception& e)
	{
		cerr << e.what() << endl;
	}

}



void Universe_Run(float time, float delta, Result* result)
{
	BaseObjectsInfo in = baseObjectsInfo[currentObject];
	BaseObjectsInfo out = baseObjectsInfo[(currentObject + 1) % 2];
	currentObject += 1;
	currentObject %= 2;
	int count = in.asteroidCount + in.deathStarsCount + in.planetsCount + in.xWingsCount;
	vector<std::pair<int, int>> toDestruct;
	vector<std::pair<int, int>> collisions;
	for (int i = 0; i < count; i += 2)
	{
		embree::avxf current(in.positions[i / 2]);
		float mass1 = in.masses[i], mass2 = in.masses[i + 1];
		//embree::avxf currentMass = embree::avxf(mass1, mass1, mass1, mass1, mass2, mass2, mass2, mass2);
		in.masses[i] = 0;
		in.masses[i + 1] = 0;
		out.positions[i / 2] = embree::avxf(0.0);
		BodyTypes firstBodyType = in.bodyTypes[i], secondBodyType = in.bodyTypes[i + 1];

		for (int j = 0; j < count; ++j)
		{
			if (j == i || j == i + 1)
			{
				continue;
			}
			embree::ssef otherSse = j % 2 == 0 ? embree::extract<0>(in.positions[j / 2]) : embree::extract<1>(in.positions[j / 2]);
			embree::avxf other(otherSse, otherSse);
			embree::avxf distance = (other - current);

			auto distSqr = dot(distance, distance);
			float distance1 = abs(distance.v[0] + distance.v[1] + distance.v[2]);
			float distance2 = abs(distance.v[4] + distance.v[5] + distance.v[6]);
			if (distance1 < in.sizes[i] + in.sizes[j])
			{
				collisions.push_back(std::pair<int, int>(i, j));
			}

			if (distance2 < in.sizes[i] + in.sizes[j])
			{
				collisions.push_back(std::pair<int, int>(i + 1, j));
			}

			if (firstBodyType == Death_Star)
			{
				if (in.bodyTypes[j] == Planet || in.bodyTypes[j] == Asteroid)
				{
					if (distance1 < additionalObjectsInfo.ranges[i - in.plainBodies])
					{
						toDestruct.push_back(std::pair<int, int>(i, j));
					}
				}
			}

			if (secondBodyType == Death_Star)
			{
				if (in.bodyTypes[j] == Planet || in.bodyTypes[j] == Asteroid)
				{
					if (distance2 < additionalObjectsInfo.ranges[i + 1 - in.plainBodies])
					{
						toDestruct.push_back(std::pair<int, int>(i + 1, j));
					}
				}
			}

			if (in.bodyTypes[j] == Death_Star)
			{
				if (firstBodyType == Planet || firstBodyType == Asteroid)
				{
					if (distance1 < additionalObjectsInfo.ranges[j - in.plainBodies])
					{
						toDestruct.push_back(std::pair<int, int>(j, i));
					}
				}

				if (secondBodyType == Planet || secondBodyType == Asteroid)
				{
					if (distance2 < additionalObjectsInfo.ranges[j - in.plainBodies])
					{
						toDestruct.push_back(std::pair<int, int>(j, i + 1));
					}
				}
			}

			if (firstBodyType == X_Wing)
			{
				if (in.bodyTypes[j] == Death_Star)
				{
					if (distance1 < additionalObjectsInfo.ranges[i - in.xWingIndex])
					{
						toDestruct.push_back(std::pair<int, int>(i, j));
					}
				}
			}

			if (secondBodyType == X_Wing)
			{
				if (in.bodyTypes[j] == Death_Star)
				{
					if (distance2 < additionalObjectsInfo.ranges[i + 1 - in.xWingIndex])
					{
						toDestruct.push_back(std::pair<int, int>(i + 1, j));
					}
				}
			}

			if (in.bodyTypes[j] == X_Wing)
			{
				if (firstBodyType == Planet)
				{
					if (distance1 < additionalObjectsInfo.ranges[j - in.xWingIndex])
					{
						toDestruct.push_back(std::pair<int, int>(j, i));
					}
				}

				if (secondBodyType == Planet)
				{
					if (distance2 < additionalObjectsInfo.ranges[j - in.xWingIndex])
					{
						toDestruct.push_back(std::pair<int, int>(j, i + 1));
					}
				}
			}




			distance /= sqrt(distSqr); //normalized
			distance *= in.masses[j]; // first object mass is concidered in a = F/m;
			distance /= distSqr;
			distance *= G; //acceleration

			out.positions[i / 2] += distance; // used for accelerations
		}

		embree::ssef lower = embree::extract<0>(current), higher = embree::extract<1>(current);
		embree::ssef distance = higher - lower;

		embree::ssef distanceSqr = distance * distance;
		float distSqrFloat = distanceSqr[0] + distanceSqr[1] + distanceSqr[2];
		float dist = sqrt(distSqrFloat);

		if (firstBodyType == Death_Star)
		{
			if (secondBodyType == Asteroid || secondBodyType == Planet)
			{
				if (dist < additionalObjectsInfo.ranges[i - in.plainBodies])
				{
					toDestruct.push_back(std::pair<int, int>(i + 1, i));
				}
			}
		}

		if (secondBodyType == Death_Star)
		{
			if (firstBodyType == Asteroid || firstBodyType == Planet)
			{
				if (dist < additionalObjectsInfo.ranges[i + 1 - in.plainBodies])
				{
					toDestruct.push_back(std::pair<int, int>(i, i + 1));
				}
			}
		}

		if (firstBodyType == X_Wing)
		{
			if (secondBodyType == Death_Star || secondBodyType == Death_Star)
			{
				if (dist < additionalObjectsInfo.ranges[i - in.xWingIndex])
				{
					toDestruct.push_back(std::pair<int, int>(i, i + 1));
				}
			}
		}

		if (secondBodyType == X_Wing)
		{
			if (firstBodyType == Death_Star || firstBodyType == Death_Star)
			{
				if (dist < additionalObjectsInfo.ranges[i + 1 - in.xWingIndex])
				{
					toDestruct.push_back(std::pair<int, int>(i + 1, i));
				}
			}
		}

		if (dist > in.sizes[i] + in.sizes[i + 1])
		{
			distance /= dist;
			distance /= distSqrFloat;
			distance *= G;

			out.positions[i / 2] += embree::avxf(distance * mass2, distance * 0);


			std::swap(lower, higher);
			distance = higher - lower;
			distanceSqr = distance * distance;
			distSqrFloat = distanceSqr[0] + distanceSqr[1] + distanceSqr[2];
			distance /= sqrt(distSqrFloat);
			distance /= distSqrFloat;
			distance *= G;

			out.positions[i / 2] += embree::avxf(distance * 0, distance*mass1);

			embree::avxf at = out.positions[i / 2] * delta;
			out.speeds[i / 2] = in.speeds[i / 2] + at;
			out.positions[i / 2] = in.positions[i / 2] + in.speeds[i / 2] * delta + at*delta / 2.0f;
		}
		else
		{
			collisions.push_back(std::pair<int, int>(i, i + 1));
		}

		for (int i = 0; i < collisions.size(); ++i)
		{
			if (in.sizes[collisions[i].second  > 0])
			{
				/*Collision& d = result->collisions[result->collisions_count];
				d.body1 = collisions[i].first;
				d.body2 = collisions[i].second;
				d.time = time;*/
				in.sizes[collisions[i].first] = -1;
				in.masses[collisions[i].first] = 0;
				in.sizes[collisions[i].second] = -1;
				in.masses[collisions[i].second] = 0;
			}

			//TODO hande collision
		}

		for (int i = 0; i < toDestruct.size(); ++i)
		{
			if (in.sizes[toDestruct[i].second  > 0])
			{
				/*Destruction& d = result->destructions[toDestruct[i].second];
				d.destructed = true;
				d.destructor = toDestruct[i].first;
				d.time = time;*/
				in.sizes[toDestruct[i].second] = -1;
				in.masses[toDestruct[i].second] = 0;
			}

			//TODO hande collision
		}

		int plainBodyCount = in.plainBodies;
		for (int a = 0; a < in.deathStarsCount + in.xWingsCount; a += 2)
		{
			if (additionalObjectsInfo.fuelAmounts[a] > 0)
			{
				in.positions[plainBodyCount / 2] + additionalObjectsInfo.accelerations[a / 2];
			}

		}
		for (int a = 0; a < in.deathStarsCount + in.xWingsCount; a += 1)
		{
			additionalObjectsInfo.fuelAmounts[a] -= additionalObjectsInfo.consumptionAmounts[a];

		}

		in.masses[i] = mass1;
		in.masses[i + 1] = mass2;
		/*memcpy(&result->positions[i].position, embree::extract<0>(out.positions[i]).f, 4 * sizeof(float));
		if (i + 1 < in.all)
		{
			memcpy(&result->positions[i + 1].position, embree::extract<1>(out.positions[i]).f, 4 * sizeof(float));
		}*/
	}
	cout << endl;
	cout << endl;
}

int main()
{
	Universe_Initialize("input.txt");

	for (int i = 1; i < 2000000; ++i)
	{
		Universe_Run(10.0, i * 1, nullptr);
	}
	return 0;
}



