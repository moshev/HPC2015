
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

enum Types {PLANET = 0, ASTEROID = 1, DEATH_STAR = 2, X_WING = 3, DEATH};
const float GRAVITY = 4.302 / 1000.f;

/*It's only for the result*/
struct Vector3D {
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
  int bodys;
  Vector3D position;
};

struct Result
{
	void init(int size)
	{
		destructions_count = 0;
		collisions_count = 0;
		positions_count = 0;
	}
	// TODO allocate memory
    Destruction* destructions;
    Collision* collisions;
    BodyPosition* positions;

    int destructions_count;
    int collisions_count;
    int positions_count;
};

struct Vectors3D {
	~Vectors3D()
	{
		delete []x;
		delete []y;
		delete []z;
	}
	void init(int size)
	{
		x = new float[size];
		y = new float[size];
		z = new float[size];
		vectorsCount = size;
	}
	int vectorsCount;
    float *x, *y, *z;
};

struct FloatArray {
	~FloatArray()
	{
		delete [] data;
	}
	void init(int size)
	{
		count = size;
		data = new float[size];
	}

	void reset()
	{
		memset(data, 0,count);
	}
	int count;
	float *data;
};

FloatArray gBodysMass,gFuel, gConsumption, gRange, gDummy;

struct VectorsAndLengths {
	void init(int size)
	{
		pos.init(size);
		len.init(size);
	}
	Vectors3D pos;
	FloatArray len;
};

struct AdditionalInfo {
	FloatArray ids;
	FloatArray types;
	FloatArray sizes;
} info;

Vectors3D allBodys, gBodysSpeed, gAcceleration, gFs, gPaths, gAccelerationWithFuel;

void Universe_Initialize(const char* file)
{
	int size;
	int id, type, cnt = 0;

	FILE *fd = fopen(file, "r");
	if(!fd)
	{
		printf("Wrong file\n");
		return;
	}
	fscanf(fd, "%d", &size);
	allBodys.init(size);
	gFs.init(size);
	gBodysMass.init(size);
	gBodysSpeed.init(size);
	gAcceleration.init(size);
	gFuel.init(size);
	gConsumption.init(size);
	gRange.init(size);
	info.ids.init(size);
	info.types.init(size);
	info.sizes.init(size);
	gDummy.init(size * 4); // 4 because x, y and z coordinate + length
	gPaths.init(size);
	gAccelerationWithFuel.init(size);

	while(!feof(fd))
	{
		fscanf(fd, "%d %d", &id, &type);
		info.ids.data[cnt] = id;
		info.types.data[cnt] = type;
		switch(type)
		{
		case PLANET:
		case ASTEROID:
			fscanf(
				fd,
				"%f %f %f %f %f %f %f %f",
				&gBodysMass.data[cnt],
				&info.sizes.data[cnt],
				&allBodys.x[cnt],
				&allBodys.y[cnt],
				&allBodys.z[cnt],
				&gBodysSpeed.x[cnt],
				&gBodysSpeed.y[cnt],
				&gBodysSpeed.z[cnt]);
			gAcceleration.x[cnt] = gAcceleration.y[cnt] = gAcceleration.z[cnt] = 0;
			gFuel.data[cnt] = gConsumption.data[cnt] = gRange.data[cnt++] = 0;
			break;
		case DEATH_STAR:
		case X_WING:
			fscanf(
				fd,
				"%f %f %f %f %f %f %f %f",
				&gBodysMass.data[cnt],
				&info.sizes.data[cnt],
				&allBodys.x[cnt],
				&allBodys.y[cnt],
				&allBodys.z[cnt],
				&gBodysSpeed.x[cnt],
				&gBodysSpeed.y[cnt],
				&gBodysSpeed.z[cnt]);
			fscanf(
				fd,
				"%f %f %f %f %f %f",
				&gAcceleration.x[cnt],
				&gAcceleration.y[cnt],
				&gAcceleration.z[cnt],
				&gFuel.data[cnt],
				&gConsumption.data[cnt],
				&gRange.data[cnt++]);
			break;
		}
	}
}

void generatePaths(int delta)
{

	memset(gFs.x, 0, gFs.vectorsCount);
	memset(gFs.y, 0, gFs.vectorsCount);
	memset(gFs.z, 0, gFs.vectorsCount);

	int size = allBodys.vectorsCount;
	for(int from = 0; from < size - 1; ++from)
	{
		for(int to = from + 1, cnt = 0; to < size; ++to)
		{
			gDummy.data[cnt++] = allBodys.x[to] - allBodys.x[from];
			gDummy.data[cnt++] = allBodys.y[to] - allBodys.y[from];
			gDummy.data[cnt++] = allBodys.z[to] - allBodys.z[from];
			gDummy.data[cnt] = sqrt(
								(gDummy.data[cnt - 1] * gDummy.data[cnt - 1]) +
								(gDummy.data[cnt - 2] * gDummy.data[cnt - 2]) +
								(gDummy.data[cnt - 3] * gDummy.data[cnt - 3]));
			gDummy.data[cnt - 3] /= gDummy.data[cnt];
			gDummy.data[cnt - 2] /= gDummy.data[cnt];
			gDummy.data[cnt - 1] /= gDummy.data[cnt];
		}

		for(int to = from + 1; to < size; ++to)
		{
			float mass = gBodysMass.data[from] * gBodysMass.data[to];

			gFs.x[to] +=
					((GRAVITY * mass) / gDummy.data[(from * 4) + 3]) *
					gDummy.data[(from * 4)];

			gFs.y[to] +=
					((GRAVITY * mass) / gDummy.data[(from * 4) + 3]) *
					gDummy.data[(from * 4) + 1];

			gFs.z[to] +=
					((GRAVITY * mass) / gDummy.data[(from * 4) + 3]) *
					gDummy.data[(from * 4) + 2];
		}

		for(int to = from + 1; to < size; ++to)
		{
			// cache miss
			gFs.x[from] += gFs.x[to];
			gFs.y[from] += gFs.y[to];
			gFs.z[from] += gFs.z[to];
		}

	}

	// already have all Fs

	for(int i = 0; i < size; ++i)
	{
		float tmpX = 0, tmpY = 0, tmpZ = 0;
		if(gFuel.data[i] > 0)
		{
			tmpX = gAccelerationWithFuel.x[i];
			tmpY = gAccelerationWithFuel.y[i];
			tmpZ = gAccelerationWithFuel.z[i];

			gFuel.data[i] -= delta * gConsumption.data[i];
		}

		gAcceleration.x[i] = (gFs.x[i] / gBodysMass.data[i]) + tmpX;
		gAcceleration.y[i] = (gFs.y[i] / gBodysMass.data[i]) + tmpY;
		gAcceleration.z[i] = (gFs.z[i] / gBodysMass.data[i]) + tmpZ;
	}

	// already have all accelerations

	for(int i = 0; i < size; ++i)
	{
		gPaths.x[i] = gBodysSpeed.x[i] * delta + (gAcceleration.x[i] * delta * delta) / 2;
		gPaths.y[i] = gBodysSpeed.y[i] * delta + (gAcceleration.y[i] * delta * delta) / 2;
		gPaths.z[i] = gBodysSpeed.z[i] * delta + (gAcceleration.z[i] * delta * delta) / 2;
	}

	// have all paths

	for(int i = 0; i < size; ++i)
	{
		gBodysSpeed.x[i] += gAcceleration.x[i] * delta;
		gBodysSpeed.y[i] += gAcceleration.y[i] * delta;
		gBodysSpeed.z[i] += gAcceleration.z[i] * delta;
	}


}

void moveObjects()
{
	for(int i = 0; i < allBodys.vectorsCount; ++i)
	{
		allBodys.x[i] += gPaths.x[i];
		allBodys.y[i] += gPaths.y[i];
		allBodys.z[i] += gPaths.z[i];
	}
}

void calcState(Result* result, float currentTime)
{
	int size = allBodys.vectorsCount;
	float distSqr;
	for(int i = 0; i < size; ++i)
	{
		for(int j = 0; j < size; ++j)
		{
			if(i == j || info.types.data[i] == DEATH || info.types.data[j] == DEATH)
			{
				continue;
			}
			distSqr = (allBodys.x[j] - allBodys.x[i]) * (allBodys.x[j] - allBodys.x[i]);
			distSqr += (allBodys.y[j] - allBodys.y[i]) * (allBodys.y[j] - allBodys.y[i]);
			distSqr += (allBodys.z[j] - allBodys.z[i]) * (allBodys.z[j] - allBodys.z[i]);

			if(distSqr < ((
					(info.sizes.data[i] + info.sizes.data[j]) / 2) *
					((info.sizes.data[i] + info.sizes.data[j]) / 2)))
			{
				result->collisions[i].body1 = info.ids.data[i];
				result->collisions[j].body2 = info.ids.data[j];
				result->collisions->time = currentTime;
				info.types.data[i] = DEATH;
				info.types.data[j] = DEATH;
				result->collisions_count++;
				continue;
			}

			if(distSqr < (gRange.data[j] * gRange.data[j]))
			{
				Types type1 = (Types)info.types.data[i];
				Types type2 = (Types)info.types.data[j];
				int id1, id2;
				id1 = info.ids.data[i];
				id2 = info.ids.data[j];

				if(type2 == DEATH_STAR)
				{
					if(type1 < type2)
					{
						result->destructions[i].destructed = info.ids.data[i];
						result->destructions[i].destructor = info.ids.data[j];
						result->destructions_count++;
						info.types.data[i] = DEATH;
						continue;
					}
				}
				else if(type2 == X_WING)
				{
					if(type1 < type2 && type1 != PLANET)
					{
						result->destructions[i].destructed = id1;
						result->destructions[i].destructor = id2;
						result->destructions_count++;
						info.types.data[i] = DEATH;
						continue;
					}
				}
			}
		}
	}
}

void Universe_Run(float time, float delta, Result* result)
{

	float currentTime = 0;

	while(currentTime <= time)
	{
		generatePaths(delta);
		moveObjects();
		calcState(result, currentTime);
		currentTime += delta;
		gDummy.reset();
	}

	int size = allBodys.vectorsCount;
	int cnt = 0;
	for(int i = 0; i < size; ++i)
	{

		if(info.types.data[i] == DEATH) continue;

		result->positions[i].position.x = allBodys.x[i];
		result->positions[i].position.y = allBodys.y[i];
		result->positions[i].position.z = allBodys.z[i];
		result->positions->bodys = info.ids.data[i];
		++cnt;
	}
	result->positions_count = cnt;
}
