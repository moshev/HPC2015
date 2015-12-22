#include <iostream>
#include <stdarg.h>
#include <vector>
using namespace std;

vector<Collision> v;

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

struct objects {
	void init(unsigned n) {
		ids = new int[n];
		positions = new Vector3D[n];
		accelerations = new Vector3D[n];
		speeds = new Vector3D[n];
		masses = new int[n];
		sizes = new float[n];
		fuels = new float[n];
		consumptions = new float[n];
		ranges = new float[n];
		N = n;
	}
	Vector3D* positions;
	Vector3D* accelerations;
	Vector3D* speeds;
	int* masses;
	float* sizes;
	float* fuels;
	float* consumptions;
	float* ranges;
	int* ids;
	unsigned N;
}o;

void ReadStuff(FILE* f,const char* str, ...) {
	va_list args;
	va_start(args, str);
	vfscanf(f, str, args);
	va_end(args);
}

void Universe_initialize(const char * file) {
	FILE* f = fopen(file, "r");
	int id;
	char type[10];
	int mass;
	float size;
	Vector3D pos;
	Vector3D speed;
	Vector3D acc;
	float fuel;
	float consumption;
	float range;
	ReadStuff(f, "%d", o.N);
	o.init(o.N);
	for (unsigned i = 0; i < N; ++i) {
		ReadStuff(f, "%d %s %f %f", &id, type, &mass, &size,
			&pos.x, &pos.y, &pos.z, &speed.x, &speed.y, &speed.y);
		o.positions[i] = pos;
		o.speeds[i] = speed;
		o.sizes[i] = size;
		o.masses[i] = mass;
		if (type[0] == 'D' || type[0] == 'X') {
			ReadStuff(f, "", &acc.x, &acc.y, &acc.z, &fuel, &consumption, &range);
			o.accelerations[i] = acc;
			o.fuels[i] = fuel;
			o.consumptions[i] = consumption;
			o.ranges[i] = range;
		}
	}
	fclose(f);
}

float distance(Vector3D first, Vector3D second) {
	return sqrtf((first.x - second.x) * (first.x - second.x) +
			     (first.y - second.y) * (first.y - second.y) +
				 (first.z - second.z) * (first.z - second.z));
}

void Universe_run(float time, float delta, Result* result) {
	float cur = 0;
	Collision c;
	while (time > cur) {
		for (int i = 0; i < o.N; ++i) {
			o.positions[i].x = o.positions[i].x + delta* o.speeds[i].x;
			o.positions[i].y = o.positions[i].y + delta* o.speeds[i].y;
			o.positions[i].z = o.positions[i].z + delta* o.speeds[i].z;
			for (int j = 0; j < o.N; ++j) {
				if (o.ids[i] && 0 > distance(o.positions[i], o.positions[j]) - o.sizes[i] - o.sizes[j]) {
					c.time = cur;
					c.body1 = o.ids[i];
					c.body2 = o.ids[j];
					v.push_back(c);
				}
			}

			//if destruction file a destruction
		}
		cur += delta;
	}
	result->positions_count = 0;
	for (int i = 0; i < o.N; ++i) {
		if (o.ids[i]) {
			result->positions->body = o.ids[i];
			result->positions->position = o.positions[i];
			++(result->positions_count);
		}
	}
}

int main() {
	return 0;
}