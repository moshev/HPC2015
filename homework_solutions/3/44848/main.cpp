#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>
#include <ctime>
#include <random>

using namespace std;

const float GCONST = 6.674E-11; // km per s*s
const float EPS = 1e-6;

struct Vector3D {
	float x, y, z;
};

struct Destruction {
	float time; // timestamp of the destruction
	int destructor; // id of the Death Star or X-Wing
	int destructed; // id of the destroyed body
};

struct Collision {
	float time; // timestamp of the collision
	int body1; // id of the first collided body
	int body2; // id of the second collided body
};

struct BodyPosition {
	int body;
	Vector3D position;
};

struct Result {
	Destruction* destructions;
	Collision* collisions;
	BodyPosition* positions;

	int destructions_count;
	int collisions_count;
	int positions_count;
};

enum class NodeType {
	Planet = 0,
	Asteroid = 1,
	Death_Star = 2,
	X_Wing = 3,
};


void Universe_Initialize(const char* file);
void Universe_Run(float time, float delta, Result* result);

int main() {
	Universe_Initialize("uni.dat");
	Universe_Run(10000, 0.1f, nullptr);

	cout << "done";
	cin.get();
	return 0;
}

const int CHUNK_SHIFT = 2;
const int CHUNK_SIZE = 1 << CHUNK_SHIFT;
const int CHUNK_MASK = (1 << CHUNK_SHIFT) - 1;

struct PhysycalObject {
	PhysycalObject() {
		for (int c = 0; c < CHUNK_SIZE; ++c) {
			x[c] = y[c] = z[c] = 0.f;
			vx[c] = vy[c] = vz[c] = 0.f;
			mass[c] = size[c] = 1.f;
		}
	}

	float x[CHUNK_SIZE];
	float y[CHUNK_SIZE];
	float z[CHUNK_SIZE];

	float vx[CHUNK_SIZE];
	float vy[CHUNK_SIZE];
	float vz[CHUNK_SIZE];

	float mass[CHUNK_SIZE];
	float size[CHUNK_SIZE];
};


struct ShipObject {
	ShipObject() {
		for (int c = 0; c < CHUNK_SIZE; ++c) {
			ax[c] = ay[c] = az[c] = 0.f;
			fuel[c] = 1.f;
			consumption[c] = range[c] = 0.f;
		}
	}

	float ax[CHUNK_SIZE];
	float ay[CHUNK_SIZE];
	float az[CHUNK_SIZE];

	float fuel[CHUNK_SIZE];
	float consumption[CHUNK_SIZE];
	float range[CHUNK_SIZE];
};

struct ObjectMeta {
	ObjectMeta() {
		for (int c = 0; c < CHUNK_SIZE; ++c) {
			type[c] = NodeType::Asteroid;
			id[c] = -1;
		}
	}

	NodeType type[CHUNK_SIZE];
	int id[CHUNK_SIZE];
};


#define CHUNK_ACCES(v, prop, idx)\
	(this->v[idx >> CHUNK_SHIFT].##prop[idx & CHUNK_MASK])

struct Universe {
	// first ships.size() are ships

	int shipCount, objCount;
	vector<PhysycalObject> objects;
	vector<ShipObject> ships;
	vector<ObjectMeta> meta;

	void swap(int l, int r) {
		std::swap(CHUNK_ACCES(objects, x, l), CHUNK_ACCES(objects, x, r));
		std::swap(CHUNK_ACCES(objects, y, l), CHUNK_ACCES(objects, y, r));
		std::swap(CHUNK_ACCES(objects, z, l), CHUNK_ACCES(objects, z, r));
		std::swap(CHUNK_ACCES(objects, mass, l), CHUNK_ACCES(objects, mass, r));
		std::swap(CHUNK_ACCES(objects, size, l), CHUNK_ACCES(objects, size, r));

		std::swap(CHUNK_ACCES(meta, type, l), CHUNK_ACCES(meta, type, r));
		std::swap(CHUNK_ACCES(meta, id, l), CHUNK_ACCES(meta, id, r));
	}

	void swapShips(int l, int r) {
		std::swap(CHUNK_ACCES(ships, fuel, l), CHUNK_ACCES(ships, fuel, r));
		std::swap(CHUNK_ACCES(ships, consumption, l), CHUNK_ACCES(ships, consumption, r));
		std::swap(CHUNK_ACCES(ships, range, l), CHUNK_ACCES(ships, range, r));
	}


	// kill functions do not work properly, off by 1 errors?

	void kill(int r) {
		swap(r, objCount - 1);

		--objCount;
		if ((objCount & CHUNK_MASK) == 0) {
			objects.pop_back();
			meta.pop_back();
		}
	}

	void kill(int chunk, int idx) {
		kill((chunk << CHUNK_SHIFT) | (idx & CHUNK_MASK));
	}

	void killShip(int r) {
		// swap with last ship
		swap(r, shipCount - 1);
		swapShips(r, shipCount - 1);

		// swap with last obj
		swap(shipCount - 1, objCount - 1);


		--objCount;
		if ((objCount & CHUNK_MASK) == 0) {
			objects.pop_back();
			meta.pop_back();
		}

		--shipCount;
		if ((shipCount & CHUNK_MASK) == 0) {
			ships.pop_back();
		}
	}

	void killShip(int chunk, int idx) {
		killShip((chunk << CHUNK_SHIFT) | (idx & CHUNK_MASK));
	}

	void splitIdx(int idx, int &chunk, int &inner) {
		chunk = idx >> CHUNK_SHIFT;
		inner = idx & CHUNK_MASK;
	}
} uni;

#define PROP_ACCESS(data, prop, chunk, idx)\
	(uni.data[chunk].##prop[idx])

struct Node {
	Vector3D pos, vel;
	float size, mass;

	Vector3D acc;
	float fuel, consumption, range;

	NodeType type;
	int id;
};

vector<Node> objects;

void Universe_Initialize(const char* file) {
	ifstream in(file);
	int n = 0;
	in >> n;
	uni.shipCount = 0;

	// insert ships only
	for (int c = 0; c < n; ++c) {
		int chunk, idx;
		uni.splitIdx(uni.shipCount, chunk, idx);

		NodeType type;
		int id;

		in >> id >> reinterpret_cast<int&>(type);
		if (type >= NodeType::Death_Star) {
			
			if (idx == 0) {
				uni.objects.push_back(PhysycalObject());
				uni.ships.push_back(ShipObject());
				uni.meta.push_back(ObjectMeta());
			}

			PROP_ACCESS(meta, id, chunk, idx) = id;
			PROP_ACCESS(meta, type, chunk, idx) = type;
			in >> PROP_ACCESS(objects, mass, chunk, idx) >> PROP_ACCESS(objects, size, chunk, idx);
			in >> PROP_ACCESS(objects, x, chunk, idx) >> PROP_ACCESS(objects, y, chunk, idx) >> PROP_ACCESS(objects, z, chunk, idx);
			in >> PROP_ACCESS(objects, vx, chunk, idx) >> PROP_ACCESS(objects, vy, chunk, idx) >> PROP_ACCESS(objects, vz, chunk, idx);

			in >> PROP_ACCESS(ships, ax, chunk, idx) >> PROP_ACCESS(ships, ay, chunk, idx) >> PROP_ACCESS(ships, az, chunk, idx);
			in >> PROP_ACCESS(ships, fuel, chunk, idx) >> PROP_ACCESS(ships, consumption, chunk, idx) >> PROP_ACCESS(ships, range, chunk, idx);
			++uni.shipCount;
		} else {
			in.ignore(2048, '\n');
		}
	}
	in.seekg(0, ios::beg);
	in.ignore(2048, '\n');

	// insert non ships
	int objs = uni.shipCount;
	for (int c = 0; c < n; ++c) {

		int chunk, idx;
		uni.splitIdx(objs, chunk, idx);

		NodeType type;
		int id;

		in >> id >> reinterpret_cast<int&>(type);
		if (type <= NodeType::Asteroid) {
			if (idx == 0) {
				uni.objects.push_back(PhysycalObject());
				uni.meta.push_back(ObjectMeta());
			}

			PROP_ACCESS(meta, id, chunk, idx) = id;
			PROP_ACCESS(meta, type, chunk, idx) = type;

			in >> PROP_ACCESS(objects, mass, chunk, idx) >> PROP_ACCESS(objects, size, chunk, idx);
			in >> PROP_ACCESS(objects, x, chunk, idx) >> PROP_ACCESS(objects, y, chunk, idx) >> PROP_ACCESS(objects, z, chunk, idx);
			in >> PROP_ACCESS(objects, vx, chunk, idx) >> PROP_ACCESS(objects, vy, chunk, idx) >> PROP_ACCESS(objects, vz, chunk, idx);

			++objs;
		} else {
			in.ignore(2048, '\n');
		}
	}
	uni.objCount = n;
}


inline float sq(float x) {
	return x * x;
}


void Universe_Run(float time, float delta, Result* result) {
	float currentTime = 0.f;
	const float dtSq = delta * delta;
	
	vector<int> dead;

	while (currentTime < time) {

		// move all ships
		for (int chunk = 0; chunk < uni.ships.size(); ++chunk) {
			for (int idx = 0; idx < CHUNK_SIZE; ++idx) {
				float fx = 0.f, fy = 0.f, fz = 0.f;
				const float myMass = PROP_ACCESS(objects, mass, chunk, idx);

				for (int ochunk = 0; ochunk < uni.objects.size(); ++ochunk) {
					for (int oidx = 0; oidx < CHUNK_SIZE; ++oidx) {
						if (chunk == ochunk && idx == oidx) {
							continue;
						}

						const float difx = PROP_ACCESS(objects, x, chunk, idx) - PROP_ACCESS(objects, x, ochunk, oidx) + EPS;
						const float dify = PROP_ACCESS(objects, y, chunk, idx) - PROP_ACCESS(objects, y, ochunk, oidx) + EPS;
						const float difz = PROP_ACCESS(objects, z, chunk, idx) - PROP_ACCESS(objects, z, ochunk, oidx) + EPS;

						const float distSq = sq(difx) + sq(dify) + sq(difz);
						const float dist = sqrtf(distSq);

						const float F = (GCONST * myMass * PROP_ACCESS(objects, mass, ochunk, oidx)) / distSq;

						fx += (difx / dist) * F;
						fy += (dify / dist) * F;
						fz += (difz / dist) * F;
					}
				}

				fx /= myMass;
				fy /= myMass;
				fz /= myMass;

				fx += PROP_ACCESS(ships, ax, chunk, idx);
				fy += PROP_ACCESS(ships, ay, chunk, idx);
				fz += PROP_ACCESS(ships, az, chunk, idx);

				PROP_ACCESS(objects, x, chunk, idx) += PROP_ACCESS(objects, vx, chunk, idx) * delta + (fx * dtSq) / 2;
				PROP_ACCESS(objects, y, chunk, idx) += PROP_ACCESS(objects, vy, chunk, idx) * delta + (fy * dtSq) / 2;
				PROP_ACCESS(objects, z, chunk, idx) += PROP_ACCESS(objects, vz, chunk, idx) * delta + (fz * dtSq) / 2;

				PROP_ACCESS(ships, fuel, chunk, idx) -= PROP_ACCESS(ships, consumption, chunk, idx) * delta;
			}
		}

		// move all non ships
		const int objectStartChunk = uni.shipCount >> CHUNK_SHIFT;
		const int objectIdxStart = uni.shipCount & CHUNK_MASK;
		for (int chunk = objectStartChunk; chunk < uni.objects.size(); ++chunk) {
			for (int idx = 0; idx < objectIdxStart; ++idx) {
				float fx = 0.f, fy = 0.f, fz = 0.f;
				const float myMass = PROP_ACCESS(objects, mass, chunk, idx);

				for (int ochunk = 0; ochunk < uni.objects.size(); ++ochunk) {
					for (int oidx = 0; oidx < CHUNK_SIZE; ++oidx) {
						if (chunk == ochunk && idx == oidx) {
							continue;
						}

						const float difx = PROP_ACCESS(objects, x, chunk, idx) - PROP_ACCESS(objects, x, ochunk, oidx) + EPS;
						const float dify = PROP_ACCESS(objects, y, chunk, idx) - PROP_ACCESS(objects, y, ochunk, oidx) + EPS;
						const float difz = PROP_ACCESS(objects, z, chunk, idx) - PROP_ACCESS(objects, z, ochunk, oidx) + EPS;

						const float distSq = sq(difx) + sq(dify) + sq(difz);
						const float dist = sqrtf(distSq);

						const float F = (GCONST * myMass * PROP_ACCESS(objects, mass, ochunk, oidx)) / distSq;

						fx += (difx / dist) * F;
						fy += (dify / dist) * F;
						fz += (difz / dist) * F;
					}
				}

				fx /= myMass;
				fy /= myMass;
				fz /= myMass;

				PROP_ACCESS(objects, x, chunk, idx) += PROP_ACCESS(objects, vx, chunk, idx) * delta + (fx * dtSq) / 2;
				PROP_ACCESS(objects, y, chunk, idx) += PROP_ACCESS(objects, vy, chunk, idx) * delta + (fy * dtSq) / 2;
				PROP_ACCESS(objects, z, chunk, idx) += PROP_ACCESS(objects, vz, chunk, idx) * delta + (fz * dtSq) / 2;

			}
		}

		dead.clear();
		
		// destructions/collisions
		for (int chunk = 0; chunk < uni.objects.size(); ++chunk) {
			for (int idx = 0; idx < CHUNK_SIZE; ++idx) {

				for (int ochunk = 0; ochunk < uni.objects.size(); ++ochunk) {
					for (int oidx = 0; oidx < CHUNK_SIZE; ++oidx) {

						if (chunk == ochunk && idx == oidx) {
							continue;
						}

						const float difx = PROP_ACCESS(objects, x, chunk, idx) - PROP_ACCESS(objects, x, ochunk, oidx) + EPS;
						const float dify = PROP_ACCESS(objects, y, chunk, idx) - PROP_ACCESS(objects, y, ochunk, oidx) + EPS;
						const float difz = PROP_ACCESS(objects, z, chunk, idx) - PROP_ACCESS(objects, z, ochunk, oidx) + EPS;

						const float distSq = sq(difx) + sq(dify) + sq(difz);

						if (distSq <= sq(PROP_ACCESS(objects, size, chunk, idx) + PROP_ACCESS(objects, size, ochunk, oidx))) {
							// colision
#ifndef DONT_RESULT
							result->collisions[result->collisions_count].body1 = PROP_ACCESS(meta, id, chunk, idx);
							result->collisions[result->collisions_count].body2 = PROP_ACCESS(meta, id, ochunk, oidx);
							result->collisions[result->collisions_count].time = currentTime;
							++result->collisions_count;
#endif
							cout << currentTime << " " << "Collision " << PROP_ACCESS(meta, id, chunk, idx) << " <> " << PROP_ACCESS(meta, id, ochunk, oidx) << endl;

							dead.push_back(PROP_ACCESS(meta, id, chunk, idx));
							dead.push_back(PROP_ACCESS(meta, id, ochunk, oidx));

						} else if (PROP_ACCESS(meta, type, ochunk, oidx) >= NodeType::Death_Star &&
							distSq <= sq(PROP_ACCESS(ships, range, ochunk, oidx))) {

							// destruction

							if (PROP_ACCESS(meta, type, ochunk, oidx) == NodeType::Death_Star && PROP_ACCESS(meta, type, chunk, idx) <= NodeType::Asteroid) {
#ifndef DONT_RESULT
								result->destructions[result->destructions_count].time = currentTime;
								result->destructions[result->destructions_count].destructed = PROP_ACCESS(meta, id, chunk, idx);
								result->destructions[result->destructions_count].destructor = PROP_ACCESS(meta, id, ochunk, oidx);
								++result->destructions_count;
#endif
								cout <<  currentTime << " " << "Kill " << PROP_ACCESS(meta, id, ochunk, oidx) << " -> " << PROP_ACCESS(meta, id, chunk, idx);
								dead.push_back(PROP_ACCESS(meta, id, chunk, idx));

							} else if (PROP_ACCESS(meta, type, ochunk, oidx) == NodeType::X_Wing &&
								(PROP_ACCESS(meta, type, chunk, idx) == NodeType::Asteroid || PROP_ACCESS(meta, type, chunk, idx) == NodeType::Death_Star)) {
#ifndef DONT_RESULT
								result->destructions[result->destructions_count].time = currentTime;
								result->destructions[result->destructions_count].destructed = PROP_ACCESS(meta, id, chunk, idx);
								result->destructions[result->destructions_count].destructor = PROP_ACCESS(meta, id, ochunk, oidx);
								++result->destructions_count;
#endif
								cout <<  currentTime << " " << "Kill " << PROP_ACCESS(meta, id, ochunk, oidx) << " -> " << PROP_ACCESS(meta, id, chunk, idx);
								dead.push_back(PROP_ACCESS(meta, id, chunk, idx));
							}
						}
					}
				}
			}
		}

		for (const auto &id : dead) {
			for (int chunk = 0; chunk < uni.objects.size(); ++chunk) {
				bool stop = false;
				for (int idx = 0; idx < CHUNK_SIZE; ++idx) {
					if (PROP_ACCESS(meta, id, chunk, idx) == id) {
						if (PROP_ACCESS(meta, type, chunk, idx) >= NodeType::Death_Star) {
							uni.killShip(chunk, idx);
						} else {
							uni.kill(chunk, idx);
						}
						stop = true;
						break;
					}
				}
				if (stop) {
					break;
				}
			}
		}

		currentTime += delta;
	}
#ifndef DONT_RESULT

	for (int chunk = 0; chunk < uni.objects.size(); ++chunk) {
		for (int idx = 0; idx < CHUNK_SIZE; ++idx) {
			result->positions[result->positions_count].body = PROP_ACCESS(meta, id, chunk, idx);

			result->positions[result->positions_count].position.x = PROP_ACCESS(objects, x, chunk, idx);
			result->positions[result->positions_count].position.y = PROP_ACCESS(objects, y, chunk, idx);
			result->positions[result->positions_count].position.z = PROP_ACCESS(objects, z, chunk, idx);
		}
	}
#endif
}