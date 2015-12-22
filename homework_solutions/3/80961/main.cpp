#include <fstream>
#include <math.h>
#include <cstring>

#define TYPE_MAX_LEN 10

#define PLANET_STR      "Planet"
#define ASTEROID_STR    "Asteroid"
#define DEATH_STAR_STR1 "Death"
#define DEATH_STAR_STR2 "Star"
#define X_WING_STR      "X-Wing"

typedef enum {
    PLANET, ASTEROID, DEATH_STAR, X_WING, DESTROYED
} body_type;

#define G 6.67408e-11

struct Vector3D
{
    float x, y, z;

    Vector3D()
    {
        x = y = z = 0.0f;
    }

    Vector3D(float _x, float _y, float _z) { set(_x, _y, _z); }

	void set(float _x, float _y, float _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

    inline float length(void) const
	{
		return sqrt(x * x + y * y + z * z);
	}
	inline float lengthSqr(void) const
	{
		return (x * x + y * y + z * z);
	}
	void scale(float multiplier)
	{
		x *= multiplier;
		y *= multiplier;
		z *= multiplier;
	}
	void operator *= (float multiplier)
	{
		scale(multiplier);
	}
    Vector3D operator * (float multiplier)
	{
        Vector3D other(*this);
        other.scale(multiplier);
        return other;
	}
	void operator += (const Vector3D& other)
	{
		this->x += other.x;
		this->y += other.y;
		this->z += other.z;
	}
	void operator /= (float divider)
	{
		scale(1.0 / divider);
	}
	void normalize(void)
	{
		float multiplier = 1.0 / length();
		scale(multiplier);
	}

    void print() const { printf("(%.9f, %.9f, %.9f)", x, y, z); }
};

inline Vector3D operator + (const Vector3D& a, const Vector3D& b)
{
	return Vector3D(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Vector3D operator - (const Vector3D& a, const Vector3D& b)
{
	return Vector3D(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Vector3D operator - (const Vector3D& a)
{
	return Vector3D(-a.x, -a.y, -a.z);
}

// dot product
inline float operator * (const Vector3D& a, const Vector3D& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

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

int N;
int *id;
int *type;
float *mass;
float *size;
Vector3D *position;
Vector3D *speed;
Vector3D *acceleration;
Vector3D *acceleration_engine;
float *fuel;
float *consumption;
float *range;


void Universe_Initialize(const char* file)
{
    std::fstream fs(file, std::fstream::in);
    char temp_type[TYPE_MAX_LEN];
    fs >> N;

    id = new int[N];
    type = new int[N];
    mass = new float[N];
    size = new float[N];
    position = new Vector3D[N];
    speed = new Vector3D[N];
    acceleration = new Vector3D[N];
    acceleration_engine = new Vector3D[N];
    fuel = new float[N];
    consumption = new float[N];
    range = new float[N];

    for(int i = 0; i < N; ++i) {
        fs >> id[i];
        fs >> temp_type;

        if(strcmp(temp_type, DEATH_STAR_STR1) == 0) {
            fs >> (temp_type + 6); //nor checking the "Star" part
            type[i] = DEATH_STAR;
        } else if(strcmp(temp_type, X_WING_STR) == 0) {
            type[i] = X_WING;
        } else if(strcmp(temp_type, PLANET_STR) == 0) {
            type[i] = PLANET;
        } else if(strcmp(temp_type, ASTEROID_STR) == 0) {
            type[i] = ASTEROID;
        }

        fs >> mass[i] >> size[i];
        fs >> position[i].x >> position[i].y >> position[i].z;
        fs >> speed[i].x >> speed[i].y >> speed[i].z;
        if(type[i] == DEATH_STAR || type[i] == X_WING) {
            fs >> acceleration_engine[i].x >> acceleration_engine[i].y >> acceleration_engine[i].z;
            fs >> fuel[i] >> consumption[i] >> range[i];
        }
    }
}

void Universe_Run(float time, float delta, Result* result)
{
    // magic
    Vector3D r_ij, n_ij;
    Vector3D F_ij, F_i = Vector3D();
    Vector3D a, P, v, norm_p, new_position, norm_v, dv;
    Collision collision;
    bool collision_happened = false;
    Destruction destruction;
    Destruction *destructions = new Destruction[N];
    int *destr_id = new int[N];
    int desctr_i = 0;
    result->collisions_count = 0;
    result->destructions_count = 0;
    result->positions_count = 0;


    Vector3D pos[N], vel[N], acel[N];

    for(float t = delta; t <= time; t += delta) {

        for(int i = 0; i < N; ++i) {
            if(type[i] == DESTROYED) {
                continue;
            }
            desctr_i = 0;
            F_i.set(0, 0, 0);
            for(int j = 0; j < N; ++j) {
                if(i != j && type[j] != DESTROYED) {
                    r_ij = n_ij = position[j] - position[i];
                    n_ij.normalize();

                    if(fabs(r_ij.length()) - size[i] - size[j] <= 1e-4) {
                        collision.time = t;
                        collision.body1 = id[i];
                        collision.body2 = id[j];
                        result->collisions[result->collisions_count++] = collision;
                        type[i] = DESTROYED;
                        type[j] = DESTROYED;
                        collision_happened = true;
                        break;

                    }

                    if((fabs(r_ij.length()) - size[i] - size[j]) <= range[i]) {
                        if((type[i] == DEATH_STAR && type[j] == PLANET || type[j] == ASTEROID) || (type[i] == X_WING && type[j] == DEATH_STAR || type[j] == ASTEROID)) {
                            destruction.time = t;
                            destruction.destructor = id[i];
                            destruction.destructed = id[j];
                            destructions[desctr_i] = destruction;
                            destr_id[desctr_i++] = j;
                        }
                    }

                    if(type[i] == DEATH_STAR || type[i] == X_WING) {
                        fuel[i] -= consumption[i]*delta;
                        if(fuel[i] <= 0) {
                            acceleration_engine[i] = Vector3D();
                            printf("asdasd\n");
                        }
                    }

                    F_ij = n_ij * ((G * mass[i] * mass[j]) / (r_ij * r_ij));
                    F_i += F_ij;
                }
            }

            if(collision_happened) {
                desctr_i = 0; //void destructions
                collision_happened = false;
                continue;
            }

            for(int i = 0; i < desctr_i; ++i) {
                result->destructions[result->destructions_count++] = destructions[i];
                type[destr_id[i]] = DESTROYED;
            }

            a = F_i * (1.0f / mass[i]);
            if(type[i] == DEATH_STAR || type[i] == X_WING) {
                a += acceleration_engine[i];
            }

            P = speed[i]*delta + (a*delta*delta) * 0.5; // P = v*t + a*t^2 / 2

            new_position = P - position[i];
            pos[i] = position[i] + new_position;

            v = speed[i] + a*delta; //XXX

            vel[i] = v;
            acel[i] = a;
        }

        for(int i = 0; i < N; ++i) {
            if(type[i] != DESTROYED) {
                position[i] += pos[i];
                speed[i] = vel[i];
                acceleration[i] = acel[i];
            }
        }
    }

    BodyPosition bodypos;
    for(int i = 0; i < N; ++i) {
        if(type[i] != DESTROYED) {
            bodypos.body = id[i];
            bodypos.position = position[i];
            result->positions[result->positions_count++] = bodypos;
        }
    }

    delete [] destructions;
    delete [] destr_id;
}


int main() {
    Universe_Initialize("input");

    Result result;
    result.destructions = new Destruction[N];
    result.collisions = new Collision[N];
    result.positions = new BodyPosition[N];

    Universe_Run(50, 0.1, &result);


    delete [] id;
    delete [] type;
    delete [] mass;
    delete [] size;
    delete [] position;
    delete [] speed;
    delete [] acceleration;
    delete [] acceleration_engine;
    delete [] fuel;
    delete [] consumption;
    delete [] range;


	return 0;
}
