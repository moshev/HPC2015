#include <sstream>
#include <string>
#include <fstream>

#if defined(__GNUC__)
# define likely(x) __builtin_expect(!!(x),1)
# define unlikely(x) __builtin_expect(!!(x),0)
#else
# define likely(x) (x)
# define unlikely(x) (x)
#endif

const double GravitationalConstant = 6.674e-11;

struct Vector3D
{
    float x, y, z;

    inline Vector3D Normalized();
};

inline Vector3D& operator+=(Vector3D& left, const Vector3D& other)
{
    left.x += other.x;
    left.y += other.y;
    left.z += other.z;
    return left;
}

inline Vector3D& operator-=(Vector3D& left, const Vector3D& other)
{
    left.x -= other.x;
    left.y -= other.y;
    left.z -= other.z;
    return left;
}

inline Vector3D& operator*=(Vector3D& left, float scalar)
{
    left.x *= scalar;
    left.y *= scalar;
    left.z *= scalar;
    return left;
}

inline float operator*=(Vector3D& left, const Vector3D& other)
{
    return std::sqrtf((left.x - other.x) * (left.x - other.x) + (left.y - other.y) * (left.y - other.y) + (left.z - other.z) * (left.z - other.z));
}

inline float operator^=(Vector3D& left, const Vector3D& other)
{
    return (left.x - other.x) * (left.x - other.x) + (left.y - other.y) * (left.y - other.y) + (left.z - other.z) * (left.z - other.z);
}

inline Vector3D operator+(const Vector3D& left, const Vector3D& other)
{
    Vector3D result = left;
    result += other;
    return result;
}

inline Vector3D operator-(const Vector3D& left, const Vector3D& other)
{
    Vector3D result = left;
    result -= other;
    return result;
}

inline Vector3D operator*(const Vector3D& left, float scalar)
{
    Vector3D result = left;
    result *= scalar;
    return result;
}

inline float operator*(const Vector3D& left, const Vector3D& other)
{
    Vector3D result = left;
    return result *= other;
}

inline float operator^(const Vector3D& left, const Vector3D& other)
{
    Vector3D result = left;
    return result ^= other;
}

inline Vector3D Vector3D::Normalized()
{
    Vector3D normalized = *this;
    float length = normalized * normalized;
    normalized *= (1.0f / length);
    return normalized;
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

struct XWing
{
    BodyPosition position;
    int mass;
    float size;
    Vector3D speed;
    Vector3D acceleration;
    float fuel;
    float consumption;
    float range;
};

struct DeathStar
{
    BodyPosition position;
    int mass;
    float size;
    Vector3D speed;
    Vector3D acceleration;
    float fuel;
    float consumption;
    float range;
};

struct Planet
{
    BodyPosition position;
    int mass;
    float size;
    Vector3D speed;
};

struct Asteroid
{
    BodyPosition position;
    int mass;
    float size;
    Vector3D speed;
};

struct Bodies
{
    XWing* x_wings;
    DeathStar* death_stars;
    Planet* planets;
    Asteroid* asteroids;

    int x_wings_count;
    int death_stars_count;
    int asteroids_count;
    int planets_count;
};

void Universe_Initialize(const char* file, Bodies* bodies)
{
    std::ifstream infile(file);
    std::string line;
    int n;
    infile >> n;
    int x_wings_count = 0;
    int death_stars_count = 0;
    int asteroids_count = 0;
    int planets_count = 0;
    bodies->x_wings = new XWing[n];
    bodies->death_stars = new DeathStar[n];
    bodies->planets = new Planet[n];
    bodies->asteroids = new Asteroid[n];

    for (int i = 0; i < n; i++)
    {
        int id, mass;
        char* type = new char[20];
        float size, position_x, position_y, position_z, speed_x, speed_y, speed_z;

        infile >> id >> type >> mass >> size >>
            position_x >> position_y >> position_y >>
            position_z >> speed_x >> speed_y >> speed_z;

        bool isPlanet = strcmp(type, "Planet") == 0;
        bool isAsteroid = strcmp(type, "Asteroid") == 0;
        bool isDeathStar = strcmp(type, "Death Star") == 0;
        bool isXWing = strcmp(type, "X-Wing") == 0;

        if (isXWing)
        {
            bodies->x_wings[x_wings_count].position.body = id;
            bodies->x_wings[x_wings_count].position.position.x = position_x;
            bodies->x_wings[x_wings_count].position.position.y = position_y;
            bodies->x_wings[x_wings_count].position.position.z = position_z;
            bodies->x_wings[x_wings_count].mass = mass;
            bodies->x_wings[x_wings_count].size = size;
            bodies->x_wings[x_wings_count].speed.x = speed_x;
            bodies->x_wings[x_wings_count].speed.y = speed_y;
            bodies->x_wings[x_wings_count].speed.z = speed_z;

            float acceleration_x, acceleration_y, acceleration_z, fuel, consumption, range;
            infile >> acceleration_x >> acceleration_y >> acceleration_z >> fuel >> consumption >> range;

            bodies->x_wings[x_wings_count].acceleration.x = acceleration_x;
            bodies->x_wings[x_wings_count].acceleration.y = acceleration_y;
            bodies->x_wings[x_wings_count].acceleration.z = acceleration_z;
            bodies->x_wings[x_wings_count].fuel = fuel;
            bodies->x_wings[x_wings_count].consumption = consumption;
            bodies->x_wings[x_wings_count].range = range;
            ++x_wings_count;
        }
        else if (isDeathStar)
        {
            bodies->death_stars[death_stars_count].position.body = id;
            bodies->death_stars[death_stars_count].position.position.x = position_x;
            bodies->death_stars[death_stars_count].position.position.y = position_y;
            bodies->death_stars[death_stars_count].position.position.z = position_z;
            bodies->death_stars[death_stars_count].mass = mass;
            bodies->death_stars[death_stars_count].size = size;
            bodies->death_stars[death_stars_count].speed.x = speed_x;
            bodies->death_stars[death_stars_count].speed.y = speed_y;
            bodies->death_stars[death_stars_count].speed.z = speed_z;

            float acceleration_x, acceleration_y, acceleration_z, fuel, consumption, range;
            infile >> acceleration_x >> acceleration_y >> acceleration_z >> fuel >> consumption >> range;

            bodies->death_stars[death_stars_count].acceleration.x = acceleration_x;
            bodies->death_stars[death_stars_count].acceleration.y = acceleration_y;
            bodies->death_stars[death_stars_count].acceleration.z = acceleration_z;
            bodies->death_stars[death_stars_count].fuel = fuel;
            bodies->death_stars[death_stars_count].consumption = consumption;
            bodies->death_stars[death_stars_count].range = range;
            ++death_stars_count;
        }
        else if (isPlanet)
        {
            bodies->planets[planets_count].position.body = id;
            bodies->planets[planets_count].position.position.x = position_x;
            bodies->planets[planets_count].position.position.y = position_y;
            bodies->planets[planets_count].position.position.z = position_z;
            bodies->planets[planets_count].mass = mass;
            bodies->planets[planets_count].size = size;
            bodies->planets[planets_count].speed.x = speed_x;
            bodies->planets[planets_count].speed.y = speed_y;
            bodies->planets[planets_count].speed.z = speed_z;
            ++planets_count;
        }
        else if (isAsteroid)
        {
            bodies->asteroids[asteroids_count].position.body = id;
            bodies->asteroids[asteroids_count].position.position.x = position_x;
            bodies->asteroids[asteroids_count].position.position.y = position_y;
            bodies->asteroids[asteroids_count].position.position.z = position_z;
            bodies->asteroids[asteroids_count].mass = mass;
            bodies->asteroids[asteroids_count].size = size;
            bodies->asteroids[asteroids_count].speed.x = speed_x;
            bodies->asteroids[asteroids_count].speed.y = speed_y;
            bodies->asteroids[asteroids_count].speed.z = speed_z;
            ++asteroids_count;
        }
    }

    bodies->asteroids_count = asteroids_count;
    bodies->death_stars_count = death_stars_count;
    bodies->planets_count = planets_count;
    bodies->x_wings_count = x_wings_count;
}

void Universe_Run(float time, float delta, Bodies* bodies, Result* result)
{
    int destructions_count = 0;
    int collisions_count = 0;

#pragma unroll
    for (float time_stamp = 0; time_stamp < time; time_stamp += delta)
    {
        // Move asteroids
#pragma unroll
        for (int i = 0; i < bodies->asteroids_count; i++)
        {
            Vector3D acceleration;
            acceleration.x = acceleration.y = acceleration.z = 0;
            Vector3D vectorBetween;
            Vector3D vectorBetweenNormalized;

#pragma unroll
            for (int j = 0; j < bodies->asteroids_count; j++)
            {
                if (likely(i != j))
                {
                    vectorBetween = bodies->asteroids[i].position.position - bodies->asteroids[j].position.position;
                    vectorBetweenNormalized = vectorBetween.Normalized();
                    acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->asteroids[i].mass * bodies->asteroids[j].mass)
                        / (vectorBetween * vectorBetween));
                }
            }

#pragma unroll
            for (int j = 0; j < bodies->death_stars_count; j++)
            {
                vectorBetween = bodies->asteroids[i].position.position - bodies->death_stars[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->asteroids[i].mass * bodies->death_stars[j].mass)
                    / (vectorBetween * vectorBetween));
            }

#pragma unroll
            for (int j = 0; j < bodies->planets_count; j++)
            {
                vectorBetween = bodies->asteroids[i].position.position - bodies->planets[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->asteroids[i].mass * bodies->planets[j].mass)
                    / (vectorBetween * vectorBetween));
            }

#pragma unroll
            for (int j = 0; j < bodies->x_wings_count; j++)
            {
                vectorBetween = bodies->asteroids[i].position.position - bodies->x_wings[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->asteroids[i].mass * bodies->x_wings[j].mass)
                    / (vectorBetween * vectorBetween));
            }

            acceleration *= (1.0f / bodies->asteroids[i].mass);
            bodies->asteroids[i].position.position = bodies->asteroids[i].position.position +
                (bodies->asteroids[i].speed * delta + acceleration * (delta * delta / 2));
            bodies->asteroids[i].speed = bodies->asteroids[i].speed + acceleration * delta;
        }

        // Move planets
#pragma unroll
        for (int i = 0; i < bodies->planets_count; i++)
        {
            Vector3D acceleration;
            acceleration.x = acceleration.y = acceleration.z = 0;
            Vector3D vectorBetween;
            Vector3D vectorBetweenNormalized;

#pragma unroll
            for (int j = 0; j < bodies->asteroids_count; j++)
            {
                vectorBetween = bodies->planets[i].position.position - bodies->asteroids[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->asteroids[i].mass * bodies->asteroids[j].mass)
                    / (vectorBetween * vectorBetween));
            }

#pragma unroll
            for (int j = 0; j < bodies->death_stars_count; j++)
            {
                vectorBetween = bodies->planets[i].position.position - bodies->death_stars[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->planets[i].mass * bodies->death_stars[j].mass)
                    / (vectorBetween * vectorBetween));
            }

#pragma unroll
            for (int j = 0; j < bodies->planets_count; j++)
            {
                if (likely(i != j))
                {
                    vectorBetween = bodies->planets[i].position.position - bodies->planets[j].position.position;
                    vectorBetweenNormalized = vectorBetween.Normalized();
                    acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->planets[i].mass * bodies->planets[j].mass)
                        / (vectorBetween * vectorBetween));
                }
            }

#pragma unroll
            for (int j = 0; j < bodies->x_wings_count; j++)
            {
                vectorBetween = bodies->planets[i].position.position - bodies->x_wings[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->planets[i].mass * bodies->x_wings[j].mass)
                    / (vectorBetween * vectorBetween));
            }

            acceleration *= (1.0f / bodies->planets[i].mass);
            bodies->planets[i].position.position = bodies->planets[i].position.position +
                (bodies->planets[i].speed * delta + acceleration * (delta * delta / 2));
            bodies->planets[i].speed = bodies->planets[i].speed + acceleration * delta;
        }

        // Move x_wings
#pragma unroll
        for (int i = 0; i < bodies->x_wings_count; i++)
        {
            Vector3D acceleration;
            acceleration.x = acceleration.y = acceleration.z = 0;
            Vector3D vectorBetween;
            Vector3D vectorBetweenNormalized;

#pragma unroll
            for (int j = 0; j < bodies->asteroids_count; j++)
            {
                vectorBetween = bodies->x_wings[i].position.position - bodies->asteroids[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->x_wings[i].mass * bodies->asteroids[j].mass)
                    / (vectorBetween * vectorBetween));
            }

#pragma unroll
            for (int j = 0; j < bodies->death_stars_count; j++)
            {
                vectorBetween = bodies->x_wings[i].position.position - bodies->death_stars[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->x_wings[i].mass * bodies->death_stars[j].mass)
                    / (vectorBetween * vectorBetween));
            }

#pragma unroll
            for (int j = 0; j < bodies->planets_count; j++)
            {
                vectorBetween = bodies->x_wings[i].position.position - bodies->planets[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->x_wings[i].mass * bodies->planets[j].mass)
                    / (vectorBetween * vectorBetween));
            }

#pragma unroll
            for (int j = 0; j < bodies->x_wings_count; j++)
            {
                if (likely(i != j))
                {
                    vectorBetween = bodies->x_wings[i].position.position - bodies->x_wings[j].position.position;
                    vectorBetweenNormalized = vectorBetween.Normalized();
                    acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->x_wings[i].mass * bodies->x_wings[j].mass)
                        / (vectorBetween * vectorBetween));
                }
            }

            acceleration *= (1.0f / bodies->x_wings[i].mass);
            acceleration += bodies->x_wings[i].acceleration * (bodies->x_wings[i].fuel <= 1e-7);
            bodies->x_wings[i].acceleration = acceleration;
            bodies->x_wings[i].position.position = bodies->x_wings[i].position.position +
                (bodies->x_wings[i].speed * delta + acceleration * (delta * delta / 2));
            bodies->x_wings[i].speed = bodies->x_wings[i].speed + acceleration * delta;
            bodies->x_wings[i].fuel -= bodies->x_wings[i].consumption * delta * (bodies->x_wings[i].fuel <= 1e-7);
        }

        // Move death_stars
#pragma unroll
        for (int i = 0; i < bodies->death_stars_count; i++)
        {
            Vector3D acceleration;
            acceleration.x = acceleration.y = acceleration.z = 0;
            Vector3D vectorBetween;
            Vector3D vectorBetweenNormalized;

#pragma unroll
            for (int j = 0; j < bodies->asteroids_count; j++)
            {
                vectorBetween = bodies->death_stars[i].position.position - bodies->asteroids[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->death_stars[i].mass * bodies->asteroids[j].mass)
                    / (vectorBetween * vectorBetween));
            }

#pragma unroll
            for (int j = 0; j < bodies->death_stars_count; j++)
            {
                if (likely(i != j))
                {
                    vectorBetween = bodies->death_stars[i].position.position - bodies->death_stars[j].position.position;
                    vectorBetweenNormalized = vectorBetween.Normalized();
                    acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->death_stars[i].mass * bodies->death_stars[j].mass)
                        / (vectorBetween * vectorBetween));
                }
            }

#pragma unroll
            for (int j = 0; j < bodies->planets_count; j++)
            {
                vectorBetween = bodies->death_stars[i].position.position - bodies->planets[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->death_stars[i].mass * bodies->planets[j].mass)
                    / (vectorBetween * vectorBetween));
            }

#pragma unroll
            for (int j = 0; j < bodies->x_wings_count; j++)
            {
                vectorBetween = bodies->death_stars[i].position.position - bodies->x_wings[j].position.position;
                vectorBetweenNormalized = vectorBetween.Normalized();
                acceleration += vectorBetweenNormalized * ((GravitationalConstant * bodies->death_stars[i].mass * bodies->x_wings[j].mass)
                    / (vectorBetween * vectorBetween));
            }

            acceleration *= (1.0f / bodies->death_stars[i].mass);
            acceleration += bodies->death_stars[i].acceleration * (bodies->x_wings[i].fuel <= 1e-7);
            bodies->death_stars[i].acceleration = acceleration;
            bodies->death_stars[i].position.position = bodies->death_stars[i].position.position +
                (bodies->death_stars[i].speed * delta + acceleration * (delta * delta / 2));
            bodies->death_stars[i].speed = bodies->death_stars[i].speed + acceleration * delta;
            bodies->death_stars[i].fuel -= bodies->death_stars[i].consumption * delta * (bodies->x_wings[i].fuel <= 1e-7);
        }

        // Collisions and destructions with death_stars attacking
#pragma unroll
        for (int i = 0; i < bodies->death_stars_count; i++)
        {
#pragma unroll
            for (int j = 0; j < bodies->asteroids_count; j++)
            {
                bool inRangeOfDeathStar = (bodies->death_stars[i].position.position ^ bodies->asteroids[j].position.position) <=
                    (bodies->asteroids[j].size + bodies->death_stars[i].size + bodies->death_stars[i].range) *
                    (bodies->asteroids[j].size + bodies->death_stars[i].size + bodies->death_stars[i].range) + 1e-7;

                if (inRangeOfDeathStar)
                {
                    result->destructions[destructions_count].destructed = bodies->asteroids[j].position.body;
                    result->destructions[destructions_count].destructor = bodies->death_stars[i].position.body;
                    result->destructions[destructions_count].time = time_stamp;
                    ++destructions_count;
                    std::swap(bodies->asteroids[j], bodies->asteroids[bodies->asteroids_count - 1]);
                    --bodies->asteroids_count;
                    --j;
                }
            }

#pragma unroll
            for (int j = 0; j < bodies->planets_count; j++)
            {
                bool inRangeOfDeathStar = (bodies->death_stars[i].position.position ^ bodies->planets[j].position.position) <=
                    (bodies->planets[j].size + bodies->death_stars[i].size + bodies->death_stars[i].range) *
                    (bodies->planets[j].size + bodies->death_stars[i].size + bodies->death_stars[i].range) + 1e-7;

                if (inRangeOfDeathStar)
                {
                    result->destructions[destructions_count].destructed = bodies->planets[j].position.body;
                    result->destructions[destructions_count].destructor = bodies->death_stars[i].position.body;
                    result->destructions[destructions_count].time = time_stamp;
                    ++destructions_count;
                    std::swap(bodies->planets[j], bodies->planets[bodies->planets_count - 1]);
                    --bodies->planets_count;
                    --j;
                }
            }

            bool deathStarDestroyed = false;
#pragma unroll
            for (int j = 0; j < bodies->death_stars_count; j++)
            {
                if (likely(i != j))
                {
                    bool inRangeOfDeathStar = (bodies->death_stars[i].position.position ^ bodies->death_stars[j].position.position) <=
                        (bodies->death_stars[j].size + bodies->death_stars[i].size) *
                        (bodies->death_stars[j].size + bodies->death_stars[i].size) + 1e-7;

                    if (inRangeOfDeathStar)
                    {
                        result->collisions[collisions_count].body1 = bodies->death_stars[i].position.body;
                        result->collisions[collisions_count].body2 = bodies->death_stars[j].position.body;
                        result->collisions[collisions_count].time = time_stamp;
                        ++collisions_count;

                        deathStarDestroyed = true;

                        std::swap(bodies->death_stars[j], bodies->death_stars[bodies->death_stars_count - 1]);
                        --bodies->death_stars_count;
                        --j;
                    }
                }
            }

            // Can't attack x-wing (it destroys death-stars).

            if (deathStarDestroyed)
            {
                std::swap(bodies->death_stars[i], bodies->death_stars[bodies->death_stars_count - 1]);
                --bodies->death_stars_count;
                --i;
            }
        }
        
        // Collisions and destructions with x_wings attacking
#pragma unroll
        for (int i = 0; i < bodies->x_wings_count; i++)
        {
#pragma unroll
            for (int j = 0; j < bodies->asteroids_count; j++)
            {
                bool inRangeOfXWing = (bodies->x_wings[i].position.position ^ bodies->asteroids[j].position.position) <=
                    (bodies->asteroids[j].size + bodies->x_wings[i].size + bodies->x_wings[i].range) *
                    (bodies->asteroids[j].size + bodies->x_wings[i].size + bodies->x_wings[i].range) + 1e-7;

                if (inRangeOfXWing)
                {
                    result->destructions[destructions_count].destructed = bodies->asteroids[j].position.body;
                    result->destructions[destructions_count].destructor = bodies->x_wings[i].position.body;
                    result->destructions[destructions_count].time = time_stamp;
                    ++destructions_count;
                    std::swap(bodies->asteroids[j], bodies->asteroids[bodies->asteroids_count - 1]);
                    --bodies->asteroids_count;
                    --j;
                }
            }

#pragma unroll
            for (int j = 0; j < bodies->death_stars_count; j++)
            {
                bool inRangeOfXWing = (bodies->x_wings[i].position.position ^ bodies->death_stars[j].position.position) <=
                    (bodies->death_stars[j].size + bodies->x_wings[i].size + bodies->x_wings[i].range) *
                    (bodies->death_stars[j].size + bodies->x_wings[i].size + bodies->x_wings[i].range) + 1e-7;

                if (inRangeOfXWing)
                {
                    result->destructions[destructions_count].destructed = bodies->death_stars[j].position.body;
                    result->destructions[destructions_count].destructor = bodies->x_wings[i].position.body;
                    result->destructions[destructions_count].time = time_stamp;
                    ++destructions_count;
                    std::swap(bodies->death_stars[j], bodies->death_stars[bodies->death_stars_count - 1]);
                    --bodies->death_stars_count;
                    --j;
                }
            }

            bool xWingDestroyed = false;
#pragma unroll
            for (int j = 0; j < bodies->planets_count; j++)
            {
                bool inRangeOfXWing = (bodies->x_wings[i].position.position ^ bodies->planets[j].position.position) <=
                    (bodies->planets[j].size + bodies->x_wings[i].size) *
                    (bodies->planets[j].size + bodies->x_wings[i].size) + 1e-7;

                if (inRangeOfXWing)
                {
                    result->collisions[collisions_count].body1 = bodies->x_wings[i].position.body;
                    result->collisions[collisions_count].body2 = bodies->planets[j].position.body;
                    result->collisions[collisions_count].time = time_stamp;
                    ++collisions_count;

                    xWingDestroyed = true;

                    std::swap(bodies->planets[j], bodies->planets[bodies->planets_count - 1]);
                    --bodies->planets_count;
                    --j;
                }
            }

#pragma unroll
            for (int j = 0; j < bodies->x_wings_count; j++)
            {
                if (likely(i != j))
                {
                    bool inRangeOfXWing = (bodies->x_wings[i].position.position ^ bodies->x_wings[j].position.position) <=
                        (bodies->x_wings[j].size + bodies->x_wings[i].size) *
                        (bodies->x_wings[j].size + bodies->x_wings[i].size) + 1e-7;

                    if (inRangeOfXWing)
                    {
                        result->collisions[collisions_count].body1 = bodies->x_wings[i].position.body;
                        result->collisions[collisions_count].body2 = bodies->x_wings[j].position.body;
                        result->collisions[collisions_count].time = time_stamp;
                        ++collisions_count;

                        xWingDestroyed = true;

                        std::swap(bodies->x_wings[j], bodies->x_wings[bodies->x_wings_count - 1]);
                        --bodies->x_wings_count;
                        --j;
                    }
                }
            }

            if (xWingDestroyed)
            {
                std::swap(bodies->x_wings[i], bodies->x_wings[bodies->x_wings_count - 1]);
                --bodies->x_wings_count;
                --i;
            }
        }

#pragma unroll
        // Asteroids collisions
        for (int i = 0; i < bodies->asteroids_count; i++)
        {
#pragma unroll
            bool asteroidDestroyed = false;
            for (int j = 0; j < bodies->asteroids_count; j++)
            {
                if (likely(i != j))
                {
                    bool inRangeOfAsteroid = (bodies->asteroids[i].position.position ^ bodies->asteroids[j].position.position) <=
                        (bodies->asteroids[j].size + bodies->asteroids[i].size) *
                        (bodies->asteroids[j].size + bodies->asteroids[i].size) + 1e-7;

                    if (inRangeOfAsteroid)
                    {
                        result->collisions[collisions_count].body1 = bodies->asteroids[i].position.body;
                        result->collisions[collisions_count].body2 = bodies->asteroids[j].position.body;
                        result->collisions[collisions_count].time = time_stamp;
                        ++collisions_count;

                        asteroidDestroyed = true;

                        std::swap(bodies->asteroids[j], bodies->asteroids[bodies->asteroids_count - 1]);
                        --bodies->asteroids_count;
                        --j;
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < bodies->planets_count; j++)
            {
                bool inRangeOfAsteroids = (bodies->asteroids[i].position.position ^ bodies->planets[j].position.position) <=
                    (bodies->planets[j].size + bodies->asteroids[i].size) *
                    (bodies->planets[j].size + bodies->asteroids[i].size) + 1e-7;

                if (inRangeOfAsteroids)
                {
                    result->collisions[collisions_count].body1 = bodies->asteroids[i].position.body;
                    result->collisions[collisions_count].body2 = bodies->planets[j].position.body;
                    result->collisions[collisions_count].time = time_stamp;
                    ++collisions_count;

                    asteroidDestroyed = true;

                    std::swap(bodies->planets[j], bodies->planets[bodies->planets_count - 1]);
                    --bodies->planets_count;
                    --j;
                }
            }

            if (asteroidDestroyed)
            {
                std::swap(bodies->asteroids[i], bodies->asteroids[bodies->asteroids_count - 1]);
                --bodies->asteroids_count;
                --i;
            }
        }

        // Planets collisions
#pragma unroll
        for (int i = 0; i < bodies->planets_count; i++)
        {
#pragma unroll
            bool planetDestroyed = false;

            for (int j = 0; j < bodies->planets_count; j++)
            {
                if (likely(i != j))
                {
                    bool inRangeOfPlanet = (bodies->planets[i].position.position ^ bodies->planets[j].position.position) <=
                        (bodies->planets[j].size + bodies->planets[i].size) *
                        (bodies->planets[j].size + bodies->planets[i].size) + 1e-7;

                    if (inRangeOfPlanet)
                    {
                        result->collisions[collisions_count].body1 = bodies->planets[i].position.body;
                        result->collisions[collisions_count].body2 = bodies->planets[j].position.body;
                        result->collisions[collisions_count].time = time_stamp;
                        ++collisions_count;

                        planetDestroyed = true;

                        std::swap(bodies->planets[j], bodies->planets[bodies->planets_count - 1]);
                        --bodies->planets_count;
                        --j;
                    }
                }
            }

            if (planetDestroyed)
            {
                std::swap(bodies->planets[i], bodies->planets[bodies->planets_count - 1]);
                --bodies->planets_count;
                --i;
            }
        }
    }

    result->destructions_count = destructions_count;
    result->collisions_count = collisions_count;

    int positionsCount = 0;

#pragma unroll
    for (int i = 0; i < bodies->asteroids_count; i++)
    {
        result->positions[positionsCount] = bodies->asteroids[i].position;
        ++positionsCount;
    }

#pragma unroll
    for (int i = 0; i < bodies->death_stars_count; i++)
    {
        result->positions[positionsCount] = bodies->death_stars[i].position;
        ++positionsCount;
    }

#pragma unroll
    for (int i = 0; i < bodies->planets_count; i++)
    {
        result->positions[positionsCount] = bodies->planets[i].position;
        ++positionsCount;
    }

#pragma unroll
    for (int i = 0; i < bodies->x_wings_count; i++)
    {
        result->positions[positionsCount] = bodies->x_wings[i].position;
        ++positionsCount;
    }

    result->positions_count = positionsCount;
}
