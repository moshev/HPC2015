#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

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
} result;
enum Type
{
    Planet = 0,
    Asteroid = 1,
    DEATH_STAR = 2,
    X_WING = 3
};
float distance(const Vector3D& v1, const Vector3D& v2)
{
    return  std::sqrt( (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z) ) ;
}
float distance_sqr(const Vector3D& v1, const Vector3D& v2)
{
    return  (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z) ;
}
void multiply(Vector3D& v, float mul)
{
    v.x *= mul;
    v.y *= mul;
    v.z *= mul;
}
Vector3D sum(const Vector3D& v1, const Vector3D& v2)
{
    Vector3D result;
    result.x = v1.x + v2.x;
    result.y = v1.y + v2.y;
    result.z = v1.z + v2.z;
    return result;
}
float length(const Vector3D& v)
{
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}
float dot(const Vector3D& v1, const Vector3D& v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}
Vector3D r_ij(Vector3D first, Vector3D second)
{
    Vector3D result;
    result.x = second.x - first.x;
    result.y = second.y - first.y;
    result.z = second.z - first.z;
    return result;
}
Vector3D n_ij(Vector3D first, Vector3D second)
{
    Vector3D result = r_ij(first , second);
    double multiplier = 1.0 / std::sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
    result.x *= multiplier;
    result.y *= multiplier;
    result.z *= multiplier;
    return result;
}
Vector3D n_ij(Vector3D x)
{
    double multiplier = 1.0 / std::sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
    x.x *= multiplier;
    x.y *= multiplier;
    x.z *= multiplier;
    return x;
}
Vector3D F_ij(Vector3D first, float mass_first, Vector3D second, float mass_second)
{
    Vector3D rij = r_ij(first, second);
    Vector3D normal = n_ij(first,second);
    float top = (G * mass_first * mass_second);
    float bot = dot(rij, rij);
    multiply(normal, top/bot);
    return normal;
}
Vector3D F_i(size_t i, Vector3D* positions, float* masses, size_t count)
{
    Vector3D suma;suma.x = 0; suma.y = 0; suma.z= 0;
    for(size_t index = 0 ; index < count ; ++index)
    {
        if(index != i)
            suma = sum(suma, F_ij(positions[i], masses[i], positions[index], masses[index]));
    }
    return suma;
}
float acceleration(float force, float mass)
{
    return force/mass;
}
float path_passed(float t, float v, float a)
{
    return v*t + (a*t*t) / 2;
}

struct Bodies
{
    int* ids;
    Type* types;
    float* masses;
    float* sizes;

    Vector3D* positions;
    Vector3D* speeds;

    Vector3D* accelerations;
    float* fuels;
    float* consumptions;
    float* ranges;

    size_t count;
};
Bodies everything;
void init_size(size_t n)
{
    everything.ids = new int[n];
    everything.types = new Type[n];
    everything.masses = new float[n];
    everything.sizes = new float[n];
    everything.positions = new Vector3D[n];
    everything.speeds = new Vector3D[n];
    everything.accelerations = new Vector3D[n];
    everything.fuels = new float[n];
    everything.consumptions = new float[n];
    everything.ranges = new float[n];

    everything.count = 0;
}
void Universe_Initialize(const char* file)
{
    std::ifstream input(file, std::ios::in);

    if(!input)
        throw "file not okay";
    char read;
    int temp;
    size_t N;
    input >> N;
    init_size(N);
    std::cout << N << std::endl;
    std::string str;
    std::getline(input, str);

    for(size_t i = 0 ; i < N; ++i)
    {
        input >> everything.ids[everything.count];
        input >> temp;
        everything.types[everything.count] = (Type)temp;
        input >> everything.masses[everything.count];
        input >> everything.sizes[everything.count];
        input >> everything.positions[everything.count].x;
        input >> everything.positions[everything.count].y;
        input >> everything.positions[everything.count].z;
        input >> everything.speeds[everything.count].x;
        input >> everything.speeds[everything.count].y;
        input >> everything.speeds[everything.count].z;

        do
        {
            input.get(read);
        }while(read != '[' && read != '\n');

        if(read == '[')
        {
            input >> everything.accelerations[everything.count].x;
            input >> everything.accelerations[everything.count].y;
            input >> everything.accelerations[everything.count].z;
            input >> everything.fuels[everything.count];
            input >> everything.consumptions[everything.count];
            input >> everything.ranges[everything.count];
        }
        else
        {
            everything.accelerations[everything.count].x = 0;
            everything.accelerations[everything.count].y = 0;
            everything.accelerations[everything.count].z = 0;
            everything.fuels[everything.count] = 0;
            everything.consumptions[everything.count] = 0;
            everything.ranges[everything.count] = 0;
        }
        ++everything.count;
        if(read != '\n')
            std::getline(input, str);

    }
}

template<typename T>
void shift(T* s, size_t pos, size_t length)
{
    --length;
    while(pos < length)
    {
        s[pos] = s[pos + 1];
        ++pos;
    }
}

void remove_body(size_t index)
{
    shift(everything.accelerations, index, everything.count);
    shift(everything.consumptions, index, everything.count);
    shift(everything.fuels, index, everything.count);
    shift(everything.ids, index, everything.count);
    shift(everything.masses, index, everything.count);
    shift(everything.positions, index, everything.count);
    shift(everything.ranges, index, everything.count);
    shift(everything.sizes, index, everything.count);
    shift(everything.speeds, index, everything.count);
    shift(everything.types, index, everything.count);
    --everything.count;
}
void print_vector(const Vector3D& v)
{
    std::cout <<"(" <<  v.x << "," << v.y <<","<< v.z << ")";
}
bool in_colision(const Vector3D& v1, const Vector3D& v2, float range)
{
    print_vector(v1);
    print_vector(v2);
    std::cout << range << std::endl;
    if(distance_sqr(v1, v2) <= range*range)
        return true;
    return false;
}
void colisions(float t)
{
    for(size_t i = 0 ; i < everything.count; ++i)
    {
        for(size_t j = i + 1; j < everything.count ; ++j)
        {
            if(everything.types[i] == Type::X_WING)
            {
                if(everything.types[j] == Type::X_WING)
                {
                    // ako i-toto XWING e s pogolqm radius ot j-toto XWINGche togava itoto unishtojava j-toto
                    if(everything.ranges[i] >= everything.ranges[j])
                    {
                        if(in_colision(everything.positions[i], everything.positions[j], everything.ranges[i]))
                        {
                            // the j-toto is now destroyed
                            result.destructions[result.destructions_count].destructor = everything.ids[i];
                            result.destructions[result.destructions_count].destructed = everything.ids[j];
                            result.destructions[result.destructions_count++].time = t;
                            //now remove the j-object
                            remove_body(j--);
                        }
                    }
                    else
                    {
                        if(in_colision(everything.positions[j], everything.positions[i], everything.ranges[j]))
                        {
                            // the i-toto is now destroyed
                            result.destructions[result.destructions_count].destructor = everything.ids[j];
                            result.destructions[result.destructions_count].destructed = everything.ids[i];
                            result.destructions[result.destructions_count++].time = t;
                            //now remove the i-object
                            remove_body(i);
                            j = i+1;
                        }
                    }
                }
                else
                {
                    if(in_colision(everything.positions[i], everything.positions[j], everything.ranges[i]))
                    {
                        // the j-toto is now destroyed
                        result.destructions[result.destructions_count].destructor = everything.ids[i];
                        result.destructions[result.destructions_count].destructed = everything.ids[j];
                        result.destructions[result.destructions_count++].time = t;
                        //now remove the j-object
                        remove_body(j--);
                    }
                }
            }
            else if(everything.types[i] == Type::DEATH_STAR)
            {
                if(everything.types[j] == Type::DEATH_STAR)
                {
                    // ako i-toto XWING e s pogolqm radius ot j-toto XWINGche togava itoto unishtojava j-toto
                    if(everything.ranges[i] >= everything.ranges[j])
                    {
                        if(in_colision(everything.positions[i], everything.positions[j], everything.ranges[i]))
                        {
                            // the j-toto is now destroyed
                            result.destructions[result.destructions_count].destructor = everything.ids[i];
                            result.destructions[result.destructions_count].destructed = everything.ids[j];
                            result.destructions[result.destructions_count++].time = t;
                            //now remove the j-object
                            remove_body(j--);
                        }
                    }
                    else
                    {
                        if(in_colision(everything.positions[j], everything.positions[i], everything.ranges[j]))
                        {
                            // the i-toto is now destroyed
                            result.destructions[result.destructions_count].destructor = everything.ids[j];
                            result.destructions[result.destructions_count].destructed = everything.ids[i];
                            result.destructions[result.destructions_count++].time = t;
                            //now remove the i-object
                            remove_body(i);
                            j = i+1;
                        }
                    }
                }
                else if (everything.types[j] != Type::X_WING &&
                         (in_colision(everything.positions[i], everything.positions[j], everything.ranges[i])))
                {
                    // the j-toto is now destroyed
                    result.destructions[result.destructions_count].destructor = everything.ids[i];
                    result.destructions[result.destructions_count].destructed = everything.ids[j];
                    result.destructions[result.destructions_count++].time = t;
                    //now remove the j-object
                    remove_body(j--);
                }
                else // it's an XWING
                {
                    if(in_colision(everything.positions[j], everything.positions[i], everything.ranges[j]))
                    {
                        // the j-toto is now destroyed
                        result.destructions[result.destructions_count].destructor = everything.ids[j];
                        result.destructions[result.destructions_count].destructed = everything.ids[i];
                        result.destructions[result.destructions_count++].time = t;
                        //now remove the j-object
                        remove_body(i);
                        j = i + 1;
                    }
                }
            }
            else // i-th element is Planet or Asteroid
            {
                if(in_colision(everything.positions[j], everything.positions[i], everything.ranges[j]))
                {
                    // the i-toto is now destroyed
                    result.destructions[result.destructions_count].destructor = everything.ids[j];
                    result.destructions[result.destructions_count].destructed = everything.ids[i];
                    result.destructions[result.destructions_count++].time = t;
                    //now remove the i-object
                    remove_body(i);
                    j = i+1;
                }
            }
        }
    }
}

void move(size_t i, float time)
{
    if((everything.types[i] == X_WING || everything.types[i] == DEATH_STAR) &&
       (everything.fuels[i] > everything.consumptions[i]))
    {
        Vector3D a = everything.accelerations[i];
        multiply(a, time*time* 0.5);
        Vector3D v = everything.speeds[i];
        multiply(v, time);
        everything.positions[i] = sum(a, v);
    }
    else
    {
        Vector3D a = F_i(i, everything.positions, everything.masses, everything.count);
        Vector3D pos = everything.speeds[i];
        multiply( pos, time);
        multiply(a, time*time* 0.5);

        pos = sum(pos, a);
        everything.positions[i] = pos;
    }
}

void move_all(float delta)
{
    for(size_t i = 0 ; i < everything.count ; ++i)
    {
        move(i, delta);
    }
}
void Universe_Run(float time, float delta, Result* result)
{
    for(float t = 0 ; t < time ; t+=delta)
    {
        colisions(t);
        move_all(delta);
    }
}

int main()
{
    Universe_Initialize("C:\\Users\\KejoT\\Desktop\\map.txt");
    result.destructions = new Destruction[1000];
    result.collisions = new Collision[1000];
    result.positions = new BodyPosition[1000];
    for(size_t i = 0 ; i < everything.count ; ++i)
    {
        std::cout << everything.ids[i] << " "
        << (int)everything.types[i] << " "
        << everything.masses[i] << " "
        << everything.sizes[i] << " "
        << everything.positions[i].x << " "
        << everything.positions[i].y << " "
        << everything.positions[i].z << " "
        << everything.speeds[i].x << " "
        << everything.speeds[i].y << " "
        << everything.speeds[i].z << std::endl;
    }
    Universe_Run(10, 1, &result);
    std::cout << " " << std::endl << std::endl;
    std::cout << "Destructons" << std::endl;
    for(size_t i = 0 ; i < result.destructions_count ; ++i)
    {
        std::cout << result.destructions[i].time << " " << result.destructions[i].destructor << " " << result.destructions[i].destructed << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Collisions" << std::endl;
    for(size_t i = 0 ; i < result.collisions_count ; ++i)
    {
        std::cout << result.collisions[i].time << " " << result.collisions[i].body1 << " " << result.collisions[i].body2 << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Positions" << std::endl;
    for(size_t i = 0 ; i < result.positions_count ; ++i)
    {
        Vector3D temp = result.positions[i].position;
        std::cout << result.positions[i].body << " (" << temp.x << "," << temp.y << "," << temp.z << ")" << std::endl;
    }
    std::cout << std::endl;
    return 0;
}
