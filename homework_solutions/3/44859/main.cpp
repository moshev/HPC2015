#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include <cmath>


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


struct ActiveBody
{
    int body;

    // x - fuel, y - consumption, z - range
    Vector3D power;
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


std::vector<float> masses;
std::vector<float> sizes;
std::vector<std::string> types;
std::vector<BodyPosition> positions;
std::vector<Vector3D> speeds;
std::vector<Vector3D> accelarations;
std::vector<ActiveBody> active_bodies;
int N = 0;
int destructions_cnt = 0;
int collisions_cnt = 0;
#define G 6.673e-11
 

void Universe_Initialize(const char* filename)
{
    std::ifstream file;
    file.open(filename);
    if (!file.is_open())
    {
        return;
    }

    file >> N;
    // reserve size for mandatory properties
    masses.resize(N);
    sizes.resize(N);
    types.resize(N);
    positions.resize(N);
    speeds.resize(N);
    accelarations.resize(N);

    BodyPosition pos;
    std::string type;
    float mass;
    float size;
    Vector3D speed;
    Vector3D acc;
    ActiveBody active_body;

    for(int i = 0; i < N; ++i)
    {
        memset(&pos, 0, sizeof(BodyPosition));
        type = "";
        mass = 0;
        size = 0;
        speed = {0, 0, 0};
        acc = {0, 0, 0};

        file >> pos.body;
        file >> type;
        if (type == "Death")
        {
            file >> type;
            type = "Death " + type;
        }

        file >> mass >> size >> pos.position.x >> pos.position.y >> pos.position.z >> speed.x >> speed.y >> speed.z;
        if (type == "Death Star" || type == "X-Wing")
        {
            memset(&active_body, 0, sizeof(ActiveBody));
            file >> acc.x >> acc.y >> acc.z;
            file >> active_body.power.x >> active_body.power.y >> active_body.power.z;
//             printf("acc: %f, %f, %f, fuel: %f, cons: %f, range: %f\n",
//                     acc.x, acc.y, acc.z, fuel, consumption, range);

            active_body.body = pos.body;
            active_bodies.push_back(active_body);
        }
        else
        {
            acc.x = 1;
            acc.y = 1;
            acc.z = 1;
        }

        positions[pos.body] = pos;
        types[pos.body] = type;
        masses[pos.body] = mass;
        sizes[pos.body] = size;
        speeds[pos.body] = speed;
        accelarations[pos.body] = acc;
    }
}


float VectorLength(const Vector3D& v)
{
    return sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
}


// find active body data by id and return index
// in active_bodies array
int FindActiveBody(int index, ActiveBody& active_body)
{
    size_t size = active_bodies.size();
    for(size_t i = 0; i < size; ++i)
    {
        if (active_bodies[i].body == index)
        {
            active_body = active_bodies[i];
            return i;
        }
    }

    return (-1);
}


bool InDeadRange(const BodyPosition& body)
{
    std::string body_type = types[body.body];
    std::string killer = "";    
    if (body_type == "Death Star")
    {
        killer = "X-Wing";
    }
    else
    {
        killer = "Death Star";
    }

    size_t size = active_bodies.size();
    for(int i = 0; i < size; ++i)
    {
        // get death bodies positions
        ActiveBody& active_body = active_bodies[i];
        Vector3D& active_pos = positions[active_body.body].position;
        std::string active_type = types[active_body.body];
        if (active_type != killer)
            continue;

        float range = active_body.power.z;
        Vector3D dist_vect;
        dist_vect.x = active_pos.x - body.position.x;
        dist_vect.x = active_pos.y - body.position.y;
        dist_vect.x = active_pos.z - body.position.z;

        float distance = VectorLength(dist_vect);
        if (distance < range)
        {
             return true;
        }
    }

    return false;
}

Destruction DestructObject(int id)
{
//     masses.erase(id);
//     sizes.erase(id);
//     types.erase(id);
//     positions.erase(id);
//     speeds.erase(id);
//     accelarations.erase(id);
// 
    ActiveBody active;
    int index = FindActiveBody(id, active);
    if (index != -1)
    {
        // remove
    }

    Destruction info;
    info.time = 0;
    info.destructed = id;
    info.destructor = active.body;

    return info;
}

void Universe_Run(float time, float delta, Result* result)
{
    float start_time = 0;
    while(start_time <= time)
    {
        start_time += delta;
        for(size_t i = 0; i < N; ++i)
        {
            BodyPosition& obj1 = positions[i];
            float obj1_mass = masses[i];
            Vector3D force_vect = {0, 0, 0};
            for(size_t j = 0; j < N; ++j)
            {
                if (i == j)
                    continue;

                const BodyPosition& obj2 = positions[j];
                if (obj1.position.x == obj2.position.x &&
                    obj1.position.y == obj2.position.y &&
                    obj1.position.z == obj2.position.z)
                {
                    ++collisions_cnt;
                }
                
                Vector3D d_vect;
                d_vect.x = obj1.position.x - obj2.position.x;
                d_vect.y = obj1.position.y - obj2.position.y;
                d_vect.z = obj1.position.z - obj2.position.z;
                float distance = VectorLength(d_vect);
                float force = (G * obj1_mass * masses[obj2.body]) / pow(distance, 2);

                force_vect.x += (force * d_vect.x) / distance;
                force_vect.y += (force * d_vect.y) / distance;
                force_vect.z += (force * d_vect.z) / distance;
            }

            Vector3D& obj_speed = speeds[i];
            Vector3D& obj_acc = accelarations[i];
            obj_speed.x += (start_time * force_vect.x * obj_acc.x) / obj1_mass;
            obj_speed.y += (start_time * force_vect.y * obj_acc.y) / obj1_mass;
            obj_speed.z += (start_time * force_vect.z * obj_acc.z) / obj1_mass;

            obj1.position.x = (start_time * obj_speed.x);
            obj1.position.y = (start_time * obj_speed.y);
            obj1.position.z = (start_time * obj_speed.z);

            // recalculate fuel and accelaration
            if (obj_acc.x > 1 || obj_acc.y > 1 || obj_acc.z > 1)
            {
                ActiveBody active_body;
                int found = FindActiveBody(i, active_body);
                // found == -1 => wrong input data

                active_body.power.x -= active_body.power.y * time;

                if (active_body.power.x <= 0)
                {
                    obj_acc.x = 1;
                    obj_acc.y = 1;
                    obj_acc.z = 1;
                }
            }

            // check if body is in death range
            if (InDeadRange(obj1)) {
                Destruction info = DestructObject(obj1.body); // remove all data for this object
                info.time = start_time;
                result->destructions[i] = info;
                ++destructions_cnt;
            }

        }
    }

    result->collisions_count = collisions_cnt;
    result->destructions_count = destructions_cnt;
    result->positions_count = positions.size();
    for(int i = 0; i < positions.size(); ++i)
    {
        result->positions[i] = positions[i];
    }
}


void DebugPrint()
{
    // print global state of universe
    for(int i = 0; i < N; ++i)
    {
        const BodyPosition& body = positions[i];
        const Vector3D& speed = speeds[i];

        printf("(id: %d)\n(type: %s)\n(mass: %f)\n(size: %f)\n(x, y, z: %f, %f, %f)\n(speed: %f %f %f)\n",
               body.body, types[body.body].c_str(), masses[body.body],
               sizes[body.body], body.position.x, body.position.y,
               body.position.z, speed.x, speed.y, speed.z);

        if (types[body.body] == "Death Star" || types[body.body] == "X-Wing")
        {
            const Vector3D& acc = accelarations[body.body];

            const ActiveBody* active_body = NULL;
            size_t size = active_bodies.size();
            for (int i = 0; i < size; ++i)
            {
                if (active_bodies[i].body == body.body)
                {
                    active_body = &active_bodies[i];
                }
            }

            assert(active_body != NULL);

            printf("(acc: (x, y, z): (%f, %f, %f))\n(fuel: %f)\n(consumption: %f)\n(range: %f)\n",
                   acc.x, acc.y, acc.z, active_body->power.x, active_body->power.y, active_body->power.z);
        }

        printf("\n");
    }
}

