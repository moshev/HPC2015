#include <iostream>
#include <cstring>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include <list>
#include <cmath>
#include <algorithm>
struct Vector3D{
    float x, y, z;

    struct Vector3D& operator+(Vector3D const& rhs){
        this->x += rhs.x;
        this->y += rhs.y;
        this->z += rhs.z;
        return *this;
    }
    struct Vector3D& operator*(float c){
        this->x *= c;
        this->y *= c;
        this->z *= c;
        return *this;
    }
    Vector3D(){}
    Vector3D(float xx, float yy, float zz): x(xx), y(yy), z(zz){

    }
};
struct Destruction{
    float time; // timestamp of the destruction
    int destructor; // id of the Death Star or X-Wing
    int destructed; // id of the destroyed body

    Destruction(){}
    Destruction(float time, int dor, int ded):time(time), destructor(dor), destructed(ded){}
};

struct Collision{
    float time; // timestamp of the collision
    int body1; // id of the first collided body
    int body2; // id of the second collided body
    Collision(){}
    Collision(float time, int body1, int body2):time(time), body1(body1), body2(body2){}
};

struct BodyPosition{
  int body;
  Vector3D position;
  BodyPosition(){}
  BodyPosition(int body, Vector3D position):body(body), position(position){}
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
std::vector<int> IDs;
std::vector<char> types; // 1-Planet 2-Asteroid 3-Death Star 4-X-Wing
std::vector<float> masses;
std::vector<float> sizes;
std::vector<Vector3D> positions;
std::vector<Vector3D> speeds;

std::list<int> alive;

std::map<int, Vector3D> accelerations;
std::map<int, float> fuels;
std::map<int, float> consumptions;
std::map<int, float> ranges;

std::map<std::string, char> objectTypesToNumber;


const float GRAVITY_CONSTANT = 6.67408e-11;
const char PLANET = 1;
const char ASTEROID = 2;
const char DEATH_STAR = 3;
const char X_WING = 4;
const float PI_TIMES_3divBy4 = 2.356194f;
float time_elapsed = 0.f;

void Universe_Initialize(const char* file){

    objectTypesToNumber["Planet"] = 1;
    objectTypesToNumber["Asteroid"] = 2;
    objectTypesToNumber["Death Star"] = 3;
    objectTypesToNumber["X-Wing"] = 4;

    std::ifstream infile(file);
    std::string line;

    std::getline(infile, line);
    std::istringstream iss(line);
    iss >> N;

    IDs.resize(N);
    types.resize(N);
    masses.resize(N);
    sizes.resize(N);
    positions.resize(N);
    speeds.resize(N);


    int count = 0;


    while(std::getline(infile, line)){

        std::istringstream iss(line);
        
        int id;
        std::string type;
        int mass;
        float size;
        int px, py, pz;
        int sx, sy, sz;
        
        iss >> id >> type;
        if(type == "Death"){
            iss >> type;
            type = "Death Star";
        }
        
        iss >> mass >> size >> px >> py >> pz >> sz >>sy >>sz;
        
        IDs[count] =id;
        alive.push_back(count);
        types[count] =objectTypesToNumber[type];
        masses[count] =mass;
        sizes[count] =size;
        positions[count] = Vector3D(px, py, pz);
        speeds[count] =Vector3D(sx, sy, sz);

        if(type == "Death Star" || type == "X-Wing"){
            int ax, ay, az;
            float fuel;
            float consumption;
            float range;
            iss >> ax >> ay >> az >> fuel >> consumption >> range;
            accelerations[count] = Vector3D(ax, ay, az);
            fuels[count] = fuel;
            consumptions[count] = consumption;
            ranges[count] = range;

        }
        ++count;
        
    }
}

float Distance_Between_i_j(int i, int j){
    Vector3D a = positions[i];
    Vector3D b = positions[j];

    return std::sqrt( (a.x-b.x)*(a.x-b.x) + (a.y- b.y)*(a.y- b.y) + (a.z- b.z)*(a.z- b.z));
}

std::vector<std::list<int>::const_iterator> Check_For_Collision(std::list<int>::const_iterator i, Result* result){

    std::vector<std::list<int>::const_iterator> dead;
    int IthElNum = *i;
    char TypeOfIthEl = types[IthElNum];
    std::list<int>::const_iterator j;
    float IthElementRadius = std::pow(sizes[IthElNum]*PI_TIMES_3divBy4, 1/3);

    bool Idied = false;
    for(j = alive.begin(); j != alive.end(); ++j){
        int JthElNum = *j;
        float JthElementRadius = std::pow(sizes[JthElNum]*PI_TIMES_3divBy4, 1/3);

        if(IthElNum != JthElNum){

            float distance = Distance_Between_i_j(IthElNum, JthElNum);
            if (distance < (IthElementRadius+JthElementRadius)){
                //COLLISON
                Collision* coll = new Collision(time_elapsed, IDs[IthElNum], IDs[JthElNum]);
                dead.push_back(i);
                dead.push_back(j);

                Collision* res = result->collisions + result->collisions_count;
                *res = *coll;

                break;

            }else if((TypeOfIthEl == DEATH_STAR && 
                    (types[JthElNum] == PLANET || types[JthElNum] == ASTEROID)) ||
                    (TypeOfIthEl == X_WING && 
                    (types[JthElNum] == ASTEROID || types[JthElNum] == DEATH_STAR))){
                    //DESTRUCTION
                    if(JthElementRadius+ranges[IthElNum]>distance){
                        dead.push_back(j);
                        Destruction* destr = new Destruction(time_elapsed, IDs[IthElNum], IDs[JthElNum]);
                        Destruction* res = result->destructions + result->destructions_count;
                        *res = *destr;
                        result->destructions_count += 1;
                    }
            }
        }
    }
    return dead;
}



void Universe_Run(float time, float delta, Result* result){

    result->collisions_count = 0;
    result->destructions_count = 0;
    result->positions_count = 0;

    while(time_elapsed < time){
        //Check universe for DEATHS
        std::list<int>::const_iterator i;

        for(i = alive.begin(); i != alive.end();){
            int IthElementNumber = *i;
            std::vector<std::list<int>::const_iterator> dead = Check_For_Collision(i, result);
            
            bool again = true;
            ++i;
            while(i!=alive.end() && again){
                again = false;
                for(auto x: dead){
                    if(i == x){
                        ++i;
                        again = true;
                        break;
                    }
                }
            }
            for(auto x: dead){
                alive.erase(x);
            }
        }

        //Calculate new positions
        for(i=alive.begin(); i!=alive.end(); ++i){
            int IthElNum = *i;
            float ix = positions[IthElNum].x;
            float iy = positions[IthElNum].y;
            float iz = positions[IthElNum].z;
            float imass = masses[IthElNum];
            std::list<int>::const_iterator j;
            Vector3D Fi(0, 0, 0);
            for(j=alive.begin(); j!=alive.end(); ++j){
                if(i!=j){
                    float jx = positions[*j].x;
                    float jy = positions[*j].y;
                    float jz = positions[*j].z;
                    Vector3D Rij(jx-iy, jy-iy, jz-iz);
                    float LRij = Distance_Between_i_j(*i, *j);
                    Fi = Fi + ( Rij * (GRAVITY_CONSTANT*masses[*i]*masses[*j]/(LRij*LRij*LRij)));
                }
            }
            Vector3D a = Fi*(1/imass);
            if(types[IthElNum] == DEATH_STAR || types[IthElNum] == X_WING){
                a = a + accelerations[IthElNum];
            }
            Vector3D posTraveled = speeds[IthElNum]*delta + a*((delta*delta)/2);
            speeds[IthElNum] = speeds[IthElNum] + a*delta;
            positions[IthElNum] = positions[IthElNum] + posTraveled;
        }
        time_elapsed += delta;
    }
    std::list<int>::const_iterator i;
    for(i=alive.begin(); i!=alive.end(); ++i){
        BodyPosition* bpos = new BodyPosition(IDs[*i], positions[*i]);
        BodyPosition* res = result->positions + result->positions_count;
        *res = *bpos;
    }
}



int main(){
    return 0;
}
