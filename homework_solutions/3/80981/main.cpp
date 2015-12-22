#include<fstream>
#include<cstring>
#include<cstdio>
#include<cstdlib>
#include<cctype>
#include<immintrin.h>

/******/
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
/******/

const float G  = 6.67300E-11; 
enum ObjType { Planet, Asteroid, DeathStar, XWing };

struct Point3D{
    union{
        struct {
            float X;
            float Y;
            float Z;
            float buff;
        };
        __m128 data;
    };
    Point3D(){
        data = _mm_setzero_ps();
    }
    Point3D(__m128 vec) : data(vec){}
};

void operator*=(Point3D& p1, const Point3D& p2){
   p1.data = _mm_mul_ps(p1.data, p2.data); 
}

void operator/=(Point3D& p1, const Point3D p2){
   p1.data = _mm_div_ps(p1.data, p2.data); 
}

void operator+=(Point3D& p1, const Point3D& p2){
   p1.data = _mm_add_ps(p1.data, p2.data); 
}

void operator-=(Point3D& p1, Point3D & p2){
   p1.data = _mm_sub_ps(p1.data, p2.data); 
}
Point3D operator+(const Point3D& p1, const Point3D& p2){
   return Point3D(_mm_add_ps(p1.data, p2.data)); 
}

Point3D operator-(const Point3D& p1, const Point3D& p2){
   return Point3D(_mm_sub_ps(p1.data, p2.data)); 
}
float getLengthSq(const Point3D& p){
   //0x71 = 0111 0001 
   return _mm_cvtss_f32(_mm_dp_ps(p.data, p.data, 0x71));
}
Point3D operator*(const Point3D& p, float scalar){
    __m128 mult = _mm_set1_ps(scalar);
    return Point3D(_mm_mul_ps(p.data, mult)); 
}
void operator*=(Point3D& p, float scalar){
    __m128 mult = _mm_set1_ps(scalar);
    p.data = _mm_mul_ps(p.data, mult); 
}
void operator/=(Point3D& p, float scalar){
    __m128 mult = _mm_set1_ps(scalar);
    p.data = _mm_div_ps(p.data, mult); 
}
Point3D getNormal(const Point3D& p){
    __m128 iNorm = _mm_rsqrt_ps(_mm_dp_ps(p.data, p.data, 0x77));
    return Point3D(_mm_mul_ps(p.data, iNorm));
}
void clear(Point3D& p){
    p.data = _mm_setzero_ps();
}

struct BodyList{
    BodyList(int len) : flistIndex(0), plistIndex(0), listLength(len)
    {
        id            = new int[listLength];
        type          = new ObjType[listLength];
        mass          = new float[listLength];
        size          = new float[listLength];
        fuel          = new float[listLength];
        consumption   = new float[listLength];
        range         = new float[listLength];
        position      = new Point3D[listLength];
        speed         = new Point3D[listLength];
        acceleration  = new Point3D[listLength];
    }
    BodyList(const BodyList&) = delete;
    BodyList$ operator=(const BodyList&) = delete;
    ~BodyList()
    {
       delete[] id;          
       delete[] type;        
       delete[] mass;        
       delete[] size;        
       delete[] position;    
       delete[] speed;       
       delete[] acceleration;
       delete[] fuel;        
       delete[] consumption; 
       delete[] range;       
    }

    int     *id;
    ObjType *type;
    union{
        struct{
            float   *mass;
            float   *size;
            float   *fuel;
            float   *consumption;
            float   *range;
        };
        float *flist[5];
    };
    union{
        struct{
            Point3D *position;
            Point3D *speed;
            Point3D *acceleration;
        };
        Point3D *plist[3];
    };
    int flistIndex;
    int plistIndex;
    const int listLength;

    float* getFIndex(){
        ++flistIndex;
        return flist[flistIndex - 1];
    }

    Point3D* getPIndex(){
        ++plistIndex;
        return plist[plistIndex - 1];
    }

    void resetIndexes(){
        flistIndex = 0;
        plistIndex = 0;
    }

};

BodyList *objects;

void putInt(int i, char *str){
    objects->id[i] = std::atoi(str);
}

void putType(int i, char *str){
    if(std::strcmp(str, "Planet") == 0)
        objects->type[i] = Planet;
    else if(std::strcmp(str, "Asteroid") == 0)
        objects->type[i] = Asteroid;
    else if(std::strcmp(str, "Death Star") == 0)
        objects->type[i] = DeathStar;
    else if(std::strcmp(str, "X-Wing") == 0)
        objects->type[i] = XWing;
}

void putFloat(int i, char *str){
    objects->getFIndex()[i] = std::atof(str);
}

void putPoint(int pos, char *str){
    int del1 = -1;
    int del2 = -1;
    bool inFloat = false;
    for(int i = 0; str[i] != '\0' && del2 == -1; ++i){
        if(!inFloat && std::isdigit(str[i]))
            inFloat = true;
        else if(inFloat && str[i] == ' '){
            (del1 == -1 ? del1 : del2) = i;
            inFloat = false;
            str[i] = '\0';
        }
    }
    Point3D *currPointer = objects->getPIndex();

    currPointer[pos].X = std::atof(str);
    currPointer[pos].Y = std::atof(str+del1+1);
    currPointer[pos].Z = std::atof(str+del2+1);
}

bool isWhitespace(const char *buff){
    while(*buff == ' ')
        ++buff;
    return *buff == '\0';
}

void Universe_Initialize(const char* file)
{
    std::ifstream initFile(file);
    if(!initFile.is_open())
        return;

    typedef std::string::iterator SymbIt;
    std::string line;

    std::getline(initFile, line);
    int bodyCount = std::atoi(line.c_str());

    objects = new BodyList(bodyCount);
    void (*funcTable[10]) (int, char*);

    funcTable[0] = putInt;
    funcTable[1] = putType;
    funcTable[2] = putFloat;
    funcTable[3] = putFloat;
    funcTable[4] = putPoint;
    funcTable[5] = putPoint;
    funcTable[6] = putPoint;
    funcTable[7] = putFloat;
    funcTable[8] = putFloat;
    funcTable[9] = putFloat;

    char lineBuff[256];
    int currVal;
    int lineCounter = 0;
    while(std::getline(initFile, line)){
        std::strcpy(lineBuff, line.c_str());
        char *buff = std::strtok(lineBuff, "<>[]");
        currVal = 0;
        objects->resetIndexes();
        while(buff != NULL){
            if(!isWhitespace(buff))
                funcTable[currVal++](lineCounter, buff);
            buff = std::strtok(NULL, "<>[]");
        }
        ++lineCounter;
    }

    initFile.close();
}


void Universe_Run(float time, float delta, Result* result)
{
    float deltaSq = delta*delta;
    Point3D currPos;
    float GtimesCurrMass;
    Point3D vec;
    Point3D force;
    for(float currTime = 0.0f; currTime < time; currTime += delta){
        for(int i = 0; i < objects->listLength; ++i){
            if(objects->position[i].buff == -1.0f)
                continue;
            currPos = objects->position[i];
            GtimesCurrMass = G*objects->mass[i];
            clear(force);
            float currSize = objects->size[i];
            ObjType currObjType = objects->type[i];
            float vecLengthSq;
            for(int j = 0; j < objects->listLength; ++j){
                if(i==j || objects->position[j].buff == -1.0f) continue;
                vec.data = objects->position[j].data;
                vec-=currPos;
                currSize += objects->size[j];
                currSize *= currSize;
                vecLengthSq = getLengthSq(vec);
                if(vecLengthSq <= currSize){
                    objects->position[i].buff = -1.0f;
                    objects->position[j].buff = -1.0f;
                    result->collisions[result->collisions_count].body1 = objects->id[i];
                    result->collisions[result->collisions_count].body2 = objects->id[j];
                    result->collisions[result->collisions_count].time = currTime;
                    ++(result->collisions_count);
                }
                if((currObjType == DeathStar && (objects->type[j] == Planet || objects->type[j] == Asteroid))||
                        (currObjType == XWing && (objects->type[j] == DeathStar || objects->type[j] == Asteroid))){
                    if(vecLengthSq <= objects->range[i]*objects->range[i]){
                        objects->position[j].buff = -1.0f;
                        result->destructions[result->destructions_count].destructor = objects->id[i];
                        result->destructions[result->destructions_count].destructed = objects->id[j];
                        result->destructions[result->destructions_count].time = currTime;
                    }
                }
                force+=getNormal(vec)*((GtimesCurrMass*objects->mass[j])/getLengthSq(vec));
            }
            force/=objects->mass[i];
            if(objects->type[i] == DeathStar || objects->type[i] == XWing){
                objects->fuel[i] -= objects->consumption[i]*delta;
                if(objects->fuel[i] <= 0.0f)
                    clear(objects->acceleration[i]);
                force+=objects->acceleration[i];
            }
            force*=(deltaSq/2.0f);
            objects->position[i]+=(objects->speed[i]*delta)+force;
        }
    }
}

