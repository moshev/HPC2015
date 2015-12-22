#include<iostream>
#include<fstream>
#include<cmath>

using namespace std;
const float G = 6.674e-11;

int main()
{
    
    
    return 0;
}

struct Vector3D
{
    float x, y, z;
    
    Vector3D(float x = 0.f, float y = 0.f, float z = 0.f): x(x), y(y), z(z) {}
    
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

struct Bodies
{
    int* ids;
    string* types;
    float* mass;
    float* size;
    float* xs;
    float* ys;
    float* zs;
    float* speedsX;
    float* speedsY;
    float* speedsZ;
    float* accX;
    float* accY;
    float* accZ;
    float* fuels;
    float* cons;
    float* ranges;


    int countAll;
    
    
    void allocateFirst(int n)
    {
        this->ids = new int[n];
        this->types = new string[n];
        this->mass = new float[n];
        this->size = new float[n];
        this->xs = new float[n];
        this->ys = new float[n];
        this->zs = new float[n];
        this->speedsX = new float[n];
        this->speedsY = new float[n];
        this->speedsZ = new float[n];
        this->accX = new float[n];
        this->accY = new float[n];
        this->accZ = new float[n];
        this->fuels = new float[n];
        this->cons = new float[n];
        this->ranges = new float[n];

        this->countAll = n;
    }
    
    
    void deallocate()
    {
        delete [] this->ids;
        delete [] this->types;
        delete [] this->mass;
        delete [] this->size;
        delete [] this->xs;
        delete [] this->ys;
        delete [] this->zs;
        delete [] this->speedsX;
        delete [] this->speedsY;
        delete [] this->speedsZ;
        delete [] this->accX;
        delete [] this->accY;
        delete [] this->accZ;
        delete [] this->fuels;
        delete [] this->cons;
        delete [] this->ranges;
        
    }
};



Bodies bod;
void Universe_Initialize(const char* file)
{
    ifstream data (file);
    
    int n;
    data >> n;
    bod.allocateFirst(n);

    
    for(int i = 0; i < n; ++i)
    {
        data >> bod.ids[i]  >> bod.types[i] >> bod.mass[i] >> bod.size[i] >> bod.xs[i];
        data >> bod.ys[i] >>  bod.zs[i] >> bod.speedsX[i] >> bod.speedsY[i] >> bod.speedsZ[i];
        
        if(bod.types[i] == "X-Wing" || bod.types[i] == "Death Star")
        {
            data >> bod.accX[i] >> bod.accY[i] >> bod.accZ[i] >> bod.fuels[i];
            data >> bod.cons[i] >> bod.ranges[i];
        } else {
            bod.accX[i] = bod.accY[i] = bod.accZ[i]  = 0;
            bod.fuels[i] = 1;
            bod.cons[i] = 0;
            bod.ranges[i] = 0;
        }
    }
}


float length(Vector3D &vec) {
    return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}


void Universe_Run(float time, float delta, Result* result)
{
    // magic индийд
    
    float currentTime = 0.f;
    
    while (currentTime < time)
    {
        
        for (int i = 0; i < bod.countAll; ++i)
        {
            Vector3D theForce;
            
            for (int j = 0; j < bod.countAll; ++j)
            {
                if(i == j || bod.types[j] == "Dead") continue;
                
                float distance = (bod.xs[j] - bod.xs[i]) +
                (bod.ys[j] - bod.ys[i]) +
                (bod.zs[j] - bod.zs[i]);
                float force = (G * bod.mass[i] * bod.mass[j]) / distance ;
                
                Vector3D toNormalize = Vector3D(bod.xs[j] - bod.xs[i],
                                                bod.ys[j] - bod.ys[i], bod.zs[j] - bod.zs[i]);
                
                theForce.x = theForce.x + force * (toNormalize.x/length(toNormalize));
                theForce.y = theForce.y + force * (toNormalize.y/length(toNormalize));
                theForce.z = theForce.z + force * (toNormalize.z/length(toNormalize));
            }
            
            theForce.x = theForce.x/bod.mass[i] + (bod.fuels[i] >  bod.cons[i] ? bod.accX[i] : 0);
            theForce.y = theForce.y/bod.mass[i] + (bod.fuels[i] >  bod.cons[i] ? bod.accY[i] : 0);
            theForce.z = theForce.z/bod.mass[i] + (bod.fuels[i] >  bod.cons[i] ? bod.accZ[i] : 0);
            
            bod.xs[i] = bod.xs[i] + bod.speedsX[i] * delta + (theForce.x * delta * delta) * 0.5;
            bod.ys[i]  = bod.ys[i] + bod.speedsY[i] * delta + (theForce.x * delta * delta) * 0.5;
            bod.zs[i] = bod.zs[i] + bod.speedsZ[i] * delta + (theForce.x * delta * delta) * 0.5;
            
            bod.fuels[i] -= bod.cons[i] * delta;
            
            bod.speedsX[i] = bod.speedsX[i] + theForce.x * delta;
            bod.speedsY[i] = bod.speedsY[i] + theForce.y * delta;
            bod.speedsZ[i] = bod.speedsZ[i] + theForce.z * delta;
            
        }
        
        for (int i = 0; i < bod.countAll; ++i)
        {
            for (int j = 0; j < bod.countAll; ++j)
            {
                if(i == j || bod.types[i] == "Dead" || bod.types[j] == "Dead" ) continue;
                
                float distance = (bod.xs[j] - bod.xs[i]) +
                (bod.ys[j] - bod.ys[i]) +
                (bod.zs[j] - bod.zs[i]);
                
                float collisionDist = (bod.size[i] / 2 + bod.size[j] / 2 ) * (bod.size[i] / 2 + bod.size[j] / 2 );
                
                if(distance < collisionDist)
                {
                    bod.types[i] = bod.types[j] = "Dead";
                    Collision collision;
                    collision.time = currentTime;
                    collision.body1 = bod.ids[i];
                    collision.body2 = bod.ids[j];
                    result->collisions[result->collisions_count++] = collision;
                    break;
                }
                
                if(distance < bod.ranges[j] * bod.ranges[j])
                {
                    if(bod.types[j] == "X-Wing")
                    {
                        if(bod.types[i] == "Asteroid" || bod.types[i] == "Death Star")
                        {
                            bod.types[i] = "Dead";
                            Destruction destruction;
                            destruction.time = currentTime;
                            destruction.destructor = bod.ids[j];
                            destruction.destructed = bod.ids[i];
                            
                            result->destructions[result->destructions_count++] = destruction;
                            
                        }
                        else if(bod.types[j] == "Death Star")
                        {
                            if(bod.types[i] == "Asteroid" || bod.types[i] == "Planet")
                            {
                                bod.types[i] = "Dead";
                                Destruction destruction;
                                destruction.time = currentTime;
                                destruction.destructor = bod.ids[j];
                                destruction.destructed = bod.ids[i];
                                result->destructions[result->destructions_count++] = destruction;
                            }
                        }
                    }
                }
                
            }
        }
        
        currentTime += delta;
    }
    
    for(int i = 0; i < bod.countAll; ++i)
    {
        Vector3D pos = Vector3D(bod.xs[i], bod.ys[i], bod.zs[i]);
        BodyPosition bodPos;
        bodPos.body = bod.ids[i];
        bodPos.position = pos;
        result->positions[result->positions_count++] = bodPos;

    }
    
    bod.deallocate();
}

