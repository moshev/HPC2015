
#include <fstream>
#include <string>
#include <vector>

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

struct SOAVector
{
    float *Xs;
    float *Ys;
    float *Zs;
};

// Objects are stored like this XXXDDDAAAPPP
// Where X-s are X-Wings
// D-s are Death Stars
// A-s are Asteroids
// P-s are Planets

struct UniverseSimulatedBodies
{
    int NumXWings;
    int NumDeathStars; // Death starts from index NumXWings
    int NumAsteroirds; // These start at index NumXWings + NumDeathStars
    int NumPlanets;    // These start at index NumXwings + NumDeathStars + NumAsteroirds

    int *Ids;

    float *Masses;
    float *Sizes;

    SOAVector Positions;

    SOAVector Speeds;

    // Only X-Wings and DeathStars has those There number are NumXWings + NumDeathStars
    SOAVector Accels;

    float *Fuels;
    float *Consumtions;
    float *Ranges;

    void Allocate(int n, int m)
    {
        NumXWings = 0;
        NumDeathStars = 0;
        NumAsteroirds = 0;
        NumPlanets = 0;

        Ids = new int[n];
        Masses = new float[n];
        Sizes = new float[n];
        Positions.Xs = new float[n];
        Positions.Ys = new float[n];
        Positions.Zs = new float[n];

        Speeds.Xs = new float[n];
        Speeds.Ys = new float[n];
        Speeds.Zs = new float[n];

        // These are needed only for part of the object so we have another size
        Fuels = new float[m];
        Consumtions = new float[m];
        Ranges = new float[m];

        Accels.Xs = new float[m];
        Accels.Ys = new float[m];
        Accels.Zs = new float[m];
    }

    ~UniverseSimulatedBodies()
    {
        delete[] Ids;
        delete[] Masses;
        delete[] Sizes;

        delete[] Positions.Xs;
        delete[] Positions.Ys;
        delete[] Positions.Zs;

        delete[] Speeds.Xs;
        delete[] Speeds.Ys;
        delete[] Speeds.Zs;

        delete[] Fuels;
        delete[] Consumtions;
        delete[] Ranges;

        delete[] Accels.Xs;
        delete[] Accels.Ys;
        delete[] Accels.Zs;
    }
};

UniverseSimulatedBodies gUniverse;

void Universe_Initialize(const char* file)
{
    std::ifstream f(file);

    unsigned Num;
    f >> Num;

    // These are temporary vectors to store the result. After everything is read they will be stored in the final struct
    struct Body { int id; float mass; float size; float posx, posy, posz; float speedx, speedy, speedz; };
    struct BodyWithAcc : public Body { float accx, accy, accz; float fuel; float cons; float range; };
    std::vector<Body> planets;
    std::vector<Body> asteroids;
    std::vector<BodyWithAcc> xwings;
    std::vector<BodyWithAcc> deathstars;

    planets.reserve(Num);
    asteroids.reserve(Num);
    xwings.reserve(Num);
    deathstars.reserve(Num);

    enum Type
    {
        Planet, Asteroid, DeathStar, XWing
    };

    while(Num--) {
        int id;
        f >> id;

        Type t;
        std::string type;
        f >> type;
        if(type == "Planet") {
            t = Type::Planet;
        } else if(type == "Asteroid") {
            t = Type::Asteroid;
        } else if(type == "Death") {
            t = Type::DeathStar;
            f >> type; // Eat Star string
        } else if(type == "X-Wing") {
            t = Type::XWing;
        }

        float mass;
        float size;
        float px, py, pz, sx, sy, sz;
        f >> mass >> size >> px >> py >> pz >> sx >> sy >> sz;

        if(t == Type::DeathStar || t == Type::XWing) {
            float accx, accy, accz;
            float fuel, consumation, range;
            f >> accx >> accy >> accz >> fuel >> consumation >> range;

            std::vector<BodyWithAcc>* vec;
            switch(t) {
                case DeathStar:
                    vec = &deathstars;
                    break;
                case XWing:
                    vec = &xwings;
                    break;
            }

            // VS compiler is stupid and cannot compile push_back with initializer list directly so i have to populate in manually
            BodyWithAcc b;
            
            b.id = id;
            b.mass = mass;
            b.size = size;
            b.posx = px;
            b.posy = py;
            b.posz = pz;
            b.speedx = sx;
            b.speedy = sy;
            b.speedz = sz;
            b.accx = accx;
            b.accy = accy;
            b.accz = accz;
            b.fuel = fuel;
            b.cons = consumation;
            b.range = range;

            vec->push_back(b);
        }
        else
        {
            std::vector<Body>* vec;
            switch(t) {
                case Asteroid:
                    vec = &asteroids;
                    break;
                case Planet:
                    vec = &planets;
                    break;
            }

            vec->push_back({id,
                              mass,
                              size,
                              px, py, pz,
                sx, sy, sz});
        }
    }

    // Now transfer data to Universe struct in X D A P form
    gUniverse.Allocate(asteroids.size() + planets.size() + xwings.size() + deathstars.size(),
                       xwings.size() + deathstars.size());

    gUniverse.NumXWings = xwings.size();
    gUniverse.NumDeathStars = deathstars.size();
    gUniverse.NumAsteroirds = asteroids.size();
    gUniverse.NumPlanets = planets.size();

    int currentIndex = 0;

    auto transferLambda = [&currentIndex](Body* body) {
        gUniverse.Ids[currentIndex] = body->id;
        gUniverse.Masses[currentIndex] = body->mass;
        gUniverse.Sizes[currentIndex] = body->size;
        gUniverse.Positions.Xs[currentIndex] = body->posx;
        gUniverse.Positions.Ys[currentIndex] = body->posy;
        gUniverse.Positions.Zs[currentIndex] = body->posz;
        gUniverse.Speeds.Xs[currentIndex] = body->speedx;
        gUniverse.Speeds.Ys[currentIndex] = body->speedy;
        gUniverse.Speeds.Zs[currentIndex] = body->speedz;
    };

    auto transferExtraLambda = [&currentIndex](BodyWithAcc* b) {
        gUniverse.Accels.Xs[currentIndex] = b->accx;
        gUniverse.Accels.Ys[currentIndex] = b->accy;
        gUniverse.Accels.Zs[currentIndex] = b->accz;
        gUniverse.Fuels[currentIndex] = b->fuel;
        gUniverse.Consumtions[currentIndex] = b->cons;
        gUniverse.Ranges[currentIndex] = b->range;
    };

    for(auto& body : xwings) { transferLambda(&body); transferExtraLambda(&body); ++currentIndex;}
    for(auto& body : deathstars) { transferLambda(&body); transferExtraLambda(&body); ++currentIndex;}
    for(auto& body : asteroids) { transferLambda(&body); ++currentIndex;}
    for(auto& body : planets) { transferLambda(&body); ++currentIndex;}

    f.close();
}

void Universe_Run(float time, float delta, Result* result)
{
    result->collisions_count = 0;
    result->positions_count = 0;
    result->destructions_count = 0;
    auto numBodies = gUniverse.NumXWings
        + gUniverse.NumDeathStars
        + gUniverse.NumAsteroirds
        + gUniverse.NumPlanets;

    // These either need to be taken outside this function or should use temporary allocator
    // which is cleared every frame
    SOAVector accels;
    accels.Xs = new float[numBodies];
    accels.Ys = new float[numBodies];
    accels.Zs = new float[numBodies];

    std::vector<int> destroyedObjects;
    destroyedObjects.reserve(numBodies);

    for(float currentTime = 0.0f; currentTime < time; currentTime += delta) {
        // These are for tracking of object not destroyed
        int newNumXWings = gUniverse.NumXWings;
        int newNumDeathStars = gUniverse.NumDeathStars;
        int newNumPlanets = gUniverse.NumPlanets;
        int newNumAsteroids = gUniverse.NumAsteroirds;

        // G is given in meters but we operate on km directly
        const float G = 0.0000000000667 * (1.0 / 1000000);
        // Step 1: Calculate Acceleration from gravity pulling
        for(int i = 0; i < numBodies; ++i) {
            float totalForcex = 0;
            float totalForcey = 0;
            float totalForcez = 0;
            for(int j = 0; j < numBodies; ++j) {
                // TODO: Make this use SSE
                if(j != i) {
                    float rx = gUniverse.Positions.Xs[j] - gUniverse.Positions.Xs[i];
                    float ry = gUniverse.Positions.Ys[j] - gUniverse.Positions.Ys[i];
                    float rz = gUniverse.Positions.Zs[j] - gUniverse.Positions.Zs[i];

                    float lenSq = rx * rx + ry * ry + rz * rz;
                    float len = sqrt(lenSq);

                    float forceScalar = (G * gUniverse.Masses[j] * gUniverse.Masses[i]) / lenSq;

                    // Divice by len to normalize
                    totalForcex += forceScalar * (rx / len);
                    totalForcey += forceScalar * (ry / len);
                    totalForcez += forceScalar * (rz / len);
                }
            }

            accels.Xs[i] = totalForcex / gUniverse.Masses[i];
            accels.Ys[i] = totalForcey / gUniverse.Masses[i];
            accels.Zs[i] = totalForcez / gUniverse.Masses[i];
        }

        // Step 2: Apply acceleration from engine of the X-Wings and DeathStar
        for(int i = 0; i < gUniverse.NumXWings + gUniverse.NumDeathStars; ++i) {
            float fuelConsumed = gUniverse.Consumtions[i] * delta;
            float actualFuelConsumed = std::fmin(fuelConsumed, gUniverse.Fuels[i]);
            // actualFuelConsumed will be 1 most of the cases, except when there is not enough fuel
            // and this will hold number between 0.0 and 1.0 representing how many time we had fuel to burn
            actualFuelConsumed /= fuelConsumed;
        
            accels.Xs[i] += gUniverse.Accels.Xs[i] * actualFuelConsumed;
            accels.Ys[i] += gUniverse.Accels.Ys[i] * actualFuelConsumed;
            accels.Zs[i] += gUniverse.Accels.Zs[i] * actualFuelConsumed;

            gUniverse.Fuels[i] -= actualFuelConsumed;
        }

        // Step 3 : Move objects
        for(int i = 0; i < numBodies; ++i) {
            // Update position
            gUniverse.Positions.Xs[i] += gUniverse.Speeds.Xs[i] * delta + (accels.Xs[i] * delta * delta) / 2.0f;
            gUniverse.Positions.Ys[i] += gUniverse.Speeds.Ys[i] * delta + (accels.Ys[i] * delta * delta) / 2.0f;
            gUniverse.Positions.Zs[i] += gUniverse.Speeds.Zs[i] * delta + (accels.Zs[i] * delta * delta) / 2.0f;

            // Update speed
            gUniverse.Speeds.Xs[i] += accels.Xs[i] * delta;
            gUniverse.Speeds.Ys[i] += accels.Ys[i] * delta;
            gUniverse.Speeds.Zs[i] += accels.Zs[i] * delta;
        }

        // Step 4 : Check X-Wing ranges
        for(int i = 0; i < gUniverse.NumXWings; ++i) {
            float lenSq = gUniverse.Ranges[i] * gUniverse.Ranges[i];
            // death stars start at NumXWings index
            // asteroids start right after death stars
            for(int j = gUniverse.NumXWings; j < gUniverse.NumXWings + gUniverse.NumDeathStars + gUniverse.NumAsteroirds; ++j) {
                float x = gUniverse.Positions.Xs[i] - gUniverse.Positions.Xs[j];
                float y = gUniverse.Positions.Ys[i] - gUniverse.Positions.Ys[j];
                float z = gUniverse.Positions.Zs[i] - gUniverse.Positions.Zs[j];
                float sizeSq = gUniverse.Sizes[j] * gUniverse.Sizes[j];

                if((x * x + y * y + z * z) <= (lenSq - sizeSq)) {
                    // We have a hit !
                    // Torpedos are launched from the X-Wing and the object is destroyed !!!

                    destroyedObjects.push_back(j);
                    (j < gUniverse.NumXWings + gUniverse.NumDeathStars) ? --newNumDeathStars : --newNumAsteroids; // Check what have we destoyed


                    auto ind = result->destructions_count;
                    result->destructions_count++;
                    result->destructions[ind].time = currentTime;
                    result->destructions[ind].destructor = gUniverse.Ids[i];
                    result->destructions[ind].destructed = gUniverse.Ids[j];
                }
            }
        }

        // Step 5 : Check Death Stars ranges
        for(int i = gUniverse.NumXWings; i < gUniverse.NumXWings + gUniverse.NumDeathStars; ++i) {
            float lenSq = gUniverse.Ranges[i] * gUniverse.Ranges[i];
            // death stars start at NumXWings index
            // asteroids start right after death stars
            for(int j = gUniverse.NumXWings + gUniverse.NumDeathStars; j < numBodies; ++j) {
                float x = gUniverse.Positions.Xs[i] - gUniverse.Positions.Xs[j];
                float y = gUniverse.Positions.Ys[i] - gUniverse.Positions.Ys[j];
                float z = gUniverse.Positions.Zs[i] - gUniverse.Positions.Zs[j];
                float sizeSq = gUniverse.Sizes[j] * gUniverse.Sizes[j];

                if((x * x + y * y + z * z) <= (lenSq - sizeSq)) {
                    // We have a hit !
                    // Laser beam is prepared and fired upon the unlucky celestial body !!

                    destroyedObjects.push_back(j);
                    (j < gUniverse.NumXWings + gUniverse.NumDeathStars + gUniverse.NumAsteroirds) ? --newNumAsteroids : --newNumPlanets; // Check what have we destoyed

                    auto ind = result->destructions_count;
                    result->destructions_count++;
                    result->destructions[ind].time = currentTime;
                    result->destructions[ind].destructor = gUniverse.Ids[i];
                    result->destructions[ind].destructed = gUniverse.Ids[j];
                }
            }
        }

        // Step 6 : Check collisions
        for(int i = 0; i < numBodies; ++i) {
            for(int j = 0; j < numBodies; ++j) {
                if(i != j) {
                    float x = gUniverse.Positions.Xs[i] - gUniverse.Positions.Xs[j];
                    float y = gUniverse.Positions.Ys[i] - gUniverse.Positions.Ys[j];
                    float z = gUniverse.Positions.Zs[i] - gUniverse.Positions.Zs[j];
                    float sizeSquare = (gUniverse.Sizes[i] + gUniverse.Sizes[j]) * (gUniverse.Sizes[i] + gUniverse.Sizes[j]);

                    if((x * x + y * y + z * z) <= sizeSquare) {
                        // Oh these object collided with each other.

                        destroyedObjects.push_back(i);
                        destroyedObjects.push_back(j);
                        if(i < gUniverse.NumXWings) --newNumXWings;
                        else if(i < gUniverse.NumXWings + gUniverse.NumDeathStars) --newNumDeathStars;
                        else if(i < gUniverse.NumXWings + gUniverse.NumDeathStars + gUniverse.NumAsteroirds) --newNumAsteroids;
                        else --newNumPlanets;

                        if(j < gUniverse.NumXWings) --newNumXWings;
                        else if(j < gUniverse.NumXWings + gUniverse.NumDeathStars) --newNumDeathStars;
                        else if(j < gUniverse.NumXWings + gUniverse.NumDeathStars + gUniverse.NumAsteroirds) --newNumAsteroids;
                        else --newNumPlanets;

                        auto ind = result->collisions_count;
                        result->collisions_count++;
                        result->collisions[ind].time = currentTime;
                        result->collisions[ind].body1 = gUniverse.Ids[i];
                        result->collisions[ind].body2 = gUniverse.Ids[j];
                    }
                }
            }
        }

#define REMOVE_MACRO(arr, end) std::remove(arr, arr + end, arr[ind])
        // This is higly unefficient, but was a scheme that required least amount of code
        auto endEngine = gUniverse.NumXWings + gUniverse.NumDeathStars;
        for(auto ind : destroyedObjects) {
            REMOVE_MACRO(gUniverse.Ids, numBodies);
            REMOVE_MACRO(gUniverse.Masses, numBodies);
            REMOVE_MACRO(gUniverse.Sizes, numBodies);
            REMOVE_MACRO(gUniverse.Positions.Xs, numBodies);
            REMOVE_MACRO(gUniverse.Positions.Ys, numBodies);
            REMOVE_MACRO(gUniverse.Positions.Zs, numBodies);
            REMOVE_MACRO(gUniverse.Speeds.Xs, numBodies);
            REMOVE_MACRO(gUniverse.Speeds.Ys, numBodies);
            REMOVE_MACRO(gUniverse.Speeds.Zs, numBodies);
            if(ind < endEngine) {
                REMOVE_MACRO(gUniverse.Accels.Xs, endEngine);
                REMOVE_MACRO(gUniverse.Accels.Ys, endEngine);
                REMOVE_MACRO(gUniverse.Accels.Zs, endEngine);
                REMOVE_MACRO(gUniverse.Fuels, endEngine);
                REMOVE_MACRO(gUniverse.Consumtions, endEngine);
                REMOVE_MACRO(gUniverse.Ranges, endEngine);
            }
        }
#undef REMOVE_MACRO

        destroyedObjects.clear();
        numBodies = newNumAsteroids + newNumDeathStars + newNumPlanets + newNumXWings;
        gUniverse.NumAsteroirds = newNumAsteroids;
        gUniverse.NumDeathStars = newNumDeathStars;
        gUniverse.NumPlanets = newNumPlanets;
        gUniverse.NumXWings = newNumXWings;
    }

    result->positions_count = numBodies;
    for(int i = 0; i < numBodies; ++i) {
        result->positions[i].body = gUniverse.Ids[i];
        result->positions[i].position = Vector3D{ gUniverse.Positions.Xs[i], gUniverse.Positions.Ys[i], gUniverse.Positions.Zs[i] };
    }

    delete[] accels.Xs;
    delete[] accels.Ys;
    delete[] accels.Zs;
}