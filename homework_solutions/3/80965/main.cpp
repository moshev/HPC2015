#include<vector>
#include<immintrin.h>
using std::vector;
#define G 6.674*(1e-11)

struct Vector3D
{
    float x, y, z;
    Vector3D& operator +=(const Vector3D& a)
    {
        x+=a.x;
        y+=a.y;
        z+=a.z;
        return *this;
    }
    Vector3D& operator -=(const Vector3D& a)
    {
        x-=a.x;
        y-=a.y;
        z-=a.z;
        return *this;
    }
    Vector3D& operator +(const Vector3D& a)
    {
        x+=a.x;
        y+=a.y;
        z+=a.z;
        return *this;
    }
    Vector3D& operator / (float num)
    {
        x/=num;
        y/=num;
        z/=num;
        return *this;
    }
     Vector3D& operator * (float num)
    {
        x*=num;
        y*=num;
        z*=num;
        return *this;
    }
     bool operator <(const Vector3D& a)
    {
        return x<a.x & y<a.y & z<a.z;
    }
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
/*Вселената е описана в следния формат:

N
<id> <Type> <Mass> <Size> <Position> <Speed> [<Acceleration> <Fuel> <Consumption> <Range>]
...
<id> <Type> <Mass> <Size> <Position> <Speed> [<Acceleration> <Fuel> <Consumption> <Range>]
*/
int n;
struct UniverseBodys
{
    vector<int> id;
    vector<int> Type; //0=Planet 1=Asteroid 2=Death Star 3=X-Wing
    vector<float> Mass;
    vector<float> Size;
    vector<BodyPosition> Position;
    vector<Vector3D> Speed;
    vector<Vector3D> Acceleration;
    vector<float> Fuel;
    vector<float> Consumption;
    vector<float> Range;
    UniverseBodys()
    {
        id.reserve(n);
        Type.reserve(n);
        Mass.reserve(n);
        Size.reserve(n);
        Position.reserve(n);
        Speed.reserve(n);
        Acceleration.reserve(n);
        Fuel.reserve(n);
        Consumption.reserve(n);
        Range.reserve(n);
    }
    void DeleteEl(int i)
    {
        std::swap(id[i],id[n]);
        id.pop_back();
        std::swap(Type[i],Type[n]);
        Type.pop_back();
        std::swap(Mass[i],Mass[n]);
        Mass.pop_back();
        std::swap(Size[i],Size[n]);
        Size.pop_back();
        std::swap(Position[i],Position[n]);
        Position.pop_back();
        std::swap(Speed[i],Speed[n]);
        Speed.pop_back();
        std::swap(Acceleration[i],Acceleration[n]);
        Acceleration.pop_back();
        std::swap(Fuel[i],Fuel[n]);
        Fuel.pop_back();
        std::swap(Consumption[i],Consumption[n]);
        Consumption.pop_back();
        std::swap(Range[i],Range[n]);
        Range.pop_back();
        n--;
    }

};
UniverseBodys Universe;
void Universe_Initialize(const char* file)
{


}

float Scalar_Product(Vector3D a, Vector3D b)
{
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

void Universe_Run(float time, float delta, Result* result)
{
    vector< vector<Vector3D> > F_ij;
    vector< vector<Vector3D> > r_ij;
    vector< vector<float> > nhelp_ij;
    vector< vector<float> > scalar;
    vector< vector<Vector3D> > n_ij;
    vector<Vector3D> F_i;
    vector<Vector3D> accel_i;
    vector<Vector3D> P_i;

    for(int t=0;t<time;t+=delta)
    {
            for(int i=0;i<n;++i)
            {
                for(int j=0;j<n;++j)
                {
                    r_ij[i][j].x=Universe.Position[j].position.x - Universe.Position[i].position.x;
                    r_ij[i][j].y=Universe.Position[j].position.y - Universe.Position[i].position.y;
                    r_ij[i][j].z=Universe.Position[j].position.z - Universe.Position[i].position.z;
                }
            }
             for(int i=0;i<n;++i)
            {
                for(int j=0;j<n;++j)
                {
                     nhelp_ij[i][j]=r_ij[i][j].x*r_ij[i][j].x+r_ij[i][j].y*r_ij[i][j].y+r_ij[i][j].z*r_ij[i][j].z;
                     n_ij[i][j].x=r_ij[i][j].x/nhelp_ij[i][j];
                }
            }

             for(int i=0;i<n;++i)
            {
                for(int j=0;j<n;++j)
                {
                    float k=(G*Universe.Mass[i]*Universe.Mass[j])/Scalar_Product(r_ij[i][j],r_ij[i][j]);
                    F_ij[i][j].x=k*n_ij[i][j].x;
                    F_ij[i][j].y=k*n_ij[i][j].y;
                    F_ij[i][j].z=k*n_ij[i][j].z;
                }
            }
            for(int i=0;i<n;++i)
            {
                for(int j=0;i<n;++j)
                  {F_i[i]+=F_ij[i][j];}
                  F_i[i]-=F_ij[i][i];
            }

             for(int i=0;i<n;++i)
            {
                    accel_i[i]=F_i[i]/Universe.Mass[i];

            }
            for(int i=0;i<n;++i)
            {
                if(Universe.Type[i]>1 && Universe.Fuel[i]>0 )
                    accel_i[i]+=Universe.Acceleration[i];

            }

             for(int i=0;i<n;++i)
            {
                    P_i[i]= Universe.Speed[i]*delta + (accel_i[i]*(delta*delta))/2;

            }

             for(int i=0;i<n;++i)
            {
                for(int j=0;j<n;++j)
                {
                    float d=(Universe.Size[i]*Universe.Size[j])/2;
                    if(i!=j && Universe.Type[i]<2 && Universe.Type[j]<2 && Universe.Position[i].position<(Universe.Position[j].position*d))
                    {
                        int r=result->collisions_count;
                        result->collisions[r].time=t;
                        result->collisions[r].body1=Universe.id[i];
                        result->collisions[r].body2=Universe.id[j];
                        result->collisions_count++;
                        Universe.DeleteEl(i);
                        Universe.DeleteEl(j);
                    }
                    if(i!=j & Universe.Type[i]<2 & Universe.Type[j]==2 & Universe.Position[i].position<(Universe.Position[j].position*d*Universe.Range[j]))
                    {
                        int r=result->destructions_count;
                        result->destructions[r].time=t;
                        result->destructions[r].destructor=Universe.id[j];
                        result->destructions[r].destructed=Universe.id[i];
                        result->destructions_count++;
                        Universe.DeleteEl(i);
                    }
                    if(i!=j & (Universe.Type[i]==1 || Universe.Type[i]==2) & Universe.Type[j]==3 & Universe.Position[i].position<(Universe.Position[j].position*d*Universe.Range[j]))
                    {
                        int r=result->destructions_count;
                        result->destructions[r].time=t;
                        result->destructions[r].destructor=Universe.id[j];
                        result->destructions[r].destructed=Universe.id[i];
                        result->destructions_count++;
                        Universe.DeleteEl(i);
                    }
                }
                for(int i=0;i<n;i++)
                {
                    if (Universe.Type[i]>2)
                    Universe.Fuel[i]-=Universe.Consumption[i]*delta;
                }

            }

        for(int i=0;i<n;++i)
        result->positions[result->positions_count]=Universe.Position[i];

        }

}

