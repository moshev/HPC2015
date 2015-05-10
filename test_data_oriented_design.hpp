//
//  data_oriented_design.h
//  GPAPI
//
//  Created by savage309 on 9.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef GPAPI_data_oriented_design_h
#define GPAPI_data_oriented_design_h

#include "diffclock.h"

namespace DataOrientedDesign {
struct MobBool {
    MobBool() {
        canJump = randomInt(0, 1);
        canSwim = randomInt(0, 1);
        canFly = randomInt(0, 1);
        canRun = randomInt(0, 1);
        canBite = randomInt(0, 1);
        canShoot = randomInt(0, 1);
    }
    bool canJump;
    bool canSwim;
    float life;
    float damage;
    float size[3];
    bool canFly;
    int numLegs;
    bool canRun;
    bool canBite;
    bool canShoot;
};

struct MobFlags {
    enum {
        CAN_JUMP = 1 << 0,
        CAN_SWIM = 1 << 1,
        CAN_FLY = 1 << 2,
        CAN_RUN = 1 << 3,
        CAN_BITE = 1 << 4,
        CAN_SHOOT = 1 << 5,
        LAST_FLAG = 1 << 6,
    };
    MobFlags() { flags = randomInt(0, LAST_FLAG) ;}
    float life;
    float damage;
    int numLegs;
    float size[3];
    int flags;
    bool canJump() const { return flags & CAN_JUMP; }
    bool canSwim() const { return flags & CAN_SWIM; }
    bool canFly() const { return flags & CAN_FLY; }
    bool canRun() const { return flags & CAN_RUN; }
    bool canBite() const { return flags & CAN_BITE; }
    bool canShoot() const { return flags & CAN_SHOOT; }

};

    bool isSuperMob(const MobBool& mob) {
        return mob.canSwim && mob.canShoot && mob.canRun && mob.canJump && mob.canFly && mob.canBite;
    }

    bool isSuperMob(const MobFlags& mob) {
         return mob.canSwim() && mob.canShoot() && mob.canRun() && mob.canJump() && mob.canFly() && mob.canBite();
    }
    
    size_t getTestSize() {
        return 50000000;
    }
    
    void testMobBool() {
        const auto TEST_SIZE = getTestSize();
        std::unique_ptr<MobBool[]> mob(new MobBool[TEST_SIZE]);
        bool areSuperMobs = true;
        for (auto i = 0; i < TEST_SIZE; ++i){
            areSuperMobs &= isSuperMob(mob[i]);
        }
    }
    
    void testMobFlags() {
        const auto TEST_SIZE = getTestSize();
        std::unique_ptr<MobFlags[]> mob(new MobFlags[TEST_SIZE]);
        bool areSuperMobs = true;
        for (auto i = 0; i < TEST_SIZE; ++i){
            areSuperMobs &= isSuperMob(mob[i]);
        }
    }
    
    void test() {
        std::cout << "Testing data oriented design ..." << std::endl;
        auto t2 = getTime();
        testMobBool();
        auto t3 = getTime();
        
        std::cout << '\t' << "Mob bool " << diffclock(t3, t2) << std::endl;
        auto t0 = getTime();
        testMobFlags();
        auto t1 = getTime();
        std::cout << '\t' << "Mob flags " << diffclock(t1, t0) << std::endl;
        
        std::cout << "\n **** \n\n";
    }
    
}//namespace DataOrientedDesign

#endif
