//
//  diffclock.h
//  GPAPI
//
//  Created by savage309 on 6.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef GPAPI_diffclock_h
#define GPAPI_diffclock_h


#include <ctime>

double diffclock(clock_t clock1,clock_t clock2){
    double diffticks = clock1 - clock2;
    return (diffticks) / CLOCKS_PER_SEC;
}

#endif
