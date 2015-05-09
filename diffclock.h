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
#include <chrono>
#include "common.h"

inline std::chrono::time_point<std::chrono::system_clock> getTime() {
    return std::chrono::system_clock::now();
}

inline double diffclock(std::chrono::time_point<std::chrono::system_clock> end, std::chrono::time_point<std::chrono::system_clock> start){
      std::chrono::duration<double> elapsed_seconds = end-start;
    return elapsed_seconds.count();
}

#endif
