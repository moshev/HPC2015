//
//  cache_miss.h
//  GPAPI
//
//  Created by savage309 on 6.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef GPAPI_cache_miss_h
#define GPAPI_cache_miss_h

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <list>

#include "diffclock.h"

namespace CacheMiss {
template <typename T>
class List {
public:
    List():first(nullptr), last(nullptr){}
    
    bool empty() const {
        return first == nullptr;
    }
    
    bool add(T data) {
        Node* newNode = new Node(data);
        if (first == nullptr) {
            last = first = newNode;
        } else {
            last->next = newNode;
            newNode->prev = last;
            newNode->next = nullptr;
            last = newNode;
        }
        return true;
    }
    
    bool remove(T data) {
        Node* temp = first;
        while (temp && temp->data != data)
            temp = temp->next;
        
        if (!temp) return false;
        if (temp == first) {
            Node* newFirst = first->next;
            delete first;
            if (newFirst) newFirst->prev = nullptr;
            first = newFirst;
        } else if (temp == last) {
            Node* newLast = last->prev;
            delete last;
            if (newLast) newLast->next = nullptr;
            last = newLast;
        } else {
            Node* left = temp->prev;
            Node* right = temp->next;
            delete temp;
            left->next = right;
            right->prev = left;
        }
        return true;
    }
private:
    struct Node {
        Node(T dataRhs):data(dataRhs), prev(nullptr), next(nullptr){}
        T data;
        Node* prev;
        Node* next;
    };
    
    Node* first;
    Node* last;
};

template <typename T>
class Vector {
public:
    Vector():ptr(nullptr), index(0), maxSize(0){}
    
    bool empty() const {
        return index == 0;
    }
    
    bool reset(int newSize) {
        delete[] ptr;
        ptr = new T[newSize];
        maxSize = newSize;
        return true;
    }
    
    bool add(T data) {
        if (index == maxSize) return false;
        ptr[index++] = data;
        return true;
    }
    
    bool remove(T data) {
        int remIndex = -1;
        for (int i = 0; i < maxSize; ++i) {
            if (ptr[i] == data)  {
                remIndex = i;
                break;
            };
        }
        if (remIndex == -1) return false;
        
        memmove(&ptr[remIndex], &ptr[remIndex + 1], maxSize - remIndex);
        --index;
        return true;
    }
private:
    T* ptr;
    int index;
    int maxSize;
};

    
inline size_t getTestSize() {
    return 81000;
}

template <typename T>
inline void eraseContainer(T& container) {
    for (int i = 0; i < getTestSize(); ++i)
        container.remove(i);
}

template <typename T>
inline void eraseContainer(std::vector<T>& vec) {
    for (int i = 0; i < getTestSize(); ++i) {
        auto iter = std::find(vec.begin(), vec.end(), i);
        vec.erase(iter);
    }
}

double stdVecTime = 0.0;
double stdListTime = 0.0;
double customVecTime = 0.0;
double customListTime = 0.0;

inline void cacheMissTest() {
    std::vector<int> vecStd;
    vecStd.reserve(getTestSize());
    for (int i = 0; i < getTestSize(); ++i)
        vecStd.push_back(i);
    
    std::random_shuffle(vecStd.begin(), vecStd.end());
    
    using namespace std;
    std::list<int> listStd;
    for (int i = 0; i < getTestSize(); ++i)
        listStd.push_back(vecStd[i]);
    
    Vector<int> vecCustom;
    vecCustom.reset((int)getTestSize());
    for (int i = 0; i < getTestSize(); ++i)
        vecCustom.add(vecStd[i]);
    
    List<int> listCustom;
    for (int i = 0; i < getTestSize(); ++i)
        listCustom.add(vecStd[i]);
    
    auto beginStdVec = getTime();
    eraseContainer(vecStd);
    auto endStdVec = getTime();
    stdVecTime+= double(diffclock(endStdVec,beginStdVec));
    if(!vecStd.empty()) printf("Bug in std vector test\n");
    
    auto beginStdList = getTime();
    eraseContainer(listStd);
    auto endStdList = getTime();
    stdListTime += double(diffclock(endStdList,beginStdList));
    if(!listStd.empty()) printf("Bug in std list test\n");
    
    auto beginVec = getTime();
    eraseContainer(vecCustom);
    auto endVec = getTime();
    customVecTime += double(diffclock(endVec,beginVec)) ;
    if(!vecCustom.empty()) printf("Bug in custom vector test\n");
    
    auto beginList = getTime();
    eraseContainer(listCustom);
    auto endList = getTime();
    customListTime += double(diffclock(endList,beginList));
    if(!listCustom.empty()) printf("Bug in custom list test\n");
}

inline void test() {
    std::cout << "Testing cache misses ..." << std::endl;
    const double testTimes = 1.0;
    srand((unsigned)time(0));
    for (int i = 0; i < testTimes; ++i)
        cacheMissTest();
    
    std::cout << '\t' << "Std vec time:" << stdVecTime / testTimes << std::endl;
    std::cout << '\t' << "Std list time:" << stdListTime / testTimes << std::endl;
    std::cout << '\t' << "Custom vec time:" << customVecTime / testTimes << std::endl;
    std::cout << '\t' << "Custom list time:" << customListTime / testTimes<< std::endl;
    
    std::cout << "\n **** \n\n";
    
}
} //namespace CacheMiss

#endif
