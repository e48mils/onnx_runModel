#ifndef STRUCT_INDEXED_PROBABILITY_H
#define STRUCT_INDEXED_PROBABILITY_H

#include<string>

using namespace std;

struct struct_indexed_probability{

    int index;
    float probability;
    std::string label;

    struct_indexed_probability(int indexInit, float probabilityInit, std::string labelInit);

    bool operator<(const struct_indexed_probability& other);
};

#endif // STRUCT_INDEXED_PROBABILITY_H