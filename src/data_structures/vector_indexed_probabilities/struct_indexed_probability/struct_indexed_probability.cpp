#include "struct_indexed_probability.h"
#include <string>

using namespace std;

struct_indexed_probability::struct_indexed_probability(int indexInit, float probabilityInit, std::string labelInit){
    index = indexInit;
    probability = probabilityInit;
    label = labelInit;
};

bool struct_indexed_probability::operator<(const struct_indexed_probability& other){
    return (other.probability < probability);
};