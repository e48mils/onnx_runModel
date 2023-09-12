#include "struct_indexed_probability/struct_indexed_probability.h"
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

struct vector_indexed_probabilities{
    vector<struct_indexed_probability> data;

    vector_indexed_probabilities(cv::Mat probabilities, std::vector<std::string> labels);

    void sortArray();

    void printFirstN(int n);
};