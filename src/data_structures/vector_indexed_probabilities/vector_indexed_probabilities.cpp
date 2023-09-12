#include "struct_indexed_probability/struct_indexed_probability.h"
#include "vector_indexed_probabilities.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;


vector_indexed_probabilities::vector_indexed_probabilities(cv::Mat probabilities, std::vector<std::string> labels){
    for (int i = 0; i < probabilities.cols; i++){
        float probability = probabilities.at<float>(0, i);
        data.emplace_back(i, probability, labels[i]);
    }
    sortArray();
};

void vector_indexed_probabilities::sortArray(){
    std::sort(data.begin(), data.end());
};

void vector_indexed_probabilities::printFirstN(int n){
    for(int i = 0; i < n; i++){
        if (i < data.size()){
            std::cout << data[i].label << ", probability: " << data[i].probability << std::endl;
        } else {
            std::cout << "Not enough classes!" << std::endl;
        };
    }
}