#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>

class onnx_model{
    private: 
        int modelNumber;
        cv::dnn::Net ResnetModel; // model
        cv::Mat inputBlob; // model input
        cv::Mat loadImage(std::string path);
        cv::Mat getInputBlob(cv::Mat image);
        std::vector<std::string> getLabels(std::string path_to_labels);



    public:
        onnx_model(int n);
        int printModelNumber();
        int runModel(cv::Mat &Image, std::vector<std::string> &labels);
        int runModelOnCamera();
};