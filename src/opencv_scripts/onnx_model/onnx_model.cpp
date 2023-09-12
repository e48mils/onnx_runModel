#include "onnx_model.h"
#include "../../data_structures/vector_indexed_probabilities/vector_indexed_probabilities.h"
#include "../../data_structures/vector_indexed_probabilities/struct_indexed_probability/struct_indexed_probability.h"
#include <iostream>
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <time.h>

using namespace std;
using namespace cv;

onnx_model::onnx_model(int n){
    modelNumber = n;

    string modelPath = "../../models/resnet50/model.onnx";
    try {
        ResnetModel = cv::dnn::readNetFromONNX(modelPath);
    } catch (cv::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return; // Handle the error appropriately
    }
};

std::vector<std::string> onnx_model::getLabels(std::string path_to_labels){
    std::vector<std::string> labels;
    std::ifstream file(path_to_labels);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            // Remove leading and trailing whitespace from the line
            line.erase(line.find_last_not_of(" \t") + 1);
            line.erase(0, line.find_first_not_of(" \t"));

            // Add the cleaned label to the vector
            labels.push_back(line);
        }
        file.close();
    } else {
        std::cerr << "Error: Unable to open file " << path_to_labels << std::endl;
    }
    return labels;
};

cv::Mat onnx_model::loadImage(std::string path){
    cv::Mat image = cv::imread(path);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Unable to load the image." << std::endl;
        return image;
    } else {
        return image;
    };
};

cv::Mat onnx_model::getInputBlob(cv::Mat image){
    cv::Mat blob;
    cv::resize(image, blob, cv::Size(224, 224));
    blob.convertTo(blob, CV_32F, 1.0, 0);

    //cv::cvtColor(blob, blob, cv::COLOR_RGB2BGR);
    
    cv::Scalar mean_values(103.939, 116.779, 123.68);
    cv::subtract(blob, mean_values, blob);

    blob = blob.reshape(1, {1, 224, 224, 3});

    return blob;
};

int onnx_model::runModelOnCamera(){
    // Image frames
    cv::Mat myImage; 

    // Declaring a window
    cv::namedWindow("Video Classifier");

    // Capture frames from default camera
    VideoCapture cap(0);

    // Track time elapsed for computing FPS
    time_t startTime, curTime;

    time(&startTime);
    int numFramesCaptured = 0;
    double secElapsed;
    double curFPS;
    double averageFPS = 0.0;
    
    //This section prompt an error message if no video stream is found
    if (!cap.isOpened()){ 
        cout << "No video stream detected" << endl;
        system("pause");
        return -1;
    }

    std::vector<std::string> labels = getLabels("../../models/resnet50/labels/Resnet_labels.txt");

    //Taking an everlasting loop to show the video
    while (true){ 
        cap >> myImage;

        numFramesCaptured++;
        time(&curTime);
        secElapsed = difftime(curTime, startTime);
        curFPS = numFramesCaptured / secElapsed;

        if (secElapsed > 0){
            averageFPS = (averageFPS * (numFramesCaptured - 1) + curFPS) / numFramesCaptured;
        }

        cv::putText(myImage, to_string(averageFPS),cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);

        //runModel(myImage, labels);

        //Breaking the loop if no video frame is detected
        if (myImage.empty()){ 
            break;
        }
        //Showing the video
        imshow("Video Classifier", myImage);
    
        //Allowing 25 milliseconds frame processing time and initiating break condition
        char c = (char) waitKey(25);

        //If 'Esc' is entered break the loop
        if (c == 27){ 
            break;
        }
        
   }
   //Releasing memory
   cap.release();
   return 0;
};



int onnx_model::runModel(cv::Mat &image, std::vector<std::string> &labels){
    // Load test image
    //cv::Mat image = loadImage(inputPath);

    // Get image in right shape
    cv::Mat blob = getInputBlob(image);

    //cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(224,224));

    // Input and Output layer names
    std::string inputLayerName = "input_2";
    std::string outputLayerName = "predictions";

    // Set the input blob (assuming you have an OpenCV Mat image)
    ResnetModel.setInput(blob, inputLayerName);

    cv::Mat outputBlob;

    try {
        // Forward pass
        outputBlob = ResnetModel.forward(outputLayerName);
        // Process the output
    } catch (cv::Exception& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
        return -1;
    }

    // Create result as user-defined STRUCT
    vector_indexed_probabilities result(outputBlob, labels);

    // Print first n rows in the sorted struct
    if (result.data[0].probability > 0.5){
        result.printFirstN(1);
    }

    return 0;
};

int onnx_model::printModelNumber(){
    cout << to_string(modelNumber) << endl;
    return 1;
};


