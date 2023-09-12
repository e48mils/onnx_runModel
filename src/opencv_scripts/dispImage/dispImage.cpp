#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include "dispImage.h"

using namespace cv;
using namespace std;

void dispImage(){
    namedWindow("PhotoFrame");
    
    cv::Mat image; 
    
    image = imread("../../data/img/Elephant.png");

    if (image.empty()) {
        cout << "Image File " << "Not Found" << endl;
  
        // wait for any key press
        cin.get();
        return;
    }

    imshow("PhotoFrame", image);
    // Wait for any keystroke
    waitKey(0);
    destroyWindow("PhotoFrame");
    
};