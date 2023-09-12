#include "printMessages/printMessages.h"
#include "opencv_scripts/dispImage/dispImage.h"
#include "opencv_scripts/onnx_model/onnx_model.h"

#include <iostream>

using namespace std;
using namespace cv;

int main(){
    onnx_model model1(1);
    onnx_model model2(2);

    printMessage("Hej hopp. Nu är vi igång.");



    model1.printModelNumber();
    model2.printModelNumber();
    model1.printModelNumber();

    //model1.runModel("../../data/img/elephant.jpg");

    model1.runModelOnCamera();

    //dispImage();

    return 0;
}