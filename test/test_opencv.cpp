#include "opencv2/opencv.hpp"

using namespace cv;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    cv::VideoWriter writer;
    if(!cap.isOpened())  // check if we succeeded
	return -1;
    while(1)
    {
	Mat frame;
	cap >> frame; // get a new frame from camera
	imshow("edges", frame);
	if(waitKey(30) >= 0) break;
    }
}
