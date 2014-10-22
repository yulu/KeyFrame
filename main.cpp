#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <iostream>

#include "FrameTracker.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{ 
	//read video
	VideoCapture capture(0);
	FrameTracker frameTracker(10, 20, 40);
	
	Mat currentFrame; 
	bool shouldQuit;
	char* window="result";
	
	namedWindow(window, CV_WINDOW_AUTOSIZE);
	//detect keyframe
	do
	{
		capture >> currentFrame; 
		if(currentFrame.empty())
		{
			shouldQuit = true;
			continue;
		}
		
		bool f = frameTracker.focus(currentFrame);
		
		imshow(window, currentFrame);
		waitKey(27);
		
	}while(!shouldQuit);
	
}

