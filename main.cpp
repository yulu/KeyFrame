#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <iostream>
#include <fstream>

#include "FrameTracker.hpp"

using namespace cv;   
using namespace std; 

int main(int argc, char** argv)
{ 
		
	Mat currentFrame; 
	bool shouldQuit;
	char* window="result";
	ofstream myfile;
	myfile.open ("./result.csv");
	int low = 10;
	int high = 20;
	
	//read video
	VideoCapture capture("./test1.mp4");
	FrameTracker frameTracker(20, low, high);

	
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
		
		int b = 0;
		if(f)
			b = 100;
			
		myfile<<low<<"\t"<<high<<"\t"<<frameTracker.numFeature<<"\t"<<b<<endl;

		
		imshow(window, currentFrame);
		waitKey(27);
		
	}while(!shouldQuit);
	
	myfile.close();
	
}

