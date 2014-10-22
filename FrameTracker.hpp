#ifndef FRAME_TRACKER_HPP
#define FRAME_TRACKER_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

using namespace cv;
using namespace std;

class FrameTracker {
	
	public:
			FrameTracker(
			int spaceThreshold,
			int lowThres,
			int highThres,
			Ptr<FeatureDetector> detector = FeatureDetector::create("ORB"),
			Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("ORB"),
			Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming")
			);
			
			bool focus(Mat& frame);
			void getGray(const Mat& frame, Mat& gray_frame);
	
	private:
			vector<vector<DMatch> >		m_matches;
			
			Ptr<FeatureDetector> 		m_detector;
			Ptr<DescriptorExtractor>	m_extractor;
			Ptr<DescriptorMatcher>		m_matcher;
			
			int spaceT;
			int lowT;
			int highT;
			bool init;
			bool good;
};

FrameTracker::FrameTracker  (
			int spaceThreshold,
			int lowThres,
			int highThres,
			Ptr<FeatureDetector> detector, 
			Ptr<DescriptorExtractor> extractor,
			Ptr<DescriptorMatcher> matcher
							) : spaceT(spaceThreshold), lowT(lowThres), highT(highThres), m_detector(detector), m_extractor(extractor), m_matcher(matcher)
			{
				init = true;
			}

/*
 * feed in the frame and extract keypoint and descriptor
 * check focus
 */
bool FrameTracker::focus(Mat& image)
{	
	vector<KeyPoint> keypoint;
	Mat				 desc;
	Mat				 frame;
	
	getGray(image, frame);
	
	//if first frame
	if(init) {
		init = false;
				
		m_detector->detect(frame, keypoint);
		if(keypoint.empty())
			return false;

		m_extractor->compute(frame, keypoint, desc);
		if(keypoint.empty())
			return false;
			
		m_matcher->clear();
		
		std::vector<cv::Mat> descriptors(1);
		descriptors[0] = desc.clone();
		m_matcher->add(descriptors);

		m_matcher->train();
			
		good = false;
		
	//not first frame
	} else {
	
		m_detector->detect(frame, keypoint);
		if(keypoint.empty())
			return false;

		m_extractor->compute(frame, keypoint, desc);
		if(keypoint.empty())
			return false;	
		
		m_matcher->radiusMatch(desc, m_matches, (float)spaceT);
		
		//check results
		if (m_matches.size() > highT)
			good = true;
		if (m_matches.size() < lowT)
			good = false;	
		
		//update matcher
		m_matcher->clear();
		
		std::vector<cv::Mat> descriptors(1);
		descriptors[0] = desc.clone();
		m_matcher->add(descriptors);
		
		m_matcher->train();	
		
		cout<<"valid desc: "<<m_matches.size()<<endl;
	}
	
	return good;
}

void FrameTracker::getGray(const cv::Mat& image, cv::Mat& gray)
{
	if(image.channels() == 3)
	{
		cvtColor(image, gray, CV_RGB2GRAY);
	}
	else if (image.channels() == 4)
	{
		cvtColor(image, gray, CV_RGBA2GRAY);
	}
	else if(image.channels() == 1)
	{
		gray = image;
	}
}
	
	
#endif
