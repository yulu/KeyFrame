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
			ORB detector = ORB(100),
			Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("ORB"),
			Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming")
			);
			
			bool focus(Mat& frame);
			void getGray(const Mat& frame, Mat& gray_frame);
			
			int numFeature;
	
	private:
			void checkDistance(vector<KeyPoint>& keypoint);
	
			vector<vector<DMatch> >		m_matches;
			
			ORB					 		m_detector;
			Ptr<DescriptorExtractor>	m_extractor;
			Ptr<DescriptorMatcher>		m_matcher;
			vector<KeyPoint>			m_keypoints;
			
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
			ORB detector, 
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
	
	numFeature = 0;
	
	getGray(image, frame);
	
	//if first frame
	if(init) {
		init = false;
				
		//m_detector.detect(frame, keypoint);
		m_detector(frame, Mat(), keypoint, desc);
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
		
		m_keypoints = keypoint;
			
		good = false;
		
	//not first frame
	} else {
	
		//m_detector.detect(frame, keypoint);
		m_detector(frame, Mat(), keypoint, desc);
		if(keypoint.empty())
			return false;

		m_extractor->compute(frame, keypoint, desc);
		if(keypoint.empty())
			return false;	
		
		m_matcher->radiusMatch(desc, m_matches, (float)spaceT);
		
		//numFeature = m_matches.size();
		
		checkDistance(keypoint);
		
		//check results
		if ( numFeature > highT)
			good = true;
		if ( numFeature < lowT)
			good = false;	
			

		
		//update matcher
		m_matcher->clear();
		
		std::vector<cv::Mat> descriptors(1);
		descriptors[0] = desc.clone();
		m_matcher->add(descriptors);
		
		m_matcher->train();	
		
		m_keypoints = keypoint;
		
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

void FrameTracker::checkDistance(vector<KeyPoint>& keypoint){
	int size = m_matches.size();
	numFeature = 0;
	for(int i = 0; i < size; i++) {
		vector<DMatch> bestMatch = m_matches[i];
		if(bestMatch.size() > 0) {	
			int qid = bestMatch[0].queryIdx;
			int tid = bestMatch[0].trainIdx;
			
			int x = keypoint[qid].pt.x - m_keypoints[tid].pt.x;
			int y = keypoint[qid].pt.y - m_keypoints[tid].pt.y;
			
			if(x * x + y * y < 12*12)
				numFeature++;
		}
	}
}
	
	
#endif
