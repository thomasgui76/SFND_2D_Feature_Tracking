#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType,bool bVis=false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);

// declaration of keypionts detector type
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,bool bVis=false);
void detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,bool bVis=false);
void detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);

// declaration of descriptor type
void descKeypointsOtherType(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors, std::string descriptorType);
void descKeypointsBrisk(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors);
void descKeypointsBrief(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors);
void descKeypointsOrb(cv::Ptr<cv::DescriptorExtractor>& extractor, std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors);
void descKeypointsFreak(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors);
void descKeypointsAkaze(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img, cv::Mat &descriptors);
void descKeypointsSift(cv::Ptr<cv::DescriptorExtractor>& extractor,std::vector<cv::KeyPoint> &keypoints, 
cv::Mat &img,cv::Mat &descriptors);

class spreadSheet
{
    private:
        std::vector<std::string> fieldVector;
        std::vector<unsigned int> rowVector;
        std::vector<std::vector<float>> vlaue;

    public:
        spreadSheet(std::vector<unsigned int> rowVector,std::vector<std::string> fieldVector)
        {
            this->rowVector = rowVector;
            this->fieldVector = fieldVector;
            int row = rowVector.size();
            int col = fieldVector.size();
            for (int i=0;i<row;i++){
                std::vector<float> rowValue(col,0.0);
                this->vlaue.push_back(rowValue);
            }
        }
        ~spreadSheet(){}
        void setValue(int row,std::string field, float t)
        {
            std::vector<std::string>::iterator iter;
            iter = std::find(this->fieldVector.begin(),this->fieldVector.end(),field);
            int col = iter - this->fieldVector.begin();
            this->vlaue[row][col] = t;
        }
        
        float getValue(int row, std::string field)
        {
            std::vector<std::string>::iterator iter;
            iter = std::find(this->fieldVector.begin(),this->fieldVector.end(),field);
            int col = iter - this->fieldVector.begin();
            return this->vlaue[row][col];
        }

};

#endif /* matching2D_hpp */
