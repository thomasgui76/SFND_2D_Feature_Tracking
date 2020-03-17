/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;


/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results
    
    vector<unsigned int> rowVector = {0,1,2,3,4,5,6,7,8,9};
    vector<std::string> keypointsdetectorType ={"SHITOMASI","HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};  // SHITOMASI,HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    vector<std::string> descriptorsSelectorType ={"BRISK","BRIEF", "ORB","FREAK","AKAZE", "SIFT"};               // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    spreadSheet performance_detector =spreadSheet(rowVector,keypointsdetectorType);
    spreadSheet performance_decscriptor =spreadSheet(rowVector,descriptorsSelectorType);

    for(size_t i = 0; i < keypointsdetectorType.size(); i++)
    {
        for (size_t j = 0; j < descriptorsSelectorType.size(); j++)
        {
            vector<int> numofKeyPoints;   // for Task 7, record number of keypoints of each image.
            vector<int> numofMatchedKeyPoints;    // for Task8, record number of matched keypoints.

            std::string detectorType = keypointsdetectorType[i];
            std::string descriptorType = descriptorsSelectorType[j];

            /* MAIN LOOP OVER ALL IMAGES */
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */
                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                // push image into data frame buffer
                /*
                DataFrame frame;
                frame.cameraImg = imgGray;
                dataBuffer.push_back(frame);
                */
                DataFrame frame;
                frame.cameraImg = imgGray;
                if (dataBuffer.size()>dataBufferSize-1)
                    dataBuffer.clear();
                dataBuffer.push_back(frame);
        
                //// EOF STUDENT ASSIGNMENT
                cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image
                // string detectorType = "SHITOMASI";
                // string detectorType ="SIFT";
                // string detectorType = keypointsdetectorType[i];

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
                double t = (double)cv::getTickCount();
                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, false);
                }
                else
                {
                    detKeypointsModern(keypoints,imgGray,detectorType,false);
                }
                t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
                performance_detector.setValue(imgIndex,detectorType,t*1000/1.0);
                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                // only keep keypoints on the preceding vehicle
                // std::cout<<"before cropping, num of keypoints: "<<keypoints.size()<<std::endl;
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);
                if (bFocusOnVehicle)
                {
                    // ...
                    float x_min = vehicleRect.x;
                    float x_max = vehicleRect.x + vehicleRect.width;
                    float y_min = vehicleRect.y;
                    float y_max = vehicleRect.y + vehicleRect.height;
                    for(size_t i=0; i<keypoints.size(); i++)
                    {
                        if((x_min <= keypoints[i].pt.x)&& (keypoints[i].pt.x <= x_max) && (y_min <= keypoints[i].pt.y) && (keypoints[i].pt.y <= y_max)) 
                        {
                            keypoints.erase(keypoints.begin()+i);
                        }
                    }
                }
                numofKeyPoints.push_back(keypoints.size());     // Task 7
                //// EOF STUDENT ASSIGNMENT

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;
                cout << "#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;
                // string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
                // string descriptorType = "SIFT";
                string descriptorType = descriptorsSelectorType[j];
                t = (double)cv::getTickCount();
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
                t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
                performance_decscriptor.setValue(imgIndex,descriptorType,t*1000/1.0);
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {
                    // std::cout<<"entering match"<<std::endl;
                    // std::cout<<"size of dataBuffer: "<<dataBuffer.size()<<std::endl;
                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    // string matcherType = "MAT_FLANN";
                    // string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG,  choose normalization in matchDescriptors
                    // string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN
                    string selectorType = "SEL_KNN";

                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                    matches, descriptorType, matcherType, selectorType);

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    numofMatchedKeyPoints.push_back(matches.size());   // Task 8
                    cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    // bVis = true;
                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                }

            } // eof loop over all images
        }
    }

    // count performance
    std::vector<float> detectorVector(keypointsdetectorType.size(),0.0);
    std::vector<float> descriptorVector(descriptorsSelectorType.size(),0.0);
    
    for (int i=0; i<rowVector.size();i++)
    {
        for (int k = 0; k< keypointsdetectorType.size(); k++)
        {
            float temp_detector_value = performance_detector.getValue(rowVector[i],keypointsdetectorType[k]);
            detectorVector[k] += temp_detector_value;
            cout<<detectorVector[k]<<endl;
        }
    }
    
    std::cout<<"Top 3 keypoint detectors"<<std::endl;
    for (int i=0; i<3; i++)
    {
        float min_value = 1e5;
        int min_idx = -1;
        for(int j=0;j<detectorVector.size();j++)
        {
            if (min_value>detectorVector[j])
            {
                min_value = detectorVector[j];
                min_idx = j;
            }
        }
        detectorVector[min_idx]=1e5;                  // mask the maximum value, then pick the next.
        std::cout<<"detector "<<keypointsdetectorType[min_idx]<<" consume "<<min_value/10.0<<" ms"<<std::endl;
    }
    
    for (int i=0; i<rowVector.size();i++)
    {
        for (int k = 0; k< descriptorsSelectorType.size(); k++)
        {
            float temp_detector_value = performance_decscriptor.getValue(rowVector[i],descriptorsSelectorType[k]);
            descriptorVector[k] += temp_detector_value;
        }
    }
    std::cout<<"Top 3 feature descriptors"<<std::endl;
    for (int i=0; i<3; i++)
    {
        float min_value = 1e5;
        int min_idx = -1;
        for(int j=0;j<descriptorVector.size();j++)
        {
            if (min_value>descriptorVector[j])
            {
                min_value = descriptorVector[j];
                min_idx = j;
            }
        }
        descriptorVector[min_idx]=1e5;                  // mask the maximum value, then pick the next.
        std::cout<<"descriptor "<<descriptorsSelectorType[min_idx]<<" consume "<<min_value/10.0<<" ms"<<std::endl;
    }

    
    return 0;
}
