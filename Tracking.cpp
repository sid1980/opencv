
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <stdio.h>
#include <iostream>
#include "contentfinder2.h"
using namespace cv;
using namespace std;


Rect tack(Mat image_copy)
{

	cv::Mat in_hsv(image_copy.size(),image_copy.type());
		cv::cvtColor(image_copy,in_hsv,CV_BGR2HSV);

		cv::cvtColor(image_copy,in_hsv,CV_BGR2HSV);
		
		std::vector<cv::Mat> hsvChannels;

		cv::split(in_hsv,hsvChannels);

		//cv::Mat kern = (cv::Mat_<float>(4,4) << 0.272, 0.534, 0.131, 0,0.349, 0.686, 0.168, 0,0.393, 0.769, 0.189, 0,0, 0, 0, 1); 
		//cv::transform(image_copy, frame2, kern); 


        //imshow("MyVideo", frame); //show the frame in "MyVideo" window
		//imshow("MyVideo2", in_hsv);
		//imshow("hsvChannels0", hsvChannels[0]);
		//imshow("hsvChannels1", hsvChannels[1]);
		//imshow("hsvChannels2", hsvChannels[2]);

		cv::Mat mask1(image_copy.size(),CV_8UC1);
		cv::Mat minMat(image_copy.size(),CV_8UC1);
		cv::Mat maxMat(image_copy.size(),CV_8UC1);
		minMat = 180/2;
		maxMat = 220/2;
		cv::inRange(hsvChannels[0],minMat,maxMat,mask1);


		cv::Mat mask2(image_copy.size(),CV_8UC1);
		cv::threshold(hsvChannels[1],mask2,100,255,CV_THRESH_BINARY);
		cv::Mat mask3( image_copy.size(), CV_8UC1 );
		mask3 = mask1 & mask2;
		//mask1 &= mask2;

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierachy;
		cv::findContours(mask2,contours,hierachy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));


		cv::Mat result = image_copy.clone();
		std::vector<std::vector<cv::Point>> contours_poly(contours.size());
		std::vector<cv::Rect> boundRect(contours.size());
		int test=0;
		for(std::size_t i = 0;i<contours.size();i++)
		{
			cv::approxPolyDP(cv::Mat(contours[i]),contours_poly[i],3,true);
			boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
			cv::rectangle(image_copy,boundRect[i],CV_RGB(0,255,0),3);

			if(boundRect[i].height==26){
			test = i;
			printf("%d     height       //////",boundRect[i].height);
			printf("%d     width       //////",boundRect[i].width);
			}
			
		}

		int maxarea = 0 , maxindex = -1;
		for(std::size_t i = 0;i<boundRect.size();i++)
		{
			if((boundRect[i].width * boundRect[i].height)>maxarea)
			{
				maxarea = boundRect[i].width * boundRect[i].height;
				maxindex = i;
			}

			

		}
		boundRect[maxindex].height = boundRect[maxindex].height;
		boundRect[maxindex].width = boundRect[maxindex].width;
		//cv::rectangle(image_copy,boundRect[maxindex],CV_RGB(255,0,0),3);
		cv::Mat combine ( image_copy.rows, (image_copy.cols * 4), CV_8UC3 );
		cv::Rect roi = cv::Rect(0,0,image_copy.cols,image_copy.rows);
		cv::Mat crop = combine(roi);
		cv::cvtColor( hsvChannels[0], crop, CV_GRAY2BGR );
		roi.x += image_copy.cols;
		crop = combine(roi);
		cv::cvtColor( mask1, crop, CV_GRAY2BGR );
		roi.x += image_copy.cols;
		crop = combine(roi);
		cv::cvtColor( mask3, crop, CV_GRAY2BGR );
		roi.x += image_copy.cols;
		crop = combine(roi);
		result.copyTo( crop, cv::noArray() );
		cv::imshow( "getBiggestColorContourBox", image_copy);
		//printf("%d            //////",boundRect[maxindex]);
		return boundRect[maxindex];
}





int main(int argc, char* argv[])
{
    VideoCapture capture("C:/Users/MAC/Desktop/NIDA/437199242.755174.mp4"); // open the video file for reading

    if ( !capture.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the video file" << endl;
         return -1;
    }

	ContentFinder finder;
	cv::Mat frame;
	cv::Mat modelimg_bgr;
	cv::Mat inimg_bgr;
	cv::Mat inimg_backproj_grey;
	cv::Mat inimg_backproj_bgr;
	cv::Mat outimg_bgr;
	cv::Rect roi_init ;
	cv::Rect roi;
	Mat frame2;
	bool bSuccess2 = capture.read(frame2);
	cv::Mat image_copy = frame2.clone();
	roi_init = tack(image_copy);

   
	std::vector<cv::DMatch> mmm;
	mmm.clear();
	cv::BruteForceMatcher<cv::L2<float>> tt;
	//tt.match();
	
	

	for(;;)
	{
		capture >> frame;
		
		if(frame.empty())
		{
			//capture.set(CV_CAP_PROP_POS_FRAMES,0);
			roi = roi_init;
			continue;
		}
		//printf("%d            //////",roi);

		if(inimg_bgr.empty())
		{
			inimg_bgr.create(frame.cols,frame.rows,frame.type());
			//cv::transpose(frame,inimg_bgr);
			frame.copyTo(inimg_bgr,cv::noArray());
			//cv::flip(inimg_bgr,inimg_bgr,1);
			//cv::imshow("Model 1" , inimg_bgr);
			//cv::imshow("Model 2" , frame);

			modelimg_bgr = (inimg_bgr(roi_init)).clone();
			finder.setModel(modelimg_bgr,CONTENTFINDER_MODE_BGR3D);
			//finder.setModel(inimg_bgr(roi_init),CONTENTFINDER_MODE_BGR3D);
			roi = roi_init;
		}
		else{
			//cv::transpose(frame,inimg_bgr);
			frame.copyTo(inimg_bgr,cv::noArray());
			//cv::flip(inimg_bgr,inimg_bgr,1);
		}

		roi = finder.finderModel(inimg_bgr,roi ,&inimg_backproj_grey);

		if(inimg_backproj_bgr.empty())
			inimg_backproj_bgr.create(inimg_backproj_grey.size(),CV_8UC3);
		cv::applyColorMap(inimg_backproj_grey,inimg_backproj_bgr,cv::COLORMAP_JET);

		if(outimg_bgr.empty())
			outimg_bgr.create(inimg_bgr.size(),inimg_bgr.type());
		inimg_bgr.copyTo(outimg_bgr,cv::noArray());
		cv::rectangle(outimg_bgr,roi,CV_RGB(0,255,0),3,8,0);

		cv::imshow("Model Image" , modelimg_bgr);
		cv::imshow("MeanShif Tracking" , outimg_bgr);
		//cv::imshow("Model 3" , inimg_backproj_grey);
		//cv::imshow("Model 4" , inimg_backproj_bgr);


		
      if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
       {
                cout << "esc key is pressed by user" << endl; 
                break; 
       }
	}

	
    //cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms

    double fps = capture.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video

     cout << "Frame per seconds : " << fps << endl;

    //namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	// namedWindow("MyVideo2",CV_WINDOW_AUTOSIZE);




    while(1)
    {

		

        Mat frame;

        bool bSuccess = capture.read(frame); // read a new frame from video

		 Mat frame2;

		 if(frame.empty())
		 {
		  capture.set(CV_CAP_PROP_POS_FRAMES,0);
		  continue;
	     }



        bool bSuccess2 = capture.read(frame2);
     
		cv::Mat image_copy = frame2.clone();

		
		//tack(image_copy);

		//cv::Rect roi_init = cv::Rect(image_copy,boundRect[maxindex],CV_RGB(255,0,0),3);


		//imshow("mask1", tack(image_copy));

		//cv::Rect roi_init = cv::Rect(140,265,180,80);
		Rect roi =  tack(image_copy);
		//printf("%f",roi);
		//Rect roi_init222 = getBiggestColorContourBox(image_copy,90.0,110.0);




      if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
       {
                cout << "esc key is pressed by user" << endl; 
                break; 
       }

	  


    }
	
	
	

	 
    return 0;
	
}

