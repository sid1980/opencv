
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

int main(int argc, char *argv[])
{
	bool useTestData=false;
	
	
	cv::Mat model_bgr = cv::imread("C:/Users/MAC/Desktop/test2.jpg", CV_LOAD_IMAGE_COLOR);
	cv::VideoCapture capture(0);

	if( !capture.isOpened() ){
		return -1;
	}

	std::vector<cv::KeyPoint> model_keypoints, in_keypoints;
	cv::SurfFeatureDetector surf(2500);
	cv::SurfDescriptorExtractor surfDesc;

	cv::BruteForceMatcher<cv::L2<float>> matcher;
	std::vector<cv::DMatch> matches;

	cv::Mat model_descriptor, in_descriptor;

	surf.detect( model_bgr, model_keypoints );
	surfDesc.compute( model_bgr, model_keypoints, model_descriptor );

	cv::Mat in_bgr;

	for(;;){
		capture >> in_bgr;

		if (in_bgr.empty()){
			break;
		}

		in_keypoints.clear();
		matches.clear();
		in_descriptor.release();

		//Find Surf keypoints and descriptors of the new camera image
		surf.detect(in_bgr,in_keypoints);
		surfDesc.compute(in_bgr,in_keypoints,in_descriptor);

		//Find matches
		matcher.match(model_descriptor,in_descriptor,matches);


		std::nth_element(matches.begin(),
			matches.begin() + 24,
			matches.end());

		matches.erase(matches.begin() + 25,matches.end());

		//convert keypoints in cv::point2f
		std::vector<cv::Point2f> model_points,in_points;
		for(std::size_t i= 0; i<matches.size();i++)
		{
			float model_x = model_keypoints[matches[i].queryIdx].pt.x;
			float model_y = model_keypoints[matches[i].queryIdx].pt.y;
			model_points.push_back(cv::Point2f(model_x,model_y));

			float in_x = in_keypoints[matches[i].trainIdx].pt.x;
			float in_y = in_keypoints[matches[i].trainIdx].pt.y;
			in_points.push_back(cv::Point2f(in_x,in_y));
		}

		cv::Mat homography_model2cam = cv::findHomography(model_points,in_points,CV_RANSAC);


		
		//transform four corners of the template to camera coordinates
		std:: vector<cv::Point> corner_model,corner_in;
		corner_model.push_back(cv::Point(0,0));
		corner_model.push_back(cv::Point(model_bgr.cols-1,0));
		corner_model.push_back(cv::Point(model_bgr.cols-1,model_bgr.rows-1));
		corner_model.push_back(cv::Point(0,model_bgr.rows-1));

	
		for (std::size_t i= 0 ; i< corner_model.size();i++)
		{
		/////empty ////////// you must coding by your self
			cv::Mat point_model(3,1,CV_64F);
			point_model.at<double>(0,0) = corner_model[i].x;
			point_model.at<double>(1,0) = corner_model[i].y;
			point_model.at<double>(2,0) = 1;

			cv::Mat point_in = homography_model2cam * point_model;
			point_in.at<double>(0,0) /= point_in.at<double>(2,0);
			point_in.at<double>(1,0) /= point_in.at<double>(2,0);


			corner_in.push_back(cv::Point(point_in.at<double>(0,0), point_in.at<double>(1,0)));


		}

			//Draw four lines representing
		cv:: Mat out_bgr= in_bgr.clone();
		
		cv::line(out_bgr,corner_in[0],corner_in[1],CV_RGB(255,0,0),2);
		cv::line(out_bgr,corner_in[1],corner_in[2],CV_RGB(0,255,0),2);
		cv::line(out_bgr,corner_in[2],corner_in[3],CV_RGB(0,0,255),2);
		cv::line(out_bgr,corner_in[3],corner_in[0],CV_RGB(0,0,0),2);
		

		cv::Mat match_img;
		cv::drawMatches(
			model_bgr,model_keypoints,
			in_bgr,in_keypoints,
			matches,
			match_img,
			cv::Scalar::all(-1));

		cv::imshow("SURF Feature Descriptor and Matching",match_img);
		cv::imshow("Homography",out_bgr);







		

		if ( cv::waitKey(1) == 27 ) {
			break;
		}
	}

	return 0;
}
