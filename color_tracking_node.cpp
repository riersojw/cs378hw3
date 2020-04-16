#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ctime>
#include "std_msgs/Int16.h"
//#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>

//CHANGE
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>

#include <csignal>
#include <cstdio>
#include <math.h>


using namespace std;
using namespace cv;

//#define MEASURE_TIME 1
#define COLOR_TRACKING 1

#ifdef MEASURE_TIME
clock_t t_begin = 0;
#endif

static int counter = 0;
static int goal = 1000;
static double time_record[] = {};

int detected_count = 0;

void cv_process_img(const Mat& input_img, Mat& output_img)
{
    Mat gray_img;
    cvtColor(input_img, gray_img, CV_RGB2GRAY);
    
    double t1 = 20;
    double t2 = 50;
    int apertureSize = 3;
    
    Canny(gray_img, output_img, t1, t2, apertureSize);
}

void cv_publish_img(image_transport::Publisher &pub, Mat& pub_img)
{
    //sensor_msgs::ImagePtr pub_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", pub_img).toImageMsg();
    sensor_msgs::ImagePtr pub_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", pub_img).toImageMsg();
    pub.publish(pub_msg);
}

void cv_color_tracking(const Mat& input_img, ros::Publisher &controlPub)
{
    int iLowH = 160;
    int iHighH = 179;
    
    int iLowS = 150;
    int iHighS = 255;
    
    int iLowV = 60;
    int iHighV = 255;
    
    Mat imgLines = Mat::zeros(input_img.size(), CV_8UC3);    
    Mat imgHSV;
    //convert input image from RGB to HSV
    cvtColor(input_img, imgHSV, CV_RGB2HSV);

    Mat1b mask1;
    Mat1b mask2;
    
    //Uncommented to Track the Color Red
    cv::inRange(imgHSV, Scalar(105, 150, 50), Scalar(125, 255, 255), mask1); // red
    cv::Mat1b mask = mask1;

     //commented out the next three lines to stop tracking Blue
    //cv::inRange(imgHSV, cv::Scalar(0, 70, 50), cv::Scalar(10, 255, 255), mask1);    //Blue
    //cv::inRange(imgHSV, cv::Scalar(170, 70, 50), cv::Scalar(180, 255, 255), mask2); // Blue
    //cv::Mat1b mask = mask1 | mask2;

    erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    
    //morphological closing (removes small holes from the foreground)
    dilate( mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    
    //Find the contour, then indentify rectangle, center, radius using openCV build in functions
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
   

    double limit = 12000;
    int detected = false;
    std_msgs::Int16 msg;
    vector<Point> approx;
    
    //Vector to keep track of moment points
    vector<Moments> mu(contours.size());
    //Vector to keep track of Bounding Rectangle points
    vector<Rect> boundRect(contours.size());
    size_t num_contours = contours.size();
    /*two scalor functions to make it easier to include color. color = bounding rect
      color2 = Text color*/ 
    Scalar color = Scalar(0,255,0);
    Scalar color2 = Scalar(0,0,255);

    for (size_t i = 0; i < num_contours; i++) {
	double area = contourArea(contours[i]);
    //adding moment points to the vector based on the contour points
    mu[i] = moments( contours[i]);
    
	//cout << "Area:" << area << endl;
	if (area > limit) {
            cout << "Area: " << area << endl;
	    detected = true;
        //Calculate the contour perimeter
        double peri = arcLength(contours[i], true);
        // approximate a polygonal based on contour perimeter and contour point
        approxPolyDP(contours[i], approx, 0.02 * peri, true);
        //Grab the bounding Rectangle points and dump them to the Bounding Rect Vector
        boundRect[i] = boundingRect( approx );

	}
    }
    if (detected){
        detected_count += 1;
    }else{
        detected_count = 0;
    }
    //calculate the center point of the detected area
    float cX = (mu[0].m10 / (mu[0].m00 + 1e-5));
    float cY = (mu[0].m01 / (mu[0].m00 + 1e-5));
    
   
    
    if (detected_count >= 2) {
        cout << "Detected" << std::endl;
        for ( int i = 0; i < contours.size(); i++){
            //cout << "CX = " << cX << std::endl;
            //cout << "CY = " << cY << std::endl;
        //Draw the X in the center of the detected Area
        line(input_img, Point2f((cX - 10), (cY - 10)), Point2f((cX + 10), (cY + 10)), color, 2);
        line(input_img, Point2f((cX + 10), (cY - 10)), Point2f((cX - 10), (cY + 10)), color, 2);
        /* Determine the location of the object in relation to the center x value of the frame
           center value of our video feed is 640. If we are between the values of 600 and 680 we
           assume that we are in front of the object. If we are 600 or less we are left of the object.
           If we are 680 or greater than we are right of the object */
        
        if (cX <= 600){
            putText(input_img, "Left", Point(900,715), FONT_HERSHEY_DUPLEX, 1, color2, 2);
        } else if (cX >= 680) {
            putText(input_img, "Right", Point(900,715), FONT_HERSHEY_DUPLEX, 1, color2, 2);
        } else {
            putText(input_img, "Front", Point(900,715), FONT_HERSHEY_DUPLEX, 1, color2, 2);
        }
        rectangle( input_img, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
        }
        msg.data = 1;
        controlPub.publish(msg);
    }else{
        cout << "Haven't detected" << std::endl;
	msg.data = 0;
        controlPub.publish(msg);
    }


    
    
    
    cv::imshow("color_tracking_input_image", input_img);
    cv::imshow("RedCircle_Tracker", mask); 
    
    waitKey(1);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg, image_transport::Publisher &pub, ros::Publisher &controlPub)
{
    
    auto start = std::chrono::high_resolution_clock::now();
    cv_bridge::CvImageConstPtr cv_ori_img_ptr;
    try{
        Mat cv_ori_img = cv_bridge::toCvShare(msg, "bgr8")->image;
        Mat cv_output_img;
        
        cv_process_img(cv_ori_img, cv_output_img);
        
#ifdef COLOR_TRACKING
        cv_color_tracking(cv_ori_img, controlPub);
#endif
        
        cv_publish_img(pub, cv_output_img);
        
#ifdef MEASURE_TIME
        clock_t t_end = clock();
        double delta_time= double(t_end - t_begin) / CLOCKS_PER_SEC;
        //cout << "Delta_t = " << 1/delta_time << "\n";
        //t_begin = t_end;
#endif
        
        //imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
        //imshow("view", cv_output_img);
        waitKey(30);
        
#ifdef MEASURE_TIME
        t_begin = clock();
#endif
    }catch(cv_bridge::Exception& e){
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }

    // process time measurement
    /*auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    time_record[counter] = (double)elapsed.count();
    counter += 1;

    if(counter == goal){
        double sum = 0.0;
        for(int i = 0; i < goal; i ++){
            sum += time_record[i];
        }
        double mean = sum/goal;
        const char *path = "/home/nvidia/catkin_ws/time_test/color_tracking_recording.txt";
        ofstream file;
        file.open(path, ios::app);
        file << "The mean is: " << mean*1000 << "ms" << endl;
        time_t now = time(0);
        char* dt = ctime(&now);
        file << dt << '\n' << '\n';
        counter = 0;
    }*/
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener");
    
    ros::NodeHandle nh;
    
    //namedWindow("Vview");
    //startWindowThread();
    image_transport::ImageTransport it(nh);
    
    ros::NodeHandle nh_pub;
    image_transport::ImageTransport itpub(nh_pub);
    image_transport::Publisher pub = itpub.advertise("sample/cannyimg", 1);
    
    ros::NodeHandle node;
    uint32_t queue_size = 500;
    ros::Publisher controlPub = node.advertise<std_msgs::Int16>("imagecontrol", queue_size);
    
    
#ifdef MEASURE_TIME
    t_begin = clock();
#endif
    
    image_transport::Subscriber sub = it.subscribe("rgb/image_rect_color", 1, boost::bind(imageCallback, _1, pub, controlPub));
    //ros::Subscriber controlSub = it.subscribe("rbg/image_rect_color")
    ros::spin();
    

    
    //destroyWindow("view");
    
}

