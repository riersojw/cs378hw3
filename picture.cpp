#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

int main(int argc, char** argv) {

Mat img, grayMat, dst, canny_output;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;

img = imread("RedCircle.png");

cvtColor(img, grayMat, COLOR_BGR2GRAY);
threshold(grayMat, dst, 127, 255, 0);
Canny( grayMat, canny_output, thresh, thresh*2, 3);
findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );

vector<Moments> mu(contours.size());
for( size_t i = 0; i < contours.size(); i++)
{
	mu[i] = moments( contours[i] );
}

float cX = (mu[0].m10 / (mu[0].m00 + 1e-5));
float cY = (mu[0].m01 / (mu[0].m00 + 1e-5));

line(img, Point2f((cX - 10), (cY - 10)), Point2f((cX + 10), (cY + 10)), (0, 255, 0), 2);
line(img, Point2f((cX + 10), (cY - 10)), Point2f((cX - 10), (cY + 10)), (0, 255, 0), 2);

/// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<Point2f>center( contours.size() );
  vector<float>radius( contours.size() );

  for( int i = 0; i < contours.size(); i++ )
     { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
     }


  /// Draw polygonal contour + bonding rects + circles
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       rectangle( img, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
     }

 /// Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
     }

  /// Show in a window
 // namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  //imshow( "Contours", drawing );


String windowName = "Red Circle - Normal";
String gwindowName = "Red Circle - Gray";
String twindowName = "Red Circle - Thresh";

namedWindow(windowName, WINDOW_AUTOSIZE);
imshow(windowName, img);

//namedWindow(gwindowName, WINDOW_AUTOSIZE);
//imshow(gwindowName, grayMat);

//namedWindow(twindowName, WINDOW_AUTOSIZE);
//imshow(twindowName, dst);

waitKey(0);

destroyAllWindows();

return 0;
}
