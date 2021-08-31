#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

string PATH = "funk.jpg"; //Image Path
int AREA_FILTER = 1000;
Mat imgOrg, imgProc, imgWarp;
vector<Point> initialPoints, docPoints;
int w = 420, h = 596;

Mat preProcessing(Mat img)
{
    cvtColor(img, imgProc, COLOR_BGR2GRAY); // to gray scale
    GaussianBlur(imgProc, imgProc, Size(3,3), 3, 0); // blurring for better canny performance
    Canny(imgProc, imgProc, 25, 75); // edge detection
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgProc, imgProc, kernel);
    return imgProc;
}

vector<Point> getContours(Mat imgDil){
    //detects the biggest rectangle in image
    vector<vector<Point>> contours; //vectors example: {{Point(20,30),Point(50,60)},{},{}}
    vector<Vec4i> hierarchy;
    findContours(imgDil,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE); //finding contours
    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<Point> biggest;
    int maxArea=0;
    for (int i=0;i<contours.size();i++){
        int area = contourArea(contours[i]);
        string objectType;
        if(area>AREA_FILTER){ //filter small rectangles
            float peri = arcLength(contours[i],true);
            approxPolyDP(contours[i],conPoly[i],0.02*peri,true);
            if(area>maxArea && conPoly[i].size()==4){ //find biggest (4 for rectangle)
                maxArea = area;
                biggest = {conPoly[i][0],conPoly[i][1],conPoly[i][2],conPoly[i][3]};
            }
        }
    }
    return biggest;
}

void drawPoints(vector<Point> points, Scalar color){
    for(int i=0;i<points.size();i++)
    {
        circle(imgOrg,points[i], 5,color,FILLED);
        putText(imgOrg, to_string(i),points[i],FONT_HERSHEY_PLAIN,4,color,4);
    }
}

vector<Point> reorder(vector<Point> points ){
    vector<Point> newPoints;
    vector<int> sumPoints, subPoints;
    //get corners
    for(int i = 0;i<4;i++){
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x - points[i].y);
    }
    newPoints.push_back(points[min_element(sumPoints.begin(),sumPoints.end()) - sumPoints.begin()]);
    newPoints.push_back(points[max_element(subPoints.begin(),subPoints.end()) - subPoints.begin()]);
    newPoints.push_back(points[min_element(subPoints.begin(),subPoints.end()) - subPoints.begin()]);
    newPoints.push_back(points[max_element(sumPoints.begin(),sumPoints.end()) - sumPoints.begin()]);
    return newPoints;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h)
{
    Point2f src[4] = {points[0],points[1],points[2],points[3]};
    Point2f dst[4] = {{0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h}};
    Mat matrix = getPerspectiveTransform(src,dst);
    warpPerspective(img, imgWarp, matrix, Point(w, h));
    return imgWarp;
}

void main() {
     //sample
    imgOrg = imread(PATH);
    resize(imgOrg,imgOrg,Size(),0.5,0.5); // reduce the size of the photo in half
    //preprocessing
    imgProc = preProcessing(imgOrg);
    //get contours
    initialPoints = getContours(imgProc);
    //drawPoints(initialPoints,Scalar(0,0,255));
    docPoints = reorder(initialPoints);
    //drawPoints(docPoints,Scalar(0,255,0));
    //warp
    imgWarp = getWarp(imgOrg, docPoints, w, h);

    imshow("Image imgWarp",imgWarp);
    waitKey(0);

}
