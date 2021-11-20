#include <fstream>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <mutex>
#include <thread>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class Yolo_detect{
    private:
        float confThreshold = 0.5; // Confidence threshold
        float nmsThreshold = 0.4;  // Non-maximum suppression threshold 輸出邊界匡的改變
        int inpWidth = 416;  // Width of network's input image
        int inpHeight = 416;
        vector<string> classes;
        vector<String> getOutputsNames(const Net& net);
        std::mutex image_mutex;
        Mat image;
        bool has_image=false;

    public:
        Yolo_detect(float confTh, float nmsTh, int inpWid, int inpHeight,string coco_names, string weights, string cfg_file);
        // Remove the bounding boxes with low confidence using non-maxima suppression
        void postprocess(Mat& frame, const vector<Mat>& out);
        // Draw the predicted bounding box
        void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
        vector<String> getOutputsNames();
        Net net;

    
};
  