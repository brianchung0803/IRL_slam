#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <mutex>
#include <thread>

#include "IRL_network.h"
#include "Yolo_detect.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <fstream>
#include <ctime>
#include "line_lbd/include/line_lbd_allclass.h"




#define SERVER_ADDRESS "192.168.50.6" //伺服器端IP地址    
#define PORT           9490      //伺服器的埠號    
#define MSGSIZE        65536        //收發緩衝區的大小    

using namespace std;
using namespace cv;
std::mutex image_mutex;
Mat image;
bool has_image = false;


void ReceiveImage(int SocketFD)
{
    int ConnectFD;
    char buffer[65536];
    while(1)
    {
        image_mutex.lock();
        ConnectFD = accept(SocketFD, NULL, NULL);
        // cout << "client connected" << endl;
        
        if(0 > ConnectFD)
        {
            perror("error accept failed");
            close(SocketFD);
            exit(EXIT_FAILURE);
        }

        std::vector<uchar> decode;
        decode.clear();
        
        // image_mutex.lock();
        while(true)    
        {   
            bzero(buffer, 65536);
            int ret = recv(ConnectFD, buffer, sizeof(buffer), 0); // file_discriptor, void* buf, size_t len(buffer大小)
            // cout << "receive client chuck: "<< ret << "bytes"<<endl;
            if (ret <= 0)    
            {    
                break;    
            }    
            
            decode.insert(decode.end(),buffer,buffer+ret);
            
        }
        if(decode.size()>0)
        {
            //
            cout << "changing image..."<<endl;
            image = imdecode(decode, IMREAD_COLOR);//圖像解碼
            // has_image=true;
            // socket_.has_image = true;
            has_image = true;
            
            //      
        }
        // image_mutex.unlock();
        close(ConnectFD);
        image_mutex.unlock();
    }
}

int main(int argc, char **argv)
{
    cout << "testing ..." <<endl;
    IRL_network test(SERVER_ADDRESS,PORT);
    cout << "ok ..." <<endl;

    Yolo_detect yolo(0.5,0.4,416,416,"/Users/Brian/Documents/brian_SLAM/IRL_slam/data/coco.names","/Users/Brian/Documents/brian_SLAM/IRL_slam/data/yolov3.weights","/Users/Brian/Documents/brian_SLAM/IRL_slam/data/yolov3.cfg");

    Mat blob;
    
    // ReceiveImage(test,image);
    thread *ImageSocket= new thread(ReceiveImage,test.SocketFD);
    // thread ImageSocket(test.ReceiveImage);
    while(true)
    {
        image_mutex.lock();
        if(has_image)
        {
            cout << "new image..."<<endl;
            has_image = false;
            
            
            blobFromImage(image, blob, 1/255.0, cv::Size(416, 416), Scalar(0,0,0), true, false);
            yolo.net.setInput(blob);
        
            // Runs the forward pass to get output of the output layers
            vector<Mat> outs;
            yolo.net.forward(outs, yolo.getOutputsNames());
            
            // Remove the bounding boxes with low confidence
            yolo.postprocess(image, outs);
            
            
            imshow("image", image);
            waitKey(300);
            
        }
        image_mutex.unlock();
        
    }
    return 0;
}