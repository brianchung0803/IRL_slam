#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <mutex>
#include <thread>


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/core.hpp>


using namespace std;
using namespace cv;

class IRL_network
{
private:
    

public:
    IRL_network(char* host_address, int port);
    ~IRL_network();
    int SocketFD;
    // void ReceiveImage();
    bool HasImage();
    void CancelImage();
    bool SendPose();
    bool SendObjects();
    Mat image;
    bool has_image = false;

};

