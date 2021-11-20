#include "../include/IRL_network.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <mutex>
#include <thread>

using namespace std;

IRL_network::IRL_network(char* host_ip, int port)
{
    cout <<"\n creating server socket, ip: "<<host_ip<<", port: "<<port<<endl;
    struct sockaddr_in stSockAddr;

    SocketFD = socket(AF_INET , SOCK_STREAM , 0); // IPv4, TCP連線 
    if (SocketFD== -1){
        cout<<"Fail to create a socket."<<endl;
    }else{
        cout << "socket created, waiting for clients."<<endl;
    }

    memset(&stSockAddr, 0, sizeof(struct sockaddr_in));
 
    stSockAddr.sin_family = AF_INET;
    stSockAddr.sin_port = htons(port);
    stSockAddr.sin_addr.s_addr = inet_addr(host_ip);

    if(-1 == ::bind(SocketFD,(const struct sockaddr *)&stSockAddr, sizeof(struct sockaddr_in)))
    {
        perror("error bind failed");
        close(SocketFD);
        exit(EXIT_FAILURE);
    }
 
    if(-1 == listen(SocketFD, 10))
    {
        perror("error listen failed");
        close(SocketFD);
        exit(EXIT_FAILURE);
    }
}

IRL_network::~IRL_network()
{
    close(SocketFD);
}


bool IRL_network::HasImage(){
    return has_image;
}

void IRL_network::CancelImage(){
    has_image = false;
}