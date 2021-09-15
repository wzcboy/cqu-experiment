//
//  client.cpp
//  DES
//
//  Created by wzc on 2021/5/21.
//  Copyright © 2021 wzc. All rights reserved.
//

#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "des.h"
#define SERVER_PORT 5555

// DH协议中的参数（要求两个大素数）
long long g = 7;
long long n = 11;
int main(){
    int clientSocket;
    struct sockaddr_in serverAddr;
    char sendbuf[2000];
    char recvbuf[2000];
    int iDataNum;
    if((clientSocket = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("socket");
        return 1;
    }
    
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(SERVER_PORT);
    serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if(connect(clientSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0)
    {
        perror("connect");
        return 1;
    }
    
    printf("[Client]:connect with destination host...\n");
    
    //  选取一个大的随机数a (1<a<n)，将a保密，计算
    // X = g^a mod n，将X发送给对方
    int key[64];
    long long a;
    srand((int)time(0) % 100);
    a = 1 + rand() % (n-1);
    printf("[Client]:choose a random num for DH: %lld\n", a);
    // 把缓冲区的换行符读掉
//    getchar();
    long long X = (long long)pow(g, a) % n;
    // 发送X
    std::string strX = std::to_string(X);
    send(clientSocket, strX.c_str(), strX.length(), 0);
    // 接收Y
    iDataNum = (int)recv(clientSocket, recvbuf, 2000, 0);
    recvbuf[iDataNum] = '\0';
    long long Y = atoi(recvbuf);
    printf("DataNum=%d, Y=%lld\n", iDataNum, Y);
    // 计算Ka=Y^a mod n
    long long Ka = (long long)pow(Y, a) % n;
//    printf("ka=%lld Y=%lld a=%lld", Ka, Y, a);
    intToBinArr(Ka, key);
    calcSubKeys(key);
    
    while(1)
    {
        printf("[Client]:Input data:>");
        char c;
        int len=0;
        while(scanf("%c", &c)){
            if(c == '\n') break;
            sendbuf[len++] = c;
        }
        
        // 加密
        std::string sendCipher = encode(sendbuf, len);
        send(clientSocket, sendCipher.c_str(), sendCipher.length(), 0);
        if(strcmp(sendbuf, "quit") == 0){
            printf("[Client]:exit!\n");
            break;
        }
        // 接受到信息，并解密
        iDataNum = (int)recv(clientSocket, recvbuf, 200, 0);
        recvbuf[iDataNum] = '\0';
        printf("[Client]:%d recv encrypted data:< %s\n", iDataNum, recvbuf);

        std::string recvCipher = recvbuf;
        std::string decryption = decode(recvCipher);
        if(strcmp(decryption.c_str(), "quit") == 0){
            printf("[Client]:exit!\n");
            break;
        }
        printf("[Client]:decryption data: %s\n", decryption.c_str());
    }
    close(clientSocket);
    return 0;
}
