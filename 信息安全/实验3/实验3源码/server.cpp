//
//  server.cpp
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
#include <stdlib.h>
#include <unistd.h>
#include "des.h"

#define SERVER_PORT 8888
// DH协议中的参数（要求两个大素数）
long long g = 7;
long long n = 11;

int main(){
    int serverSocket;
    struct sockaddr_in server_addr;
    struct sockaddr_in clientAddr;
    int addr_len = sizeof(clientAddr);
    int client;
    char buffer[2000];
    int iDataNum;
    
    if((serverSocket = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("socket");
        return 1;
    }
    
    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if(bind(serverSocket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        perror("connect");
        return 1;
    }
    
    if(listen(serverSocket, 5) < 0)
    {
        perror("listen");
        return 1;
    }

    while(1)
    {
        printf("Listening on port: %d\n", SERVER_PORT);
        client = accept(serverSocket, (struct sockaddr*)&clientAddr, (socklen_t*)&addr_len);
        if(client < 0)
        {
            perror("accept");
            continue;
        }
        printf("[Server]:recv client data...\n");
        printf("[Server]:IP is %s\n", inet_ntoa(clientAddr.sin_addr));
        printf("[Server]:Port is %d\n", htons(clientAddr.sin_port));
        
        //  选取一个大的随机数b (1<b<n)，将b保密，计算
        // Y = g^b mod n，将X发送给对方
        // 接收X
        long long b;
        srand((int)time(0));
        b = 1 + rand() % (n-1);
        printf("[Server]:choose a random num for DH: %lld\n", b);
        
        iDataNum = (int)recv(client, buffer, 2000, 0);
        buffer[iDataNum] = '\0';
        long long X = atoi(buffer);
        printf("DataNum=%d, X=%lld\n", iDataNum, X);
        
        int key[64];

        // 把缓冲区的换行符读掉
//        getchar();
        long long Y = (long long)pow(g, b) % n;
        // 发送Y
        std::string strY = std::to_string(Y);
        send(client, strY.c_str(), strY.length(), 0);
        // 计算Ka=X^a mod n
        long long Kb = (long long)pow(X, b) % n;
//        printf("kb=%lld X=%lld b=%lld", Kb, X, b);
        intToBinArr(Kb, key);
        calcSubKeys(key);
        while(1)
        {

            iDataNum = (int)recv(client, buffer, 1024, 0);
            if(iDataNum < 0)
            {
                perror("recv");
                continue;
            }
            buffer[iDataNum] = '\0';
            printf("[Server]:%d recv encrypted data:< %s\n", iDataNum, buffer);
            // 将接受到的信息解密
            std::string recvCipher = buffer;
            std::string decryption = decode(recvCipher);
            if(strcmp(decryption.c_str(), "quit") == 0){
                printf("[Server]:%d exit!\n", iDataNum);
                break;
            }
            printf("[Server]: decryption data: %s\n", decryption.c_str());
            printf("[Server]:Input data:>");
            char c;
            int len=0;
            while(scanf("%c", &c)){
                if(c == '\n') break;
                buffer[len++] = c;
            }
            // 加密
            std::string sendCipher = encode(buffer, len);
            send(client, sendCipher.c_str(), sendCipher.length(), 0);
            if(strcmp(buffer, "quit") == 0){
                printf("[Server]:%d exit!\n", iDataNum);
                break;
            }
        }
        close(client);
    }
    close(serverSocket);
    return 0;
}
