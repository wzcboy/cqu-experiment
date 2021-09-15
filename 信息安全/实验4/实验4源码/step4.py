#!/usr/bin/env python
# coding: utf-8

import json
from socket import *

def R2toPC2(pc2_server_ip,pc2_server_port):
    #创建服务器套接字
    server_socket = socket(AF_INET,SOCK_STREAM)
    server_socket.bind((pc2_server_ip, 8080))
    server_socket.listen(128)
    #接收连接
    client_socket, client_addr = server_socket.accept()
    #接受数据
    str_ipPack = client_socket.recv(1024).decode("utf-8")
    #处理数据
    dic = json.loads(str_ipPack)
    client_socket.send("-------成功接收到网络包-------".encode("utf-8"))
    server_socket.close()
    client_socket.close()
    return dic

def main():
    pc2_server_ip = "192.168.45.53"
    dic = R2toPC2(pc2_server_ip, 8080)
    print("成功接收数据")
    print(dic)
    
if __name__ == "__main__":
    main()
