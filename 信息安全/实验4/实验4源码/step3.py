#!/usr/bin/env python
# coding: utf-8

import json
from socket import *

def decodePackage(dic):
    src_arr = dic["src"]
    dst_arr = dic["dst"]
    proto_arr = dic["proto"]
    dic["src"] = src_arr[0:-1]
    dic["dst"] = dst_arr[0:-1]
    dic["proto"] = proto_arr[0:-1]
    return dic

def R1toR2(pc2_server_ip,pc2_server_port):
    #创建服务器套接字
    server_socket = socket(AF_INET,SOCK_STREAM)
    server_socket.bind(("192.168.45.55", 8080))
    server_socket.listen(128)
    #接收连接
    client_socket, client_addr = server_socket.accept()
    #接受数据
    str_ipPack = client_socket.recv(1024).decode("utf-8")
    #处理数据
    ipPack = json.loads(str_ipPack)
    print(ipPack)
    print(type(ipPack))
    dic = decodePackage(ipPack)
    str_dic = json.dumps(dic)
    #发送数据到PC2
    msg = R2ToPC2(pc2_server_ip, pc2_server_port, str_dic)

    server_socket.close()
    client_socket.close()
    return msg

def R2ToPC2(server_ip, server_port, data):
    #建立连接
    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.connect((server_ip,server_port))
    #发送数据
    client_socket.send(data.encode("utf-8"))
    print("——————成功发送数据至PC-2:",server_ip,"——————")
    #接收返回数据
    recv_data = client_socket.recv(1024)
    msg = recv_data.decode("utf-8")
    client_socket.close()
    return msg

def main():
    pc2_server_ip = "192.168.45.53"
    msg = R1toR2(pc2_server_ip, 8080)
    print(msg)
    
if __name__ == "__main__":
    main()
