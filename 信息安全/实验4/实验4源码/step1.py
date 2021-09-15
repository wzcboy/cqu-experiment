#!/usr/bin/env python
# coding: utf-8

import json
from socket import *

def createIpPackage(src_ip, dst_ip):
    dic    = {
        "version": 4,
        "ihl"    : None,
        "tos"    : 0x0,
        "length" : None,
        "id"     : 1,
        "flags"  : 0x0,
        "ttl"    : 64,
        "proto"  : [1],
        "cksum"  : [None],
        "src"    : [src_ip],
        "dst"    : [dst_ip]
    }
    return dic

def PC1ToR1(server_ip, server_port, data):
    #建立连接
    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.connect((server_ip,server_port))
    #发送数据
    client_socket.send(data.encode("utf-8"))
    print("——————成功发送数据至Router-1:",server_ip,"——————")
    print(data)
    #接收返回数据
    recv_data = client_socket.recv(1024)
    msg = recv_data.decode("utf-8")
    print(msg)
    client_socket.close()
    return msg

def main():
    src_ip = "192.168.45.57"
    dst_ip = "192.168.45.53"
    router1_ip = "192.168.45.56"
    #生成IP网络包
    package = createIpPackage(src_ip, dst_ip)
    #序列化
    str_pack = json.dumps(package)
    #发送至路由器
    msg = PC1ToR1(router1_ip, 8080, str_pack)
    print(msg)
    
if __name__ == "__main__":
    main()
