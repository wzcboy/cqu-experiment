#!/usr/bin/env python
# coding: utf-8

from socket import *
import json

def encodePackage(dic, src_ip, dst_ip, proto=4):
    src_arr = dic["src"]
    src_arr = src_arr + [src_ip]
    print(src_arr)
    dst_arr = dic["dst"]
    dst_arr = dst_arr + [dst_ip]
    proto_arr = dic["proto"]
    proto_arr = proto_arr + [proto]
    dic["src"] = src_arr
    dic["dst"] = dst_arr
    dic["proto"] = proto_arr
    return dic

def PC1toR1(r1_server_ip,r2_server_ip,r2_server_port):
    #创建服务器套接字
    server_socket = socket(AF_INET,SOCK_STREAM)
    server_socket.bind((r1_server_ip, 8080))
    server_socket.listen(128)
    #接收连接
    client_socket, client_addr = server_socket.accept()
    #接受数据
    str_ipPack = client_socket.recv(1024).decode("utf-8")
    #处理数据
    ipPack = json.loads(str_ipPack)
    print(ipPack)
    dic = encodePackage(ipPack,r1_server_ip,r2_server_ip)
    
    str_dic = json.dumps(dic)
    #发送数据到R2
    msg = R1ToR2(r2_server_ip, r2_server_port, str_dic)
    #回传结果
    client_socket.send(msg.encode("utf-8"))
    server_socket.close()
    client_socket.close()
    return msg

def R1ToR2(server_ip, server_port, data):
    #建立连接
    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.connect((server_ip,server_port))
    #发送数据
    client_socket.send(data.encode("utf-8"))
    print("——————成功发送数据至Router-2:",server_ip,"——————")
    #接收返回数据
    recv_data = client_socket.recv(1024)
    msg = recv_data.decode("utf-8")
    client_socket.close()
    return msg

def main():
    r1_server_ip = "192.168.45.56"
    r2_server_ip = "192.168.45.55"
    msg = PC1toR1(r1_server_ip,r2_server_ip, 8080)
    print(msg)
    
if __name__ == "__main__":
    main()
