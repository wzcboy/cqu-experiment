import socket
def main():
    s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #UDP通信
    send_data = input("please input string :")
    dst_ip_port = ("192.168.45.5",8080)
    s.sendto(send_data.encode("UTF-8"),dst_ip_port)
    s.close()



if __name__ == "__main__":
    main()				