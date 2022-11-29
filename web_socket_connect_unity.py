import socket


def get_rect(x1, y1, x2, y2, hardware_max_x=7, hardware_max_y=15, software_max_x=800, software_max_y=600):
    top_left = [int((x1 / software_max_x) * hardware_max_x), int((y1 / software_max_y) * hardware_max_y)]
    bottom_right = [int((x2 / software_max_x) * hardware_max_x), int((y2 / software_max_y) * hardware_max_y)]
    x1 = top_left[0]
    y1 = top_left[1]
    x2 = bottom_right[0]
    y2 = bottom_right[1]
    ar = []
    for x in range(x1, x2):
        ar.append([x, y1])
    for y in range(y1, y2):
        ar.append([x2, y])
    for x in range(x2, x1):
        ar.append([x, y2])
    for y in range(y2, y1):
        ar.append([x1, y])
    #return [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
    return ar


def tap_points(array):
    HOST = "127.0.0.1"
    PORT = 4426

    client_socket = socket.socket()  # instantiate
    client_socket.connect((HOST, PORT))  # connect to the server

    message = None

    for i in array:
        message = str(i[0]) + "," + str(i[1])
        client_socket.send(message.encode())  # send message
        data = client_socket.recv(1024).decode()  # receive response

        print('Received from server: ' + data)  # show in terminal

    client_socket.close()  # close the connection
