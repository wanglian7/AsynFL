import pickle, struct, socket

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())

def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg


def recv_msg(sock, expect_msg_type=None):
    # 修改为循环接收，确保收到完整的4字节长度信息
    msg_len_bytes = bytearray()
    while len(msg_len_bytes) < 4:
        received = sock.recv(4 - len(msg_len_bytes))
        if not received:  # 如果没有数据接收到，表示连接可能已关闭
            raise ConnectionAbortedError("Connection closed unexpectedly while waiting for message length.")
        msg_len_bytes.extend(received)

    msg_len = struct.unpack(">I", bytes(msg_len_bytes))[0]

    # 同样地，确保完整接收消息内容
    msg_data = bytearray()
    while len(msg_data) < msg_len:
        chunk = sock.recv(msg_len - len(msg_data))
        if not chunk:  # 连接意外关闭的检查
            raise ConnectionAbortedError("Connection closed unexpectedly while receiving message data.")
        msg_data.extend(chunk)

    msg = pickle.loads(msg_data)
    print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + str(msg[0]))

    return msg