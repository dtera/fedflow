# coding=utf-8

import logging
import pickle
import socket
import threading
import time
from queue import SimpleQueue

import numpy as np
import termplotlib as tpl
import torch
from prettytable import PrettyTable, ALL

END_OF_MESSAGE = "\n\t".encode()
END_OF_GENERATE = "finish".encode()


def get_hist_str(data, bins=40, orientation="vertical"):
    counts, bin_edges = np.histogram(data, bins=bins)
    fig = tpl.figure()
    fig.hist(counts, bin_edges, force_ascii=False, orientation=orientation)
    return fig.get_string()


class CommProfiler:
    def __init__(self) -> None:
        self.amt = []
        self.encode_t = []
        self.send_t = []

    def add_comm(self, amt=0, encode_t=0, send_t=0):
        self.amt.append(amt)
        self.encode_t.append(encode_t)
        self.send_t.append(send_t)

    def get_report(self):
        amt = np.array(self.amt)
        encode_t = np.array(self.encode_t)
        send_t = np.array(self.send_t)

        table = PrettyTable()
        table.field_names = ["", "Histogram"]

        table.align[""] = "l"  # "l" 对应左对齐
        table.align["Histogram"] = "r"  # "l" 对应左对齐
        table.add_row(
            [
                f"""Bytes\n    Total:{np.sum(amt)}\n    AVG:{round(np.mean(amt), 2)}""",
                get_hist_str(amt),
            ]
        )
        table.add_row(
            [
                f"""Encode Time\n    Total:{round(np.sum(encode_t), 6)} sec\n    AVG:{round(np.mean(encode_t), 6)} sec""",
                get_hist_str(amt),
            ]
        )
        table.add_row(
            [
                f"""Encode Throughput\n    AVG:{round(np.sum(amt) / np.sum(encode_t) / 1e6, 2)} MBps""",
                get_hist_str(amt / encode_t),
            ]
        )
        table.add_row(
            [
                f"""Send Time\n    Total:{round(np.sum(send_t), 6)} sec\n    AVG:{round(np.mean(send_t), 6)} sec""",
                get_hist_str(send_t),
            ]
        )
        table.add_row(
            [
                f"""Send bandwidth\n    AVG:{round(np.sum(amt) / np.sum(send_t) / 1e6, 2)} MBps""",
                get_hist_str(amt / send_t),
            ]
        )
        table.hrules = ALL
        print(table)


class AttnProfiler:
    def __init__(self) -> None:
        self.qkv = []
        self.lora = []

    def log_qkv(self, t):
        self.qkv.append(t)

    def log_lora(self, t):
        self.lora.append(t)

    def get_report(self):
        print(f"qkv mean time {np.mean(self.qkv)}")
        print(f"lora mean time {np.mean(self.lora)}")


def send_tensor(
        s: socket.socket,
        tensor: torch.Tensor,
        processing_method="numpy_pickle",
        trans_protocol="tcp",
        profiler: CommProfiler = None,
):
    # 记录开始时间
    if profiler is not None:
        start_time = time.time()

    data = ENCODING_MAP[processing_method](tensor)
    if profiler is not None:
        encode_time = time.time()
    logging.debug(f"Shape {list(tensor.shape)} tensor of size [{len(data)}] Bytes sent")
    # 发送数据
    SEND_METHOD_MAP[trans_protocol](s, data)

    # 记录结束时间
    if profiler is not None:
        end_time = time.time()

    # 计算耗时和速度
    if profiler is not None:
        profiler.add_comm(len(data), encode_time - start_time, end_time - encode_time)


def numpy_pickle_encoding(tensor):
    try:
        n = tensor.numpy()  # cpu
    except Exception:
        n = tensor.cpu().numpy()  # gpu
    return pickle.dumps(n)


def numpy_pickle_decoding(data):
    return torch.Tensor(pickle.loads(data))


def init_tcp_client(ip="127.0.0.1", port=12345):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    print(f"Client is connected to {ip}:{port}")
    return s


def post_tcp_sends():
    pass


def tcp_sends(s: socket.socket, data):
    s.sendall(data + END_OF_MESSAGE)


def recv_tensor(
        s: socket.socket,
        buffer_size=1024,
        encoding_method="numpy_pickle",
        trans_protocol="tcp",
):
    """receive tensor

    Args:
        s (socket.socket): _description_
        buffer_size (int, optional): actually not used. Defaults to 1024.
        encoding_method (str, optional): convert `torch.Tensor` to `bytes`. Defaults to "numpy_pickle".
        trans_protocol (str, optional): Currently only implemented TCP. Defaults to "tcp".

    Returns:
        `torch.Tensor`
    """
    data = RECV_METHOD_MAP[trans_protocol](s, buffer_size)
    logging.debug(f"received data length {len(data)}")
    tensor = DECODING_MAP[encoding_method](data)

    return tensor


def init_tcp_server(ip="0.0.0.0", port=12345):
    """init tcp socket for central server

    Args:
        ip (str, optional): _description_. Defaults to "0.0.0.0".
        port (int, optional): _description_. Defaults to 12345.

    Returns:
        _type_: _description_
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, port))
    print(f"Server is listening on {socket.gethostbyname(socket.gethostname())}:{port}")
    s.listen(1)
    conn, addr = s.accept()
    print(f"Client{addr} is connected")
    return conn


def tcp_recv(s: socket.socket = None, buffer_size=1024, data_consumer=None, is_break=True, is_running=lambda: True,
             except_hook=None):
    """recursively read tensor from buffer

    Args:
        s (socket.socket, optional): _description_. Defaults to None.
        buffer_size (int, optional): _description_. Defaults to 1024.

    Returns:
        _type_: _description_
    """
    data = b""
    while is_running():
        try:
            temp = s.recv(buffer_size)
            data += temp
            if temp.endswith(END_OF_MESSAGE):
                if data_consumer is not None:
                    data = data.rstrip(END_OF_MESSAGE)
                    data_consumer(data)
                    data = b""
                if is_break:
                    break
            if temp == END_OF_GENERATE:
                print("received END_OF_GENERATE")
                return
        except:
            if except_hook is not None:
                except_hook(False)

    return data.rstrip(END_OF_MESSAGE)


RECV_METHOD_MAP = {"tcp": tcp_recv}
SEND_METHOD_MAP = {"tcp": tcp_sends}
ENCODING_MAP = {
    "numpy_pickle": numpy_pickle_encoding,
}
DECODING_MAP = {
    "numpy_pickle": numpy_pickle_decoding,
}


#############################################Channel Communication Begin#############################################
class SocketChannel(threading.Thread):
    def __init__(self, channel_conn: socket.socket, send_callback=tcp_sends, recv_callback=tcp_recv, **kwargs):
        threading.Thread.__init__(self)

        self.daemon = True
        self.send_callback = send_callback
        self.recv_callback = recv_callback
        self.buffer_size = kwargs.get("buffer_size", 1024)
        self.data_encoder = kwargs.get("data_encoder", None)
        self.data_decoder = kwargs.get("data_decoder", None)
        self.auto_start = kwargs.get("auto_start", True)
        self.recv_queue = SimpleQueue()
        self.channel_conn = channel_conn
        self.set_running(True)

        if self.auto_start:
            self.start()

    def run(self):
        self.recv_callback(self.channel_conn, self.buffer_size, self._recv, False,
                           self.is_running, self.set_running)

    def send(self, data):
        enc_data = self.data_encoding(data) if self.data_encoder is None else self.data_encoder(data)
        self.send_callback(self.channel_conn, enc_data)

    def _recv(self, data):
        dec_data = self.data_decoding(data) if self.data_decoder is None else self.data_decoder(data)
        return self.recv_queue.put(dec_data)

    def get_data(self, cid=1):
        return self.recv_queue.get()

    def data_encoding(self, data):
        return pickle.dumps(data)

    def data_decoding(self, data):
        return pickle.loads(data)

    def is_running(self):
        return self.running

    def set_running(self, running=True):
        self.running = running

    def shutdown(self, __how: int = socket.SHUT_RDWR):
        self.channel_conn.shutdown(__how)

    def close(self):
        self.set_running(False)
        self.channel_conn.close()


class ServerChannel(threading.Thread):
    def __init__(self, ip="0.0.0.0", port=12345, send_callback=tcp_sends, recv_callback=tcp_recv, **kwargs):
        threading.Thread.__init__(self)

        self.daemon = True
        self.running = True
        self.port = port
        self.send_callback = send_callback
        self.recv_callback = recv_callback
        self.kwargs = kwargs
        self.clients: dict[str, SocketChannel] = {}
        self.send_cond: dict[str, threading.Condition] = {}

        self.channel_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.channel_sock.bind((ip, port))
        print(f"Server is listening on {self.public_address()}")
        self.channel_sock.listen(1)
        self.channel_sock.settimeout(3)

        self.start()

    def run(self):
        while self.running:
            try:
                channel_conn, addr = self.channel_sock.accept()
            except:
                continue
            print(f"Client{addr} is connected")
            client = SocketChannel(channel_conn, self.send_callback, self.recv_callback, **self.kwargs)
            cid = client.get_data()
            cid = addr[1] if cid == -1 else cid
            if cid in self.send_cond:
                with self.send_cond[cid]:
                    self.send_cond[cid].notify()
            self.clients[cid] = client
            self.clients = dict([(id_, self.clients[id_]) for id_ in self.clients if self.clients[id_].is_running()])

    def is_connected(self, cid):
        if cid in self.clients and self.clients[cid].is_running():
            return True
        logging.warning(f"Client[{cid}] is not connected, waiting for connection...")
        return False

    def send_wait_for(self, cid=1):
        if not self.is_connected(cid):
            if cid not in self.send_cond:
                self.send_cond[cid] = threading.Condition()
            with self.send_cond[cid]:
                self.send_cond[cid].wait()

    def send(self, cid, data):
        self.send_wait_for(cid)

        self.clients[cid].send(data)

    def sendall(self, data):
        if len(self.clients) == 0:
            self.send_wait_for()
        for cid in self.clients:
            self.send(cid, data)

    def get_data(self, cid):
        if self.is_connected(cid):
            return self.clients[cid].get_data()
        return

    def get_alldata(self):
        data = dict([(cid, self.get_data(cid)) for cid in self.clients])
        data = [(cid, data[cid]) for cid in self.clients if data[cid] is not None]
        if len(data) > 0:
            return data[0][1] if len(data) == 1 else dict(data)
        else:
            return None

    def public_address(self):
        return f"tcp://{socket.gethostbyname(socket.gethostname())}:{self.port}"

    def close(self):
        for cid in self.clients:
            self.clients[cid].close()
        self.running = False
        self.channel_sock.close()


class ClientChannel(SocketChannel):
    def __init__(self, cid=-1, channel_address="tcp://127.0.0.1:12345", send_callback=tcp_sends, recv_callback=tcp_recv,
                 **kwargs):
        host, s_port = channel_address.split("//")[-1].split(":")
        port = int(s_port)
        channel_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        channel_conn.connect((host, port))
        print(f"Client is connected to {channel_address}")
        kwargs["auto_start"] = False

        super().__init__(channel_conn, send_callback, recv_callback, **kwargs)
        self.send(cid)  # send client id to server
        self.start()
#############################################Channel Communication End###############################################
