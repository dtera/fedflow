# coding=utf-8

import logging
import pickle
import socket
import time

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
    print("device connected")
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
    print(f"Client[addr:{addr}] is connected")
    return conn


def tcp_recv(s: socket.socket = None, buffer_size=1024):
    """recursively read tensor from buffer

    Args:
        s (socket.socket, optional): _description_. Defaults to None.
        buffer_size (int, optional): _description_. Defaults to 1024.

    Returns:
        _type_: _description_
    """
    data = b""
    while True:
        temp = s.recv(buffer_size)
        data += temp
        if temp.endswith(END_OF_MESSAGE):
            break
        if temp == END_OF_GENERATE:
            print("received END_OF_GENERATE")
            return

    return data.rstrip(END_OF_MESSAGE)


RECV_METHOD_MAP = {"tcp": tcp_recv}
SEND_METHOD_MAP = {"tcp": tcp_sends}
ENCODING_MAP = {
    "numpy_pickle": numpy_pickle_encoding,
}
DECODING_MAP = {
    "numpy_pickle": numpy_pickle_decoding,
}
