# coding=utf-8

import logging
import pickle
import socket
import threading
import time
from multiprocessing.pool import ThreadPool
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
        self._send_callback = send_callback
        self._recv_callback = recv_callback
        self._buffer_size = kwargs.get("buffer_size", 1024)
        self._data_encoder = kwargs.get("data_encoder", None)
        self._data_decoder = kwargs.get("data_decoder", None)
        self._auto_start = kwargs.get("auto_start", True)
        self._recv_queue = SimpleQueue()
        self._send_pool = ThreadPool(processes=1)
        self._channel_conn = channel_conn
        self._set_running(True)

        if self._auto_start:
            self.start()

    def run(self):
        self._recv_callback(self._channel_conn, self._buffer_size, self._recv, False, self.is_running,
                            self._set_running)

    def _recv(self, data):
        dec_data = self.data_decoding(data) if self._data_decoder is None else self._data_decoder(data)
        return self._recv_queue.put(dec_data)

    def send(self, data, pre_data_encoder=None):
        enc_data = data if pre_data_encoder is None else pre_data_encoder(data)
        enc_data = self.data_encoding(enc_data) if self._data_encoder is None else self._data_encoder(enc_data)

        self._send_pool.apply_async(self._send_callback, (self._channel_conn, enc_data))

    def recv(self, post_data_decoder=None):
        dec_data = self._recv_queue.get()
        return dec_data if post_data_decoder is None else post_data_decoder(dec_data)

    def data_encoding(self, data):
        return pickle.dumps(data)

    def data_decoding(self, data):
        return pickle.loads(data)

    def is_running(self):
        return self.running

    def _set_running(self, running=True):
        self.running = running

    def shutdown(self, __how: int = socket.SHUT_RDWR):
        self._channel_conn.shutdown(__how)

    def close(self):
        self._set_running(False)
        self._channel_conn.close()

    def to_tensor(self, data):
        return torch.Tensor(data)

    def from_tensor(self, tensor):
        try:
            data = tensor.numpy()  # cpu
        except Exception:
            data = tensor.cpu().numpy()  # gpu
        return data

    def send_tensor(self, data):
        return self.send(data, self.from_tensor)

    def recv_tensor(self):
        return self.recv(self.to_tensor)


class ServerChannel(threading.Thread):
    def __init__(self, ip="0.0.0.0", port=12345, send_callback=tcp_sends, recv_callback=tcp_recv, **kwargs):
        threading.Thread.__init__(self)

        self.daemon = True
        self._running = True
        self._port = port
        self._send_callback = send_callback
        self._recv_callback = recv_callback
        self._kwargs = kwargs
        self._clients: dict[str, SocketChannel] = {}
        self._send_cond: dict[str, threading.Condition] = {}

        self._channel_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._channel_sock.bind((ip, port))
        print(f"Server is listening on {self.public_address()}")
        self._channel_sock.listen(1)
        self._channel_sock.settimeout(3)

        self.start()

    def run(self):
        while self._running:
            try:
                channel_conn, addr = self._channel_sock.accept()
            except:
                continue
            print(f"Client{addr} is connected")
            client = SocketChannel(channel_conn, self._send_callback, self._recv_callback, **self._kwargs)
            cid = client.recv()
            cid = str(addr[1] if cid == -1 else cid)
            if cid in self._send_cond:
                with self._send_cond[cid]:
                    self._send_cond[cid].notify()
            self._clients[cid] = client
            self._clients = dict(
                [(id_, self._clients[id_]) for id_ in self._clients if self._clients[id_].is_running()])

    def is_connected(self, cid=1):
        cid = str(cid)
        if cid in self._clients and self._clients[cid].is_running():
            return True
        logging.warning(f"Client[{cid}] is not connected, waiting for connection...")
        return False

    def _send_wait_for(self, cid=2000):
        cid = str(cid)
        if not self.is_connected(cid):
            if cid not in self._send_cond:
                self._send_cond[cid] = threading.Condition()
            with self._send_cond[cid]:
                self._send_cond[cid].wait()

    def send_(self, cid, data, pre_data_encoder=None):
        self._send_wait_for(cid)
        self._clients[cid].send(data, pre_data_encoder)

    def send_tensor_(self, cid, data):
        self._send_wait_for(cid)
        self._clients[cid].send_tensor(data)

    def send(self, data, pre_data_encoder=None):
        if len(self._clients) == 0:
            self._send_wait_for()
        for cid in self._clients:
            self.send_(cid, data, pre_data_encoder)

    def send_tensor(self, data):
        if len(self._clients) == 0:
            self._send_wait_for()
        for cid in self._clients:
            self.send_tensor_(cid, data)

    def recv_(self, cid, post_data_decoder=None):
        cid = str(cid)
        if self.is_connected(cid):
            return self._clients[cid].recv(post_data_decoder)
        return

    def recv_tensor_(self, cid):
        cid = str(cid)
        if self.is_connected(cid):
            return self._clients[cid].recv_tensor()
        return

    def recv(self, post_data_decoder=None):
        data = dict([(cid, self.recv_(cid, post_data_decoder)) for cid in self._clients])
        data = [(cid, data[cid]) for cid in self._clients if data[cid] is not None]
        if len(data) > 0:
            return data[0][1] if len(data) == 1 else dict(data)
        else:
            return None

    def recv_tensor(self):
        data = dict([(cid, self.recv_tensor_(cid)) for cid in self._clients])
        data = [(cid, data[cid]) for cid in self._clients if data[cid] is not None]
        if len(data) > 0:
            return data[0][1] if len(data) == 1 else dict(data)
        else:
            return None

    def public_address(self):
        return f"tcp://{socket.gethostbyname(socket.gethostname())}:{self._port}"

    def close(self):
        for cid in self._clients:
            self._clients[cid].close()
        self._running = False
        self._channel_sock.close()


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
