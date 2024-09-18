# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import asyncio
import queue
import sys
import time
from socket import socket
from time import sleep

from fedflow.util.comm_utils import ClientChannel, ServerChannel

send_queue = queue.SimpleQueue()
recv_queue = queue.SimpleQueue()


async def send_messages_polling(s: socket):
    while True:
        if not send_queue.empty():
            data = send_queue.get()
            s.sendall(data)


async def recv_messages_polling(s: socket):
    while True:
        data = s.recv(1024)
        recv_queue.put(data)
        print(data)


async def send_msg(s: socket, role):
    for i in range(10):
        time.sleep(1)
        send_queue.put(f"{role}: {i}")
        print(f"send_queue_size: {send_queue.qsize()}")


async def run(s: socket):
    await asyncio.gather(send_messages_polling(s), recv_messages_polling(s))


def async_server_test():
    from fedflow.util import init_tcp_server
    s: socket = init_tcp_server()
    asyncio.run(run(s))
    print("server send messages")
    send_msg(s, "server")


def async_client_test():
    from fedflow.util import init_tcp_client
    s: socket = init_tcp_client()
    asyncio.run(run(s))
    print("client send messages")
    send_msg(s, "client")


def socket_server_test():
    server = ServerChannel()
    sleep(10)
    for i in range(10):
        server.sendall(f"server: {i}")
        sleep(1)
        print(server.get_alldata())
    server.close()


def socket_client_test(cid):
    client = ClientChannel(cid)
    sleep(10)
    for i in range(10):
        sleep(1)
        client.send(f"client[{cid}]: {i}")
        print(client.get_data())


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == 'client':
        socket_client_test(sys.argv[2] if len(sys.argv) > 2 else 1)
    else:
        socket_server_test()
