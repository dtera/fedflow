# coding=utf-8

import socket
import torch
from fedflow.utils import send_tensor, recv_tensor, CommProfiler


class PLAB(torch.nn.Module):
    def __init__(self, in_features: int, rcd: int, rdc: int, out_features: int):
        """A,B matrices on Cloud, PrivateLorA of Q,K,V are stacked to execute in parallel

        Args:
            in_features (int):
            rcd (int):
            rdc (int):
            out_features (int):
        """
        super().__init__()
        self.lora_A = torch.nn.Parameter(
            torch.zeros((3, rcd, in_features)), requires_grad=False
        )  # 3 for q,k,v, stacked together for parallel execution
        self.lora_B = torch.nn.Parameter(
            torch.zeros((3, out_features, rdc)), requires_grad=False
        )

    def forward(self, x, s: socket.socket, buffer_size=int(2048e3)):
        x = x.to(self.lora_A.dtype).to(self.lora_A.device)
        # compress activations
        x = x @ self.lora_A.transpose(2, 1)
        # send compressed activations to edge device
        send_tensor(s, x)
        # receive comparessed and transformed activations from edge device
        x = recv_tensor(s, buffer_size).to(self.lora_B.device).to(self.lora_B.dtype)
        # de compress to hidden dimension
        x = x @ self.lora_B.transpose(2, 1)
        # chunk for q,k,v
        return x.chunk(3, 0)


class PLM(torch.nn.Module):
    def __init__(self, rcd: int, rdc: int, **kwargs) -> None:
        """PrivateLoRA M matrix
        1. receive compressed activations from cloud as input
        2. transform on activations
        3. send back activations

        Args:
            rcd (int): dimension of cloud 2 device
            rdc (int): dimension of device 2 cloud
        """
        super().__init__(**kwargs)
        # q,k,v lora stacked together
        self.lora_M = torch.nn.Parameter(
            torch.zeros((3, rdc, rcd)), requires_grad=False
        )

    def forward(
            self, s: socket.socket, buffer_size=int(2048e3), profiler: CommProfiler = None
    ):
        """
        Args:
            s (socket.socket):
            buffer_size (int, optional): useless but i'm lazy. Defaults to int(2048e3).
            profiler (CommProfiler, optional): if not None profiler will do performance profile. Defaults to None.
        """
        # receive compressed activations
        x = recv_tensor(s, buffer_size).to(self.lora_M.device).to(self.lora_M.dtype)
        # transform them
        x = x @ self.lora_M.transpose(2, 1)
        # send them back to cloud
        send_tensor(s, x, profiler=profiler)


class PLMStack(torch.nn.Module):
    def __init__(self, num_hidden_layers: int, rcd: int, rdc: int, **kwargs) -> None:
        """Stack of PrivateLoRA M

        Args:
            num_hidden_layers (int): number of m does not necessarily equal to number of decoder layers.
            rcd (int): dimension of M matrix, indicates transmission base for cloud 2 device connection
            rdc (int): dimension of M matrix, indicates transmission base for device 2 cloud connection
        """
        super().__init__(**kwargs)
        self.layers = torch.nn.ModuleList(
            [PLM(rcd, rdc) for _ in range(num_hidden_layers)]
        )

    def forward(
            self, s: socket.socket, buffer_size=int(2048e3), profiler: CommProfiler = None
    ):
        """
        Args:
            s (socket.socket):
            buffer_size (int, optional): useless but i'm lazy. Defaults to int(2048e3).
            profiler (CommProfiler, optional): if not None profiler will do performance profile. Defaults to None.
        """
        for i, layer in enumerate(self.layers):
            # print(f"{i} th layer")
            layer(s, buffer_size, profiler=profiler)
