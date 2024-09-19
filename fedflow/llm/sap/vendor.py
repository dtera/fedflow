import argparse
import logging
import time

import torch

from fedflow.llm.sap.model import SAPLlamaConfig, LlamaForVender

parser = argparse.ArgumentParser()

parser.add_argument(
    "weight_path",
    default=None,
    help="prepared cloud weight",
)
parser.add_argument(
    "llama_path",
    default=None,
    help="root dir of huggingface llama model, should contain weight files and config",
)
parser.add_argument(
    "--ip",
    default="127.0.0.1",
    help="socket ip of cloud",
)
parser.add_argument(
    "--port",
    default=12345,
    help="socket port of cloud",
)
parser.add_argument(
    "--device",
    default="cuda",
    help="device of model",
)
parser.add_argument(
    "--debug",
    default=False,
)
args = parser.parse_args()

log_format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"

logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO, format=log_format
)

if __name__ == "__main__":
    from fedflow.util.comm_utils import init_tcp_server

    logging.info("Loading state dict...")
    vendor_dict = torch.load(args.weight_path)
    logging.info("state dict loaded")

    config = SAPLlamaConfig.from_pretrained(args.llama_path)
    # ranks of A and B, hard code
    config.rcd = 128
    config.rdc = 128
    # initializing model
    logging.info("Initializing Model")
    model = LlamaForVender(config)
    print(model)
    logging.info("model initialized")

    count = 0
    for n, p in model.named_parameters():
        count += p.numel()
    logging.info(f"model has {count} parameters")
    logging.info("Start loading state dict...")
    model.load_state_dict(vendor_dict, strict=False)
    logging.info("State dict loaded")

    model = model.to(args.device)
    model.vendor_to_fp16()
    logging.info("model ready, you can now launch customer.py")

    s = init_tcp_server(args.ip, args.port)

    logging.info("Enter listening loop")
    # listening loop of server
    with torch.no_grad():
        while True:
            time.sleep(1)
            try:
                # toy handshake protocol
                data = s.recv_(3)
                if "new" == data.decode():
                    s.send("new".encode())
                    logging.info("===========Request Coming=========")
                    model.my_generate(s=s)
                else:
                    print("Waiting")
            except Exception:
                print()
                continue
