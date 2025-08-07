import os
import random


def _use_torchrun():
    nproc_per_node = os.getenv("NPROC_PER_NODE")
    if nproc_per_node is None:
        return False
    return True


def _get_torchrun_args():
    if not _use_torchrun():
        return None

    nnodes = int(os.getenv("NNODES", 1))
    nproc_per_node = int(os.getenv("NPROC_PER_NODE"))
    node_rank = os.getenv("NODE_RANK", 0)
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT")
    if master_port is None:
        master_port = random.randint(20000, 29999)

    torchrun_args = [
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
    ]
    return torchrun_args
