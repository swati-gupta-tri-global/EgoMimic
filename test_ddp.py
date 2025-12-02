#!/usr/bin/env python3
"""Minimal DDP test to debug 4-GPU initialization issue."""
# docker exec -it swati-egomimic /bin/bash -c "cd /workspace/externals/EgoMimic && torchrun --nproc_per_node=4 test_ddp.py"
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    # Set environment variables
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    # Get rank from environment or default to 0
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"Process {rank}/{world_size} starting...")
    
    # Initialize process group
    if world_size > 1:
        print(f"Rank {rank}: Initializing process group...")
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        print(f"Rank {rank}: Process group initialized!")
    
    # Set device
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: Using GPU {rank}")
    
    return rank, world_size

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    try:
        rank, world_size = setup()
        
        # Create a simple model
        model = torch.nn.Linear(10, 10).cuda()
        
        if world_size > 1:
            model = DDP(model, device_ids=[rank])
        
        # Test forward pass
        x = torch.randn(20, 10).cuda()
        y = model(x)
        
        print(f"Rank {rank}: Forward pass successful! Output shape: {y.shape}")
        
        if world_size > 1 and rank == 0:
            print("\n✓ All ranks initialized successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()

if __name__ == "__main__":
    main()
