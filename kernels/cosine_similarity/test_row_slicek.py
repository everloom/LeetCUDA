import torch
import numpy as np

if __name__ == "__main__":
    aa = torch.arange(12).reshape(3, 4)
    bb = torch.arange(12, 24).reshape(3, 4)
    bbt = bb.T
    cc = aa @ bbt
    
    print(f"{'Lane':<5} | {'Col':<3} | {'b0 (Row)':<10} | {'b1 (Row)':<10} | {'b2 (Row)':<10} | {'b3 (Row)':<10}")
    print("-" * 65)
    
    for lane_id in range(32):
        # groupID = laneid >> 2 (决定列)
        col = lane_id >> 2
        
        # threadID_in_group = laneid % 4 (决定行偏移)
        tid_in_group = lane_id % 4
        
        # 根据图示逻辑计算行索引
        # b0: threadID_in_group * 2
        # b1: threadID_in_group * 2 + 1
        # b2: threadID_in_group * 2 + 8
        # b3: threadID_in_group * 2 + 9
        
        row_b0 = tid_in_group * 2
        row_b1 = tid_in_group * 2 + 1
        row_b2 = tid_in_group * 2 + 8
        row_b3 = tid_in_group * 2 + 9
        
        print(f"{lane_id:<5} | {col:<3} | {row_b0:<10} | {row_b1:<10} | {row_b2:<10} | {row_b3:<10}")

    # breakpoint()