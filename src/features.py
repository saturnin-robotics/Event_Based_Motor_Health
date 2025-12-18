import torch
import numpy as np
import h5py
from tqdm import tqdm  # Progress bar

#  SENSOR CONSTANTS
# Resolution of the DVS/DAVIS sensor
WIDTH = 240
HEIGHT = 180
C_COUNT = 2            # Number of channels (0: OFF, 1: ON)

# HYPERPARAMETERS 
# Aggregation time window in microseconds (1ms)
TIME_WINDOW_US = 1000 

def slice_events_into_frames(file_path, time_window_us=TIME_WINDOW_US):
    """
    Transforms the raw event stream into a sequence of 3D Voxel Grids (Frames).
    
    This function uses a vectorized approach (NumPy/PyTorch) for high performance.
    It aggregates events occurring within 'time_window_us' into a (W, H, C) tensor.

    Args:
        file_path (str): Path to the HDF5 file.
        time_window_us (int): Duration of each frame in microseconds.

    Returns:
        list[torch.Tensor]: A list of 3D tensors, each representing a time slice.
    """
    print(f" ---Processing features from: {file_path} ---")
    
 
    # (We bypass EventDataset here because __getitem__ loop is too slow for slicing)
    with h5py.File(file_path, 'r') as f:
        # Access the event table
        events_db = f['CD']['events']
        
        #  Fast reading of all timestamps
        print("Reading all timestamps...")
        all_ts = events_db['t'] # HDF5 allows fast column access
        
        if len(all_ts) == 0:
            print("Error: No events found in file.")
            return []

        start_t = all_ts[0]
        last_t = all_ts[-1]
        
        #  Vectorized Time Slicing
        # We calculate where to cut the array to create 1ms packets
        print("Calculating slicing indices...")
        
        # Create time bins: [t0, t0+dt, t0+2dt, ...]
        bins = np.arange(start_t, last_t + time_window_us, time_window_us)
        
        # 'indices' holds the array positions where each new millisecond starts
        # np.searchsorted is extremely fast (binary search)
        indices = np.searchsorted(all_ts, bins)
        
        num_frames = len(indices) - 1
        print(f"Generating {num_frames} frames...")
        
        event_frames_list = []
        
        # Loop over BLOCKS (Chunks) instead of single events
        # tqdm provides a progress bar
        for i in tqdm(range(num_frames), desc="Building Frames"):
            
            # Start and End index for the current frame in the event array
            idx_start = indices[i]
            idx_end = indices[i+1]
            
            # Initialize an empty frame (Voxel Grid)
            frame = torch.zeros(WIDTH, HEIGHT, C_COUNT)
            
            # If there are events in this time window
            if idx_end > idx_start:
                # Read the full chunk from disk/memory at once
                chunk = events_db[idx_start:idx_end]
                
                # Direct conversion to Tensors (Vectorized)
                # We avoid the slow Python for-loop here
                xs = torch.from_numpy(chunk['x'].astype(np.int64))
                ys = torch.from_numpy(chunk['y'].astype(np.int64))
                ps = torch.from_numpy(chunk['p'].astype(np.int64))
                
                # Convert Polarity: (-1, 1) -> Channel Index (0, 1)
                # If p=1 (ON) -> Index 1, If p=-1 (OFF) -> Index 0
                cs = (ps == 1).long() 
                
                # Safety Filter: Ensure coordinates are within sensor bounds
                mask = (xs < WIDTH) & (ys < HEIGHT)
                
                if mask.any():
                    # Apply mask to keep only valid events
                    xs = xs[mask]
                    ys = ys[mask]
                    cs = cs[mask]
                    
                    # Vectorized Accumulation
                    # Create a tensor of indices [3, N]
                    indices_tensor = torch.stack((xs, ys, cs), dim=0)
                    
                    # Values to add (1 per event)
                    values = torch.ones(len(xs))
                    
                    # 'index_put_' adds 1 to frame[x, y, c] for all events efficiently
                    frame.index_put_(tuple(indices_tensor), values, accumulate=True)
            
            # Add the completed frame to the list
            event_frames_list.append(frame)
            
    print(f"Feature extraction complete. Total frames: {len(event_frames_list)}")
    return event_frames_list