import os
import tempfile
from collections import defaultdict
from copy import deepcopy
from queue import Empty, Queue

import h5py
import imageio
import numpy as np

from droid.misc.subprocess_utils import run_threaded_command


def write_dict_to_hdf5(hdf5_file, data_dict, keys_to_ignore=["image", "depth", "pointcloud"]):
    for key in data_dict.keys():
        # --- FIX: Skip empty keys to prevent HDF5 crash ---
        if not key: 
            continue
        # --------------------------------------------------
        
        if key in keys_to_ignore:
            continue

        curr_data = data_dict[key]
        
        # 1. Handle Dictionaries (Recursion)
        if isinstance(curr_data, dict):
            # Encode key for safety if hdf5 requires bytes
            try:
                # Check if group exists using string
                if key in hdf5_file:
                    group_key = key
                else:
                    # If creating new, try encoding (but keep string if encoding fails)
                    try:
                        group_key = key.encode('utf-8')
                    except:
                        group_key = key
            except:
                group_key = key
                
            if group_key not in hdf5_file:
                hdf5_file.create_group(group_key)
            write_dict_to_hdf5(hdf5_file[group_key], curr_data)
            continue

        # 2. Robust Data Conversion
        if isinstance(curr_data, list):
            curr_data = np.array(curr_data)
        
        if not isinstance(curr_data, np.ndarray):
            curr_data = np.array(curr_data)

        # 3. Handle String Data (Unicode -> Bytes)
        if curr_data.dtype.kind == 'U':
            curr_data = np.char.encode(curr_data, 'utf-8')
        elif curr_data.dtype.kind == 'O':
            try:
                curr_data = curr_data.astype('S')
            except:
                pass

        # 4. Handle Dataset Name (Key) Encoding
        safe_key = key

        # 5. Create or Resize Dataset
        # Try finding the dataset with the string key first
        dataset_exists = safe_key in hdf5_file
        
        if not dataset_exists:
            dshape = curr_data.shape
            dtype = curr_data.dtype
            
            try:
                hdf5_file.create_dataset(
                    safe_key, 
                    (1, *dshape), 
                    maxshape=(None, *dshape), 
                    dtype=dtype
                )
            except (TypeError, ValueError):
                # Fallback: Try encoded bytes key
                try:
                    safe_key_bytes = safe_key.encode('utf-8')
                    if not safe_key_bytes: raise ValueError("Empty key")
                    
                    hdf5_file.create_dataset(
                        safe_key_bytes, 
                        (1, *dshape), 
                        maxshape=(None, *dshape), 
                        dtype=dtype
                    )
                    safe_key = safe_key_bytes # Update for next resize
                except:
                    # If all else fails, skip this specific key to save the rest
                    print(f"Warning: Could not save key '{key}'. Skipping.")
                    continue
        else:
            # Extend the dataset by 1
            # Note: We need to use the exact key format (str or bytes) that exists in the file
            try:
                hdf5_file[safe_key].resize(hdf5_file[safe_key].shape[0] + 1, axis=0)
                hdf5_file[safe_key][-1] = curr_data
            except KeyError:
                # Maybe it exists as bytes?
                safe_key_bytes = safe_key.encode('utf-8')
                if safe_key_bytes in hdf5_file:
                    hdf5_file[safe_key_bytes].resize(hdf5_file[safe_key_bytes].shape[0] + 1, axis=0)
                    hdf5_file[safe_key_bytes][-1] = curr_data

class TrajectoryWriter:
    def __init__(self, filepath, metadata=None, exists_ok=False, save_images=True):
        assert (not os.path.isfile(filepath)) or exists_ok
        self._filepath = filepath
        self._save_images = save_images
        self._hdf5_file = h5py.File(filepath, "w")
        self._queue_dict = defaultdict(Queue)
        
        self._video_writers = {}
        self._video_files = {}
        self._video_buffers = {} 
        
        self._open = True

        if metadata is not None:
            self._update_metadata(metadata)

        def hdf5_writer(data):
            return write_dict_to_hdf5(self._hdf5_file, data)

        run_threaded_command(self._write_from_queue, args=(hdf5_writer, self._queue_dict["hdf5"]))

    def write_timestep(self, timestep):
        if self._save_images:
            self._update_video_files(timestep)
        self._queue_dict["hdf5"].put(timestep)

    def _update_metadata(self, metadata):
        for key in metadata:
            self._hdf5_file.attrs[key] = deepcopy(metadata[key])

    def _write_from_queue(self, writer, queue):
        while self._open:
            try:
                data = queue.get(timeout=1)
            except Empty:
                continue
            writer(data)
            queue.task_done()

    def _update_video_files(self, timestep):
        if "observation" in timestep:
            obs_key = "observation"
        elif "observations" in timestep:
            obs_key = "observations"
        else:
            return
        
        image_dict = timestep[obs_key]["image"]

        for video_id in image_dict:
            img = image_dict[video_id]
            
            if video_id not in self._video_writers:
                filename = self.create_video_file(video_id, ".mp4")
                self._video_writers[video_id] = imageio.get_writer(filename, macro_block_size=1)
                run_threaded_command(
                    self._write_from_queue, args=(self._video_writers[video_id].append_data, self._queue_dict[video_id])
                )

            self._queue_dict[video_id].put(img)

        del timestep[obs_key]["image"]

    def create_video_file(self, video_id, suffix):
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix)
        self._video_files[video_id] = temp_file
        return temp_file.name

    def close(self, metadata=None):
        if metadata is not None:
            self._update_metadata(metadata)

        [queue.join() for queue in self._queue_dict.values()]

        for video_id in self._video_writers:
            self._video_writers[video_id].close()

        root_key = "observation" if "observation" in self._hdf5_file else "observations"
        if root_key not in self._hdf5_file:
             self._hdf5_file.create_group("observation")
             root_key = "observation"

        if "videos" not in self._hdf5_file[root_key]:
            self._hdf5_file[root_key].create_group("videos")

        for video_id in self._video_files:
            self._video_files[video_id].seek(0)
            serialized_video = np.asarray(self._video_files[video_id].read())
            self._hdf5_file[root_key]["videos"].create_dataset(video_id, data=serialized_video)
            self._video_files[video_id].close()

        self._hdf5_file.close()
        self._open = False