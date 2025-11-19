from pathlib import Path

import numpy as np
from utils import load_raster


def get_global_wind_velocity(all_wind_velocity_files:list[Path])->np.float32:
   """Get the global maximum wind velocity for normalization"""
   global_max_wind_velocity = -np.inf
   for file_name in all_wind_velocity_files:
      wind_velocity_grid = load_raster(file_name)
      global_max_wind_velocity = max(global_max_wind_velocity, wind_velocity_grid.data.max())
   return float(global_max_wind_velocity)


def wind_angle_speed_to_uv(speed:np.ndarray, wd_deg:np.ndarray, nodata:float=-9999.0):
    """
       Returns u (east), v (north) as masked numpy arrays (mask True=invalid).
    """
    mask = (speed == nodata) | (wd_deg == nodata)
    # fill masked with 0 for trig computations 
    s = np.where(mask, 0.0, speed)
    wd = np.where(mask, 0.0, wd_deg)
    rad = np.deg2rad(wd)

    u = -s * np.sin(rad)   # eastward
    v = -s * np.cos(rad)   # northward
    return u, v, mask.astype(np.float32)  # mask channel: 1=valid, 0=invalid

def get_all_wind_grids(all_wind_velocity_files:list[Path], global_max_wind_velocity:float)->tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
   """
    Loading the u,v and mask for all possible wind directions 
   """
   all_u, all_v, all_mask = [],[],[]
   for wind_velocity_filename in all_wind_velocity_files:
      wind_ang_filename = wind_velocity_filename.with_name(wind_velocity_filename.name.replace("_vel.asc", "_ang.asc"))
      wind_velocity_grid = load_raster(wind_velocity_filename)
      wind_angle_grid = load_raster(wind_ang_filename)
      u,v,mask  = wind_angle_speed_to_uv(wind_velocity_grid.data, wind_angle_grid.data)
      all_u.append(u/global_max_wind_velocity) #normalizing by max speed
      all_v.append(v/global_max_wind_velocity)#normalizing by max speed
      all_mask.append(mask)
   return all_u, all_v, all_mask

def sample_wind_grids(all_u:list, all_v:list, all_mask:list, sampling:str="all"):
   """
    Sampling the wind grids from the loaded u,v. 
    Options: dist, all
        dist: get the mean and std for u,v
        all: stack all the directions together
    """
   combined_mask = np.logical_or.reduce(all_mask)
   if sampling=="dist":
      stacked_u = np.stack(all_u, axis=-1)  # (H,W,n_dir)
      stacked_v = np.stack(all_v, axis=-1)

      u_ma = np.ma.array(stacked_u, mask=np.broadcast_to(combined_mask[:,:,np.newaxis], stacked_u.shape))
      v_ma = np.ma.array(stacked_v, mask=np.broadcast_to(combined_mask[:,:,np.newaxis], stacked_v.shape))

      mean_u = u_ma.mean(axis=-1).filled(0)
      mean_v = v_ma.mean(axis=-1).filled(0)

      std_u = u_ma.std(axis=-1).filled(1e-6)  # avoid divide-by-zero
      std_v = v_ma.std(axis=-1).filled(1e-6)
      
      wind_grid = np.stack([mean_u.data, mean_v.data, std_u.data, std_v.data], axis=-1)
   else:
      u_v_combined = [val for pair in zip(all_u, all_v, strict=False) for val in pair] #Alternating between u,v pairs
      wind_grid = np.stack(u_v_combined, axis=-1) #(H,W,2*n_dir)
   
   wind_grid = np.ma.array(wind_grid, mask=np.broadcast_to(combined_mask[:,:,np.newaxis], wind_grid.shape))
   return wind_grid

def get_wind_grid_sample(path_wind_grids:str,sampling:str="all")->np.ndarray:
   """
    Get the final wind grid
   """
   path_wind_grids = Path(path_wind_grids)
   all_wind_velocity_files = list(path_wind_grids.glob("w???_vel.asc"))
   global_max_wind_velocity = get_global_wind_velocity(all_wind_velocity_files)
   
   all_u, all_v, all_mask = get_all_wind_grids(all_wind_velocity_files, global_max_wind_velocity)
   wind_grid = sample_wind_grids(all_u, all_v, all_mask, sampling=sampling)
   return wind_grid

#TODO: Delete Later
if __name__=="__main__":
   data_folder = "../yan_bp3/hex05"
   folders = ["burning_conditions_module", "dictionary", "ignitions_module",
             "mapped_inputs",  "outputs"]
   path_wind_grids = data_folder + "/" + folders[0] + "/wind_grids"
   wind_grid = get_wind_grid_sample(path_wind_grids,sampling="all")
   print(wind_grid.shape)
