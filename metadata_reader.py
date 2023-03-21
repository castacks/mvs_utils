import json
import os, sys
from os.path import join
import numpy as np

from .frame_io import read_frame_graph
from .shape_struct import ShapeStruct

class MetadataReader(object):

    def __init__(self, data_dir):
        super().__init__()

        self.data_dir = data_dir
        self.frame_graph = None

        # Member variables that will be assigned later.
        self.metadata = None

        self.num_cams = None
        self.cam_paths_list = None
        self.cam_to_poses_dict = None
        self.cam_to_camdata = None

        self.rig_is_cam = None
        self.rig_paths_list = None

        self.init_cam_list = None
        self.init_imgtype_list = None
    
    def populate_typed_path_for_single_cam(self, types, single_cam_path):
        '''
        types (list of str): The simulator image types.
        '''

        return { t : single_cam_path for t in types }

    def read_metadata_and_initialize_dirs(self, metadata_path, frame_graph_path, create_dirs=True):
        '''
        Reads in the specified metadata file, which sets important variables such as number of cameras and their extrinsics.
        Also sets up the directory structure according to the specified metadata.
        
        creawte_dirs: If True, creates the associated directory structure. 
        '''
        
        # Read the frame graph first.
        self.frame_graph = read_frame_graph(frame_graph_path)
        print('Frame graph read successfully. ')
        
        with open(metadata_path) as metadata_file:

            #Load Metadata JSON and set the number of cameras
            self.metadata = json.load(metadata_file)
            self.num_cams = len(self.metadata['cams'])

            #Initialize indexing lists  
            self.cam_paths_list = []
            self.rig_paths_list = []
            self.cam_to_poses_dict = dict()

            #Print the number of found cameras
            print(f"Number of cameras found... {self.num_cams}!")

            #Initialize the camera name/id to camera data dictionary. 
            #This is used during data collection to streamline the data collection procedures.
            self.cam_to_camdata = dict()

            #Make a rig directory and initialize the rigdata struct.
            rig_out_dir = join(self.data_dir, "rig")
            if create_dirs:
                os.makedirs(rig_out_dir, exist_ok=True)

            rigdata = dict(
                path='rig', 
                types=self.metadata["rig_img_types"],
                typed_path=self.populate_typed_path_for_single_cam(
                    self.metadata["rig_img_types"], 
                    "rig"),
                is_rig=True,
                rig_is_cam=False,
                rig_is_cam_data=dict(
                    cam_idx=None, # Should be an non-negative integer.
                    cam_overlapped_types=[]
                ), 
                data=dict(frame="rbf", image_frame="rif", is_rig=True) # NOTE: Hardcoded "rbf" for the simulator.
            )

            cam_headers = [] # The table header for the physical cameras. If rig is cam, then
                             # There will not be a "rig" column. Not sure if it is a good design.
            # If the rig is a virtual camera of the simulator. Then we set this to True.
            self.rig_is_cam = False

            #Iterate through each found camera...
            for i, c in enumerate(self.metadata['cams']):
                
                #For each camera, create a directory and index that directory in the csv index.
                c_str = f"cam{i}"
                cam_headers.append(c_str)
                
                cpath = join(self.data_dir, c_str)
                if create_dirs:
                    os.makedirs(cpath, exist_ok=True)

                #Also create a camera data struct that holds all important data for data collection
                cdata = dict(
                    path=c_str,
                    types=c["img_types"],
                    typed_path=self.populate_typed_path_for_single_cam(
                        c["img_types"], 
                        c_str),
                    is_rig=False,
                    data=c
                )

                # Get the pose of the camera by querying the frame graph.
                frame_name = c["frame"]
                T_rig_cam = self.frame_graph.query_transform(f0="rbf", f1=frame_name) # FTensor0
                cam_position = T_rig_cam.translation.cpu().numpy()

                print("METADATAREADER-----")
                print(frame_name, cam_position)
                # cam_orientation = T_rig_cam.rotation.cpu().numpy()

                #If a camera is the first to have an origin position at the rig frame, set the rig_is_cam
                #flag to be true. Also include the rigdata in the list of data associated with the camera number.
                #If it is not the first camera, warn the user. Otherwise, add the camera data to the index.
                # if np.array_equal(cam_position,np.array([0.0,0.0,0.0])):
                if c["is_rig"]:
                    if self.rig_is_cam:
                        print(f"Camera {i} also is positioned at the origin (Numbering starts at 0). \
                              Since a previous camera was also positioned at the rig frame, this camera will \
                              not be indexed as the rig camera. To ensure that camera {i} is the rig camera, \
                              change the order of the camera in the metadata to be the first camera.")
                        continue
                    else:
                        self.rig_is_cam = True

                        # Figure out the image types.
                        s_types_rig = set(rigdata["types"])
                        s_types_cam = set(cdata["types"])
                        s_types_overlapped = s_types_cam.intersection( s_types_rig )
                        s_types_rig_only = s_types_rig.difference( s_types_overlapped )
                        
                        # Update rigdata.
                        rigdata["rig_is_cam"] = True
                        rigdata["rig_is_cam_data"]["cam_idx"] = i
                        rigdata["rig_is_cam_data"]["cam_overlapped_types"] = list(s_types_overlapped)
                        # Update the file path for the overlapped types. Overlapped types live in
                        # the virtual camera directory during data collection.
                        for overlapped_type in s_types_overlapped:
                            rigdata["typed_path"][overlapped_type] = c_str
                        
                        # Update the types of the underlying virtual camera in the simulator.
                        # This camera must contain all the types including the rig types.
                        cdata["types"] = list( s_types_cam.union( s_types_rig_only ) )

                        # Update the file path for the rig-only types. Rig-only types live in
                        # the rig virtual camera directory during data collection.
                        for rig_only_type in s_types_rig_only:
                            cdata["typed_path"][rig_only_type] = 'rig'

                        cdata.update({"is_rig":True})
                        self.cam_to_camdata.update({
                            i:cdata
                        })

                    
                self.cam_to_camdata.update({
                    i:cdata
                })

                self.cam_to_poses_dict.update({
                    i:list()
                })

            # rig always has metadata no matter if it is originally a virtual camera or not.
            self.cam_to_camdata.update({
                "rig":rigdata
            })

            # If the rig_is_cam flag was not set to true, then add the rig as a seperate virtual 
            # camera to the dictionary (and will be populated in the simulator).
            # This indicates that none of the virtual cameras identify itself as the rig. 
            # Also add the "rig" to the csv index.                
            if not self.rig_is_cam:
                self.cam_to_poses_dict.update({
                    "rig":list()
                })

                cam_headers.append("rig")

            # Place the headers at the top of the csv index
            self.cam_paths_list.append(cam_headers)
            # rig_paths_list only records the filename mapping for the rig.
            self.rig_paths_list.append(self.metadata["rig_img_types"])

            print(f'MetadataReader: self.rig_paths_list = \n{self.rig_paths_list}')

            #To initialize the ImageClient, a default cam list and image type is needed.
            #The image types can be changed on-the-fly later in the pipeline.
            self.init_cam_list = self.metadata["cams"][0]["airsim_cam_nums"]
            self.init_imgtype_list = self.metadata["cams"][0]["img_types"]
    
            #Print the camera to data conversion dictionary to ensure everything was read correctly.
            print(f'MetadataReader: self.cam_to_camdata = \n{self.cam_to_camdata}')
            