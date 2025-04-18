"""
This script performs object detection, tracking, and location inference. 

YOLOv7 is used to detect the ships/floating obstacles. It is then tracked using the SORT algorithm. Based on the object's
bounding box and the camera's properties, we can estimate the depth and position of the object. We then calculate the object's
absolute location using its relative position to the camera. The data is then sent out through
the nano_pub socket.

"""

import os

import base64
import cv2 
import time 
import numpy as np
from math import *

import dill
from collections import deque

from config import *
from utils.general import get_logger, get_socket, Detected_Object_Data
from utils.cv import draw_bbox
from CV.sort import * 
from CV.byte_track.byte_tracker import BYTETracker

if USE_TRT: from CV.yolov7_trt import YoloTRT
else: import yolov7

logger = get_logger(os.path.splitext(os.path.basename(__file__))[0], "INFO")

# ZMQ sockets
yolo_pub = get_socket(PORTS["YOLO_PUB"], "PUB")
video_pub = get_socket(PORTS["VIDEO_PUB"], "PUB")

# Record video
record = False

if __name__ == "__main__":

    try:
        logger.info("Started main program")
        
        # Create video write object
        vid_holder = cv2.VideoWriter(
                                    filename=f"track.mp4", 
                                    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),  # MP4 codec, 
                                    fps=15, 
                                    frameSize=RESIZED_VID_RESOLUTION)

        if USE_TRT: # load YOLOv7 tensorRT engine
            logger.info("Using TRT")
            logger.info(f"""\n\tLibrary Plugin: {LIBRARYPLUGIN}
                            \n\tENGINE: {ENGINE}""")
            
            model = YoloTRT(library=LIBRARYPLUGIN, engine=ENGINE,
                            conf=CONF_THRES, 
                            iou=IOUS_THRES,
                            obj_filter=[])
            
        else: # load YOLOv7 weights
            
            logger.info("Not using TRT")
            logger.info(f"\nModel: {V7_WEIGHTS}")
            model = yolov7.load(V7_WEIGHTS)

            # set model parameters
            model.conf = CONF_THRES      
            model.iou = IOUS_THRES    
            # model.classes = [YOLO["categories"].index("bottle")]   # (optional list) filter by class
            
        logger.info(f"""
                    \n\tIOU Thres: {IOUS_THRES}
                    \n\tCONF Thres: {CONF_THRES}""")

        # Initialise bytetrack
        byte_track = BYTETracker(BYTE_TRK['track_thresh'],
                              BYTE_TRK['track_buffer'],
                              BYTE_TRK['match_thresh'],
                              BYTE_TRK['min_box_area'],
                              BYTE_TRK['frame_rate'])

        prev_t = 0
        object_tracks = {}         # Store object traces

        cap = cv2.VideoCapture(VIDEO_SRC)
        logger.info(f"Running inference on: " + ("camera" if VIDEO_SRC == 0 else VIDEO_SRC))
        
        while True:
            while True:

                if cap.isOpened():
                    
                    # read video feed
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, RESIZED_VID_RESOLUTION)

                    edited_frame = frame.copy()
                    
                    # Perform inference 
                    if USE_TRT:  
                        detections = model.Inference(frame) # np array 
                    else:
                        detections = model(frame).pred[0] # pytorch tensor
                        detections.cpu().numpy()
                        
                    # Update object tracker
                    tracks = byte_track.update(output_results=detections)
                    
                    # Unpack tracks and store in list
                    obj_list = []
                    for i in range(len(tracks)):
                        obj_list.append(
                                        Detected_Object_Data(
                                            id=tracks[i].track_id, # int
                                            cls=0, # int
                                            bbox=tracks[i].tlbr.astype(int), # (x1,y1,x2,y2)
                                        )
                        )
                        # Draw bounding box
                        draw_bbox(edited_frame, obj_list[i].bbox, obj_list[i].id, CATEGORIES[obj_list[i].cls])
                 
                        # Track object positions
                        if obj_list[i].id not in object_tracks: object_tracks[obj_list[i].id] = deque([])
                        
                        # Get center x,y of bbox
                        tlwh = tracks[i].tlwh.astype(int) 
                        x, y = tlwh[0] + tlwh[2]//2, tlwh[1] + tlwh[3]//2
                        object_tracks[obj_list[i].id].append((x,y)) # add bbox center
                        if len(object_tracks[obj_list[i].id]) > 200: object_tracks[obj_list[i].id].popleft()
                        
                        # Draw tracelines
                        # for n in range(1, len(object_tracks[obj_list[i].id])):
                        #     if n % 5 == 0:
                        #         cv2.line(edited_frame, object_tracks[obj_list[i].id][n - 1], object_tracks[obj_list[i].id][n], (100, 150, 255), 8)
                 
                    # Draw fps 
                    time_elapsed_secs = time.time() - prev_t
                    prev_t = time.time() 
                    fps = 1/time_elapsed_secs
                    cv2.putText(edited_frame, f"FPS: {fps:.2f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(255,0,0))

                    cv2.imshow("Output", edited_frame)
                    
                    # Encode the frame as a base64 string
                    _, buffer = cv2.imencode('.jpg', frame)
                    encoded_frame = base64.b64encode(buffer)
                    
                    # Record video stream
                    if record: vid_holder.write(edited_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    logger.error("Cannot open capture.")
                    break
                
                yolo_pub.send(dill.dumps(obj_list))     # Publish detected objs
                video_pub.send(dill.dumps(encoded_frame)) # Publish unedited video frame
                    
            # # Rewind video if not running on camera
            if VIDEO_SRC != 0 or LOOP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    except Exception as E:
        logger.error(f"Exception occured: {E}")
        logger.info("Stopping program...")

    finally:
        cap.release()
        vid_holder.release()
        cv2.destroyAllWindows()

        logger.info("Ended main program")
