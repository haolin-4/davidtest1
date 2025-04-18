# Ports
#---------------------------------------------------------------------------------------------------
PORTS = {
  "FC_PUB":                 "tcp://127.0.0.1:5555",     #"tcp://192.168.137.218:5555"  # from FC to nano
  "YOLO_PUB":               "tcp://127.0.0.1:5556",     # from yolo to obj processing
  "OBJ_PROCESS_PUB":        "tcp://127.0.0.1:5557",     #"tcp://192.168.2.3:5555" # from nano to pi (jetson nano's static ip)
  "RL_PUB":                 "tcp://127.0.0.1:5558", 
  "VIDEO_PUB":              "tcp://127.0.0.1:5559",     # for live video feed from camera
  "PC_TO_JETSON_PUB":       "tcp://192.168.137.1:8000"  # for communication with Jetson Nano from the PC
}

# Flight Controller
#---------------------------------------------------------------------------------------------------
JETSON_UART_PORT = '/dev/ttyTHS1'
SITL_PORT = 'tcp:127.0.0.1:5762'
FC_BAUD_RATE = 57600
FC_TIMEOUT =  60
SERVO_CHANNEL =  1
THROTTLE_CHANNEL = 3

USE_SITL_SIM = True  # Set to true for simulation use

# RL
#---------------------------------------------------------------------------------------------------
RL_MODEL = "RL\model"


# YOLO
#---------------------------------------------------------------------------------------------------
USE_TRT = False # set accordingly to the device
RESIZED_VID_RESOLUTION = [640, 480]

# For Tensor RT
LIBRARYPLUGIN = "cv/yolo/libmyplugins_fp16.so"
ENGINE = "cv/yolo/ship_detection_fp16.engine"

# For default yolo
V7_WEIGHTS = "cv/yolo/v4.pt"

CONF_THRES = 0.4 
IOUS_THRES = 0.5

MAX_NO_OF_OBJS = 10

CATEGORIES = ["vessel"]

VIDEO_SRC = r"ships.mp4" # 0 for webcam
LOOP = False # Loop video

# FOr byte track object tracking
BYTE_TRK = {
  "track_thresh" : 0.5, 
  "track_buffer" : 15*30,
  "match_thresh" : 0.9, 
  "min_box_area" : 10, # pixels^2 (w x h)
  "frame_rate" : 15 # FPS
}

# For SORT object tracking
# SORT = {
#   "max_age": 5,
#   "min_hits": 2,
#   "iou_thres": 0.2,
# }


# Jetboard Physical Attributes
#---------------------------------------------------------------------------------------------------
CAMERA = {
  "resolution": [3840, 2160], # [w, h] pixels
  "FOV": [50, 100], # Vertical, Horizontal FOV in degrees
  "elevation": 0.84, # meters 
  "model": "IMX179",
}

JETBOARD = {  
  "FC_to_cam_dist" : 0.72, # meters
  "mass_kg" : 40, # kg
  "length_m" : 2, # meters
  "width_m" : 0.83, # meters
  "height_m" : 1, # meters
  "COG_to_nozzel": 1 # meters
}

# Default YOLO categories (COCO dataset):
# CATEGORIES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
#             "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
#             "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
#             "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#             "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#             "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
#             "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
#             "hair drier", "toothbrush"]
