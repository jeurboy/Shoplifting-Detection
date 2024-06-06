import os
import queue
import time
import threading
import cv2
import numpy as np

from keras.models import model_from_json
from termcolor import colored

import Alert
from data_pip_shoplifting import Shoplifting_Live

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='error', category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# from object_detection.utils import label_map_util
# from object_detection.utils import config_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder

def get_abuse_model_and_weight_json():
    # read model json
    # load json and create model
    weight_abuse = r"E:\FINAL_PROJECT_DATA\2021\Silence_Vision__EDS_Demo\Event_detection\Event_weight\Abuse\weights_at_epoch_3_28_7_21_round2.h5"
    json_path = r"E:\FINAL_PROJECT_DATA\2021\Yolov5_DeepSort_Pytorch-master\Yolov5_DeepSort_Pytorch-master\EMS\model_Abuse_at_epoch_3_28_7_21_round2.json"
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    abuse_model = model_from_json(loaded_model_json)
    # load weights into new model
    abuse_model.load_weights(weight_abuse)
    print("Loaded EMS model,weight_steals from disk")
    return abuse_model


# ABUSE_MODEL = get_abuse_model_and_weight_json()
q = queue.Queue(maxsize=3000)
frame_set = []

model_type = "rgb"  # rgb / opt

Frame_set_to_check = []
Frame_INDEX = 0
lock = threading.Lock()
Email_alert_flag = False
email_alert = Alert.Email_Alert()
shoplifting_SYS = Shoplifting_Live(model_type=model_type)
W = 0
H = 0
src_main_dir_path = r"result/shoplifter/"


def Receive():
    global W, H
    dataset_path = "/Users/pornprasithmahasith/Documents/workspace/video-classifier-cnn-lstm/dataset"

    # print("start Receive")
    # rtsp://SIMCAM:2K93AG@192.168.1.2/live
    # video_cap_ip = 'rtsp://SIMCAM:S6BG9J@192.168.1.20/live'
    # video_cap_ip = r'rtsp://barloupo@gmail.com:ziggy2525!@192.168.1.9:554/stream2'

    # Normal
    # video_cap_ip = dataset_path+"/test/normal/2024-05-14_12-37-17.mp4"
    # video_cap_ip = dataset_path+"/test/normal/2024-04-03_14-40-14.mp4"  # Lotus store
    video_cap_ip = dataset_path+"/test/normal/2024-05-13_14-57-39.mp4"  # Pop home

    # Steal
    # video_cap_ip = dataset_path+"/train/shoplifter/2024-05-14_12-56-34_หยิบใส่เสื้อคลุม_ชั้นวาง_ซ้าย.mp4"

    cap = cv2.VideoCapture(video_cap_ip)
    # cap.set(3, 640)
    # cap.set(4, 480)
    W = int(cap.get(3))
    H = int(cap.get(4))
    # print("H={}\nW={}".format(H,W))
    ret, frame = cap.read()
    # print(colored(ret, 'green'))
    q.put(frame)
    # while cap.isOpened():
    while ret:
        ret, frame = cap.read()
        # cv2.imshow("video", frame)
        q.put(frame)


def Display():
    global Frame_set_to_check, Frame_INDEX
    print(colored('Start Displaying', 'blue'))

    while True:
        if q.empty() != True:
            frame = q.get()
            if isinstance(frame, type(None)):
                print("[-][-] NoneType frame {}".format(type(frame)))
                break

            frame_set.append(frame.copy())

            if len(frame_set) == 149:
                Frame_set_to_check = frame_set.copy()

                Predict()
                time.sleep(1)
                frame_set.clear()

            # cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def Predict():
    global Frame_set_to_check, Frame_INDEX
    # ems = EMS_Live()
    with lock:

        if model_type == "opt":
            # RGB + OPT NET
            shoplifting_SYS.build_shoplifting_net_models()
        else:
            # RGB NET ONLY
            shoplifting_SYS.get_new_model_shoplifting_net()

        print(colored("model loaded complete", 'green'))

        Frame_set_to_check_np = np.array(Frame_set_to_check.copy())

        Frame_set = shoplifting_SYS.make_frame_set_format(
            Frame_set_to_check_np)

        reports = shoplifting_SYS.run_Shoplifting_frames_check_live_demo_2_version(
            Frame_set, Frame_INDEX)

        Frame_INDEX = Frame_INDEX + 1

        Bag = reports[0]
        Clotes = reports[1]
        Normal = reports[2]
        state = reports[3]
        # todo event_index maybe paas a dict
        event_index = reports[4]
        # print("event_index {}".format(event_index))
        ##

        if (state):
            print(colored("---------------------", 'red'))
            print(colored('Found shopLifting event', 'red'))
            print(
                colored(f"Bag: {Bag}\nClotes: {Clotes}\nNormal: {Normal}", 'red'))
            # print(colored(f"reports {reports[0], reports[1],reports[2]}", 'green'))
            print(
                colored(f"Test number:{Frame_INDEX-1}\n---------------------\n", 'red'))

            prob = [Bag, Clotes, Normal]

            found_fall_video_path = shoplifting_SYS.save_frame_set_after_pred_live_demo(src_main_dir_path,
                                                                                        Frame_set_to_check,
                                                                                        Frame_INDEX-1, prob,
                                                                                        0, W, H)

            if Email_alert_flag:
                file_name = found_fall_video_path.split("\\")[-1]
                print(f"path = to email{found_fall_video_path}")
                print(f"file name: {file_name}")
                absulutefilepath = found_fall_video_path
                email_alert.send_email_alert(email_alert.user_email_address3, file_name,
                                             absulutefilepath)

        else:
            print(colored("---------------------", 'green'))
            print(colored("Normal event", 'green'))
            print(
                colored(f"Bag: {Bag}\nClotes: {Clotes}\nNormal: {Normal}", 'green'))
            print(
                colored(f"Test number:{Frame_INDEX - 1}\n---------------------\n", 'green'))
            Frame_set_to_check.clear()

        # lock.release()
        time.sleep(1)


if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)

    p1.start()
    p2.start()
