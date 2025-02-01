# main script

import os
import sys
import platform
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
import time
import numpy as np
import torch
from crown import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

# Ensure that the correct path separator is used for pathlib
Path = Path if platform.system() == "Linux" else Path

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.general import Profile, colorstr, cv2, increment_path, strip_optimizer, print_args
import time

num_of_wins = 1
left_rock_detected = False
right_rock_detected = False
left_countdown_start_time = None
right_countdown_start_time = None
countdown_duration = 5  # Countdown duration in seconds

left_player_score = 0
right_player_score = 0

is_left_clown = False
is_right_clown = False

who_won = ""


def start_round(model, dataset, conf_thres, iou_thres):
    print("Waiting for both players to show 'rock'...")
    left_ready = False
    right_ready = False

    while not (left_ready and right_ready):
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)

            for det in pred:
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        x_min, x_max = xyxy[0], xyxy[2]
                        frame_width = im0s[0].shape[1]
                        if x_min > frame_width / 2:  # Right side
                            if int(cls) == 1:  # Class 1 is "rock"
                                right_ready = True
                        elif x_min <= frame_width / 2:  # Left side
                            if int(cls) == 1:  # Class 1 is "rock"
                                left_ready = True

            if left_ready and right_ready:
                print("Both players are ready!")
                break

    print("Starting in 3 seconds...")
    time.sleep(3)
    print("Game started!")

def display_num_of_wins(im0s):
    if not isinstance(im0s, np.ndarray):
        im0s = np.array(im0s)  # Convert to NumPy array if needed

    # Adjust font scale and position for visibility
    font_scale = 0.5  # Reasonable size for text
    thickness = 2  # Line thickness for better readability
    position = (50, 20)  # Text position on the frame

    cv2.putText(
        im0s,  # Frame to draw on
        f"Number of Wins: {num_of_wins}",  # Text to display
        position,  # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        font_scale,  # Font scale
        (0, 0, 0),  # Color (Green)
        thickness,  # Thickness
        cv2.LINE_AA  # Line type
    )


def display_countdown(im0, start_time, countdown_duration=5000):
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = max(0, countdown_duration - elapsed_time)
    position = (150, 50)

    if remaining_time > 0:
        countdown_text = f"Countdown: {int(remaining_time)}"
        cv2.putText(
            im0,
            countdown_text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return remaining_time


def detect_rock(cnt,x_min,width,cls,right_box_found,left_box_found):
    if x_min > width / 2 and not right_box_found:
        right_hand = int(cls)
        if cls == 1:
            right_hand = 'rock'
        elif cls == 2:
            right_hand = 'scissors'
        print("Right class: ", right_hand)
        right_box_found = True
    elif x_min <= width / 2 and not left_box_found:
        left_hand = int(cls)
        if cls == 1:
            left_hand = 'rock'
        left_box_found = True
@smart_inference_mode()
def webcam_inference(
        weights=ROOT / 'weights/best.pt',  # model.pt path(s)
        source=0,  # file/dir/URL/glob, 0 for webcam
        imgsz=(512, 512),  # inference size (pixels)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        data=ROOT / 'Rock-Paper-Scissor-4/data.yaml',  # path to data.yaml
):
    source = str(source)
    webcam = source.isnumeric() or source.endswith('.streams')
    left_box_found = False
    right_box_found = False
    game_start = False
    game_finish = False
    left_hand = None
    right_hand = None
    time_to_lose = 5
    global num_of_wins

    device = select_device("")
    model = DetectMultiBackend(weights, device=device, dnn=False, data=ROOT / 'data', fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgz = check_img_size(imgsz, s=stride)

    batch_size = 1
    view_img = check_imshow(warn=True)
    dataset = LoadStreams(source, img_size=imgz, stride=stride, auto=pt, vid_stride=1)
    batch_size = len(dataset)

    model.warmup(imgsz=(1 if pt or model.triton else batch_size, 3, *imgz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    width, length = 0, 0
    start_time = time.time()
    left_time_to_lose = time_to_lose
    right_time_to_lose = time_to_lose
    global is_left_clown , is_right_clown , who_won
    left_cheat = False
    right_cheat = False

    # start_round(model, dataset, conf_thres=conf_thres, iou_thres=iou_thres)

    game_has_started = False

    for path, im, im0s, vid_cap, s in dataset:

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, chunks=im.shape[0], dim=0)

        with dt[1]:
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=False, visualize=False).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=False, visualize=False).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=False, visualize=False)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    x_min, y_min, x_max, y_max = xyxy

                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if x_min > width / 2 and not right_box_found:
                        right_hand = int(cls)
                        if cls == 0:
                            right_hand = 'paper'
                        elif cls == 1:
                            right_hand = 'rock'
                        elif cls == 2:
                            right_hand = 'scissors'
                        print("Right class: ", right_hand)
                        right_box_found = True
                    elif x_min <= width / 2 and not left_box_found:
                        left_hand = int(cls)
                        if cls == 0:
                            left_hand = 'paper'
                        elif cls == 1:
                            left_hand = 'rock'
                        elif cls == 2:
                            left_hand = 'scissors'
                        print("Left class: ", left_hand)
                        left_box_found = True

                    if left_box_found and right_box_found:
                        if right_hand == left_hand:
                            if game_has_started:
                                print("DRAW!!!")
                                left_hand = None
                                right_hand = None
                            left_box_found = False
                            right_box_found = False
                            # cv2.destroyAllWindows()
                            # return length, width, x_min, y_min, x_max, y_max, confidence, cls
                        if (right_hand, left_hand) in [('paper', 'rock'), ('scissors', 'paper'), ('rock', 'scissors')]:
                            if game_has_started:
                                print("Right hand wins!!!!")
                                left_hand = None
                                right_hand = None
                            left_box_found = False
                            right_box_found = False
                            # cv2.destroyAllWindows()
                            # return length, width, x_min, y_min, x_max, y_max, confidence, cls
                        else:
                            if game_has_started:
                                print("Left hand wins!!!!")
                                left_hand = None
                                right_hand = None
                            left_box_found = False
                            right_box_found = False
                            # cv2.destroyAllWindows()
                            # return length, width, x_min, y_min, x_max, y_max, confidence,
                        game_has_started = False

            if(right_hand == left_hand  ==  'rock' and not game_has_started):
                game_has_started = True
                print("ROCK ROCK")
                # Start the countdown with movement detection
                left_cheat , right_cheat = start_countdown(game_has_started, dataset, view_img, p, windows)
                'TODO : ADD WAIT'

                winner = 0
                if not left_cheat and not right_cheat:
                   left_cheat, right_cheat, winner = check_cheating_and_decide_winner(dataset, model, conf_thres, iou_thres, left_hand, right_hand, view_img, p, windows , seen , dt, webcam, names)
                update_score(left_pl_cheat= left_cheat, right_pl_cheat = right_cheat, winner = winner)
            
            if not is_left_clown:
                is_left_clown = left_cheat

            if not is_right_clown:
                is_right_clown = right_cheat
                       
            im0 = annotator.result()
            display_player_scores(im0)
            display_num_of_wins(im0)
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                length = im0.shape[0]
                width = im0.shape[1]
                cv2.line(im0, (int(width / 2), 0), (int(width / 2), length), (0, 255, 0), 3)
                im0 = show_clown(im0 , is_left_clown , is_right_clown , width)
                if (left_player_score >= num_of_wins):
                    who_won = "left"
                    show_crown(im0 , who_won , width)
                elif (right_player_score >= num_of_wins):
                    who_won = 'right'
                    show_crown(im0 , who_won , width)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
        
        if who_won != "":
            break
    cv2.waitKey(3000)
    cv2.destroyAllWindows()



    
def restart_game():
    print('GAME RESTARTED')


def main_loop(weights, data, source):
    while True:
        webcam_inference(weights=weights, data=data, source=source)
        end_game()
        # key = input(print('press r to restart the game and press q to quit the game'))

        # if key == 'r':
        #     restart_game()
        # if key == 'q':
        #     print('GAME OVER')
        #     break
        break


def detect_movement_in_half(prev_frame, curr_frame, half, threshold=100):
    """
    Detects movement in the specified half of the frame.

    Args:
        prev_frame (ndarray): The previous grayscale frame.
        curr_frame (ndarray): The current grayscale frame.
        half (str): 'left' or 'right', indicating which half to analyze.
        threshold (int): The threshold for detecting movement.

    Returns:
        bool: True if movement is detected in the specified half, False otherwise.
    """
    height, width = prev_frame.shape
    if half == 'left':
        prev_half = prev_frame[:, :width // 2]
        curr_half = curr_frame[:, :width // 2]
    elif half == 'right':
        prev_half = prev_frame[:, width // 2:]
        curr_half = curr_frame[:, width // 2:]
    else:
        raise ValueError("Invalid half specified. Use 'left' or 'right'.")

    diff = cv2.absdiff(prev_half, curr_half)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)


    # print("tresh sum: ", np.sum(thresh))

    movement_detected = np.sum(thresh) > 10000  # Tune this value for sensitivity
    return movement_detected


def start_countdown(game_has_started, dataset, view_img, p, windows):
    """
    Starts the countdown and detects movement for each player during the countdown.

    Args:
        game_has_started (bool): Whether the game has started.
        dataset: The YOLO dataset stream.
        view_img (bool): Whether to display the webcam feed.
        p (str): Path or identifier for the current frame.
        windows (list): List of currently open window identifiers.

    Returns:
        None
    """
    if not game_has_started:
        return

    # Get the first frame from dataset
    for path, im, im0s, vid_cap, s in dataset:
        prev_frame = cv2.cvtColor(im0s[0], cv2.COLOR_BGR2GRAY)  # Convert first frame to grayscale
        break

    player_movement = {'left': False, 'right': False}
    countdown = 3  # Start countdown from 3
    start_time = time.time()  # Track time for countdown updates

    while countdown >= 0:
        frame_updated = False  # Ensure the loop doesn't repeat indefinitely

        for path, im, im0s, vid_cap, s in dataset:
            frame = im0s[0]  # Use the current frame from YOLO stream
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect movement in each half
            if detect_movement_in_half(prev_frame, curr_frame, 'left'):
                player_movement['left'] = True
            if detect_movement_in_half(prev_frame, curr_frame, 'right'):
                player_movement['right'] = True

            prev_frame = curr_frame  # Update previous frame

            # Update countdown every 1 second
            if time.time() - start_time >= 1:
                start_time = time.time()  # Reset timer
                countdown -= 1  # Decrease countdown
                frame_updated = True  # Ensure we exit the loop after updating

            # Clear previous countdown and display the new countdown number
            cv2.rectangle(frame, (80, 50), (200, 120), (0, 0, 0), -1)  # Black box to erase old number
            if countdown > 0:
                cv2.putText(frame, f"{countdown}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

            # Display the frame
            display_frame(frame, Annotator(frame), lambda x: None, view_img, p, windows)

            # Allow multiple frame updates per second for smooth movement detection
            cv2.waitKey(1000)

            # Exit dataset loop early to allow the countdown to progress
            if frame_updated:
                break

        # Exit while loop when countdown reaches 0
        if countdown == 0:
            break

    left_cheat =  False
    right_cheat = False
    # Final movement validation
    if not player_movement['left'] :
        print("Cheating detected! Left player did not move.")
        left_cheat = True
    elif not player_movement['right']:
        print("Cheating detected! Right player did not move.")
        right_cheat = True
    else:
        print("No cheating detected. Game starts now!")
    return left_cheat, right_cheat



def display_frame(im0, annotator, display_wins_func, view_img, p, windows):
    """
    Displays the annotated frame with additional elements like number of wins and a dividing line.

    Args:
        im0 (ndarray): The current frame to display.
        annotator (Annotator): YOLOv5 annotator for adding labels and boxes.
        display_wins_func (callable): A function to display additional information like number of wins.
        view_img (bool): Whether to display the image in a window.
        p (str): Path or identifier for the current frame.
        windows (list): List of currently open window identifiers.

    Returns:
        None
    """
    im0 = annotator.result()  # Get the annotated frame
    display_wins_func(im0)  # Display number of wins or other details

    if view_img:
        # Create a window for the frame if not already created
        if platform.system() == "Linux" and p not in windows:
            windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # Allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])  # Resize window to match the frame dimensions

        # Get frame dimensions
        length = im0.shape[0]
        width = im0.shape[1]

        # Draw a dividing line in the middle of the frame
        cv2.line(im0, (int(width / 2), 0), (int(width / 2), length), (0, 255, 0), 3)

        # Display the frame
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # Wait 1 millisecond




def check_cheating_and_decide_winner(dataset, model, conf_thres, iou_thres, left_hand_initial, right_hand_initial, view_img, p, windows, seen , dt, webcam, names):
    """
    Check for cheating and decide the winner after 3 seconds.

    Args:
        dataset: The YOLO dataset stream.
        model: The YOLO detection model.
        conf_thres: Confidence threshold for detections.
        iou_thres: IOU threshold for non-max suppression.
        left_hand_initial: Initial hand decision for the left player.
        right_hand_initial: Initial hand decision for the right player.
        view_img (bool): Whether to display the webcam feed.
        p (str): Path or identifier for the current frame.
        windows (list): List of currently open window identifiers.

    Returns:
        None
    """
    print("Checking for cheating...")
    start_time = time.time()
    cheating_detected = False

    # Initialize variables for final hand decisions

    left_hand_initial = None
    right_hand_initial = None


    left_hand_final = left_hand_initial
    right_hand_final = right_hand_initial

    global is_left_clown , is_right_clown


    for path, im, im0s, vid_cap, s in dataset:
        if time.time() - start_time >= 15:
            break

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, chunks=im.shape[0], dim=0)

        with dt[1]:
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=False, visualize=False).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=False, visualize=False).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=False, visualize=False)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    x_min, y_min, x_max, y_max = xyxy

                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    frame_length = im0.shape[0]
                    frame_width = im0.shape[1]

                    # Update hand decisions based on bounding box location
                    if x_min > frame_width / 2:  # Right side
                        right_hand_final = int(cls)
                        if cls == 0:
                            right_hand_final = 'paper'
                        elif cls == 1:
                            right_hand_final = 'rock'
                        elif cls == 2:
                            right_hand_final = 'scissors'
                    elif x_min <= frame_width / 2:  # Left side
                        left_hand_final = int(cls)
                        if cls == 0:
                            left_hand_final = 'paper'
                        elif cls == 1:
                            left_hand_final = 'rock'
                        elif cls == 2:
                            left_hand_final = 'scissors'
                    if left_hand_initial is None or right_hand_initial is None:
                        left_hand_initial = left_hand_final
                        right_hand_initial = right_hand_final



        # Display the frame with annotations
        im0 = annotator.result()
        display_num_of_wins(im0)  # Display additional info
        if view_img:
            if platform.system() == "Linux" and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # Allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            length = im0.shape[0]
            width = im0.shape[1]
            # Draw dividing line
            cv2.line(im0, (int(width / 2), 0), (int(width / 2), length), (0, 255, 0), 3)
            show_clown(im0 , is_left_clown , is_right_clown , width)
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

    print("final_left:", left_hand_final)
    print("initial_left:", left_hand_initial)
    print("final_right:", right_hand_final)
    print("initial_right:", right_hand_initial)

    left_cheat = False
    right_cheat = False
    winner = 0
    # Check for changes in hand decisions
    if left_hand_final != left_hand_initial:
        cheating_detected = True
        print("Cheating detected! Left player changed decision")
        left_cheat = True

    if right_hand_final != right_hand_initial:
        cheating_detected = True
        print("Cheating detected! Right player changed decision")
        right_cheat = True

    if cheating_detected:
        print("Cheating detected! One of the players changed their decision.")
    else:
        print("No cheating detected. Deciding winner...")
        winner = decide_winner(left_hand_initial, right_hand_initial)
    return left_cheat, right_cheat, winner



def decide_winner(left_hand, right_hand):
    """
    تعیین برنده بر اساس انتخاب بازیکنان
    """

    if left_hand == right_hand:
        print("It's a DRAW!")
        return 0
    elif (right_hand, left_hand) in [('paper', 'rock'), ('scissors', 'paper'), ('rock', 'scissors')]:  # سنگ کاغذ قیچی
        print("Right player wins!")
        return 1
    else:
        print("Left player wins!")
        return -1

def update_score(left_pl_cheat, right_pl_cheat, winner):
    global left_player_score, right_player_score  
    if left_pl_cheat:
        left_player_score -= 1
    if right_pl_cheat:
        right_player_score -= 1

    if left_pl_cheat or right_pl_cheat:
        return
    if winner == 1:
        right_player_score += 1
    elif winner == -1:
        left_player_score += 1
    return


def display_player_scores(im0s):
    """
    Displays the scores for the left and right players in their respective halves of the frame.

    Args:
        im0s (ndarray): The frame to draw on.
        left_player_score (int): The score of the left player.
        right_player_score (int): The score of the right player.

    Returns:
        None
    """
    global left_player_score, right_player_score  # Declare global variables

    if not isinstance(im0s, np.ndarray):
        im0s = np.array(im0s)  # Convert to NumPy array if needed

    # Frame dimensions
    height, width = im0s.shape[:2]

    # Adjust font scale and thickness for visibility
    font_scale = 0.7
    thickness = 2

    # Define text positions for left and right players
    left_position = (50, 50)  # Top-left corner for the left player
    right_position = (width - 200, 50)  # Top-right corner for the right player

    # Display the left player's score
    cv2.putText(
        im0s,  # Frame to draw on
        f"Left Player: {left_player_score}",  # Text to display
        left_position,  # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        font_scale,  # Font scale
        (255, 0, 0),  # Text color (Blue)
        thickness,  # Line thickness
        cv2.LINE_AA  # Line type
    )

    # Display the right player's score
    cv2.putText(
        im0s,  # Frame to draw on
        f"Right Player: {right_player_score}",  # Text to display
        right_position,  # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        font_scale,  # Font scale
        (0, 0, 255),  # Text color (Red)
        thickness,  # Line thickness
        cv2.LINE_AA  # Line type
    )


def end_game():
    global who_won , is_left_clown , is_right_clown 
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    start_time = time.time()
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break
        width = frame.shape[1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_center = x + w // 2  
            if face_center < width // 2:  
                if who_won == 'left':
                    overlay_crown(frame, crown, x, y, w, h)  
                if is_left_clown:
                    overlay_mask(frame, mask, x, y, w, h)
            else:  
                if who_won == 'right':
                    overlay_crown(frame, crown, x, y, w, h) 
                if is_right_clown:
                    overlay_mask(frame, mask, x, y, w, h)


        # Display the resulting frame
        cv2.imshow('Mask Overlay', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Set your desired parameters
    weights_path = ROOT / 'runs/train/exp/weights/best.pt'
    data_path = ROOT / 'Rock-Paper-Scissor-4/data.yaml'
    source = 0  # Change this to the appropriate source, e.g., path to video file or URL

    # Run the webcam_inference function
    main_loop(weights=weights_path, data=data_path, source=source)