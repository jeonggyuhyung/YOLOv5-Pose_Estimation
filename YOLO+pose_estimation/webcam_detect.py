import argparse
import chainer

import cv2
import torch
from numpy import random
import re
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.pose_detector import PoseDetector, draw_person_pose
from utils.general import check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging
from utils.torch_utils import select_device
from multiprocessing import Process, Queue

# producer
def setup(q1, q2, weights='yolov5s.pt', conf=0.4,
                   img=640, iou=0.5, device='', view='store_true', save='store_true',
                   classes='+', agnostic='store_true', augment='store_true', update='store_true'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=img, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=conf, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=iou, help='IOU threshold for NMS')
    parser.add_argument('--device', default=device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action=view, help='display results')
    parser.add_argument('--save-txt', action=save, help='save results to *.txt')
    parser.add_argument('--classes', nargs=classes, type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action=agnostic, help='class-agnostic NMS')
    parser.add_argument('--augment', action=augment, help='augmented inference')
    parser.add_argument('--update', action=update, help='update all models')
    opt = parser.parse_args()

    weights, view_img, save_txt, imgsz, argment, conf, iou, classes, agnostic_nms = \
        opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.augment, opt.conf_thres, opt.iou_thres, opt.classes,\
        opt.agnostic_nms

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    while True:
        if not q1.empty():
            frame = q1.get()
            yolo(frame, q2, imgsz, device, half, model, augment, conf, iou, classes, agnostic_nms, names, colors)


def yolo(q1, q2, imgsz, device, half, model, augment, conf, iou, classes, agnostic_nms, names, colors):
    source = q1
    dataset = LoadImages(source, img_size=imgsz)
    for img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf, iou, classes=classes, agnostic=agnostic_nms)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                if len(det) >= 2:
                    det_num = []
                    for i in det:
                        det_num.append((i[2] + i[3]) - (i[0] + i[1]))
                    largest = det_num[0]
                    for i in range(len(det_num)):
                        if det_num[i] > largest:
                            largest = det_num[i]
                    det = [det[det_num.index(largest)]]

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    num = []
                    f = str(xyxy).split(", ")
                    for string in f:
                        axes = re.findall("tensor\((.*).", string)
                        if axes != []:
                            num.append(int(axes[0]))
                    outq = num, label, colors[int(cls)], names[int(cls)]
                    q2.put(outq)
        break

def pose_estimation(q1, q3, q4):
    chainer.config.enable_backprop = False
    chainer.config.train = False
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=0)
    while True:
        if not q4.empty() and not q1.empty():
            num = q4.get()
            img = q1.get()[num[1]:num[3], num[0]:num[2]]
            poses, _ = pose_detector(img)
            outq3 = poses, num
            q3.put(outq3)

def webcam_producer(q1):
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        q1.put(frame)
    capture.release()

def webcam_out(q1, q2, q3, q4):
    score = 0
    Before_flag = False
    while True:
        if not q1.empty():
            frame = q1.get()
            if not q2.empty():
                q2num, label, colors, name = q2.get()
                q4.put(q2num)
            if not q3.empty():
                poses, num = q3.get()
            try:
                frame2 = frame[num[1]:num[3], num[0]:num[2]].copy()
                canvas, score, Before_flag, status = draw_person_pose(frame2, poses, score, Before_flag, name)
                frame[num[1]:num[3], num[0]:num[2]] = canvas
            except:
                pass
            try:
                plot_one_box(q2num, frame, label=label, color=colors, line_thickness=3)
            except:
                pass
            cv2.imshow("webcam", frame)
        if cv2.waitKey(1) > 0: break

if __name__ == "__main__":
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()

    p1 = Process(name="webcam_producer", target=webcam_producer, args=(q1, ), daemon=True)
    p2 = Process(name="yolo", target=setup, args=(q1, q2,'weights/health_best.pt', 0.7, ), daemon=True)
    p3 = Process(name="webcam_out", target=webcam_out, args=(q1, q2, q3, q4, ), daemon=True)
    p4 = Process(name='pose_estimation', target=pose_estimation, args=(q1, q3, q4, ), daemon=True)
    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()