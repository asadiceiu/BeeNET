
import glob
from time import time
import os
import cv2
import torch
from ultralytics import YOLO

class BeeDetectorYolo:
    def __init__(self, yolo_path):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO(yolo_path)
        self.yolo.to(self.device)
    
    def process_yolo_results(self, result, class_names, conf_thresh=0.5):
        """
        processes the yolo results and returns the frame with the bounding boxes.
        results: list of results from yolo inference
        frame: np.array
        class_names: dict
        draw_bounding_box: bool
        return: frame, np.array
        """
        box_list = {}
        for i in range(len(class_names)):
            box_list[class_names[i]] = []
        if result is not None and len(result.boxes.cpu().numpy().data)>0:
            # class_names = {0: 'fly', 1: 'tag'}
            data = result.boxes.cpu().numpy().data
            boxes = data[:, 0:4].astype(int)
            confs = data[:, 4]
            class_ids = data[:, 5].astype(int)
            # for each box, draw the bounding box and label
            for i, box in enumerate(boxes):
                if confs[i] > conf_thresh and class_ids[i] in class_names:
                    box = [int(x) for x in box] #: x1, y1, x2, y2 = box
                    # add confidence to the box
                    box.append(round(confs[i],2))
                    box_list[class_names[class_ids[i]]].append(box)
        return box_list
    
    def detect_insects_yolo(self, video_path, conf_threshold=0.65):
        """
        Process the dronefly video
        video_path: str
        yolo: YOLO
        class_names: dict
        output_path: str
        output_video_path: str
        output_video_fps: int
        """
        extension = video_path.split('.')[-1]
        detected_bee_video_path = video_path.replace("."+extension, "_yolo_detections.mp4")
        detected_bee_csv_path = video_path.replace("."+extension, "_yolo_detections.csv") 
        if os.path.exists(detected_bee_video_path) and os.path.exists(detected_bee_csv_path):
            print(f"Files already exist: {detected_bee_video_path}, {detected_bee_csv_path}")
            return 0
        
        class_names = {0: 'bee'}
        # load the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return
        extension = video_path.split('.')[-1]
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   
        csv_string = "frame_id, center_x, center_y, box_x1, box_y1, box_x2, box_y2, confidence\n"
        
        writer = cv2.VideoWriter(detected_bee_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))
        t1 = time()
        # process the video frame by frame
        frame_counter = 0
        while cap.isOpened():
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                break
            
            # process the frame
            results = self.yolo(frame, verbose=False)
            frame_counter = 0
            for result in results:
                box_list = self.process_yolo_results(result, class_names, conf_thresh=conf_threshold)
                for class_name, boxes in box_list.items():
                    for box in boxes:
                        x1, y1, x2, y2, conf = box
                        center = (int((x1+x2)/2), int((y1+y2)/2))
                        csv_string += f"{frame_no},{center[0]},{center[1]},{box[0]},{box[1]},{box[2]},{box[3]},{conf:0.2f}\n"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} {conf:0.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # write the frame to the output video
                t = time() - t1
                fps = frame_no / t
                print(f"Processed frame: {frame_no:5d}/{n_frames:5d} ({frame_no/n_frames:.2%}), fps={fps:0.2f}", end='\r')
                writer.write(frame)
            frame_counter += 1
        # release the video capture and video writer objects
        print()
        cap.release()
        writer.release()
        with open(detected_bee_csv_path, 'w') as f:
            f.write(csv_string)
        print(f"Processing Done. Took {time()-t1:.2f} seconds")
        return time()-t1