# This script contains the BeeDetectorYolo class for detecting bees in a video using YOLO.
# Created by: Asaduz Zaman
# Created on: 10 August 2024
# Updated on: 10:00 PM 10 August 2024

import os
import cv2
from time import time
from ultralytics import YOLO

class BeeDetectorYolo:
    """
    Class for detecting bees in a video using YOLO.
    Attributes:
        yolo: YOLO model for bee detection.
        device: Device to run the YOLO model on (default: 'cpu').

    Methods:
        process_yolo_results(result, class_names, conf_thresh=0.5):
            Process YOLO results and return a list of bounding boxes.
        detect_insects_yolo(video_path, conf_threshold=0.65):
            Detect insects in a video and save the results as a processed video and CSV.
        _get_output_paths(video_path):
            Generate paths for the output video and CSV files.
        _init_video_processing(video_path, output_video_path):
            Initialize video capture and writer.
        _process_frame(frame, frame_no, results, class_names, conf_threshold):
            Process a single frame and return the CSV data for detections.
        _finalize_processing(cap, writer, csv_path, csv_data, start_time):
            Finalize video processing by saving the CSV and releasing resources.
    """
    def __init__(self, yolo_path, device='cpu'):
        """Initialize the YOLO model for bee detection."""
        self.device = device
        self.yolo = YOLO(yolo_path)
        self.yolo.to(self.device)
    
    def process_yolo_results(self, result, class_names, conf_thresh=0.5):
        """
        Process YOLO results and return a list of bounding boxes.
        Args:
            result: YOLO inference result.
            class_names: Dictionary mapping class IDs to class names.
            conf_thresh: Confidence threshold for filtering detections.
        Returns:
            box_list: Dictionary containing class-wise bounding boxes and confidences.
        """
        box_list = {name: [] for name in class_names.values()}
        if result is not None and len(result.boxes.cpu().numpy().data) > 0:
            data = result.boxes.cpu().numpy().data
            boxes = data[:, 0:4].astype(int)
            confs = data[:, 4]
            class_ids = data[:, 5].astype(int)

            for i, box in enumerate(boxes):
                if confs[i] > conf_thresh and class_ids[i] in class_names:
                    box = list(map(int, box))
                    box.append(round(confs[i], 2))
                    box_list[class_names[class_ids[i]]].append(box)
        return box_list
    
    def detect_insects_yolo(self, video_path, conf_threshold=0.65):
        """
        Detect insects in a video and save the results as a processed video and CSV.
        Args:
            video_path: Path to the input video file.
            conf_threshold: Confidence threshold for detections.
        Returns:
            Time taken for processing the video.
        """
        detected_bee_video_path, detected_bee_csv_path = self._get_output_paths(video_path)
        if os.path.exists(detected_bee_video_path) and os.path.exists(detected_bee_csv_path):
            print(f"Files already exist: {detected_bee_video_path}, {detected_bee_csv_path}")
            return 0

        class_names = {0: 'bee'}
        cap, writer, n_frames = self._init_video_processing(video_path, detected_bee_video_path)

        if not cap:
            return

        csv_data = ["frame_id,center_x,center_y,box_x1,box_y1,box_x2,box_y2,confidence\n"]
        t1 = time()

        while cap.isOpened():
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                break

            results = self.yolo(frame, verbose=False)
            csv_data.extend(self._process_frame(frame, frame_no, results, class_names, conf_threshold))

            t = time() - t1
            fps = frame_no / t
            print(f"Processed frame: {frame_no:5d}/{n_frames:5d} ({frame_no/n_frames:.2%}), fps={fps:0.2f}", end='\r')

            writer.write(frame)

        print()
        self._finalize_processing(cap, writer, detected_bee_csv_path, csv_data, t1)
        return time() - t1

    def _get_output_paths(self, video_path):
        """Generate paths for the output video and CSV files."""
        extension = video_path.split('.')[-1]
        detected_bee_video_path = video_path.replace(f".{extension}", "_yolo_detections.mp4")
        detected_bee_csv_path = video_path.replace(f".{extension}", "_yolo_detections.csv")
        return detected_bee_video_path, detected_bee_csv_path

    def _init_video_processing(self, video_path, output_video_path):
        """Initialize video capture and writer."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return None, None, 0

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                                 (int(cap.get(3)), int(cap.get(4))))
        return cap, writer, n_frames

    def _process_frame(self, frame, frame_no, results, class_names, conf_threshold):
        """Process a single frame and return the CSV data for detections."""
        csv_rows = []
        for result in results:
            box_list = self.process_yolo_results(result, class_names, conf_thresh=conf_threshold)
            for class_name, boxes in box_list.items():
                for box in boxes:
                    x1, y1, x2, y2, conf = box
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    csv_rows.append(f"{frame_no},{center[0]},{center[1]},{x1},{y1},{x2},{y2},{conf:.2f}\n")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return csv_rows

    def _finalize_processing(self, cap, writer, csv_path, csv_data, start_time):
        """Finalize video processing by saving the CSV and releasing resources."""
        cap.release()
        writer.release()
        with open(csv_path, 'w') as f:
            f.writelines(csv_data)
        print(f"Processing Done. Took {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    print("Note: This script is not meant to be run directly.")
    print("Please import this script in another script and use the BeeDetectorYolo class.")
    print("=========================================")
    print(BeeDetectorYolo.__doc__)
    print("=========================================")
    print("Example usage:")
    print("detector = BeeDetectorYolo('yolov5s.pt')")
    print("detector.detect_insects_yolo('video.mp4')")
    print("This will detect bees in the video and save the processed video and CSV.")