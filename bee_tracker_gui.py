# Create a GUI for the bee tracker
import math
import os
import shutil
import tkinter as tk

from tkinter import filedialog, messagebox
import cv2
import numpy as np

from bee_detector_yolo import BeeDetectorYolo
from kalman_tracker import KalmanFilterTracker


class BeeTrackerGUI():
    def __init__(self, root):
        self.root = root
        self.root.title('Bee Tracking GUI')
        self.root.geometry('1200x600')
        self.bee_detector = BeeDetectorYolo('models/stinglessbee-yolov8-s-best.pt')
        self.bee_tracker = KalmanFilterTracker()
        self.initUI()

    def initUI(self):
        self.left_frame = tk.Frame(self.root, width=600, height=600)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.top_left_control_frame = tk.Frame(self.left_frame)
        self.top_left_control_frame.pack(pady=10, padx=10)
        self.load_video_button = tk.Button(self.top_left_control_frame, text='Load Video', command=self.load_video)
        self.detect_bee_button = tk.Button(self.top_left_control_frame, text='Detect Bees', command=self.detect_bees)
        self.assign_zone_button = tk.Button(self.top_left_control_frame, text='Assign Zone', command=self.assign_zone)
        self.load_video_button.grid(row=0, column=0, padx=5)
        self.detect_bee_button.grid(row=0, column=1, padx=5)
        self.assign_zone_button.grid(row=0, column=2, padx=5)

        # create a canvas to display the video
        self.video_canvas = tk.Canvas(self.left_frame, bg='lightblue')
        # make the canvas expandable to fill the window
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        self.left_control_frame = tk.Frame(self.left_frame)
        self.left_control_frame.pack(pady=10, padx=10)

        self.play_button = tk.Button(self.left_control_frame, text='Play', command=self.play_video)
        self.pause_button = tk.Button(self.left_control_frame, text='Pause', command=self.pause_video)
        self.stop_button = tk.Button(self.left_control_frame, text='Stop', command=self.stop_video)
        self.next_frame_button = tk.Button(self.left_control_frame, text='>>', command=self.next_frame)
        self.prev_frame_button = tk.Button(self.left_control_frame, text='<<', command=self.prev_frame)

        self.play_button.grid(row=0, column=0, padx=5)
        self.pause_button.grid(row=0, column=1, padx=5)
        self.stop_button.grid(row=0, column=2, padx=5)
        self.prev_frame_button.grid(row=0, column=3, padx=5)
        self.next_frame_button.grid(row=0, column=4, padx=5)

        self.zone = None

        self.right_frame = tk.Frame(self.root, width=600, height=600)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.top_control_frame = tk.Frame(self.right_frame)
        self.top_control_frame.pack(pady=10, padx=10)

        
        self.track_button = tk.Button(self.top_control_frame, text='Track', command=self.track)
        self.show_track_video_button = tk.Button(self.top_control_frame, text='Show Track Video', command=self.show_track_video)
        self.save_track_video_button = tk.Button(self.top_control_frame, text='Save Track Video', command=self.save_track_video)

        self.track_button.grid(row=0, column=0, padx=5)
        self.show_track_video_button.grid(row=0, column=1, padx=5)
        self.save_track_video_button.grid(row=0, column=2, padx=5)

        self.track_canvas = tk.Canvas(self.right_frame, bg='orange')
        self.track_canvas.pack(fill=tk.BOTH, expand=True)

        self.bottom_control_frame = tk.Frame(self.right_frame)
        self.bottom_control_frame.pack(pady=10, padx=10)

        self.prev_track_button = tk.Button(self.bottom_control_frame, text='<<', command=self.prev_track)
        self.next_track_button = tk.Button(self.bottom_control_frame, text='>>', command=self.next_track)

        self.prev_track_button.grid(row=0, column=0, padx=5)
        self.next_track_button.grid(row=0, column=1, padx=5)

    def detect_bees(self):
        # check if the video is loaded
        if self.video is None:
            messagebox.showerror('Error', 'No video loaded')
            return
        # set the video to not playing
        self.playing = False
        time_needed = self.bee_detector.detect_insects_yolo(self.original_video_path, conf_threshold=0.65)
        messagebox.showinfo('Info', f'Detection Done. Took {time_needed:.2f} seconds')
        extension = self.video_path.split('.')[-1]
        # detected bee video: video_path.replace(".mp4", "_yolo_detections.mp4")
        # detected bee csv: video_path.replace(".mp4", "_yolo_detections.csv")
        self.detected_bee_video_path = self.video_path.replace("."+extension, "_yolo_detections.mp4")
        self.detected_bee_csv_path = self.video_path.replace("."+extension, "_yolo_detections.csv")
        self.video_path = self.detected_bee_video_path
        self.init_video()
        # load the detected bee video


        
    def assign_zone(self):
        # check if the video is loaded
        if self.video is None:
            print('No video loaded')
            return
        # set the video to not playing
        self.playing = False
        self.polygon_points = []
        # get the video frame and open it
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.video_frame_index)
        ret, frame = self.video.read()
        if not ret:
            print('Error reading frame')
            return
        self.assign_zone_original_frame = frame.copy()
        self.assign_zone_frame, self.anchor_point = self.show_frame(frame, drawOnCanvas=False, canvas=self.track_canvas)
        x, y = self.anchor_point
        # draw frame on the canvas
        self.track_canvas.create_image(x, y, image=self.assign_zone_frame, anchor=tk.NW)
        # delete all the anchors and polygons
        self.track_canvas.delete('anchors')
        self.track_canvas.delete('polygon')
        # bind the canvas to the click event to assign the zone
        self.track_canvas.bind('<Button-1>', self.on_click_zone)
        #
    
    def draw_zone(self):
        # draw the zone on the canvas
        self.track_canvas.delete('polygon')
        self.track_canvas.delete('anchors')
        for x, y in self.polygon_points:
            self.track_canvas.create_oval(x-2, y-2, x+2, y+2, fill='red', tags='anchors')
        if len(self.polygon_points) < 3:
            return
        points = [p for point in self.polygon_points for p in point]
        points.extend(self.polygon_points[0])
        self.track_canvas.create_polygon(points, outline='red', width=2, fill='', tags='polygon')

    def on_click_zone(self, event):
        # get the x and y coordinates of the click
        x, y = event.x, event.y
        # add the point to the polygon points
        self.polygon_points.append((x, y))
        self.draw_zone()
        if len(self.polygon_points) >= 3:
            self.zone = self.polygon_points
            

    def track(self):
        # check if the video is loaded
        if self.video is None:
            print('No video loaded')
            return
        # check if the zone is assigned
        if self.zone is None:
            print('No zone assigned')
            return
        # check if detected bee csv exists
        if not os.path.exists(self.detected_bee_csv_path):
            print('No detected bee csv file')
            return
        # set the video to not playing
        self.playing = False
        # unbind the canvas from the click event
        self.track_canvas.unbind('<Button-1>')
        # load the detected bee csv
        self.tracks, counts = self.bee_tracker.track_insects(self.detected_bee_csv_path, self.zone)
        print(f'Tracks: {len(self.tracks)}')
        if self.tracks is None:
            print('No tracks found')
            return
        self.n_tracks = len(self.tracks)
        self.current_track_index = 0
        messagebox.showinfo('Info', f'Tracking Done. Found {self.n_tracks} tracks.\nEntering: {counts["entering"]}\nExiting: {counts["exiting"]}\nInside: {counts["inside"]}')
        self.draw_tracks(self.tracks[self.current_track_index])
        ...
    
    def draw_tracks(self, track):
        # check if the video is loaded
        if self.video is None:
            print('No video loaded')
            return
        # check if the zone is assigned
        if self.zone is None:
            print('No zone assigned')
            return
        # check if the tracks are assigned
        if self.tracks is None:
            print('No tracks assigned')
            return
        # set the video to not playing
        self.playing = False
        # get the video frame
        observations = self.bee_tracker.get_observations()
        # draw observations on the frame
        frame = self.assign_zone_original_frame.copy()
        for observation in observations:
            x, y = observation
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        tracked_points = track['positions']
        cv2.polylines(frame, [np.array(tracked_points)], False, (0, 255, 0), 3)
        # draw circle on the last point
        x, y = tracked_points[-1]
        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
        
        self.tracking_image, (x,y) = self.show_frame(frame, drawOnCanvas=False, canvas=self.track_canvas)
        self.track_canvas.create_image(x, y, image=self.tracking_image, anchor=tk.NW)
        self.draw_zone()

    def show_track_video(self):
        track_id = self.current_track_index
        track = self.tracks[track_id]
        frame_ids = track['frame_ids']
        positions = track['positions']
        for frame_id, position in zip(frame_ids, positions):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = self.video.read()
            
            if not ret:
                print('Error reading frame')
                return
            observations = self.bee_tracker.get_observations(frame_id=frame_id)
            for observation in observations:
                x, y = observation
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            cv2.polylines(frame, [np.array(positions)], False, (0, 255, 0), 3)
            # draw circle on the last point
            x, y = position
            cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
            self.track_video_frame, anchor = self.show_frame(frame, drawOnCanvas=False, canvas=self.track_canvas)
            x, y = anchor
            self.track_canvas.create_image(x, y, image=self.track_video_frame, anchor=tk.NW)
            self.track_canvas.update()
            self.track_canvas.after(int(1000/self.fps))

    def save_track_video(self):
        track_id = self.current_track_index
        track = self.tracks[track_id]
        frame_ids = track['frame_ids']
        positions = track['positions']
        track_type = track['classification']
        extension = self.video_path.split('.')[-1]
        out_file_name = self.video_path.replace("."+extension, f"_track_{track_id}_{track_type}.mp4")
        out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        for frame_id, position in zip(frame_ids, positions):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = self.video.read()
            if not ret:
                print('Error reading frame')
                return
            observations = self.bee_tracker.get_observations(frame_id=frame_id)
            for observation in observations:
                x, y = observation
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            cv2.polylines(frame, [np.array(positions)], False, (0, 255, 0), 3)
            # draw circle on the last point
            x, y = position
            cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
            out.write(frame)
        out.release()
        messagebox.showinfo('Info', f'Track video saved to {out_file_name}')

    def save(self):
        ...

    def prev_track(self):
        self.current_track_index -= 1
        if self.current_track_index < 0:
            self.current_track_index = self.n_tracks - 1
        self.draw_tracks(self.tracks[self.current_track_index])
        ...

    def next_track(self):
        self.current_track_index += 1
        if self.current_track_index >= self.n_tracks:
            self.current_track_index = 0
        self.draw_tracks(self.tracks[self.current_track_index])
        ...

    def show_frame(self, frame, drawOnCanvas=True, canvas=None):
        canvas = self.video_canvas if canvas is None else canvas
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize the frame to fit the canvas
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        height_ratio = height / frame.shape[0]
        width_ratio = width / frame.shape[1]
        ratio = min(height_ratio, width_ratio)
        frame = cv2.resize(frame, (math.floor(frame.shape[1] * ratio), math.floor(frame.shape[0] * ratio)))
        if not drawOnCanvas:
            frame = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
            anchor_y = canvas.winfo_height() // 2 - frame.height() // 2
            anchor_x = canvas.winfo_width() // 2 - frame.width() // 2
            return frame, (anchor_x, anchor_y)
        self.video_image = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
        # set image anchor to center to make it easier to draw on the image
        anchor_y = canvas.winfo_height() // 2 - self.video_image.height() // 2
        anchor_x = canvas.winfo_width() // 2 - self.video_image.width() // 2
        self.video_canvas.create_image(anchor_x, anchor_y, image=self.video_image, anchor=tk.NW)
        return 

    def play_video(self):
        # play the video. but first check if the video is loaded
        if self.video is None:
            messagebox.showerror('Error', 'No video loaded')
            return
        self.playing = True
        while True and self.video_frame_index < self.total_frames and self.playing:
            self.video_frame_index += 1
            self.update_video_frame()
            self.video_canvas.update()
            self.video_canvas.after(int(1000/self.fps))

    def pause_video(self):
        # pause the video. check if the video is loaded
        if self.video is None:
            return
        self.playing = False

    def stop_video(self):
        # stop the video. check if the video is loaded
        if self.video is None:
            return
        self.video_frame_index = 0
        self.update_video_frame()
        self.playing = False

    def next_frame(self):
        # move to the next frame. check if the video is loaded
        if self.video is None:
            return
        
        self.video_frame_index += 1
        self.update_video_frame()
        self.playing = False

    def prev_frame(self):
        # move to the previous frame. check if the video is loaded
        if self.video is None:
            return
        
        self.video_frame_index -= 1
        # make sure the frame index is not negative. if it is, set it to total_frames - 1
        if self.video_frame_index < 0:
            self.video_frame_index = self.total_frames - 1
        self.update_video_frame()
        self.playing = False

    def load_video(self):
        self.file_path = filedialog.askopenfilename()
        # check if the video file is selected
        if not self.file_path:
            return
        # check if the file is a video file
        if not self.file_path.endswith(('.mp4', '.avi', '.mov')):
            messagebox.showerror('Error', 'Please select a video file')
            return
        # try opening the video file using opencv
        self.video = cv2.VideoCapture(self.file_path)
        if not self.video.isOpened():
            messagebox.showerror('Error', 'Error opening video file')
            return
        self.video.release()
        # copy the video to 'videos' folder
        self.original_video_path = os.path.join('outputs', os.path.basename(self.file_path))
        self.video_path = os.path.join('outputs', os.path.basename(self.file_path))
        os.makedirs('outputs', exist_ok=True)
        if not os.path.exists(self.video_path):
            shutil.copyfile(self.file_path, self.video_path) # copy the video to the videos folder only if it is not already there
        self.init_video()

    def init_video(self):
        self.video = cv2.VideoCapture(self.video_path)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_frame_index = 0
        self.update_video_frame()
        # bind the canvas to the click event to move to the next frame
        self.video_canvas.bind('<Button-1>', self.on_click)
        self.playing = False

    def on_click(self, event):
        self.video_frame_index += 1
        self.update_video_frame()

    def update_video_frame(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.video_frame_index)
        ret, frame = self.video.read()
        if not ret:
            return
        self.show_frame(frame)
        
 
if __name__ == '__main__':
    root = tk.Tk()
    app = BeeTrackerGUI(root)
    root.mainloop()
