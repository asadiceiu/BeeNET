# Description: A GUI application for bee tracking using YOLO and Kalman Filter.
#
# Created By: Asaduz Zaman
# Created On: 10 August 2024
# Updated By: Asaduz Zaman
# Updated On: 10:00 PM 10 August 2024
import math
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from bee_detector_yolo import BeeDetectorYolo
from kalman_tracker import KalmanFilterTracker


class BeeTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Bee Tracking GUI')
        self.root.geometry('1200x600')
        self.bee_detector = BeeDetectorYolo('models/stinglessbee-yolov8-s-best.pt', device='mps')
        self.bee_tracker = KalmanFilterTracker()
        self.zone = None
        self.video = None
        self.tracks = None
        self.playing = False
        self.current_track_index = 0
        self.polygon_points = []
        self.detected_bee_csv_path = None
        self.detected_bee_video_path = None
        self.setup_ui()

    def setup_ui(self):
        """Initialize the UI components."""
        self.left_frame = self.create_frame(self.root, side=tk.LEFT)
        self.setup_video_controls()

        self.video_canvas = self.create_canvas(self.left_frame, bg='lightblue')

        self.right_frame = self.create_frame(self.root, side=tk.RIGHT)
        self.setup_tracking_controls()

        self.track_canvas = self.create_canvas(self.right_frame, bg='orange')

    def create_frame(self, parent, **kwargs):
        """Helper function to create a frame."""
        frame = tk.Frame(parent, width=600, height=600)
        frame.pack(fill=tk.BOTH, expand=True, **kwargs)
        return frame

    def create_canvas(self, parent, **kwargs):
        """Helper function to create a canvas."""
        canvas = tk.Canvas(parent, **kwargs)
        canvas.pack(fill=tk.BOTH, expand=True)
        return canvas

    def setup_video_controls(self):
        """Setup the video control buttons."""
        control_frame = tk.Frame(self.left_frame)
        control_frame.pack(pady=10, padx=10)

        self.create_button(control_frame, 'Load Video', self.load_video, 0, 0)
        self.create_button(control_frame, 'Detect Bees', self.detect_bees, 0, 1)
        self.create_button(control_frame, 'Assign Zone', self.assign_zone, 0, 2)

        play_controls = tk.Frame(self.left_frame)
        play_controls.pack(pady=10, padx=10)

        self.create_button(play_controls, 'Play', self.play_video, 0, 0)
        self.create_button(play_controls, 'Pause', self.pause_video, 0, 1)
        self.create_button(play_controls, 'Stop', self.stop_video, 0, 2)
        self.create_button(play_controls, '<<', self.prev_frame, 0, 3)
        self.create_button(play_controls, '>>', self.next_frame, 0, 4)

    def setup_tracking_controls(self):
        """Setup the tracking control buttons."""
        control_frame = tk.Frame(self.right_frame)
        control_frame.pack(pady=10, padx=10)

        self.create_button(control_frame, 'Track', self.track, 0, 0)
        self.create_button(control_frame, 'Show Track Video', self.show_track_video, 0, 1)
        self.create_button(control_frame, 'Save Track Video', self.save_track_video, 0, 2)

        bottom_frame = tk.Frame(self.right_frame)
        bottom_frame.pack(pady=10, padx=10)

        self.create_button(bottom_frame, '<<', self.prev_track, 0, 0)
        self.create_button(bottom_frame, '>>', self.next_track, 0, 1)

    def create_button(self, parent, text, command, row, column, **kwargs):
        """Helper function to create a button."""
        button = tk.Button(parent, text=text, command=command)
        button.grid(row=row, column=column, padx=5, **kwargs)
        return button

    def detect_bees(self):
        """Detect bees in the loaded video using YOLO."""
        if not self.video:
            messagebox.showerror('Error', 'No video loaded')
            return

        self.playing = False
        time_needed = self.bee_detector.detect_insects_yolo(self.original_video_path, conf_threshold=0.65)
        messagebox.showinfo('Info', f'Detection Done. Took {time_needed:.2f} seconds')

        extension = self.video_path.split('.')[-1]
        self.detected_bee_video_path = self.video_path.replace(f".{extension}", "_yolo_detections.mp4")
        self.detected_bee_csv_path = self.video_path.replace(f".{extension}", "_yolo_detections.csv")
        self.video_path = self.detected_bee_video_path
        self.init_video()

    def assign_zone(self):
        """Assign a zone within the video frame."""
        if not self.video:
            print('No video loaded')
            return

        self.playing = False
        self.polygon_points = []

        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.video_frame_index)
        ret, frame = self.video.read()

        if not ret:
            print('Error reading frame')
            return

        self.assign_zone_original_frame = frame.copy()
        self.assign_zone_frame, self.anchor_point = self.show_frame(frame, drawOnCanvas=False, canvas=self.track_canvas)
        x, y = self.anchor_point

        self.track_canvas.create_image(x, y, image=self.assign_zone_frame, anchor=tk.NW)
        self.track_canvas.delete('anchors')
        self.track_canvas.delete('polygon')
        self.track_canvas.bind('<Button-1>', self.on_click_zone)

    def draw_zone(self):
        """Draw the zone polygon on the canvas."""
        self.track_canvas.delete('polygon')
        self.track_canvas.delete('anchors')

        for x, y in self.polygon_points:
            self.track_canvas.create_oval(x-2, y-2, x+2, y+2, fill='red', tags='anchors')

        if len(self.polygon_points) >= 3:
            points = [p for point in self.polygon_points for p in point]
            points.extend(self.polygon_points[0])
            self.track_canvas.create_polygon(points, outline='red', width=2, fill='', tags='polygon')

    def on_click_zone(self, event):
        """Handle click event on the canvas to define the zone."""
        self.polygon_points.append((event.x, event.y))
        self.draw_zone()

        if len(self.polygon_points) >= 3:
            self.zone = self.polygon_points

    def track(self):
        """Track bees based on the assigned zone and detected data."""
        if not self.video or not self.zone or not os.path.exists(self.detected_bee_csv_path):
            print('Missing data: video, zone, or detected bee CSV')
            return

        self.playing = False
        self.track_canvas.unbind('<Button-1>')
        self.tracks, counts = self.bee_tracker.track_insects(self.detected_bee_csv_path, self.zone)

        if not self.tracks:
            print('No tracks found')
            return

        self.n_tracks = len(self.tracks)
        self.current_track_index = 0
        messagebox.showinfo(
            'Info', 
            f'Tracking Done. Found {self.n_tracks} tracks.\n'
            f'Entering: {counts["entering"]}\n'
            f'Exiting: {counts["exiting"]}\n'
            f'Inside: {counts["inside"]}\n'
            f'Outside: {counts["outside"]}'
        )
        self.draw_tracks(self.tracks[self.current_track_index])

    def draw_tracks(self, track):
        """Draw the tracks on the canvas."""
        if not self.video or not self.zone or not self.tracks:
            print('Missing data: video, zone, or tracks')
            return

        self.playing = False
        frame = self.assign_zone_original_frame.copy()

        for observation in self.bee_tracker.get_observations():
            x, y = observation
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        tracked_points = track['positions']
        cv2.polylines(frame, [np.array(tracked_points)], False, (0, 255, 0), 3)
        x, y = tracked_points[-1]
        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

        self.tracking_image, (x, y) = self.show_frame(frame, drawOnCanvas=False, canvas=self.track_canvas)
        self.track_canvas.create_image(x, y, image=self.tracking_image, anchor=tk.NW)
        self.draw_zone()

    def show_track_video(self):
        """Show the video with the tracked bees."""
        track = self.tracks[self.current_track_index]

        for frame_id, position in zip(track['frame_ids'], track['positions']):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = self.video.read()

            if not ret:
                print('Error reading frame')
                return

            for observation in self.bee_tracker.get_observations(frame_id=frame_id):
                x, y = observation
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            cv2.polylines(frame, [np.array(track['positions'])], False, (0, 255, 0), 3)
            cv2.circle(frame, position, 10,(0, 255, 255), -1)

            self.track_video_frame, anchor = self.show_frame(frame, drawOnCanvas=False, canvas=self.track_canvas)
            x, y = anchor
            self.track_canvas.create_image(x, y, image=self.track_video_frame, anchor=tk.NW)
            self.track_canvas.update()
            self.track_canvas.after(int(1000/self.fps))

    def save_track_video(self):
        """Save the video with the tracked bees."""
        track = self.tracks[self.current_track_index]
        extension = self.video_path.split('.')[-1]
        out_file_name = self.video_path.replace(f".{extension}", f"_track_{self.current_track_index}.mp4")

        out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                              (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        for frame_id, position in zip(track['frame_ids'], track['positions']):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = self.video.read()

            if not ret:
                print('Error reading frame')
                return

            for observation in self.bee_tracker.get_observations(frame_id=frame_id):
                x, y = observation
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            cv2.polylines(frame, [np.array(track['positions'])], False, (0, 255, 0), 3)
            cv2.circle(frame, position, 10, (0, 255, 255), -1)
            out.write(frame)

        out.release()
        messagebox.showinfo('Info', f'Track video saved to {out_file_name}')

    def prev_track(self):
        """Navigate to the previous track."""
        self.current_track_index = (self.current_track_index - 1) % self.n_tracks
        self.draw_tracks(self.tracks[self.current_track_index])

    def next_track(self):
        """Navigate to the next track."""
        self.current_track_index = (self.current_track_index + 1) % self.n_tracks
        self.draw_tracks(self.tracks[self.current_track_index])

    def show_frame(self, frame, drawOnCanvas=True, canvas=None):
        """Show the current frame on the canvas."""
        canvas = self.video_canvas if canvas is None else canvas
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width, height = canvas.winfo_width(), canvas.winfo_height()
        ratio = min(height / frame.shape[0], width / frame.shape[1])
        frame = cv2.resize(frame, (math.floor(frame.shape[1] * ratio), math.floor(frame.shape[0] * ratio)))

        if not drawOnCanvas:
            frame = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
            anchor_x = (canvas.winfo_width() - frame.width()) // 2
            anchor_y = (canvas.winfo_height() - frame.height()) // 2
            return frame, (anchor_x, anchor_y)

        self.video_image = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
        anchor_x = (canvas.winfo_width() - self.video_image.width()) // 2
        anchor_y = (canvas.winfo_height() - self.video_image.height()) // 2
        self.video_canvas.create_image(anchor_x, anchor_y, image=self.video_image, anchor=tk.NW)

    def play_video(self):
        """Play the video."""
        if not self.video:
            messagebox.showerror('Error', 'No video loaded')
            return

        self.playing = True
        while self.playing and self.video_frame_index < self.total_frames:
            self.video_frame_index += 1
            self.update_video_frame()
            self.video_canvas.update()
            self.video_canvas.after(int(1000 / self.fps))

    def pause_video(self):
        """Pause the video."""
        if self.video:
            self.playing = False

    def stop_video(self):
        """Stop the video and reset to the first frame."""
        if self.video:
            self.video_frame_index = 0
            self.update_video_frame()
            self.playing = False

    def next_frame(self):
        """Move to the next frame."""
        if self.video:
            self.video_frame_index += 1
            self.update_video_frame()
            self.playing = False

    def prev_frame(self):
        """Move to the previous frame."""
        if self.video:
            self.video_frame_index = (self.video_frame_index - 1) % self.total_frames
            self.update_video_frame()
            self.playing = False

    def load_video(self):
        """Load a video file for tracking."""
        self.file_path = filedialog.askopenfilename()

        if not self.file_path or not self.file_path.endswith(('.mp4', '.avi', '.mov')):
            messagebox.showerror('Error', 'Please select a valid video file')
            return

        self.video = cv2.VideoCapture(self.file_path)

        if not self.video.isOpened():
            messagebox.showerror('Error', 'Error opening video file')
            return

        self.video.release()
        self.original_video_path = os.path.join('outputs', os.path.basename(self.file_path))
        self.video_path = os.path.join('outputs', os.path.basename(self.file_path))
        os.makedirs('outputs', exist_ok=True)

        if not os.path.exists(self.original_video_path):
            shutil.copyfile(self.file_path, self.video_path)

        self.init_video()

    def init_video(self):
        """Initialize the video properties and display the first frame."""
        self.video = cv2.VideoCapture(self.video_path)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_frame_index = 0
        self.update_video_frame()
        self.video_canvas.bind('<Button-1>', self.on_click)
        self.playing = False

    def on_click(self, event):
        """Handle click events on the video canvas to move to the next frame."""
        self.next_frame()

    def update_video_frame(self):
        """Update the canvas with the current video frame."""
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.video_frame_index)
        ret, frame = self.video.read()

        if ret:
            self.show_frame(frame)


if __name__ == '__main__':
    root = tk.Tk()
    app = BeeTrackerGUI(root)
    root.mainloop()