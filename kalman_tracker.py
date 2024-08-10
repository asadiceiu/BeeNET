# Description: Class to track insects using a Kalman filter.
#
# Created By: Asaduz Zaman
# Created On: 10 August 2024
# Updated By: Asaduz Zaman
# Updated On: 10:00 PM 10 August 2024
from matplotlib.path import Path
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import pandas as pd

class KalmanFilterTracker():
    """
    Class to track insects using a Kalman filter.
    Attributes:
        all_tracks: List of dictionaries containing track information.
        process_noise: Process noise for the Kalman filter.
        measurement_noise: Measurement noise for the Kalman filter.
        distance_threshold: Maximum distance between a track and an observation for assignment.
        max_frames_before_stopped: Maximum number of frames before a track is considered stopped.
        kf_dim_x: Dimension of the state vector for the Kalman filter.
        kf_dim_z: Dimension of the observation vector for the Kalman filter.
        F: State transition matrix for the Kalman filter.
        H: Measurement matrix for the Kalman filter.
        R: Measurement noise covariance matrix for the Kalman filter.
        Q: Process noise covariance matrix for the Kalman filter.
        P: Initial state covariance matrix for the Kalman filter
    """
    def __init__(self) -> None:
        self.all_tracks = []
        self.process_noise = 0.1
        self.measurement_noise = 0.1
        self.distance_threshold = 50
        self.max_frames_before_stopped = 10
        self.kf_dim_x = 4
        self.kf_dim_z = 2
        self.F = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.R = np.eye(self.kf_dim_z) * self.measurement_noise
        self.Q = np.eye(self.kf_dim_x) * self.process_noise
        self.P = np.eye(self.kf_dim_x)

    def track_insects(self, csv_file_name, zone) -> pd.DataFrame:
        self.df = pd.read_csv(csv_file_name)
        self.df.columns = self.df.columns.str.strip()
        self.df['frame_id'] = self.df['frame_id'].astype(int)
        frame_ids = self.df['frame_id'].unique()
        frame_ids.sort()

        start_frame = frame_ids[0]
        end_frame = frame_ids[-1]
        tracks = []
        all_tracks = []
        track_id = 0
        for frame_id in range(start_frame, end_frame):
            print(f"Processing frame: {frame_id}/{end_frame}", end='\r')
            observations = self.df[self.df['frame_id'] == frame_id][['center_x', 'center_y']].values
            if len(observations) == 0:
                continue
            for track in tracks:
                track['age'] += 1
                track['kf'].predict()

            # create cost matrix for the hungarian algorithm
            cost_matrix = np.zeros((len(tracks), len(observations)))
            for i, track in enumerate(tracks):
                for j, obs in enumerate(observations):
                    cost_matrix[i, j] = np.linalg.norm(track['kf'].x[:2] - obs)
            
            # apply hungarian algorithm
            track_indices, observation_indices = linear_sum_assignment(cost_matrix)

            # update tracks
            assigned_tracks = set()
            assigned_observations = set()

            for i, j in zip(track_indices, observation_indices):
                if cost_matrix[i, j] < self.distance_threshold:
                    tracks[i]['age'] = 0
                    tracks[i]['kf'].update(observations[j])
                    tracks[i]['positions'].append(observations[j])
                    tracks[i]['frame_ids'].append(frame_id)
                    tracks[i]['distance'] += cost_matrix[i, j]
                    assigned_tracks.add(i)
                    assigned_observations.add(j)

            # create new tracks
            for j in range(len(observations)):
                if j not in assigned_observations:
                    kf = KalmanFilter(dim_x=self.kf_dim_x, dim_z=self.kf_dim_z)
                    kf.x = np.array([observations[j][0], observations[j][1], 0, 0])
                    kf.F = self.F
                    kf.H = self.H
                    kf.R = self.R
                    kf.Q = self.Q
                    kf.P = self.P
                    tracks.append({'track_id': track_id, 'age': 0, 
                                   'kf': kf, 'positions': [observations[j]], 
                                   'frame_ids': [frame_id], 'distance': 0,
                                   'classification':'unknown'})
                    track_id += 1
            # remove tracks that have not been updated for a while
            tracks = [track for track in tracks if track['age'] < self.max_frames_before_stopped]

            for track in tracks:
                if track['track_id'] not in [t['track_id'] for t in all_tracks]:
                    all_tracks.append(track)
        print()
        self.all_tracks = [track for track in all_tracks if len(track['positions']) > 5 and track['distance'] > 100]
        # sort tracks by number of positions
        self.all_tracks.sort(key=lambda x: len(x['positions']), reverse=True)
        # classify tracks
        counts = {
            'inside': 0,
            'outside': 0,
            'entering': 0,
            'exiting': 0
        }
        for track in self.all_tracks:
            track['classification'] = self.flexible_zone_track_classifier(track, zone)
            counts[track['classification']] += 1
        return self.all_tracks, counts
    
    def get_track_info(self, track_id) -> dict:
        for track in self.all_tracks:
            if track['track_id'] == track_id:
                return track
        return None
    
    def get_observations(self, frame_id=None) -> np.array:
        if frame_id is None:
            return self.df[['center_x', 'center_y']].values
        return self.df[self.df['frame_id'] == frame_id][['center_x', 'center_y']].values
    
    def flexible_zone_track_classifier(self, track, zone) -> str:
        """
        Using a ray casting algorithm, classify the track as either inside or outside the zone.
            zone: list of points that define the zone. polygon points in the form of [[x1, y1], [x2, y2], ...]
            track: dict containing the track information
        return: str 'inside' or 'outside' or 'entering' or 'exiting'
        """
        start_point = track['positions'][0]
        end_point = track['positions'][-1]

        polygon_path = Path(zone)
        start_inside = polygon_path.contains_point(start_point)
        end_inside = polygon_path.contains_point(end_point)
        if start_inside and end_inside:
            return 'inside'
        elif start_inside and not end_inside:
            return 'exiting'
        elif not start_inside and end_inside:
            return 'entering'
        else:
            return 'outside'

if __name__ == '__main__':
    # print the docstring of the class
    print("Note: This is a class definition file and cannot be executed directly.")
    print("=========================================")
    print(KalmanFilterTracker.__doc__)
    print("=========================================")
    print("Example usage:")
    print("kf_tracker = KalmanFilterTracker()")
    print("all_tracks, counts = kf_tracker.track_insects('detected_bee.csv', zone)")