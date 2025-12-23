import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

# Load the cleaned data
csv_data = """Block,X,Y,Z,Dir,cx,cy,cz
RoadTechStart,16,1,21,North,528.0,12.0,688.0
RoadTechStraight,16,1,22,North,528.0,12.0,720.0
RoadTechStraight,16,1,23,North,528.0,12.0,752.0
RoadTechStraight,16,1,24,North,528.0,12.0,784.0
RoadTechCurve1,16,1,25,East,528.0,12.0,816.0
RoadTechStraight,15,1,25,East,496.0,12.0,816.0
RoadTechStraight,14,1,25,East,464.0,12.0,816.0
RoadTechCurve1,13,1,25,North,432.0,12.0,816.0
RoadTechStraight,13,1,26,North,432.0,12.0,848.0
RoadTechStraight,13,1,27,North,432.0,12.0,880.0
RoadTechCurve1,13,1,28,West,432.0,12.0,912.0
RoadTechStraight,14,1,28,West,464.0,12.0,912.0
RoadTechStraight,15,1,28,West,496.0,12.0,912.0
RoadTechFinish,16,1,28,West,528.0,12.0,912.0"""

df = pd.read_csv(io.StringIO(csv_data))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TrackLineProcessor:
    def __init__(self, df):
        """
        Expects a cleaned DataFrame with columns: [Block, cx, cz, Dir]
        """
        self.df = df
        # Extract the sequence of points (the median trajectory)
        self.points = df[['cx', 'cz']].values
        self.num_points = len(self.points)
        
        # Precompute segment vectors and lengths for distance-to-end calculations
        self.segments = np.diff(self.points, axis=0)
        self.segment_lengths = np.linalg.norm(self.segments, axis=1)
        self.cum_dist_from_end = np.zeros(self.num_points)
        # Cumulative distance from the end (backwards sum)
        self.cum_dist_from_end[:-1] = np.cumsum(self.segment_lengths[::-1])[::-1]

    def _get_closest_segment(self, pos):
        """Finds the closest point on the piecewise linear trajectory."""
        min_dist = float('inf')
        closest_p = self.points[0]
        active_segment_idx = 0

        for i in range(len(self.segments)):
            p1 = self.points[i]
            p2 = self.points[i+1]
            # Vector from p1 to pos
            v = pos - p1
            # Segment vector
            u = self.segments[i]
            t = np.dot(v, u) / np.dot(u, u)
            t = np.clip(t, 0, 1) # Closest point must be on the segment
            
            projection = p1 + t * u
            dist = np.linalg.norm(pos - projection)
            
            if dist < min_dist:
                min_dist = dist
                closest_p = projection
                active_segment_idx = i
                
        return closest_p, min_dist, active_segment_idx

    def get_car_stats(self, car_pos, car_angle_deg):
        """
        Computes distance to line and relative angle.
        car_angle_deg: 0 is North (+Z), 90 is East (-X) [Custom TM coords]
        """
        car_pos = np.array(car_pos)
        closest_p, dist_to_line, idx = self._get_closest_segment(car_pos)
        
        # Direction of the trajectory at this segment
        seg_vec = self.segments[idx]
        # Angle in degrees (Standard Math: atan2(y, x))
        # In your system: North is +Z, East is -X. 
        # So we use atan2(-dx, dz) to align with your Dir mapping
        line_angle_rad = np.atan2(-seg_vec[0], seg_vec[1])
        line_angle_deg = np.degrees(line_angle_rad) % 360
        
        # Angle difference (normalized to -180, 180)
        angle_diff = (car_angle_deg - line_angle_deg + 180) % 360 - 180
        
        return dist_to_line, angle_diff

    def get_distance_to_finish(self, car_pos):
        """Computes remaining distance along the trajectory line."""
        car_pos = np.array(car_pos)
        closest_p, _, idx = self._get_closest_segment(car_pos)
        
        # Distance from closest_p to the end of its current segment
        dist_to_segment_end = np.linalg.norm(self.points[idx+1] - closest_p)
        
        # Total distance = dist to segment end + distance of all following segments
        remaining_dist = dist_to_segment_end + self.cum_dist_from_end[idx+1]
        return remaining_dist

    def draw_track(self, car_pos=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        # Draw Blocks
        for _, row in self.df.iterrows():
            rect = patches.Rectangle((row['cx']-16, row['cz']-16), 32, 32, 
                                     linewidth=0.5, edgecolor='gray', facecolor='whitesmoke')
            ax.add_patch(rect)
        
        # Draw Trajectory Line
        ax.plot(self.points[:, 0], self.points[:, 1], 'r--', label='Median Trajectory', alpha=0.7)
        ax.scatter(self.points[:, 0], self.points[:, 1], c='blue', s=10)
        
        if car_pos is not None:
            ax.scatter(car_pos[0], car_pos[1], c='green', marker='^', s=100, label='Car')
            
        ax.set_aspect('equal')
        ax.invert_xaxis() # Since East is decreasing X
        plt.legend()
        plt.show()

# Assuming 'df' is your cleaned dataframe
processor = TrackLineProcessor(df)

# Example car state
my_car_pos = [525.0, 820.0] 
my_car_dir = 0 # Slightly off-North

dist, angle_err = processor.get_car_stats(my_car_pos, my_car_dir)
dist_remaining = processor.get_distance_to_finish(my_car_pos)

print(f"Distance to Line: {dist:.2f}m")
print(f"Angle Error: {angle_err:.2f} degrees")
print(f"Distance to Finish: {dist_remaining:.2f}m")

processor.draw_track(car_pos=my_car_pos)