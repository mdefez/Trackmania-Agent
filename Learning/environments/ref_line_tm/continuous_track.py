import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

class TrackmaniaTrack:
    def __init__(self, csv_str):
        self.df = pd.read_csv(io.StringIO(csv_str))
        self.block_size = 32
    
    ## Compute pivot point of each block
    def compute_pivots(self):
        init_angle = 90  # Facing North
        angle = init_angle
        for idx, row in self.df.iterrows():
            block_type = row['Block']
            cx =  - row['cx']
            cz = row['cz']
            rotation = row['Rotate']

            pointing_vec = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            if 'Curve' in block_type:
                if 'Left' in rotation:
                    rotation_angle = 90
                    angle += rotation_angle
                elif 'Right' in rotation:
                    rotation_angle = -90
                    angle += rotation_angle

                rotation_matrix = np.array([[np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
                                             [np.sin(np.radians(rotation_angle)),  np.cos(np.radians(rotation_angle))]])
                offset_vec = rotation_matrix @ pointing_vec
                
                begin_point = [cx, cz]
                begin_point -= pointing_vec * (self.block_size / 2)
                begin_point += offset_vec * (self.block_size / 2)

                pivot = begin_point
                ## Set pivot in dataframe
                self.df.at[idx, 'pivot_x'] = pivot[0]
                self.df.at[idx, 'pivot_z'] = pivot[1]

                output_vec = [cx, cz] + offset_vec * self.block_size / 2
                self.df.at[idx, "output_x"] = output_vec[0]
                self.df.at[idx, "output_z"] = output_vec[1]
            else :
                self.df.at[idx, 'angle'] = angle % 360
                self.df.at[idx, 'output_x'] = cx + pointing_vec[0] * self.block_size / 2
                self.df.at[idx, 'output_z'] = cz + pointing_vec[1] * self.block_size / 2
    
    def _conpute_block_idx(self, pos):
        x_block = int(pos[0] // self.block_size)
        z_block = int(pos[1] // self.block_size)
        for idx, row in self.df.iterrows():
            block_x = int(row['cx'] // self.block_size)
            block_z = int(row['cz'] // self.block_size)
            if block_x == x_block and block_z == z_block:
                return idx
        return None
    
    def _draw_blocks(self):
        plt.figure(figsize=(10,10))
        for idx, row in self.df.iterrows():
            cx = - row['cx']
            cz = row['cz']
            corners = np.array([
                [cx - self.block_size/2, cz - self.block_size/2],
                [cx + self.block_size/2, cz - self.block_size/2],
                [cx + self.block_size/2, cz + self.block_size/2],
                [cx - self.block_size/2, cz + self.block_size/2]
            ])
            plt.fill(corners[:,0], corners[:,1], color='lightgray', edgecolor='black')
        
        plt.plot(-self.df['cx'], self.df['cz'], 'r-o', label='Centerline')
        plt.title("Trackmania Track Shape")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    def compute_line_angle(self, pos) :
        idx = self._conpute_block_idx(pos)
        if idx is None or idx >= len(self.df) - 1:
            return None
        
        current_block = self.df.iloc[idx]
        
        angle_deg = 0

        if 'Curve' in current_block['Block']:
            pivot = np.array([current_block['pivot_x'], current_block['pivot_z']])
            if 'Left' in current_block['Rotate']:
                # compute tangent angle for left curve
                vec_from_pivot = np.array([-pos[0] - pivot[0], pos[1] - pivot[1]])
                angle_deg = np.degrees(np.arctan2(vec_from_pivot[1], vec_from_pivot[0])) + 90
            elif 'Right' in current_block['Rotate']:
                # compute tangent angle for right curve
                vec_from_pivot = np.array([-pos[0] - pivot[0], pos[1] - pivot[1]])
                angle_deg = np.degrees(np.arctan2(vec_from_pivot[1], vec_from_pivot[0])) - 90
        else:
            angle_deg = current_block['angle']

        return np.radians(angle_deg) % (2 * np.pi)
    
    def plot_sampled_angles(self, density=5):
        self._draw_blocks()
        for idx, row in self.df.iterrows():
            cx = row['cx']
            cz = row['cz']
            for i in range(density + 1):
                for j in range(density + 1):
                    sample_x = cx - self.block_size/2 + (i / density) * self.block_size
                    sample_z = cz - self.block_size/2 + (j / density) * self.block_size
                    print(sample_x, sample_z)
                    angle_deg = self.compute_line_angle([sample_x, sample_z])
                    print(angle_deg)
                    if angle_deg is not None:
                        angle_rad = np.radians(angle_deg)
                        dx = 5 * np.cos(angle_rad)
                        dz = 5 * np.sin(angle_rad)
                        plt.arrow(-sample_x, sample_z, dx, dz, head_width=1.0, head_length=1.5, fc='blue', ec='blue', alpha=0.6)
        plt.show()
    
    def compute_distance_to_finish(self, pos):
        idx = self._conpute_block_idx(pos)
        if idx is None:
            return None
        
        dist = 0.0

        corrected_pos = np.array([-pos[0], pos[1]])

        # Distance from current position to end of current block
        current_block = self.df.iloc[idx]

        end_point = np.array([current_block['output_x'], current_block['output_z']])
        dist += np.linalg.norm(end_point - corrected_pos)

        for j in range(idx + 1, len(self.df)):
            next_block = self.df.iloc[j]
            if 'Curve' in next_block['Block']:
                dist += np.pi * (self.block_size / 2) / 4 # Quarter circle arc length
            else : 
                dist += self.block_size
        
        return dist
     
    def distance_to_next_curve(self, pos):
        idx = self._conpute_block_idx(pos)
        if idx is None:
            return None
        
        dist = 0.0

        # Distance from current position to end of current block
        current_block = self.df.iloc[idx]
        if 'Curve' in current_block['Block']:
            block_center = np.array([-current_block['cx'], current_block['cz']])
            to_center_vec = block_center - np.array([-pos[0], pos[1]])
            dist = np.linalg.norm(to_center_vec)
            return dist
        
        # Check remaining blocks for next curve
        for j in range(idx + 1, len(self.df)):
            next_block = self.df.iloc[j]
            if 'Curve' in next_block['Block']:
                cx = - next_block['cx']
                cz = next_block['cz']
                block_center = np.array([cx, cz])
                print(block_center)
                to_center_vec = block_center - np.array([-pos[0], pos[1]])
                dist = np.linalg.norm(to_center_vec)
                return dist
        
        return None  # No more curves ahead

    def _next_curve_angle(self, pos):
        idx = self._conpute_block_idx(pos)
        print(idx)
        if idx is None:
            return None
        
        # Check remaining blocks for next curve
        for j in range(idx + 1, len(self.df)):
            next_block = self.df.iloc[j]
            if 'Curve' in next_block['Block']:
                # Compute angle at the start of this curve block
                angle = 1 if self.df.iloc[j]['Rotate'] == 'Right' else -1
                return angle
            
        return 0  # No more curves ahead

    def distance_to_median_line(self, pos):
        idx = self._conpute_block_idx(pos)
        if idx is None:
            return None

        corrected_pos = np.array([-pos[0], pos[1]])
        current_block = self.df.iloc[idx]
        
        
        if 'Curve' in current_block['Block']:
            pivot = current_block['pivot_x'], current_block['pivot_z']
            to_center_vec = corrected_pos - np.array([pivot[0], pivot[1]])
            dist_to_center = np.linalg.norm(to_center_vec)
            radius = self.block_size / 2
            dist_to_median = dist_to_center - radius
        else:
            ## Straight block
            block_center = np.array([-current_block['cx'], current_block['cz']])
            to_center_vec = corrected_pos - block_center
            if current_block['angle'] % 180 == 0:
                # East-West
                dist_to_median = np.abs(to_center_vec[1])  # Assuming median line
            else:
                # North-South
                dist_to_median = np.abs(to_center_vec[0])  # Assuming median line
        
        return dist_to_median
            

if __name__ == "__main__":
    # Use the track data provided
    data = """Block,X,Y,Z,Rotate,cx,cy,cz
    RoadTechStart,16,1,21,None,528.0,12.0,688.0
    RoadTechStraight,16,1,22,None,528.0,12.0,720.0
    RoadTechStraight,16,1,23,None,528.0,12.0,752.0
    RoadTechStraight,16,1,24,None,528.0,12.0,784.0
    RoadTechCurve1,16,1,25,Right,528.0,12.0,816.0
    RoadTechStraight,15,1,25,None,496.0,12.0,816.0
    RoadTechStraight,14,1,25,None,464.0,12.0,816.0
    RoadTechCurve1,13,1,25,Left,432.0,12.0,816.0
    RoadTechStraight,13,1,26,None,432.0,12.0,848.0
    RoadTechStraight,13,1,27,None,432.0,12.0,880.0
    RoadTechCurve1,13,1,28,Left,432.0,12.0,912.0
    RoadTechStraight,14,1,28,None,464.0,12.0,912.0
    RoadTechStraight,15,1,28,None,496.0,12.0,912.0
    RoadTechFinish,16,1,28,None,528.0,12.0,912.0"""

    track = TrackmaniaTrack(data)
    # track.plot_sampled_angles(density=3)

    track.compute_pivots()
    print(track._conpute_block_idx([528, 720]))  # Should return index of the block at (16,22)
    print(track.compute_line_angle([528, 720]))  # Should return 90 (North)
    print(track.compute_line_angle([496, 816]))  # Should return 0 (East)

    # Curve
    print(track.compute_line_angle([528, 816]))  # Should return angle tangent to right curve
    print(track.compute_line_angle([434, 816]))  # Should return angle tangent to right curve
    # track.plot_sampled_angles(density=5)

    print(track.compute_distance_to_finish([528, 800]))  # Distance from (16,22) to finish
    print(track.compute_distance_to_finish([510, 820]))  # Distance from (16,22) to finish


    print("Distance to next curve:", track.distance_to_next_curve([528, 816]))  # Distance to next curve from (16,22)
    print("Next curve angle:", track._next_curve_angle([500, 816]))  # Distance to next curve from (16,22)

    print("Distance to median line:", track.distance_to_median_line([528, 816]))  # Distance to median line from (16,22)
    print("Distance to median line:", track.distance_to_median_line([500, 818]))  # Distance to median line from (16,22)

    print("distance to finish:", track.compute_distance_to_finish([528, 688]))
    print("distance to finish:", track.compute_distance_to_finish([528, 816]))
    print("distance to finish:", track.compute_distance_to_finish([434, 816]))