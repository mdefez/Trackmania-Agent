import pandas as pd
import numpy as np

def distance_and_angle_to_racing_line(car_pos, car_heading, racing_line):
    car_pos = np.array(car_pos)
    min_dist = np.inf
    closest_point = None
    segment_idx = None
    t_closest = None
    tangent_vec = None
    
    for i in range(len(racing_line)-1):
        p1 = racing_line[i]
        p2 = racing_line[i+1]
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len == 0:
            continue
        seg_dir = seg_vec / seg_len
        
        # Project car position onto segment
        v = car_pos - p1
        t = np.dot(v, seg_dir)
        t_clamped = np.clip(t, 0, seg_len)
        proj = p1 + seg_dir * t_clamped
        
        dist = np.linalg.norm(car_pos - proj)
        if dist < min_dist:
            min_dist = dist
            closest_point = proj
            segment_idx = i
            t_closest = t_clamped / seg_len
            tangent_vec = seg_dir
    
    # Angle between car heading and racing line tangent
    theta = car_heading - np.arctan2(tangent_vec[1], tangent_vec[0])
    theta = (theta + np.pi) % (2*np.pi) - np.pi
    
    return min_dist, closest_point, segment_idx, t_closest, theta

def line_ref_loss(car_pos, car_heading, racing_line, k=1.0):
    d, _, _, _, theta = distance_and_angle_to_racing_line(car_pos, car_heading, racing_line)
    loss = d**2 + k * theta**2
    return loss

if __name__ == "__main__":
    df = pd.read_csv("clean_blocks.csv", sep=';')
    racing_line = df[['X','Z']].to_numpy()

    # Example car positions and headings
    car_positions = [
        [16, 22],
        [15.5, 23.2],
        [14.8, 25]
    ]

    car_headings = [
        np.pi/2,  # North
        np.pi/2,
        np.pi/2
    ]

    for pos, heading in zip(car_positions, car_headings):
        loss = line_ref_loss(pos, heading, racing_line, k=1.0)
        print(f"Car at {pos}, heading {np.degrees(heading):.1f}° -> Loss: {loss:.3f}")
