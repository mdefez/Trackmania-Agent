import pandas as pd
import numpy as np


def _world_to_block(pos) :
    return pos[0] / 32, pos[1] / 32

def _block_to_world(pos):
    return pos[0] * 32, pos[1] * 32


def distance_and_angle_to_racing_line_blocks(car_pos, car_heading, racing_line):
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

## Compute distance from car to end on racing line
def distance_to_end_on_racing_line_blocks(car_pos, racing_line):
    car_pos = np.array(car_pos)
    min_dist = np.inf
    segment_idx = None
    t_closest = None
    
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
            segment_idx = i
            t_closest = t_clamped / seg_len
    
    # Compute distance to end of racing line
    dist_to_end = 0.0
    if segment_idx is not None:
        # Add remaining distance on current segment
        p1 = racing_line[segment_idx]
        p2 = racing_line[segment_idx+1]
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        dist_to_end += (1 - t_closest) * seg_len
        
        # Add distances of remaining segments
        for j in range(segment_idx+1, len(racing_line)-1):
            p1 = racing_line[j]
            p2 = racing_line[j+1]
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            dist_to_end += seg_len
    
    return dist_to_end

def distance_to_next_curve_blocks(car_pos, racing_line):
    car_pos = np.array(car_pos)
    min_dist = np.inf
    segment_idx = None
    
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
            segment_idx = i
    
    # Find next curve (change in direction)
    next_curve_idx = None
    for j in range(segment_idx+1, len(racing_line)-2):
        p1 = racing_line[j]
        p2 = racing_line[j+1]
        p3 = racing_line[j+2]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        angle_change = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angle_change = (angle_change + np.pi) % (2*np.pi) - np.pi
        
        if abs(angle_change) > 0.1:  # threshold for curve detection
            next_curve_idx = j+1
            break
    
    if next_curve_idx is not None:
        dist_to_curve = 0.0
        # Add remaining distance on current segment
        p1 = racing_line[segment_idx]
        p2 = racing_line[segment_idx+1]
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        v = car_pos - p1
        t = np.dot(v, seg_vec / seg_len)
        t_clamped = np.clip(t, 0, seg_len)
        dist_to_curve += seg_len - t_clamped
        
        # Add distances of segments until next curve
        for k in range(segment_idx+1, next_curve_idx):
            p1 = racing_line[k]
            p2 = racing_line[k+1]
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            dist_to_curve += seg_len
        # Compute angle at next curve
        p_prev = racing_line[next_curve_idx-1]
        p_curr = racing_line[next_curve_idx]
        p_next = racing_line[next_curve_idx+1]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        curve_angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        curve_angle = (curve_angle + np.pi) % (2*np.pi) - np.pi
        return dist_to_curve, curve_angle
    else:
        return None, None


def line_ref_loss_world(car_pos, car_heading, racing_line, k=1.0):
    d, _, _, _, theta = distance_and_angle_to_racing_line_blocks(_world_to_block(car_pos), car_heading, racing_line)
    print("Distance to line:", d, "Angle to line:", theta)
    loss = (d * 15) ** 2 + k * (theta * 15) ** 2
    # print("Loss:", loss)
    return loss, d

def end_racing_line_loss_world(car_pos, racing_line):
    d = distance_to_end_on_racing_line_blocks(_world_to_block(car_pos), racing_line)
    return d

def _curve_feature_world(car_pos, racing_line):
    dist_to_curve, curve_angle = distance_to_next_curve_blocks(_world_to_block(car_pos), racing_line)
    return dist_to_curve, curve_angle

if __name__ == "__main__":
    df = pd.read_csv("../../../setup/data/blocks.csv", sep=',')
    racing_line = df[['X','Z']].to_numpy()

    # Example car positions and headings
    car_positions = [
        [16, 21, 0],
        [15, 25, 0],
        [13, 26, 0],
        [13, 28, 0]
    ]

    car_headings = [
        np.pi/2,
        np.pi,
        np.pi/2,
        np.pi/3
    ]

    for pos, heading in zip(car_positions, car_headings):
        print(pos)
        line_loss = line_ref_loss_world(_block_to_world(pos), heading, racing_line)
        end_loss = end_racing_line_loss_world(_block_to_world(pos), racing_line)
        dist_to_curve, curve_angle = _curve_feature_world(_block_to_world(pos), racing_line)
        
        print("World car pos" f"{_block_to_world(pos)}, car heading {heading}")
        print("Curve feature: ", (dist_to_curve, curve_angle))
        print("end_loss:", end_loss)
        print("line_loss:", line_loss)
        