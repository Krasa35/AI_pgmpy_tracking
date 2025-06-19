import numpy as np
import re
from collections import defaultdict
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.optimize import linear_sum_assignment
import math
import cv2
import os
import sys

class BBox:
    """Represents a bounding box with tracking information"""
    def __init__(self, x, y, w, h, frame_num, track_id=-1, image_region=None, velocity_x=0, velocity_y=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame_num = frame_num
        self.track_id = track_id
        self.center_x = x + w / 2
        self.center_y = y + h / 2
        self.area = w * h
        self.image_region = image_region  # Store cropped image region for color analysis
        self.velocity_x = velocity_x  # Velocity in x direction
        self.velocity_y = velocity_y  # Velocity in y direction
        self.velocity_magnitude = math.sqrt(velocity_x*velocity_x + velocity_y*velocity_y)
    
    def __repr__(self):
        return f"BBox(x={self.x:.2f}, y={self.y:.2f}, w={self.w:.2f}, h={self.h:.2f}, id={self.track_id})"

class Frame:
    def __init__(self, frame_num, frame_name, bboxes):
        self.bboxes = bboxes  # List of BBox objects
        self.frame_num = frame_num  # Frame number (sequential)
        self.frame_name = frame_name  # Frame name (e.g., "frame_0001.jpg")

    def __repr__(self):
        return f"Frame(num={self.frame_num}, name={self.frame_name}, bboxes={self.bboxes})"

class ProbabilisticTracker:
    """Probabilistic tracker using Bayesian networks"""
    
    def __init__(self, max_distance=100, max_frames_missing=10, iou_threshold=0.3):
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks = {}  # track_id -> list of BBox objects
        self.last_seen = {}  # track_id -> frame_number
        self.previous_frame_ids = set()  # Track IDs from previous frame
        
        # Create Bayesian Network for tracking
        self.model = self._create_bayesian_network()
        self.inference = VariableElimination(self.model)
    
    def _create_bayesian_network(self):
        """Create a Bayesian Network for object tracking"""
        # Define the network structure
        model = DiscreteBayesianNetwork([
            ('Pos_prev', 'Pos_curr'),
            ('Size_prev', 'Size_curr'),
            ('Color_prev', 'Color_curr'),
            ('Velocity_prev', 'Velocity_curr'),
            ('Pos_curr', 'Obs_pos'),
            ('Size_curr', 'Obs_size'),
            ('Color_curr', 'Obs_color'),
            ('Velocity_curr', 'Obs_velocity')
        ])

        # Define CPDs (Conditional Probability Distributions)

        # Position states: 0=same, 1=near, 2=far
        pos_prev_cpd = TabularCPD(
            variable='Pos_prev',
            variable_card=3,
            values=[[0.4], [0.4], [0.2]]
        )

        pos_curr_cpd = TabularCPD(
            variable='Pos_curr',
            variable_card=3,
            values=[
                [0.7, 0.2, 0.1],
                [0.25, 0.6, 0.3],
                [0.05, 0.2, 0.6]
            ],
            evidence=['Pos_prev'],
            evidence_card=[3]
        )

        # Size states: 0=same, 1=similar, 2=different
        size_prev_cpd = TabularCPD(
            variable='Size_prev',
            variable_card=3,
            values=[[0.5], [0.3], [0.2]]
        )

        size_curr_cpd = TabularCPD(
            variable='Size_curr',
            variable_card=3,
            values=[
                [0.8, 0.3, 0.1],
                [0.15, 0.6, 0.3],
                [0.05, 0.1, 0.6]
            ],
            evidence=['Size_prev'],
            evidence_card=[3]
        )

        # Color states: 0=same, 1=similar, 2=different
        color_prev_cpd = TabularCPD(
            variable='Color_prev',
            variable_card=3,
            values=[[0.6], [0.3], [0.1]]
        )

        color_curr_cpd = TabularCPD(
            variable='Color_curr',
            variable_card=3,
            values=[
                [0.85, 0.4, 0.1],
                [0.1, 0.5, 0.3],
                [0.05, 0.1, 0.6]
            ],
            evidence=['Color_prev'],
            evidence_card=[3]
        )

        # Velocity states: 0=same, 1=similar, 2=different
        velocity_prev_cpd = TabularCPD(
            variable='Velocity_prev',
            variable_card=3,
            values=[[0.7], [0.2], [0.1]]
        )

        velocity_curr_cpd = TabularCPD(
            variable='Velocity_curr',
            variable_card=3,
            values=[
                [0.8, 0.3, 0.1],
                [0.15, 0.6, 0.3],
                [0.05, 0.1, 0.6]
            ],
            evidence=['Velocity_prev'],
            evidence_card=[3]
        )

        # Observation models
        obs_pos_cpd = TabularCPD(
            variable='Obs_pos',
            variable_card=3,
            values=[
                [0.9, 0.3, 0.1],
                [0.08, 0.6, 0.3],
                [0.02, 0.1, 0.6]
            ],
            evidence=['Pos_curr'],
            evidence_card=[3]
        )

        obs_size_cpd = TabularCPD(
            variable='Obs_size',
            variable_card=3,
            values=[
                [0.85, 0.4, 0.1],
                [0.12, 0.5, 0.3],
                [0.03, 0.1, 0.6]
            ],
            evidence=['Size_curr'],
            evidence_card=[3]
        )

        obs_color_cpd = TabularCPD(
            variable='Obs_color',
            variable_card=3,
            values=[
                [0.9, 0.4, 0.1],
                [0.08, 0.5, 0.3],
                [0.02, 0.1, 0.6]
            ],
            evidence=['Color_curr'],
            evidence_card=[3]
        )

        obs_velocity_cpd = TabularCPD(
            variable='Obs_velocity',
            variable_card=3,
            values=[
                [0.85, 0.4, 0.1],
                [0.12, 0.5, 0.3],
                [0.03, 0.1, 0.6]
            ],
            evidence=['Velocity_curr'],
            evidence_card=[3]
        )

        # Add CPDs to the model
        model.add_cpds(
            pos_prev_cpd, pos_curr_cpd, size_prev_cpd, size_curr_cpd,
            color_prev_cpd, color_curr_cpd, velocity_prev_cpd, velocity_curr_cpd,
            obs_pos_cpd, obs_size_cpd, obs_color_cpd, obs_velocity_cpd
        )
        # model_graphviz = model.to_graphviz()
        # model_graphviz.draw("model.png", prog="dot")

        return model
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate Euclidean distance between bbox centers"""
        return math.sqrt((bbox1.center_x - bbox2.center_x)**2 + 
                        (bbox1.center_y - bbox2.center_y)**2)
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1 = max(bbox1.x, bbox2.x)
        y1 = max(bbox1.y, bbox2.y)
        x2 = min(bbox1.x + bbox1.w, bbox2.x + bbox2.w)
        y2 = min(bbox1.y + bbox1.h, bbox2.y + bbox2.h)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = bbox1.area + bbox2.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_position_state(self, distance):
        """Convert distance to discrete state"""
        if distance < 80:
            return 0  # same
        elif distance < 150:
            return 1  # near
        else:
            return 2  # far
    
    def get_size_state(self, area1, area2):
        """Convert size difference to discrete state"""
        if area1 == 0 or area2 == 0:
            return 2  # different
        
        ratio = min(area1, area2) / max(area1, area2)
        if ratio > 0.6:
            return 0  # same
        elif ratio > 0.4:
            return 1  # similar
        else:
            return 2  # different
    
    def compare_color(self, bbox1, bbox2):
        """Compare color based on histogram analysis using OpenCV"""
        # Check if both bounding boxes have image regions
        if bbox1.image_region is None or bbox2.image_region is None:
            return 2  # different color (no image region available)
        
        # Calculate color histograms using OpenCV
        # Convert to HSV for better color representation
        hsv1 = cv2.cvtColor(bbox1.image_region, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(bbox2.image_region, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for H, S, V channels
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Compare histograms using correlation method
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Convert correlation to discrete color state
        if correlation > 0.65:
            return 0  # same color (high correlation)
        elif correlation > 0.4:
            return 1  # similar color (medium correlation)
        else:
            return 2  # different color (low correlation)
    
    def compare_velocity(self, bbox1, bbox2):
        """Compare velocity between two consecutive bounding boxes"""
        # Compare the stored velocity magnitudes
        vel1_mag = bbox1.velocity_magnitude
        vel2_mag = bbox2.velocity_magnitude
        
        # Calculate the difference in velocity magnitudes
        vel_diff = abs(vel2_mag - vel1_mag)
        
        # Also consider the direction difference
        if vel1_mag > 0 and vel2_mag > 0:  # Both have non-zero velocity
            # Calculate angle difference
            angle1 = math.atan2(bbox1.velocity_y, bbox1.velocity_x)
            angle2 = math.atan2(bbox2.velocity_y, bbox2.velocity_x)
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Get smaller angle
            
            # Weight the difference by both magnitude and direction
            combined_diff = vel_diff + (angle_diff / math.pi) * max(vel1_mag, vel2_mag)
        else:
            combined_diff = vel_diff
        
        # Convert to discrete velocity state
        if combined_diff < 5:
            return 0  # same velocity
        elif combined_diff < 15:
            return 1  # similar velocity
        else:
            return 2  # different velocity
    
    def calculate_probability(self, prev_bbox, curr_bbox):
        """Calculate probability of association using Bayesian network"""
        if prev_bbox is None:
            return 0.1  # Low probability for new tracks
        
        distance = self.calculate_distance(prev_bbox, curr_bbox)
        pos_state = self.get_position_state(distance)
        size_state = self.get_size_state(prev_bbox.area, curr_bbox.area)
        
        color_state = self.compare_color(prev_bbox, curr_bbox)  # Actual color comparison logic
        velocity_state = self.compare_velocity(prev_bbox, curr_bbox)  # Compare stored velocities
        
        # Set evidence and query the network
        evidence = {
            'Obs_pos': pos_state,
            'Obs_size': size_state,
            'Obs_color': color_state,
            'Obs_velocity': velocity_state
        }
        
        try:
            # Query probability of current position, size, color, and velocity states
            prob_pos = self.inference.query(['Pos_curr'], evidence=evidence)
            prob_size = self.inference.query(['Size_curr'], evidence=evidence)
            prob_color = self.inference.query(['Color_curr'], evidence=evidence)
            prob_velocity = self.inference.query(['Velocity_curr'], evidence=evidence)
            
            # Combine probabilities (simple multiplication for now)
            combined_prob = (
                prob_pos.values[0] * prob_size.values[0] * prob_color.values[0] * prob_velocity.values[0] +
                prob_pos.values[1] * prob_size.values[1] * prob_color.values[1] * prob_velocity.values[1] * 0.5
            )
            
            return combined_prob
        except:
            # Fallback to distance-based probability
            return max(0.1, 1.0 - distance / self.max_distance)
    
    def update_velocity(self, bbox, prev_bbox):
        """Update velocity of bbox based on movement from previous bbox"""
        if prev_bbox is None:
            bbox.velocity_x = 0
            bbox.velocity_y = 0
            bbox.velocity_magnitude = 0
        else:
            # Calculate velocity based on displacement and frame difference
            frame_diff = max(bbox.frame_num - prev_bbox.frame_num, 1)
            bbox.velocity_x = (bbox.center_x - prev_bbox.center_x) / frame_diff
            bbox.velocity_y = (bbox.center_y - prev_bbox.center_y) / frame_diff
            bbox.velocity_magnitude = math.sqrt(bbox.velocity_x*bbox.velocity_x + bbox.velocity_y*bbox.velocity_y)

    def track_frame(self, detections, frame_num):
        """Track objects in a single frame"""
        # Remove old box IDs that are no longer active
        self.tracks = {track_id: track_history for track_id, track_history in self.tracks.items()
                       if frame_num - self.last_seen[track_id] <= self.max_frames_missing}
        self.last_seen = {track_id: last_seen for track_id, last_seen in self.last_seen.items()
                          if frame_num - last_seen <= self.max_frames_missing}

        if not self.tracks:
            # First frame or no active tracks - assign new IDs to all detections
            for detection in detections:
                detection.track_id = self.next_id  # Assign new ID
                self.tracks[self.next_id] = [detection]  # Add to tracks
                self.last_seen[self.next_id] = frame_num  # Update last seen
                self.next_id += 1  # Increment next ID
            return detections

        # Get active tracks (seen recently)
        active_tracks = {track_id: track_history[-1] for track_id, track_history in self.tracks.items()}

        # Create cost matrix for Hungarian algorithm
        track_ids = list(active_tracks.keys())
        cost_matrix = np.zeros((len(detections), len(track_ids)))

        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                prev_bbox = active_tracks[track_id]
                prob = self.calculate_probability(prev_bbox, detection)
                cost_matrix[i, j] = 1.0 - prob  # Convert to cost

        # Solve assignment problem
        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        # Process assignments
        assigned_detections = set()
        assigned_tracks = set()

        for det_idx, track_idx in zip(det_indices, track_indices):
            cost = cost_matrix[det_idx, track_idx]
            if cost < 0.995:  # Threshold for valid assignment
                track_id = track_ids[track_idx]
                prev_bbox = active_tracks[track_id]
                
                # Update velocity before assigning
                self.update_velocity(detections[det_idx], prev_bbox)
                
                detections[det_idx].track_id = track_id
                self.tracks[track_id].append(detections[det_idx])
                self.last_seen[track_id] = frame_num
                assigned_detections.add(det_idx)
                assigned_tracks.add(track_id)

        # Handle unassigned detections (new tracks)
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                # Assign new track ID to unassigned detections
                detection.track_id = self.next_id
                self.tracks[self.next_id] = [detection]
                self.last_seen[self.next_id] = frame_num
                self.next_id += 1

        return detections

    def extract_image_region(self, image, bbox):
        """Extract the image region corresponding to a bounding box"""
        try:
            # Ensure coordinates are within image bounds
            x_start = max(0, int(bbox.x))
            y_start = max(0, int(bbox.y))
            x_end = min(image.shape[1], int(bbox.x + bbox.w))
            y_end = min(image.shape[0], int(bbox.y + bbox.h))
            
            # Extract the region
            region = image[y_start:y_end, x_start:x_end]
            
            # Ensure minimum size for histogram calculation
            if region.shape[0] < 10 or region.shape[1] < 10:
                return None
                
            return region
        except:
            return None

    def get_display_id(self, track_id, detection_index):
        """Get the display ID - returns detection index if ID existed in previous frame, -1 if new"""
        if track_id in self.previous_frame_ids:
            return detection_index  # Use array index for continuing tracks
        else:
            return -1  # New detection
    
    def update_previous_frame_ids(self, current_frame_ids):
        """Update the set of IDs from the current frame for next frame comparison"""
        self.previous_frame_ids = set(current_frame_ids)

def parse_bboxes_file(data_folder) -> list[Frame]:
    """Parse the bboxes.txt file and return a list of Frame objects"""
    frames_data = []

    with open(f'{data_folder}/bboxes.txt', 'r') as f:
        lines = f.readlines()

    frame_counter = 0  # Sequential frame counter
    i = 0
    while i < len(lines):
        if lines[i].strip().endswith('.jpg'):
            frame_name = lines[i].strip()
            i += 1

            if i < len(lines):
                num_bboxes = int(lines[i].strip())
                i += 1

                # Load the actual image for color analysis
                image_path = f"{data_folder}/frames/{frame_name}"
                image = None
                try:
                    image = cv2.imread(image_path)
                except:
                    print(f"Warning: Could not load image {image_path}")

                bboxes = []
                for _ in range(num_bboxes):
                    if i < len(lines):
                        coords = list(map(float, lines[i].strip().split()))
                        x, y, w, h = coords

                        # Extract image region for color analysis
                        image_region = None
                        if image is not None:
                            tracker = ProbabilisticTracker()  # Temporary instance for helper method
                            image_region = tracker.extract_image_region(image, BBox(x, y, w, h, frame_counter))

                        bbox = BBox(x, y, w, h, frame_counter, image_region=image_region)
                        bboxes.append(bbox)
                        i += 1

                frames_data.append(Frame(frame_num=frame_counter, 
                                         frame_name=frame_name,
                                         bboxes=bboxes))

            frame_counter += 1  # Increment frame counter for each new frame
        else:
            i += 1

    return frames_data

def main(debug=False):
    """Main function to run the tracking system"""
    # Parse the input file
    if len(sys.argv) < 2:
        print("Error: Please provide the data folder as an argument.")
        return
    data_folder = sys.argv[1]
    if not os.path.exists(data_folder):
        print("Error: 'data' folder not found.")
        return
    frames_data = parse_bboxes_file(data_folder)
    
    # Initialize tracker
    tracker = ProbabilisticTracker()

    if debug:
        print("Pedestrian and Cyclist Tracking Results:")
        print("=" * 50)
    
    for frame in frames_data:
        detections = frame.bboxes
        frame_num = frame.frame_num
        
        # Track objects in this frame
        tracked_detections = tracker.track_frame(detections, frame_num)
        
        # Get current frame IDs and determine display IDs
        current_frame_ids = [bbox.track_id for bbox in tracked_detections if bbox.track_id != -1]

        if debug:
            # Display the frame with bounding boxes
            frame_img = cv2.imread(f'{data_folder}/frames/{frame.frame_name}')
            for i, bbox in enumerate(tracked_detections):
                display_id = tracker.get_display_id(bbox.track_id, i)
                color = (0, 255, 0) if display_id != -1 else (0, 0, 255)
                
                cv2.rectangle(frame_img, (int(bbox.x), int(bbox.y)), 
                    (int(bbox.x + bbox.w), int(bbox.y + bbox.h)), color, 2)
                cv2.putText(frame_img, f"ID: {display_id}", 
                    (int(bbox.x), int(bbox.y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imshow("Tracking", frame_img)
            cv2.waitKey(10)

            print(f"\nFrame: {frame.frame_name}")
            print(f"Number of detections: {len(tracked_detections)}")

        print_string = ''
        
        for i, bbox in enumerate(tracked_detections):
            display_id = tracker.get_display_id(bbox.track_id, i)
            if display_id == -1:
                print_string += '-1 '
                if debug:
                    print(f"  BBox {i+1}: ID = -1 (new detection, original track_id: {bbox.track_id})")
            else:
                print_string += f'{display_id} '
                if debug:
                    print(f"  BBox {i+1}: ID = {display_id} (continuing track, original track_id: {bbox.track_id})")
            if debug:
                print(f"    Coordinates: ({bbox.x:.2f}, {bbox.y:.2f}, {bbox.w:.2f}, {bbox.h:.2f})")
        # Update previous frame IDs for next iteration
        if not debug:
            print(print_string.strip())
        tracker.update_previous_frame_ids(current_frame_ids)

if __name__ == "__main__":
    main(debug=False)  # Set debug=True to see detailed output and frame images