"""
Restaurant Crowd Analysis AI System using YOLOv8
Real-time crowd monitoring and analytics with computer vision
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import json
from ultralytics import YOLO

class RestaurantCrowdAnalyzer:
    def __init__(self, video_source=0, max_capacity=50, model_path='yolov8n.pt'):
        """
        Initialize the crowd analyzer with YOLOv8
        
        Args:
            video_source: Camera index or video file path
            max_capacity: Maximum restaurant capacity
            model_path: YOLOv8 model path (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
        """
        self.video_source = video_source
        self.max_capacity = max_capacity
        self.cap = None
        
        # Initialize YOLOv8
        print(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        print("✓ YOLOv8 model loaded successfully")
        
        # Analytics data
        self.people_count_history = deque(maxlen=300)  # Last 10 seconds at 30fps
        self.entry_line = None
        self.exit_line = None
        self.entries = 0
        self.exits = 0
        self.tracked_objects = {}
        self.next_object_id = 0
        
        # Heatmap for crowd density
        self.heatmap = None
        self.heatmap_accumulator = None
        
        # Define zones (customizable - x1, y1, x2, y2)
        self.zones = {
            'entrance': None,
            'dining_area': None,
            'waiting_area': None,
            'exit': None
        }
        
        # Performance metrics
        self.avg_wait_time = 0
        self.peak_count = 0
        self.analytics_data = []
        
    def setup_zones(self, frame_width, frame_height):
        """Auto-setup zones based on frame dimensions"""
        self.zones = {
            'entrance': (0, 0, frame_width//4, frame_height),
            'waiting_area': (frame_width//4, 0, frame_width//2, frame_height//3),
            'dining_area': (frame_width//4, frame_height//3, 3*frame_width//4, frame_height),
            'exit': (3*frame_width//4, 0, frame_width, frame_height)
        }
        
        # Entry/exit lines for flow tracking
        self.entry_line = frame_width // 4
        self.exit_line = 3 * frame_width // 4
    
    def detect_people(self, frame, conf_threshold=0.5):
        """
        Detect people using YOLOv8
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold for detection
        
        Returns:
            List of detections with boxes and confidence
        """
        # Run YOLOv8 inference
        results = self.model(frame, conf=conf_threshold, classes=[0], verbose=False)
        # classes=[0] filters for 'person' class only
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                center_x = x + w // 2
                center_y = y + h // 2
                
                detections.append({
                    'box': (x, y, w, h),
                    'confidence': confidence,
                    'center': (center_x, center_y),
                    'id': None  # Will be assigned by tracker
                })
        
        return detections
    
    def simple_tracker(self, detections, max_distance=50):
        """
        Simple object tracking using centroid distance
        
        Args:
            detections: List of current frame detections
            max_distance: Maximum distance to consider same object
        """
        if not self.tracked_objects:
            # Initialize tracking for first frame
            for det in detections:
                det['id'] = self.next_object_id
                self.tracked_objects[self.next_object_id] = {
                    'center': det['center'],
                    'last_seen': 0,
                    'crossed_entry': False,
                    'crossed_exit': False
                }
                self.next_object_id += 1
            return detections
        
        # Match detections to tracked objects
        matched = set()
        for det in detections:
            min_dist = float('inf')
            matched_id = None
            
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_id in matched:
                    continue
                    
                dist = np.sqrt(
                    (det['center'][0] - obj_data['center'][0])**2 +
                    (det['center'][1] - obj_data['center'][1])**2
                )
                
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    matched_id = obj_id
            
            if matched_id is not None:
                det['id'] = matched_id
                matched.add(matched_id)
                self.tracked_objects[matched_id]['center'] = det['center']
                self.tracked_objects[matched_id]['last_seen'] = 0
                
                # Check for entry/exit crossing
                if (self.entry_line and 
                    not self.tracked_objects[matched_id]['crossed_entry'] and
                    det['center'][0] > self.entry_line):
                    self.tracked_objects[matched_id]['crossed_entry'] = True
                    self.entries += 1
                
                if (self.exit_line and 
                    not self.tracked_objects[matched_id]['crossed_exit'] and
                    det['center'][0] > self.exit_line):
                    self.tracked_objects[matched_id]['crossed_exit'] = True
                    self.exits += 1
            else:
                # New object
                det['id'] = self.next_object_id
                self.tracked_objects[self.next_object_id] = {
                    'center': det['center'],
                    'last_seen': 0,
                    'crossed_entry': False,
                    'crossed_exit': False
                }
                self.next_object_id += 1
        
        # Remove old tracked objects
        to_remove = []
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['last_seen'] += 1
            if self.tracked_objects[obj_id]['last_seen'] > 30:  # 1 second at 30fps
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
        
        return detections
    
    def update_heatmap(self, frame_shape, detections):
        """Generate crowd density heatmap"""
        if self.heatmap_accumulator is None:
            self.heatmap_accumulator = np.zeros(
                (frame_shape[0], frame_shape[1]), 
                dtype=np.float32
            )
        
        # Decay existing heatmap
        self.heatmap_accumulator *= 0.98
        
        # Add new detections with Gaussian blur
        for det in detections:
            cx, cy = det['center']
            if 0 <= cx < frame_shape[1] and 0 <= cy < frame_shape[0]:
                cv2.circle(self.heatmap_accumulator, (cx, cy), 40, (1,), -1)
        
        # Apply Gaussian blur for smooth heatmap
        self.heatmap = cv2.GaussianBlur(self.heatmap_accumulator, (51, 51), 0)
    
    def count_by_zone(self, detections):
        """Count people in each defined zone"""
        zone_counts = {zone: 0 for zone in self.zones}
        
        for det in detections:
            cx, cy = det['center']
            for zone_name, zone_coords in self.zones.items():
                if zone_coords is None:
                    continue
                x1, y1, x2, y2 = zone_coords
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    zone_counts[zone_name] += 1
                    break
        
        return zone_counts
    
    def get_occupancy_level(self, count):
        """Calculate occupancy level and color"""
        percentage = (count / self.max_capacity) * 100
        
        if percentage < 40:
            return "Low", (0, 255, 0), percentage
        elif percentage < 70:
            return "Moderate", (0, 200, 255), percentage
        elif percentage < 90:
            return "High", (0, 140, 255), percentage
        else:
            return "Critical", (0, 0, 255), percentage
    
    def draw_analytics(self, frame, detections):
        """Draw comprehensive analytics overlay"""
        people_count = len(detections)
        self.people_count_history.append(people_count)
        
        if people_count > self.peak_count:
            self.peak_count = people_count
        
        # Draw detections with IDs
        for det in detections:
            x, y, w, h = det['box']
            color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw ID and confidence
            if det['id'] is not None:
                label = f"ID:{det['id']} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            cv2.circle(frame, det['center'], 4, (0, 0, 255), -1)
        
        # Draw zones
        for zone_name, zone_coords in self.zones.items():
            if zone_coords is None:
                continue
            x1, y1, x2, y2 = zone_coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(frame, zone_name.replace('_', ' ').title(), 
                       (x1 + 5, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        
        # Draw entry/exit lines
        if self.entry_line:
            cv2.line(frame, (self.entry_line, 0), 
                    (self.entry_line, frame.shape[0]), (0, 255, 255), 2)
            cv2.putText(frame, "ENTRY", (self.entry_line + 5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.exit_line:
            cv2.line(frame, (self.exit_line, 0), 
                    (self.exit_line, frame.shape[0]), (255, 0, 255), 2)
            cv2.putText(frame, "EXIT", (self.exit_line + 5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Zone counts
        zone_counts = self.count_by_zone(detections)
        
        # Create analytics panel
        panel_height = 220
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        # Main metrics
        occupancy_level, color, percentage = self.get_occupancy_level(people_count)
        
        cv2.putText(panel, f"CURRENT COUNT: {people_count}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(panel, f"Capacity: {self.max_capacity} | Occupancy: {percentage:.1f}%", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Occupancy bar
        bar_width = 300
        bar_height = 20
        bar_x, bar_y = 20, 75
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (80, 80, 80), -1)
        fill_width = int(bar_width * (percentage / 100))
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                     color, -1)
        cv2.putText(panel, occupancy_level, (bar_x + bar_width + 10, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Flow metrics
        cv2.putText(panel, f"Entries: {self.entries} | Exits: {self.exits} | Net: {self.entries - self.exits}", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(panel, f"Peak Count: {self.peak_count}", 
                   (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
        
        # Zone breakdown
        y_offset = 170
        zone_text = " | ".join([f"{name.replace('_', ' ').title()}: {count}" 
                                for name, count in zone_counts.items()])
        cv2.putText(panel, f"Zones: {zone_text}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)
        
        # Average count
        if len(self.people_count_history) > 0:
            avg_count = np.mean(list(self.people_count_history))
            cv2.putText(panel, f"Avg (10s): {avg_count:.1f}", (20, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Trend graph
        if len(self.people_count_history) > 1:
            graph_width = 400
            graph_height = 120
            graph_x = frame.shape[1] - graph_width - 20
            graph_y = 20
            
            # Graph background
            cv2.rectangle(panel, (graph_x, graph_y), 
                         (graph_x + graph_width, graph_y + graph_height),
                         (50, 50, 50), -1)
            cv2.rectangle(panel, (graph_x, graph_y), 
                         (graph_x + graph_width, graph_y + graph_height),
                         (100, 100, 100), 2)
            
            # Plot line
            max_val = max(self.people_count_history) if max(self.people_count_history) > 1 else 1
            points = []
            for i, val in enumerate(self.people_count_history):
                x = graph_x + int(i * graph_width / len(self.people_count_history))
                y = graph_y + graph_height - int(val * graph_height / max_val) - 5
                points.append((x, max(graph_y, y)))
            
            for i in range(len(points) - 1):
                cv2.line(panel, points[i], points[i + 1], (0, 255, 150), 2)
            
            cv2.putText(panel, "Count Trend (10s)", (graph_x + 5, graph_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(panel, f"Max: {max_val}", (graph_x + graph_width - 80, graph_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, timestamp, (frame.shape[1] - 220, panel_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
        # Combine frame and panel
        combined = np.vstack([frame, panel])
        
        return combined
    
    def run(self, show_heatmap=True, save_video=False):
        """
        Main processing loop
        
        Args:
            show_heatmap: Display heatmap overlay
            save_video: Save output video to file
        """
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            print("✗ Error: Could not open video source")
            return
        
        # Get video properties
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup zones based on frame size
        self.setup_zones(frame_width, frame_height)
        
        # Video writer
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_filename = f'crowd_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            video_writer = cv2.VideoWriter(output_filename, fourcc, fps, 
                                          (frame_width, frame_height + 220))
            print(f"✓ Recording to {output_filename}")
        
        print("\n" + "="*60)
        print("Restaurant Crowd Analysis System - YOLOv8")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save analytics snapshot")
        print("  'r' - Reset counters")
        print("  'h' - Toggle heatmap")
        print("="*60 + "\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.video_source, str):
                        # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                frame_count += 1
                
                # Detect people with YOLOv8
                detections = self.detect_people(frame, conf_threshold=0.45)
                
                # Track objects
                detections = self.simple_tracker(detections)
                
                # Update heatmap
                if show_heatmap:
                    self.update_heatmap(frame.shape, detections)
                
                # Draw analytics
                output_frame = self.draw_analytics(frame, detections)
                
                # Show heatmap overlay
                if show_heatmap and self.heatmap is not None:
                    # Normalize floating-point heatmap to 0-255 and convert to uint8 single-channel
                    heatmap_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap_uint8 = heatmap_norm.astype(np.uint8)
                    # applyColorMap expects an 8-bit single-channel (grayscale) image or a 3-channel image
                    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    heatmap_resized = cv2.resize(heatmap_colored, (250, 150))
                    
                    # Position heatmap on frame
                    h, w = heatmap_resized.shape[:2]
                    output_frame[20:20+h, frame.shape[1]-w-20:frame.shape[1]-20] = heatmap_resized
                    cv2.putText(output_frame, "Density Heatmap", 
                               (frame.shape[1]-w-15, 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Save frame
                if video_writer:
                    video_writer.write(output_frame)
                
                # Display
                cv2.imshow('Restaurant Crowd Analysis - YOLOv8', output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_analytics()
                elif key == ord('r'):
                    self.entries = 0
                    self.exits = 0
                    self.peak_count = 0
                    print("✓ Counters reset")
                elif key == ord('h'):
                    show_heatmap = not show_heatmap
                    print(f"✓ Heatmap {'enabled' if show_heatmap else 'disabled'}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            if video_writer:
                video_writer.release()
                print("✓ Video saved")
            cv2.destroyAllWindows()
            
            # Final analytics
            self.save_analytics(final=True)
    
    def save_analytics(self, final=False):
        """Save analytics data to JSON file"""
        analytics = {
            'timestamp': datetime.now().isoformat(),
            'current_count': self.people_count_history[-1] if self.people_count_history else 0,
            'average_count': float(np.mean(list(self.people_count_history))) if self.people_count_history else 0,
            'peak_count': self.peak_count,
            'total_entries': self.entries,
            'total_exits': self.exits,
            'net_flow': self.entries - self.exits,
            'capacity': self.max_capacity,
            'occupancy_percentage': (self.people_count_history[-1] / self.max_capacity * 100) if self.people_count_history else 0
        }
        
        filename = f"analytics_{'final_' if final else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(analytics, f, indent=2)
        
        print(f"✓ Analytics saved to {filename}")
        
        if final:
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Peak Count: {analytics['peak_count']}")
            print(f"Average Count: {analytics['average_count']:.1f}")
            print(f"Total Entries: {analytics['total_entries']}")
            print(f"Total Exits: {analytics['total_exits']}")
            print(f"Net Flow: {analytics['net_flow']}")
            print("="*60)


if __name__ == "__main__":
    # Initialize analyzer with YOLOv8
    analyzer = RestaurantCrowdAnalyzer(
        video_source= "input2.mp4",              # 0 for webcam, or path to video file
        max_capacity=50,              # Your restaurant capacity
        model_path='yolov8n.pt'       # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    )
    
    # Run the analysis
    analyzer.run(show_heatmap=True, save_video=True)
