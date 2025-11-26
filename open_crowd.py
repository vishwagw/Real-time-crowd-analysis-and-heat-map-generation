"""
Open Crowd Analysis AI System using YOLOv8
General-purpose crowd monitoring and analytics for any location.

Features:
- Detects and tracks people using YOLOv8
- Simple centroid-based tracker
- Zone counting, heatmap, occupancy/flow metrics
- CLI-friendly: set video source, model, capacity, and save options
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import json
import argparse
from ultralytics import YOLO


class OpenCrowdAnalyzer:
    def __init__(self, video_source=0, max_capacity=None, model_path='yolov8n.pt', location_name=None):
        """
        Initialize the crowd analyzer with YOLOv8

        Args:
            video_source: Camera index or video file path
            max_capacity: Optional maximum capacity for occupancy calculations
            model_path: YOLOv8 model path
            location_name: Optional friendly name for the monitored place
        """
        self.video_source = video_source
        self.max_capacity = max_capacity or 0
        self.location_name = location_name or "Open Location"
        self.cap = None

        # Initialize YOLOv8
        print(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        print("✓ YOLOv8 model loaded successfully")

        # Analytics data
        self.people_count_history = deque(maxlen=300)
        self.entry_line = None
        self.exit_line = None
        self.entries = 0
        self.exits = 0
        self.tracked_objects = {}
        self.next_object_id = 0

        # Heatmap for crowd density
        self.heatmap = None
        self.heatmap_accumulator = None

        # Generic zones (can be overridden by setup_zones)
        self.zones = {
            'zone_1': None,
            'zone_2': None,
            'zone_3': None,
            'zone_4': None
        }

        # Performance metrics
        self.peak_count = 0

    def setup_zones(self, frame_width, frame_height):
        """Auto-setup simple zones based on frame dimensions"""
        self.zones = {
            'left_area': (0, 0, frame_width // 3, frame_height),
            'center_area': (frame_width // 3, 0, 2 * frame_width // 3, frame_height),
            'right_area': (2 * frame_width // 3, 0, frame_width, frame_height),
            'full_frame': (0, 0, frame_width, frame_height)
        }

        # Entry/exit lines for simple flow tracking (vertical lines)
        self.entry_line = frame_width // 3
        self.exit_line = 2 * frame_width // 3

    def detect_people(self, frame, conf_threshold=0.45):
        """
        Detect people using YOLOv8
        Returns list of detections with boxes, confidence, and center
        """
        results = self.model(frame, conf=conf_threshold, classes=[0], verbose=False)
        detections = []

        for result in results:
            boxes = getattr(result, 'boxes', [])
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                center_x = x + w // 2
                center_y = y + h // 2

                detections.append({
                    'box': (x, y, w, h),
                    'confidence': confidence,
                    'center': (center_x, center_y),
                    'id': None
                })

        return detections

    def simple_tracker(self, detections, max_distance=60):
        """Simple centroid-based tracker"""
        if not self.tracked_objects:
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

        matched = set()
        for det in detections:
            min_dist = float('inf')
            matched_id = None

            for obj_id, obj_data in self.tracked_objects.items():
                if obj_id in matched:
                    continue
                dist = np.hypot(det['center'][0] - obj_data['center'][0], det['center'][1] - obj_data['center'][1])
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    matched_id = obj_id

            if matched_id is not None:
                det['id'] = matched_id
                matched.add(matched_id)
                self.tracked_objects[matched_id]['center'] = det['center']
                self.tracked_objects[matched_id]['last_seen'] = 0

                # Simple entry/exit by x-position crossing
                if (self.entry_line and not self.tracked_objects[matched_id]['crossed_entry'] and det['center'][0] > self.entry_line):
                    self.tracked_objects[matched_id]['crossed_entry'] = True
                    self.entries += 1

                if (self.exit_line and not self.tracked_objects[matched_id]['crossed_exit'] and det['center'][0] > self.exit_line):
                    self.tracked_objects[matched_id]['crossed_exit'] = True
                    self.exits += 1
            else:
                det['id'] = self.next_object_id
                self.tracked_objects[self.next_object_id] = {
                    'center': det['center'],
                    'last_seen': 0,
                    'crossed_entry': False,
                    'crossed_exit': False
                }
                self.next_object_id += 1

        # Age tracked objects
        to_remove = []
        for obj_id in list(self.tracked_objects.keys()):
            self.tracked_objects[obj_id]['last_seen'] += 1
            if self.tracked_objects[obj_id]['last_seen'] > 60:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.tracked_objects[obj_id]

        return detections

    def update_heatmap(self, frame_shape, detections):
        if self.heatmap_accumulator is None:
            self.heatmap_accumulator = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)

        self.heatmap_accumulator *= 0.98

        for det in detections:
            cx, cy = det['center']
            if 0 <= cx < frame_shape[1] and 0 <= cy < frame_shape[0]:
                cv2.circle(self.heatmap_accumulator, (cx, cy), 40, (1,), -1)

        self.heatmap = cv2.GaussianBlur(self.heatmap_accumulator, (51, 51), 0)

    def count_by_zone(self, detections):
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
        if not self.max_capacity:
            return "N/A", (200, 200, 200), 0

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
        people_count = len(detections)
        self.people_count_history.append(people_count)

        if people_count > self.peak_count:
            self.peak_count = people_count

        for det in detections:
            x, y, w, h = det['box']
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if det['id'] is not None:
                label = f"ID:{det['id']} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, det['center'], 4, (0, 0, 255), -1)

        # Draw zones
        for zone_name, zone_coords in self.zones.items():
            if zone_coords is None:
                continue
            x1, y1, x2, y2 = zone_coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(frame, zone_name.replace('_', ' ').title(), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # Entry/exit lines
        if self.entry_line:
            cv2.line(frame, (self.entry_line, 0), (self.entry_line, frame.shape[0]), (0, 255, 255), 2)
            cv2.putText(frame, "ENTRY", (self.entry_line + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self.exit_line:
            cv2.line(frame, (self.exit_line, 0), (self.exit_line, frame.shape[0]), (255, 0, 255), 2)
            cv2.putText(frame, "EXIT", (self.exit_line + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Analytics panel
        panel_height = 200
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        occupancy_level, color, percentage = self.get_occupancy_level(people_count)
        cv2.putText(panel, f"Location: {self.location_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.putText(panel, f"CURRENT COUNT: {people_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(panel, f"Capacity: {self.max_capacity if self.max_capacity else 'Unknown'} | Occupancy: {percentage:.1f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Flow metrics
        cv2.putText(panel, f"Entries: {self.entries} | Exits: {self.exits} | Net: {self.entries - self.exits}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(panel, f"Peak Count: {self.peak_count}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)

        # Zone breakdown
        zone_counts = self.count_by_zone(detections)
        zone_text = " | ".join([f"{name}: {count}" for name, count in zone_counts.items()])
        cv2.putText(panel, f"Zones: {zone_text}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, timestamp, (frame.shape[1] - 260, panel_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        combined = np.vstack([frame, panel])
        return combined

    def run(self, show_heatmap=True, save_video=False, output_prefix=None):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print("✗ Error: Could not open video source")
            return

        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        self.setup_zones(frame_width, frame_height)

        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_filename = (output_prefix or 'crowd_analysis') + f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height + 200))
            print(f"✓ Recording to {output_filename}")

        print("\n" + "=" * 60)
        print(f"Open Crowd Analysis - {self.location_name}")
        print("=" * 60)
        print("Controls: 'q' - Quit | 's' - Save analytics | 'r' - Reset counters | 'h' - Toggle heatmap")
        print("=" * 60 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.video_source, str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                detections = self.detect_people(frame)
                detections = self.simple_tracker(detections)

                if show_heatmap:
                    self.update_heatmap(frame.shape, detections)

                output_frame = self.draw_analytics(frame, detections)

                if show_heatmap and self.heatmap is not None:
                    heatmap_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap_uint8 = heatmap_norm.astype(np.uint8)
                    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    heatmap_resized = cv2.resize(heatmap_colored, (240, 140))
                    h, w = heatmap_resized.shape[:2]
                    output_frame[10:10+h, frame.shape[1]-w-10:frame.shape[1]-10] = heatmap_resized
                    cv2.putText(output_frame, "Density Heatmap", (frame.shape[1]-w-8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if video_writer:
                    video_writer.write(output_frame)

                window_title = f"Open Crowd Analysis - {self.location_name}"
                cv2.imshow(window_title, output_frame)

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
            self.cap.release()
            if video_writer:
                video_writer.release()
                print("✓ Video saved")
            cv2.destroyAllWindows()
            self.save_analytics(final=True)

    def save_analytics(self, final=False):
        analytics = {
            'timestamp': datetime.now().isoformat(),
            'current_count': self.people_count_history[-1] if self.people_count_history else 0,
            'average_count': float(np.mean(list(self.people_count_history))) if self.people_count_history else 0,
            'peak_count': self.peak_count,
            'total_entries': self.entries,
            'total_exits': self.exits,
            'net_flow': self.entries - self.exits,
            'capacity': self.max_capacity,
            'occupancy_percentage': (self.people_count_history[-1] / self.max_capacity * 100) if (self.people_count_history and self.max_capacity) else 0
        }

        filename = f"open_analytics_{'final_' if final else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(analytics, f, indent=2)

        print(f"✓ Analytics saved to {filename}")

        if final:
            print("\n" + "=" * 60)
            print("SESSION SUMMARY")
            print("=" * 60)
            print(f"Peak Count: {analytics['peak_count']}")
            print(f"Average Count: {analytics['average_count']:.1f}")
            print(f"Total Entries: {analytics['total_entries']}")
            print(f"Total Exits: {analytics['total_exits']}")
            print(f"Net Flow: {analytics['net_flow']}")
            print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description='Open Crowd Analysis')
    parser.add_argument('--source', '-s', default=0, help='Video source (0 for webcam or path to file)')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='Path to YOLOv8 model')
    parser.add_argument('--capacity', '-c', type=int, default=0, help='Optional max capacity for occupancy')
    parser.add_argument('--name', '-n', default='Open Location', help='Friendly name for the location')
    parser.add_argument('--no-heatmap', dest='heatmap', action='store_false', help='Disable heatmap overlay')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--output-prefix', default='open_crowd', help='Prefix for saved video filename')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Convert numeric source if possible
    try:
        source = int(args.source)
    except Exception:
        source = args.source

    analyzer = OpenCrowdAnalyzer(video_source=source, max_capacity=args.capacity, model_path=args.model, location_name=args.name)
    analyzer.run(show_heatmap=args.heatmap, save_video=args.save, output_prefix=args.output_prefix)
