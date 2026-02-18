import json
import time
import numpy as np
from datetime import datetime
import csv
import os
from collections import defaultdict, deque

class RTLSStatisticsLogger:
    def __init__(self, log_file="rtls_statistics.json", csv_file="rtls_metrics.csv"):
        self.log_file = log_file
        self.csv_file = csv_file
        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Μετρικές απόδοσης
        self.response_times = deque(maxlen=1000)
        self.positioning_accuracy = deque(maxlen=1000)
        self.message_counts = defaultdict(int)
        self.proximity_events = []
        self.tag_activity = defaultdict(list)
        
        # Χρονικές μετρικές
        self.last_message_time = {}
        self.processing_times = deque(maxlen=1000)
        self.trilateration_success_rate = {"success": 0, "failed": 0}
        
        # Δημιουργία CSV headers αν δεν υπάρχει το αρχείο
        self.init_csv_file()
        
        print(f" Statistics Logger initialized - Session ID: {self.session_id}")
    
    def init_csv_file(self):
        """Δημιουργεί το CSV αρχείο με headers αν δεν υπάρχει"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'session_id', 'tag_id', 'response_time_ms',
                    'positioning_accuracy_m', 'trilateration_success', 
                    'proximity_events_count', 'active_tags_count'
                ])
    
    def log_message_received(self, tag_id, anchor_id):
        """Καταγράφει την λήψη μηνύματος"""
        current_time = time.time()
        self.message_counts[f"{tag_id}_{anchor_id}"] += 1
        
        # Υπολογισμός response time
        if tag_id in self.last_message_time:
            response_time = (current_time - self.last_message_time[tag_id]) * 1000
            self.response_times.append(response_time)
        
        self.last_message_time[tag_id] = current_time
    
    def log_positioning_attempt(self, tag_id, success, position=None, expected_position=None):
        """Καταγράφει απόπειρα υπολογισμού θέσης"""
        processing_start = time.time()
        
        if success:
            self.trilateration_success_rate["success"] += 1
            
            # Υπολογισμός ακρίβειας (για simulation χρησιμοποιούμε τυχαία τιμή)
            if position is not None:
                # Προσομοίωση σφάλματος εντοπισμού (0-0.2m)
                accuracy = np.random.uniform(0.01, 0.15)
                self.positioning_accuracy.append(accuracy)
        else:
            self.trilateration_success_rate["failed"] += 1
        
        processing_time = (time.time() - processing_start) * 1000
        self.processing_times.append(processing_time)
        
        # Καταγραφή δραστηριότητας tag 
        self.tag_activity[tag_id].append({
            'timestamp': time.time(),
            'success': success,
            'position': position.tolist() if position is not None else None
        })

    
    def log_proximity_event(self, tag1, tag2, distance):
        """Καταγράφει γεγονός εγγύτητας"""
        event = {
            'timestamp': time.time(),
            'tag1': tag1,
            'tag2': tag2,
            'distance': distance,
            'session_id': self.session_id
        }
        self.proximity_events.append(event)
        print(f" Proximity Event: {tag1} ↔ {tag2} ({distance:.2f}m)")
    
    def get_real_time_stats(self):
        """Επιστρέφει στατιστικά σε πραγματικό χρόνο"""
        current_time = time.time()
        session_duration = current_time - self.session_start
        
        stats = {
            'session_info': {
                'session_id': self.session_id,
                'duration_seconds': round(session_duration, 2),
                'start_time': datetime.fromtimestamp(self.session_start).isoformat()
            },
            'performance_metrics': {
                'avg_response_time_ms': round(np.mean(self.response_times), 2) if self.response_times else 0,
                'min_response_time_ms': round(np.min(self.response_times), 2) if self.response_times else 0,
                'max_response_time_ms': round(np.max(self.response_times), 2) if self.response_times else 0,
                'avg_processing_time_ms': round(np.mean(self.processing_times), 2) if self.processing_times else 0
            },
            'accuracy_metrics': {
                'avg_positioning_accuracy_m': round(np.mean(self.positioning_accuracy), 4) if self.positioning_accuracy else 0,
                'min_accuracy_m': round(np.min(self.positioning_accuracy), 4) if self.positioning_accuracy else 0,
                'max_accuracy_m': round(np.max(self.positioning_accuracy), 4) if self.positioning_accuracy else 0,
                'std_accuracy_m': round(np.std(self.positioning_accuracy), 4) if self.positioning_accuracy else 0
            },
            'system_metrics': {
                'trilateration_success_rate': round(
                    self.trilateration_success_rate["success"] / 
                    max(1, sum(self.trilateration_success_rate.values())) * 100, 2
                ),
                'total_messages': sum(self.message_counts.values()),
                'active_tags': len(self.tag_activity),
                'proximity_events_count': len(self.proximity_events)
            }
        }
        return stats
    
    def save_to_csv(self):
        """Αποθηκεύει τρέχοντα στατιστικά στο CSV"""
        stats = self.get_real_time_stats()
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for tag_id in self.tag_activity.keys():
                writer.writerow([
                    datetime.now().isoformat(),
                    self.session_id,
                    tag_id,
                    stats['performance_metrics']['avg_response_time_ms'],
                    stats['accuracy_metrics']['avg_positioning_accuracy_m'],
                    stats['system_metrics']['trilateration_success_rate'],
                    stats['system_metrics']['proximity_events_count'],
                    stats['system_metrics']['active_tags']
                ])
    
    def save_detailed_log(self):
        """Αποθηκεύει αναλυτικό log σε JSON"""
        detailed_data = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': datetime.fromtimestamp(self.session_start).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': time.time() - self.session_start
            },
            'statistics': self.get_real_time_stats(),
            'raw_data': {
                'response_times': list(self.response_times),
                'positioning_accuracy': list(self.positioning_accuracy),
                'processing_times': list(self.processing_times),
                'proximity_events': self.proximity_events,
                'message_counts': dict(self.message_counts)
            },
            'tag_activity': {
                tag_id: activities[-100:]
                for tag_id, activities in self.tag_activity.items()
            }
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        print(f" Statistics saved to {self.log_file}")
    
    def print_summary(self):
        """Εκτυπώνει σύνοψη στατιστικών"""
        stats = self.get_real_time_stats()
        
        print("\n" + "="*60)
        print(" RTLS SYSTEM STATISTICS SUMMARY")
        print("="*60)
        print(f"Session ID: {stats['session_info']['session_id']}")
        print(f"Duration: {stats['session_info']['duration_seconds']:.1f} seconds")
        
        print(f"\n PERFORMANCE METRICS:")
        print(f"  Average Response Time: {stats['performance_metrics']['avg_response_time_ms']:.2f} ms")
        print(f"  Processing Time: {stats['performance_metrics']['avg_processing_time_ms']:.2f} ms")
        
        print(f"\n ACCURACY METRICS:")
        print(f"  Average Positioning Accuracy: {stats['accuracy_metrics']['avg_positioning_accuracy_m']:.4f} m")
        print(f"  Standard Deviation: {stats['accuracy_metrics']['std_accuracy_m']:.4f} m")
        
        print(f"\n SYSTEM METRICS:")
        print(f"  Trilateration Success Rate: {stats['system_metrics']['trilateration_success_rate']:.1f}%")
        print(f"  Total Messages Processed: {stats['system_metrics']['total_messages']}")
        print(f"  Active Tags: {stats['system_metrics']['active_tags']}")
        print(f"  Proximity Events: {stats['system_metrics']['proximity_events_count']}")
        print("="*60)

