import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os

class RTLSStatisticsViewer:
    def __init__(self, json_file="rtls_statistics.json", csv_file="rtls_metrics.csv"):
        self.json_file = json_file
        self.csv_file = csv_file
        self.json_data = None
        self.load_data()
    
    def load_data(self):
        """Φορτώνει τα δεδομένα από JSON"""
        print(f" Searching for files in: {os.getcwd()}")
        print(f" Files in directory: {[f for f in os.listdir('.') if f.endswith(('.json', '.csv'))]}")
        
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    self.json_data = json.load(f)
                print(f" Loaded JSON data from {self.json_file}")
            except Exception as e:
                print(f" Error loading JSON: {e}")
        else:
            print(f" JSON file {self.json_file} not found")
            print(" Make sure to run rtls_server.py first and terminate with Ctrl+C")
    
    def plot_response_times(self):
        """Γράφημα χρόνων απόκρισης"""
        if not self.json_data:
            print(" No data available")
            return
        
        try:
            response_times = self.json_data['raw_data']['response_times']
            if not response_times:
                print(" No response time data")
                return
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.hist(response_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Response Time (ms)')
            plt.ylabel('Frequency')
            plt.title('Response Time Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(response_times, alpha=0.7, color='blue')
            plt.xlabel('Message Number')
            plt.ylabel('Response Time (ms)')
            plt.title('Response Time Over Time')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f" Error plotting: {e}")
    
    def plot_positioning_accuracy(self):
        """Γράφημα ακρίβειας εντοπισμού"""
        if not self.json_data:
            return
        
        try:
            accuracy_data = self.json_data['raw_data']['positioning_accuracy']
            if not accuracy_data:
                print("⚠️ No accuracy data")
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(accuracy_data, alpha=0.7, color='green', marker='o', markersize=3)
            plt.xlabel('Positioning Attempt')
            plt.ylabel('Error (m)')
            plt.title('Positioning Accuracy Over Time')
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except Exception as e:
            print(f" Error plotting accuracy: {e}")
    
    def plot_proximity_events(self):
        """Γράφημα γεγονότων εγγύτητας"""
        if not self.json_data:
            return
        
        try:
            events = self.json_data['raw_data']['proximity_events']
            if not events:
                print(" No proximity events")
                return
            
            distances = [event['distance'] for event in events]
            
            plt.figure(figsize=(10, 6))
            plt.hist(distances, bins=15, alpha=0.7, color='orange', edgecolor='black')
            plt.xlabel('Distance (m)')
            plt.ylabel('Frequency')
            plt.title('Proximity Distance Distribution')
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except Exception as e:
            print(f" Error plotting proximity: {e}")
    
    def generate_report(self):
        """Δημιουργεί αναλυτική αναφορά"""
        if not self.json_data:
            print(" No data available for report")
            return

        try:
            stats = self.json_data['statistics']
            
            print("\n" + "="*60)
            print(" RTLS PERFORMANCE REPORT")
            print("="*60)
            
            session = stats['session_info']
            print(f"Session ID: {session['session_id']}")
            print(f"Duration: {session['duration_seconds']:.1f} seconds")
            
            perf = stats['performance_metrics']
            print(f"\n⚡ PERFORMANCE:")
            print(f"  Avg Response Time: {perf['avg_response_time_ms']:.2f} ms")
            print(f"  Min Response Time: {perf['min_response_time_ms']:.2f} ms")
            print(f"  Max Response Time: {perf['max_response_time_ms']:.2f} ms")
            
            acc = stats['accuracy_metrics']
            print(f"\n ACCURACY:")
            print(f"  Avg Error: {acc['avg_positioning_accuracy_m']:.4f} m")
            print(f"  Std Deviation: {acc['std_accuracy_m']:.4f} m")
            
            sys = stats['system_metrics']
            print(f"\n SYSTEM:")
            print(f"  Success Rate: {sys['trilateration_success_rate']:.1f}%")
            print(f"  Total Messages: {sys['total_messages']}")
            print(f"  Active Tags: {sys['active_tags']}")
            print(f"  Proximity Events: {sys['proximity_events_count']}")
            print("="*60)
            
        except Exception as e:
            print(f" Error generating report: {e}")

if __name__ == "__main__":
    print(" RTLS Statistics Viewer")
    print("=" * 40)
    
    viewer = RTLSStatisticsViewer()
    
    if viewer.json_data:
        print("\n Generating visualizations...")
        viewer.plot_response_times()
        viewer.plot_positioning_accuracy()
        viewer.plot_proximity_events()
        viewer.generate_report()
    else:
        print("\n To generate statistics:")
        print("1. Run: python rtls_server.py")
        print("2. Run: python tag_simulator.py")
        print("3. Wait 2-3 minutes")
        print("4. Stop server with Ctrl+C")
        print("5. Run: python view_statistics.py")

