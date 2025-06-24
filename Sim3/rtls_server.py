import paho.mqtt.client as mqtt
import json
import time
import numpy as np
from math import sqrt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import threading
import signal
import sys
from statistics_logger import RTLSStatisticsLogger

# Signal handler για clean shutdown
def signal_handler(sig, frame):
    print("\n🛑 Shutting down gracefully...")
    global running
    running = False

# Global flag για clean shutdown
running = True

# --- Διαμόρφωση Χρήστη ---
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_DATA_TOPIC = "uwb/anchor_data"
MQTT_MOTOR_CMD_TOPIC_PREFIX = "uwb/tags/"

ANCHOR_POSITIONS = {
    "anchor1": np.array([0.0, 0.0]),
    "anchor2": np.array([5.0, 0.0]),
    "anchor3": np.array([0.0, 7.0]),
    "anchor4": np.array([5.0, 7.0])
}

MIN_ANCHORS_FOR_POSITIONING = 3
PROXIMITY_THRESHOLD = 1.0

# --- Στατιστικά ---
stats_logger = RTLSStatisticsLogger()
stats_update_interval = 10

# --- Global Variables ---
tag_distances = {}
tag_positions = {}
motor_states = {}

# --- Matplotlib Global Variables ---
fig, ax = None, None
tag_plot_artists = {}

def setup_plot():
    """Ρυθμίζει το αρχικό γράφημα."""
    global fig, ax
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel("Συντεταγμένη X (m)")
    ax.set_ylabel("Συντεταγμένη Y (m)")
    ax.set_title("Σύστημα Εντοπισμού Θέσης UWB - Real Time")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    anchor_x_coords = [pos[0] for pos in ANCHOR_POSITIONS.values()]
    anchor_y_coords = [pos[1] for pos in ANCHOR_POSITIONS.values()]
    
    if anchor_x_coords and anchor_y_coords:
        ax.set_xlim(min(anchor_x_coords) - 1, max(anchor_x_coords) + 1)
        ax.set_ylim(min(anchor_y_coords) - 1, max(anchor_y_coords) + 1)

    for anchor_id, pos in ANCHOR_POSITIONS.items():
        ax.plot(pos[0], pos[1], 's', markersize=12, label=f"Anchor: {anchor_id}", color='black', markeredgecolor='gray')
        ax.text(pos[0] + 0.1, pos[1] + 0.1, anchor_id, fontsize=9, color='black')

    ax.legend(loc='upper right')
    plt.show()

def update_plot(tags_in_proximity_set):
    """Ενημερώνει το γράφημα με προστασία από interrupts."""
    global ax, tag_plot_artists
    
    try:
        # Έλεγχος αν το figure είναι ακόμα ανοιχτό
        if not plt.fignum_exists(fig.number):
            return
        
        for tag_id in list(tag_plot_artists.keys()):
            artists = tag_plot_artists.pop(tag_id, [])
            for artist in artists:
                try:
                    artist.remove()
                except:
                    pass

        for tag_id, data in tag_positions.items():
            if time.time() - data["timestamp"] < 5.0:
                pos = data["position"]
                is_in_proximity = tag_id in tags_in_proximity_set
                color = 'red' if is_in_proximity else 'blue'
                marker_size = 10 if is_in_proximity else 7
                z_order = 5 if is_in_proximity else 3

                try:
                    point = ax.plot(pos[0], pos[1], 'o', markersize=marker_size, color=color, zorder=z_order)
                    text = ax.text(pos[0] + 0.08, pos[1] + 0.08, tag_id, fontsize=8, color=color, zorder=z_order)
                    tag_plot_artists[tag_id] = point + [text]
                except:
                    pass

        try:
            fig.canvas.draw_idle()
            plt.pause(0.01)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            pass
            
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        pass

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code {rc}")
    client.subscribe(MQTT_DATA_TOPIC)
    print(f"Subscribed to {MQTT_DATA_TOPIC}")

def trilaterate_position(distances_to_anchors, anchor_coords):
    A = []
    b = []
    anchor_ids_used = []

    for anchor_id, dist in distances_to_anchors.items():
        if anchor_id not in anchor_coords:
            continue
        anchor_ids_used.append(anchor_id)

    if len(anchor_ids_used) < MIN_ANCHORS_FOR_POSITIONING:
        return None

    ref_anchor_id = anchor_ids_used[0]
    x_ref, y_ref = anchor_coords[ref_anchor_id]
    d_ref_sq = distances_to_anchors[ref_anchor_id]**2

    for i in range(1, len(anchor_ids_used)):
        current_anchor_id = anchor_ids_used[i]
        x_i, y_i = anchor_coords[current_anchor_id]
        d_i_sq = distances_to_anchors[current_anchor_id]**2

        A.append([2 * (x_i - x_ref), 2 * (y_i - y_ref)])
        b.append(d_ref_sq - d_i_sq - (x_ref**2 - x_i**2) - (y_ref**2 - y_i**2))

    if not A or len(A[0]) == 0:
        return None

    A_matrix = np.array(A)
    b_vector = np.array(b)

    try:
        position, _, _, _ = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
        return position
    except np.linalg.LinAlgError:
        print("Trilateration failed: LinAlgError")
        return None

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        anchor_id = payload.get("anchor_id")
        tag_id = payload.get("tag_id")
        distance = payload.get("distance")

        if not all([anchor_id, tag_id, isinstance(distance, (int, float))]):
            return

        # Καταγραφή λήψης μηνύματος
        stats_logger.log_message_received(tag_id, anchor_id)

        if tag_id not in tag_distances:
            tag_distances[tag_id] = {}

        tag_distances[tag_id][anchor_id] = distance

        if len(tag_distances[tag_id]) >= MIN_ANCHORS_FOR_POSITIONING:
            current_anchor_coords = {aid: ANCHOR_POSITIONS[aid] for aid in tag_distances[tag_id] if aid in ANCHOR_POSITIONS}
            position = trilaterate_position(tag_distances[tag_id], current_anchor_coords)
            
            if position is not None:
                tag_positions[tag_id] = {"position": position, "timestamp": time.time()}
                
                # Καταγραφή επιτυχούς positioning
                stats_logger.log_positioning_attempt(tag_id, True, position)
            else:
                # Καταγραφή αποτυχημένου positioning
                stats_logger.log_positioning_attempt(tag_id, False)

    except Exception as e:
        print(f"Error processing message: {e}")

def check_proximity_and_control_motors(client_mqtt):
    """Ελέγχει την εγγύτητα και στέλνει εντολές στους κινητήρες."""
    active_tags = list(tag_positions.keys())
    tags_currently_in_proximity = set()

    for i in range(len(active_tags)):
        for j in range(i + 1, len(active_tags)):
            tag_id1 = active_tags[i]
            tag_id2 = active_tags[j]

            if time.time() - tag_positions[tag_id1]["timestamp"] > 2.0 or \
               time.time() - tag_positions[tag_id2]["timestamp"] > 2.0:
                continue

            pos1 = tag_positions[tag_id1]["position"]
            pos2 = tag_positions[tag_id2]["position"]
            distance_between_tags = np.linalg.norm(pos1 - pos2)

            if distance_between_tags < PROXIMITY_THRESHOLD:
                print(f"⚠️ ΕΓΓΥΤΗΤΑ: {tag_id1} και {tag_id2} είναι κοντά ({distance_between_tags:.2f}m)!")
                
                # Καταγραφή proximity event
                stats_logger.log_proximity_event(tag_id1, tag_id2, distance_between_tags)
                
                tags_currently_in_proximity.add(tag_id1)
                tags_currently_in_proximity.add(tag_id2)

    all_known_tags = set(motor_states.keys()).union(set(tag_positions.keys()))

    for t_id in all_known_tags:
        topic = f"{MQTT_MOTOR_CMD_TOPIC_PREFIX}{t_id}/motor"
        current_motor_state = motor_states.get(t_id, "OFF")

        if t_id in tags_currently_in_proximity:
            if current_motor_state == "OFF":
                client_mqtt.publish(topic, "ON")
                motor_states[t_id] = "ON"
                print(f"Εντολή: Κινητήρας ON για {t_id}")
        else:
            if current_motor_state == "ON":
                client_mqtt.publish(topic, "OFF")
                motor_states[t_id] = "OFF"
                print(f"Εντολή: Κινητήρας OFF για {t_id}")

    return tags_currently_in_proximity

def periodic_stats_update():
    """Περιοδική ενημέρωση και εκτύπωση στατιστικών"""
    while running:
        time.sleep(stats_update_interval)
        if running:
            stats_logger.save_to_csv()
            
            # Εκτύπωση στατιστικών κάθε 30 δευτερόλεπτα
            if int(time.time()) % 30 == 0:
                stats_logger.print_summary()

# --- Κύριο Πρόγραμμα ---
if __name__ == "__main__":
    # Ρύθμιση signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Δημιουργία MQTT client με compatibility
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            client = mqtt.Client()

        client.on_connect = on_connect
        client.on_message = on_message

        try:
            client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        except Exception as e:
            print(f"Δεν ήταν δυνατή η σύνδεση στον MQTT Broker: {e}")
            sys.exit(1)

        client.loop_start()

        # Ξεκίνησε το thread για στατιστικά
        stats_thread = threading.Thread(target=periodic_stats_update, daemon=True)
        stats_thread.start()

        setup_plot()

        # Main loop με καλύτερο error handling
        while running:
            try:
                tags_in_alarm = check_proximity_and_control_motors(client)
                update_plot(tags_in_alarm)
                
                # Μικρή παύση για να επιτρέψουμε interrupt handling
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received...")
    
    finally:
        print("🧹 Cleaning up...")
        running = False
        
        # Αποθήκευση στατιστικών
        try:
            stats_logger.save_detailed_log()
            stats_logger.print_summary()
        except:
            pass
        
        # Κλείσιμο matplotlib
        try:
            plt.close('all')
            plt.ioff()
        except:
            pass
        
        # Κλείσιμο MQTT
        try:
            client.loop_stop()
            client.disconnect()
        except:
            pass
        
        print("✅ Program terminated cleanly.")
