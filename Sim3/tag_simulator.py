import paho.mqtt.client as mqtt
import json
import time
import numpy as np
import random

# --- Διαμόρφωση Προσομοιωτή ---
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_DATA_TOPIC = "uwb/anchor_data"

ANCHOR_POSITIONS = {
    "anchor1": np.array([0.0, 0.0]),
    "anchor2": np.array([5.0, 0.0]),
    "anchor3": np.array([0.0, 7.0]),
    "anchor4": np.array([5.0, 7.0])
}

NUM_SIMULATED_TAGS = 3
SIMULATED_TAG_IDS = [f"sim_tag{i+1}" for i in range(NUM_SIMULATED_TAGS)]

MIN_X = min(p[0] for p in ANCHOR_POSITIONS.values()) - 1
MAX_X = max(p[0] for p in ANCHOR_POSITIONS.values()) + 1
MIN_Y = min(p[1] for p in ANCHOR_POSITIONS.values()) - 1
MAX_Y = max(p[1] for p in ANCHOR_POSITIONS.values()) + 1

UPDATE_INTERVAL_SECONDS = 0.5
MAX_STEP_SIZE = 0.3
NOISE_LEVEL = 0.05

# --- Αρχικές θέσεις και στόχοι ---
simulated_tag_current_positions = {
    tag_id: np.array([random.uniform(MIN_X, MAX_X), random.uniform(MIN_Y, MAX_Y)])
    for tag_id in SIMULATED_TAG_IDS
}

simulated_tag_targets = {
    tag_id: np.array([random.uniform(MIN_X, MAX_X), random.uniform(MIN_Y, MAX_Y)])
    for tag_id in SIMULATED_TAG_IDS
}

def update_tag_positions_and_targets():
    """Ενημερώνει τις θέσεις των προσομοιωμένων tags προς τους στόχους τους."""
    global simulated_tag_current_positions, simulated_tag_targets

    for tag_id in SIMULATED_TAG_IDS:
        current_pos = simulated_tag_current_positions[tag_id]
        target_pos = simulated_tag_targets[tag_id]
        direction = target_pos - current_pos
        distance_to_target = np.linalg.norm(direction)

        if distance_to_target < MAX_STEP_SIZE * 2:
            simulated_tag_targets[tag_id] = np.array([random.uniform(MIN_X, MAX_X), random.uniform(MIN_Y, MAX_Y)])
        else:
            move_vector = (direction / distance_to_target) * MAX_STEP_SIZE
            new_pos = current_pos + move_vector
            new_pos[0] = np.clip(new_pos[0], MIN_X, MAX_X)
            new_pos[1] = np.clip(new_pos[1], MIN_Y, MAX_Y)
            simulated_tag_current_positions[tag_id] = new_pos

def calculate_distance(pos1, pos2):
    """Υπολογίζει την Ευκλείδεια απόσταση μεταξύ δύο σημείων."""
    return np.linalg.norm(pos1 - pos2)

def on_connect_simulator(client, userdata, flags, rc):
    if rc == 0:
        print(f"Tag Simulator: Connected to MQTT Broker: {MQTT_BROKER_HOST}")
    else:
        print(f"Tag Simulator: Failed to connect, return code {rc}\n")

# --- Κύριο Πρόγραμμα Προσομοιωτή ---
try:
    # Δημιουργία MQTT client με compatibility
    try:
        sim_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        sim_client = mqtt.Client()

    sim_client.on_connect = on_connect_simulator

    try:
        sim_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    except Exception as e:
        print(f"Tag Simulator: Could not connect to MQTT Broker: {e}")
        exit(1)

    sim_client.loop_start()

    print("Tag Simulator: Starting simulation...")
    print(f"Simulating {NUM_SIMULATED_TAGS} tags: {', '.join(SIMULATED_TAG_IDS)}")
    print(f"Publishing data every {UPDATE_INTERVAL_SECONDS} seconds.")
    print(f"Simulation area X: [{MIN_X:.1f}, {MAX_X:.1f}], Y: [{MIN_Y:.1f}, {MAX_Y:.1f}]")

    try:
        while True:
            update_tag_positions_and_targets()

            for tag_id in SIMULATED_TAG_IDS:
                tag_pos = simulated_tag_current_positions[tag_id]

                for anchor_id, anchor_pos in ANCHOR_POSITIONS.items():
                    dist_no_noise = calculate_distance(tag_pos, anchor_pos)
                    simulated_distance = dist_no_noise + random.uniform(-NOISE_LEVEL, NOISE_LEVEL)
                    simulated_distance = max(0, simulated_distance)

                    payload = {
                        "anchor_id": anchor_id,
                        "tag_id": tag_id,
                        "distance": round(simulated_distance, 2)
                    }

                    sim_client.publish(MQTT_DATA_TOPIC, json.dumps(payload))

            time.sleep(UPDATE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nTag Simulator: Stopping simulation...")

finally:
    sim_client.loop_stop()
    sim_client.disconnect()
    print("Tag Simulator: Disconnected and stopped.")
