#!/usr/bin/env python3
"""
MAVLink Line Follower Mission:
 - Arm and Take off
 - Receive real-time camera feed
 - Use HoughLinesP to detect path orientation and offset
 - Dynamically adjust velocity (vx, vy) and yaw to follow the line
 - Land when the line is lost or mission completes
"""
import time
import argparse
import threading
import socket
import struct
import numpy as np
import cv2
import math
from datetime import datetime
from pymavlink import mavutil

# --- MAVLink Control Functions ---

def wait_heartbeat(m):
    print('Waiting for heartbeat...')
    m.wait_heartbeat()
    print('Heartbeat received: system', m.target_system, 'component', m.target_component)

def set_mode_guided(m):
    try:
        m.set_mode_apm('GUIDED')
        print('Requested GUIDED mode')
    except Exception:
        m.mav.set_mode_send(m.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
        print('Sent set_mode GUIDED fallback')
    time.sleep(1)

def is_armed(m):
    hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
    if not hb: return False
    return bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

def arm(m, force=False):
    print(f'Arming (force={force})...')
    if not force:
        m.arducopter_arm()
        end = time.time() + 5
        while time.time() < end:
            if is_armed(m): return True
            time.sleep(0.5)
    
    # Force arm if necessary
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 21196, 0, 0, 0, 0, 0)
    time.sleep(2)
    return is_armed(m)

def takeoff(m, alt):
    print(f'Takeoff to {alt} m...')
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, float(alt))
    time.sleep(5) # Simplified wait for logic flow

def send_velocity(m, vx, vy, yaw_rate=0.0):
    """Sends velocity commands in BODY_NED frame."""
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000101111000111, # Velocity and yaw_rate enabled
        0, 0, 0, vx, vy, 0, 0, 0, 0, 0, yaw_rate
    )

def land(m):
    print('Landing...')
    m.set_mode_apm('LAND')
    time.sleep(5)

# --- Line Following Algorithm ---

def process_frame(frame, speed_base):
    """
    Analyzes frame to find the line angle and center offset.
    Returns (vx, vy, yaw_rate, processed_frame)
    """
    h, w = frame.shape
    # 1. Edge Detection
    edges = cv2.Canny(frame, 50, 150)
    
    # 2. Hough Line Detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=15)
    
    vx, vy, yaw_rate = 0.0, 0.0, 0.0
    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        # Calculate average line center and angle
        centers_x = []
        angles = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            centers_x.append((x1 + x2) / 2)
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        avg_center_x = np.mean(centers_x)
        avg_angle = np.mean(angles)

        # Correction Logic:
        # 1. Forward Speed (constant or based on confidence)
        vx = speed_base
        
        # 2. Horizontal Correction (Lateral Error)
        # If line is to the right of center, vy should be positive (move right)
        error_x = (avg_center_x - (w / 2)) / (w / 2) # Normalized -1 to 1
        vy = error_x * 0.5 # Lateral gain

        # 3. Heading Correction (Angular Error)
        # Target angle is usually -90 for a vertical line in image space
        # We want to rotate to align with the line's orientation
        yaw_rate = (avg_angle + 90) * 0.05 # Angular gain
        
        cv2.putText(display_frame, f"Angle: {round(avg_angle, 1)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        # If no line is found, stop or hover
        vx, vy, yaw_rate = 0.0, 0.0, 0.0
        cv2.putText(display_frame, "LOST LINE", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return vx, vy, yaw_rate, display_frame

# --- Main Threads ---

def _camera_and_control_loop(host, port, stop_event, m, speed_base):
    try:
        sock = socket.create_connection((host, port), timeout=5)
        print(f'Connected to Camera at {host}:{port}')
    except Exception as e:
        print(f'Camera Error: {e}')
        return

    try:
        while not stop_event.is_set():
            # Read Frame
            hdr = _recvn(sock, 4)
            if not hdr: break
            w, h = struct.unpack('<HH', hdr)
            buf = _recvn(sock, int(w) * int(h))
            if not buf: break
            
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w))
            
            # Process Frame for Navigation
            vx, vy, yaw_rate, result_img = process_frame(frame, speed_base)
            
            # Send commands to MAVLink
            send_velocity(m, vx, vy, yaw_rate)
            
            # Display
            cv2.imshow('Line Follower Debug', result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    finally:
        sock.close()
        cv2.destroyAllWindows()

def _recvn(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk: return None
        data.extend(chunk)
    return bytes(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alt', type=float, default=1.5)
    ap.add_argument('--speed', type=float, default=0.4, help='Base forward speed')
    ap.add_argument('--mav', type=str, default='udp:0.0.0.0:14550')
    ap.add_argument('--host', type=str, default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5599)
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.mav)
    wait_heartbeat(m)
    set_mode_guided(m)

    if not arm(m):
        print("Arming failed")
        return

    takeoff(m, args.alt)
    
    stop_event = threading.Event()
    cam_thread = threading.Thread(
        target=_camera_and_control_loop, 
        args=(args.host, args.port, stop_event, m, args.speed), 
        daemon=True
    )
    cam_thread.start()

    print("Line following active. Press 'q' in window or Ctrl+C to land.")
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    land(m)

if __name__ == '__main__':
    main()