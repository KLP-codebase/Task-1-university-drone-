#!/usr/bin/env python3
"""
MAVLink Line Follower & AprilTag Mission:
 - Connect to MAVLink and wait for Heartbeat
 - Arm (with fallback force-arm) and Take off
 - Receive real-time camera feed via socket
 - Simultaneously follow lines and scan AprilTags using 'apriltag' library
 - Center over tags, verify via Hamming distance, and print ID
 - SPECIAL CONDITION: Rotate 90 degrees right after the FIRST tag is scanned.
 - Land and Disarm when mission ends
"""
import time
import argparse
import threading
import socket
import struct
import numpy as np
import cv2
import math
import apriltag
from datetime import datetime
from pymavlink import mavutil

# --- AprilTag Configuration ---
TAG_OPTIONS = apriltag.DetectorOptions(families="tag36h11")
TAG_DETECTOR = apriltag.Detector(TAG_OPTIONS)

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
    if not hb:
        return False
    return bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

def arm(m, force=False):
    print(f'Arming (force={force})...')
    if not force:
        try:
            m.arducopter_arm()
            while True:
                ack = m.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)
                if ack and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    print("ARM ACK Received")
                    break
                if not ack: break
            time.sleep(0.5)
        except Exception as e:
            print(f"Normal arm request error: {e}")
        
        end = time.time() + 5
        while time.time() < end:
            if is_armed(m):
                print('Armed Successfully')
                return True
            time.sleep(0.5)
    
    print('Sending force-arm command...')
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0
    )
    time.sleep(2)
    if is_armed(m):
        print('Armed (Forced)')
        return True
    return False

def takeoff(m, alt):
    print(f'Takeoff to {alt}m...')
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, float(alt)
    )
    
    end = time.time() + 20
    while time.time() < end:
        msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if not msg: continue
        alt_m = msg.relative_alt / 1000.0
        print(f' Current Altitude: {alt_m:.2f} m')
        if alt_m >= alt - 0.3:
            print('Reached takeoff altitude target')
            return True
    return False

def rotate_90_clockwise(m, duration_s=4.0):
    print(f'Action: Rotating 90 degrees clockwise over {duration_s}s')
    angular_speed = 90.0 / duration_s 
    
    m.mav.command_long_send(
        m.target_system,
        m.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        90.0,           # target angle (relative)
        angular_speed,  # speed (deg/sec)
        1.0,            # direction (1 = CW)
        1.0,            # relative (1)
        0, 0, 0
    )
    time.sleep(duration_s)

def send_velocity(m, vx, vy, yaw_rate=0.0):
    """Sends velocity commands in BODY_NED frame (vx=forward, vy=right)."""
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000101111000111, 
        0, 0, 0, vx, vy, 0, 0, 0, 0, 0, yaw_rate
    )

def land(m):
    print('Initiating Landing Sequence...')
    try:
        m.set_mode_apm('LAND')
    except Exception:
        m.mav.set_mode_send(m.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)
    
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    
    end = time.time() + 15
    while time.time() < end:
        if not is_armed(m):
            print('Drone Disarmed; Mission Complete')
            return True
        time.sleep(0.5)
    return False

# --- Vision Processing ---

def process_frame(frame, speed_base, last_scanned_id):
    """
    Simultaneously handles AprilTag detection (priority) and Line Following.
    """
    gray = frame.copy()
    h, w = gray.shape
    vx, vy, yaw_rate = 0.0, 0.0, 0.0
    display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    center_x, center_y = w // 2, h // 2
    
    # 1. AprilTag Detection
    detections = TAG_DETECTOR.detect(gray)
    tag_active = False
    tag_info = None 

    if len(detections) > 0:
        detection = detections[0]
        tag_id = detection.tag_id
        hamming = detection.hamming
        tag_cx, tag_cy = detection.center
        
        pts = np.array(detection.corners).astype(int)
        cv2.polylines(display_frame, [pts], True, (0, 0, 255), 3)
        cv2.circle(display_frame, (int(tag_cx), int(tag_cy)), 5, (0, 0, 255), -1)
        
        if tag_id != last_scanned_id:
            tag_active = True
            tag_info = (tag_id, hamming)
            
            # --- REDUCED ADJUSTMENT PARAMETERS ---
            tol = 50           # Increased tolerance to reduce micro-movements (deadzone)
            gain_p = 0.002     # Reduced P-gain for smoother approach (was 0.005)
            max_adj_vel = 0.2  # Speed cap for tag centering movements
            
            dx = tag_cx - center_x
            dy = tag_cy - center_y
            
            if abs(dx) < tol and abs(dy) < tol:
                vx, vy, yaw_rate = 0.0, 0.0, 0.0 
                cv2.putText(display_frame, f"LOCKED: ID {tag_id}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Calculate smooth adjustment velocity
                vx = np.clip(-dy * gain_p, -max_adj_vel, max_adj_vel)
                vy = np.clip(dx * gain_p, -max_adj_vel, max_adj_vel)
                
                cv2.putText(display_frame, "ALIGNING (SMOOTH)", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 2. Line Following
    if not tag_active:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
        
        if lines is not None:
            centers_x = []
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                centers_x.append((x1 + x2) / 2)
                angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            avg_cx = np.mean(centers_x)
            avg_ang = np.mean(angles)

            vx = speed_base
            error_x = (avg_cx - center_x) / center_x
            vy = error_x * 0.6
            yaw_rate = (avg_ang + 90) * 0.04
            
            cv2.putText(display_frame, "FOLLOWING PATH", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, "SEARCHING FOR PATH", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return vx, vy, yaw_rate, display_frame, tag_info, tag_active

# --- Execution Loops ---

def _camera_and_control_loop(host, port, stop_event, m, speed_base):
    try:
        sock = socket.create_connection((host, port), timeout=5)
        print(f'Socket: Connected on {host}:{port}')
    except Exception as e:
        print(f'Socket Error: {e}')
        stop_event.set()
        return

    last_scanned_id = None
    scan_timer = None
    tags_scanned_count = 0

    try:
        while not stop_event.is_set():
            hdr = _recvn(sock, 4)
            if not hdr: break
            w, h = struct.unpack('<HH', hdr)
            buf = _recvn(sock, int(w) * int(h))
            if not buf: break
            
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w))
            
            vx, vy, yaw_rate, res_img, tag_data, tag_active = process_frame(frame, speed_base, last_scanned_id)
            
            # Hover & Scan logic
            if tag_active and vx == 0 and vy == 0: 
                if scan_timer is None:
                    scan_timer = time.time()
                
                # Hold for 1.5 seconds to scan
                if time.time() - scan_timer > 1.5:
                    tag_id, hamming = tag_data
                    status = "VERIFIED" if hamming == 0 else f"UNCERTAIN (H:{hamming})"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] SCAN SUCCESS -> ID: {tag_id} | {status}")
                    
                    last_scanned_id = tag_id
                    scan_timer = None
                    tags_scanned_count += 1
                    
                    # CONDITION: If this is the FIRST tag, turn 90 degrees right
                    if tags_scanned_count == 1:
                        print("First tag detected! Performing 90-degree turn.")
                        rotate_90_clockwise(m)
            else:
                if not tag_active:
                    scan_timer = None

            # Send MAVLink velocity
            send_velocity(m, vx, vy, yaw_rate)
            
            cv2.imshow('Drone Vision', res_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    except Exception as e:
        print(f"Loop error: {e}")
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
    ap.add_argument('--alt', type=float, default=0.8)
    ap.add_argument('--speed', type=float, default=0.35)
    ap.add_argument('--mav', type=str, default='udp:0.0.0.0:14550')
    ap.add_argument('--host', type=str, default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5599)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.mav)
    wait_heartbeat(m)
    set_mode_guided(m)

    if not arm(m, force=args.force):
        print("Failed to arm.")
        return

    if not takeoff(m, args.alt):
        print("Takeoff timed out.")
        land(m)
        return

    stop_event = threading.Event()
    cam_thread = threading.Thread(
        target=_camera_and_control_loop, 
        args=(args.host, args.port, stop_event, m, args.speed), 
        daemon=True
    )
    cam_thread.start()

    print("Mission Running. Monitoring Camera...")
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()

    land(m)

if __name__ == '__main__':
    main()