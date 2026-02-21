#!/usr/bin/env python3
"""
MAVLink Line Follower & AprilTag Mission:
 - Connect to MAVLink and wait for Heartbeat
 - Arm and Take off
 - Receive real-time camera feed via socket
 - Simultaneously follow lines and scan AprilTags
 - CONDITION 1: Rotate 90 degrees right after the FIRST tag is scanned.
 - CONDITION 2: After scanning the SECOND tag, perform a precision land on it.
 - UI: Displays detected line angle, Hough lines, and scanned AprilTag IDs.
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
    print('Switching to GUIDED mode...')
    # Ensure we clear any existing overrides or old states
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_GUIDED,
        0, 0, 0, 0, 0, 0
    )
    # APM Specific set_mode
    try:
        m.set_mode_apm('GUIDED')
    except Exception:
        pass
    time.sleep(2) # Give SITL time to stabilize mode

def is_armed(m):
    m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
    if not hb: return False
    return bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

def arm(m, force=False):
    print(f'Arming (force={force})...')
    # Pre-arm check: ensure we are in GUIDED
    set_mode_guided(m)
    
    if not force:
        try:
            m.arducopter_arm()
            for _ in range(5):
                ack = m.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
                if ack and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    break
        except Exception: pass
        
        end = time.time() + 5
        while time.time() < end:
            if is_armed(m): return True
            time.sleep(0.5)
    
    # Force arming if standard arming fails
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 21196, 0, 0, 0, 0, 0)
    time.sleep(2)
    return is_armed(m)

def takeoff(m, alt):
    print(f'Takeoff to {alt}m...')
    # Use standard command_long for takeoff
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, float(alt))
    
    # Wait for the drone to actually leave the ground
    end = time.time() + 20
    while time.time() < end:
        msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if not msg: continue
        alt_m = msg.relative_alt / 1000.0
        if alt_m >= alt - 0.3: 
            print(f"Takeoff Complete. Altitude: {alt_m:.2f}m")
            return True
        time.sleep(0.2)
    return False

def rotate_90_clockwise(m, duration_s=4.0):
    print(f'Action: Rotating 90 degrees clockwise')
    angular_speed = 90.0 / duration_s 
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, 90.0, 
                            angular_speed, 1.0, 1.0, 0, 0, 0)
    time.sleep(duration_s)

def send_velocity(m, vx, vy, vz=0.0, yaw_rate=0.0):
    """Sends velocity commands in BODY_NED frame (vx=forward, vy=right, vz=down)."""
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000101111000111, 
        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate
    )

def land(m):
    print('Initiating Final Land Command...')
    try:
        m.set_mode_apm('LAND')
    except Exception:
        m.mav.set_mode_send(m.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
    
    end = time.time() + 15
    while time.time() < end:
        if not is_armed(m):
            print('Landed and Disarmed.')
            return True
        time.sleep(0.5)
    return False

# --- Vision Processing ---

def process_frame(frame, speed_base, last_scanned_id, tags_count, force_hover=False):
    gray = frame.copy()
    h, w = gray.shape
    vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0
    display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    center_x, center_y = w // 2, h // 2
    
    # UI Overlay: Global Stats
    cv2.putText(display_frame, f"Tags Scanned: {tags_count}", (w - 180, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if last_scanned_id is not None:
        cv2.putText(display_frame, f"Last ID: {last_scanned_id}", (w - 180, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    detections = TAG_DETECTOR.detect(gray)
    tag_active = False
    tag_info = None 

    if len(detections) > 0:
        det = detections[0]
        tag_id, hamming = det.tag_id, det.hamming
        tag_cx, tag_cy = det.center
        
        pts = np.array(det.corners).astype(int)
        cv2.polylines(display_frame, [pts], True, (0, 0, 255), 3)
        cv2.putText(display_frame, f"TAG ID: {tag_id}", (int(tag_cx) - 40, int(tag_cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if tag_id != last_scanned_id or force_hover:
            tag_active = True
            tag_info = (tag_id, hamming)
            tol = 50 
            dx = tag_cx - center_x
            dy = tag_cy - center_y
            
            if abs(dx) < tol and abs(dy) < tol:
                vx, vy = 0.0, 0.0
                cv2.putText(display_frame, "LOCK: CENTERED", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                vx = np.clip(-dy * 0.002, -0.2, 0.2)
                vy = np.clip(dx * 0.002, -0.2, 0.2)
                cv2.putText(display_frame, "LOCK: ALIGNING", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Line Following
    if not tag_active and not force_hover:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
        if lines is not None:
            c_x = [ (l[0][0]+l[0][2])/2 for l in lines ]
            angs = [ np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines ]
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            avg_ang = np.mean(angs)
            vx = speed_base
            vy = ((np.mean(c_x) - center_x) / center_x) * 0.6
            yaw_rate = (avg_ang + 95) * 0.04
            
            cv2.putText(display_frame, f"ANGLE: {round(avg_ang, 1)}", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "MISSION: FOLLOW PATH", (10, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, "MISSION: SEARCHING", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return vx, vy, yaw_rate, display_frame, tag_info, tag_active

# --- Execution Loops ---

def _camera_and_control_loop(host, port, stop_event, m, speed_base):
    # Short sleep to let simulation clock settle after reset
    time.sleep(1)
    
    try:
        sock = socket.create_connection((host, port), timeout=5)
        # Flush any stale frames in the socket
        sock.setblocking(False)
        try:
            while sock.recv(4096): pass
        except Exception: pass
        sock.setblocking(True)
    except Exception as e:
        print(f'Socket Error: {e}'); stop_event.set(); return

    last_scanned_id = None
    scan_timer = None
    tags_scanned_count = 0
    landing_mode = False

    try:
        while not stop_event.is_set():
            hdr = _recvn(sock, 4)
            if not hdr: break
            w, h = struct.unpack('<HH', hdr)
            buf = _recvn(sock, int(w) * int(h))
            if not buf: break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w))
            
            vx, vy, yaw_rate, res_img, tag_data, tag_active = process_frame(
                frame, speed_base, last_scanned_id, tags_scanned_count, force_hover=landing_mode
            )
            
            vz = 0.0
            
            if landing_mode:
                if vx == 0 and vy == 0:
                    vz = 0.25 
                    cv2.putText(res_img, "ALT: DESCENDING ON TAG", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                
                msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
                if msg:
                    curr_alt = msg.relative_alt / 1000.0
                    if curr_alt < 0.3:
                        land(m)
                        stop_event.set()
            
            elif tag_active and vx == 0 and vy == 0: 
                if scan_timer is None: scan_timer = time.time()
                if time.time() - scan_timer > 1.5:
                    tag_id, hamming = tag_data
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] SCAN SUCCESS -> ID: {tag_id}")
                    last_scanned_id = tag_id
                    scan_timer = None
                    tags_scanned_count += 1
                    
                    if tags_scanned_count == 1:
                        rotate_90_clockwise(m)
                    elif tags_scanned_count == 2:
                        landing_mode = True
            else:
                if not tag_active: scan_timer = None

            send_velocity(m, vx, vy, vz, yaw_rate)
            cv2.imshow('Drone Vision AI', res_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): stop_event.set(); break
            
    finally:
        sock.close(); cv2.destroyAllWindows()

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
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.mav)
    wait_heartbeat(m)
    
    # 1. Reset Mode and ensure we are cleared for a new run
    set_mode_guided(m)
    
    # 2. Re-arm and check status
    if not arm(m):
        print("Failed to arm. Suggest restarting Python script AFTER resetting Webots.")
        return

    # 3. Fresh Takeoff
    if not takeoff(m, args.alt): 
        land(m)
        return

    stop_event = threading.Event()
    cam_thread = threading.Thread(target=_camera_and_control_loop, args=(args.host, args.port, stop_event, m, args.speed), daemon=True)
    cam_thread.start()

    try:
        while not stop_event.is_set(): time.sleep(1)
    except KeyboardInterrupt: stop_event.set()
    if is_armed(m): land(m)

if __name__ == '__main__':
    main()