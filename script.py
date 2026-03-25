#!/usr/bin/env python3
"""
MAVLink Line Follower & AprilTag Airport Mission:
 - Connect to MAVLink and wait for Heartbeat
 - Arm and Take off to cruise altitude (1.5 m)
 - Receive real-time greyscale camera feed via socket
 - Follow the bright path using brightness-threshold + contour centroid steering
 - Detect AprilTags (family tag36h11); each tag encodes a 3-digit number:
     Digit 1 -> Country code
     Digit 2 -> Landing status  (1 = safe to land, 0 = unsafe)
     Digit 3 -> Number of connected paths leaving this airport
 - Target countries declared in:  Airports = [country1, country2]
   (second value is 0 if only one country is needed)
 - On centring over a tag:
     * Parse the three digits
     * If country in Airports AND digit-2 == 1 AND not yet visited -> land 5 s
     * Take a locked snapshot, analyse path directions from it, rotate if needed
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
from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Mission configuration  -- evaluators change these two values
# ---------------------------------------------------------------------------
Airports = [1, 2]

CRUISE_ALT        = 2.0    # line-following altitude (m)
FORWARD_SPEED     = 0.40   # m/s along the path
TAG_CENTRE_TOL_PX = 40     # px radius within which tag is "centred"
TAG_LOCK_FRAMES   = 5      # consecutive centred frames before acting
BRANCH_MIN_AREA   = 600    # minimum bright-pixel count to count as a path branch

# ---------------------------------------------------------------------------
# Global flags
# ---------------------------------------------------------------------------
is_rotating       = False
visited_countries = set()

# ---------------------------------------------------------------------------
# AprilTag detector
# ---------------------------------------------------------------------------
TAG_OPTIONS  = apriltag.DetectorOptions(families="tag36h11")
TAG_DETECTOR = apriltag.Detector(TAG_OPTIONS)

# ---------------------------------------------------------------------------
# Shared display state  (written by control thread, read by display thread)
# ---------------------------------------------------------------------------
_display_lock      = threading.Lock()
_display_frame     = None    # latest BGR debug frame to show
_display_status    = ''      # status banner text (e.g. "LANDING…")
_display_armed     = True    # when False the window shows a dark "disarmed" banner


def _display_thread_fn(stop_event):
    """
    Dedicated OpenCV display thread.

    Runs independently of the control loop so that cv2.imshow / cv2.waitKey
    are always called at ~30 fps regardless of how long land_and_hold() blocks.
    This prevents the OS from reporting the window as 'not responding'.
    """
    while not stop_event.is_set():
        with _display_lock:
            frame  = _display_frame
            status = _display_status
            armed  = _display_armed

        if frame is not None:
            disp = frame.copy()
            if not armed:
                # Dark translucent overlay to show the feed is paused
                overlay = disp.copy()
                cv2.rectangle(overlay, (0, 0), (disp.shape[1], disp.shape[0]),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.55, disp, 0.45, 0, disp)
                cv2.putText(disp, 'DISARMED — feed paused',
                            (10, disp.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            if status:
                cv2.putText(disp, status,
                            (5, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 165, 255), 1)
            cv2.imshow('Drone Vision AI', disp)

        key = cv2.waitKey(33) & 0xFF   # ~30 fps; 33 ms
        if key == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Socket helpers
# ---------------------------------------------------------------------------

def _recvn(sock, n):
    """Receive exactly n bytes from sock, returning None on disconnect or error."""
    data = bytearray()
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except (OSError, socket.timeout):
            return None
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def grab_frame(sock):
    """Receive one colour frame (3 bytes/pixel) from webots_vehicle.py."""
    hdr = _recvn(sock, 4)
    if not hdr:
        return None
    fw, fh = struct.unpack('<HH', hdr)
    buf = _recvn(sock, int(fw) * int(fh) * 3)
    if not buf:
        return None
    return np.frombuffer(buf, dtype=np.uint8).reshape((fh, fw, 3))

# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

def wait_heartbeat(m):
    print('Waiting for heartbeat...')
    m.wait_heartbeat()
    print(f'  Heartbeat: sys={m.target_system} comp={m.target_component}')


def set_mode_guided(m):
    print('  -> GUIDED mode')
    try:
        m.set_mode_apm('GUIDED')
    except Exception:
        m.mav.set_mode_send(
            m.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            4)
    time.sleep(2)


def is_armed(m):
    hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
    if not hb:
        return False
    return bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


def arm(m, force=False):
    print(f'Arming (force={force})...')
    set_mode_guided(m)
    if not force:
        try:
            m.arducopter_arm()
            for _ in range(5):
                ack = m.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
                if ack and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    break
        except Exception:
            pass
        end = time.time() + 5
        while time.time() < end:
            if is_armed(m):
                return True
            time.sleep(0.5)
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0)
    time.sleep(2)
    return is_armed(m)


def send_velocity(m, vx, vy, vz=0.0, yaw_rate=0.0):
    """Send velocity in BODY_NED frame. Silently skipped while rotating."""
    if is_rotating:
        return
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000101111000111,
        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate)


def get_altitude(m):
    msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
    if msg:
        return msg.relative_alt / 1000.0
    return 0.0


def takeoff(m, alt):
    print(f'Takeoff -> {alt} m ...')
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, float(alt))
    end = time.time() + 20
    while time.time() < end:
        msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if not msg:
            continue
        if msg.relative_alt / 1000.0 >= alt - 0.3:
            print(f'  Takeoff complete: {msg.relative_alt/1000.0:.2f} m')
            return True
        time.sleep(0.2)
    return False


def _set_angle_limit(m, max_deg):
    value = 3000.0 if max_deg == 0 else float(max_deg * 100)
    m.mav.param_set_send(
        m.target_system, m.target_component,
        b'ANGLE_MAX',
        value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.1)


def land_and_hold(m, sock, hold_s=5):
    print(f'  Landing and holding {hold_s} s ...')
    try:
        m.set_mode_apm('LAND')
    except Exception:
        m.mav.command_long_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            1, 9, 0, 0, 0, 0, 0)

    end = time.time() + 20
    while time.time() < end:
        if get_altitude(m) < 0.25:
            break
        time.sleep(0.3)

    time.sleep(hold_s)
    print('  Re-arming and climbing back...')
    if not arm(m, force=True):
        print('  WARNING: re-arm failed.')
        return False

    _set_angle_limit(m, 8)
    ok = takeoff(m, CRUISE_ALT)
    _set_angle_limit(m, 0)

    if not ok:
        return False

    print('  Holding position — waiting for camera feed...')
    FEED_WAIT_MAX = 15.0
    feed_deadline = time.time() + FEED_WAIT_MAX
    got_frame = False

    sock.setblocking(False)
    try:
        while time.time() < feed_deadline:
            m.mav.set_position_target_local_ned_send(
                0, m.target_system, m.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000101111000111, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0)
            try:
                peek = sock.recv(1, socket.MSG_PEEK)
                if peek:
                    got_frame = True
                    break
            except BlockingIOError:
                pass
            except OSError:
                break
            time.sleep(0.2)
    finally:
        sock.setblocking(True)

    if got_frame:
        print('  Camera feed live — resuming mission.')
    else:
        print(f'  WARNING: camera feed did not return within {FEED_WAIT_MAX:.0f} s.')

    _drain_socket(sock, label='post-land-hold')
    return True


def land_final(m):
    """Final landing — drone stays disarmed."""
    print('Final landing...')
    try:
        m.set_mode_apm('LAND')
    except Exception:
        m.mav.command_long_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            1, 9, 0, 0, 0, 0, 0)
    end = time.time() + 20
    while time.time() < end:
        if not is_armed(m):
            print('  Disarmed. Mission complete.')
            return True
        time.sleep(0.5)
    return False


def rotate_relative(m, degrees, speed_deg_s=25.0):
    global is_rotating
    cw_flag = 1 if degrees >= 0 else -1
    abs_deg = abs(degrees)
    duration = abs_deg / speed_deg_s + 0.8
    print(f'  Rotating {degrees:+.0f} deg ({"CW" if cw_flag > 0 else "CCW"}, {duration:.1f} s)...')
    is_rotating = True
    try:
        m.mav.command_long_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0, abs_deg, speed_deg_s, cw_flag, 1, 0, 0, 0)
        time.sleep(duration)
    finally:
        is_rotating = False
    print('  Rotation done.')

# ---------------------------------------------------------------------------
# AprilTag decoding
# ---------------------------------------------------------------------------

def decode_tag(tag_id):
    s           = f'{int(tag_id):03d}'
    country     = int(s[0])
    status      = int(s[1])
    connections = int(s[2])
    return country, status, connections

# ---------------------------------------------------------------------------
# Vision: greyscale path detection (line following)
# ---------------------------------------------------------------------------

def _yellow_mask(bgr_frame):
    blurred = cv2.GaussianBlur(bgr_frame, (15, 15), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15,  80, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    k_open  = np.ones((5,  5),  np.uint8)
    k_close = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def _path_centreline(mask, scan_bottom):
    pts = []
    for x in range(mask.shape[1]):
        col = mask[:scan_bottom, x]
        ys  = np.where(col > 0)[0]
        if len(ys) < 4: continue
        mid_y = int((int(ys[0]) + int(ys[-1])) // 2)
        pts.append((x, mid_y))
    if len(pts) < 6: return None
    return np.array(pts, dtype=np.int32)


def _pick_path_contour(contours, frame_h):
    best = None
    best_top  = frame_h
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200: continue
        top_y = int(cnt[:, 0, 1].min())
        if top_y < best_top or (abs(top_y - best_top) <= 5 and area > best_area):
            best      = cnt
            best_top  = top_y
            best_area = area
    return best


def detect_path(frame):
    """
    Detect the yellow path line and return steering commands.
    """
    if frame is None:
        return 0.0, 0.0, 0.0, None, None

    h, w   = frame.shape[:2]
    cx_img = w // 2
    THIRD  = h // 3
    debug  = frame.copy()
    mask = _yellow_mask(frame)

    cv2.line(debug, (0, THIRD), (w, THIRD), (0, 140, 255), 1)

    contours_full, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vx, vy, yaw_rate = 0.0, 0.0, 0.0

    if not contours_full:
        return vx, vy, yaw_rate, debug, None

    path_cnt = _pick_path_contour(contours_full, h)
    if path_cnt is None:
        return vx, vy, yaw_rate, debug, None

    cv2.drawContours(debug, [path_cnt], -1, (0, 255, 0), 2)

    cnt_xs    = path_cnt[:, 0, 0]
    x_min, x_max = int(cnt_xs.min()), int(cnt_xs.max())
    masked_cl = np.zeros_like(mask)
    masked_cl[:, x_min:x_max + 1] = mask[:, x_min:x_max + 1]

    cl_pts = _path_centreline(masked_cl, h // 2)

    if cl_pts is not None:
        step = max(1, len(cl_pts) // 40)
        for px, py in cl_pts[::step]:
            cv2.circle(debug, (int(px), int(py)), 2, (255, 255, 0), -1)

    if cl_pts is not None and len(cl_pts) >= 2:
        top_idx   = int(np.argmin(cl_pts[:, 1]))
        top_x     = int(cl_pts[top_idx, 0])
        top_y     = int(cl_pts[top_idx, 1])
        mid_x     = int(cl_pts[:, 0].mean())
    else:
        top_y     = int(path_cnt[:, 0, 1].min())
        top_x     = int(path_cnt[path_cnt[:, 0, 1] == top_y, 0, 0].mean())
        mid_x     = int(path_cnt[:, 0, 0].mean())

    in_top_third = top_y < THIRD
    path_zone = {
        'in_top_third': in_top_third,
        'top_x':        top_x,
        'top_y':        top_y,
        'mid_x':        mid_x,
    }

    cv2.circle(debug, (top_x, top_y), 6, (0, 128, 255), -1)

    if not in_top_third:
        cv2.putText(debug, 'PATH BELOW ZONE — awaiting reacq',
                    (5, THIRD - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 255), 1)
        cv2.line(debug, (cx_img, 0), (cx_img, h), (255, 0, 0), 1)
        return vx, vy, yaw_rate, debug, path_zone

    if cl_pts is not None and len(cl_pts) >= 6:
        fit_pts = cl_pts
    else:
        fit_pts = path_cnt[:, 0, :]
        upper   = fit_pts[fit_pts[:, 1] < h // 2]
        fit_pts = upper if len(upper) >= 6 else fit_pts

    fit  = cv2.fitLine(fit_pts.reshape(-1, 1, 2).astype(np.float32),
                       cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx_f, vy_f = float(fit[0]), float(fit[1])
    x0,   y0   = float(fit[2]), float(fit[3])

    if vy_f > 0:
        vx_f, vy_f = -vx_f, -vy_f

    heading_err = math.atan2(vx_f, -vy_f)

    top_exit_offset = (top_x - cx_img) / cx_img
    top_exit_yaw    = float(np.clip(top_exit_offset * 0.4, -0.5, 0.5))

    yaw_rate = float(np.clip(heading_err * 1.5 + top_exit_yaw, -1.8, 1.8))
    vx       = float(np.clip(FORWARD_SPEED * (1.0 - 0.7 * abs(heading_err)),
                             0.08, FORWARD_SPEED))

    sc  = 50
    pt1 = (int(x0 - vx_f * sc), int(y0 - vy_f * sc))
    pt2 = (int(x0 + vx_f * sc), int(y0 + vy_f * sc))
    cv2.arrowedLine(debug, pt1, pt2, (0, 255, 255), 2, tipLength=0.3)

    if cl_pts is not None:
        front_pts = cl_pts[cl_pts[:, 1] < THIRD]
    else:
        front_pts = np.array([])

    if len(front_pts) >= 2:
        cx = int(front_pts[:, 0].mean())
        cy = int(front_pts[:, 1].mean())
    else:
        M = cv2.moments(path_cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            return vx, vy, yaw_rate, debug, path_zone

    cv2.circle(debug, (cx, cy), 8, (0, 0, 255), -1)
    cv2.line(debug, (cx_img, 0), (cx_img, h), (255, 0, 0), 1)

    lateral_error = (cx - cx_img) / cx_img
    vy = float(np.clip(lateral_error * 0.6, -0.4, 0.4))

    n_cl = len(cl_pts) if cl_pts is not None else 0
    cv2.putText(debug,
                f'hdg={math.degrees(heading_err):+.1f}d '
                f'tex={top_exit_yaw:+.2f} '
                f'lat={lateral_error:+.2f} '
                f'yr={yaw_rate:+.2f} cl={n_cl}',
                (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 0), 1)

    return vx, vy, yaw_rate, debug, path_zone

def _bright_pixel_count(bgr_roi, thresh=None):
    if bgr_roi is None or bgr_roi.size == 0: return 0
    mask = _yellow_mask(bgr_roi)
    return cv2.countNonZero(mask)


def decide_turn(snapshot):
    """
    Analyse a locked BGR colour snapshot to decide whether to rotate at this
    airport, and by how many degrees.
    """
    h, w = snapshot.shape[:2]

    masked = snapshot.copy()
    pad_y0, pad_y1 = h // 4, 3 * h // 4
    pad_x0, pad_x1 = w // 4, 3 * w // 4
    masked[pad_y0:pad_y1, pad_x0:pad_x1] = 0

    strip, stripw = h // 4, w // 4
    roi_top    = masked[0:strip,   :]
    roi_bottom = masked[3*strip:,  :]
    roi_left   = masked[:,         0:stripw]
    roi_right  = masked[:,         3*stripw:]

    score_top    = _bright_pixel_count(roi_top)
    score_bottom = _bright_pixel_count(roi_bottom)
    score_left   = _bright_pixel_count(roi_left)
    score_right  = _bright_pixel_count(roi_right)

    print('\n' + '-'*50)
    print('  [decide_turn] Yellow pixel scores (pad masked):')
    print(f'    AHEAD  (top)   : {score_top}')
    print(f'    BEHIND (bottom): {score_bottom}  <-- arrival, excluded')
    print(f'    LEFT           : {score_left}')
    print(f'    RIGHT          : {score_right}')
    print(f'    Branch min area: {BRANCH_MIN_AREA}')

    right_ok = score_right > BRANCH_MIN_AREA
    left_ok  = score_left  > BRANCH_MIN_AREA
    top_ok   = score_top   > BRANCH_MIN_AREA

    if top_ok:
        rotation_deg = 0
        reason = f'path ahead ({score_top} px) — going straight'
    elif right_ok and left_ok:
        rotation_deg = 90
        reason = f'both lateral — preferring RIGHT ({score_right} vs {score_left})'
    elif right_ok:
        rotation_deg = 90
        reason = f'turning RIGHT ({score_right} px)'
    elif left_ok:
        rotation_deg = -90
        reason = f'turning LEFT ({score_left} px)'
    else:
        rotation_deg = 0
        reason = 'no clear branch detected — defaulting straight'

    print(f'  => Decision: {rotation_deg:+d} deg  ({reason})')
    print('-'*50 + '\n')
    return rotation_deg

# ---------------------------------------------------------------------------
# Airport navigation and Reacquisition
# ---------------------------------------------------------------------------

def _drain_socket(sock, label=''):
    """Discard all frames currently buffered in the TCP socket."""
    sock.setblocking(False)
    drained = 0
    while True:
        try:
            peek = sock.recv(1, socket.MSG_PEEK)
            if not peek: break
        except (BlockingIOError, OSError):
            break

        sock.setblocking(True)
        hdr = _recvn(sock, 4)
        if hdr is None: break
        fw2, fh2 = struct.unpack('<HH', hdr)
        payload   = _recvn(sock, int(fw2) * int(fh2) * 3)
        if payload is None: break
        drained += 1
        sock.setblocking(False)

    sock.setblocking(True)
    if drained:
        print(f'  Drained {drained} stale frames ({label}).')


def execute_airport_navigation(m, sock, rotation_deg):
    """
    Execute the post-airport navigation decision in ONE bulk rotation.
    Includes a timer criteria to wait for camera feed sync afterwards.
    """
    _drain_socket(sock, label='pre-rotation backlog')

    if abs(rotation_deg) <= 1:
        print('  Navigation: going straight, no rotation needed.')
        return

    print(f'  Navigation: rotating {rotation_deg:+d} deg in one go...')
    rotate_relative(m, rotation_deg, speed_deg_s=25.0)

    # ── Wait for camera sync after a bulk turn ────────────────────────────────
    print('  Navigation: waiting for camera sync...')
    _drain_socket(sock, label='post-rotation drain')
    
    deadline = time.time() + 3.0  # timer/wait criteria
    while time.time() < deadline:
        fresh = grab_frame(sock)
        if fresh is not None:
            break
        time.sleep(0.15)

    print('  Navigation: rotation complete.')


def perform_reacquisition_step(m, sock, direction):
    """
    A function to rotate the drone in 5 degree intervals to search for the path.
    Implements a wait criteria for camera feed sync to ensure stability mid-search.
    """
    print(f'  [reacq] Rotating 5 deg {"CW" if direction > 0 else "CCW"} to search for path...')
    rotate_relative(m, 5 * direction, speed_deg_s=25.0)

    # ── Wait for camera sync after a 5-degree step ───────────────────────────
    _drain_socket(sock, label='reacq step drain')
    
    step_deadline = time.time() + 2.0  # timer/wait criteria
    while time.time() < step_deadline:
        step_frame = grab_frame(sock)
        if step_frame is not None:
            # We've synchronized! 
            return True
        time.sleep(0.1)

    print('  [reacq] WARNING: Camera feed sync timed out after rotation step.')
    return False

# ---------------------------------------------------------------------------
# Main camera + control loop
# ---------------------------------------------------------------------------

def _camera_and_control_loop(host, port, stop_event, m):
    global _display_frame, _display_status, _display_armed
    time.sleep(1)
    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.settimeout(None)
    except Exception as e:
        print(f'Socket error: {e}')
        stop_event.set()
        return

    target_countries   = set(c for c in Airports if c != 0)
    at_airport         = False
    last_tag_id        = None
    tag_lock_frames    = 0
    tag_announced      = False
    tag_cooldown_until = 0.0

    # Reacquisition state
    REACQ_TIMEOUT       = 2.0
    REACQ_BOTTOM_EXCL   = None
    path_lost_since     = None
    reacq_active        = False
    reacq_direction     = 0

    try:
        while not stop_event.is_set():
            # ── 1. Receive frame ──────────────────────────────────────────────
            hdr = _recvn(sock, 4)
            if not hdr: break
            fw, fh = struct.unpack('<HH', hdr)
            buf = _recvn(sock, int(fw) * int(fh) * 3)
            if not buf: break

            frame = np.frombuffer(buf, dtype=np.uint8).reshape((fh, fw, 3))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            debug = frame.copy()

            REACQ_BOTTOM_EXCL = int(fh * 3 / 4)
            
            # Reset motion commands for this tick
            vx = vy = yaw_rate = vz = 0.0
            tag_handled = False
            status_txt  = ''

            # ── 2. AprilTag Detection (HIGHEST PRIORITY) ──────────────────────
            # We scan for tags first. If a tag is visible, it cancels any ongoing
            # 5-degree reacquisition searches immediately and aligns the drone.
            detections = TAG_DETECTOR.detect(gray)
            in_cooldown = time.time() < tag_cooldown_until

            if in_cooldown:
                remaining  = tag_cooldown_until - time.time()
                status_txt = f'Tag scan blocked ({remaining:.1f}s)'

            if detections and not at_airport and not in_cooldown:
                tag_handled = True
                det    = detections[0]
                tid    = det.tag_id
                tx, ty = det.center
                dx     = tx - fw / 2
                dy     = ty - fh / 2

                # Enforce Priority: Cancel Reacquisition immediately
                if reacq_active:
                    print('  [reacq] TAG PRIORITY - Aborting 5° path search to align with tag.')
                reacq_active    = False
                reacq_direction = 0
                path_lost_since = None

                if abs(dx) < TAG_CENTRE_TOL_PX and abs(dy) < TAG_CENTRE_TOL_PX:
                    if tid != last_tag_id:
                        tag_lock_frames = 0
                        tag_announced   = False
                        last_tag_id     = tid
                    tag_lock_frames += 1

                    if not tag_announced:
                        country, status, connections = decode_tag(tid)
                        print('\n' + '='*60)
                        print(f'  AIRPORT IN SIGHT   Tag ID : {tid}')
                        print(f'  Country            : {country}')
                        print(f'  Landing status     : {status}  (1=safe, 0=unsafe)')
                        print(f'  Connections        : {connections}')
                        print('='*60 + '\n')
                        tag_announced = True

                    if tag_lock_frames >= TAG_LOCK_FRAMES:
                        at_airport = True
                        send_velocity(m, 0, 0, 0, 0)
                        time.sleep(0.5)

                        country, status, connections = decode_tag(tid)
                        print('\n' + '='*60)
                        print(f'  AIRPORT LOCKED     Tag ID : {tid}')
                        print('='*60 + '\n')

                        for _ in range(6): grab_frame(sock)
                        snapshot = grab_frame(sock)
                        if snapshot is None: snapshot = frame

                        should_land = (status == 1 and country in target_countries and country not in visited_countries)
                        if should_land:
                            print(f'  >> Landing at country {country}')
                            with _display_lock:
                                _display_status = f'LANDING at country {country}…'
                                _display_armed  = False
                            land_and_hold(m, sock, hold_s=5)
                            with _display_lock:
                                _display_status = 'Resuming mission…'
                                _display_armed  = True
                            visited_countries.add(country)

                            if target_countries.issubset(visited_countries):
                                print('  All targets reached — final landing.')
                                with _display_lock:
                                    _display_status = 'Mission complete — final landing'
                                    _display_armed  = False
                                land_final(m)
                                stop_event.set()
                                break
                        else:
                            print('  No landing, continuing mission.')

                        if not stop_event.is_set():
                            # Regular navigation executed in one go!
                            rotation_deg = decide_turn(snapshot)
                            execute_airport_navigation(m, sock, rotation_deg)

                            at_airport         = False
                            tag_lock_frames    = 0
                            tag_announced      = False
                            tag_cooldown_until = time.time() + 8.0
                            path_lost_since    = None 
                            reacq_active       = False
                            with _display_lock:
                                _display_status = ''

                else:
                    # Keep aligning to the tag
                    tag_lock_frames = 0
                    tag_announced   = False
                    vx = float(np.clip(-dy * 0.001, -0.10, 0.10))
                    vy = float(np.clip( dx * 0.001, -0.10, 0.10))
                    yaw_rate = 0.0
                    cv2.putText(debug, 'ALIGNING TO TAG', (5, fh // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elif not detections:
                if tag_lock_frames > 0 and not at_airport:
                    tag_lock_frames = 0
                    tag_announced   = False

            # ── 3. Line Following & Reacquisition (Only if tag not handled) ───
            if not tag_handled and not at_airport:
                vx_path, vy_path, yaw_path, debug_path, path_zone = detect_path(frame)
                debug = debug_path

                if path_zone is not None and path_zone['in_top_third']:
                    # Path is perfectly in the 1/3 boundary
                    path_lost_since = None
                    reacq_active    = False
                    reacq_direction = 0
                    vx, vy, yaw_rate = vx_path, vy_path, yaw_path
                else:
                    # Path is lost entirely or strictly below the 1/3 boundary
                    if path_lost_since is None:
                        path_lost_since = time.time()

                    # Check if timer elapsed to start 5-degree searching
                    if not reacq_active and (time.time() - path_lost_since >= REACQ_TIMEOUT):
                        reacq_active = True
                        if path_zone is not None and path_zone['top_y'] <= REACQ_BOTTOM_EXCL:
                            reacq_direction = 1 if path_zone['mid_x'] >= fw // 2 else -1
                        else:
                            reacq_direction = 1

                    if reacq_active:
                        # Stop all standard motion while reacquiring
                        vx = vy = yaw_rate = 0.0
                        
                        cv2.putText(debug, f'REACQ SEARCH {"CW" if reacq_direction > 0 else "CCW"}',
                                    (5, fh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Push interface update immediately before we block for the camera step
                        with _display_lock:
                            _display_frame  = debug
                            _display_status = status_txt

                        # Trigger the requested 5-degree interval function with wait criteria
                        perform_reacquisition_step(m, sock, reacq_direction)
                        
                        # We jump straight back to the top of the loop.
                        # The next iteration will grab the newly synced frame and, crucially,
                        # CHECK FOR APRILTAGS FIRST before deciding to do another 5-deg step.
                        continue
                    else:
                        # Waiting for timeout before search kicks in; coastal movement
                        vx, vy, yaw_rate = vx_path, vy_path, yaw_path

            # ── 4. Send Motion Command ────────────────────────────────────────
            if not at_airport and not reacq_active:
                send_velocity(m, vx, vy, vz, yaw_rate)

            # ── 5. Push frame to display ──────────────────────────────────────
            if debug is not None:
                overlay_txt = (f"Visited:{sorted(visited_countries)}  "
                               f"Targets:{sorted(target_countries)}  "
                               f"Lock:{tag_lock_frames}")
                cv2.putText(debug, overlay_txt, (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                with _display_lock:
                    _display_frame  = debug
                    _display_status = status_txt

    finally:
        sock.close()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alt',   type=float, default=CRUISE_ALT,
                    help='Cruise altitude in metres (default 1.5)')
    ap.add_argument('--speed', type=float, default=FORWARD_SPEED,
                    help='Forward speed in m/s')
    ap.add_argument('--mav',   type=str,   default='udp:0.0.0.0:14550',
                    help='MAVLink connection string')
    ap.add_argument('--host',  type=str,   default='127.0.0.1',
                    help='Camera stream host')
    ap.add_argument('--port',  type=int,   default=5599,
                    help='Camera stream port')
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.mav)
    wait_heartbeat(m)
    set_mode_guided(m)

    if not arm(m):
        print('Arming failed.')
        return
    if not takeoff(m, args.alt):
        land_final(m)
        return

    stop_event = threading.Event()

    cam_thread = threading.Thread(
        target=_camera_and_control_loop,
        args=(args.host, args.port, stop_event, m),
        daemon=True)
    cam_thread.start()

    disp_thread = threading.Thread(
        target=_display_thread_fn,
        args=(stop_event,),
        daemon=True)
    disp_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    disp_thread.join(timeout=2)

    if is_armed(m):
        land_final(m)

if __name__ == '__main__':
    main()
