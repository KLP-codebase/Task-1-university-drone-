"""
Functions to control Iris ArduCopter drone in Webots simulation
"""

from pymavlink import mavutil
import time
import sys


class DroneController:
    """Controller class for ArduPilot SITL drone"""
    
    def __init__(self, sitl_ip="127.0.0.1", sitl_port=5760):
        """
        Initialize drone controller connection to ArduPilot SITL
        
        Args:
            sitl_ip (str): IP address of SITL instance
            sitl_port (int): Port of SITL instance (default MAVLink port)
        """
        self.sitl_ip = sitl_ip
        self.sitl_port = sitl_port
        self.master = None
        self.target_system = 1
        self.target_component = 1
        self.connect()
    
    def connect(self):
        """Connect to ArduPilot SITL via MAVLink"""
        try:
            connection_string = f"udp:{self.sitl_ip}:{self.sitl_port}"
            self.master = mavutil.mavlink_connection(connection_string)
            self.master.wait_heartbeat()
            print(f"Connected to ArduPilot SITL at {connection_string}")
        except Exception as e:
            print(f"Failed to connect to SITL: {e}")
            sys.exit(1)
    
    def send_command(self, command, param1=0, param2=0, param3=0, param4=0, 
                     param5=0, param6=0, param7=0):
        """
        Send a command message to the drone
        
        Args:
            command: MAV command ID
            param1-7: Command parameters
        """
        msg = self.master.mav.command_long_encode(
            self.target_system,
            self.target_component,
            command,
            0,              # confirmation
            param1,
            param2,
            param3,
            param4,
            param5,
            param6,
            param7
        )
        self.master.mav.send(msg)
    
    def set_mode(self, mode_name):
        """
        Set drone flight mode
        
        Args:
            mode_name (str): Mode name (e.g., "STABILIZE", "LOITER", "LAND")
        """
        mode_map = {
            "STABILIZE": 0,
            "ACRO": 1,
            "ALT_HOLD": 2,
            "AUTO": 3,
            "GUIDED": 4,
            "LOITER": 5,
            "RTL": 6,
            "CIRCLE": 7,
            "LAND": 9,
            "OF_LOITER": 10,
            "DRIFT": 11,
            "SPORT": 13,
            "FLIP": 14,
            "AUTOTUNE": 15,
            "POSHOLD": 16,
            "BRAKING": 17,
            "THROW": 18,
            "AVOID_ADSR": 19,
            "GUIDED_NOGPS": 20,
            "SMART_RTL": 21,
            "FLOWHOLD": 22,
            "FOLLOW": 23,
            "ZIGZAG": 24,
            "SYSTEMID": 25,
            "AUTOROTATE": 26,
        }
        
        if mode_name not in mode_map:
            print(f"Unknown mode: {mode_name}")
            return False
        
        mode = mode_map[mode_name]
        msg = self.master.mav.set_mode_encode(
            self.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode
        )
        self.master.mav.send(msg)
        print(f"Set mode to {mode_name}")
        return True
    
    def arm(self, arm=True):
        """
        Arm or disarm the drone
        
        Args:
            arm (bool): True to arm, False to disarm
        
        Returns:
            bool: True if successful
        """
        # MAV_CMD_COMPONENT_ARM_DISARM = 400
        # param1: 1 to arm, 0 to disarm
        arm_value = 1 if arm else 0
        
        self.send_command(
            command=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            param1=arm_value
        )
        
        action = "Arming" if arm else "Disarming"
        print(f"{action} drone...")
        time.sleep(1)
        return True
    
    def take_off(self, altitude=10):
        """
        Take off to specified altitude
        
        Args:
            altitude (float): Target altitude in meters (default 10m)
        
        Returns:
            bool: True if successful
        """
        # MAV_CMD_NAV_TAKEOFF = 22
        self.send_command(
            command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            param1=0,      # pitch value (0 for default)
            param2=0,
            param3=0,
            param4=0,      # yaw angle (0 for current heading)
            param5=0,
            param6=0,
            param7=altitude  # target altitude
        )
        print(f"Taking off to {altitude}m altitude...")
        return True
    
    def hover(self, duration=10, altitude=None):
        """
        Hover at current or specified altitude
        
        Args:
            duration (float): Duration to hover in seconds
            altitude (float): Target altitude in meters (None = current altitude)
        
        Returns:
            bool: True if successful
        """
        # Switch to LOITER mode to hover
        self.set_mode("LOITER")
        
        if altitude is not None:
            # Climb/descend to specified altitude while hovering
            msg = self.master.mav.set_position_target_local_ned_encode(
                time_boot_ms=0,
                target_system=self.target_system,
                target_component=self.target_component,
                coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask=0b1111_1111_1111_1000,  # position only
                x=0,
                y=0,
                z=-altitude,  # negative z = up in NED frame
                vx=0,
                vy=0,
                vz=0,
                afx=0,
                afy=0,
                afz=0,
                yaw=0,
                yaw_rate=0
            )
            self.master.mav.send(msg)
            print(f"Hovering at altitude {altitude}m for {duration}s...")
        else:
            print(f"Hovering for {duration}s...")
        
        time.sleep(duration)
        return True
    
    def land(self):
        """
        Land the drone
        
        Returns:
            bool: True if successful
        """
        # Switch to LAND mode
        self.set_mode("LAND")
        print("Landing drone...")
        return True
    
    def goto(self, lat, lon, altitude):
        """
        Fly to GPS waypoint
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            altitude (float): Altitude in meters
        
        Returns:
            bool: True if successful
        """
        # MAV_CMD_NAV_WAYPOINT = 16
        self.send_command(
            command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            param1=0,      # delay at waypoint in seconds
            param2=0,      # acceptance radius
            param3=0,      # pass through distance
            param4=0,      # yaw angle
            param5=lat,    # latitude
            param6=lon,    # longitude
            param7=altitude  # altitude
        )
        print(f"Flying to waypoint: ({lat}, {lon}) at {altitude}m...")
        return True
    
    def disconnect(self):
        """Disconnect from SITL"""
        if self.master:
            self.master.close()
            print("Disconnected from SITL")


# Convenience functions for simple operations

def arm_drone(controller):
    """Arm the drone"""
    return controller.arm(arm=True)


def disarm_drone(controller):
    """Disarm the drone"""
    return controller.arm(arm=False)


def hover_drone(controller, duration=10, altitude=10):
    """
    Hover the drone at specified altitude
    
    Args:
        controller: DroneController instance
        duration (float): Duration to hover in seconds
        altitude (float): Altitude in meters
    """
    # First ensure drone is armed and in air
    if not controller.arm(arm=True):
        print("Failed to arm drone")
        return False
    
    time.sleep(1)
    
    # Take off
    if not controller.take_off(altitude=altitude):
        print("Failed to take off")
        return False
    
    time.sleep(5)  # Wait for drone to reach altitude
    
    # Hover
    return controller.hover(duration=duration, altitude=altitude)


def land_drone(controller):
    """
    Land the drone
    
    Args:
        controller: DroneController instance
    """
    return controller.land()


def full_mission(controller, altitude=10, hover_duration=10):
    """
    Execute full mission: arm -> takeoff -> hover -> land
    
    Args:
        controller: DroneController instance
        altitude (float): Flight altitude in meters
        hover_duration (float): Duration to hover in seconds
    """
    print("\n=== Starting Full Mission ===\n")
    
    # Arm
    print("Step 1: Arming drone...")
    if not controller.arm(arm=True):
        print("Failed to arm drone")
        return False
    time.sleep(2)
    
    # Take off
    print("\nStep 2: Taking off...")
    if not controller.take_off(altitude=altitude):
        print("Failed to take off")
        return False
    time.sleep(7)  # Wait for drone to reach altitude
    
    # Hover
    print(f"\nStep 3: Hovering at {altitude}m for {hover_duration}s...")
    if not controller.hover(duration=hover_duration, altitude=altitude):
        print("Failed to hover")
        return False
    
    # Land
    print("\nStep 4: Landing...")
    if not controller.land():
        print("Failed to land")
        return False
    time.sleep(5)  # Wait for landing
    
    # Disarm
    print("\nStep 5: Disarming drone...")
    controller.arm(arm=False)
    
    print("\n=== Mission Complete ===\n")
    return True


def iris_mission_2m(controller, hover_duration=10):
    """
    Execute Iris copter mission: arm -> fly to 2m -> hover -> land
    
    This is a simplified mission specifically for 2 meter flight altitude.
    
    Args:
        controller: DroneController instance
        hover_duration (float): Duration to hover at 2m altitude in seconds
    
    Returns:
        bool: True if mission completed successfully
    """
    altitude = 2  # 2 meters as specified
    
    print("\n" + "="*50)
    print("  IRIS COPTER MISSION: ARM -> FLY 2M -> LAND")
    print("="*50 + "\n")
    
    try:
        # Step 1: Arm the drone
        print("[1/4] Arming Iris copter...")
        if not controller.arm(arm=True):
            print("ERROR: Failed to arm drone")
            return False
        time.sleep(2)
        print("✓ Drone armed successfully\n")
        
        # Step 2: Take off to 2 meters
        print(f"[2/4] Taking off to {altitude}m altitude...")
        if not controller.take_off(altitude=altitude):
            print("ERROR: Failed to take off")
            return False
        time.sleep(6)  # Wait for drone to reach 2m altitude
        print(f"✓ Reached {altitude}m altitude\n")
        
        # Step 3: Hover at 2 meters
        print(f"[3/4] Hovering at {altitude}m for {hover_duration}s...")
        if not controller.hover(duration=hover_duration, altitude=altitude):
            print("ERROR: Failed to hover")
            return False
        print(f"✓ Hover complete\n")
        
        # Step 4: Land
        print("[4/4] Landing...")
        if not controller.land():
            print("ERROR: Failed to initiate landing")
            return False
        time.sleep(6)  # Wait for landing to complete
        print("✓ Landed successfully\n")
        
        # Disarm after landing
        print("Disarming drone...")
        controller.arm(arm=False)
        time.sleep(1)
        
        print("="*50)
        print("  MISSION COMPLETED SUCCESSFULLY!")
        print("="*50 + "\n")
        return True
        
    except Exception as e:
        print(f"ERROR: Mission failed with exception: {e}")
        print("Attempting emergency disarm...")
        try:
            controller.arm(arm=False)
        except:
            pass
        return False

# ============================================================================
# MAIN SCRIPT - Create instance and run drone missions
# ============================================================================

if __name__ == "__main__":
    """
    Main script to instantiate DroneController and execute missions
    """
    
    print("\n" + "="*60)
    print("  IRIS DRONE CONTROLLER - MISSION LAUNCHER")
    print("="*60 + "\n")
    
    try:
        # Create DroneController instance
        print("Creating DroneController instance...")
        controller = DroneController(sitl_ip="127.0.0.1", sitl_port=5760)
        print("✓ DroneController instance created successfully\n")
        
        # Example 1: Simple arm/disarm
        print("--- Example 1: Arm/Disarm ---")
        print("Arming drone...")
        controller.arm(arm=True)
        time.sleep(2)
        print("Disarming drone...\n")
        controller.arm(arm=False)
        time.sleep(1)
        
        # Example 2: Basic full mission (arm -> takeoff -> hover -> land)
        print("--- Example 2: Full Mission (10m altitude) ---")
        full_mission(controller, altitude=10, hover_duration=5)
        
        # Example 3: 2-meter Iris mission
        print("--- Example 3: Iris 2-Meter Mission ---")
        iris_mission_2m(controller, hover_duration=5)
        
        # Disconnect
        print("Disconnecting from SITL...")
        controller.disconnect()
        print("✓ All operations complete!\n")
        
    except KeyboardInterrupt:
        print("\n\nMission interrupted by user")
        try:
            controller.arm(arm=False)
            controller.disconnect()
        except:
            pass
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Failed to execute mission")
        try:
            if 'controller' in locals():
                controller.arm(arm=False)
                controller.disconnect()
        except:
            pass