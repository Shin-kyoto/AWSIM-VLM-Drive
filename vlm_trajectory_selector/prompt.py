# prompt.py

initial_position = (89633.29, 43127.57, 0.8778)

def calculate_distance(point1: tuple, point2: tuple) -> float:
    """Calculate the Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def create_trajectory_prompt(last_trajectory_action: str, last_sector: int, current_velocity: float, current_position: tuple, last_commands: list[str]) -> str:
    track_knowledge = f"""Track sectors:
    1: 30m Starting Straight (speed limit 5 m/s). At the end of the straight in sector 1, there is the white big sign board. Before you pass 28m, you can go straight. After you pass 28m, the blue and white stripe wall became closer, and please slow down with big braking for sector 2. distance from braking point {calculate_distance(current_position, (89616, 43156, 0))} m
    2: R-Hairpin. distance from current position to sector 2 entry point: {calculate_distance(current_position, (89613.9, 43157.7, 0))}m. distance from current position to sector 2 middle point: {calculate_distance(current_position, (89617, 43166, 0))}m. (exapmple: ("time": 1.0,"velocity": 0.1,"x": 0.1,"y": -0.3,"z": 0.0) ("time": 2.0,"velocity": 0.1,"x": 0.12,"y": -0.4,"z": 0.0))
    3: 30m Short Straight
    4: L-Hairpin
    5: 30m Short Straight
    6: U-shaped Right
    7: 90-degree Left
    8: R-Hairpin
    9: L-Hairpin
    10: S-Curve (L->R)
    11: 90-degree Right
    12: S-Curve (L->R)
    13: Sweeping Right Corner (leads to main straight)"""

    # コマンド履歴（最大20個）
    last_commands_str = ', '.join(last_commands[-20:]) if last_commands else 'None'

    prompt = f"""
    Assume I am at the coordinate (0,0) facing forward along the x-axis.
    Based on the given image of a race track, choose the best trajectory to drive the car.
    The road is the dark gray area. Stay on the road. The wall has a blue and white pattern.
    
    Command options (output one of these for the next action):
    - turn right
    - turn left
    - go straight

    History:
    - Last Trajectory: {last_trajectory_action}
    - Last sector: {last_sector}
    - Current Velocity: {current_velocity:.2f} m/s
    - Current Position: distance from initial position: {calculate_distance(current_position, initial_position):.2f} m
    - Last 20 commands: {last_commands_str}
    {track_knowledge}
    Your task: Determine the current sector and generate a trajectory and command.
    What is my future trajectory in next 5 seconds under vehicle coordinate?

    Output format:
    {{
      "command": "turn right|turn left|go straight",
      "trajectory_points": [
        {{"x": float, "y": float, "z": float, "time": float, "velocity": float}},
        ... (10 points)
      ]
    }}

    Generate 10 trajectory points with:
    - x, y, z coordinates (vehicle coordinate system, x: forward, y: left, z: up)
    - time: time from now (0.0 to 5.0 seconds)
    - velocity: target velocity in m/s

    Ensure smooth trajectory that follows the road and maintains safe driving.
    Don't forget that the sector 2 is a right hairpin. Before that, you should slow down.
    Don't forget that if you want to turn right, y shoud be negative.
    What is my future trajectory in next 5 seconds under vehicle coordinate?
    """
    return prompt
