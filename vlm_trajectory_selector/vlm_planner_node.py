#!/usr/bin/env python3
# vlm_planner_node.py

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import json
import numpy as np

# メッセージ型
from autoware_auto_planning_msgs.msg import Trajectory as OutputTrajectory
from autoware_auto_planning_msgs.msg import TrajectoryPoint as OutputPoint
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import AccelWithCovarianceStamped
from builtin_interfaces.msg import Duration

# 画像処理
import cv2
from cv_bridge import CvBridge
from PIL import Image

# VLMPlannerのインポート
from vlm_planner import VLMPlanner
from prompt import initial_position


class VlmPlannerNode(Node):
    def __init__(self):
        super().__init__('vlm_planner_node')

        # パラメータ宣言
        self.declare_parameter('output_topic', '/output/trajectory')
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        # VLMPlannerのインスタンス化
        self.vlm_planner = VLMPlanner(self.get_logger())

        # QoSプロファイルの設定
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ROS 2 Publisher & Subscribers
        self.trajectory_publisher = self.create_publisher(OutputTrajectory, output_topic, reliable_qos)
        
        # 画像購読
        self.image_sub = self.create_subscription(
            RosImage, 
            '/sensing/camera/image_raw', 
            self.image_callback, 
            sensor_qos
        )
        
        # Odometry購読 (kinematic state)
        self.odometry_sub = self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.odometry_callback,
            reliable_qos
        )
        
        # Acceleration購読
        self.acceleration_sub = self.create_subscription(
            AccelWithCovarianceStamped,
            '/localization/acceleration',
            self.acceleration_callback,
            sensor_qos
        )

        # 状態変数
        self.bridge = CvBridge()
        self.latest_image = None
        self.current_odometry = None
        self.current_acceleration = None
        
        # VLM推論のレート制限
        self.last_trajectory_action = "S"
        self.last_known_sector = 1
        self.last_inference_time_sec = time.monotonic()
        self.inference_interval_sec = 5.0  # 推論を実行する間隔 (実時間で5秒)
        self.last_trajectory_msg = None

        self.get_logger().info(f"VLM Planner Node started. Publishing to '{output_topic}'.")

    def image_callback(self, msg: RosImage):
        """カメラ画像を受信し、最新の画像を保存し、trajectory生成または再利用を行う"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        now_sec = time.monotonic()
        if (now_sec - self.last_inference_time_sec) >= self.inference_interval_sec:
            self.get_logger().info("Generating new trajectory with VLM...")

            # 現在の車両状態情報を取得
            current_velocity = 0.0
            if self.current_odometry:
                current_velocity = self.current_odometry.twist.twist.linear.x
                current_position = (self.current_odometry.pose.pose.position.x, 
                                    self.current_odometry.pose.pose.position.y,
                                    self.current_odometry.pose.pose.position.z)

            # VLMでtrajectory生成
            trajectory_points, current_sector = self.vlm_planner.generate_trajectory(
                self.latest_image,
                self.last_trajectory_action,
                self.last_known_sector,
                current_velocity,
                current_position
            )

            if len(trajectory_points) > 0:
                # 成功した場合、結果と時刻を更新
                self.last_known_sector = current_sector
                self.last_inference_time_sec = now_sec
                trajectory_msg = self.create_trajectory_message(trajectory_points)
                self.last_trajectory_msg = trajectory_msg
                self.trajectory_publisher.publish(trajectory_msg)
                self.get_logger().info(f"Published trajectory with {len(trajectory_points)} points")
            else:
                self.get_logger().warn("Failed to generate trajectory. No trajectory published.")
        else:
            # 5秒経過していない場合は前回のtrajectoryを再利用
            if self.last_trajectory_msg is not None:
                self.trajectory_publisher.publish(self.last_trajectory_msg)
                self.get_logger().info("Reusing last trajectory (interval not passed)")
            else:
                self.get_logger().warn("No previous trajectory to reuse.")

    def odometry_callback(self, msg: Odometry):
        """Odometryデータを受信し、現在の位置・速度情報を保存する"""
        self.current_odometry = msg

    def acceleration_callback(self, msg: AccelWithCovarianceStamped):
        """加速度データを受信し、現在の加速度情報を保存する"""
        self.current_acceleration = msg

    def create_trajectory_message(self, trajectory_points: list) -> OutputTrajectory:
        """VLMからの軌道点リストをROS 2 Trajectoryメッセージに変換する"""
        output_msg = OutputTrajectory()
        output_msg.header.stamp = self.get_clock().now().to_msg()
        output_msg.header.frame_id = "base_link"

        for point_data in trajectory_points:
            output_point = OutputPoint()
            
            # 時間設定
            time_sec = float(point_data.get("time", 0.0))
            output_point.time_from_start = Duration()
            output_point.time_from_start.sec = int(time_sec)
            output_point.time_from_start.nanosec = int((time_sec - int(time_sec)) * 1e9)
            
            # 位置設定
            output_point.pose.position.x = float(point_data.get("x", 0.0))
            output_point.pose.position.y = float(point_data.get("y", 0.0))
            output_point.pose.position.z = float(point_data.get("z", 0.0))
            
            # 方向設定（簡単のため前方向を向いていると仮定）
            output_point.pose.orientation.x = 0.0
            output_point.pose.orientation.y = 0.0
            output_point.pose.orientation.z = 0.0
            output_point.pose.orientation.w = 1.0
            
            # 速度設定
            target_velocity = float(point_data.get("velocity", 5.0))  # デフォルト5m/s
            output_point.longitudinal_velocity_mps = target_velocity
            output_point.lateral_velocity_mps = 0.0
            output_point.acceleration_mps2 = 0.0
            output_point.heading_rate_rps = 0.0
            output_point.front_wheel_angle_rad = 0.0
            output_point.rear_wheel_angle_rad = 0.0

            output_msg.points.append(output_point)

        return output_msg




def main(args=None):
    rclpy.init(args=args)
    try:
        node = VlmPlannerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ValueError) as e:
        print(f"Node shutting down: {e}")
    finally:
        if 'node' in locals() and rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
