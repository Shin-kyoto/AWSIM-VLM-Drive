#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import google.generativeai as genai
import time
import io
import base64
from PIL import Image
from pynput.keyboard import Controller
import numpy as np
import typing_extensions as typing
import json
import math
# ROS-CV連携のためのライブラリ
import cv2
from cv_bridge import CvBridge
# ROSの画像メッセージ型
from sensor_msgs.msg import Image as RosImage

# --- 型定義 ---
class ControlCommand(typing.TypedDict):
    command: str
    reason: str

class VlmDriverNode(Node):
    """
    ROSの画像トピックを購読し、VLMで判断してキーボード操作を行うノード
    """
    def __init__(self):
        super().__init__('vlm_driver_node')

        # --- Geminiとキーボードの初期化 ---
        self.setup_gemini()
        self.keyboard = Controller()
        self.counters = {'right': 0, 'left': 0}
        
        # --- ROS 2関連の初期化 ---
        self.bridge = CvBridge()
        
        # ROSトピック情報に基づいたQoSプロファイルの設定
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # 画像トピックのサブスクライバーを作成
        self.subscription = self.create_subscription(
            RosImage,
            '/sensing/camera/image_raw',
            self.image_callback,
            qos_profile)
        
        # 処理中の重複実行を防ぐためのフラグ
        self.is_processing = False
        
        self.get_logger().info("VLM Driver Node has been started. Waiting for images...")

    def setup_gemini(self):
        """
        Geminiモデルをセットアップします。
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.get_logger().error("環境変数 'GEMINI_API_KEY' が設定されていません。")
            raise ValueError("APIキーがありません。")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.get_logger().info("Gemini model initialized successfully.")

    def image_callback(self, msg: RosImage):
        """
        画像トピックを受信するたびに呼び出されるコールバック関数。
        """
        if self.is_processing:
            # 前の処理が完了するまで新しい画像は無視する
            return
            
        self.is_processing = True
        try:
            self.get_logger().info("Received an image, starting processing...")
            
            # 1. ROS ImageメッセージをPIL Imageに変換
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # 2. キャプチャ画像を前処理
            processed_pil_image = self.preprocess_image(pil_image)
            
            # 3. 画像をGeminiに送信して制御コマンドを取得
            action_response, latency = self.send_to_gemini(processed_pil_image)
            
            # 4. コマンドに基づきシミュレータを操作
            self.execute_action(action_response)

        except Exception as e:
            self.get_logger().error(f"An error occurred in image_callback: {e}")
        finally:
            # 処理が完了したらフラグをリセット
            self.is_processing = False

    def preprocess_image(self, image: Image) -> Image:
        """
        画像をクロップ、リサイズします。
        """
        pil_image = image.copy()
        pil_image = pil_image.crop((0, pil_image.height // 2 - 200, pil_image.width, pil_image.height // 2 + 200))
        pil_image = pil_image.resize((pil_image.width // 4, pil_image.height // 4))
        return pil_image

    def send_to_gemini(self, img: Image):
        """
        画像とプロンプトをGeminiに送信し、制御コマンドを取得します。
        """
        prompt = f"""
        Based on the given image, choose an action to drive the cart so that it does not deviate from the course.
        The choices are as follows:
        - "↑": Accelerate
        - "↓": Decelerate
        - "←": Steer left
        - "→": Steer right
        """
        start = time.perf_counter()
        response = self.model.generate_content(
            [prompt, img],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ControlCommand
            ),
        )
        end = time.perf_counter()
        latency = end - start
        self.get_logger().info(f"Gemini latency: {latency:.4f}s")
        return response, latency

    def execute_action(self, response):
        """
        Geminiからのレスポンスに基づき、キーボード操作を実行します。
        """
        try:
            probability = math.exp(response.to_dict()["candidates"][0]["avg_logprobs"])
            response_dict = json.loads(response.text.strip())
            command = response_dict["command"]
            reason = response_dict["reason"]
            self.get_logger().info(f"Executing Command: {command}, Reason: {reason}, Probability: {probability:.4f}")

            key_map = {"←": 'a', "→": 'd', "↑": 'w', "↓": 's'}
            if command not in key_map:
                self.get_logger().warn("No valid command received or command is none.")
                return

            # カウンターとキー操作のロジック (元コードから流用)
            base_duration = 0.1
            increase_duration = 0.2
            thresh_duration = 1.0
            duration = base_duration
            
            if command == "→":
                self.counters['right'] += 1
                duration = base_duration + (self.counters['right'] * increase_duration)
                self.counters['left'] = 0
            elif command == "←":
                self.counters['left'] += 1
                duration = base_duration + (self.counters['left'] * increase_duration)
                self.counters['right'] = 0
            else:
                self.counters['right'] = 0
                self.counters['left'] = 0

            key_to_press = key_map[command]
            if duration > thresh_duration and command in ["←", "→"]:
                self.keyboard.press(key_to_press)
            else:
                if command in ["↑", "↓"]:
                    self.keyboard.press(key_to_press)
                else:
                    self.keyboard.press(key_to_press)
                    time.sleep(duration)
                    self.keyboard.release(key_to_press)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            self.get_logger().error(f"Error parsing Gemini response: {e}\nReceived text: {response.text}")
        except Exception as e:
            self.get_logger().error(f"An error occurred in execute_action: {e}")

def main(args=None):
    rclpy.init(args=args)
    try:
        node = VlmDriverNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ValueError) as e:
        print(f"Node shutting down: {e}")
    finally:
        if 'node' in locals() and rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
