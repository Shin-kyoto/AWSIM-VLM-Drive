#!/usr/bin/env python3
import os
import google.generativeai as genai
import time
import io
import base64
from PIL import Image
from pynput.keyboard import Controller
import pyautogui
import numpy as np
import typing_extensions as typing
import subprocess
import json
import math

# --- 関数定義 ---

def get_window_geometry(window_title):
    """
    指定されたウィンドウタイトルのウィンドウジオメトリ（位置とサイズ）を取得します。
    """
    try:
        # `wmctrl`でウィンドウリストを取得
        wmctrl_output = subprocess.check_output(['wmctrl', '-l']).decode('utf-8')
        target_line = None

        for line in wmctrl_output.splitlines():
            if len(line.split()) > 3 and line.split(None, 3)[-1] == window_title:
                target_line = line
                break

        if not target_line:
            raise ValueError(f"Window with title '{window_title}' not found.")

        # ウィンドウIDを取得
        window_id = target_line.split()[0]
        
        # `xwininfo`でウィンドウの座標を取得
        xwininfo_output = subprocess.check_output(['xwininfo', '-id', window_id]).decode('utf-8')
        coords = {}
        for line in xwininfo_output.splitlines():
            if "Absolute upper-left X" in line:
                coords['x'] = int(line.split(':')[-1].strip())
            elif "Absolute upper-left Y" in line:
                coords['y'] = int(line.split(':')[-1].strip())
            elif "Width" in line:
                coords['width'] = int(line.split(':')[-1].strip())
            elif "Height" in line:
                coords['height'] = int(line.split(':')[-1].strip())

        if not all(k in coords for k in ['x', 'y', 'width', 'height']):
            raise ValueError("Failed to get all window coordinates.")

        return coords

    except Exception as e:
        raise RuntimeError(f"Failed to get window geometry: {e}")

def capture_window(window_title, output_path="screenshot.png", use_imagemagick=False):
    """
    指定されたウィンドウをキャプチャします。
    """
    try:
        # ウィンドウの位置とサイズを取得
        geometry = get_window_geometry(window_title)

        if use_imagemagick:
            # ImageMagick (`import` コマンド) を使用してスクリーンショットを取得
            window_id = subprocess.check_output(
                ['wmctrl', '-l'], text=True
            ).splitlines()
            # This logic for getting window_id for imagemagick seems complex,
            # sticking to the pyautogui implementation which is more direct.
            # A more robust way would be needed if imagemagick is a hard requirement.
            # For now, we assume use_imagemagick=False.
            pass
        else:
            # PyAutoGUIでスクリーンショットを取得
            screenshot = pyautogui.screenshot(region=(
                geometry['x'],
                geometry['y'],
                geometry['width'],
                geometry['height']
            ))
            screenshot.save(output_path)
            return screenshot

    except Exception as e:
        raise RuntimeError(f"Failed to capture window: {e}")

def preprocess_image(image):
    """
    画像をクロップ、リサイズし、Geminiに送信できる形式に変換します。
    """
    pil_image = image.copy()
    pil_image = pil_image.crop((0, pil_image.height // 2 - 200, pil_image.width, pil_image.height // 2 + 200))
    pil_image = pil_image.resize((pil_image.width // 4, pil_image.height // 4))
    
    # display(pil_image)  # .pyスクリプトではdisplayは使えないためコメントアウト
    
    # Base64エンコード
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    image_str = base64.b64encode(buffered.getvalue()).decode()
    
    # PIL Imageオブジェクトも返すように変更（send_to_geminiがPIL Imageを想定しているため）
    return pil_image, image_str

def send_to_gemini(img, model):
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
    response = model.generate_content(
        [prompt, img],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            # `ControlCommand`のスキーマを直接渡す
            response_schema=ControlCommand
        ),
    )
    end = time.perf_counter()
    latency = end - start
    print(f"Latency: {latency}")
    return response, latency

def execute_action(response, keyboard, counters):
    """
    Geminiからのレスポンスに基づき、キーボード操作を実行します。
    """
    try:
        probability = math.exp(response.to_dict()["candidates"][0]["avg_logprobs"])
        response_dict = json.loads(response.text.strip())
        command = response_dict["command"]
        reason = response_dict["reason"]
        print(f"Executing Command: {command}, Reason: {reason}, Probability: {probability}")
        print()
        
        base_duration = 0.1
        increase_duration = 0.2
        thresh_duration = 1.0
        duration = base_duration
        
        if command == "→":
            counters['right'] += 1
            duration = base_duration + (counters['right'] * increase_duration)
            print(f"Right counter: {counters['right']}, Increased duration: {duration}")
            counters['left'] = 0 # 反対方向のカウンターはリセット
        elif command == "←":
            counters['left'] += 1
            duration = base_duration + (counters['left'] * increase_duration)
            print(f"Left counter: {counters['left']}, Increased duration: {duration}")
            counters['right'] = 0 # 反対方向のカウンターはリセット
        else:
            counters['right'] = 0
            counters['left'] = 0
            duration = base_duration
        
        # キー操作
        key_map = {"←": 'a', "→": 'd', "↑": 'w', "↓": 's'}
        if command not in key_map:
            print("No valid command received or command is none.")
            return

        key_to_press = key_map[command]
        
        if duration > thresh_duration and command in ["←", "→"]:
            keyboard.press(key_to_press)
        else:
            if command in ["↑", "↓"]:
                keyboard.press(key_to_press) # 加速・減速は押しっぱなし
            else:
                keyboard.press(key_to_press)
                time.sleep(duration)
                keyboard.release(key_to_press)

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Received text: {response.text}")
    except Exception as e:
        print(f"An error occurred in execute_action: {e}")


# --- 型定義 ---
class ControlCommand(typing.TypedDict):
    command: str
    reason: str


# --- メイン実行ブロック ---
def main():
    """
    メインの実行ループ
    """
    # APIキーの設定 (環境変数から取得することを推奨)
    api_key = os.environ['GEMINI_API_KEY'] # ここにあなたのAPIキーを設定してください
    if api_key == "YOUR_API_KEY":
        print("警告: 'YOUR_API_KEY' を実際のAPIキーに置き換えてください。")
        return
    genai.configure(api_key=api_key)

    # モデルの初期化
    model = genai.GenerativeModel("gemini-1.5-flash")

    # キーボードコントローラー
    keyboard = Controller()

    # カウンターの初期化
    counters = {'right': 0, 'left': 0}
    
    print("VLM-based driving script started. Press Ctrl+C to exit.")

    while True:
        try:
            # 1. シミュレータの画面をキャプチャ
            frame = capture_window("AWSIM", use_imagemagick=False)
            
            # 2. キャプチャ画像を前処理
            img_pil, _ = preprocess_image(frame)
            
            # 3. 画像をGeminiに送信して制御コマンドを取得
            action_response, latency = send_to_gemini(img_pil, model)
            
            # 4. コマンドに基づきシミュレータを操作
            execute_action(action_response, keyboard, counters)
            
            # 5. 実行レートを調整 (約4秒に1回)
            sleep_time = 4.0 - latency
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        except RuntimeError as e:
            print(f"A runtime error occurred: {e}")
            print("Is the 'AWSIM' window open and visible?")
            time.sleep(5) # 5秒待ってリトライ
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            break

if __name__ == "__main__":
    main()
