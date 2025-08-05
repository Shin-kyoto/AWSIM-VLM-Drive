
# vlm_planner.py

import google.generativeai as genai
import os
import time
import json

import math
from PIL import Image
import typing_extensions as typing
from prompt import create_trajectory_prompt, initial_position

class TrajectoryPoint3D(typing.TypedDict):
    """Geminiに渡すTrajectoryPointの3D座標情報"""
    x: float
    y: float
    z: float
    time: float
    velocity: float

class TrajectoryResponse(typing.TypedDict):
    """Geminiに渡すレスポンススキーマ"""
    current_sector: int
    trajectory_points: list[TrajectoryPoint3D]
    reasoning: str
    command: str

class VLMPlanner:
    """
    画像認識とVLMによるtrajectory生成を行うクラス
    """
    def __init__(self, logger):
        self.logger = logger
        self.model = None
        self.last_commands = []  # VLMが推論したコマンド履歴
        self._setup_gemini()

    def _setup_gemini(self):
        """
        Geminiモデルをセットアップします。
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.logger.error("環境変数 'GEMINI_API_KEY' が設定されていません。")
            raise ValueError("APIキーがありません。")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        self.logger.info("VLMPlanner: Gemini model initialized successfully.")

    def _preprocess_image(self, image: Image) -> Image:
        """
        画像をクロップ、リサイズします。
        """
        pil_image = image.copy()
        # 認識範囲を調整
        w, h = pil_image.size
        crop_box = (0, h // 2 - 200, w, h // 2 + 200)
        pil_image = pil_image.crop(crop_box)
        pil_image = pil_image.resize((w // 4, (crop_box[3] - crop_box[1]) // 4))
        return pil_image

    def generate_trajectory(self, image: Image, last_trajectory_action: str, last_sector: int, current_velocity: float, current_position: tuple) -> tuple[list, int]:
        """
        与えられた画像を元に、VLMでtrajectoryを生成します。
        """
        if self.model is None:
            self.logger.warn("VLM model is not ready.")
            return [], last_sector

        processed_image = self._preprocess_image(image)
        
        prompt = create_trajectory_prompt(last_trajectory_action, last_sector, current_velocity, current_position, self.last_commands)
        
        try:
            start = time.perf_counter()
            response = self.model.generate_content(
                [prompt, processed_image],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=TrajectoryResponse
                ),
            )
            end = time.perf_counter()
            latency = end - start
            self.logger.info(f"Gemini trajectory generation latency: {latency:.4f}s")
            self.logger.info(f"Response: {response.text.strip()}")
            response_dict = json.loads(response.text.strip())
            trajectory_points = response_dict.get("trajectory_points", [])
            current_sector = response_dict.get("current_sector", last_sector)
            reasoning = response_dict.get("reasoning", "")
            command = response_dict.get("command", "go straight")
            self.last_commands.append(command)

            self.logger.info(f"Generated {len(trajectory_points)} points. sector: {current_sector}")
            self.logger.info(f"Reasoning: {reasoning}")
            self.logger.info(f"Command: {command}")

            return trajectory_points, current_sector
                
        except Exception as e:
            self.logger.error(f"Error during VLM trajectory generation: {e}")
            return [], last_sector
