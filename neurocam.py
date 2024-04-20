import os
import cv2
import threading
import requests
import base64
import time
import json
import queue
from enum import Enum, auto
from dotenv import load_dotenv
from picamera2 import Picamera2

from akida import Model as AkidaModel, devices
from akida_models.detection.processing import (
    decode_output,
)
import numpy as np

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
if OPENAI_API_KEY is None:
    print("Please set the OPENAI_API_KEY environment variable.")
else:
    print(f"Loaded API Key")

CAMERA_SRC = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

INFERENCE_PER_SECOND = 2
PREDICTION_CONFIDENCE_MIN = 0.6
VWW_THRESHOLD = 0.5

TARGET_WIDTH = 224
TARGET_HEIGHT = 224

YOLO = "yolo"
VWW = "vww"
YOLO_FBZ = "models/yolo.fbz"
VWW_FBZ = "models/vww.fbz"
NUM_ANCHORS = 5
GRID_SIZE = (7, 7)
NUM_CLASSES = 2

CAR = "car"
PERSON = "person"
LABELS = [CAR, PERSON]
COLOURS = {
    CAR: (255, 0, 0),
    PERSON: (255, 255, 255),
}
ANCHORS = [
    [0.56658, 1.05302],
    [1.09512, 2.04102],
    [2.39016, 3.01487],
    [2.46033, 4.92333],
    [5.17334, 5.56817],
]

SYSTEM_PROMPT = """
You are an advanced Raspberry Pi-based security system, equipped with neuromorphic hardware 
that utilizes a YoloV2 neural network to identify cars and people. As a vigilant security guard 
monitoring the feed, your main tasks involve analyzing the behavior and physical details of 
any detected person with a forensic level of detail. Focus on capturing and reporting every 
critical element to ensure thorough and precise surveillance.
"""

IMAGE_PROMPT = """
Analyze the still image from the security feed and write a detailed security report about the frame: 
Describe the person's actions and provide a detailed account of their physical appearance. Focus on 
identifying any unusual or suspicious behaviors, attire that may be inappropriate for the setting or 
season, distinguishing features such as tattoos or scars, and any objects they might be carrying that 
could be of security concern. Evaluate how these observations could relate to security protocols.
"""


class SecurityLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

    def __str__(self):
        return self.name


class SecurityReport:
    def __init__(self, report):
        """
        Initializes the SecurityReport class with a dictionary containing the report details.

        Parameters:
        - report (dict): A dictionary with keys 'overview', 'persons', and 'security_level'.
        """
        self.report = report
        self.colors = {
            "LOW": "\033[32m",  # Green
            "MEDIUM": "\033[33m",  # Yellow
            "HIGH": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
            "RESET": "\033[0m",  # Reset to default
        }

    def print_colored(self, text, color):
        """
        Prints text in the specified ANSI color.
        """
        print(
            f"{self.colors.get(color, self.colors['RESET'])}{text}{self.colors['RESET']}"
        )

    def display(self):
        """
        Prints the complete security report with appropriate color coding for the security level.
        """
        print("\n\n")
        print("Security Report Overview:")
        print(self.report["overview"])
        print("\n")
        print("Persons Detected:")
        print(self.report["persons"])
        print("\n")
        # Get the security level in uppercase to match dictionary keys
        security_level = self.report["security_level"].upper()
        self.print_colored(f"Security Level: {security_level}", security_level)
        print("\n\n")


class Vision:
    """
    A class to handle the communication with OpenAI's API for analyzing images.

    Attributes:
        api_key (str): The API key for accessing OpenAI's API.
        headers (dict): Headers to use in the API request.
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def encode_image_to_base64(self, image_data):
        """Encode the image to base64 string after converting it to JPEG format."""
        success, buffer = cv2.imencode(".jpg", image_data)
        if not success:
            raise ValueError("Could not encode image to JPEG")
        return base64.b64encode(buffer).decode("utf-8")

    def construct_payload(self, base64_image):
        """Construct the payload to be sent to OpenAI's API based on the image."""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_detailed_security_report",
                    "description": "Write a detailed security report about a frame security camera feed",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "overview": {
                                "type": "string",
                                "description": "A comprehensive description of the scene depicted in the frame, including environmental context, noticeable activities, and potential security concerns.",
                            },
                            "persons": {
                                "type": "string",
                                "description": "A detailed narrative about each person observed in the frame, including their appearance, behavior, and other relevant observations.",
                            },
                            "security_level": {
                                "type": "string",
                                "description": "Specifies the severity level of a security concern, categorized into LOW, MEDIUM, HIGH, and CRITICAL. Each level indicates the urgency and potential impact of the issue.",
                                "enum": [level.name for level in SecurityLevel],
                            },
                        },
                        "required": ["overview", "persons", "security_level"],
                    },
                },
            }
        ]

        return {
            "model": "gpt-4-turbo",
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": "write_detailed_security_report"},
            },
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": IMAGE_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 300,
        }

    def ask_openai(self, image_data):
        """
        Send a request to OpenAI's API to analyze the image.

        Args:
            image_data (array): The image data to analyze.

        Returns:
            dict: The response from the API.
        """
        base64_image = self.encode_image_to_base64(image_data)
        payload = self.construct_payload(base64_image)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload,
        )
        if response.status_code == 200:
            response_message = response.json()["choices"][0]["message"]
            response_content = response_message["content"]
            tool_calls = response_message.get("tool_calls")
            if tool_calls:
                function_name = tool_calls[0]["function"]["name"]
                function_args = json.loads(tool_calls[0]["function"]["arguments"])

                return {
                    "function": function_name,
                    "args": function_args,
                }

        else:
            error_message = (
                response.json().get("error", {}).get("message", "Unknown error")
            )
            return None


class Camera:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=10)
        self.camera = Picamera2()
        self.stream_config = self.camera.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        self.camera.configure(self.stream_config)
        self.camera.start()
        self.running = True

        self.label = 0
        self.pred_boxes = []

        self.terminator_vision = False

        threading.Thread(target=self.capture_frames).start()
        threading.Thread(target=self.show_window).start()

    def capture_frames(self):
        while self.running:
            frame = self.camera.capture_array()
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def get_frame(self):
        return self.frame_queue.get()

    def get_input_array(self, target_width, target_height):
        frame = self.frame_queue.get()
        if frame is not None:
            processed_frame = self.process_frame(frame, target_width, target_height)
            return processed_frame

    def show_window(self):
        while self.running:
            frame = self.render_boxes(self.get_frame())

            if frame is not None:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False

    def process_frame(self, frame, target_width, target_height):
        if frame is not None:
            resized_array = cv2.resize(frame, (target_width, target_height))
            expanded_array = np.expand_dims(resized_array, axis=0)
            int8_array = expanded_array.astype("uint8")
            return int8_array

    def set_pred_boxes(self, pred_boxes):
        self.pred_boxes = pred_boxes

    @staticmethod
    def apply_terminator_vision(input_array):
        """
        Applies a Terminator-style vision effect to an input image array.

        Args:
        input_array (numpy.ndarray): The input image array in BGR format.

        Returns:
        numpy.ndarray: The image with the Terminator vision effect applied.
        """
        if input_array is None:
            print("Error: No input image provided.")
            return None

        # Ensure the input is in float format for manipulation
        image = input_array.astype(float)

        # Reduce the contributions of the blue and green channels to emphasize red
        image[:, :, 0] *= 0.2  # Reduce blue channel
        image[:, :, 1] *= 0.2  # Reduce green channel

        # Clip the values to be in the valid range [0, 255] and convert back to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Optionally apply a color map for additional styling (e.g., heat map)
        terminator_vision_image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)

        return terminator_vision_image

    def render_boxes(self, frame):
        """
        Draw bounding boxes around detected objects in the frame.

        Args:
            frame (array): The frame to draw bounding boxes on.

        Returns:
            array: The frame with bounding boxes drawn.
        """

        if self.terminator_vision:
            frame = self.apply_terminator_vision(frame)

        for box in self.pred_boxes:
            if box[5] > PREDICTION_CONFIDENCE_MIN:
                x1, y1 = int(box[0]), int(box[1])
                x2, y2 = int(box[2]), int(box[3])
                label = LABELS[int(box[4])]
                score = "{:.2%}".format(box[5])
                colour = COLOURS[label]
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
                cv2.putText(
                    frame,
                    "{} - {}".format(label, score),
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.4,
                    colour,
                    1,
                    cv2.LINE_AA,
                )

        return frame


class YOLOInference:
    def __init__(self, camera, anchors, show_stats=False):
        self.camera = camera
        self.anchors = anchors
        self.show_stats = show_stats
        self.model = AkidaModel(filename=YOLO_FBZ)
        self.pred_boxes = []
        self.active = False

        # start the thread
        self.thread = threading.Thread(target=self.infer)
        self.thread.start()

    def set_active(self):
        print("YOLO Active")
        self.active = True
        self.camera.terminator_vision = True
        self.map_hardware()

    def set_inactive(self):
        self.active = False
        self.camera.terminator_vision = False
        self.pred_boxes = []

    def map_hardware(self):
        if len(devices()) > 0:
            device = devices()[0]
            if self.show_stats:
                device.soc.power_measurement_enabled = True
            self.model.map(device, hw_only=True)

    def infer(self):
        while True:
            if self.active:
                input_array = self.camera.get_input_array(224, 224)
                if self.model:
                    self.yolo_infer(input_array)
                if self.show_stats:
                    print(self.model.statistics)

    def yolo_infer(self, input_array):
        pots = self.model.predict(input_array)[0]
        w, h, c = pots.shape
        pots = pots.reshape((h, w, len(self.anchors), 4 + 1 + len(LABELS)))
        raw_boxes = decode_output(pots, self.anchors, len(LABELS))
        pred_boxes = [
            [
                box.x1 * FRAME_WIDTH,
                box.y1 * FRAME_HEIGHT,
                box.x2 * FRAME_WIDTH,
                box.y2 * FRAME_HEIGHT,
                box.get_label(),
                box.get_score(),
            ]
            for box in raw_boxes
        ]
        self.pred_boxes = pred_boxes
        self.camera.set_pred_boxes(pred_boxes)


class VWWInference:
    def __init__(self, camera, show_stats=False):
        self.camera = camera
        self.show_stats = show_stats
        self.model = AkidaModel(filename=VWW_FBZ)
        self.person_detected = False
        self.person_detected_confidence = 0
        self.active = False

        # start the thread
        self.thread = threading.Thread(target=self.infer)
        self.thread.start()

    def set_active(self):
        print("VWW Active")
        self.active = True
        self.camera.set_pred_boxes([])
        self.person_detected = False
        self.map_hardware()

    def set_inactive(self):
        self.active = False

    def map_hardware(self):
        if len(devices()) > 0:
            device = devices()[0]
            if self.show_stats:
                device.soc.power_measurement_enabled = True
            self.model.map(device, hw_only=True)

    def infer(self):
        while True:
            if self.active:
                input_array = self.camera.get_input_array(96, 96)
                if self.model:
                    self.vww_infer(input_array)
                if self.show_stats:
                    print(self.model.statistics)

    def vww_infer(self, input_array):
        prediction = self.model.predict(input_array).reshape(-1)
        probabilities = self.softmax(prediction)
        self.person_detected = probabilities[1] > VWW_THRESHOLD
        self.person_detected_confidence = probabilities[1]

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


class Sentry:
    def __init__(self):
        self.camera = Camera()
        self.vision = Vision(OPENAI_API_KEY)
        self.vww_inference = VWWInference(self.camera)
        self.yolo_inference = YOLOInference(self.camera, ANCHORS)
        self.security_level = SecurityLevel.LOW
        self.timer = None
        self.report_timer = None
        self.running = True

        self.vww_inference.set_active()

        threading.Thread(target=self.monitor, daemon=True).start()

    def monitor(self):
        while self.running:
            if self.vww_inference.person_detected:
                # Disable VWW and enable YOLO when a person is detected
                if not self.yolo_inference.active:
                    self.vww_inference.set_inactive()
                    self.yolo_inference.set_active()
                    self.set_security_level(SecurityLevel.HIGH)
                    self.start_security_report_timer()

                    self.timer = threading.Timer(30, self.reset_security_level)
                    self.timer.start()

            time.sleep(1)

    def start_security_report_timer(self):
        # Cancel existing report timer if running
        if self.report_timer:
            self.report_timer.cancel()

        # Start a new report timer that executes every 5 seconds
        self.report_timer = threading.Timer(5, self.generate_security_report)
        self.report_timer.start()

    def generate_security_report(self):
        self.get_security_report()

    def get_security_report(self):
        report = self.vision.ask_openai(self.camera.get_frame())
        if report:
            security_report = SecurityReport(report["args"])
            security_report.display()

    def set_security_level(self, level):
        if self.security_level != level:
            self.security_level = level
            print(f"Security Level set to {level}")

    def reset_security_level(self):
        # Reset the security level and swap to VWW detection
        self.vww_inference.person_detected = False
        print(f"Security Level reset to {SecurityLevel.LOW}")
        self.set_security_level(SecurityLevel.LOW)
        self.yolo_inference.set_inactive()
        self.yolo_inference.pred_boxes = []
        self.vww_inference.set_active()
        if self.report_timer:
            self.report_timer.cancel()

    def stop(self):
        self.running = False
        if self.timer:
            self.timer.cancel()
        if self.report_timer:
            self.report_timer.cancel()


def main():
    try:
        # Initialize the Sentry object
        sentry = Sentry()
        print("Sentry system is running.")

        # The system will keep running in the background, you can add more logic here if needed
        # For example, running some tests or adding user interaction
        time.sleep(
            600
        )  # Keep the main thread alive for 10 minutes or until Ctrl+C is pressed

    except Exception as e:
        print(f"Failed to start Sentry system: {e}")

    finally:
        # Ensuring that the system stops properly when the main execution is interrupted
        if "sentry" in locals():  # Check if 'sentry' was successfully defined
            sentry.stop()
            print("Sentry system has been stopped.")


if __name__ == "__main__":
    main()
