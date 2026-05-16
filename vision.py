import subprocess
import cv2
import numpy as np
from ultralytics import YOLO


YOUTUBE_URL = "https://www.youtube.com/watch?v=1EiC9bvVGnk"

VEHICLE_CLASSES = [2, 3, 5, 7] #detect only transport

STOP_LINE_Y = 620
TRAFFIC_LIGHT_ROI = (1660, 320, 1770, 400)

model = YOLO("yolo11n.pt")

def get_stream_url(url):
    result = subprocess.check_output([
        "python",
        "-m",
        "yt_dlp",
        "-g",
        "-f", "best[ext=mp4]/best",
        url
    ])

    return result.decode().strip().splitlines()[0]

stream_url = get_stream_url(YOUTUBE_URL)

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Failed to open stream")
    exit()


def detect_traffic_light(frame):

    x1, y1, x2, y2 = TRAFFIC_LIGHT_ROI

    roi = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # RED
    red1 = cv2.inRange(
        hsv,
        (0, 80, 80),
        (10, 255, 255)
    )

    red2 = cv2.inRange(
        hsv,
        (170, 80, 80),
        (180, 255, 255)
    )

    red_mask = red1 + red2

    # YELLOW
    yellow_mask = cv2.inRange(
        hsv,
        (15, 80, 80),
        (35, 255, 255)
    )

    # GREEN
    green_mask = cv2.inRange(
        hsv,
        (40, 60, 60),
        (90, 255, 255)
    )

    red_count = cv2.countNonZero(red_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    green_count = cv2.countNonZero(green_mask)

    counts = {
        "RED": red_count,
        "YELLOW": yellow_count,
        "GREEN": green_count
    }

    signal = max(counts, key=counts.get)

    if counts[signal] < 50:
        return "UNKNOWN"

    return signal

vehicle_history = {}
violators = set()


while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame read failed")
        break

    signal = detect_traffic_light(frame)

    RED_LIGHT = signal == "RED"


    rx1, ry1, rx2, ry2 = TRAFFIC_LIGHT_ROI

    cv2.rectangle(
        frame,
        (rx1, ry1),
        (rx2, ry2),
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"LIGHT: {signal}",
        (rx1, ry1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    line_color = (0, 0, 255) if RED_LIGHT else (0, 255, 0)

    cv2.line(
        frame,
        (0, STOP_LINE_Y),
        (frame.shape[1], STOP_LINE_Y),
        line_color,
        3
    )


    results = model.track(
        frame,
        persist=True,
        classes=VEHICLE_CLASSES,
        conf=0.35,
        verbose=False
    )

    result = results[0]

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            vehicle_id = int(track_id)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if vehicle_id not in vehicle_history:
                vehicle_history[vehicle_id] = []

            vehicle_history[vehicle_id].append((cx, cy))

            vehicle_history[vehicle_id] = vehicle_history[vehicle_id][-10:]

            points = vehicle_history[vehicle_id]

            if len(points) >= 2:

                prev_y = points[-2][1]
                curr_y = points[-1][1]

                crossed_line = (
                    prev_y < STOP_LINE_Y <= curr_y
                )

                if crossed_line and RED_LIGHT:

                    if vehicle_id not in violators:

                        violators.add(vehicle_id)

                        print(
                            f"RED LIGHT VIOLATION: Vehicle {vehicle_id}"
                        )

                        filename = (
                            f"violation_{vehicle_id}.jpg"
                        )

                        cv2.imwrite(filename, frame)

            color = (
                (0, 0, 255)
                if vehicle_id in violators
                else (0, 255, 0)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                5,
                color,
                -1
            )

            label = f"ID {vehicle_id}"

            if vehicle_id in violators:
                label += " VIOLATION"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow(
        "Parkapumi",
        frame
    )

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()