import cv2 as cv
import mediapipe as mp
import numpy as np
import serial
import time
import math
import csv
import os
import wave
import struct
import threading
from datetime import datetime


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ============================================================
# SERIAL COMMUNICATION
# ============================================================

SERIAL_PORT = "COM7"
BAUD_RATE = 115200

try:

    ser = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=1
    )

    time.sleep(2)

    print(f"Serial connected: {SERIAL_PORT}")

except Exception as e:

    print("Serial connection failed:")
    print(e)

    ser = None


# ============================================================
# OPENCV DISPLAY COLORS (BGR)
# ============================================================

BLACK   = (0, 0, 0)
WHITE   = (255, 255, 255)
GREEN   = (0, 255, 0)
YELLOW  = (0, 255, 255)
ORANGE  = (0, 165, 255)
RED     = (0, 0, 255)
CYAN    = (255, 255, 0)
MAGENTA = (255, 0, 255)


# ============================================================
# EYE LANDMARKS
# ============================================================

LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]


# ============================================================
# MOUTH LANDMARKS
# ============================================================

UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]


# ============================================================
# IRIS LANDMARKS
# ============================================================

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


# ============================================================
# FACE / EYE / IRIS LISTS
# ============================================================

FACE_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

LEFT_EYE = [
    362, 385, 387, 263, 373, 380
]

RIGHT_EYE = [
    33, 160, 158, 133, 153, 144
]

LEFT_IRIS_POINTS = LEFT_IRIS
RIGHT_IRIS_POINTS = RIGHT_IRIS


# ============================================================
# EYE SETTINGS
# ============================================================

EYE_CLOSED_RATIO_THRESHOLD = 6.0

# Existing drowsiness threshold
EYE_CLOSED_DURATION_SEC = 1.5

STATE_GRACE_SEC = 0.3

# Emergency stop if eyes continuously closed
EYE_EMERGENCY_STOP_SEC = 5.0


# ============================================================
# DRIVER NOT DETECTED SETTINGS
# ============================================================

# Emergency stop if driver is not detected continuously
# for 5 seconds or more.
NO_DRIVER_STOP_SEC = 5.0


# ============================================================
# YAWN SETTINGS
# ============================================================

YAWN_RATIO_THRESHOLD = 1.8

# Mouth must remain open continuously
# before declaring a yawn.
YAWN_DURATION_SEC = 1.5

# Prevent repeated yawn detection
YAWN_COOLDOWN_SEC = 3.0


# ============================================================
# DISTRACTION SETTINGS
# ============================================================

DISTRACTION_DELAY_SEC = 4.0

warning_delay = DISTRACTION_DELAY_SEC

distraction_start_time = None
distraction_detected = False


# ============================================================
# EYE WARNING
# ============================================================

EYE_WARNING_DURATION_SEC = 0.8

# Stronger warning after 3 seconds of continuous eye closure
STRONG_WARNING_DURATION_SEC = 3.0


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0


# ============================================================
# STATE VARIABLES
# ============================================================

current_state = "UNKNOWN"
previous_state = None

# Continuous eye-closure timer
eye_closed_start_time = None

# Prevent repeated strong-warning commands during one eye-closure event
strong_warning_sent = False

# Continuous no-driver timer
no_driver_start_time = None

# Yawn timer
yawn_start_time = None

# Current yawn status
yawn_detected = False

# Previous yawn status
# Used to detect YAWN_DETECTED -> YAWN_CLEAR
yawn_was_active = False

last_yawn_time = 0

session_yawn_count = 0
session_drowsiness_count = 0
session_distraction_count = 0

last_event_time = 0


# ============================================================
# EMERGENCY VEHICLE STOP
# ============================================================

# Once True, vehicle remains stopped.
#
# It can be activated by:
#
# 1. Eyes closed continuously >= 5 seconds
# OR
# 2. Driver not detected continuously >= 5 seconds
#
# Opening eyes or detecting the driver again will NOT
# automatically restart the vehicle.
# ============================================================

vehicle_emergency_stopped = False


# ============================================================
# EMERGENCY STOP REASON
# ============================================================

emergency_stop_reason = ""


# ============================================================
# CSV LOGGING
# ============================================================

LOG_FILE = "driver_monitoring_log.csv"


def initialize_log():

    if not os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Timestamp",
                "Event",
                "Eye Ratio",
                "Mouth Ratio",
                "State"
            ])


def log_event(
    event,
    eye_ratio="",
    mouth_ratio="",
    state=""
):

    try:

        with open(
            LOG_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                event,
                eye_ratio,
                mouth_ratio,
                state
            ])

    except Exception as e:

        print(
            "Logging error:",
            e
        )


initialize_log()


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def calculate_distance(point1, point2):

    return np.linalg.norm(
        np.array(point1) -
        np.array(point2)
    )


# ============================================================
# ASPECT RATIO
# ============================================================

def get_aspect_ratio(
    image,
    face_landmarks,
    top_bottom_points,
    left_right_points
):

    h, w, _ = image.shape

    top = face_landmarks.landmark[
        top_bottom_points[0]
    ]

    bottom = face_landmarks.landmark[
        top_bottom_points[1]
    ]

    left = face_landmarks.landmark[
        left_right_points[0]
    ]

    right = face_landmarks.landmark[
        left_right_points[1]
    ]

    top_point = (
        int(top.x * w),
        int(top.y * h)
    )

    bottom_point = (
        int(bottom.x * w),
        int(bottom.y * h)
    )

    left_point = (
        int(left.x * w),
        int(left.y * h)
    )

    right_point = (
        int(right.x * w),
        int(right.y * h)
    )

    vertical_distance = calculate_distance(
        top_point,
        bottom_point
    )

    horizontal_distance = calculate_distance(
        left_point,
        right_point
    )

    if vertical_distance == 0:

        return 0

    return (
        horizontal_distance /
        vertical_distance
    )


# ============================================================
# IRIS / GAZE CHECK
# ============================================================

def check_iris_in_middle(
    image,
    face_landmarks
):

    h, w, _ = image.shape

    try:

        # ----------------------------------------------------
        # LEFT IRIS
        # ----------------------------------------------------

        left_iris_points = []

        for index in LEFT_IRIS:

            point = face_landmarks.landmark[index]

            left_iris_points.append([
                point.x * w,
                point.y * h
            ])

        left_iris_center = np.mean(
            left_iris_points,
            axis=0
        )

        # ----------------------------------------------------
        # RIGHT IRIS
        # ----------------------------------------------------

        right_iris_points = []

        for index in RIGHT_IRIS:

            point = face_landmarks.landmark[index]

            right_iris_points.append([
                point.x * w,
                point.y * h
            ])

        right_iris_center = np.mean(
            right_iris_points,
            axis=0
        )

        # ----------------------------------------------------
        # LEFT EYE BOUNDARY
        # ----------------------------------------------------

        left_eye_left = face_landmarks.landmark[
            LEFT_EYE_LEFT_RIGHT[0]
        ]

        left_eye_right = face_landmarks.landmark[
            LEFT_EYE_LEFT_RIGHT[1]
        ]

        left_eye_center_x = (
            (left_eye_left.x + left_eye_right.x)
            * w / 2
        )

        # ----------------------------------------------------
        # RIGHT EYE BOUNDARY
        # ----------------------------------------------------

        right_eye_left = face_landmarks.landmark[
            RIGHT_EYE_LEFT_RIGHT[0]
        ]

        right_eye_right = face_landmarks.landmark[
            RIGHT_EYE_LEFT_RIGHT[1]
        ]

        right_eye_center_x = (
            (right_eye_left.x + right_eye_right.x)
            * w / 2
        )

        left_deviation = abs(
            left_iris_center[0] -
            left_eye_center_x
        )

        right_deviation = abs(
            right_iris_center[0] -
            right_eye_center_x
        )

        deviation_threshold = 2.8

        if (
            left_deviation <= deviation_threshold
            and
            right_deviation <= deviation_threshold
        ):

            return True

        return False

    except Exception:

        return True


# ============================================================
# YAWN DETECTION
# ============================================================

def check_yawn(
    image,
    face_landmarks
):

    global yawn_start_time
    global yawn_detected
    global last_yawn_time

    current_time = time.time()

    h, w, _ = image.shape

    # --------------------------------------------------------
    # MOUTH LANDMARKS
    # --------------------------------------------------------

    upper = face_landmarks.landmark[
        UPPER_LOWER_LIPS[0]
    ]

    lower = face_landmarks.landmark[
        UPPER_LOWER_LIPS[1]
    ]

    left = face_landmarks.landmark[
        LEFT_RIGHT_LIPS[0]
    ]

    right = face_landmarks.landmark[
        LEFT_RIGHT_LIPS[1]
    ]

    upper_point = np.array([
        upper.x * w,
        upper.y * h
    ])

    lower_point = np.array([
        lower.x * w,
        lower.y * h
    ])

    left_point = np.array([
        left.x * w,
        left.y * h
    ])

    right_point = np.array([
        right.x * w,
        right.y * h
    ])

    vertical_distance = np.linalg.norm(
        upper_point -
        lower_point
    )

    horizontal_distance = np.linalg.norm(
        left_point -
        right_point
    )

    if vertical_distance <= 0:

        yawn_start_time = None
        yawn_detected = False

        return False, None, False

    # --------------------------------------------------------
    # MOUTH RATIO
    # --------------------------------------------------------

    mouth_ratio = (
        horizontal_distance /
        vertical_distance
    )

    # --------------------------------------------------------
    # MOUTH OPEN CONDITION
    # --------------------------------------------------------

    mouth_open = (
        mouth_ratio <
        YAWN_RATIO_THRESHOLD
    )

    if mouth_open:

        # Draw mouth landmarks
        cv.circle(
            image,
            tuple(
                upper_point.astype(int)
            ),
            4,
            GREEN,
            -1
        )

        cv.circle(
            image,
            tuple(
                lower_point.astype(int)
            ),
            4,
            GREEN,
            -1
        )

        cv.circle(
            image,
            tuple(
                left_point.astype(int)
            ),
            4,
            GREEN,
            -1
        )

        cv.circle(
            image,
            tuple(
                right_point.astype(int)
            ),
            4,
            GREEN,
            -1
        )

        # ----------------------------------------------------
        # START MOUTH TIMER
        # ----------------------------------------------------

        if yawn_start_time is None:

            yawn_start_time = current_time

        open_duration = (
            current_time -
            yawn_start_time
        )

        cv.putText(
            image,
            f"Mouth Open: {open_duration:.1f}s",
            (20, 155),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            GREEN,
            2
        )

        # ----------------------------------------------------
        # YAWN CONFIRMED
        # ----------------------------------------------------

        if (
            open_duration >=
            YAWN_DURATION_SEC
            and
            not yawn_detected
            and
            (
                current_time -
                last_yawn_time
            ) >= YAWN_COOLDOWN_SEC
        ):

            yawn_detected = True

            last_yawn_time = current_time

            return True, mouth_ratio, True

    else:

        # ----------------------------------------------------
        # MOUTH CLOSED
        # ----------------------------------------------------

        yawn_start_time = None

        yawn_detected = False

        return False, mouth_ratio, False

    return False, mouth_ratio, mouth_open


# ============================================================
# SEND COMMAND TO ESP32
# ============================================================

def send_command(command):

    if ser is None:

        print(
            "Serial unavailable. "
            f"Command not sent: {command}"
        )

        return

    try:

        ser.write(
            (command + "\n").encode()
        )

        print(
            "Sent:",
            command
        )

    except Exception as e:

        print(
            "Serial send error:",
            e
        )


# ============================================================
# READ SERIAL REPLIES
# ============================================================

def read_serial_replies():

    if ser is None:

        return

    try:

        while ser.in_waiting:

            data = ser.readline().decode(
                errors="ignore"
            ).strip()

            if data:

                print(
                    "ESP32:",
                    data
                )

    except Exception as e:

        print(
            "Serial read error:",
            e
        )


# ============================================================
# CHANGE VEHICLE STATE
# ============================================================

def change_state(
    new_state,
    eye_ratio=None,
    mouth_ratio=None
):

    global current_state
    global session_drowsiness_count

    # --------------------------------------------------------
    # NEVER overwrite emergency stop
    # --------------------------------------------------------

    if vehicle_emergency_stopped:

        current_state = "EMERGENCY_STOP"

        return

    if new_state == current_state:

        return

    current_state = new_state

    # --------------------------------------------------------
    # DRIVER AWAKE
    # --------------------------------------------------------

    if new_state == "FACE_DETECTED":

        send_command(
            "FACE_DETECTED:Driver Awake"
        )

        log_event(
            "Driver Awake",
            eye_ratio,
            mouth_ratio,
            new_state
        )

    # --------------------------------------------------------
    # DROWSINESS
    # --------------------------------------------------------

    elif new_state == "ALERT":

        send_command(
            "ALERT:Drowsiness Detected"
        )

        session_drowsiness_count += 1

        log_event(
            "Drowsiness Detected",
            eye_ratio,
            mouth_ratio,
            new_state
        )

    # --------------------------------------------------------
    # NO FACE
    # --------------------------------------------------------

    elif new_state == "NO_FACE":

        send_command(
            "NO_FACE:No Driver Detected"
        )

        log_event(
            "No Driver Detected",
            eye_ratio,
            mouth_ratio,
            new_state
        )


# ============================================================
# ALERT SOUND
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def create_alert_sound():

    filename = os.path.join(
        SCRIPT_DIR,
        "drowsiness_alert.wav"
    )

    sample_rate = 44100
    duration = 0.6
    frequency = 2500

    num_samples = int(
        sample_rate *
        duration
    )

    with wave.open(
        filename,
        "w"
    ) as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(
            sample_rate
        )

        for i in range(num_samples):

            value = int(
                32767 *
                0.5 *
                math.sin(
                    2 *
                    math.pi *
                    frequency *
                    i /
                    sample_rate
                )
            )

            data = struct.pack(
                "<h",
                value
            )

            wav_file.writeframes(data)

    return filename


ALERT_SOUND_FILE = create_alert_sound()

print(
    f"Alert sound file: "
    f"{ALERT_SOUND_FILE}"
)

print(
    "Alert sound file exists:",
    os.path.exists(ALERT_SOUND_FILE)
)


_alert_sound_lock = threading.Lock()

_alert_sound_playing = False


def _play_alert_sound_worker():

    global _alert_sound_playing

    try:

        import winsound

        for _ in range(3):

            winsound.PlaySound(
                ALERT_SOUND_FILE,
                winsound.SND_FILENAME
            )

    except Exception as e:

        print(
            "Sound error:",
            e
        )

    finally:

        with _alert_sound_lock:

            _alert_sound_playing = False


def play_alert_sound():

    global _alert_sound_playing

    with _alert_sound_lock:

        if _alert_sound_playing:

            return

        _alert_sound_playing = True

    threading.Thread(
        target=_play_alert_sound_worker,
        daemon=True
    ).start()


# ============================================================
# CAMERA
# ============================================================

cap = cv.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    if ser is not None:

        ser.close()

    raise SystemExit


# ============================================================
# MEDIAPIPE FACE MESH
# ============================================================

with mp_face_mesh.FaceMesh(

    static_image_mode=False,

    max_num_faces=2,

    refine_landmarks=True,

    min_detection_confidence=0.4,

    min_tracking_confidence=0.4

) as face_mesh:

    print()
    print("==========================================")
    print(" DRIVER DROWSINESS MONITORING SYSTEM")
    print("==========================================")
    print()
    print("Eye drowsiness threshold : 1.5 seconds")
    print("Emergency eye stop       : 5 seconds")
    print("Emergency no-face stop   : 5 seconds")
    print("Yawn duration             : 1.5 seconds")
    print()
    print("Press Q to exit.")
    print()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "Camera frame could not be read."
            )

            break


        # ----------------------------------------------------
        # FLIP IMAGE
        # ----------------------------------------------------

        image = cv.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        rgb_image = cv.cvtColor(
            image,
            cv.COLOR_BGR2RGB
        )

        outputs = face_mesh.process(
            rgb_image
        )


        current_time = time.time()


        # ====================================================
        # NO DRIVER DETECTED
        # ====================================================

        if (
            outputs.multi_face_landmarks
            is None
            or
            len(
                outputs.multi_face_landmarks
            ) == 0
        ):

            # ------------------------------------------------
            # Reset eye timer
            # ------------------------------------------------

            eye_closed_start_time = None
            strong_warning_sent = False

            # ------------------------------------------------
            # Reset yawn state
            # ------------------------------------------------

            yawn_start_time = None
            yawn_detected = False
            yawn_was_active = False

            # ------------------------------------------------
            # Reset distraction
            # ------------------------------------------------

            distraction_start_time = None
            distraction_detected = False


            # ------------------------------------------------
            # START NO-DRIVER TIMER
            # ------------------------------------------------

            if no_driver_start_time is None:

                no_driver_start_time = (
                    current_time
                )

            no_driver_duration = (
                current_time -
                no_driver_start_time
            )


            # ------------------------------------------------
            # NO DRIVER TIMER DISPLAY
            # ------------------------------------------------

            if not vehicle_emergency_stopped:

                cv.putText(
                    image,
                    "NO DRIVER DETECTED",
                    (20, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    RED,
                    3
                )

                cv.putText(
                    image,
                    f"No Driver: "
                    f"{no_driver_duration:.1f}s",
                    (20, 90),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    YELLOW,
                    2
                )


            # ------------------------------------------------
            # 5-SECOND EMERGENCY STOP
            # ------------------------------------------------

            if (
                no_driver_duration >=
                NO_DRIVER_STOP_SEC
                and
                not vehicle_emergency_stopped
            ):

                # --------------------------------------------
                # LATCH STOP
                # --------------------------------------------

                vehicle_emergency_stopped = True

                emergency_stop_reason = (
                    "DRIVER NOT DETECTED >= 5 SECONDS"
                )

                current_state = "EMERGENCY_STOP"


                # --------------------------------------------
                # SEND STOP COMMAND
                # --------------------------------------------

                send_command(
                    "STOP_VEHICLE"
                )


                # --------------------------------------------
                # LOG
                # --------------------------------------------

                log_event(
                    "EMERGENCY VEHICLE STOP - "
                    "Driver Not Detected >= 5 Seconds",
                    "",
                    "",
                    "EMERGENCY_STOP"
                )


                # --------------------------------------------
                # CONSOLE
                # --------------------------------------------

                print()
                print("==========================================")
                print("!!! EMERGENCY VEHICLE STOP !!!")
                print(
                    "Driver not detected continuously "
                    "for 5 seconds or more."
                )
                print(
                    "STOP_VEHICLE command sent to ESP32."
                )
                print(
                    "Vehicle remains stopped."
                )
                print("==========================================")
                print()


            # ------------------------------------------------
            # EMERGENCY DISPLAY
            # ------------------------------------------------

            if vehicle_emergency_stopped:

                cv.putText(
                    image,
                    "!!! VEHICLE STOPPED !!!",
                    (20, 140),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    RED,
                    3
                )

                cv.putText(
                    image,
                    "DRIVER NOT DETECTED >= 5 SECONDS",
                    (20, 180),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    RED,
                    2
                )

                cv.putText(
                    image,
                    "EMERGENCY STOP ACTIVE",
                    (20, 220),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    RED,
                    2
                )


            # ------------------------------------------------
            # STATE
            # ------------------------------------------------

            if vehicle_emergency_stopped:

                current_state = "EMERGENCY_STOP"

            else:

                current_state = "NO_FACE"


            cv.putText(
                image,
                f"STATE: {current_state}",
                (20, 520),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                RED if vehicle_emergency_stopped else WHITE,
                2
            )


            # ------------------------------------------------
            # READ ESP32
            # ------------------------------------------------

            read_serial_replies()


            # ------------------------------------------------
            # SHOW
            # ------------------------------------------------

            cv.imshow(
                "AI Driver Drowsiness Detection",
                image
            )


            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            continue


        # ====================================================
        # DRIVER DETECTED
        # ====================================================

        # Reset no-driver timer immediately because a face
        # has been detected again.
        #
        # IMPORTANT:
        # This does NOT reset vehicle_emergency_stopped.
        # If the 5-second emergency stop already happened,
        # the vehicle remains stopped.

        no_driver_start_time = None


        # ====================================================
        # FIRST FACE
        # ====================================================

        face_landmarks = (
            outputs.multi_face_landmarks[0]
        )


        # ====================================================
        # DRAW FACE MESH
        # ====================================================

        mp_drawing.draw_landmarks(

            image,

            face_landmarks,

            mp_face_mesh.FACEMESH_TESSELATION,

            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=1,
                circle_radius=1
            ),

            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 180, 0),
                thickness=1
            )
        )


        # ====================================================
        # EYE RATIO
        # ====================================================

        left_eye_ratio = get_aspect_ratio(
            image,
            face_landmarks,
            LEFT_EYE_TOP_BOTTOM,
            LEFT_EYE_LEFT_RIGHT
        )

        right_eye_ratio = get_aspect_ratio(
            image,
            face_landmarks,
            RIGHT_EYE_TOP_BOTTOM,
            RIGHT_EYE_LEFT_RIGHT
        )

        eye_ratio = (
            left_eye_ratio +
            right_eye_ratio
        ) / 2


        # ====================================================
        # YAWN DETECTION
        # ====================================================

        yawn_now, mouth_ratio, mouth_open = (
            check_yawn(
                image,
                face_landmarks
            )
        )


        # ====================================================
        # DISPLAY RATIOS
        # ====================================================

        cv.putText(
            image,
            f"Eye Ratio: {eye_ratio:.2f}",
            (20, 85),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            CYAN,
            2
        )

        if mouth_ratio is not None:

            cv.putText(
                image,
                f"Mouth Ratio: {mouth_ratio:.2f}",
                (20, 120),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                CYAN,
                2
            )


        # ====================================================
        # YAWN DETECTED
        # ====================================================

        if yawn_now:

            session_yawn_count += 1

            yawn_was_active = True

            # Send to ESP32
            # ESP32 can activate vibration / LCD.
            send_command(
                "YAWN_DETECTED"
            )

            log_event(
                "Yawn Detected",
                eye_ratio,
                mouth_ratio,
                current_state
            )

            print()
            print("================================")
            print("YAWN DETECTED")
            print(
                f"Mouth Ratio: "
                f"{mouth_ratio:.2f}"
            )
            print(
                "Yawn vibration command sent."
            )
            print("================================")


        # ====================================================
        # YAWN CLEAR
        #
        # IMPORTANT:
        # This is NOT displayed on the computer screen.
        #
        # It is only sent to the ESP32/LCD.
        # ====================================================

        elif (
            yawn_was_active
            and
            not mouth_open
        ):

            yawn_was_active = False

            # Send clear command ONLY ONCE
            send_command(
                "YAWN_CLEAR"
            )

            log_event(
                "Yawn Clear",
                eye_ratio,
                mouth_ratio,
                current_state
            )

            print(
                "YAWN CLEAR - Mouth closed."
            )


        # ====================================================
        # EYE CLOSURE
        # ====================================================

        eyes_closed = (
            eye_ratio >
            EYE_CLOSED_RATIO_THRESHOLD
        )


        if eyes_closed:

            if eye_closed_start_time is None:

                eye_closed_start_time = (
                    current_time
                )

            eye_closed_duration = (
                current_time -
                eye_closed_start_time
            )

        else:

            eye_closed_start_time = None
            strong_warning_sent = False

            eye_closed_duration = 0


        # ====================================================
        # EYE TIMER DISPLAY
        # ====================================================

        if eyes_closed:

            cv.putText(
                image,
                f"Eyes Closed: "
                f"{eye_closed_duration:.1f}s",
                (20, 190),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                YELLOW,
                2
            )


        # ====================================================
        # 3-SECOND STRONG WARNING
        #
        # After 3 seconds of continuous eye closure, send a
        # stronger warning command to the ESP32. The ESP32 can
        # use this command to increase vibration/buzzer intensity.
        # ====================================================

        if (
            eyes_closed
            and
            eye_closed_duration >=
            STRONG_WARNING_DURATION_SEC
            and
            not strong_warning_sent
            and
            not vehicle_emergency_stopped
        ):

            strong_warning_sent = True

            send_command(
                "STRONG_WARNING"
            )

            log_event(
                "Strong Drowsiness Warning",
                eye_ratio,
                mouth_ratio,
                current_state
            )

            print()
            print("================================")
            print("!!! STRONG DROWSINESS WARNING !!!")
            print("Eyes closed continuously for 3 seconds.")
            print("Strong vibration/buzzer command sent to ESP32.")
            print("Vehicle will stop at 5 seconds if eyes remain closed.")
            print("================================")

            cv.putText(
                image,
                "STRONG WARNING!",
                (20, 250),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                ORANGE,
                3
            )


        # ====================================================
        # 5-SECOND EMERGENCY EYE STOP
        # ====================================================

        if (
            eyes_closed
            and
            eye_closed_duration >=
            EYE_EMERGENCY_STOP_SEC
            and
            not vehicle_emergency_stopped
        ):

            vehicle_emergency_stopped = True

            emergency_stop_reason = (
                "EYES CLOSED >= 5 SECONDS"
            )

            current_state = "EMERGENCY_STOP"


            # Send actual vehicle stop command
            send_command(
                "STOP_VEHICLE"
            )


            log_event(
                "EMERGENCY VEHICLE STOP - "
                "Eyes Closed >= 5 Seconds",
                eye_ratio,
                mouth_ratio,
                "EMERGENCY_STOP"
            )


            print()
            print("==========================================")
            print("!!! EMERGENCY VEHICLE STOP !!!")
            print(
                "Driver eyes closed continuously "
                "for 5 seconds or more."
            )
            print(
                "STOP_VEHICLE command sent to ESP32."
            )
            print(
                "Vehicle remains stopped."
            )
            print("==========================================")
            print()


        # ====================================================
        # EMERGENCY STOP DISPLAY
        # ====================================================

        if vehicle_emergency_stopped:

            cv.putText(
                image,
                "!!! VEHICLE STOPPED !!!",
                (20, 270),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                RED,
                3
            )

            cv.putText(
                image,
                emergency_stop_reason,
                (20, 305),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                RED,
                2
            )

            cv.putText(
                image,
                "EMERGENCY STOP ACTIVE",
                (20, 340),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                RED,
                2
            )


        # ====================================================
        # NORMAL EYE / DROWSINESS LOGIC
        #
        # Do not execute after emergency stop.
        # ====================================================

        if not vehicle_emergency_stopped:

            # ------------------------------------------------
            # EYE WARNING
            # ------------------------------------------------

            if (
                eyes_closed
                and
                eye_closed_duration >=
                EYE_WARNING_DURATION_SEC
                and
                eye_closed_duration <
                EYE_CLOSED_DURATION_SEC
            ):

                cv.putText(
                    image,
                    "EYES CLOSING...",
                    (20, 225),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    YELLOW,
                    2
                )


            # ------------------------------------------------
            # STRONG WARNING DISPLAY
            # ------------------------------------------------

            if (
                eyes_closed
                and
                eye_closed_duration >=
                STRONG_WARNING_DURATION_SEC
                and
                eye_closed_duration <
                EYE_EMERGENCY_STOP_SEC
            ):

                cv.putText(
                    image,
                    "STRONG WARNING - OPEN EYES!",
                    (20, 250),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    ORANGE,
                    3
                )


            # ------------------------------------------------
            # DROWSINESS
            # ------------------------------------------------

            if eyes_closed:

                if (
                    eye_closed_duration >=
                    EYE_CLOSED_DURATION_SEC
                ):

                    change_state(
                        "ALERT",
                        eye_ratio,
                        mouth_ratio
                    )

                    cv.putText(
                        image,
                        "DROWSINESS DETECTED",
                        (20, 270),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        RED,
                        3
                    )


                    # ----------------------------------------
                    # SOUND ALERT
                    # ----------------------------------------

                    if (
                        current_time -
                        last_event_time
                    ) > 3:

                        last_event_time = (
                            current_time
                        )

                        play_alert_sound()


            else:

                # --------------------------------------------
                # EYES OPEN
                # --------------------------------------------

                change_state(
                    "FACE_DETECTED",
                    eye_ratio,
                    mouth_ratio
                )

                cv.putText(
                    image,
                    "DRIVER AWAKE",
                    (20, 270),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    GREEN,
                    2
                )


        # ====================================================
        # DISTRACTION DETECTION
        #
        # INFORMATION ONLY
        # ====================================================

        if not check_iris_in_middle(
            image,
            face_landmarks
        ):

            if distraction_start_time is None:

                distraction_start_time = (
                    current_time
                )

            distraction_duration = (
                current_time -
                distraction_start_time
            )

            cv.putText(
                image,
                f"Looking Away: "
                f"{distraction_duration:.1f}s",
                (20, 310),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                YELLOW,
                2
            )

            if (
                distraction_duration >=
                DISTRACTION_DELAY_SEC
                and
                not distraction_detected
            ):

                distraction_detected = True

                session_distraction_count += 1

                log_event(
                    "Distraction Detected",
                    eye_ratio,
                    mouth_ratio,
                    current_state
                )

                print(
                    "Distraction detected."
                )

        else:

            distraction_start_time = None
            distraction_detected = False


        # ====================================================
        # MOUTH OPEN DISPLAY
        # ====================================================

        if mouth_open:

            cv.putText(
                image,
                "MOUTH OPEN",
                (20, 350),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                GREEN,
                2
            )


        # ====================================================
        # YAWN DISPLAY
        #
        # IMPORTANT:
        # Only show YAWN DETECTED.
        #
        # Do NOT show YAWN CLEAR on computer display.
        # YAWN_CLEAR is sent to ESP32/LCD only.
        # ====================================================

        if yawn_detected:

            cv.putText(
                image,
                "YAWN DETECTED",
                (20, 390),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                ORANGE,
                3
            )


        # ====================================================
        # SESSION INFORMATION
        # ====================================================

        cv.putText(
            image,
            f"Yawns: {session_yawn_count}",
            (20, 430),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            CYAN,
            2
        )

        cv.putText(
            image,
            f"Drowsiness: "
            f"{session_drowsiness_count}",
            (20, 455),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            MAGENTA,
            2
        )

        cv.putText(
            image,
            f"Distractions: "
            f"{session_distraction_count}",
            (20, 480),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            ORANGE,
            2
        )


        # ====================================================
        # CURRENT STATE
        # ====================================================

        if vehicle_emergency_stopped:

            current_state = "EMERGENCY_STOP"

        cv.putText(
            image,
            f"STATE: {current_state}",
            (20, 520),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            RED if vehicle_emergency_stopped else WHITE,
            2
        )


        # ====================================================
        # READ ESP32 REPLIES
        # ====================================================

        read_serial_replies()


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv.imshow(
            "AI Driver Drowsiness Detection",
            image
        )


        # ====================================================
        # EXIT
        # ====================================================

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv.destroyAllWindows()


if ser is not None:

    ser.close()


print()
print("==========================================")
print("SYSTEM STOPPED")
print(
    f"Total Yawns: "
    f"{session_yawn_count}"
)
print(
    f"Total Drowsiness Events: "
    f"{session_drowsiness_count}"
)
print(
    f"Total Distraction Events: "
    f"{session_distraction_count}"
)

if vehicle_emergency_stopped:

    print(
        "Emergency Vehicle Stop: ACTIVATED"
    )

    print(
        f"Reason: {emergency_stop_reason}"
    )

else:

    print(
        "Emergency Vehicle Stop: NOT ACTIVATED"
    )

print("==========================================")