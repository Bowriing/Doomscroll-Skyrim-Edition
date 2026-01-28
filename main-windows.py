import cv2
import mediapipe as mp
import time
import subprocess
import os
from pathlib import Path


class VideoPlayer:
    """Plays video using Windows default media player (with audio)."""

    def __init__(self, video_path: Path):
        self.video_path = str(video_path.absolute())
        self.process = None

    def start(self):
        """Start playing the video with default media player."""
        if self.process is not None:
            return

        print(f"\n[VideoPlayer] Opening video: {self.video_path}")

        # Use os.startfile to open with default player (includes audio)
        try:
            os.startfile(self.video_path)
            self.process = True  # Flag that video is playing
            print("[VideoPlayer] Video started with default player")
        except Exception as e:
            print(f"[VideoPlayer] ERROR: {e}")

    def stop(self):
        """Stop the video by killing common media player processes."""
        if self.process is None:
            return

        print("\n[VideoPlayer] Stopping video...")

        # Kill common Windows media players
        players = ['Video.UI.exe', 'Microsoft.Media.Player.exe', 'wmplayer.exe', 'vlc.exe', 'Movies & TV', 'WindowsMediaPlayer.exe']
        for player in players:
            try:
                subprocess.run(
                    ['taskkill', '/F', '/IM', player],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2
                )
            except:
                pass

        self.process = None
        print("[VideoPlayer] Video stopped")


def draw_warning(frame, text="lock in twin"):
    h, w = frame.shape[:2]
    box_w, box_h = 500, 70
    x1 = (w - box_w) // 2
    y1 = 24
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 0, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (80, 255, 160), 4)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 160), 2)

    cv2.putText(
        frame,
        text.upper(),
        (x1 + 26, y1 + 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


def main():
    timer = 2.0
    # Eye gaze thresholds
    eye_looking_down_threshold = 0.40
    eye_debounce_threshold = 0.50
    # Head tilt thresholds (higher = head tilted down more)
    head_looking_down_threshold = 0.90
    head_debounce_threshold = 0.85

    skyrim_skeleton_video = Path("./assets/skyrim-skeleton.mp4").resolve()
    if not skyrim_skeleton_video.exists():
        print(f"Could not find video at: {skyrim_skeleton_video}")
        print("Make sure skyrim-skeleton.mp4 is in the ./assets/ folder")
        return

    print("=" * 60)
    print("DOOMSCROLL DETECTOR - Windows Edition")
    print("=" * 60)
    print("Instructions:")
    print("  - Look down for 2 seconds to trigger alarm video")
    print("  - Look up to stop the video")
    print("  - Press ESC to quit")
    print("=" * 60)

    face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Could not open webcam")
        print("Make sure your webcam is connected and not in use by another app")
        return

    print("Webcam opened successfully")

    doomscroll = None
    video_playing = False
    video_player = VideoPlayer(skyrim_skeleton_video)
    print(f"Video player ready: {skyrim_skeleton_video}")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            height, width, depth = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_image = face_mesh_landmarks.process(rgb_frame)
            face_landmark_points = processed_image.multi_face_landmarks

            current = time.time()

            if not face_landmark_points:
                print("No face detected", end='\r')

            if face_landmark_points:
                one_face_landmark_points = face_landmark_points[0].landmark

                left = [one_face_landmark_points[145], one_face_landmark_points[159]]
                for landmark_point in left:
                    x = int(landmark_point.x * width)
                    y = int(landmark_point.y * height)

                right = [one_face_landmark_points[374], one_face_landmark_points[386]]
                for landmark_point in right:
                    x = int(landmark_point.x * width)
                    y = int(landmark_point.y * height)

                lx = int((left[0].x + left[1].x) / 2 * width)
                ly = int((left[0].y + left[1].y) / 2 * height)

                rx = int((right[0].x + right[1].x) / 2 * width)
                ry = int((right[0].y + right[1].y) / 2 * height)

                box = 50

                cv2.rectangle(frame, (lx - box, ly - box), (lx + box, ly + box), (10, 255, 0), 2)
                cv2.rectangle(frame, (rx - box, ry - box), (rx + box, ry + box), (10, 255, 0), 2)

                l_iris = one_face_landmark_points[468]
                r_iris = one_face_landmark_points[473]

                l_ratio = (l_iris.y - left[1].y) / (left[0].y - left[1].y + 1e-6)
                r_ratio = (r_iris.y - right[1].y) / (right[0].y - right[1].y + 1e-6)

                eye_ratio = (l_ratio + r_ratio) / 2.0

                # Head pitch detection using nose tip and forehead
                nose_tip = one_face_landmark_points[4]      # Nose tip
                forehead = one_face_landmark_points[10]     # Top of forehead
                chin = one_face_landmark_points[152]        # Chin

                # Calculate head pitch ratio: how far down the nose is relative to forehead-chin line
                # Higher value = head tilted down
                face_height = chin.y - forehead.y + 1e-6
                nose_position = (nose_tip.y - forehead.y) / face_height
                head_ratio = nose_position

                # Debug logging
                eye_thresh = eye_debounce_threshold if video_playing else eye_looking_down_threshold
                head_thresh = head_debounce_threshold if video_playing else head_looking_down_threshold

                eye_down = eye_ratio < eye_thresh
                head_down = head_ratio > head_thresh

                looking_status = "DOWN" if (eye_down or head_down) else "UP"
                time_looking = f"{current - doomscroll:.1f}s" if doomscroll else "0.0s"
                print(f"Eye: {eye_ratio:.2f}(<{eye_thresh:.2f}={eye_down}) | Head: {head_ratio:.2f}(>{head_thresh:.2f}={head_down}) | {looking_status} | {time_looking}   ", end='\r')

                # Trigger if EITHER eyes looking down OR head tilted down
                if video_playing:
                    is_looking_down = (eye_ratio < eye_debounce_threshold) or (head_ratio > head_debounce_threshold)
                else:
                    is_looking_down = (eye_ratio < eye_looking_down_threshold) or (head_ratio > head_looking_down_threshold)

                if is_looking_down:
                    if doomscroll is None:
                        doomscroll = current

                    if (current - doomscroll) >= timer:
                        if not video_playing:
                            print(f"\n>>> TRIGGERING VIDEO PLAYER <<<")
                            video_player.start()
                            video_playing = True
                            print("ALARM TRIGGERED - Stop doomscrolling!")

                else:
                    doomscroll = None
                    if video_playing:
                        video_player.stop()
                        video_playing = False
                        print("Video stopped - Keep it up!")
            else:
                doomscroll = None
                if video_playing:
                    video_player.stop()
                    video_playing = False

            if video_playing:
                draw_warning(frame, "doomscrolling alarm")

            cv2.imshow('lock in', frame)
            key = cv2.waitKey(1)

            if key == 27:  # ESC
                print("\nExiting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        if video_playing:
            video_player.stop()

        cam.release()
        cv2.destroyAllWindows()
        print("Cleanup complete. Goodbye!")


if __name__ == '__main__':
    main()