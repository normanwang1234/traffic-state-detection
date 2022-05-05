import os
import numpy as np
import cv2
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt


class ObjectDetection:
    BRIGHT_FACTOR = 2
    # Taken from CoCo detection labels
    PERSON_ID = 1
    BICYCLE_ID = 2
    CAR_ID = 3
    MOTORCYCLE_ID = 4
    BUS_ID = 6
    TRUCK_ID = 8
    TRAFFIC_LIGHT_ID = 10
    STOP_SIGN = 12

    def __init__(self, video_filepath: str):
        self.base_dir = os.getcwd()
        self.video_path = self.base_dir + video_filepath
        self.relevant_labels = {1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Motorcycle', 6: 'Bus', 8: 'Truck',
                                10: 'Traffic', 12: 'Stop'}
        self.detection_model = tf.saved_model.load('models/centernet_hg104_512x512_coco17_tpu-8/saved_model')
        self.preprocessed_frames = []

    def run_detection(self):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 15000)
        prev_frame = None
        while cap.isOpened():
            ret, frame = cap.read()
            try:
                processed_frame = self._preprocess_for_image(frame)
            except:
                with open("video_frames3.pickle", "wb") as f:
                    pickle.dump(self.preprocessed_frames, f, protocol=pickle.HIGHEST_PROTOCOL)
                break
            lane_marked_frame = self._find_lane_markers(processed_frame)
            if prev_frame is None:
                prev_frame = lane_marked_frame
                continue
            rgb_flow, state = self._apply_dense_optical_flow(lane_marked_frame, prev_frame)
            prev_frame = lane_marked_frame

            # Plotting bounding boxes and current state of car
            self._detect_objects(frame)
            cv2.putText(frame, 'Current State: ' + state,
                        (25, 450), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            try:
                self.preprocessed_frames.append(frame)
            except:
                with open("video_frames3.pickle", "wb") as f:
                    pickle.dump(self.preprocessed_frames, f, protocol=pickle.HIGHEST_PROTOCOL)
                break

            if len(self.preprocessed_frames) == 5000:
                with open("video_frames3.pickle", "wb") as f:
                    pickle.dump(self.preprocessed_frames, f, protocol=pickle.HIGHEST_PROTOCOL)
                break

        cap.release()

    def _preprocess_for_image(self, frame: np.ndarray):
        frame = self._crop_frame(frame)
        frame = self._adjust_brightness(frame)
        frame = self._apply_gaussian_blur(frame)
        return frame

    def _adjust_brightness(self, frame: np.ndarray):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame[:, :, 2] = hsv_frame[:, :, 2] * self.BRIGHT_FACTOR
        frame_rgb = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
        return frame_rgb

    def _apply_gaussian_blur(self, frame: np.ndarray):
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 2)
        return blurred_frame

    def _apply_dense_optical_flow(self, frame: np.ndarray, prev_frame: np.ndarray):
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        gray_current = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hsv = np.zeros((250, 550, 3))
        hsv[:, :, 1] = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[:, :, 1]
        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_current,
                                            None,
                                            0.5,
                                            1,
                                            5,
                                            2,
                                            5,
                                            2,
                                            0)
        magnitude = 0
        for i in range(flow.shape[0]):
            for j in range(flow.shape[1]):
                magnitude += abs(flow[i][j][0])

        state = 'Not Moving'
        if magnitude < 7500:
            state = 'Not Moving'
        elif 7500 <= magnitude < 20000:
            state = 'Decelerating/Going Slow'
        elif magnitude > 20000:
            state = 'Accelerating'
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[:, :, 0] = ang * (90 / np.pi)
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = np.asarray(hsv, dtype=np.float32)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return [rgb_flow, state]

    def _crop_frame(self, frame: np.ndarray):
        return frame[100:350, :-90]

    def _detect_objects(self, frame):
        loaded_image = np.asarray(frame[100:350, :-90])
        image_tensor = tf.convert_to_tensor(loaded_image)
        image_tensor = image_tensor[tf.newaxis, ...]
        output = self.detection_model(image_tensor)
        height, width, _ = loaded_image.shape
        detected_boxes = output['detection_boxes'][0]
        detected_scores = output['detection_scores'][0]
        detected_classes = output['detection_classes'][0]
        for index in range(len(detected_boxes)):
            image_id = int(detected_classes[index])
            image_confidence = float(detected_scores[index])
            if image_id not in self.relevant_labels or image_confidence < 0.65:
                continue
            y_min = int(detected_boxes[index][0] * height)
            x_min = int(detected_boxes[index][1] * width)
            y_max = int(detected_boxes[index][2] * height)
            x_max = int(detected_boxes[index][3] * width)
            y_dist = y_max - y_min
            cropped_object = np.asarray(loaded_image[y_min:y_max, x_min:x_max])
            # Relevant objects are detected
            traffic_color = None
            if image_id in self.relevant_labels and image_confidence >= 0.65:
                cv2.rectangle(loaded_image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
                if image_id == self.TRAFFIC_LIGHT_ID:
                    traffic_color = self._detect_traffic_light_color(cropped_object)
                if traffic_color:
                    text_string = traffic_color
                else:
                    text_string = self.relevant_labels[image_id]
                cv2.putText(loaded_image, text_string + ': ' + str(round(image_confidence * 100, 1)) + '%',
                            (int(x_min), int(y_min + y_dist // 2)), cv2.FONT_HERSHEY_TRIPLEX,
                            0.4, (0, 0, 255), 1, cv2.LINE_AA)

    def _detect_traffic_light_color(self, cropped_object):
        gray = np.asarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2GRAY))
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=10, minRadius=0, maxRadius=10)
        cropped_circles = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                if i[2] > i[1]:
                    x_min = 0
                else:
                    x_min = i[1] - i[2]
                x_max = i[1] + i[2]
                if i[2] > i[0]:
                    y_min = 0
                else:
                    y_min = i[0] - i[2]
                y_max = i[0] + i[2]
                cropped_circles.append(cropped_object[x_min: x_max, y_min: y_max])
        else:
            return
        red_color = 1
        green_color = 1
        blue_color = 1
        for cropped_circle in cropped_circles:
            rows, cols, _ = cropped_circle.shape
            for i in range(rows):
                for j in range(cols):
                    red_color += cropped_circle[i][j][2]
                    green_color += cropped_circle[i][j][1]
                    blue_color += cropped_circle[i][j][0]
        if red_color == 1 and green_color == 1:
            return
        if red_color > 1 or green_color > 1:
            if red_color > green_color and red_color > blue_color:
                return 'Red'
            if green_color > red_color or blue_color > red_color:
                return 'Green'

    def _find_lane_markers(self, img):
        height, width, ch = img.shape
        lane_vertices = [
            (100, height),
            (250, 100),
            (350, 100),
            (width, height),
        ]
        vertices = np.array([lane_vertices], np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, (255,) * 3)
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    def visualize(self, stream=None):
        if not stream:
            stream = self.preprocessed_frames
        for frame in stream:
            cv2.imshow('Visualize', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def preload_video():
    with open(r"video_frames2.pickle", "rb") as input_file:
        video_frames = pickle.load(input_file)
        for frame in video_frames:
            cv2.imshow('Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def main():
    # preprocessor = ObjectDetection('/data/train.mp4')
    # preprocessor.run_detection()
    preload_video()


if __name__ == "__main__":
    main()
