from typing import Generator

import os
import numpy as np
import cv2
import pickle


class ObjectDetection:
    BRIGHT_FACTOR = 2
    def __init__(self, video_filepath: str):
        self.base_dir = os.getcwd()
        self.video_path = self.base_dir + video_filepath

    def run_detection(self, visual_type='flow'):
        cap = cv2.VideoCapture(self.video_path)
        #15000
        cap.set(cv2.CAP_PROP_POS_FRAMES, 14400)
        prev_frame = None
        while cap.isOpened():
            ret, frame = cap.read()
            processed_frame = self._preprocess_for_image(frame)
            lane_marked_frame = self._find_lane_markers(processed_frame)
            if prev_frame is None:
                prev_frame = lane_marked_frame
                continue
            rgb_flow, state = self._apply_dense_optical_flow(lane_marked_frame, prev_frame)
            prev_frame = lane_marked_frame
            cv2.putText(rgb_flow, 'Current State: ' + state,
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if visual_type == 'flow':
                yield rgb_flow
            elif visual_type == 'lane':
                yield lane_marked_frame

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
        image_scale = 0.5
        nb_images = 1
        win_size = 5
        nb_iterations = 2
        deg_expansion = 5
        standard_dev = 2
        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_current,
                                            None,
                                            image_scale,
                                            nb_images,
                                            win_size,
                                            nb_iterations,
                                            deg_expansion,
                                            standard_dev,
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
        hsv[:, :, 0] = ang * (180 / np.pi / 2)
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = np.asarray(hsv, dtype=np.float32)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return [rgb_flow, state]

    def _crop_frame(self, frame: np.ndarray):
        return frame[100:350, :-90]


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

        # Returning the image only where mask pixels match
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

def visualize(stream: Generator):
    for frame in stream:
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def main():
    preprocessor = ObjectDetection('/data/train.mp4')
    stream = preprocessor.run_detection(visual_type='flow')
    visualize(stream)


if __name__ == "__main__":
    main()
