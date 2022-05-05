import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob


def run_test():
    color_matching = {'Green': 'Green', 'Red': 'Red', 'Yellow': 'Red', 'None': 'None'}
    for color in color_matching:
        path = 'Traffic Data/' + color + '/*.jpg'
        images = getImages(path)
        total = len(images)
        correct = 0
        for img in images:
            loaded_image = np.asarray(cv2.imread(img))
            try:
                traffic_color = detect_traffic_light_color(loaded_image)
                if traffic_color == color_matching[color]:
                    correct += 1
            except:
                continue
        print(str(color) + ' ' + str(correct) + ' out of ' + str(total) + ' : ' + str(correct / total))


def getImages(path):
    images = []
    for imageName in glob.iglob(path, recursive=True):
        images.append(imageName)
    return images


def detect_traffic_light_color(cropped_object):
    gray = np.asarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2GRAY))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=12, minRadius=0, maxRadius=10)
    cropped_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if i[2] > i[1]:
                x_min = 0
            else:
                x_min = i[1] - i[2]
            x_max = i[1]+i[2]
            if i[2] > i[0]:
                y_min = 0
            else:
                y_min = i[0] - i[2]
            y_max = i[0]+i[2]
            cropped_circles.append(cropped_object[x_min: x_max, y_min: y_max])
    else:
        return 'None'
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
        return 'None'
    if red_color > 1 or green_color > 1:
        if red_color > green_color and red_color > blue_color:
            return 'Red'
        if green_color > red_color or blue_color > red_color:
            return 'Green'

def main():
    run_test()


if __name__ == "__main__":
    main()
