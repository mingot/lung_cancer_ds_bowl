import numpy as np
import cv2

def generate_simmetry(shape_size, theta):
    theta *= np.pi / 180
    
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square,
                                [center_square[0] + square_size, center_square[1]],
                                [center_square[0], center_square[1] + square_size]])
    pts2 = np.float32([center_square,
                                [center_square[0] + square_size * np.cos(2 * theta), center_square[1] + square_size * np.sin(2 * theta)],
                                [center_square[0] + square_size * np.sin(2 * theta), center_square[1] - square_size * np.cos(2 * theta)]])
    return cv2.getAffineTransform(pts1, pts2)