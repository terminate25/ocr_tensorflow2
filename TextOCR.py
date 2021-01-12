#!/usr/bin/python3
import argparse
import sys
import tensorflow as tf
from cv2 import cv2

from TextDetector import TextDetector
from TextRecognizer import TextRecognizer


class TextOCR(object):

    def __init__(self):
        self.detector = TextDetector()
        self.recognizer = TextRecognizer()

    def scan(self, img):
        textlines, bbox, scores = self.detector.detect(img)
        results = list()
        for textline in textlines:
            timg = img[int(textline[1]) - 5:int(textline[3]) + 5, int(textline[0]) - 5:int(textline[2]) + 5, :]
            text, _ = self.recognizer.recognize(timg)
            results.append({'image': timg, 'text': text, 'position': textline})
        return results


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_image', '-i',
    help="Path to input file",
    default='data/test/img_6.jpg'
)

if __name__ == "__main__":
    filename = parser.parse_args().input_image
    img = cv2.imread(filename)
    if img is None:
        print('failed to open image!')
        exit()
    ocr = TextOCR()
    results = ocr.scan(img)
    for result in results:
        print(result['text'])
