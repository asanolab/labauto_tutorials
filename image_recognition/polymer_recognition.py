#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import csv
import cv2
import math
import numpy as np
import os
import sys
import time

argc = len(sys.argv)

# parameters
AREA_THRESH_U = 132000  # almost full size of the frame r50mm
AREA_THRESH_B = 4500

# HSV filter for samples in Section5
h_L = 0.1 * 255
h_U = 0.2 * 255
s_L = 0.2 * 255
s_U = 0.6 * 255
v_L = 0.4 * 255
v_U = 0.6 * 255

# HSV filter for samples in Section4
#h_L = 0.15 * 255
#h_U = 0.3 * 255
#s_L = 0.1 * 255
#s_U = 0.2 * 255
#v_L = 0.6 * 255
#v_U = 0.8 * 255


class PolymerRecognition:
    def __init__(self, polymer_dense=0.956, polymer_mass=1.0, src_img_path='sample_0.jpg', result_dir='./result', img_id=0):
        # args:
        # - polymer_dense [g/cm^3]
        # - polymer_mass [g]
        self.area_1pixel = 1
        self.polymer_dense = polymer_dense
        self.polymer_mass  = polymer_mass
        self.src_img_path = src_img_path
        self.result_dir= result_dir
        self.img_id = img_id  # img_id = exp_id

        # input file
        print("Input image: {}".format(self.src_img_path))

        # generate dir
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)


    def generate_frame_mask(self, img_tmp):
        # detect frame
        r_frame_mm = 50.0  # [mm]

        gray_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
        gray_tmp = cv2.medianBlur(gray_tmp, 5)
        circles_frame = cv2.HoughCircles(gray_tmp, cv2.HOUGH_GRADIENT, dp=1, minDist=400, param1=50, param2=60, minRadius=200, maxRadius=250)  # if not detect the frame, try to decrease param2
        circles_frame = np.uint16(np.around(circles_frame))

        file_name = 'frame_circle_' + str(self.img_id) + '.jpg'
        file_path = os.path.join(self.result_dir, file_name)

        for i in circles_frame[0,:]:
            x_frame = i[0]
            y_frame = i[1]
            r_frame = i[2]
            # outer circle
            cv2.circle(img_tmp, (x_frame, y_frame), r_frame, (0,255,0), 1)
            # center
            cv2.circle(img_tmp, (x_frame, y_frame), 2, (0,0,255), 3)
            cv2.imwrite(file_path, img_tmp)

        self.area_1pixel = (r_frame_mm/r_frame)**2 # 220pixel=50mm -> w,h_1pixel=0.227mm
        print('area_1pixel: {}'.format(self.area_1pixel))

        # mask by frame
        h,w = img_tmp.shape[:2]
        mask = np.zeros((h,w), dtype=np.uint8) # make same size with the input img
        cv2.circle(mask, (x_frame, y_frame), r_frame-1, color=255, thickness=-1)  # r_frame-1 means mask inside frame

        return mask


    def blur_filter(self, mask_img):
        gray_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
        file_name = 'gray_' + str(self.img_id) + '.jpg'
        file_path = os.path.join(self.result_dir, file_name)
        cv2.imwrite(file_path, gray_img)

        blur_gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        ret, bw_img = cv2.threshold(blur_gray_img, 0, 255, cv2.THRESH_OTSU)

        file_name = 'black_white_' + str(self.img_id) + '.jpg'
        file_path = os.path.join(self.result_dir, file_name)
        cv2.imwrite(file_path, bw_img)

        return bw_img


    def hsv_filter(self, img):
        ## hsv filter
        hsvLower = np.array([h_L, s_L, v_L])
        hsvUpper = np.array([h_U, s_U, v_U])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)           # convert to HSV
        hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)      # mask by HSV
        hsvResult = cv2.bitwise_and(img, img, mask=hsv_mask) # merge img and mask

        file_name = 'hsv_filter_' + str(self.img_id) + '.jpg'
        file_path = os.path.join(self.result_dir, file_name)
        cv2.imwrite(file_path, hsvResult)
        time.sleep(1)

        ## 2値化
        bgr = cv2.cvtColor(hsvResult, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        file_name = 'thresh_' + str(self.img_id) + '.jpg'
        file_path = os.path.join(self.result_dir, file_name)
        cv2.imwrite(file_path, thresh)

        ## noise removal
        kernel = np.ones((3, 3), np.uint8)
        bw_tmp = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=15)
        bw_img = cv2.morphologyEx(bw_tmp, cv2.MORPH_OPEN, kernel, iterations=15)

        file_name = 'morph_close_' + str(self.img_id) + '.jpg'
        file_path = os.path.join(self.result_dir, file_name)
        cv2.imwrite(file_path, bw_tmp)

        file_name = 'morph_open_' + str(self.img_id) + '.jpg'
        file_path = os.path.join(self.result_dir, file_name)
        cv2.imwrite(file_path, bw_img)

        return bw_img


    def polymer_recog(self):
        recog_fail_value = 0.001  # return value used when recognition error.

        img = cv2.imread(self.src_img_path)
        img_tmp = img.copy()

        # mask by frame
        try:
            mask = self.generate_frame_mask(img_tmp)
        except:
            print('No frame was found')
            return recog_fail_value

        img[mask==0] = [0, 0, 0] # make the pixel black(0 0 0) where mask is black(==0)

        file_name = 'mask_' + str(self.img_id) + '.jpg'
        file_path = os.path.join(self.result_dir, file_name)
        cv2.imwrite(file_path, img)
        mask_img = img.copy()
        mask_img_copy = img.copy()

        # filter
        bw_img = self.hsv_filter(mask_img)

        # get contours
        if(sys.version_info.major) == 3:
            contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # for python3(OpenCV4)
        else:
            imgEdge, contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # for python2

        num_contours = len(contours)
        if num_contours == 0:
            print('No contours were detected')
            return recog_fail_value
        else:
            print('number of contours detected : {}'.format(num_contours))

        # draw all contours
        img_all_contours = cv2.drawContours(mask_img_copy, contours, -1, (0,255,0), 3)
        file_name = 'contours_all_' + str(self.img_id) + '.jpg'
        file_path = os.path.join(self.result_dir, file_name)
        cv2.imwrite(file_path, img_all_contours)


        # processing each contour
        area_largest = 0.0
        time.sleep(1)
        for contour in contours:
            # check area
            area = cv2.contourArea(contour)
            area_mm2 = area * self.area_1pixel

            if (area < AREA_THRESH_B):
                print('area is below the threshold')
                continue
            elif (area > AREA_THRESH_U):
                print('area is above the threshold')
                continue

            # calc thickness
            thickness = self.calc_thickness(self.polymer_dense, self.polymer_mass, area_mm2)  # [mm]

            # calc roundness
            roundness = self.calc_roundness(area, contour)

            # contour
            img_contour = cv2.drawContours(mask_img, contour, -1, (0,255,0), 3)

            # print contour info
            print('')
            print('Detected contour:')
            print(' area           : {}'.format(area))
            print(' area [mm^2]    : {}'.format(area_mm2))
            print(' roundness      : {}'.format(roundness))
            print(' thickness [mm] : {}'.format(thickness))

            # largest area is regarded as polymer
            if area > area_largest:
                polymer_contour = img_contour
                polymer_area_mm2 = area_mm2
                polymer_thickness = thickness
                polymer_roundness = roundness
                area_largest = area

        if area_largest == 0.0:
            print('Contours area are too small')
            return recog_fail_value

        # print polymer info
        print('')
        print('Detected polymer:')
        print(' area [mm^2]    : {}'.format(polymer_area_mm2))
        print(' roundness      : {}'.format(polymer_roundness))
        print(' thickness [mm] : {}'.format(polymer_thickness))

        # draw detected polymer contour
        file_name = 'detected_polymer_' + str(self.img_id) + '.jpg'
        file_path = os.path.join(self.result_dir, file_name)
        cv2.imwrite(file_path, polymer_contour)

        # write csv
        result_csv_path = os.path.join(self.result_dir, 'result_{}.csv'.format(self.img_id))
        with open(result_csv_path, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['exp_id', 'area[mm^2]', 'roundness', 'thickness[mm]'])
            writer.writerow([self.img_id, '{:.4f}'.format(polymer_area_mm2), '{:.4f}'.format(polymer_roundness), '{:.4f}'.format(polymer_thickness)])

        print('')
        print('Recognition completed')


    def calc_thickness(self, dense, mass, area_mm2):
        V_cm3 = mass / dense   # [cm^3]
        V     = V_cm3 * 10**3  # [mm^3]
        return V / area_mm2  # [mm]


    def calc_roundness(self, area, contour):
        perimeter = cv2.arcLength(contour, True)
        if (perimeter == 0):
            return 0

        return 4 * math.pi * area / perimeter**2


if __name__ == '__main__':
    if(argc==1):  # all image
        for i in range(20):
            sample_name = 'sample_' + str(i) + '.jpg'
            pr = PolymerRecognition(src_img_path=sample_name, img_id=i)
            pr.polymer_recog()
    elif(argc==3):  # specified image
        pr = PolymerRecognition(src_img_path=sys.argv[1], img_id=sys.argv[2])
        pr.polymer_recog()
    else:
        print('Number of arg is wrong')
        exit()
