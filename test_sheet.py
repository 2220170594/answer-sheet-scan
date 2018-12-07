from sheet import get_answer_from_sheet
import cv2,math,os

import time

start = time.clock()

image_name = "0031.jpg"; ##0006 0008 没有检测到内边缘

get_answer_from_sheet("img/new/"+image_name);
end = time.clock()
print "get_answer_from_sheet: %f s" % (end - start)


