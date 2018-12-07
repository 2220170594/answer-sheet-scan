import cv2,sys,os
from sheet import get_answer_from_sheet
import settings

if len(sys.argv) == 5:
    global CHOICES_PER_QUE,CHOICE_COL_COUNT,CHOICE_CNT_COUNT
    base_img = cv2.imread(sys.argv[1])
    CHOICES_PER_QUE = int(sys.argv[2])
    CHOICE_COL_COUNT = int(sys.argv[3])
    settings.CHOICE_CNT_COUNT = int(sys.argv[4])
    
    #print settings.CHOICE_CNT_COUNT
    get_answer_from_sheet(base_img)
else:
    print("param count not right")
    
    