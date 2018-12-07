# coding=utf-8
#import cPickle  #python2
import pickle
import os
import cv2
from imutils import contours   #pip install web.py==0.40-dev1
from e import ContourCountError, ContourPerimeterSizeError, PolyNodeCountError
import numpy as np
import settings
#from settings import ANS_IMG_THRESHOLD, CNT_PERIMETER_THRESHOLD, CHOICE_IMG_THRESHOLD, ANS_IMG_DILATE_ITERATIONS, \
#    ANS_IMG_ERODE_ITERATIONS, CHOICE_IMG_DILATE_ITERATIONS, CHOICE_IMG_ERODE_ITERATIONS, CHOICE_MAX_AREA, \
    #CHOICE_CNT_COUNT, ANS_IMG_KERNEL, CHOICE_IMG_KERNEL, CHOICE_MIN_AREA
from utils import detect_cnt_again, get_init_process_img, get_bright_process_img, get_max_area_cnt, get_ans,sort_by_row_hs,sort_by_row_hs2

'''
def get_choice_area(areas):
    areas = sorted(areas)
    segments = []
    segment_areas = [areas[0]]
    for i, c in enumerate(areas[1:]):
        if abs(c-areas[i]) < 300:   
            segment_areas.append(areas[i])
        else:
            segments.append(segment_areas)
            segment_areas = [areas[i]]
    temp = segments[0]      
    for array in segments:
        if len(array) > len(temp) or len(temp)> 250 and len(array) < 250:#小于250
            temp = array
    return temp[0]-10,temp[-1]+10
'''
def get_choice_area(areas):
    areas = sorted(areas)
    segments = []
    segment_areas = []
    for i, c in enumerate(areas[1:]):
        #print c
        if abs(c-areas[i]) < 200:   
            segment_areas.append(areas[i])
        else:
            segments.append(segment_areas)
            segment_areas = []
    segments.append(segment_areas)
    temp = segments[0]      
    for array in segments:
        if len(array) > len(temp) or len(temp)> 250:#小于250
            temp = array
    return temp[0]-60,temp[-1]+60



def brightness(im_file):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]
def get_answer_from_sheet(base_img):

    filepath,tempfilename = os.path.split(base_img);
    file_name,_ = os.path.splitext(tempfilename)
    obj_dir = os.path.curdir + "/img/new/"+ file_name
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir) 
    base_img = cv2.imread(base_img);
    #保存最原始图片
    cv2.imwrite(obj_dir+"/"+'origin.png', base_img)
    # 灰度化然后进行边缘检测、二值化等等一系列处理
    img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    img = get_init_process_img(img)
    #写入图片
    cv2.imwrite(obj_dir+"/"+'process-0.png',img)
    # 获取最大面积轮廓并和图片大小作比较，看轮廓周长大小判断是否是答题卡的轮廓
    cnt = get_max_area_cnt(img)
    cnt_perimeter = cv2.arcLength(cnt, True)
    cv2.drawContours(base_img, [cnt], 0, (0, 255, 0), 1)
    #画边框
    cv2.imwrite(obj_dir+"/"+'green_border.png', base_img)    
    base_img_perimeter = (base_img.shape[0] + base_img.shape[1]) * 2
    if not cnt_perimeter > settings.CNT_PERIMETER_THRESHOLD * base_img_perimeter:
        print ("边缘丢失")
        exit()
    # 计算多边形的顶点，并看是否是四个顶点
    poly_node_list = cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.1, True)
    if not poly_node_list.shape[0] == 4:
        raise PolyNodeCountError
    # 根据计算的多边形顶点继续处理图片，主要是是纠偏
    processed_img = detect_cnt_again(poly_node_list, base_img)
    #保存纠正图片
    #processed_img = cv2.dilate(processed_img, kernel, iterations=1)    
    wait_draw = processed_img.copy()
    cv2.imwrite(obj_dir+"/"+'correct-position.png', processed_img)
    # 调整图片的亮度
    processed_img = get_bright_process_img(processed_img)
    cv2.imwrite(obj_dir+"/"+'brighten.png', processed_img)
    
    #processed_img = processed_img[processed_img[1]+20:(processed_img[1] + processed_img[3]-20), processed_img[0]+20:(processed_img[0] + processed_img[2]-20)]
    # 通过二值化和膨胀腐蚀获得填涂区域
    #ans_img = cv2.adaptiveThreshold(processed_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,2)  
    #ret, ans_img = cv2.threshold(processed_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #新的方法 
    #ans_img = cv2.dilate(processed_img, settings.ANS_IMG_KERNEL, iterations=settings.ANS_IMG_DILATE_ITERATIONS)
    #ans_img = cv2.erode(ans_img, settings.ANS_IMG_KERNEL, iterations=settings.ANS_IMG_ERODE_ITERATIONS)
    ans_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, settings.ANS_IMG_KERNEL)
    
    ans_img = cv2.adaptiveThreshold(ans_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,45,1) 
    cv2.imwrite(obj_dir+"/"+'answer_area.png', ans_img)
    
    # 通过二值化和膨胀腐蚀获得选项框区域
    #choice_img = cv2.adaptiveThreshold(processed_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,35,2) 
    #ret, choice_img = cv2.threshold(processed_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)#新方法
    #choice_img = cv2.dilate(processed_img, settings.CHOICE_IMG_KERNEL, iterations=settings.CHOICE_IMG_DILATE_ITERATIONS)
    #choice_img = cv2.erode(processed_img, settings.CHOICE_IMG_KERNEL, iterations=settings.CHOICE_IMG_ERODE_ITERATIONS)
    choice_img = cv2.adaptiveThreshold(processed_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
    #choice_img = cv2.morphologyEx(choice_img,cv2.MORPH_GRADIENT,settings.ANS_IMG_KERNEL)
   
    cv2.imwrite(obj_dir+"/"+'choice_area.png', choice_img)
    #cv2.waitKey(0)
    
    # 查找选项框以及前面题号的轮廓
    cnts, h = cv2.findContours(choice_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #cnts = []
    #for c in cnt1s:
    #    cnts.append(cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.1, True))
    question_cnts = []
    cnts_areas = []
    for i,c in enumerate(cnts):
        cnts_areas.append(cv2.contourArea(c))
    
    CHOICE_MIN_AREA,CHOICE_MAX_AREA = get_choice_area(cnts_areas)
    #print "%d %d" %(CHOICE_MIN_AREA,CHOICE_MAX_AREA)
    for i,c in enumerate(cnts):
        w = cv2.boundingRect(c)[2]
        h = cv2.boundingRect(c)[3]
        if CHOICE_MIN_AREA < cnts_areas[i]< CHOICE_MAX_AREA \
            and ((w/h<=1 and h/w <2) or (w/h>1 and w/h <2)):
            question_cnts.append(c)
            
    cv2.drawContours(wait_draw, question_cnts, -1, (0, 0, 255), 1)
    cv2.imshow("img", wait_draw)  
    cv2.waitKey(0)
    cv2.imwrite(obj_dir+"/"+'wait_draw5.png', wait_draw)
    cv2.waitKey(0)    
    
    if len(question_cnts) < settings.CHOICE_CNT_COUNT/2:
        print ("数目错误 %d %d" %(len(question_cnts),settings.CHOICE_CNT_COUNT))
        exit()    
    #对轮廓之上而下的排序
    question_cnts, cnts_pos = contours.sort_contours(question_cnts, method="left-to-right")
    question_cnts, cnts_pos = contours.sort_contours(question_cnts, method="top-to-bottom")
    rows = sort_by_row_hs2(list(cnts_pos))
    get_ans(ans_img, rows)

   
    