# coding=utf-8
import math
import numpy as np
import cv2
from imutils import auto_canny, contours
from e import PolyNodeCountError
from score import score
import settings
#from settings import CHOICES, SHEET_AREA_MIN_RATIO, PROCESS_BRIGHT_COLS, PROCESS_BRIGHT_ROWS, BRIGHT_VALUE, \
    #CHOICE_COL_COUNT, CHOICES_PER_QUE, settings.WHITE_RATIO_PER_CHOICE, MAYBE_MULTI_CHOICE_THRESHOLD, CHOICE_CNT_COUNT


def get_corner_node_list(poly_node_list):
    """
    获得多边形四个顶点的坐标
    :type poly_node_list: ndarray
    :return: tuple
    """
    center_y, center_x = (np.sum(poly_node_list, axis=0) / 4)[0]
    top_left = bottom_left = top_right = bottom_right = None
    for node in poly_node_list:
        x = node[0, 1]
        y = node[0, 0]
        if x < center_x and y < center_y:
            top_left = node
        elif x < center_x and y > center_y:
            bottom_left = node
        elif x > center_x and y < center_y:
            top_right = node
        elif x > center_x and y > center_y:
            bottom_right = node
    return top_left, bottom_left, top_right, bottom_right


def detect_cnt_again(poly, base_img):
    """
    继续检测已截取区域是否涵盖了答题卡区域
    :param poly: ndarray
    :param base_img: ndarray
    :return: ndarray
    """
    # 该多边形区域是否还包含答题卡区域的flag
    flag = False

    # 计算多边形四个顶点，并且截图，然后处理截取后的图片
    top_left, bottom_left, top_right, bottom_right = get_corner_node_list(poly)
    roi_img = get_roi_img(base_img, bottom_left, bottom_right, top_left, top_right)
    img = get_init_process_img(roi_img)

    # 获得面积最大的轮廓
    cnt = get_max_area_cnt(img)

    # 如果轮廓面积足够大，重新计算多边形四个顶点
    if cv2.contourArea(cnt) > roi_img.shape[0] * roi_img.shape[1] * settings.SHEET_AREA_MIN_RATIO:
        flag = True
        poly = cv2.approxPolyDP(cnt, cv2.arcLength((cnt,), True) * 0.1, True)
        top_left, bottom_left, top_right, bottom_right = get_corner_node_list(poly)
        if not poly.shape[0] == 4:
            raise PolyNodeCountError

    # 多边形顶点和图片顶点，主要用于纠偏
    base_poly_nodes = np.float32([top_left[0], bottom_left[0], top_right[0], bottom_right[0]])
    base_nodes = np.float32([[0, 0],
                            [base_img.shape[1], 0],
                            [0, base_img.shape[0]],
                            [base_img.shape[1], base_img.shape[0]]])
    transmtx = cv2.getPerspectiveTransform(base_poly_nodes, base_nodes)

    if flag:
        img_warp = cv2.warpPerspective(roi_img, transmtx, (base_img.shape[1], base_img.shape[0]))
    else:
        img_warp = cv2.warpPerspective(base_img, transmtx, (base_img.shape[1], base_img.shape[0]))
    
    return img_warp


def get_init_process_img(roi_img):
    """
    对图片进行初始化处理，包括，梯度化，高斯模糊，二值化，腐蚀，膨胀和边缘检测
    :param roi_img: ndarray
    :return: ndarray
    """
    h = cv2.Sobel(roi_img, cv2.CV_32F, 0, 1, -1)
    v = cv2.Sobel(roi_img, cv2.CV_32F, 1, 0, -1)
    img = cv2.add(h, v)
    img = cv2.convertScaleAbs(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    img = auto_canny(img)
    return img


def get_roi_img(base_img, bottom_left, bottom_right, top_left, top_right):
    """
    截取合适的图片区域
    :param base_img: ndarray
    :param bottom_left: ndarray
    :param bottom_right: ndarray
    :param top_left: ndarray
    :param top_right: ndarray
    :return: ndarray
    """
    min_v = top_left[0, 1] if top_left[0, 1] < bottom_left[0, 1] else bottom_left[0, 1]
    max_v = top_right[0, 1] if top_right[0, 1] > bottom_right[0, 1] else bottom_right[0, 1]
    min_h = top_left[0, 0] if top_left[0, 0] < top_right[0, 0] else top_right[0, 0]
    max_h = bottom_left[0, 0] if bottom_left[0, 0] > bottom_right[0, 0] else bottom_right[0, 0]
    roi_img = base_img[min_v + 10:max_v - 10, min_h + 10:max_h - 10]
    return roi_img


def get_bright_process_img(img):
    """
    改变图片的亮度，方便二值化
    :param img: ndarray
    :return: ndarray
    """
    '''
    
    for y in range(PROCESS_BRIGHT_COLS):
        for x in range(PROCESS_BRIGHT_ROWS):
            col_low = 1.0 * img.shape[0] / PROCESS_BRIGHT_COLS * y
            col_high = 1.0 * img.shape[0] / PROCESS_BRIGHT_COLS * (y + 1)
            row_low = 1.0 * img.shape[1] / PROCESS_BRIGHT_ROWS * x
            row_high = 1.0 * img.shape[1] / PROCESS_BRIGHT_ROWS * (x + 1)
            roi = img[int(col_low):int(col_high), int(row_low): int(row_high)]
            mean = cv2.mean(roi)
            for each_roi in roi:
                for each_p in each_roi:
                    each_p += BRIGHT_VALUE - np.array(mean, dtype=np.uint8)[:3]
    
    '''
    
    
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def get_max_area_cnt(img):
    """
    获得图片里面最大面积的轮廓
    :param img: ndarray
    :return: ndarray
    """
    cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=lambda c: cv2.contourArea(c))
    return cnt


def get_ans(ans_img, rows):
    # 选项个数
    interval = get_item_interval()
    my_score = 0
    items_per_row = get_items_per_row()
    answer_list = []
    all_percents = []
    for i,row in enumerate(rows):
        for k in range(len(row)/settings.CHOICES_PER_QUE):
            for j, c in enumerate(row[k * interval:interval + k * interval]):
                new = ans_img[c[1]:(c[1] + c[3]), c[0]:(c[0] + c[2])]
                white_count = np.count_nonzero(new)
                percent = white_count * 1.0 / new.size 
                all_percents.append(percent)                
    all_percents.sort()
    stand_th_low = 0
    stand_th_up = 0
    # 找边界值
    for i,p in enumerate(all_percents[1:]):
        if p - all_percents[i] > 0.1:
            if stand_th_low == 0 and i < len(all_percents)*2/3:#填涂的选项应该不会超过三分之二
                stand_th_low = p
                break
    if get_ave(all_percents) < 0.3 and get_median(all_percents) < 0.3 and stand_th_low == 0:# 认为全涂了
        stand_th_low == 0
    
    #print stand_th_low    
    for i, row in enumerate(rows):
        # 从左到右为当前题目的气泡轮廓排序，然后初始化被涂画的气泡变量
        for k in range(len(row)/settings.CHOICES_PER_QUE):
            #print '======================================='
            percent_list = []
            for j, c in enumerate(row[k * interval:interval + k * interval]):
                try:
                    # 获得选项框的区域
                    new = ans_img[c[1]:(c[1] + c[3]), c[0]:(c[0] + c[2])]
                    # 计算白色像素个数和所占百分比
                    white_count = np.count_nonzero(new)
                    percent = white_count * 1.0 / new.size
                except IndexError:
                    percent = 1
                percent_list.append({'col': k + 1, 'row': i + 1, 'percent': percent, 'choice': settings.CHOICES[j]})
                
            temp_percent_list = percent_list[:];
            temp_percent_list.sort(key=lambda x: x['percent'])
            temp_percent_list.reverse()
            settings.WHITE_RATIO_PER_CHOICE = 0
            if stand_th_low == 0:
                for i,x in enumerate(temp_percent_list[1:]):
                    if temp_percent_list[i]['percent'] - x['percent'] > 0.1: 
                            settings.WHITE_RATIO_PER_CHOICE = temp_percent_list[i]['percent']
                            break
                if settings.WHITE_RATIO_PER_CHOICE == 0:
                    settings.WHITE_RATIO_PER_CHOICE = 0.35
                    
            ans_str = ""
            for temp_choice in percent_list:
                #print stand_th_low
                if stand_th_low != 0 and temp_choice['percent'] <stand_th_low :#or temp_choice['percent']>stand_th_up+0.3:
                    ans_str += temp_choice['choice']
                elif stand_th_low == 0 and temp_choice['percent'] < settings.WHITE_RATIO_PER_CHOICE:
                    ans_str += temp_choice['choice']
            answer_list.append(ans_str) 
            print(percent_list)

           
    #print '=====总分========'
    print (answer_list)


def get_items_per_row():
    items_per_row = settings.CHOICE_COL_COUNT / settings.CHOICES_PER_QUE #(CHOICES_PER_QUE + 1)
    return items_per_row


def get_item_interval():
    interval = settings.CHOICES_PER_QUE #+ 1
    return interval


def delete_rect(cents_pos, que_cnts):
    count = 0
    for i, c in enumerate(cents_pos):
        ratio = 1.0 * c[2] / c[3]
        if 0.5 > ratio  or ratio > 2:
            que_cnts.pop(i - count)
            count += 1
    return que_cnts


def get_choice_row_count():
    choice_row_count = int(math.ceil(settings.CHOICE_CNT_COUNT * 1.0 / settings.CHOICE_COL_COUNT))
    return choice_row_count
#中位数
def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0: # 判断列表长度为偶数
        median = (data[size//2]+data[size//2-1])/2
        data[0] = median
    if size % 2 == 1: # 判断列表长度为奇数
        median = data[(size-1)//2]
        data[0] = median
    return data[0]
def get_ave(data):
    sum = 0
    for x in data:
        sum += x;
    return sum / len(data) 

def checkRect(rect1,rect2):
    th = 10
    if abs(rect1[0]-rect2[0]) < th and abs(rect1[1] - rect2[1]) < th*2 and abs( \
        rect1[2] - rect2[2])<th and abs(rect1[3]-rect2[3])<th: #位置和大小都匹配
        print (1)
        return 1
    elif abs(rect1[2]-rect2[2]) < th and abs(rect1[3]- rect2[3])< th and abs(\
        rect1[1]-rect2[1])<th:# 在同一个水平线上，但是间距不对，中间可能有遗漏
        print (2)
        return 2
    else:
        print (3)
        return 3
    

def sort_by_row_hs(cnts_pos):
    choice_row_count = get_choice_row_count()
    count = 0
    rows = []
    print(type(cnts_pos[0]))
    threshold = get_min_row_interval(cnts_pos)
    for i in range(choice_row_count):
        cols = cnts_pos[i * settings.CHOICE_COL_COUNT - count:(i + 1) * settings.CHOICE_COL_COUNT - count]
        # threshold = _std_plus_mean(cols)
        temp_row = [cols[0]]
        for j, col in enumerate(cols[1:]):
            temp_row.append(col)
        count += settings.CHOICE_COL_COUNT - len(temp_row)
        temp_row.sort(key=lambda x: x[0])
        rows.append(temp_row)

    # insert_no_full_row(rows)

    return rows   

    
def sort_by_row_hs2(cnts_pos):
    
    #1.计算边界的值
    min_left = min(cnts_pos, key=lambda x: x[0])[0]+5
    max_right = max(cnts_pos, key=lambda x: x[0])[0]-5
    min_top = min(cnts_pos, key=lambda x: x[1])[1]+5
    max_top = max(cnts_pos, key=lambda x: x[1])[1]-5

    
    #2.将矩形按照一条龙放入队列
    queues = []
    temp_row = [cnts_pos[0]]
    choice_maigins = []
    question_margins = []
    choice_maigin_y= []
    rows = []
    for i,pos  in enumerate(cnts_pos[1:]):
        if abs(pos[1] - cnts_pos[i][1])<5:
            temp_row.append(pos)
        else:
            temp_row = sorted(temp_row,key=lambda x:x[0])
            if len(temp_row) == settings.CHOICE_COL_COUNT:#统计边界间距
                for j,pos2 in enumerate(temp_row[1:]):
                    if (j+1) % settings.CHOICES_PER_QUE == 0 and (j+1) < settings.CHOICE_COL_COUNT:#(两个选择题的边界
                        question_margins.append(pos2[0]-temp_row[j][0]-temp_row[j][2])
                    else:
                        choice_maigins.append(pos2[0]-temp_row[j][0]-temp_row[j][2])
            queues.extend(temp_row)
            rows.append(temp_row)
            temp_row = [pos] ## 不一定是第一个因为没有排序
            choice_maigin_y.append(pos[1]-queues[-1][1])
    temp_row = sorted(temp_row,key=lambda x:x[0])#排序后放入
    queues.extend(temp_row) 
    rows.append(temp_row)
    if len(question_margins)==0 or len(choice_maigins)==0 or len(choice_maigin_y)==0:
        print ("修复失败")
    
    '''
     if len(question_margins) == 0 and len(choice_maigins) == 0:# 可能一行完整的数据都没有找到,从rows里面寻找上下间距的值
        for i, row in enumerate(rows[1:]):
            choice_maigin_y.append(get_ave([x[1] for x in row]) - get_ave([x[1] for x in rows[i]]))# 计算两行间隙
    '''            
    # 选项之间左右的间距，题目之间左右的间距，选项之间的上下间距
    
    '''
    if len(choice_maigins) >0:
        choice_margin_x = get_median(choice_maigins)
    else:
        choice_margin_x = 21
    
    if len(question_margins)>0:
        question_margin_x = get_median(choice_maigins)
    else:
        question_margin_x = 137
        
    if len(choice_maigin_y) > 0:
        question_margin_y = get_ave(choice_maigin_y)
    else:
        question_margin_y = 135
    '''
    choice_margin_x =  get_median(choice_maigins)
    question_margin_x = get_median(question_margins)
    question_margin_y = get_ave(choice_maigin_y)
    
    #宽度和高度
    cell_height = get_median([x[3] for x in queues])
    cell_width = get_median([x[2] for x in queues])
    #3.对队列里面的值有效性和完整性进行判断和修复
    
    final_queue = []
    insert_cnt = 0        
    ## 算法一    
    i = 0
    while i+ insert_cnt < settings.CHOICE_CNT_COUNT and i < len(queues):
        if (len(final_queue) >0):
            last_rect = final_queue[-1]
        else:
            last_rect = queues[0]
        if i == 0: #第一个选项
            print ('++++++' + str(i))
            expect_rect =  (min_left,min_top,cell_width,cell_height) #预估计矩形
        elif (i+insert_cnt) % settings.CHOICE_COL_COUNT == 0 and (i+insert_cnt) > 1:# 非一行第一个元素
            print ('----' + str(i))
            expect_rect = (min_left,last_rect[1]+question_margin_y,cell_width,cell_height) #预估计矩形
        elif (i+insert_cnt) % settings.CHOICES_PER_QUE == 0 and (i+insert_cnt)% settings.CHOICE_COL_COUNT != 0 : # 相邻的问题  python2 <>
            print ('======' + str(i))
            expect_rect = (last_rect[0]+last_rect[2]+question_margin_x,last_rect[1],cell_width,cell_height) #预估计矩形 
        else:
            print ('-=-=-=' + str(i))
            expect_rect = (last_rect[0]+last_rect[2]+choice_margin_x,last_rect[1],cell_width,cell_height)#同一个问题相邻的选项
        check_result = checkRect(expect_rect,queues[i])    
        if check_result == 1 or check_result == 3:
            final_queue.append(queues[i])
            i = i + 1   
        elif check_result == 2:#这里没有考虑连续缺失的情况
            final_queue.append(expect_rect)
            #final_queue.append(queues[i])
            insert_cnt += 1
    if len(final_queue) != settings.CHOICE_CNT_COUNT:
        print ("题目数量检测错误")
        exit()
    ##对最后的队列进行分行
   
    choice_row_count = get_choice_row_count()
    count = 0
    rows = []
    threshold = get_min_row_interval(final_queue)
    for i in range(choice_row_count):
        cols = final_queue[i * settings.CHOICE_COL_COUNT - count:(i + 1) * settings.CHOICE_COL_COUNT - count]
        temp_row = [cols[0]]
        for j, col in enumerate(cols[1:]):
            temp_row.append(col)
        count += settings.CHOICE_COL_COUNT - len(temp_row)
        temp_row.sort(key=lambda x: x[0])
        rows.append(temp_row)    
    return rows


def sort_by_col(cnts_pos):
    # TODO
    cnts_pos.sort(key=lambda x: x[0])
    choice_row_count = get_choice_row_count()
    count = 0
    cols = []
    threshold = get_min_col_interval(cnts_pos)
    for i in range(settings.CHOICE_COL_COUNT):
        rows = cnts_pos[i * choice_row_count - count:(i + 1) * choice_row_count - count]
        # threshold = _std_plus_mean(cols)
        temp_col = [rows[0]]
        for j, row in enumerate(rows[1:]):
            if row[0] - rows[j - 1][0] < threshold:
                temp_col.append(row)
            else:
                break
        count += choice_row_count - len(temp_col)
        temp_col.sort(key=lambda x: x[1])
        cols.append(temp_col)
    return cols


def insert_null_2_rows(cols, rows):
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            try:
                if row[j] != col[0]:
                    row.insert(j, 'null')
                else:
                    col.pop(0)
            except IndexError:
                row.insert(j, 'null')


def get_min_row_interval(cnts_pos):
    choice_row_count = get_choice_row_count()
    rows_interval = []
    for i, c in enumerate(cnts_pos[1:]):
        rows_interval.append(c[1] - cnts_pos[i][1])
    rows_interval.sort(reverse=True)
    return min(rows_interval[:choice_row_count - 1])


def get_min_col_interval(cnts_pos):
    cols_interval = []
    for i, c in enumerate(cnts_pos[1:]):
        cols_interval.append(c[0] - cnts_pos[i][0])
    cols_interval.sort(reverse=True)
    return min(cols_interval[:settings.CHOICE_COL_COUNT - 1])


def insert_no_full_row(rows):
    full_row_list, not_full_row_list = sep_full_n_no_full_choice_rows(rows)
    low_up_dt = _get_choices_low_up(full_row_list)
    for row in not_full_row_list:
        miss_size = settings.CHOICE_COL_COUNT - len(row)
        for i, node in enumerate(row):
            if not (low_up_dt[i][1] >= node[0] >= low_up_dt[i][0]):
                row.insert(i, 'null')
                miss_size -= 1
            if not miss_size:
                break


def sep_full_n_no_full_choice_rows(rows):
    _full_row_list = []
    _not_full_row_list = []
    for row in rows:
        if len(row) == settings.CHOICE_COL_COUNT:
            row.sort(key=lambda x: x[0])
            _full_row_list.append(row)
        else:
            row.sort(key=lambda x: x[0])
            _not_full_row_list.append(row)
    return _full_row_list, _not_full_row_list


def _get_choices_low_up(rows):
    _dt = _get_item_choices_x(rows)
    dt = _get_items_choice_low_up(_dt)
    return dt


def _get_item_choices_x(rows):
    dt = {}
    for row in rows:
        for i in range(settings.CHOICE_COL_COUNT):
            try:
                dt[i].append(row[i][0])
            except (KeyError, AttributeError):
                dt[i] = [row[i][0]]
    return dt


def _get_items_choice_low_up(rows_dt):
    dt = {}
    for key in rows_dt.keys():
        choices_x = rows_dt[key]
        dt[key] = _std_plus_low_up_mean(choices_x)
    return dt


def _std_plus_mean(cols):
    nums = 0
    square_nums = 0
    for col in cols:
        nums += col[1]
        square_nums += col[1] ** 2
    mean = nums / len(cols)
    std = (square_nums / len(cols) - mean ** 2) ** 0.5
    return round(mean + 1.5 * std, 0)


def _std_plus_low_up_mean(nums):
    sums = 0.0
    squares = 0.0
    for num in nums:
        sums += num
        squares += num ** 2
    mean = sums / len(nums)
    std = (squares / len(nums) - mean ** 2) ** 0.5
    return mean - 3 * std, mean + 3 * std