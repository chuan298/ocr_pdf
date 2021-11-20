from test_abc import check
from time import time

from detect import TextDetector
import cv2
from PIL import Image, ImageEnhance
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os, torch
import numpy as np
import time
import utility
import pdf2image
import tempfile
from table_reconstruction.table_reconstruction import TableExtraction
textDetect = TextDetector()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
# config['cnn']['pretrained']=False
# config['device'] = 'cpu'
# config['predictor']['beamsearch']=False
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img', default="/home/ancv/Work/KhmOcr/real_data_test/crop_box/", help='foo help')
parser.add_argument('--config', default="/home/ancv/Work/ocr_pdf/vietocr/config/vgg-transformer.yml", help='foo help')

args = parser.parse_args()
config = Cfg.load_config_from_file(args.config)

detector = Predictor(config)

########################
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
cfg = get_cfg()
cfg.merge_from_file("/home/ancv/Work/TableBank/config_101.yaml")
cfg.MODEL.WEIGHTS = "/home/ancv/Work/TableBank/model_final (1).pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
table_predictor = DefaultPredictor(cfg)

def get_tables(image):

    def group_h_lines(h_lines, thin_thresh):
        new_h_lines = []
        while len(h_lines) > 0:
            thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
            # print(h_lines, thin_thresh)
            lines = [line for line in h_lines if thresh[1] -
                    thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
            h_lines = [line for line in h_lines if thresh[1] - thin_thresh >
                    line[0][1] or line[0][1] > thresh[1] + thin_thresh]
            x = []
            for line in lines:
                x.append(line[0][0])
                x.append(line[0][2])
            x_min, x_max = min(x), max(x)
            average_y = sum([line[0][1] for line in lines]) // len(lines)
            new_h_lines.append([x_min, average_y, x_max, average_y])
        return new_h_lines

    def group_v_lines(v_lines, thin_thresh):
        new_v_lines = []
        while len(v_lines) > 0:
            thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
            lines = [line for line in v_lines if thresh[0] -
                    thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
            v_lines = [line for line in v_lines if thresh[0] - thin_thresh >
                    line[0][0] or line[0][0] > thresh[0] + thin_thresh]
            y = []
            for line in lines:
                y.append(line[0][1])
                y.append(line[0][3])
            average_x = sum([line[0][0] for line in lines]) // len(lines)
            y_min, y_max = min(y), max(y)
            new_v_lines.append([average_x, y_min, average_x, y_max])
        return new_v_lines

    def preprocess(img, factor: int):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Sharpness(img).enhance(factor)
        if gray.std() < 30:
            enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
        return np.array(enhancer)

    def get_table_coordinate(hor_lines, ver_lines):
        

        tab_x1 = min(min([line[0] for line in hor_lines]), min([line[0] for line in ver_lines]))
        tab_y1 = min(min([line[1] for line in hor_lines]), min([line[1] for line in ver_lines]))
        tab_x2 = max(max([line[2] for line in hor_lines]), max([line[2] for line in ver_lines]))
        tab_y2 = max(max([line[3] for line in hor_lines]), max([line[3] for line in ver_lines]))

        return [tab_x1, tab_y1, tab_x2, tab_y2]





    image = preprocess(image, 2.0)
    outputs = table_predictor(image)
    tables = outputs["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"].tolist()
    if len(tables) == 0: return [], []

    res_tables = []
    res_horlines = []
    for table in tables:
            
        x_min = int(table[0] - (table[2]-table[0])*0.02)
        y_min = int(table[1] - (table[3]-table[1])*0.02)
        x_max = int(table[2] + (table[2]-table[0])*0.02)
        y_max = int(table[3] + (table[3]-table[1])*0.02)
        table_img = image[y_min:y_max, x_min:x_max]
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find contours and remove text inside cells
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 4000:
                cv2.drawContours(thresh, [c], -1, 0, -1)
        horizontal = thresh.copy()
        vertical = thresh.copy()

        # [horiz]
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))

        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)

        horizontal = cv2.dilate(horizontal, (15,1), iterations=5)
        horizontal = cv2.erode(horizontal, (15,1), iterations=5)
        # cv2.namedWindow("horizontal", cv2.WINDOW_NORMAL)
        # cv2.imshow('horizontal', horizontal)
        hor_lines = cv2.HoughLinesP(horizontal,rho=1,theta=np.pi/180,threshold=80,minLineLength=100,maxLineGap=10)
        
        if hor_lines is not None:
        
            kernel_len = gray.shape[1]//100
            hor_lines = group_h_lines(hor_lines, kernel_len)
            # print(hor_lines, len(hor_lines))
            temp_line = []
            for line in hor_lines:
                x1,y1,x2,y2 = line
                temp_line.append([x1,y1,x2,y2])
            #Sorted lines according to y values
            hor_lines = sorted(temp_line,key=lambda x: x[1])
            
            print("len(hor_lines)", len(hor_lines))



        # for x1, y1, x2, y2 in hor_lines:
        #     cv2.line(table_img, (x1,y1), (x2,y2), (0, 255, 0), 3)

        # [vert]
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))

        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        vertical = cv2.dilate(vertical, (1,15), iterations=5)
        vertical = cv2.erode(vertical, (1,15), iterations=5)                                  
        # cv2.namedWindow("vertical", cv2.WINDOW_NORMAL)
        # cv2.imshow('vertical', vertical)
        #Detecting vertical lines using HoughLinesP() function
        ver_lines = cv2.HoughLinesP(vertical,rho=1,theta=np.pi/180,threshold=80,minLineLength=50,maxLineGap=10)

        if ver_lines is not None:
            kernel_len = gray.shape[0]//90
            ver_lines = group_v_lines(ver_lines, kernel_len)
            temp_line = []
            for line in ver_lines: 
                x1,y1,x2,y2 = line
                temp_line.append([x1,y1,x2,y2])
            ver_lines = sorted(temp_line,key=lambda x: x[0])
            print("len(ver_lines)", len(ver_lines))

        if ver_lines is not None and hor_lines is not None:
            table_coord = get_table_coordinate(hor_lines, ver_lines)
            table_coord[0] += x_min
            table_coord[1] += y_min
            table_coord[2] += x_min
            table_coord[3] += y_min
            res_tables.append(table_coord)
            ###
            for i in range(len(hor_lines)):
                hor_lines[i][0] += x_min
                hor_lines[i][1] += y_min
                hor_lines[i][2] += x_min
                hor_lines[i][3] += y_min
            res_horlines.append(hor_lines)
        # cv2.namedWindow("hv", cv2.WINDOW_NORMAL)
        # cv2.imshow('hv', horizontal + vertical)
    return res_tables, res_horlines



def detect_box_in_table(boxes, tables):
    
    outside_boxes = boxes.copy()
    # table_boxes = [[[]*(len(horlines) + 1)] for _ in range(len(tables))]
    table_boxes = [[] for _ in range(len(tables))]
    print("len(tables)", len(tables))
    # res_boxes = [[]*(len(horlines) + 1)]
    
    for i, table in enumerate(tables):
        t_x_min, t_y_min, t_x_max, t_y_max = table
        # print(t_x_min, t_y_min, t_x_max, t_y_max)
        for box in boxes:
            # print(box)
            # num_line = 0
            # pre_line_y = t_y_min
            # while num_line < len(horlines):
            #     l_x_min, l_y_min, l_x_max, l_y_max = horlines[num_line]
            #     if  pre_line_y < (box[1]+box[3])/2 < l_y_min:
            #         table_boxes[i][num_line].append(box)
            #         outside_boxes.remove(box)
            #     num_line += 1

            iou = utility.bb_ratio_intersection(boxA=(t_x_min, t_y_min, t_x_max, t_y_max), boxB=box)
            # print(iou)
            if iou > 0.8:
                table_boxes[i].append(box)
                outside_boxes.remove(box)
    return outside_boxes, table_boxes

def save_pdf_to_image(pdf_file, out_dir):
    with tempfile.TemporaryDirectory() as path:
        images_from_path = pdf2image.convert_from_path(pdf_file, output_folder=out_dir, paths_only=True, dpi=300, fmt="png", thread_count=4, output_file="pdf")
    image_files = os.listdir(out_dir)
    image_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

    return image_files

def process_image_file(img_file, save_text=False, dir_out_txt="", type="pdf"):
    print(img_file)
    img = cv2.imread(img_file)
    img_draw_after = img.copy()
    ## detect text
    dt_boxes, elapse = textDetect(img)

    boxes = []
    for i, bbox in enumerate(dt_boxes):
        p1, p2, p3, p4 = bbox
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        xmin = int(min(x1, x4))
        xmax = int(max(x2, x3))
        ymin = int(min(y1, y2))
        ymax = int(max(y3, y4))
        boxes.append([xmin, ymin, xmax, ymax])

        #################
        # img_draw_after = cv2.rectangle(img_draw_after, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        #################
    #################
    # for i, bbox in enumerate(boxes):
    #     xmin, ymin, xmax, ymax = bbox
    #     img_draw_after = cv2.putText(img_draw_after, str(i), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    #     img_draw_after = cv2.rectangle(img_draw_after, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    # cv2.namedWindow("table", cv2.WINDOW_NORMAL)
    # cv2.imshow("table", img_draw_after)
    
    ###################
    boxes = utility.preprocess_boxes(img, boxes, img.shape[1], img.shape[0], type)

    min_box_height, idx = utility.calc_min_box_height(boxes)
    
    tables, res_horlines = get_tables(img)
    # print("tables", tables)
    # print("res_horlines", res_horlines)
    boxes, table_boxes = detect_box_in_table(boxes, tables)

    
    # print(boxes)
    
    print("min_box_height", min_box_height)
    boxes.sort(key = lambda x: x[0])
    boxes = utility.merge_boxes(boxes, min_box_height)
    boxes.sort(key = lambda x: x[1])

    for i in range(len(table_boxes)):
        table_boxes[i].sort(key = lambda x: x[0])
        table_boxes[i] = utility.merge_boxes(table_boxes[i], min_box_height)
        table_boxes[i].sort(key = lambda x: x[1])


    ordinal_number_boxes = utility.merge_text_line_by_line(boxes.copy(), min_box_height)

    list_imgs = []
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        xmin = int(xmin - (xmax-xmin)*0.05)
        xmax = int(xmax + (xmax-xmin)*0.05)
        ymin = int(ymin - (ymax-ymin)*0.06)
        ymax = int(ymax + (ymax-ymin)*0.06)
        img_crop = Image.fromarray(img[ymin: ymax, xmin: xmax])
        # img_crop.show()
        ##################
        # out_path_crop = out_crop + file.split(".")[0] + "__" + str(i) + ".jpg"
        # cv2.imwrite(out_path_crop, img_crop)
        ##################
        list_imgs.append(img_crop)
    
    #######################
    table_list_imgs = []
    table_list_boxes = []
    for table_box in table_boxes:
        for box in table_box:
            xmin, ymin, xmax, ymax = box
            table_list_boxes.append([xmin, ymin, xmax, ymax])
            xmin = int(xmin - (xmax-xmin)*0.05)
            xmax = int(xmax + (xmax-xmin)*0.05)
            ymin = int(ymin - (ymax-ymin)*0.06)
            ymax = int(ymax + (ymax-ymin)*0.06)
            img_crop = img[ymin: ymax, xmin: xmax]
            table_list_imgs.append(Image.fromarray(img_crop))
            
            # img_draw_after = cv2.putText(img_draw_after, str(i), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            # img_draw_after = cv2.rectangle(img_draw_after, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
   
    #######################
    if len(list_imgs) + len(table_list_imgs) == 0: return

    result_s, list_batch_numbers = detector.batch_predict(list_imgs + table_list_imgs, list(range(len(list_imgs) + len(table_list_imgs))))
    

    zipped_lists = zip(list_batch_numbers, result_s)
    sorted_zipped_lists = sorted(zipped_lists)
    list_batch_numbers, result_s  = zip(*sorted_zipped_lists)
    # print(result_s)
    # print(list_batch_numbers)

    list_str_lines, list_box_lines = [], []
    with open(dir_out_txt + img_file.split("/")[-1].split(".")[0] + ".txt", "w") as f:#, open("pdf_text_crop.txt", "a") as f_crop:
        for key, value in ordinal_number_boxes.items():
            
            min_x, min_y, max_x, max_y = 99999, 99999, -1, -1
            str_line = ""
            # print(key, value)
            for i, v in enumerate(value):
                
                idx, box = v

                str_line += result_s[list_batch_numbers.index(idx)].strip()
                if i < len(value) - 1:
                    str_line += " "
                
                min_x = min(min_x, box[0])
                min_y = min(min_y, box[1])
                max_x = max(max_x, box[2])
                max_y = max(max_y, box[3])

                #########
                # f_crop.write(out_crop + file.split(".")[0] + "__" + str(idx) + ".jpg" + "\t" + result_s[list_batch_numbers.index(idx)] + "\n")
                #########
            list_str_lines.append(str_line)
            list_box_lines.append([min_x, min_y, max_x, max_y])
            
            f.write(str_line + "\n")

        # print(len(list_imgs))
        table_strs = [[] for _ in range(len(table_boxes))]
        
        for i, table in enumerate(table_boxes):
            for j, box in enumerate(table):
        # for i, string in enumerate(result_s[len(list_imgs):]):
                min_x, min_y, max_x, max_y = box
                string = result_s[len(list_imgs):][i+j]
                table_strs[i].append(string)
                # f.write(string +  "\t" + str([min_x, min_y, max_x, max_y]) + "\t" + str(img.shape[1]) + "\t" +str(img.shape[0])+ "\n")
                # img_draw_after = cv2.putText(img_draw_after, string, (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                # img_draw_after = cv2.rectangle(img_draw_after, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)
            
            # for l in res_horlines[i]:
            #     img_draw_after = cv2.line(img_draw_after, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 3)
    # cv2.namedWindow("table", cv2.WINDOW_NORMAL)
    # cv2.imshow("table", img_draw_after)
    
    return list_str_lines, list_box_lines, table_boxes, table_strs, res_horlines, tables

def compare(str_lines1, box_lines1, w1, h1, str_lines2, box_lines2, w2, h2):

    is_compare_list = []
    count_true = 0
    for i, (s1, l1) in enumerate(zip(str_lines1, box_lines1)):

        tl_x, tl_y, br_x, br_y = l1
        tl_xy_tmp = utility.new_coordinates_after_resize_img(original_size=(w1, h1), new_size=(w2, h2), original_coordinate=(tl_x, tl_y))
        br_xy_tmp = utility.new_coordinates_after_resize_img(original_size=(w1, h1), new_size=(w2, h2), original_coordinate=(br_x, br_y))

        for j, (s2, l2) in enumerate(zip(str_lines2, box_lines2)):
            if j not in is_compare_list:
                iou = utility.bb_intersection_over_union(boxA=(tl_xy_tmp[0], tl_xy_tmp[1], br_xy_tmp[0], br_xy_tmp[1]), boxB=l2)
                if iou > 0.7:
                    is_compare_list.append(j)
                    if s1 == s2:
                        count_true += 1
        

    
    return count_true / max(len(str_lines1), len(str_lines2))


def merge_box_row_tables(table_list_boxes, table_list_strings, res_horlines, tables):
    list_row_boxes = [[[] for _ in range(len(res_horlines[i])+2)] for i in range(len(res_horlines))]
    
    for i, table in enumerate(tables):
        # t_x_min, t_y_min, t_x_max, t_y_max = table
        num_line = 0
        pre_line_y = table
        is_check_box = [False]*len(table_list_boxes[i])
        while num_line < len(res_horlines[i]):
            l_x_min, l_y_min, l_x_max, l_y_max = res_horlines[i][num_line]
            # print(i, res_horlines[i][num_line])
            for j, (box, str) in enumerate(zip(table_list_boxes[i], table_list_strings[i])):
                # iou = utility.bb_intersection_over_union(boxA=(pre_line_y[0], pre_line_y[1], res_horlines[i][num_line][2], res_horlines[i][num_line][3]), boxB=box)
                # if  pre_line_y < (box1[1]+box1[3])/2 < l_y_min:
                # print((pre_line_y[0], pre_line_y[1], res_horlines[i][num_line][2], res_horlines[i][num_line][3]), box, iou)
                # if iou > 0.8:
                # print((box, str), is_check_box[j] is False and (pre_line_y[1] < (box[1]+box[3])/2 < l_y_min))
                if is_check_box[j] is False and (pre_line_y[1] < (box[1]+box[3])/2 < l_y_min):
                    list_row_boxes[i][num_line].append((box, str))
                    is_check_box[j] = True
            
            pre_line_y = res_horlines[i][num_line]
            num_line += 1

        #check end line table
        for j, (box, str) in enumerate(zip(table_list_boxes[i], table_list_strings[i])): 
            # if utility.bb_intersection_over_union(boxA=(pre_line_y[0], pre_line_y[1], table[2], table[3]), boxB=box) > 0.8:
            if is_check_box[j] is False and (pre_line_y[1] < (box[1]+box[3])/2 < table[1]): 
                list_row_boxes[i][num_line].append((box, str))
                is_check_box[j] = True
        #######
    # print("AAAAAAAAAAAAAAA")
    for i, row_box in enumerate(list_row_boxes):
        list_row_boxes[i] = list(filter(lambda x: x != [], list_row_boxes[i]))

    # for i, row_box in enumerate(list_row_boxes):
    #     for j, row in enumerate(row_box):
    #         print(row, j)

    return list_row_boxes
    
def sort_row_box(row_box):
    '''
    row_box: list of (box, str)
    '''
    # print("row_box", row_box)
    row_box = sorted(row_box, key=lambda x: (x[0][0]+x[0][2])/2)
    min_box = min(row_box, key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]))

    for i in range(len(row_box)):
        for j in range(i+1, len(row_box)):
            if abs((row_box[i][0][0]+row_box[i][0][2])/2 - (row_box[j][0][0]+row_box[j][0][2])/2) < (min_box[0][2]-min_box[0][0])/2:
                # row_box_res.append()
                if (row_box[i][0][1]+row_box[i][0][3])/2 > (row_box[j][0][1]+row_box[j][0][3])/2:
                    row_box[i], row_box[j] = row_box[j], row_box[i]
    
    return row_box

def check_diff_char_in_2_strings(str1, str2):
    str1 = list(str1)
    str2 = list(str2)
    res = []
    for s1 in str1:
        if s1 in str2:
            str2.remove(s1)
        else:
            res.append(s1)
    return res + str2

import editdistance
def compare_tables(table_list_boxes1, table_list_strings1, res_horlines1, tables1, table_list_boxes2, table_list_strings2, res_horlines2, tables2, file1, file2, img_scan):
    
    
    list_row_boxes1 = merge_box_row_tables(table_list_boxes1, table_list_strings1, res_horlines1, tables1)
    list_row_boxes2 = merge_box_row_tables(table_list_boxes2, table_list_strings2, res_horlines2, tables2)
    # cv2.waitKey(0)
    count_row, count_ok = 0, 0
    
    for row_boxes1, row_boxes2 in zip(list_row_boxes1, list_row_boxes2):
        
        with open(file1, "a") as f1, open(file2, "a") as f2:
            f1.write("\n\n\n")
            for i, row_box1 in enumerate(row_boxes1):
                row_box1 = sort_row_box(row_box1)
                row_str1 = " ".join([_[1] for _ in row_box1])
                # print(row_str1)
                f1.write(row_str1 + "\n")
            print("++++++++++++++++++++++")
            f2.write("\n\n\n")
            for i, row_box2 in enumerate(row_boxes2):
                row_box2 = sort_row_box(row_box2)
                row_str2 = " ".join([_[1] for _ in row_box2])
                # print(row_str2)
                f2.write(row_str2 + "\n")

        is_check = [False]*len(row_boxes1)
        for i, row_box2 in enumerate(row_boxes2):
            row_box2 = sort_row_box(row_box2)
            row_str2 = " ".join([_[1] for _ in row_box2])

            is_ok = False
            for j in range(max(0, i-2), min(len(row_boxes1), i+2)):
                row_box1 = sort_row_box(row_boxes1[j])
                row_str1 = " ".join([_[1].strip() for _ in row_box1])

                # if is_check[j] is False and row_str1 == row_str2:
                if is_check[j] is False:
                    # diff_words = list(set(row_str2.split()) - set(row_str1.split())) + list(set(row_str1.split()) - set(row_str2.split()))

                    if row_str1 == row_str2:
                        count_ok += 1
                        is_check[j] = True
                        is_ok = True
                        break

                    distance = editdistance.eval(row_str2, row_str1)
                    if distance <=3:
                        diff_chars = check_diff_char_in_2_strings(row_str2, row_str1)
                        is_contain_num = False
                        for num in "0123456789":
                            if num in diff_chars:
                                is_contain_num = True
                                break
                        if is_contain_num is False:
                            count_ok += 1
                            is_check[j] = True
                            is_ok = True
                            break
                        

                    # if len(diff_words) == 1 and len(diff_words[0]) == 1 and not diff_words[0].isdigit():
                    #     count_ok += 1
                    #     is_check[j] = True
                    #     is_ok = True
                    #     break
                    # elif len(diff_words) == 2:
                    #     diff_chars = check_diff_char_in_2_strings(diff_words[0], diff_words[1])
                    #     if len(diff_chars) == 1 and not diff_chars[0].isdigit():
                    #         count_ok += 1
                    #         is_check[j] = True
                    #         is_ok = True
                    #         break
            
            if is_ok is False:
                min_x = min([_[0][0] for _ in row_box2])
                min_y = min([_[0][1] for _ in row_box2])
                max_x = max([_[0][2] for _ in row_box2])
                max_y = max([_[0][3] for _ in row_box2])
                img_scan = cv2.rectangle(img_scan, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)

        count_row += max(len(row_boxes1), len(row_boxes2))
    
    return count_ok, count_row

    
if __name__ == "__main__":

    pdf_file = "/home/ancv/Desktop/ori.pdf"
    out_dir = "/home/ancv/Work/ocr_pdf/img_pdf/"
    out_txt = "/home/ancv/Work/ocr_pdf/text_pdf/"
    # image_files = save_pdf_to_image(pdf_file, out_dir)

    scan_pdf_file = "/home/ancv/Desktop/scan.pdf"
    scan_out_dir = "/home/ancv/Work/ocr_pdf/img_scan/"
    scan_out_txt = "/home/ancv/Work/ocr_pdf/text_scan/"
    # scan_image_files = save_pdf_to_image(scan_pdf_file, scan_out_dir)

    # is_ok = False
    # for file in os.listdir(out_dir):
    #     # if file != "pdf0002-23.png": 
    #     #     is_ok = True
    #     #     continue
    #     # if is_ok:
    #     process_image_file(out_dir + file, dir_out_txt=out_txt, type="pdf")
    #    # break
    pdf = list(range(1, 22)) + list(range(24, 41)) + list(range(42, 61))
    scan = list(range(3, 24)) + list(range(27, 30)) + list(range(31, 34)) + list(range(35, 46)) + list(range(47, 66))
    list_pdf = ["img_pdf/" + i for i in os.listdir("img_pdf")]
    list_scan = ["img_scan/" + i for i in os.listdir("img_scan")]

    average_acc =  0
    for d, s in zip(pdf, scan):
        file_pdf = ""
        file_scan = ""
        for lf in list_pdf:
            if int(lf.split(".")[0].split("-")[-1]) == d:
                file_pdf = lf
        for ls in list_scan:
            if int(ls.split(".")[0].split("-")[-1]) == s:
                file_scan = ls
        # if file_pdf != "img_pdf/pdf0004-60.png": continue

        img_scan = cv2.imread(file_scan)

        list_str_lines1, list_box_lines1, table_list_boxes1, table_list_strings1, res_horlines1, tables1 = process_image_file(file_pdf, dir_out_txt=out_txt, type='pdf')
        list_str_lines2, list_box_lines2, table_list_boxes2, table_list_strings2, res_horlines2, tables2 = process_image_file(file_scan, dir_out_txt=scan_out_txt, type='scan')
        

        count_table_ok, count_table_row = compare_tables(table_list_boxes1, table_list_strings1, res_horlines1, tables1, table_list_boxes2, table_list_strings2, res_horlines2, tables2, \
                file1= out_txt + file_pdf.split("/")[-1].split(".")[0] + ".txt", \
                file2= scan_out_txt + file_scan.split("/")[-1].split(".")[0] + ".txt", img_scan=img_scan
            )

        count_ok = 0
        is_check = [False]*len(list_str_lines1)
        for i, (str2, box2) in enumerate(zip(list_str_lines2, list_box_lines2)):
            is_ok = False
            for j, (str1, box1) in enumerate(zip(list_str_lines1, list_box_lines1)):
                # if is_check[j] is False and str1 == str2:\
                if is_check[j] is False:
                    if " ".join(str1.split()) == " ".join(str2.split()):
                        count_ok += 1
                        is_check[j] = True
                        is_ok = True
                        break

                    distance = editdistance.eval(" ".join(str2.split()), " ".join(str1.split()))
                    if distance <=3:
                        diff_chars = check_diff_char_in_2_strings(str2, str1)
                        is_contain_num = False
                        for num in "0123456789":
                            if num in diff_chars:
                                is_contain_num = True
                                break
                        if is_contain_num is False:
                            count_ok += 1
                            is_check[j] = True
                            is_ok = True
                            break

                    # diff_words = list(set(str1.split()) - set(str2.split())) + list(set(str2.split()) - set(str1.split()))
                    # if len(diff_words) == 1 and len(diff_words[0]) == 1 and not diff_words[0].isdigit():
                    #     count_ok += 1
                    #     is_check[j] = True
                    #     is_ok = True
                    #     break
                    # elif len(diff_words) == 2:
                    #     diff_chars = check_diff_char_in_2_strings(diff_words[0], diff_words[1])
                    #     if len(diff_chars) == 1 and not diff_chars[0].isdigit():
                    #         count_ok += 1
                    #         is_check[j] = True
                    #         is_ok = True
                    #         break
            if is_ok is False:
                min_x = box2[0]
                min_y = box2[1]
                max_x = box2[2]
                max_y = box2[3]
                img_scan = cv2.rectangle(img_scan, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)

        cv2.imwrite("/home/ancv/Work/ocr_pdf/saved_dir_scan/" + file_scan.split("/")[-1], img_scan)
        
        print(file_pdf, file_scan, (count_ok + count_table_ok)/(count_table_row + max(len(list_str_lines1), len(list_str_lines2))))

        average_acc += (count_ok + count_table_ok)/(count_table_row + max(len(list_str_lines1), len(list_str_lines2)))
        # break
    print("average_acc", average_acc/len(pdf))
    # cv2.waitKey(0)
    # for file in os.listdir(scan_out_dir):
    #     if file != "pdf0001-09.png": continue
    #     process_image_file(scan_out_dir + file, dir_out_txt=scan_out_txt, type='scan')

    