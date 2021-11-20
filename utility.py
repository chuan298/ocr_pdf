import argparse
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
from paddle import inference
from collections import defaultdict
def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    use_gpu = 1
    
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=500)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str, default="./models/detect")
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=bool, default=False)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)

    return parser.parse_args()


def create_predictor(args, mode, logger):
    if mode == "det":
        model_dir = args.det_model_dir
    elif mode == 'cls':
        model_dir = args.cls_model_dir
    else:
        model_dir = args.rec_model_dir

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    model_file_path = model_dir + "/inference.pdmodel"
    params_file_path = model_dir + "/inference.pdiparams"
    if not os.path.exists(model_file_path):
        logger.info("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        logger.info("not find params file path {}".format(params_file_path))
        sys.exit(0)

    config = inference.Config(model_file_path, params_file_path)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=inference.PrecisionType.Half
                if args.use_fp16 else inference.PrecisionType.Float32,
                max_batch_size=args.max_batch_size)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(6)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            #  TODO LDOUBLEV: fix mkldnn bug when bach_size  > 1
            #config.set_mkldnn_op({'conv2d', 'depthwise_conv2d', 'pool2d', 'batch_norm'})
            args.rec_batch_num = 1

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.switch_use_feed_fetch_ops(False)

    # create predictor
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image



####################################### post processing detect

def merge_text_line_by_line(boxes, min_box_height):
    num_line = 0
    last_merge = -1
    ordinal_number_boxes = defaultdict(list)
    for i in range(len(boxes)):

        if i <= last_merge:
            continue
        num_line += 1
        ordinal_number_boxes[num_line].append((i, boxes[i]))
        
        
        for j in range(i+1, len(boxes)):
            center_i = ((boxes[i][2] + boxes[i][0])/2, (boxes[i][3] + boxes[i][1])/2)
            center_j = ((boxes[j][2] + boxes[j][0])/2, (boxes[j][3] + boxes[j][1])/2)
                        
            if abs(center_j[1]-center_i[1]) < min_box_height/1.3:
                last_merge = j
                x1 = min(boxes[i][0], boxes[j][0])
                y1 = min(boxes[i][1], boxes[j][1])
                x2 = max(boxes[i][2], boxes[j][2])
                y2 = min(boxes[i][3], boxes[j][3])
                boxes[i] = [x1, y1, x2, y2]
                ordinal_number_boxes[num_line].append((j, boxes[j]))



    for i in ordinal_number_boxes:
        ordinal_number_boxes[i].sort(key=lambda x: x[1][0])

    return ordinal_number_boxes


def detect_black_text(img):

    # converting from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0,0,0])
    upper1 = np.array([179,255,40])
    lower2 = np.array([0,0,0])
    upper2 = np.array([179,40,100])
    # lower3 = np.array([0,0,0])
    # upper3 = np.array([50,50,127])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    # img_res = cv2.bitwise_and(img, img, mask = mask)
    # Refining the mask corresponding to the detected red color
    # mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
    # mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 1)
    # mask2 = cv2.bitwise_not(mask1)

	# Generating the final output
	# res1 = cv2.bitwise_and(background,background,mask=mask1)
	# res2 = cv2.bitwise_and(img,img,mask=mask2)
	# final_output = cv2.addWeighted(res1,1,res2,1,0)


    # img_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    # contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # # if len(contours) != 0:
    # #     # draw in blue the contours that were founded
    # #     cv2.drawContours(red, contours, -1, 255, 3)

    # #     # find the biggest countour (c) by the area
    # #     c = max(contours, key = cv2.contourArea)
    # #     x,y,w,h = cv2.boundingRect(c)

    # #     # draw the biggest contour (c) in green
    # #     cv2.rectangle(red,(x,y),(x+w,y+h),(0,255,0),2)
    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     cv2.rectangle(red,(x,y),(x+w,y+h),(0,255,0),2)

    # show the images
    # cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    # cv2.imshow("Result", mask)

    # cv2.waitKey(0)
    total_pixels = hsv.size/3
    black_ratio =(cv2.countNonZero(mask))/total_pixels

    return black_ratio


def preprocess_boxes(img, boxes, w, h, type):
    res_boxes = []
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        if xmax-xmin >= 0.005*w and ymax-ymin >= 0.0065*h and (type == "pdf" or detect_black_text(img[ymin: ymax, xmin: xmax]) > 0.015):
            # xmin = int(xmin - (xmax-xmin)*0.05)
            # xmax = int(xmax + (xmax-xmin)*0.05)
            # ymin = int(ymin - (ymax-ymin)*0.06)
            # ymax = int(ymax + (ymax-ymin)*0.06)
            # if type == "scan":
            #     print(i, detect_black_text(img[ymin: ymax, xmin: xmax]))
            res_boxes.append([xmin, ymin, xmax, ymax])
    return res_boxes

# def calc_x_distance(box1, box2):
#     x_dist = min(abs(text_xmin-obj_xmin), abs(text_xmin-obj_xmax), abs(text_xmax-obj_xmin), abs(text_xmax-obj_xmax))

def calc_min_box_height(boxes):
    min_h = 9999
    idx = -1
    for i, box in enumerate(boxes):
        h = box[3]-box[1]
        if h < min_h:
            min_h = h
            idx = i
    return min_h, idx


def merge_boxes(boxes, min_box_height):

    res_boxes = []
    list_ok = []
    for i in range(len(boxes)):

        temp = boxes[i]
        
        for j in range(i+1, len(boxes)):
            center_i = ((temp[2] + temp[0])/2, (temp[3] + temp[1])/2)
            center_j = ((boxes[j][2] + boxes[j][0])/2, (boxes[j][3] + boxes[j][1])/2)
            
            # print(i, j, abs(center_j[1]-center_i[1]), min_box_height/1.3, temp[0] < boxes[j][2], temp[2] > boxes[j][0])
            
            if abs(center_j[1]-center_i[1]) < min_box_height/1.3 and temp[0] < boxes[j][2] and temp[2] > boxes[j][0]:
                # print("AAAAAAAAAAAA")
                xmin = min(boxes[j][0], temp[0])
                ymin = min(boxes[j][1], temp[1])
                xmax = max(boxes[j][2], temp[2])
                ymax = max(boxes[j][3], temp[3])
                temp = [xmin, ymin, xmax, ymax]
                list_ok.append(j)
        # print(i, list_ok)
        if i not in list_ok:
            res_boxes.append(temp)

    return res_boxes


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def bb_ratio_intersection(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxBArea)
	# return the intersection over union value
	return iou

def new_coordinates_after_resize_img(original_size, new_size, original_coordinate):
  original_size = np.array(original_size)
  new_size = np.array(new_size)
  original_coordinate = np.array(original_coordinate)
  xy = original_coordinate/(original_size/new_size)
  x, y = int(xy[0]), int(xy[1])
  return (x, y)


def remove_red_thing(inputImage):
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Convert the BGR image to HSV:
    hsvImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)

    # Create the HSV range for the red ink:
    # [128, 255, 255], [90, 50, 70]
    lowerValues1 = np.array([159, 50, 70])
    upperValues1 = np.array([180, 255, 255])
    lowerValues2 = np.array([0, 50, 70])
    upperValues2 = np.array([9, 255, 255])
    # Get binary mask of the blue ink:
    bluepenMask1 = cv2.inRange(hsvImage, lowerValues1, upperValues1)
    bluepenMask2 = cv2.inRange(hsvImage, lowerValues2, upperValues2)
    bluepenMask = cv2.bitwise_or(bluepenMask1, bluepenMask2)
    # Use a little bit of morphology to clean the mask:
    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 1
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    bluepenMask = cv2.morphologyEx(bluepenMask, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

    # Add the white mask to the grayscale image:
    colorMask = cv2.add(grayscaleImage, bluepenMask)
    _, binaryImage = cv2.threshold(colorMask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite('bwimage.jpg',binaryImage)
    thresh, im_bw = cv2.threshold(binaryImage, 210, 230, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    imgfinal = cv2.dilate(im_bw, kernel=kernel, iterations=1)

    return imgfinal