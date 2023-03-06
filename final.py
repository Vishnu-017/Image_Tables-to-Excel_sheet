'''import cv2
import layoutparser as lp
image = cv2.imread("bd2.png")
image = image[..., ::-1]

# load model
model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                                threshold=0.5,
                                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                enforce_cpu=False,
                                enable_mkldnn=True)
# detect
layout = model.detect(image)

# show result
show_img = lp.draw_box(image, layout, box_width=3, show_element_type=True)
show_img.show()'''

from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
ocr = PaddleOCR(lang = 'en')
image_path = 'D:\\Python\\healthcare\\bd2.png'
image_cv = cv2.imread(image_path)
image_height = image_cv.shape[0]
image_width = image_cv.shape[1]
print(image_height,image_width)
output = ocr.ocr(image_path)
print(output)
print(len(output))
arr2d = np.array(output)
arr1d = arr2d.flatten()
print(arr1d[1]) # appednd as it is 
boxes=[]
for out in range(0,len(arr1d),2):
    boxes.append(arr1d[out])
    print(arr1d[out])
#boxes=[line[0] for line in output]
#print(boxes)
texts=[]
#texts = [line[1][0] for line in output]
for out in range(1,len(arr1d),2):
    texts.append(arr1d[out])
    print(arr1d[out])
#probabilities = [line[1][1] for line in output]
probabilities=[]
for out in range(1,len(arr1d),2):
    probabilities.append(arr1d[out][1])
    print(arr1d[out][1])
    
image_boxes = image_cv.copy()
#print(image_boxes)
#print(result[0][0],result[0][2])
for box,text in zip(boxes,texts):
    #print(int(result[0]))
    cv2.rectangle(image_boxes, (int(box[0][0]),int(box[0][1])), (int(box[2][0]),int(box[2][1])), (0,0,255),1)
    cv2.putText(image_boxes,text[0],(int(box[0][0]),int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),0)

cv2.imwrite('detections.jpg',image_boxes) 
#Reconstruction
im = image_cv.copy()
horiz_boxes =[]
vert_boxes = []
for box in boxes:
    x_h, x_v = 0,int(box[0][0])
    y_h,y_v=int(box[0][1]),0
    
    wid_h,wid_v=image_width,int(box[2][0]-box[0][0])
    hei_h,hei_v = int(box[2][1]-box[0][1]),image_height
    
    horiz_boxes.append([x_h,y_h,x_h+wid_h,y_h+hei_h]) 
    vert_boxes.append([x_v,y_v,x_v+wid_v,y_v+hei_v])
    cv2.rectangle(im,(x_h,y_h),(x_h+wid_h,y_h+hei_h),(255,0,0),1)
    cv2.rectangle(im,(x_v,y_v),(x_v+wid_v,y_v+hei_v),(0,255,0),1)
    
print(horiz_boxes[0])
cv2.imwrite('hori_vert.jpg',im)

import tensorflow as tf
horiz_out = tf.image.non_max_suppression(
    horiz_boxes,
    probabilities,
    max_output_size = 1000,
    iou_threshold = 0.1,
    score_threshold = float('-inf'),
    name=None
)
horiz_lines = np.sort(np.array(horiz_out))
print(horiz_lines)
im_nms = image_cv.copy()
for val in horiz_lines:
    cv2.rectangle(im_nms, (int(horiz_boxes[val][0]),int(horiz_boxes[val][1])), (int(horiz_boxes[val][2]),int(horiz_boxes[val][3])), (0,0,255),1)



vert_out = tf.image.non_max_suppression(
    vert_boxes,
    probabilities,
    max_output_size=1000,
    iou_threshold=0.1,
    score_threshold=float('-inf'),
    name=None
)
vert_lines = np.sort(np.array(vert_out))
print(vert_lines)

for val in vert_lines:
    cv2.rectangle(im_nms,(int(vert_boxes[val][0]),int(vert_boxes[val][1])), (int(vert_boxes[val][2]),int(vert_boxes[val][3])), (255,0,255),1)
cv2.imwrite('im_nms.jpg',im_nms) 

#to csv
out_array = [[""for i in range(len(vert_lines))] for j in range(len(horiz_lines)) ]
print(np.array(out_array).shape)
print(out_array)

unordered_boxes = []
for i in vert_lines:
    print(vert_boxes[i])
    unordered_boxes.append(vert_boxes[i][0])
print(unordered_boxes)
ordered_boxes = np.argsort(unordered_boxes)
print(ordered_boxes)


def intersection(box_1,box_2):
    return [box_2[0],box_1[1],box_2[2],box_1[3]]

def iou(box_1,box_2):
    x_1 = max(box_1[0],box_2[0])
    y_1=max(box_1[1],box_2[1])
    x_2 = min(box_1[2],box_2[2])
    y_2 = min(box_1[3],box_2[3])
    
    inter =abs(max((x_2 - x_1,0)) * max((y_2-y_1),0))
    if inter == 0:
        return 0
    box_1_area = abs((box_1[2]-box_1[0]) * (box_1[3] - box_1[1]))
    box_2_area = abs((box_2[2]-box_2[0]) * (box_2[3] - box_2[1]))
    return inter / float(box_1_area+box_2_area - inter)

for i in range(len(horiz_lines)):
    for j in range(len(vert_lines)):
        resultant = intersection(horiz_boxes[horiz_lines[i]],vert_boxes[vert_lines[ordered_boxes[j]]])
        for b in range(len(boxes)):
            the_boxes = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
            if(iou(resultant,the_boxes)>0.1):
                out_array[i][j] = texts[b][0]
        #print(resultant)
print(out_array)
import pandas as pd
pd.DataFrame(out_array).to_csv('healthcare.csv')
        