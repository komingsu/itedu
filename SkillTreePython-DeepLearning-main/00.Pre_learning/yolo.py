import cv2
import numpy as np
import matplotlib.pyplot as plt
import IPython

def set_model(model_folder_path):
    weight_file =  model_folder_path + "yolov3.weights"
    cfg_file = model_folder_path+"yolov3.cfg"
    model = cv2.dnn.readNet(weight_file, cfg_file)
    predict_layer_names = [model.getLayerNames()[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    num_model = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_russian_plate_number.xml')
    name_file = model_folder_path+"coco.names"
    with open(name_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, predict_layer_names, class_names, num_model


def get_preds(img, model, predict_layer_names, min_confidence=0.5, threshold=0.4):
    img_h, img_w, img_c = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    preds_each_layers = model.forward(predict_layer_names)

    boxes = []
    confidences=[]
    class_ids = []

    for preds in preds_each_layers:
        for pred in preds:
            box, confidence, class_id = pred[:4], pred[4], np.argmax(pred[5:])
            if confidence > min_confidence:
                x_center, y_center, w, h = box
                x_center, w = int(x_center*img_w), int(w*img_w)
                y_center, h = int(y_center*img_h), int(h*img_h)
                x, y = x_center-int(w/2), y_center-int(h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence)) # float 처리를 해야 NMSBoxes 함수 사용 가능
                class_ids.append(class_id)

    # 중복 제거 index
    idxs = cv2.dnn.NMSBoxes(boxes,confidences,min_confidence, threshold)

    # 중복 제거
    boxes = list(map(lambda x: boxes[x[0]], idxs))
    confidences = list(map(lambda x: confidences[x[0]], idxs))
    class_ids = list(map(lambda x: class_ids[x[0]], idxs))

    # img를 벗어나는 박스 수정
    for i in boxes:
        if i[0]<0: i[0]=0

    return boxes, confidences, class_ids


def draw_img_result(img, boxes, num_model):
    for x,y,w,h in boxes:
        croped = img[y:y+h, x:x+w]
        pred_num = num_model.detectMultiScale(croped)

        for (x1,y1,w1,h1) in pred_num:
            cv2.rectangle(img, (x+x1,y+y1), (x+x1+w1, y+y1+h1), (0,255,0), 2)
    
    return img

def img2detect(img, model, predict_layer_names, num_model):
    boxes, confidences, class_ids = get_preds(img, model, predict_layer_names)
    img = draw_img_result(img, boxes, num_model)
    cv2.imshow("img", img)