import cv2
import numpy as np

def load_yolo():
    net = cv2.dnn.readNet(r"C:\Users\Guest_User\Downloads\archive (1)\yolov3.weights", r"C:\Users\Guest_User\Downloads\archive (1)\yolov3.cfg")
    with open(r"C:\Users\Guest_User\Downloads\archive (1)\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, output_layers, colors

def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            if len(detection) >= 5:
                scores = detection[5:]
                if scores.size > 0:
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

def draw_labels(img, boxes, confidences, class_ids, indexes, classes, colors):
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
    return img

def process_image(image_path, net, output_layers, classes, colors):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image file.")
        return
    boxes, confidences, class_ids, indexes = detect_objects(img, net, output_layers)
    img = draw_labels(img, boxes, confidences, class_ids, indexes, classes, colors)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_webcam(net, output_layers, classes, colors):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
        boxes, confidences, class_ids, indexes = detect_objects(frame, net, output_layers)
        frame = draw_labels(frame, boxes, confidences, class_ids, indexes, classes, colors)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:  # Press Esc to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    net, classes, output_layers, colors = load_yolo()
    mode = input("Enter 'webcam' to use webcam or 'image' to use an image file: ").strip().lower()
    if mode == 'webcam':
        process_webcam(net, output_layers, classes, colors)
    elif mode == 'image':
        image_path = input("Enter the path to the image file: ").strip()
        process_image(image_path, net, output_layers, classes, colors)
    else:
        print("Invalid mode selected. Please enter 'webcam' or 'image'.")

if __name__ == "__main__":
    main()
