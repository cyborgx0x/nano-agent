from ultralytics import YOLO
import math

model = YOLO('yolov8n.pt')
model = YOLO('model/albion/fiber.pt')

# results = model.predict('screen 1920 0 1920 1080', stream=True, vid_stride=True)


def game_detection(image):

    results = model.predict(image)
    return run(results[0])

def run(result):
    
    if len(result.boxes.xyxy) != 0:
        print(result.boxes.xyxy)
        k0,j0 = (1920/2, 1080/2)
        x0,y0 = (1920, 1080)
        nearest = 0
        boxes_data = result.boxes.xyxy

        for index, bbox in enumerate(boxes_data):
            
            if result.boxes.conf[index].item() < 0.7:
                continue
            x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()

            k1,j1 = math.ceil((x1+x2)/2), math.ceil((y1+y2)/2)
            
            if (k1-k0)**2 + (j1-j0)**2 < (x0-k0)**2 + (y0-k0)**2:
                x0,y0=k1,j1
                nearest = index
        '''
        after get the nearest point, move the mouse and click :)
        '''
        print(nearest, x0,y0)
        bbox = result.boxes.xyxy[nearest]
        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
        return dict(
            data = (x1,y1,x2,y2)
        )
    else:
        return dict(
            data=None
        )