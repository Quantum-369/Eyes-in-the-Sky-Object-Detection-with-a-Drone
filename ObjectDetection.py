import pygame
from djitellopy import tello
import time
import cv2
import cvzone

thres = 0.55
nmsThres = 0.2
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



class TelloController:

    def __init__(self):
        pygame.init() #Initializes the pygame library and sets it up for use.
        self.screen = pygame.display.set_mode((640, 480)) #Creates a window with the dimensions of 640x480 pixels on the screen.
        # The window will be used to display the live video feed from the drone.
        pygame.display.set_caption("Tello Controller") #Sets the title of the window to "Tello Controller".
        self.drone = tello.Tello() #
        self.drone.connect()
        self.drone.streamon()
        self.img = None
        self.speed = 40
        self.recording = False

    def get_key_input(self):
        lr, fb, ud, yaw = 0, 0, 0, 0
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            lr = -self.speed
        elif keys[pygame.K_RIGHT]:
            lr = self.speed
        if keys[pygame.K_UP]:
            fb = self.speed
        elif keys[pygame.K_DOWN]:
            fb = -self.speed
        if keys[pygame.K_u]:
            ud = self.speed
        elif keys[pygame.K_d]:
            ud = -self.speed
        if keys[pygame.K_c]:
            yaw = -self.speed
        elif keys[pygame.K_a]:
            yaw = self.speed
        if keys[pygame.K_t]:
            self.drone.takeoff()
        elif keys[pygame.K_l]:
            self.drone.land()
        if keys[pygame.K_p]:
            self.save_image()
        if keys[pygame.K_h]:
            self.record_video()
        return [lr, fb, ud, yaw]

    def save_image(self):
        filename = f'DRONEIMAGES/{time.time()}.jpg'
        self.img= cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, self.img)
        print(f'Image saved as {filename}')
        time.sleep(0.2)  # avoids multiple image savings

    def record_video(self):
        filename = input("Enter the filename to save the video: ")
        codec = input("Enter the codec of the video (example: 'XVID'): ")
        resolution = input("Enter the resolution of the video (example: '720x480'): ")
        width, height = resolution.split("x")

        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(f'{filename}.mp4', fourcc, 20.0, (width, height))

        self.recording = True
        while self.recording:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        self.recording = False
                    if event.key == pygame.K_p:
                        self.save_image()
            values = self.get_key_input()
            self.drone.send_rc_control(values[0], values[1], values[2], values[3])
            img = self.drone.get_frame_read().frame
            img = cv2.resize(img, (width, height))
            out.write(img)
            self.img = img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_surface = pygame.surfarray.make_surface(img)
            self.screen.blit(pygame.transform.rotate(img_surface, -90), (0, 0))
            pygame.display.update()
        out.release()
        self.recording = False


def main():
    controller = TelloController() # calling the class
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                controller.clean_up()
                exit()
        values = controller.get_key_input()
        controller.drone.send_rc_control(values[0], values[1], values[2], values[3])
        controller.img = controller.drone.get_frame_read().frame
        classIds, confs, bbox = net.detect(controller.img, confThreshold=thres, nmsThreshold=nmsThres)
        try:
            for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cvzone.cornerRect(controller.img, box)
                cv2.putText(controller.img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                            (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 255, 0), 2)
        except:
            pass
        controller.img = cv2.cvtColor(controller.img, cv2.COLOR_RGB2BGR)
        img_surface = pygame.surfarray.make_surface(controller.img)
        img_surface = pygame.transform.flip(img_surface, True, False)
        controller.screen.blit(pygame.transform.rotate(img_surface,-270),(0,0))
        pygame.display.update()
        print(f'Battery: {controller.drone.get_battery()}%')
        if ord('v') == cv2.waitKey(1) & 0xFF:
            break
        time.sleep(0.05)


def clean_up(self):
    pygame.quit()
    self.drone.land()
    self.drone.streamoff()
    self.drone.end()


if __name__ == "__main__":
    main()

