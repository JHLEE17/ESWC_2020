from imutils.video import VideoStream
import imagezmq


#path = "rtsp://192.168.1.77:8080//h264_ulaw.sdp"  # change to your IP stream address
#path = "http://121.88.204.251:8082/" 종호집
path = "http://58.141.147.118:8082"
cap = VideoStream(path)

sender = imagezmq.ImageSender(connect_to='tcp://localhost:5566')  # change to IP address and port of server thread
cam_id = 'Camera 2'  # this name will be displayed on the corresponding camera stream

stream = cap.start()

while True:

    frame = stream.read()
    sender.send_image(cam_id, frame)
