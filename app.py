from importlib import import_module
from flask import Flask, render_template, Response
import cv2
import time
import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen(camera_stream, feed_type, device):
    """Video streaming generator function."""
    unique_name = (feed_type, device)

    num_frames = 0
    total_time = 0
    while True:
        time_start = time.time()

        cam_id, frame = camera_stream.get_frame(unique_name)
        if frame is None:
            break

        num_frames += 1

        time_now = time.time()
        total_time += time_now - time_start
        fps = num_frames / total_time

        # write camera name                                        0.85 였음
        cv2.putText(frame, cam_id, (int(0.75 * frame.shape[1]), int(0.05 * frame.shape[0])), 0, 1.5e-3 * frame.shape[0],
                    (0, 255, 255), 2)
        
        #200920 11:51 이종호 - fps 문구 삭제
        if feed_type == 'yolo':
            cv2.putText(frame, "FPS: %.2f" % fps, (int(0.75 * frame.shape[1]), int(0.9 * frame.shape[0])), 0,
                        1.5e-3 * frame.shape[0], (0, 255, 255), 2)
        ###
        frame = cv2.imencode('.jpg', frame)[1].tobytes()  # Remove this line for test camera
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        
@app.route('/video_feed/<feed_type>/<device>')
def video_feed(feed_type, device):
    """Video streaming route. Put this in the src attribute of an img tag."""
    port_list = (5555, 5566)
    if feed_type == 'camera':
        camera_stream = import_module('camera_server').Camera
        return Response(gen(camera_stream=camera_stream(feed_type, device, port_list), feed_type=feed_type, device=device),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    elif feed_type == 'yolo':
        camera_stream = import_module('camera_yolo').Camera
        return Response(gen(camera_stream=camera_stream(feed_type, device, port_list), feed_type=feed_type, device=device),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090, threaded=True)
