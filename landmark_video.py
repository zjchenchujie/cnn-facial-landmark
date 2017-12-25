"""Demo code shows how to detect landmarks from video"""
import os

import numpy as np
import tensorflow as tf

import cv2
import face_detector as fd
import pts_tools as pt

CWD_PATH = os.getcwd()
MODEL_PB = 'frozen_graph/frozen_graph.pb'

INPUT_SIZE = 128

# Path to frozen detection graph..
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_PB)


def detect_marks(image_np, sess, detection_graph):
    """Detect marks from image"""
    # Get result tensor by its name.
    logits_tensor = detection_graph.get_tensor_by_name('logits/BiasAdd:0')

    # Actual detection.
    predictions = sess.run(
        logits_tensor,
        feed_dict={'input_image_tensor:0': image_np})

    # Convert predictions to landmarks.
    marks = np.array(predictions).flatten()
    marks = np.reshape(marks, (-1, 2))

    return marks


def extract_face(image):
    """Extract face area from image."""
    conf, raw_boxes = fd.get_facebox(image=image, threshold=0.9)
    # fd.draw_result(image, conf, raw_boxes)

    for box in raw_boxes:
        # Move box down.
        diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
        offset_y = int(abs(diff_height_width / 2))
        box_moved = pt.move_box(box, [0, offset_y])

        # Make box square.
        facebox = pt.get_square_box(box_moved)

        if pt.box_in_image(facebox, image):
            return facebox

    return None


def main():
    """MAIN"""
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Get frame from webcam or video file.
    cam = cv2.VideoCapture(0)

    while True:
        # Read frame
        _, frame = cam.read()

        # Get face area image.
        facebox = extract_face(frame)
        if facebox is None:
            continue
        else:
            face_img = frame[
                facebox[1]: facebox[3],
                facebox[0]: facebox[2]]

        # Detect landmarks
        face_img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        landmarks = detect_marks(face_img, sess, detection_graph)

        # Visualization of the result.
        origin_box_size = facebox[2] - facebox[0]
        for mark in landmarks:
            mark[0] = facebox[0] + mark[0] * origin_box_size
            mark[1] = facebox[1] + mark[1] * origin_box_size
            cv2.circle(frame, (int(mark[0]), int(
                mark[1])), 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == 27:
            break

    sess.close()


if __name__ == '__main__':
    main()