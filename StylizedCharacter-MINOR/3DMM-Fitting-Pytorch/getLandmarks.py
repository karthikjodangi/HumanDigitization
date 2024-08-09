from facenet_pytorch import MTCNN
import face_alignment
import cv2
import numpy as np
import os
import torch
from core.options import ImageFittingOptions
import core.utils as utils


def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
    """Draw landmarks on the image."""
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), radius, color, -1)
    return image


def save_landmarks(args):
    # Initialize face detection and landmark detection models
    print('Loading models...')
    mtcnn = MTCNN(device=args.device, select_largest=False)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.THREE_D, flip_input=False)

    # Load the image
    print('Loading image...')
    img_arr = cv2.imread(args.img_path)[:, :, ::-1]
    orig_h, orig_w = img_arr.shape[:2]
    print(f'Image loaded. Width: {orig_w}, Height: {orig_h}')

    # Detect the face using MTCNN
    bboxes, probs = mtcnn.detect(img_arr)

    if bboxes is None:
        print('No face detected.')
        return
    else:
        bbox = utils.pad_bbox(bboxes[0], (orig_w, orig_h), args.padding_ratio)
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]
        assert face_w == face_h
    print(f'A face is detected. l: {bbox[0]}, t: {bbox[1]}, r: {bbox[2]}, b: {bbox[3]}')

    # Extract and resize the face region
    face_img = img_arr[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    resized_face_img = cv2.resize(face_img, (args.tar_size, args.tar_size))

    # Detect 3D landmarks
    lms = fa.get_landmarks_from_image(resized_face_img)[0]
    lms_3d = lms[:, :3]

    # Exclude the first 17 landmarks
    lms_3d_excluded = lms_3d[17:]

    # Draw landmarks on the original image (transform landmarks back to original scale)
    lms_2d = lms_3d_excluded[:, :2]
    scale_factor_x = face_w / args.tar_size
    scale_factor_y = face_h / args.tar_size
    lms_2d_orig_scale = lms_2d * [scale_factor_x, scale_factor_y] + [bbox[0], bbox[1]]
    img_with_landmarks = draw_landmarks(img_arr.copy(), lms_2d_orig_scale)

    # Save the landmarks to a npy file
    utils.mymkdirs(args.res_folder)
    basename = os.path.basename(args.img_path)[:-4]
    out_lms_path = os.path.join(args.res_folder, basename + '_landmarks.npy')
    np.save(out_lms_path, lms_3d_excluded)

    # Save the annotated image
    out_img_path = os.path.join(args.res_folder, basename + '_annotated.jpg')
    cv2.imwrite(out_img_path, img_with_landmarks[:, :, ::-1])

    print(f'Landmarks saved at {out_lms_path}')
    print(f'Annotated image saved at {out_img_path}')


if __name__ == '__main__':
    args = ImageFittingOptions().parse()
    args.device = f'cuda:{args.gpu}'
    save_landmarks(args)
