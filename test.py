import argparse
import csv
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim

from models import *
from tqdm import tqdm

eps = 1e-6


# -----------------------Load the parameters for testing model---------------------------------------
def load_model(model_path, device):
    model = MS_UNet()
    if device == "cuda":
        model = model.cuda()
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    # Load parameters for model
    if model_path:
        print("Loading model: ", model_path)
        checkpoint = torch.load(model_path, device)
        model.load_state_dict(checkpoint["model_state_dict"], False)
    else:
        print("False loading model!")
        return None
    return model


# -----------------Using the model to output the flow map for the distortion image------------------
def estimation_flowmap(model, img, device):
    model.eval()
    img = cv2.resize(img, (512, 384))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    # img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.unsqueeze(0)
    with torch.no_grad():
        img = img.to(device)
        output = model(img)
    output = output.detach().cpu().squeeze(0).numpy()
    return output


# ----------------------The computation process of face metric ---------------------------------------
def compute_cosin_similarity(preds, gts):
    people_num = gts.shape[0]
    points_num = gts.shape[1]
    similarity_list = []
    preds = preds.astype(np.float32)
    gts = gts.astype(np.float32)
    for people_index in range(people_num):
        # the index 63 of lmk is the center point of the face, that is, the tip of the nose
        pred_center = preds[people_index, 63, :]
        pred = preds[people_index, :, :]
        pred = pred - pred_center[None, :]
        gt_center = gts[people_index, 63, :]
        gt = gts[people_index, :, :]
        gt = gt - gt_center[None, :]

        dot = np.sum((pred * gt), axis=1)
        pred = np.sqrt(np.sum(pred * pred, axis=1))
        gt = np.sqrt(np.sum(gt * gt, axis=1))

        similarity_list_tmp = []
        for i in range(points_num):
            if i != 63:
                similarity = (dot[i] / (pred[i] * gt[i] + eps))
                similarity_list_tmp.append(similarity)

        similarity_list.append(np.mean(similarity_list_tmp))

    return np.mean(similarity_list)


# --------------------The normalization function -----------------------------------------------------
def normalization(x):
    return [(float(i) - min(x)) / float(max(x) - min(x) + eps) for i in x]


# -------------------The computation process of line metric-------------------------------------------
def compute_line_slope_difference(pred_line, gt_k):
    scores = []
    for i in range(pred_line.shape[0] - 1):
        pk = (pred_line[i + 1, 1] - pred_line[i, 1]) / (pred_line[i + 1, 0] - pred_line[i, 0] + eps)
        score = np.abs(pk - gt_k)
        scores.append(score)
    scores_norm = normalization(scores)
    score = np.mean(scores_norm)
    score = 1 - score
    return score


# -------------------------------Compute the out put flow map -------------------------------------------------
def compute_ori2shape_face_line_metric(model, oriimg_paths):
    line_all_sum_pred = []
    face_all_sum_pred = []

    for oriimg_path in tqdm(oriimg_paths):
        # Get the [Source image]
        ori_img = cv2.imread(oriimg_path)  # Read the oriinal image
        ori_height, ori_width, _ = ori_img.shape  # get the size of the oriinal image
        input = ori_img.copy()  # get the image as the input of our model

        # Get the [flow map]"""
        pred = estimation_flowmap(model, input, device)
        pflow = pred.transpose(1, 2, 0)
        predflow_x, predflow_y = pflow[:, :, 0], pflow[:, :, 1]

        scale_x = ori_width / predflow_x.shape[1]
        scale_y = ori_height / predflow_x.shape[0]
        predflow_x = cv2.resize(predflow_x, (ori_width, ori_height)) * scale_x
        predflow_y = cv2.resize(predflow_y, (ori_width, ori_height)) * scale_y

        # Get the [predicted image]"""
        ys, xs = np.mgrid[:ori_height, :ori_width]
        mesh_x = predflow_x.astype("float32") + xs.astype("float32")
        mesh_y = predflow_y.astype("float32") + ys.astype("float32")
        pred_out = cv2.remap(input, mesh_x, mesh_y, cv2.INTER_LINEAR)
        cv2.imwrite(oriimg_path.replace(".jpg", "_pred.jpg"), pred_out)

        # Get the landmarks from the [gt image]
        stereo_lmk_file = open(oriimg_path.replace(".jpg", "_stereo_landmark.json"))
        stereo_lmk = np.array(json.load(stereo_lmk_file), dtype="float32")

        # Get the landmarks from the [source image]
        ori_lmk_file = open(oriimg_path.replace(".jpg", "_landmark.json"))
        ori_lmk = np.array(json.load(ori_lmk_file), dtype="float32")

        # Get the landmarks from the the pred out
        out_lmk = np.zeros_like(ori_lmk)
        for i in range(ori_lmk.shape[0]):
            for j in range(ori_lmk.shape[1]):
                x = ori_lmk[i, j, 0]
                y = ori_lmk[i, j, 1]
                if y < predflow_y.shape[0] and x < predflow_y.shape[1]:
                    out_lmk[i, j, 0] = x - predflow_x[int(y), int(x)]
                    out_lmk[i, j, 1] = y - predflow_y[int(y), int(x)]
                else:
                    out_lmk[i, j, 0] = x
                    out_lmk[i, j, 1] = y

        # Compute the face metric
        face_pred_sim = compute_cosin_similarity(out_lmk, stereo_lmk)
        face_all_sum_pred.append(face_pred_sim)
        stereo_lmk_file.close()
        ori_lmk_file.close()

        # Get the line from the [gt image]
        gt_line_file = oriimg_path.replace(".jpg", "_line_lines.json")
        lines = json.load(open(gt_line_file))

        # Get the line from the [source image]
        ori_line_file = oriimg_path.replace(".jpg", "_lines.json")
        ori_lines = json.load(open(ori_line_file))

        # Get the line from the pred out
        pred_ori2shape_lines = []
        for index, ori_line in enumerate(ori_lines):
            ori_line = np.array(ori_line, dtype="float32")
            pred_ori2shape = np.zeros_like(ori_line)
            for i in range(ori_line.shape[0]):
                x = ori_line[i, 0]
                y = ori_line[i, 1]
                pred_ori2shape[i, 0] = x - predflow_x[int(y), int(x)]
                pred_ori2shape[i, 1] = y - predflow_y[int(y), int(x)]
            pred_ori2shape = pred_ori2shape.tolist()
            pred_ori2shape_lines.append(pred_ori2shape)

        # Compute the lines score
        line_pred_ori2shape_sum = []
        for index, line in enumerate(lines):
            gt_line = np.array(line, dtype="float32")
            pred_ori2shape = np.array(pred_ori2shape_lines[index], dtype="float32")
            gt_k = (gt_line[1, 1] - gt_line[0, 1]) / (gt_line[1, 0] - gt_line[0, 0] + eps)
            pred_ori2shape_score = compute_line_slope_difference(pred_ori2shape, gt_k)
            line_pred_ori2shape_sum.append(pred_ori2shape_score)
        line_all_sum_pred.append(np.mean(line_pred_ori2shape_sum))

    return np.mean(line_all_sum_pred) * 100, np.mean(face_all_sum_pred) * 100


if __name__ == "__main__":

    # Parameters definition
    # model_path = "./checkpoint/baseline_msunet.tar"
    model_path = "./checkpoint/semi_supervised.con_seg.tar"
    test_dir = "./test"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oriimg_paths = []
    for root, dirs, files in os.walk(test_dir):
        for file_name in files:
            if file_name.endswith(".jpg"):
                if "line" not in file_name and "stereo" not in file_name and "pred" not in file_name:
                    oriimg_paths.append(os.path.join(root, file_name))
    print("The number of images: :", len(oriimg_paths))

    print("--------------------------Test--------------------------")
    model = load_model(model_path, device)
    line_score, face_score = compute_ori2shape_face_line_metric(model, oriimg_paths)
    print("Line_score = {:.3f}, Face_score = {:.3f} ".format(line_score, face_score))
