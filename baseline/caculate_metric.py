import os
import json
import argparse
import numpy as np
from collections import defaultdict

"""
baseline/caculate_metric.py — 官方评估指标计算

STrack2 = 0.2 × Sloc + 0.2 × Scls + 0.6 × Sgrade

用法（推荐通过 eval.sh 调用，或直接运行）：
  python baseline/caculate_metric.py \
      --gt_img_dir  data/Track2/NG_1154/images \
      --gt_txt_dir  data/Track2/NG_1154/level_labels \
      --pt_img_dir  data/Track2/NG_1154/images \
      --pt_txt_dir  result/grading_E8 \
      --cls_txt_dir configs/class_name.txt \
      --iou_thresh  0.25
"""


def convert_mask2bbox(points):
    if len(points) % 2 != 0:
        raise ValueError("points 数量不是偶数，无法组成 (x,y) 对")
    npoints = np.zeros((2, int(len(points)/2)), dtype=float)
    for i in range(0, int(len(points)/2)):
        npoints[:, i] = np.array([points[2*i], points[2*i+1]])
    xmin, xmax = float(min(npoints[0,:])), float(max(npoints[0,:]))
    ymin, ymax = float(min(npoints[1,:])), float(max(npoints[1, :]))
    w = xmax - xmin
    h = ymax - ymin
    return [xmin, ymin, w, h]


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def bbox_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    area1 = max(0.0, (x1_max - x1_min)) * max(0.0, (y1_max - y1_min))
    area2 = max(0.0, (x2_max - x2_min)) * max(0.0, (y2_max - y2_min))
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


class CaculateMetric:
    def __init__(self):
        self.gt_data = {}
        self.pred_data = {}
        self.grade_map = ['Acceptable', 'Marginal NG', 'NG', 'Gross NG']

    def read_data(self, img_dir, txt_dir, txt_shuffix=".json"):
        all_data = {}
        classes_index = []
        for img_name in os.listdir(img_dir):
            if img_name.split(".")[-1] not in ["bmp", "jpg", "png", "jpeg"]:
                continue
            txt_name = img_name.split(".")[0] + txt_shuffix
            txt_path = os.path.join(txt_dir, txt_name)
            if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
                all_data[img_name.split(".")[0]] = []
                continue
            with open(txt_path, 'r', encoding='utf-8') as f:
                if txt_shuffix == ".txt":
                    lines = f.readlines()
                    data = []
                    for line in lines:
                        line = line.strip().split(" ")
                        bbox = convert_mask2bbox(line[1:])
                        data.append({"cls": int(line[0]), "bbox": bbox})
                        if int(line[0]) not in classes_index:
                            classes_index.append(int(line[0]))
                    all_data[img_name.split(".")[0]] = data
                elif txt_shuffix == ".json":
                    lines = json.load(f)
                    data = []
                    for line in lines:
                        points = [float(x.strip()) for x in line["points"].split(',')]
                        bbox = convert_mask2bbox(points)
                        if "severity" in list(line.keys()) and line["severity"] in self.grade_map:
                            grade = self.grade_map.index(line["severity"])
                        else:
                            grade = None
                        data.append({"cls": int(line["class"]), "bbox": bbox, "grade": grade})
                        if int(line["class"]) not in classes_index:
                            classes_index.append(int(line["class"]))
                    all_data[img_name.split(".")[0]] = data
        return all_data, classes_index

    def caculate_screen(self):
        tp_img = dict.fromkeys(self.classes_index, 0)
        fp_img = dict.fromkeys(self.classes_index, 0)
        fn_img = dict.fromkeys(self.classes_index, 0)
        tp_img_cnt = 0
        tn_img_cnt = 0
        fp_img_cnt = 0
        fn_img_cnt = 0
        for img_name, data in self.gt_data.items():
            if len(data) == 0 and (img_name not in list(self.pred_data.keys()) or len(self.pred_data[img_name]) == 0):
                tn_img_cnt += 1
            elif len(data) == 0 and len(self.pred_data[img_name]) != 0:
                for v in self.pred_data[img_name]:
                    cls = v["cls"]
                    fp_img[cls] += 1
                fp_img_cnt += 1
            elif len(data) != 0 and (img_name not in list(self.pred_data.keys()) or len(self.pred_data[img_name]) == 0):
                for v in self.gt_data[img_name]:
                    cls = v['cls']
                    fn_img[cls] += 1
                fn_img_cnt += 1
            elif len(data) != 0 and len(self.pred_data[img_name]) != 0:
                gt_match = 0
                for i, gd in enumerate(data):
                    for pd in self.pred_data[img_name]:
                        bbox1 = xywh_to_xyxy(gd["bbox"])
                        bbox2 = xywh_to_xyxy(pd["bbox"])
                        if bbox_iou(bbox1, bbox2) > 0.25 and gd["cls"] == pd["cls"]:
                            gt_match += 1
                            cls = gd['cls']
                            tp_img[cls] += 1
                            break
                if gt_match > 0:
                    tp_img_cnt += 1
        SscreenDict = defaultdict(float)
        for cls in self.classes_index:
            cls_recall_img = tp_img[cls] / (tp_img[cls] + fn_img[cls] + 10e-6)
            cls_specificity_img = tn_img_cnt / (tn_img_cnt + fn_img[cls] + 10e-6)
            cls_sscreen = 0.5 * cls_recall_img + 0.5 * cls_specificity_img
            SscreenDict[cls] = cls_sscreen
        Recall_img = tp_img_cnt / (tp_img_cnt + fn_img_cnt + 10e-6)
        Specificity_img = tn_img_cnt / (tn_img_cnt + fn_img_cnt + 10e-6)
        Sscreen = 0.5 * Recall_img + 0.5 * Specificity_img
        SscreenDict['all'] = Sscreen
        return SscreenDict

    def caculate_Sfine(self, iou_thresh=0.25):
        TP = dict.fromkeys(self.classes_index, 0)
        FP = dict.fromkeys(self.classes_index, 0)
        FN = dict.fromkeys(self.classes_index, 0)
        for img_name, gt_data in self.gt_data.items():
            if len(gt_data) == 0 and (img_name not in list(self.pred_data.keys()) or len(self.pred_data[img_name]) == 0):
                continue
            elif len(gt_data) == 0 and len(self.pred_data[img_name]) != 0:
                for pd in self.pred_data[img_name]:
                    cls = pd["cls"]
                    FP[cls] += 1
            elif len(gt_data) != 0 and (img_name not in list(self.pred_data.keys()) or len(self.pred_data[img_name]) == 0):
                for gd in gt_data:
                    cls = gd["cls"]
                    FN[cls] += 1
            elif len(gt_data) != 0 and len(self.pred_data[img_name]) != 0:
                for i, gd in enumerate(gt_data):
                    gt_match = 0
                    for pd in self.pred_data[img_name]:
                        bbox1 = xywh_to_xyxy(gd["bbox"])
                        bbox2 = xywh_to_xyxy(pd["bbox"])
                        if bbox_iou(bbox1, bbox2) > 0.25 and gd["cls"] == pd["cls"]:
                            gt_match += 1
                            break
                    if gt_match > 0:
                        TP[gd["cls"]] += 1
                    else:
                        FN[gd["cls"]] += 1
        per_class_f1 = {}
        sfine_sum = 0.0
        SfineDict = defaultdict(float)
        for cls in self.classes_index:
            tp = TP[cls]
            fp = FP[cls]
            fn = FN[cls]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            per_class_f1[cls] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
            SfineDict[cls] = f1
            sfine_sum += f1
        Sfine = sfine_sum / len(self.classes_index)
        SfineDict['all'] = Sfine
        return SfineDict

    def caculate_cls(self):
        SfineDict = self.caculate_Sfine()
        SscreenDict = self.caculate_screen()
        SclsDict = defaultdict(float)
        for cls in self.classes_index:
            SclsDict[cls] = 0.5 * SfineDict[cls] + 0.5 * SscreenDict[cls]
        SclsDict['all'] = 0.5 * SfineDict['all'] + 0.5 * SscreenDict['all']
        return SclsDict

    def caculate_loc(self):
        ious_dict = {}
        SlocDict = dict.fromkeys(self.classes_index, 0)
        for file_name, gt_data in self.gt_data.items():
            if file_name not in list(self.pred_data.keys()):
                continue
            pt_data = self.pred_data[file_name]
            if len(gt_data) == 0 or len(pt_data) == 0:
                continue
            matched_pred = set()
            for g in gt_data:
                best_iou = 0
                best_idx = -1
                for pi, p in enumerate(pt_data):
                    if pi in matched_pred or g["cls"] != p["cls"]:
                        continue
                    bbox1 = xywh_to_xyxy(g["bbox"])
                    bbox2 = xywh_to_xyxy(p["bbox"])
                    iou = bbox_iou(bbox1, bbox2)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = pi
                if best_iou > 0 and best_idx >= 0:
                    matched_pred.add(best_idx)
                    cls = g['cls']
                    if cls not in ious_dict:
                        ious_dict[cls] = [best_iou]
                    else:
                        ious_dict[cls].append(best_iou)
        ious = []
        for cls in self.classes_index:
            if cls not in ious_dict or len(ious_dict[cls]) == 0:
                SlocDict[cls] = 0
                continue
            v = ious_dict[cls]
            cls_sloc = float(np.mean(v))
            ious.extend(v)
            SlocDict[cls] = cls_sloc
        Sloc = float(np.mean(ious))
        SlocDict['all'] = Sloc
        return SlocDict

    def severity_grading_from_confmat(self, conf_mat):
        conf_mat = np.asarray(conf_mat, dtype=float)
        K = conf_mat.shape[0]
        N = conf_mat.sum()
        if N == 0:
            return np.nan
        idx = np.arange(K)
        I, J = np.meshgrid(idx, idx, indexing="ij")
        W = ((I - J) ** 2) / float((K - 1) ** 2)
        num = N * np.sum(W * conf_mat)
        n_i_dot = conf_mat.sum(axis=1)
        n_dot_j = conf_mat.sum(axis=0)
        expected = np.outer(n_i_dot, n_dot_j)
        denom = np.sum(W * expected)
        if denom == 0:
            return 0
        return 1.0 - num / denom

    def collect_triplets(self, gt_dict, pred_dict, iou_thres=0.25):
        triplets = []
        common_imgs = set(gt_dict.keys()) & set(pred_dict.keys())
        if not common_imgs:
            raise ValueError("gt_dict 和 pred_dict 没有共同的图像名")
        for img in common_imgs:
            gt_instances = gt_dict[img]
            pred_instances = pred_dict[img]
            n = min(len(gt_instances), len(pred_instances))
            for k in range(n):
                gt_ins = gt_instances[k]
                pred_ins = pred_instances[k]
                bbox1 = xywh_to_xyxy(gt_ins["bbox"])
                bbox2 = xywh_to_xyxy(pred_ins["bbox"])
                if gt_ins["cls"] == pred_ins["cls"] and bbox_iou(bbox1, bbox2) > iou_thres:
                    cls_id = int(gt_ins["cls"])
                    gt_g = int(gt_ins["grade"])
                    pred_g = int(pred_ins["grade"])
                    triplets.append((cls_id, gt_g, pred_g))
                    break
        if not triplets:
            raise ValueError("没有匹配到任何实例，请检查数据和匹配逻辑")
        return triplets

    def caculate_grade(self, K=4):
        SgradeDict = defaultdict(float)
        triplets = self.collect_triplets(self.gt_data, self.pred_data)
        all_gt = np.array([g for _, g, _ in triplets], dtype=int)
        all_pred = np.array([p for _, _, p in triplets], dtype=int)
        if K is None:
            K = max(all_gt.max(), all_pred.max()) + 1
        if all_gt.min() == 1:
            all_gt -= 1
            all_pred -= 1
            triplets = [(c, g - 1, p - 1) for (c, g, p) in triplets]
        overall_conf = np.zeros((K, K), dtype=float)
        for g, p in zip(all_gt, all_pred):
            overall_conf[g, p] += 1
        overall_s_grade = self.severity_grading_from_confmat(overall_conf)
        SgradeDict['all'] = overall_s_grade
        per_cls_pairs = defaultdict(list)
        for cls_id, gt_g, pred_g in triplets:
            per_cls_pairs[cls_id].append((gt_g, pred_g))
        for cls_id, pairs in per_cls_pairs.items():
            conf = np.zeros((K, K), dtype=float)
            for gt_g, pred_g in pairs:
                conf[gt_g, pred_g] += 1
            SgradeDict[cls_id] = self.severity_grading_from_confmat(conf)
        return SgradeDict

    def read_classes(self, class_txt_dir):
        with open(class_txt_dir, 'r') as f:
            classes_list = [i.strip() for i in f.readlines()]
        return classes_list

    def process_data(self, gt_img_dir, gt_txt_dir, pred_img_dir, pred_txt_dir,
                     class_txt_dir, txt_shuffix, S=2):
        classes_list = self.read_classes(class_txt_dir)
        self.gt_data, gt_classes = self.read_data(gt_img_dir, gt_txt_dir, txt_shuffix=txt_shuffix)
        self.pred_data, pt_classes = self.read_data(pred_img_dir, pred_txt_dir, txt_shuffix=txt_shuffix)
        self.classes_index = list(set(gt_classes) | set(pt_classes))
        SscreenDict = self.caculate_screen()
        SfineDict = self.caculate_Sfine()
        SclsDict = defaultdict(float)
        SclsDict['all'] = 0.5 * SfineDict['all'] + 0.5 * SscreenDict['all']
        SlocDict = self.caculate_loc()
        if S == 2:
            SgradeDict = self.caculate_grade(K=4)
            S2Dict = dict.fromkeys(self.classes_index, 0)
            S2 = 0.2 * SlocDict['all'] + 0.2 * SclsDict['all'] + 0.6 * SgradeDict['all']
            S2Dict['all'] = S2
            print("class" + " " * 5 + "Sloc" + " " * 5 + "0.5*Sscreen" + " " * 5 + "0.5*Sfine" + " " * 5 + "Scls" + " " * 5 + "Sgrade" + " " * 5 + "Strack2")
            print("all" + " " * 5 + f"{SlocDict['all']:.3f}" + " " * 5 + f"{SscreenDict['all']*0.5:.3f}" + " " * 5 + f"{SfineDict['all']*0.5:.3f}" + " " * 5 + f"{SclsDict['all']:.3f}" + " " * 5 + f"{SgradeDict['all']:.3f}" + " " * 5 + f"{S2:.3f}")
            for cls_idx in self.classes_index:
                cls_sloc = SlocDict[cls_idx] if cls_idx in list(SlocDict.keys()) else 0
                cls_sfine = SfineDict[cls_idx] if cls_idx in list(SfineDict.keys()) else 0
                cls_screen = SscreenDict[cls_idx] if cls_idx in list(SscreenDict.keys()) else 0
                cls_scls = 0.5 * cls_sfine + 0.5 * cls_screen
                SclsDict[cls_idx] = cls_scls
                cls_grade = SgradeDict[cls_idx] if cls_idx in list(SgradeDict.keys()) else 0
                cls_s2 = 0.2 * cls_sloc + 0.2 * cls_scls + 0.6 * cls_grade
                S2Dict[cls_idx] = cls_s2
                print(classes_list[cls_idx] + " " * 3 + f"{cls_sloc:.3f}" + " " * 5 + f"{cls_screen*0.5:.3f}" + " " * 5 + f"{cls_sfine*0.5:.3f}" + " " * 5 + f"{cls_scls:.3f}" + " " * 5 + f"{cls_grade:.3f}" + " " * 5 + f"{cls_s2:.3f}")
            return S2Dict
        return None


def parse_args():
    p = argparse.ArgumentParser(description="计算 STrack2 官方评估指标")
    p.add_argument("--gt_img_dir",  required=True, help="GT 图像目录")
    p.add_argument("--gt_txt_dir",  required=True, help="GT 标注目录（.json）")
    p.add_argument("--pt_img_dir",  required=True, help="预测图像目录")
    p.add_argument("--pt_txt_dir",  required=True, help="预测标注目录（.json，含 severity）")
    p.add_argument("--cls_txt_dir", required=True, help="class_name.txt 路径")
    p.add_argument("--iou_thresh",  type=float, default=0.25)
    p.add_argument("--suffix",      default=".json", help="标注文件后缀")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cm = CaculateMetric()
    SDict = cm.process_data(
        args.gt_img_dir, args.gt_txt_dir,
        args.pt_img_dir, args.pt_txt_dir,
        args.cls_txt_dir, args.suffix, S=2
    )
    print("caculate finished!")
