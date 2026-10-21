"""ONNX YOLO pose inference. No torch, pickle, or color-threshold detection."""
# 【第1段：准备工具】Path找文件；ast解析模型说明；json读取参数；time计时。
# cv2负责模型推理、图像处理和画框；np负责数组与坐标计算。
from pathlib import Path
import ast
import json
import time
import cv2
import numpy as np

# 当前PY文件所在目录；不依赖你从哪个终端目录启动。
HERE = Path(__file__).resolve().parent


# 【第2段：读取ONNX内部格式——初学可折叠跳过】
# 这是只读取模型说明的辅助函数，不是神经网络推理代码。
# varint读取变长整数；field是字段编号；wire决定该字段怎样存储。
def protobuf_fields(data):
    def varint(i):
        value = shift = 0
        while True:
            byte = data[i]; i += 1
            value |= (byte & 127) << shift
            if byte < 128: return value, i
            shift += 7
    i = 0
    while i < len(data):
        tag, i = varint(i); field, wire = tag >> 3, tag & 7
        if wire == 2:
            length, i = varint(i); value = data[i:i+length]; i += length
        elif wire == 0: value, i = varint(i)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[i:i+size]; i += size
        else: raise ValueError('Unsupported protobuf wire type')
        yield field, value


# 【第3段：读取模型说明】例如类别names、输入尺寸imgsz、关键点形状kpt_shape。
# field=14对应ONNX模型的metadata_props；这里只处理本模型用到的格式。
def metadata(path):
    result = {}
    for field, value in protobuf_fields(Path(path).read_bytes()):
        if field == 14:
            entry = dict(protobuf_fields(value))
            result[entry[1].decode()] = entry[2].decode()
    return result


# 【第4段：整理四点顺序，后面PnP必须依赖正确对应】
# 先按x分出左右各两个点，再按y分出上下，返回LT、LB、RB、RT。
# 仅适用于当前视频中灯条基本竖直的场景；大角度翻滚时可能排错。
# 排序不会修正模型预测偏差，模型点的物理含义还需核对训练标注。
def order_points(points):
    # Normalize to image LT, LB, RB, RT. Does not assume training index names.
    # 每行一个点：[x,y]；4行2列；这里都是图片像素，不是空间中的米。
    points = np.asarray(points, np.float64).reshape(4, 2)
    # points[:,0]取全部点的x；argsort给出从小到大的排列索引。
    sides = points[np.argsort(points[:, 0], kind='stable')]
    # 画面坐标y向下增大，所以y较小的是上端。
    left = sides[:2][np.argsort(sides[:2, 1])]
    right = sides[2:][np.argsort(sides[2:, 1])]
    result = np.array([left[0], left[1], right[1], right[0]])
    contour = result.astype(np.float32)
    # 排除不凸或面积不足4平方像素的四边形；通过检查也不代表一定准确。
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 4:
        return None
    return result


# 【第5段：检测器类】将模型、类别表、最近结果与推理方法放在一起管理。
# self表示这个检测器实例自身；__init__只在创建实例时执行。
class YoloDetector:
    def __init__(self, model=None):
        # 未指定模型时，默认读取本文件夹/models/exp-2.onnx。
        self.model = Path(model or HERE/'models'/'exp-2.onnx')
        self.meta = metadata(self.model)
        # 把说明里的字符串转为Python字典/列表；literal_eval不会执行任意Python代码。
        self.names = ast.literal_eval(self.meta['names'])
        self.height, self.width = ast.literal_eval(self.meta['imgsz'])
        # 要求4个关键点，每点只有x、y；这个模型不提供每个点的独立置信度。
        if ast.literal_eval(self.meta['kpt_shape']) != [4, 2]:
            raise ValueError('Expected four XY keypoints')
        if self.meta.get('end2end', 'False') != 'False':
            raise ValueError('This decoder expects raw outputs, not end-to-end NMS')
        # 真正加载模型。当前没有设置CUDA后端，使用默认CPU推理流程。
        # 有NVIDIA显卡也不等于自动使用GPU；需支持CUDA的OpenCV及相应后端设置。
        self.net = cv2.dnn.readNetFromONNX(str(self.model))
        self.last = []
        self.inference_ms = 0.

    # 【第6段：核心入口——输入一帧，输出装甲板列表】
    # frame是BGR图片；color筛类别颜色；confidence过滤低分；iou用于重复框抑制。
    def infer(self, frame, color='red', confidence=.25, iou=.45):
        # 【6.1：等比例缩放与补边】原视频1440x1080变640x480，上下各补80到640x640。
        h, w = frame.shape[:2]
        # min确保缩小后的整张图片都能放进模型输入尺寸，避免拉伸变形。
        scale = min(self.width/w, self.height/h)
        nw, nh = round(w*scale), round(h*scale)
        left, top = (self.width-nw)//2, (self.height-nh)//2
        # 创建灰色底图：114是像素灰度；3是BGR三个颜色通道。
        image = np.full((self.height, self.width, 3), 114, np.uint8)
        image[top:top+nh, left:left+nw] = cv2.resize(frame, (nw, nh))
        # 【6.2：整理模型输入】除255变为0到1；swapRB把BGR换成RGB；整理为NCHW数组。
        blob = cv2.dnn.blobFromImage(image, 1/255., (self.width, self.height), swapRB=True)
        # 【6.3：模型前向推理】计时包括setInput和forward，不是整个程序每帧总耗时。
        started = time.perf_counter()
        # 把整理后的图片交给模型。
        self.net.setInput(blob)
        # 执行神经网络，取第一个输出；得到数字数组，还没有画任何框。
        raw = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]
        self.inference_ms = (time.perf_counter()-started)*1000
        # 【6.4：读懂输出】本模型14类，输出形状1x26x8400。
        # 1张图片；26=4个框参数+14个类别分数+8个四点坐标；8400是候选数。
        # 形状检查针对当前提供的固定640模型，不能直接保证兼容任何YOLO导出模型。
        nclasses = len(self.names)
        if raw.shape != (1, 4+nclasses+8, 8400):
            raise ValueError(f'Unexpected output shape: {raw.shape}')
        # 转置为8400行、26列，让每一行对应一个候选。
        predictions = raw[0].T
        # 截取14个类别分数；这个输出布局没有额外独立objectness字段。
        scores = predictions[:, 4:4+nclasses]
        # 每行最大分数所在的位置就是类别编号。
        classes = scores.argmax(axis=1)
        # 类别得分不是经验证的正确概率；0.8分不等于一定有80%概率正确。
        conf = scores.max(axis=1)
        # 【6.5：筛选】R开头保留红色，B开头保留蓝色，空前缀保留所有类别。
        # 依赖模型的分类结果，不再用R-B之类的像素颜色阈值。
        prefix = {'red':'R', 'blue':'B', 'all':''}[color]
        # 同时满足类别颜色要求和最低得分门槛，才保留。
        selected = np.array([self.names[int(c)].startswith(prefix) for c in classes]) & (conf >= confidence)
        indices = np.flatnonzero(selected)
        # 没有候选时返回空列表，不会凭空生成装甲板。
        if not len(indices): self.last = []; return []
        # 【6.6：检测框去重】先提取cx,cy,w,h，再改为左上角x,y,w,h。
        boxes = predictions[indices, :4].copy()
        # 左上角 = 中心 - 宽高的一半。这里只改框，不改四个模型关键点。
        boxes[:, :2] -= boxes[:, 2:]/2
        # Suppress overlapping class hypotheses for the same physical plate.
        # 重叠框优先留高分；iou越小越容易抑制重叠候选。
        # 这里跨类别去重，避免同一块板留下多个类别框；真实相邻板也可能被误压掉。
        kept = cv2.dnn.NMSBoxes(boxes.tolist(), conf[indices].tolist(), confidence, iou)
        result = []
        # 遍历去重后保留的候选，k对应筛选后的boxes行号。
        for k in np.asarray(kept).reshape(-1):
            # 映射回原8400个候选中的行号，才能取到对应关键点。
            idx = indices[k]
            # 【6.7：关键点还原】读取模型直接预测的4个XY点，不是矩形框的四个角。
            pts = predictions[idx, 4+nclasses:].reshape(4, 2).copy()
            # 还原公式：原图点=(模型点-补边)/缩放比例。
            # 例如模型点(320,240)、上补边80、比例4/9，还原约(720,360)。
            pts = (pts-np.array([left, top]))/scale
            # 按左上、左下、右下、右上排序，便于后面的PnP对应物理模型。
            ordered = order_points(pts)
            # 无效四边形或NaN/无穷数，跳过本候选。
            if ordered is None or not np.isfinite(ordered).all(): continue
            # Truncated targets cannot supply four observed physical corners.
            # 任一点超出原图时，当前保守策略跳过该板；不会对越界点直接裁剪造出角点。
            if (ordered[:,0]<0).any() or (ordered[:,0]>=w).any() or (ordered[:,1]<0).any() or (ordered[:,1]>=h).any(): continue
            # 检测框同样还原到原图；宽高只需除比例，不减补边。
            box = boxes[k].astype(float)
            box[:2] = (box[:2]-[left, top])/scale; box[2:] /= scale
            # 【6.8：保存结果】points是排序后像素坐标；raw_points是还原后未排序坐标。
            # box是矩形框；confidence是得分；class_id与name是类别编号与名字。
            result.append(dict(points=ordered, raw_points=pts, box=box,
                               confidence=float(conf[idx]), class_id=int(classes[idx]),
                               name=self.names[int(classes[idx])]))
        # 保存最近一次结果，画类别文字时使用；return再交给调用者。
        self.last = result
        return result


# 【第7段：缓存模型】首次调用才创建检测器，以后复用，避免每帧重新加载模型。
_DETECTOR = None


def detector():
    global _DETECTOR
    if _DETECTOR is None: _DETECTOR = YoloDetector()
    return _DETECTOR


# 【第8段：读取调参配置】yolo_config.json里只有confidence和iou。
def config():
    return json.loads((HERE/'yolo_config.json').read_text(encoding='utf-8'))


# 【第9段：兼容旧程序的入口】识别实际调用上面的YOLO，不做灯条提取。
# **unused接收旧代码传入的颜色阈值，但不会使用它们。
def detect(frame, color='red', **unused):
    settings = config()
    objects = detector().infer(frame, color, settings['confidence'], settings['iou'])
    # Preserve legacy call shape; the mask is only a display placeholder.
    # 返回三个对象是为了兼容旧接口：黑色占位图、空灯条列表、四点列表。
    # 黑图不是YOLO识别掩膜；真正识别结果在第三项。
    return np.zeros(frame.shape[:2], np.uint8), [], [item['points'] for item in objects]


# 【第10段：画图】只显示结果，不负责识别，不解算PnP，也不控制车。
def annotate(frame, lights, armors):
    # 复制图片再画，保留原始frame。
    image = frame.copy()
    for points in armors:
        # 用模型的四个关键点画绿色闭合多边形；True闭合；BGR(0,255,0)绿色；线宽2。
        cv2.polylines(image, [points.astype(np.int32)], True, (0,255,0), 2)
        # LT左上、LB左下、RB右下、RT右上。
        for label, pt in zip(('LT','LB','RB','RT'), points):
            xy = tuple(pt.astype(int))
            # 在关键点处画半径3像素的黄色实心圆，-1表示填充。
            cv2.circle(image, xy, 3, (0,255,255), -1)
            cv2.putText(image, label, xy, 0, .45, (0,255,0), 1)
    # 为最近一次识别结果画类别和得分。
    for item in detector().last:
        xy = tuple((item['points'][0]+[0,-10]).astype(int))
        cv2.putText(image, f"YOLO {item['name']} {item['confidence']:.2f}", xy, 0, .55, (255,255,0), 1)
    # 返回画好的图片，调用者再用imshow显示或VideoWriter保存。
    return image
