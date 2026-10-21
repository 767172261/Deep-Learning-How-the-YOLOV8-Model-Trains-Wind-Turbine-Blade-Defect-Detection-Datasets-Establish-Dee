深度学习YOLOV8模型如何训练风力发电叶片缺陷检测数据集 建立深度学习基于YOLOV8模型的风电叶片缺陷破损检测系统 识别裂纹 侵蚀 脏污 漏油 孔洞
@[toc]


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ac26e43a43734894ba6c469f5e11c135.png)




**数据集文字描述：**

风电叶片缺陷检测数据集，3282张，yolo和voc两种标注方式 训练集：2743张，验证集：270张，测试集：269张 6类 类别名称: 每一类图像数 ，每一类标注数 Crack-裂纹: 1144,2163 Erosion-侵蚀: 233,337 Dirt-脏污: 693,762 oil_leakage-漏油: 574,600 PU-tape: 329,621 Pin_Hole-孔洞: 303,303 image num: 3282
 6类
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2ed5ac98f270499d8810f4142426de03.png)



### 风电叶片缺陷检测数据集统计

| 类别名称    | 每一类图像数 | 每一类标注数 |
|-------------|------------|------------|
| Crack-裂纹  | 1144       | 2163       |
| Erosion-侵蚀| 233        | 337        |
| Dirt-脏污   | 693        | 762        |
| Oil_leakage-漏油 | 574   | 600        |
| PU-tape     | 329        | 621        |
| Pin_Hole-孔洞 | 303      | 303        |

### 数据集划分

- 训练集：2743张
- 验证集：270张
- 测试集：269张

### YOLOv7 训练与推理代码示例

#### 1. 安装依赖

```bash
pip install torch torchvision torchaudio
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip install -r requirements.txt
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/146a39b6f4194a6cbd9c18d162565e94.png)

#### 2. 准备数据集

将同学您的数据集按照VOC格式或YOLO格式组织好。对于YOLO格式，确保每个图像都有对应的`.txt`文件，内容为标注框信息（类别ID, x_center, y_center, width, height），所有值均相对于图像尺寸归一化。

#### 3. 修改配置文件

在`yolov7/data`目录下创建一个自定义的yaml文件，例如`wind_turbine.yaml`，指定训练、验证、测试集路径及类别信息。

```yaml
train: /path/to/train/images
val: /path/to/val/images
test: /path/to/test/images

nc: 6  # number of classes
names: ['Crack', 'Erosion', 'Dirt', 'oil_leakage', 'PU-tape', 'Pin_Hole']
```

#### 4. 开始训练

```bash
python train.py --workers 8 --device 0 --batch-size 16 --data wind_turbine.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name wind_turbine_results --hyp data/hyp.scratch.p5.yaml
```

#### 5. 推理代码

```python
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
import torch
import cv2

# 加载模型
weights = 'runs/train/wind_turbine_results/weights/best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights, map_location=device)
model.eval()

def detect(image_path):
    img = cv2.imread(image_path)
    img_size = 640
    img0 = img.copy()
    img = letterbox(img, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    for i, det in enumerate(pred):  # detections per image
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=(0, 255, 0), line_thickness=3)

    return img0

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
```

请根据自己的需求调整上述脚本中的参数和路径。同学！
