# 数据、模型来源与许可说明

## 考核素材

四段原始装甲板视频和camera_info.json来自考生已有的MountainForge视觉组2027赛季招新考核数据包。原说明将工业相机视频描述为深圳大学RP战队开源实拍测试数据。仓库保存原文件用于考核复现，不声称为考生自行采集，也不替素材权利人新增许可证。原任务文本见assessment_requirements.md。

## 自采实验材料

q2_calib内20张标定图、A4照片与尺读数来自此次本地实验归档。三组更正数据不是额外三组实验。q1_env内Ubuntu截图来自已存在的GitHub上传，保存原图；未把渲染文字当作真实终端截图。

## YOLO权重

exp-2.onnx与exp-2.pt由考生提供。ONNX元数据自述为Ultralytics YOLO26n-pose，版本8.4.165，输入640×640，14类、4个XY关键点，license字段为AGPL-3.0，并带Ultralytics文档链接。完整元数据与模型SHA256在qA_yolo/yolo_validation.json。

推理只读取ONNX，未在本项目执行PT反序列化或训练。PT作为提供的源权重归档，不代表已验证其与ONNX数值完全等价。仓库没有训练数据、训练日志及PyTorch与ONNX Runtime的公平速度对比，不申报附加题一完成；模型具体训练数据权利与授权范围以提供方记录为准。

关键点名称只有0、1、2、3，未给出物理语义；当前以图像几何关系整理点序，并经画面检查。真实训练标注是否严格对应灯条端点仍需确认。

## 软件依赖与参考

- OpenCV：图像处理、PnP、相机标定、ONNX DNN推理。
- NumPy：数组、线性代数与滤波实现。
- matplotlib：科学图表。
- PyYAML：满足公共题环境要求，用于环境核验。
- Codex：辅助实现、验证与解释，详见ai_log.md。

参考官方文档：
- https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
- https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html
- https://docs.ultralytics.com/

未为整个仓库统一添加会覆盖第三方素材与权重的许可证。第三方依赖和模型遵守各自许可证，权利归属不因本次考核整理而改变。
