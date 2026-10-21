# 复现、版本与验证口径

## 两套检测器

qA_aim是传统主结果；qA_yolo是模型扩展。两者各自产生detections、quality、pred、诊断与图表，不交叉读取观测缓存。共享输入在data/armor。

主结果不回写早期试验，历史滤波与端点优化分别留在调整前备份、spin_refined。复制到提交目录后仅修正输入查找和启动路径，不改变检测阈值或滤波模型。模型中文注释此前经过AST等价检查；本次再次进行语法与功能检查。

## 重新运行

1. 安装requirements.txt，执行q1_env/check_environment.py。
2. qA_aim/run_analysis.py --video all生成远、中距离观测与远距离预测。
3. qA_aim/run_extra.py --video all生成近距离和蓝色观测。
4. qA_aim/estimate_spin.py --video all生成中、近中心与转速。
5. qA_aim/analyze_qr.py运行25组参数实验；已有快照时复用qr_analysis/observations.csv。需要以新观测重做时传--refresh-observations。
6. qA_aim/compare_distortion.py复现有/无D的差值。
7. 模型版采用同名脚本的qA_yolo路径；先检测再估计，不复用传统版CSV。
8. python tools/validate_submission.py检查所有提交表头、帧号、单位时序约定、空值策略及视频规格和样本解码。

## 检查的边界

能导入、能运行、视频帧数正确不等于算法精度验收。没有人工逐帧标注，帧候选覆盖不是召回率；没有位置与转速真值，残差与中心波动不是绝对误差。标定则有三组尺测依据，但样本不足且偏差较大。

合成数据用于验证估计器结构、缺失处理和时间因果性，不能以合成误差替代真实录像精度。仓库tools/check_spin_model.py包含对应检查。

原视频150fps用于时间戳，不是本机运算速度。模型版抽帧CPU前向时间约34ms中位数不包含全链路，不声称150fps实时处理；线程、硬件与负载会影响速度。当前未开启GPU。

## Git与日期

保留原两次上传提交；当前新增提交对应实际整理阶段，未伪造10月16日以来的持续代码历史。文件保存时间不作为算法完成日期的唯一证明。上传目标沿用考生明确指定的仓库，未改仓库名、访问权限或覆盖远端历史。

## 文件核验

tools/submission_validation.json保存本次格式与媒体检查；docs/material_index.md和docs/file_manifest.csv列出文件用途、大小及SHA256。JSON数值中的NaN需要按原脚本空值输出逻辑解读，正式CSV空字段表示没有有效观测/估计。

## 本次整理修复

核验时发现原中距离传统识别MP4缺失moov索引而无法播放。已使用原始视频与本次提交的detections_中距离陀螺.csv逐帧重新绘制，保留全部CSV值，未重新检测或改变观测。复现脚本为qA_aim/render_from_csv.py。
