from pathlib import Path
import subprocess
import sys
import os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.plot_cache'))
choices={
 '1':['preview_yolo.py','--video','中距离陀螺'],
 '2':['preview_yolo.py','--video','远距离平移'],
 '3':['preview_yolo.py','--video','近距离陀螺'],
 '4':['preview_yolo.py','--video','装甲板_蓝'],
 '5':['run_analysis.py','--video','all'],
 '6':['run_extra.py','--video','all'],
 '7':['estimate_spin.py','--video','all'],
 '8':['run_analysis.py','--predict-only'],
 '9':['analyze_qr.py']}
while True:
 print('\nYOLO装甲板与小陀螺\n1 中距离识别窗口\n2 远距离识别窗口\n3 近距离识别窗口\n4 蓝色识别窗口\n5 处理远、中距离完整视频\n6 处理近距离、蓝色完整视频\n7 估计中、近距离车体中心与转速（先运行5和6）\n8 重新计算平移预测（先运行5）\n9 Q、R对比（先运行5）\n0 退出')
 choice=input('输入数字，按回车：').strip()
 if choice=='0':break
 if choice in choices:
  result=subprocess.run([sys.executable,*choices[choice]],cwd=HERE)
  print('处理完成，结果在当前文件夹与figures中。' if result.returncode==0 else '运行失败，请保留上方错误信息。')
