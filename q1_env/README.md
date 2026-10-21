# 题一：开发环境与工程规范

## 环境检查

本次整理使用 Python 3.12.14；numpy 2.5.4、opencv-python 5.0.0.93、matplotlib 3.11.2、PyYAML 6.0.3。版本为实际检查值，不表示所有平台都必须安装同一版本。

```bash
python -m pip install -r requirements.txt
python q1_env/check_environment.py
```

程序检查Python版本、依赖导入、NumPy基本计算、OpenCV编码与相机模型函数、matplotlib无界面渲染、YAML解析。当前输出在 environment_check.json 和 environment_check.txt。检查日期为本次整理日期，与早期截图日期区分。

## Ubuntu 证据

以下为原有GitHub提交中的真实截图：第一张显示Ubuntu 22.04.5 LTS、ls -la与top，第二张显示cat /proc/cpuinfo | head。未重新制作或冒充当前终端截图。

![Ubuntu版本与目录、进程命令](figures/屏幕截图%202026-10-16%20173549.png)
![CPU信息命令](figures/屏幕截图%202026-10-16%20173721.png)

## Git 说明

仓库保留已有网页上传提交，后续按环境、标定、传统A题、YOLO扩展、报告核验等整理步骤提交。本次未将早期工作伪装成逐日Git提交，未修改作者日期。现有仓库名为-rm-vision-recruit--dingmeiyan-，与题目推荐的rm-vision-recruit-dingmeiyan有额外连字符；沿用考生指定仓库。

```bash
git log --oneline --decorate
git status --short
```
