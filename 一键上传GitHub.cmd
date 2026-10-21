@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在将已整理并提交的材料推送到考核仓库。
echo 如出现 GitHub 登录页面，请用仓库所属账号完成登录。
echo 此操作不使用强制推送，不改写远端历史。
git -c http.proxy=http://127.0.0.1:7890 -c http.version=HTTP/1.1 push origin main
if errorlevel 1 (
 echo 上传未完成，请保留上面的错误信息。
 pause
 exit /b 1
)
echo 上传成功。正在显示远端 main 的提交编号：
git -c http.proxy=http://127.0.0.1:7890 ls-remote origin refs/heads/main
pause
