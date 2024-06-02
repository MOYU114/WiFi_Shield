@echo off

:: 切换到data_preprocess目录
cd .\data_preprocess\

:: 使用无界面的Matlab运行csireader脚本
matlab -nojvm -nodesktop -nodisplay -r csireader
:: 等待几秒
echo 等待matlab脚本成功运行
:: 等待几秒
timeout /t 10

:: 返回上一级目录
cd ..

:: 运行第一个python脚本
python3 .\malicious_detection.py

:: 本团队使用anaconda配置的环境，故有此行命令
:: C:\Work\anaconda\envs\python_for_ai_gpu\python.exe .\malicious_detection.py

:: 运行log_boxed.exe程序
.\log_boxed.exe