@echo off

:: �л���data_preprocessĿ¼
cd .\data_preprocess\

:: ʹ���޽����Matlab����csireader�ű�
matlab -nojvm -nodesktop -nodisplay -r csireader
:: �ȴ�����
echo �ȴ�matlab�ű��ɹ�����
:: �ȴ�����
timeout /t 10

:: ������һ��Ŀ¼
cd ..

:: ���е�һ��python�ű�
python3 .\malicious_detection.py

:: ���Ŷ�ʹ��anaconda���õĻ��������д�������
:: C:\Work\anaconda\envs\python_for_ai_gpu\python.exe .\malicious_detection.py

:: ����log_boxed.exe����
.\log_boxed.exe