# WiFi安防器_隐私保护的智能安全监测与防护系统

2024年全国大学生信息安全竞赛作品赛

## 项目进展情况

- [x] **基于CSI 信号的抵近人员感知与预警技术：**利用无线网络中的信道状态信息（CSI）来感知监测区域内的人员活动。
- [x] **抗CSI窃听分析的室内情境隐私保护技术：**利用基于MIMOCrypt 的双AP 加密混淆隐私保护框架，通过在室内部署AP 设备，这些设备能够利用MIMO 技术对用户发出的信道矩阵进行加密并发射干扰信号
- [x] **隐私保护的行为追溯与审计技术：**使用预训练的CSI 人体行为识别模型，本系统能够对触发警报的恶意人员行为进行识别，这些识别信息为行为记录和报警等审计活动提供了关键支持。
- [x] **客户端：**使用Qt编写客户端，能够按照日期与动作进行查询审计。

## 目录结构说明

- 防护器运行脚本：`\protect_device\encrpt`
- CSI数据存储目录：`\data\`
- 数据预处理：`\data_preprocess\`
- 模型训练：
  - `attack_model_training`：用于训练CSI人体骨架分析模型，可用于攻击。
  - `detection_model_training`：用于训练CSI人体行为识别模型，用于行为追溯。
- 运行脚本：
  - `malicious_detection`：恶意行为探测的执行脚本。
- 客户端：
  - `\log\`：存放Qt客户端源代码。
  - `log_boxed.exe`：Windows端的追溯审计客户端。
- 测试：`test.bat`文件

## 使用与测试说明

**前提说明：**

- 需要安装好matlab，并将其添加至环境变量 Path 中
- 需要按照requirement.txt安装依赖包`pip install -r requirements.txt`

**直接运行：**

运行目录下`test.bat`文件进行测试。

**具体过程：**

1. 使用发射器与接收机进行WiFi数据包抓取，得到`example.pcap`文件。
2. 利用`data_preprocess/csireader.m`处理脚本，获得`data/example.csv`。
3. 运行`python3 malicious_detection.py`，数据将存储到`output/log.txt`中。
4. 使用`log_boxed.exe`进行查看。

