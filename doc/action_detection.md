# action_detection

单人动作识别（就一个人就够了，忽略多人什么的逻辑）：

1. 特征提取：rtmlib
2. 时序模型：Continual ST-GCN
3. 输出当前动作 + 置信度

训练样本是 data/train/摇摆大拇指 里面的视频，就是需要识别摇摆大拇指
pyside6 实时预览手的骨架，画面左上角现实当前动作和置信度，相机驱动 device/ 里面
