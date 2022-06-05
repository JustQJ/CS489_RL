各文件目录和文件含义如下

****

**imgs** 目录下为训练结果

**parameters** 目录下为保存的各个训练好的游戏模型

**policyBased.py** 中实现了策略模型

**valueBased.py** 中实现了值模型

**train.py** 用于训练实现的模型，直接运行可以进行8个游戏的训练

**running.py** 为测试训练好的模型，运行命令为

`python running.py --env_name BreakoutNoFrameskip-v4`

其中`BreakoutNoFrameskip-v4` 可以改为其他游戏名(不包括 BoxingNoFrameskip-v4 )。

