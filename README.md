# rossmann-kaggle-udacity

---
## 使用的软件和库
我使用的操作系统是Win10 64位专业版, 项目中我使用R语言进行数据探索、可视化和特征工程等工作，R的版本是x64 3.4.3。需要安装的R包有：
- zoo 
- data.table
- rlist
- ggplot2
- rhdf5

我使用Python3.6 构建模型，需要安装的Python库有：
- numpy
- pandas
- matplotlib
- tensorflow
- keras
- xgboost
- lightgbm
- h5py

## 运行
1. 检查本项目应该包含以下子目录: googletrend, input, models, output, R, weather
2. 提取特征（如果直接使用output目录下的all_data.h5，则可以省略该步骤）： 
    - 改变工作目录至R子目录
    - 运行脚本data.R，这个过程大约需花费3到5分钟，它将在output目录下生成名为all_data.h5的特征文件。
3. 安装并配置Kaggle API, 参见 https://github.com/kaggle/kaggle-api
4. 简单XGBoost模型： ```python ross_xgb.py```
5. Entity-Embedding模型：
    - 修改ross_main.py的设置
    ```
    MODEL = NN_Embedding_Base
    N_NETWORKS = 1
    EPOCHS = 20
    ```
    - ```python ross_main.py```

6. EE-Residual模型：
    - 修改ross_main.py的设置
    ```
    MODEL = NN_Embedding
    N_NETWORKS = 1
    EPOCHS = 25
    ```
    - ```python ross_main.py```
    
7. EE-tree模型： ```python ross_ee_tree.py```
 
8. 最终提交的融合模型：
    - 修改ross_main.py的设置
    ```
    MODEL = NN_Embedding
    N_NETWORKS = 10
    EPOCHS = 25
    ```
    - ```python ross_main.py```

 ---
 运行截图
 ![](./Run.png)

 
 

