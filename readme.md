![image-20210627213157359](figure/image-20210627213157359.png)

## Riiid AIEd Challenge 2020

> 在这个比赛中，任务是为“知识追踪”创建算法，即随着时间的推移对学生知识进行建模，其目的是准确预测学生在未来互动中的表现。即根据学生历史的答题表现预测学生未来答题表现，也可能出现没有历史表现，预测学生第一次答题的情况（冷启动问题）。赛题的评价指标是AUC。

### 数据介绍

+ `train.csv`

<img src="figure/image-20210627223604489.png" alt="image-20210627223604489" style="zoom: 75%;" />

+ `questions.csv`

<img src="figure/image-20210627225330637.png" alt="image-20210627225330637" style="zoom:75%;" />

+ `lecture.csv`

<img src="figure/image-20210627225539573.png" alt="image-20210627225539573" style="zoom:75%;" />

更多关于数据的介绍参考[这里](https://github.com/BITprogramMan/Kaggle_Riiid_Solution/blob/master/data_analysis.md)

### 数据划分方式

> 首先计算数据集中有多少个用户，选择5%的用户只出现在验证集中，45%的用户只出现在训练集中，剩下的用户随机划分，但是保证每个用户出现在验证集的时间晚于出现在训练集的时间

### 模型

> 本此比赛的模型主要分为两大类，一类是树模型lightgbm，一类是基于神经网络的模型。

#### 基于树模型

##### 特征工程

> **用户侧特征**
>
> + avg/median/max/std elapsed_time
> + avg/median/max/std elapsed_time by part
> + avg has_explanation flag
> + avg has_explanation flag by part
> + nunique of question, part, lecture
> + cumcount / timestamp
> + avg answered_correctly with recent 10/30/100/300 questions, recent 10min/7 days
> + avg answered_correctly by part, question, bundle, order of response, question difficulty, question difficulty x part
> + question difficulty: discretize the avg answered_correctly of all users for each question into 10 levels
> + cumcount by part, question, bundle, question difficulty, question difficulty x part
> + lag from 1/2/3/4 step before
> + correctness, lag, elapsed_time, has_explanation in the same question last time
>
> **question侧特征**
>
> + number of tags
> + one-hot encoding of tag (top-10 frequent tags)
> + SVD, LDA, item2vec using user_id x content_id matrix
> + LDA, item2vec using user_id x content_id matrix (filtered by answered_correctly == 0)
> + 10%, 20%, 50%, 80% elapsed time of all users response, correct response, wrong response
> + SAINT embedding vector + PCA
>
> ......
>
> 更多特征的介绍参考[这里](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210354)以及我的[总结](https://github.com/BITprogramMan/Kaggle_Riiid_Solution/blob/master/feature_engineering.md)（PS：排行榜第九名使用264个特征训练两个LightGBM，取得0.806的效果）

##### lightGBM





















#### 基于神经网络的模型

































