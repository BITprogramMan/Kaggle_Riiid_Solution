## 特征工程

### **用户/用户属性侧特征(User)**

+ 用户的id
+ 用户回答问题的次数,不同问题数, 回答问题的准确率;(全部历史记录中以及最近的某个时间窗口内);
+ 用户最近回答问题准确率的趋势；

### **问题/问题属性侧特征(Question)**

+ 问题属性,tags,part,index in bundle等;
+ 问题的难度,问题的回答准确率,被回答的次数;
+ 相同问题被回答的距离;
+ 相同问题被回答的时间差的统计特征;
+ 相同问题Tag被回答的距离;
+ 相同问题Tag被回答的时间差的统计特征;

### **用户问题/问题属性特征(UQ)**

+ mean of mean correctness of content that certain user has done before, which judges the difficulty of tasks that users have done.
+ 用户回答问题错误的题目的难度的分布特征

```python
tmp_df = for_question_df.groupby('content_id')['answered_correctly'].agg([['corr_ratio', 'mean']]).reset_index()
tmp_fe = for_question_df[for_question_df['answered_correctly']==0].merge(tmp_df, on='content_id').groupby('user_id')['corr_ratio'].agg(['min', 'max', 'mean', 'std']).reset_index()
for_train = for_train.merge(tmp_fe, on='user_id', how='left') 
```

+ 用户回答问题的elapsed time,lag time以及二者的ratio的统计特征,可以分answer_corretly=1/0；

```python
user_task_timestamp = q_logs[['user_id', 'task_container_id', 'timestamp']].drop_duplicates()
user_task_timestamp['lag_time'] = user_task_timestamp['timestamp'] - user_task_timestamp.groupby(['user_id'])['timestamp'].shift(1)
tmp = q_logs[['user_id', 'task_container_id', 'content_id', 'answered_correctly']].merge(user_task_timestamp.drop(['timestamp'], axis=1), on=['user_id', 'task_container_id'], how='left')
group_df = tmp[tmp['answered_correctly']==1].groupby(['content_id'])['lag_time'].agg([['c_lag_time_mean', 'mean'],
                                                                                     ['c_lag_time_std', 'std'],
                                                                                     ['c_lag_time_max', 'max'],
                                                                                    ['c_lag_time_min', 'min'],
                                                                                     ['c_lag_time_median', 'median']]).reset_index()
train = train.merge(group_df, on=['content_id'], how='left')

group_df = tmp[tmp['answered_correctly']==0].groupby(['content_id'])['lag_time'].agg([['w_lag_time_mean', 'mean'],
                                                                                       ['w_lag_time_std', 'std'],
                                                                                       ['w_lag_time_max', 'max'],
                                                                                       ['w_lag_time_min', 'min'],
                                                                                       ['w_lag_time_median', 'median']]).reset_index()
train = train.merge(group_df, on=['content_id'], how='left')
```

+ 用户回答问题的QQ(Question-Question matrix)矩阵的统计特征;
+ 用户对于同一个问题的历史回答情况，例如：最近的准确率，平均准确率，过去尝试的次数，距离上一次的尝试次数，用户是否在上次尝试之后查看了解释；
+ 用户当前的question和历史correct和incorrect的question的向量的乘积；
+ 用户关于每个问题的回答率的百分比的统计特征；
+ 用户回答相同Tags问题的elapsed time,lag time以及二者的ratio的统计特征；
+ 最后一次用户回答相同Tags问题的timestamp的时间差，回答的准确率，回答题目数以及平均每题回答时间。

```python
 用户在整体的情况下的时间戳的排名差
for_question_df['rank_part']  = for_question_df.groupby(['user_id', 'part'])['timestamp'].rank(method='first')
for_question_df['rank_user']  = for_question_df.groupby(['user_id'])['timestamp'].rank(method='first')
for_question_df['rank_diff']  = for_question_df['rank_user'] - for_question_df['rank_part']

# rank的排名差
for_question_df['part_times'] = for_question_df.groupby(['user_id', 'part'])['rank_diff'].rank(method='dense')
for_question_df['rank_diff']  = for_question_df.groupby(['user_id', 'part'])['rank_diff'].rank(method='dense', ascending=False)

# 最近一次的部分
last_part      = for_question_df[for_question_df['rank_diff']==1]
# 最近用户某个part题目做的次数
part_times     = for_question_df.groupby(['user_id', 'part'])['part_times'].agg([['part_times', 'max']]).reset_index()

# 最后一次做了某个part做了几个题，准确率多少
last_part_df   = last_part.groupby(['user_id', 'part'])['answered_correctly'].agg([['last_continue_part_ratio', 'mean'], ['last_continue_part_cnt', 'count']]).reset_index()
last_part_time = last_part.groupby(['user_id', 'part'])['timestamp'].agg([['last_continue_part_time_start', 'min'], ['last_continue_part_time_end', 'max']]).reset_index()
last_part_df   = last_part_df.merge(last_part_time, on=['user_id', 'part'], how='left')
last_part_df   = last_part_df.merge(part_times, on=['user_id', 'part'], how='left')

# 平均每个题目使用的时间
last_part_df['part_time_diff'] = last_part_df['last_continue_part_time_end'] - last_part_df['last_continue_part_time_start']
last_part_df['part_time_freq'] = last_part_df['last_continue_part_cnt']/last_part_df['part_time_diff']

for_train = for_train.merge(last_part_df, on=['user_id', 'part'], how='left')
for_train['last_continue_part_time_start'] = for_train['timestamp'] - for_train['last_continue_part_time_start']
for_train['last_continue_part_time_end'] = for_train['timestamp'] - for_train['last_continue_part_time_end']
```

+ 自用户观看带有相同Tag的Lecture以来的时间(最近的时间差)，与Lecture持续时间相比，观看Lecture的时间（来自Ednet数据）
+ 新用户刚刚开始的问题都是类似的，对这些类似的问题的准确性进行信息统计