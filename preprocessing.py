# import
import numpy as np
import pandas as pd
import os
from datetime import timedelta, datetime
import glob
from itertools import chain
import json
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


# 1. Data 기본 전처리 및 [2019.02.15-2019.03.31]최신 기간 등록된 글로 Train 및 추천
class Data():
    def __init__(self, meta, read_file_list, magazine, users, validation=True):
        self.meta = meta
        self.read_list = read_file_list
        self.magazine = magazine
        self.users = users
        self.validation = validation
        
    def data_preprocessing(self):
        self.meta.drop(columns="article_id", inplace=True)
        self.meta.rename(columns={"user_id": "author_id", "id": "article_id"}, inplace=True)
        self.meta["type"] = self.meta["magazine_id"].apply(lambda x: "개인" if x == 0.0 else "매거진")
        self.meta["reg_datetime"] = self.meta["reg_ts"].apply(lambda x: datetime.fromtimestamp(x/1000.0))
        self.meta["reg_dt"] = self.meta["reg_datetime"].dt.date
        self.meta.drop(columns="reg_ts", inplace=True)
        
        self.magazine.rename(columns={"id": "magazine_id"}, inplace=True)
        self.users.rename(columns={"id": "readers_id", "keyword_list": "search_keyword_list"}, inplace=True)
        

        read_df_lst = []
        for file in tqdm(self.read_list):
            file_name = os.path.basename(file)
            file_df = pd.read_csv(file, header=None, names=["raw"])
            file_df["dt"] = file_name[:8]
            file_df["hr"] = file_name[8:10]
            file_df["readers_id"] = file_df["raw"].str.split(" ").str[0]
            file_df["article_id"] = file_df["raw"].str.split(" ").str[1:].str.join(" ").str.strip()
            read_df_lst.append(file_df)
        read_df = pd.concat(read_df_lst)
        read_count = read_df["article_id"].str.split(" ").map(len)
        self.read = pd.DataFrame({"dt": np.repeat(read_df["dt"], read_count),
                             "hr": np.repeat(read_df["hr"], read_count),
                             "readers_id": np.repeat(read_df["readers_id"], read_count),
                             "article_id": list(chain.from_iterable(read_df["article_id"].str.split(" ")))})
        
        self.read["date"] = pd.to_datetime(self.read["dt"])
        del read_df, read_count
        
        return self.meta, self.read, self.magazine, self.users
    
    ## Test, Train 만들기 위해 read, metadata 기간별 자르기
    def cut_data(self, dev):
        if self.validation is True:
            # 2019.2.7-2019.2.28 read_df 원본 데이터
            self.read_cut = self.read[(self.read.date >= datetime(2019, 2, 7))&(self.read.date <= datetime(2019, 2, 28))]
            self.read_cut.reset_index(drop=True, inplace=True)
    
            # 2019.2.22-2019.2.28 self.read에서 dev 제거 데이터
            dev_none = self.read_cut[self.read_cut.readers_id.isin(list(dev.readers_id))]
            dev_none2 = dev_none[(dev_none.date >= datetime(2019, 2, 22))&(dev_none.date <= datetime(2019, 2, 28))]
            self.read_nondev = self.read_cut[~self.read_cut.index.isin(list(dev_none2.index))]

            # 2019.2.7-2019.2.28 self.meta 원본 데이터
            self.meta_cut = self.meta[(self.meta.reg_datetime >= datetime(2019, 2, 7))&(self.meta.reg_datetime <= datetime(2019, 2, 28))]
        else: 
            # 2019.2.7-2019.3.08 read_df 원본 데이터
            self.read_cut = self.read[(self.read.date >= datetime(2019, 2, 7))&(self.read.date <= datetime(2019, 3, 8))]
            self.read_cut.reset_index(drop=True, inplace=True)
    
            # 2019.3.01-2019.3.08 self.read에서 dev 제거 데이터
            dev_none = self.read_cut[self.read_cut.readers_id.isin(list(dev.readers_id))]
            dev_none2 = dev_none[(dev_none.date > datetime(2019, 2, 28))&(dev_none.date <= datetime(2019, 3, 8))]
            self.read_nondev = self.read_cut[~self.read_cut.index.isin(list(dev_none2.index))]
            
        # 2019.2.7-2019.3.8 self.meta 원본 데이터
        self.meta_cut = self.meta[(self.meta.reg_datetime >= datetime(2019, 2, 7))&(self.meta.reg_datetime <= datetime(2019, 3, 8))]
           
        return self.read_cut, self.read_nondev, self.meta_cut
    
    ## 2019.2.7-2019.2.28기간 users데이터, dev데이터로 만들어진 target_info data 생성
    def target_info_load(self, dev):
        self.dev = dev
        """
        Target_info DataFrame is divided into users and dev and divided by number of cases
        """
        train = pd.merge(self.meta_cut, self.read_nondev, how="left", left_on='article_id',right_on="article_id")   
        self.train_meta = popular_weight_data(train)
        train_dev = pd.merge(self.dev, self.train_meta, how="left", left_on="readers_id", right_on="readers_id")
        
        self.target_info = pd.merge(train_dev, self.users, how="left", left_on="readers_id",
                               right_on="readers_id")
        return self.target_info, self.train_meta
    
    ## train data에 article_id별 keyword_list 뽑기
    def train_keyword_list(self, train):
        self.keyword = train[['article_id', 'keyword_list']]
        self.keyword_list.drop_duplicates('article_id', ignore_index=True, inplace=True)
        
        return self.keyword_list
    

    
# 2. train들어올때 건수별 나누기 만들기   

# def target_info_load(dev):
#     dev = dev
#     """
#     Target_info DataFrame is divided into users and dev and divided by number of cases
#     """
#     train = pd.merge(meta_cut, read_nondev, how="left", left_on='article_id',right_on="article_id")   
#     train_meta = popular_weight_data(train)
#     train_dev = pd.merge(dev, train_meta, how="left", left_on="readers_id", right_on="readers_id")

#     target_info = pd.merge(train_dev, users, how="left", left_on="readers_id",
#                            right_on="readers_id")
#     return target_info, train_meta


def train_article_count_division(target_info, users, start_count, stop_count):
    """
    function : division target_info dataframe by number of read.
               Only the readers_id divided by the number of cases from the users dataframe is extracted.

    input : target_info, users, start_count, stop_count
    output : group_users

    => unread : 0
    => min-50% : 1-7 (1 / 2-7)
    => 50%-upper fence : 8-64 (8-27 / 28-64)
    => upper-fence-max : 65-21059
    """
    read_count = target_info.groupby('readers_id').count().article_id
    group_list = read_count[(read_count>=start_count)&(read_count < stop_count)].index
    group_meta = target_info[target_info["readers_id"].isin(group_list)]
    group_users = users[users["readers_id"].isin(group_list)]
        
    non_list = list(set(group_meta.readers_id.unique().tolist())-set(group_users.readers_id.values.tolist()))
    df = []
    for i in tqdm(non_list):
        ls = {"search_keyword_list":[],"following_list":[],"readers_id":i}
        df.append(ls)
            
    df= pd.DataFrame(df)
    group_users= pd.concat([group_users, df])
    group_users_list = group_users.readers_id.values.tolist()
    return group_users, group_users_list


# 3. 인기글 가중치 추가 함수(각 글당 독자들의 소비 수) 
def popular_weight_data(train):
    popular = train.groupby("article_id").readers_id.nunique().sort_values(ascending=False)
    for i in tqdm(popular.index):
        train.loc[train["article_id"]== i,"popular_weight"] = popular[i]
    return train

# 4. 추천에 필요한 데이터 추가 생성: 최근 글의 인기순, 최근 순을 정렬한 글 리스트
def recent_popularity_list(train):
    # validation: 2.07~2.28 일까지 인기> 최신 순으로 내림차순 정렬되어 있는 리스트 생성
    # recommendation: 2.07~3.08 일까지 인기> 최신 순으로 내림차순 정렬되어 있는 리스트 생성
    recent_popularity = train[["article_id","reg_dt",'popular_weight']].values.tolist()
    popular_list = list(set([tuple(article) for article in recent_popularity]))
    popular_list.sort(key=lambda x: (int(x[2]),x[1]), reverse=True)
    return popular_list
