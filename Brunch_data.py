
# import
import numpy as np
import pandas as pd
import os
from datetime import timedelta, datetime
import glob
from itertools import chain
import json
import os
import warnings
warnings.filterwarnings('ignore')


# 1. Data 기본 전처리 및 [2019.01.29-2019.03.01]최신 기간 등록된 글로 Train 및 추천
class Data():
    def __init__(self, meta, read_file_list, magazine, users):
        self.meta = meta
        self.read_list = read_file_list
        self.magazine = magazine
        self.users = users
        
    def data_preprocessing(self):
        self.meta.drop(columns="article_id", inplace=True)
        self.meta.rename(columns={"user_id": "author_id", "id": "article_id"}, inplace=True)
        self.magazine.rename(columns={"id": "magazine_id"}, inplace=True)
        self.users.rename(columns={"id": "readers_id", "keyword_list": "search_keyword_list"}, inplace=True)

        read_df_lst = []
        for file in self.read_list:
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
        return self.meta, self.read, self.magazine, self.users

    ## 최신 기간에 등록된 글로만 추천 [2019.01.29-2019.03.01] (Train_data 만들기)
    def train_data(self):
        self.meta["reg_datetime"] = self.meta["reg_ts"].apply(lambda x: datetime.fromtimestamp(x/1000.0))
        self.meta["reg_dt"] = self.meta["reg_datetime"].dt.date
        self.meta.drop(columns=["display_url", "sub_title", "title"], inplace=True)

        metadata_train = self.meta[(self.meta.reg_datetime >= datetime(2019, 1, 29))&(self.meta.reg_datetime <= datetime(2019, 3, 1))]
        
        self.train = pd.merge(left=metadata_train, right=self.read, left_on="article_id", right_on="article_id", how="inner")

        self.train.drop(columns=["reg_ts", "reg_datetime"], inplace=True)
        self.train.rename(columns={"dt":"read_dt", "hr":"read_hr"}, inplace=True)

        return self.train
    
    ## train data에 article_id별 keyword_list 뽑기
    def train_keyword_list(self):
        self.keyowrd_list = self.train[['article_id', 'keyword_list']]
        self.keyowrd_list.drop_duplicates('article_id', ignore_index=True, inplace=True)
        
        return self.keyowrd_list

    
# 2. Train data에서 readers별 읽은 article 건수 및 list DataFrame 생성 및 upper_fense 이상치 제거
class Read_article_outline_remove():
    def __init__(self, train):
        self.train = train
    
    ## readers_id 별 article_id 건수 및 list Dataframe 생성
    def read_article_list(self):
        self.train.article_id = self.train.article_id+" "

        self.readers_article_list = self.train.groupby("readers_id")["article_id"].sum().reset_index()
        self.readers_article_list["article"] = self.readers_article_list.article_id.apply(lambda x: x.split(" "))
        self.readers_article_list["article"] = self.readers_article_list.article.apply(lambda x: x[:-1])

        ### readers_id 별 리스트 안의 article_id의 중복 제거해주기
        self.readers_article_list["article_list"]=self.readers_article_list.article.apply(lambda x: list(set(x)))
        self.readers_article_list.drop(columns=["article", "article_id"], inplace=True)

        self.readers_article_list["article_id_count"]=self.readers_article_list.article_list.apply(lambda x: len(x))

        return self.readers_article_list

    ## upper_fence기준 이상치 제거
    def upper_fence_remove(self):
        iqr = np.percentile(self.readers_article_list["article_id_count"], 75) - np.percentile(self.readers_article_list["article_id_count"], 25)
        self.upper_fence_df = self.readers_article_list[self.readers_article_list.article_id_count < 11+iqr*1.5]

        return self.upper_fence_df
        
    
# 3. readers_article_list 건수별 나누기 [1: 1건, 2: 2-4건, 3: 5-11건, 4: 12-25건]
def article_count_division_1(data):
    return data[data["article_id_count"] == 1]
def article_count_division_2(data):
    return data[(data["article_id_count"] >= 2) & (data["article_id_count"] <= 4)]
def article_count_division_3(data):
    return data[(data["article_id_count"] >= 5) & (data["article_id_count"] <= 11)]
def article_count_division_4(data):
    return data[(data["article_id_count"] >= 12) & (data["article_id_count"] <= 25)]


# 4. readers_article_list Data와 다른 Data의 merge를 통한 새로운 DataFrame 생성
class New_data():
    def __init__(self, data):
        self.data = data
        
    ## readers_article_list와 article별 magazine_id 데이터 생성
    def ra_magazine_df(self, read, meta, maga):
        self.read = read
        self.meta = meta
        self.maga = maga
        
        df1 = pd.merge(left=self.read, right=self.meta, left_on="article_id", right_on="article_id", how="inner")
        df2 = pd.merge(left=self.data, right=df1, left_on="readers_id", right_on="readers_id", how="left")
        self.df3 = pd.merge(left=df2, right=self.maga, left_on="magazine_id", right_on="magazine_id", how="left")
        return self.df3
        
    ## readers_article_list와 readers별 searching_keyword, Following_list 데이터 생성
    def ra_searching_following_df(self, users):
        self.users = users
        self.df4 = pd.merge(left=self.data, right=self.users, left_on="readers_id", right_on="readers_id", how="left")
        return self.df4

    ## readers_article_list와 article별 magazine_id, readers별 searching_keyword, Following_list 데이터 생성
    def ra_total_df(self):
        df5 = pd.merge(left=self.df3, right=self.df4, left_on="readers_id", right_on="readers_id", how="inner")
        return df5
