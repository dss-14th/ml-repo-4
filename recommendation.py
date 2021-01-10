import sys
sys.path.append("..")
from konlpy.tag import Mecab
from itertools import chain
from tqdm import tqdm
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BasicRecommendation:
    
    def __init__(self, train, users, target_info, popular_list):
        self.users = users
        self.train = train
        self.popular_list = popular_list
        self.target_info = target_info
        self.path = []
        
    
    ## following_list 있는 유저-> 구독하는 작가의 최신 글 추천
    def following_recom(self):
        """
        input: target data (target user id)
        output: 추천 리스트 
        """
        following = self.users[self.users["readers_id"]==self.target_id].following_list.values[0]
        article_list=[]
        for i in range(len(following)):
            article = self.train[self.train.author_id == following[i]][["article_id","reg_dt", "popular_weight"]].values
            if len(article) != 0:
                article_list.extend(article)

        # 최신 글, 인기 글 소팅 

        article_list = list(set([tuple(article) for article in article_list]))
        article_list.sort(key=lambda x: (x[2], x[1]), reverse=True)
        if len(article_list)> 100:
            article_list = article_list[:100]

        return article_list


    ## magazine_based recommandation
    def magazine_recom(self, target_id):
        magazine_list= list(set(self.train[self.train["readers_id"]==target_id].magazine_id.values))

        m_list=[]
        for i in tqdm(magazine_list):
            magazine_recom_list= self.train[self.train.magazine_id ==i].sort_values(['reg_dt', "popular_weight"], ascending=False)\
            [["article_id",'reg_dt',"popular_weight"]].values
            article_list = list(set([tuple(article) for article in magazine_recom_list]))
            m_list.extend(article_list)
        m_list.sort(key=lambda x: (x[1],int(x[2])), reverse=True)

        return m_list


    ### popularity based - 최신 글에서 가장 인기있는 글 추천
    def recent_popularity_recom(self, already_read, recom_num): 

        """
        (input) recom_num : 추천해줘야 하는 글 갯수
        (output) 추천 리스트
        """
        unread_popular_list= list(set(self.popular_list)-set(already_read))
        article_list = unread_popular_list[:recom_num]
        return article_list
    
    
    def morphs_analysis(self, data):
        mecab = Mecab(self.path)
        ls = []
        for x in tqdm(range(len(data))):
            ls.append(mecab.morphs(data[x]))
        return list(chain.from_iterable(ls))
        

    def tfidf_cosine(self, group_list):
        '''
        독자/작가별 태그리스트의 형태소를 분석하는 함수
        morphs_analysis 함수를 참조함
        (input) group_list : 분석대상 list
        (output) reader_key_sum, author_key_sum, target_key_sum : 
        reader와 author, target의 키워드 분석결과가 컬럼으로 추가되어 데이터프레임으로 반환됨
        '''
        tqdm.pandas()
        reader_key_sum = self.train.groupby('readers_id')['keyword_list'].agg(sum)
        reader_key_sum = reader_key_sum.reset_index()
        # 키워드 리스트가 없는 독자 삭제
        reader_key_sum = reader_key_sum[reader_key_sum['keyword_list'].apply(lambda x: len(x))!=0]
        # 형태소 분석
        reader_key_sum['morphs_list'] = reader_key_sum['keyword_list'].apply(lambda x: (' ').join(self.morphs_analysis(x)))

        author_key_sum = self.train.groupby('author_id')['keyword_list'].agg(sum)
        author_key_sum = author_key_sum.reset_index()
        # 키워드 리스트가 없는 독자 삭제
        author_key_sum = author_key_sum[author_key_sum['keyword_list'].apply(lambda x: len(x))!=0]
        # 형태소 분석
        author_key_sum['morphs_list'] = author_key_sum['keyword_list'].apply(lambda x: (' ').join(self.morphs_analysis(x)))
        group_list = pd.DataFrame(group_list, columns={'readers_id'})
        target_key_sum = reader_key_sum.merge(group_list, how='inner', on='readers_id')
        
        
        print('vectorizer...')
        tfidf_vectorizer = TfidfVectorizer()
        reader_vec = tfidf_vectorizer.fit_transform(reader_key_sum['morphs_list'])
        reader_vec = reader_vec.toarray()
        author_vec = tfidf_vectorizer.transform(author_key_sum['morphs_list'])
        author_vec = author_vec.toarray()
        target_vec = tfidf_vectorizer.transform(target_key_sum['morphs_list'])
        target_vec = target_vec.toarray()
        
        print('reader cosine_similarity...')
        reader_keyword_sim = cosine_similarity(reader_vec, target_vec)
        reader_keyword_sim_sorted_ind = reader_keyword_sim.argsort()[:,::-1]
        
        print('author cosine_similarity...')        
        reader_author_keyword_sim = cosine_similarity(reader_vec, author_vec)
        reader_author_keyword_sim_sorted_ind = reader_author_keyword_sim.argsort()[:,::-1]
        reader_key_sum = reader_key_sum.reset_index(drop=True)
        author_key_sum = author_key_sum.reset_index(drop=True)
        
        return reader_key_sum, author_key_sum, reader_keyword_sim_sorted_ind, reader_author_keyword_sim_sorted_ind
        

    def cosine_similarity_list(self, target_id, reader_key_sum, author_key_sum, reader_keyword_sim_sorted_ind, reader_author_keyword_sim_sorted_ind, top_n=10):
        '''
        독자/작가별 태그리스트의 형태소를 분석하는 함수
        morphs_analysis 함수를 참조함
        (input) group_list : 분석대상 list
        (output) reader_key_sum, author_key_sum : 
        reader와 author의 키워드 분석결과가 컬럼으로 추가되어 데이터프레임으로 반환됨
        '''
        
        reader_list = reader_key_sum[reader_key_sum['readers_id']==target_id]
        article_index = reader_list.index.values
        reader_similar_indexes = reader_keyword_sim_sorted_ind[article_index, 1:(top_n)+1]
        reader_similar_indexes = reader_similar_indexes.reshape(-1)
        
        author_similar_indexes = reader_author_keyword_sim_sorted_ind[article_index, 1:(top_n)+1]
        author_similar_indexes = author_similar_indexes.reshape(-1)

        return reader_key_sum.iloc[reader_similar_indexes].readers_id.values.tolist(), \
    author_key_sum.iloc[author_similar_indexes].author_id.values.tolist()
   

    def doc2vec(self):
        '''
        유사 독자를 찾아주는 doc2vec 만들기
        train2 data는 대상 구분한 초기 데이터 
        '''
        tqdm.pandas()
        reader_klist_df = self.train.groupby('readers_id')['keyword_list'].agg('sum')
        reader_klist_df = pd.DataFrame(reader_klist_df)
        reader_klist_df['keyword_list'] = reader_klist_df['keyword_list'].progress_apply(lambda x: " ".join(x))
        reader_klist_df.reset_index(inplace=True)
        reader_doc_df = reader_klist_df[['readers_id', 'keyword_list']].values.tolist()
        reader_tagged_data = [TaggedDocument(words=doc, tags=[readerid]) for readerid, doc in reader_doc_df]

        author_klist_df = self.train.groupby('author_id')['keyword_list'].agg('sum')
        author_klist_df = pd.DataFrame(author_klist_df)
        author_klist_df['keyword_list'] = author_klist_df['keyword_list'].progress_apply(lambda x: "".join(x))
        author_klist_df.reset_index(inplace=True)
        author_doc_df = author_klist_df[['author_id', 'keyword_list']].values.tolist()
        author_tagged_data = [TaggedDocument(words=doc, tags=[authorid]) for authorid, doc in author_doc_df]
        
        
        ### reader Doc2Vec model build
        max_epochs=10
        reader_model = Doc2Vec(window=10,
                       size=150,
                       alpha=0.025,
                       min_alpha=0.025,
                       min_count=2,
                       dm=1,
                       negative=3,
                       seed=9999)
        reader_model.build_vocab(reader_tagged_data)

        ### Doc2Vec model 학습
        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            reader_model.train(reader_tagged_data, 
                       total_examples=reader_model.corpus_count,
                       epochs=reader_model.iter)

            reader_model.alpha -= 0.002
            reader_model.min_alpha = reader_model.alpha
    
        reader_model.save('reader_model')
        print('reader model saved')

        # author Doc2Vec model build
        max_epochs=10
        author_model = Doc2Vec(window=10,
                       size=150,
                       alpha=0.025,
                       min_alpha=0.025,
                       min_count=2,
                       dm=1,
                       negative=3,
                       seed=9999)
        author_model.build_vocab(author_tagged_data)

        # Doc2Vec model 학습
        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            author_model.train(author_tagged_data, 
                       total_examples=author_model.corpus_count,
                       epochs=author_model.iter)

            author_model.alpha -= 0.002
            author_model.min_alpha=author_model.alpha
        
        author_model.save('author_model')
        print('author model saved')
        return reader_klist_df
           

class BrunchRecommendation(BasicRecommendation):
    
    def __init__(self, train, users, target_info, popular_list):
        super().__init__(train, users, target_info, popular_list)

        
    def total_recommendation(self, group_list):

        ### group_1 추천
        final_list=[]
        
        for target_id in tqdm(group_list):
            # (1.29~2.28) target user 가 읽은 글
            already_read = self.target_info[self.target_info["readers_id"]==target_id][["article_id","reg_dt", "popular_weight"]].values.tolist()
            already_read = list(set([tuple(read) for read in already_read]))
            """ following_list 추천 """
            try:
                recom_list = self.following_recom(target_id, train_38, users)
                recommended =list(set(recom_list)- set(already_read))
            except:
                recommended = []
                
            if len(recommended) >= 100:
                recommended = recommended[:100]
        
            if len(recommended) < 100:
                """ magazine_list 추천 """
                m_recom = self.magazine_recom(target_id)
                m_recom = list(set(m_recom)- set(already_read))
                recommended.extend(m_recom)
                
                if len(recommended) >= 100:
                    recommended = recommended[:100]

                """ recent_popular 추천 """
                if len(recommended) < 100:
                    recom_num = 100-len(recommended) 
                    p_recom= self.recent_popularity_recom(already_read, recom_num)
                    recommended.extend(p_recom)

                    if len(recommended) >= 100:
                        recommended = recommended[:100]
                    
            recommended = [recom[:][0] for recom in recommended]
            final_list.append({'target id' : target_id,
                               'recommendation': recommended})
            
        return final_list
        
        
    def total_recommendation_2(self, group_list, reader_key_sum, author_key_sum, reader_keyword_sim_sorted_ind, reader_author_keyword_sim_sorted_ind): 
        
        final_list=[]
        # group_2 추천
        # df-idf형태소 분석, 코사인 유사도 유사 독자, 작가 index 정보 가져오기
#         reader_key_sum, author_key_sum, reader_keyword_sim_sorted_ind, reader_author_keyword_sim_sorted_ind = self.tfidf_cosine(group_list)
        
        for target_id in tqdm(group_list):
            # (1.29~2.28) target user 가 읽은 글
            already_read = self.target_info[self.target_info["readers_id"]==target_id][["article_id", "reg_dt", "popular_weight"]].values.tolist()
            already_read = list(set([tuple(read) for read in already_read]))
            # id별로 유사한 reader, author list 뽑는 함수
            readers, authors = self.cosine_similarity_list(target_id, reader_key_sum, author_key_sum, reader_keyword_sim_sorted_ind, reader_author_keyword_sim_sorted_ind)
            
            # target_id 당 추천
            target_recommend_list = []
            ## 독자-작가 유사성을 통한 작가가 쓴 글 추천            
            for author in authors:
                author_recom = self.train[self.train['author_id'] == author].sort_values(by="reg_datetime", ascending=False)\
                [["article_id", "reg_dt","popular_weight"]].values.tolist()
                author_recom_list= list(set([tuple(article) for article in author_recom]))
                target_recommend_list.extend(author_recom_list)
                
            ## 독자-독자 유사성을 통한 작가가 쓴 글 추천
            for reader in readers:
                reader_recom = self.train[self.train.readers_id==reader]\
                [["article_id", "reg_dt", "popular_weight"]].values.tolist()
                reader_recom_list = list(set([tuple(article) for article in reader_recom]))
                target_recommend_list.extend(reader_recom_list)
                
            # sorting
            target_recommend_list.sort(key=lambda x: (x[2], x[1]), reverse=True)
            # 타겟 독자가 읽은 글 삭제
            final_recommend_list= list(set(target_recommend_list)-set(already_read))
            # 글 100개 넘으면 중단
            if len(final_recommend_list)> 100:
                final_recommend_list = final_recommend_list[:100]
    
            # 글 100개 안넘으면 인기-최신 글 추천 
            if len(final_recommend_list) < 100:
                recom_num = 100-len(final_recommend_list) 
                # 검증땐 Recom_1 사용, 추천땐 recom_2
                p_recom= self.recent_popularity_recom(already_read, recom_num)
                #p_recom= recent_popularity_recom_2(target_id, already_read, recom_num)
                final_recommend_list.extend(p_recom)
                if len(final_recommend_list) > 100:
                    final_recommend_list = final_recommend_list[:100]
                    break
                
            recommended = [recom[:][0] for recom in final_recommend_list]
            final_list.append({'target id': target_id,
                               'recommendation': recommended})      
                
        return final_list   
                
                               
    def total_recommendation_doc2vec(self, group_list): 
        
        final_list=[]
        # group_2 추천 
        
        reader_klist_df = self.doc2vec()
        
        # load model
        print('load model...')
        reader_model = Doc2Vec.load('reader_model')
        author_model = Doc2Vec.load('author_model')
        
        print('start recommendation for target user...')
        for target_id in tqdm(group_list):
            # (1.29~2.28) target user 가 읽은 글
            
            already_read= self.target_info[self.target_info["readers_id"]==target_id][["article_id", "reg_dt", "popular_weight"]].values.tolist()
            already_read = list(set([tuple(read) for read in already_read]))
            
            # target_id 당 추천
            target_recommend_list = []
                        
            ## 독자-작가 유사성을 통한 작가가 쓴 글 추천

            # reader id로 키워드 리스트 values값 추출
            reader_doc_list = reader_klist_df[reader_klist_df['readers_id'] == target_id]['keyword_list'].values
            author_inferred_vector = author_model.infer_vector(reader_doc_list)
            
            # top10 similar authors
            author_return_docs = author_model.docvecs.most_similar(positive=[author_inferred_vector], topn=11)
            authors=[]
            for rd in author_return_docs:
                authors.append(rd[0])
            authors = authors[1:]
            
            for author in authors:
                author_recom = self.train[self.train['author_id'] == author].sort_values(by="reg_datetime", ascending=False)\
                [["article_id", "reg_dt","popular_weight"]].values.tolist()
                author_recom_list= list(set([tuple(article) for article in author_recom]))
                target_recommend_list.extend(author_recom_list)
                
            ## top10 similar readers
            readers=[]
            reader_return_docs = reader_model.docvecs.most_similar(target_id, topn=11)
            for rd in reader_return_docs:
                readers.append(rd[0])  
            
            reders = readers[1:]
                
            for reader in readers:
                reader_recom = self.train[self.train['readers_id']==reader]\
                [["article_id", "reg_dt", "popular_weight"]].values.tolist()
                reader_recom_list = list(set([tuple(article) for article in reader_recom]))
                target_recommend_list.extend(author_recom_list)
                
            # sorting
            target_recommend_list.sort(key=lambda x: (x[2], x[1]), reverse=True)
            # 타겟 독자가 읽은 글 삭제
            final_recommend_list= list(set(target_recommend_list)-set(already_read))

            # 글 100개 넘으면 중단
            if len(final_recommend_list)> 100:
                final_recommend_list = final_recommend_list[:100]
            
            # 글 100개 안넘으면 인기-최신 글 추천 
            if len(final_recommend_list) < 100:
                recom_num = 100-len(final_recommend_list) 
                # 검증땐 Recom_1 사용, 추천땐 recom_2
                p_recom= self.recent_popularity_recom(already_read, recom_num)
                #p_recom= recent_popularity_recom_2(target_id, already_read, recom_num)
                final_recommend_list.extend(p_recom)
                if len(final_recommend_list) > 100:
                    final_recommend_list = final_recommend_list[:100]
                    break
            
            recommended = [recom[:][0] for recom in final_recommend_list]
            final_list.append({'target id': target_id,
                               'recommendation': recommended})            
                
        return final_list