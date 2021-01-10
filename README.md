## <center> Brunch-Article-Recommendation-System </center>

<p align="center"> <img src="https://user-images.githubusercontent.com/67793544/100312455-b7ad1a00-2ff5-11eb-87aa-75a57d8bfa07.png" width="15%"></p>

### Problem Description
Brunch is a platform for connecting people who loves reading and making contents. To make users find their taste, they designed article recommendation systems. For a better user experience, which can be made through personalized recommendation, they've been consistently developing their system. 

Now that many quality data is in Brunch, by using those, we need to make customized unique recommendation system.

### Objective of project
- the purpose of this project is to develop personalized contents recommendation system. 

#### Data Sources : [Kakao Arena](https://arena.kakao.com/c/6)
- data overview
    - articles(author, issue-date, title, contents, etc)
    - reader/ author information(read-date, read-article, following author list, etc) 
    
#### Dataset path settings

    res
    ├── read
    │   ├── ...
    │   └── ...
    └── predict
    │   ├── dev.users
    │   └── test.users
    ├── magazine.json
    ├── metadata.json
    └── users.json
     

### Project detail
#### 1. Period limitation
- we found that most consumption took place within 2 weeks of the publication of the article. Users tended to read more recently published articles. It was the result that reflects trends and seasonality of articles. Therefore, we decided to recommend articles to target users within two weeks of the issued date.

#### 2. Target segmentation
- The number of articles read by users during 1 months is divided into three groups using descriptive statistics.
    - group1 (0~7) : passive user (who read below average number of articles)
    - group2 (8~64): active user ( who read above average number of articles)
    - group3 (65~ ): domain worker, expert, crawler, etc. ( who read above upper-fence number of articles)

#### 3. Recommendation algorithms
- Articles of following author
    - 98 % of all users have author list who follow.
    - Each users follow an average of 8.6 authors. 
    - Recommend users recent article published by author they subscribe to.
    
- Articles of magazine
    - Article in magazines tends to be more read by users than not in.
    - New users are likely to read articles in Brunch magazines.
    - Recommend users another popular&recent articles in magazine where they read articles at least once.
    
- Articles based on similar tastes(collaborative filtering)
    - Tag list is the list that show content characteristic each articles has.
    - By aggregating and vectorizing tag list in all articles that each users read(tf-tidf vercotizer, Doc2vec.), we deduced their unique tastes.
    - Utlizing above tastes vector, we found the tastes similarity between reader and author, reader and reader.
    - Recommend users articles published by author and consumed by reader with similar tastes. 

- Articles of Popular & Recent
    - To resolve cold start, in case that the target users didn't read at least one article, we recommended popular & recent articles.

#### 4. Result of recommendation
<p align="left"><img src="https://user-images.githubusercontent.com/67793544/101129214-88755900-3644-11eb-8bb6-0cb91bec1e8c.png" width="60%"></p>
