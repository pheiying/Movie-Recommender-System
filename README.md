# Movie-Recommender-System

## Table Of Contents
- [Introduction](#introduction)
- [Graphical User Interface (GUI)](#graphical-user-interface)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)
- [References](#references)

## Introduction

## Graphical User Interface (GUI)

## Methodology

### Data Collection

The "[ml-latest-small](./ml-latest-small)" dataset presents a comprehensive collection of movie ratings and tags, amassed from a diverse user base. The dataset comprises

- **9,742 movies**, encompassing various genres and time periods.
- **610 users**, who have actively participated in rating movies and tagging content.
- **1,589 unique tags**, indicating the varied preferences and tastes of the user base.
- A substantial number of **ratings**, amounting to **100,836**, which illustrates the extensive engagement of users with the platform.

### Recommendation Techniques
1. **Content-Based Filtering:**
2. **Memory-Based Collaborative Filtering:**
3. **Model-Based Collaborative Filtering:**

### Evaluation Method
1. **RMSE (Root Mean Square Error):** RMSE quantifies the average predictive error between actual and predicted ratings, with lower values indicating better accuracy.
2. **MAE (Mean Absolute Error):** MAE calculates the average absolute difference between predicted and actual ratings, with lower values signifying improved prediction accuracy.
3. **Hit Rate:** Hit Rate measures the proportion of recommended items that match user interactions or preferences.
4. **Coverage:** Coverage assesses the percentage of items in the catalogue that the recommender system can suggest.
5. **Novelty:** Novelty evaluates the uniqueness and diversity of recommendations to introduce users to unfamiliar items.
6. **Recall @k:** Recall @ k quantifies the fraction of relevant items recommended within the top-k list.
7. **Precision@k:** Precision @ k measures the accuracy of relevant items within the top-k recommendations.


## Results

### Root Mean Squared Error (RSME)
**User-based** and **Item-based** have the lowest RMSE, suggesting they are the most accurate in predicting exact ratings. 

**Year-based** and **Weighted year based** have the highest RMSE, implying they might not be as accurate in predicting ratings.

### Mean Absolute Error (MAE)
**User-based** and **item-based** excel, indicating they are adept at accurately predicting ratings. 

**Year-based** has the highest MAE, followed closely by combined content-based.

### Hit Rate
**SVD** has an outstandingly high hit rate, implying it’s most effective at suggesting movies users will interact with. 

**NCF** model an**User-based** have very low hit rates, suggesting users might not find their recommendations as engaging.

### Coverage
Almost all recommenders have a coverage of 1.0000 or near to 1.o. This means they can potentially recommend any movie in the dataset

**NCF model** has an unusually low coverage of 0.0001 or 0.01%. This means it can only recommend a tiny fraction of the available items.      

### Novelty
The **NCF model** stands out with an extraordinarily high novelty score, suggesting it recommends less popular items. 

**SVD** has the lowest novelty, implying it tends to suggest more popular or mainstream movies.

### Precision at K (P@k)
**SVD** dominates, implying that a high proportion of its top recommendations are items users have interacted with. 

The **NCF model** has the lowest precision, suggesting its top recommendations are rarely hits with users.

###  Recall at K (R@k)
**SVD** filtering excels, suggesting it's able to capture most of the items users have interacted with in its top recommendations.

**User-based** and **NCF model** filtering have very low recall, indicating they miss out on many movies users would interact with.

## License

This project forms part of an academic course and is intended solely for educational purposes. It may include references to copyrighted materials and any such materials are utilised exclusively for scholarly use. For guidance on sharing or distributing this work, it is advisable to seek consultation from your instructor or institution.

For more details, see the [LICENSE](./LICENSE.txt) file.

## Reference
**Dataset:** [ml-latest-small](./ml-latest-small)
