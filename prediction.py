#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


matches=pd.read_csv("C:\\Users\\jacki\\Downloads\\matches.csv")
matches.head()


# In[3]:


matches.shape


# In[4]:


# 2 seasons * 20 squads * 38 matches

2 * 20 * 38


# In[5]:


# Missing Liverpool 2021-2022
matches["team"].value_counts()


# In[6]:


matches[matches["team"] == "Liverpool"].sort_values("date")


# In[7]:


matches["round"].value_counts()


# In[8]:


matches.dtypes


# In[9]:


matches['date']= pd.to_datetime(matches['date'])


# In[10]:


matches.dtypes


# In[11]:


matches["venue_code"] = matches["venue"].astype("category").cat.codes


# In[12]:


matches["opp_code"] = matches["opponent"].astype("category").cat.codes


# In[13]:


matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")


# In[14]:


matches["day_code"] = matches["date"].dt.dayofweek


# In[15]:


matches["target"] = (matches["result"] == "W").astype("int")


# In[16]:


matches


# In[17]:


from sklearn.ensemble import RandomForestClassifier


# In[18]:


rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)


# In[19]:


train = matches[matches["date"] < '2022-01-01']


# In[20]:


test = matches[matches["date"] > '2022-01-01']


# In[21]:


predictors = ["venue_code", "opp_code", "hour", "day_code"]


# In[22]:


rf.fit(train[predictors], train["target"])


# In[23]:


preds = rf.predict(test[predictors])


# In[24]:


from sklearn.metrics import accuracy_score


# In[25]:


acc = accuracy_score(test["target"], preds)


# In[26]:


acc


# In[27]:


combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))


# In[28]:


pd.crosstab(index=combined["actual"], columns=combined["predicted"])


# In[29]:


from sklearn.metrics import precision_score


# In[30]:


precision_score(test["target"], preds)


# In[31]:


grouped_matches = matches.groupby("team")


# In[32]:


group = grouped_matches.get_group("Manchester City").sort_values("date")


# In[33]:


group


# In[34]:


def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


# In[35]:


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
rolling_averages(group, cols, new_cols)


# In[37]:


matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling


# In[42]:


matches_rolling.index = range(matches_rolling.shape[0])


# In[43]:


matches_rolling


# In[44]:


def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision


# In[45]:


combined, precision = make_predictions(matches_rolling, predictors + new_cols)
precision


# In[46]:


combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
combined


# In[47]:


class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)


# In[48]:


combined["new_team"] = combined["team"].map(mapping)


# In[49]:


merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
merged


# In[50]:


merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()


# In[ ]:




