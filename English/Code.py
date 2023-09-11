#######################
# Potential Customer Revenue Calculation with Rule-Based Classification
#######################

#######################
# Business Problem
#######################
# New level-based customer definitions (personas) using some characteristics of a game company's customers
# Create segments according to these new customer definitions and identify new customers who may come to the company according to these segments.
# Wants to estimate how much money it can earn on average.

# For example It is desired to determine how much money a 25-year-old male user from Turkey, who is an IOS user, can earn on average.

#######################
# Dataset Story
#######################
# Persona.csv data set shows the prices of the products sold by an international game company and some of the users who purchased these products.
# contains demographic information. The data set consists of records created in each sales transaction. This means table
# is not deduplicated. In other words, a user with certain demographic characteristics may have made more than one purchase.

# Price: Customer's spending amount
# Source: The type of device the customer is connected to
# Sex: Customer's gender
# Country: Customer's country
# Age: Customer's age

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# SOLUTION
#############################################

#############################################
# Step-1: Understanding the Dataset
#############################################

# Define the persona.csv file and show general information about the data set.

import pandas as pd
pd.set_option("display.max_rows", None)
df = pd.read_csv('datasets/persona.csv')
df.head()
df.shape
df.info()

# How many unique SOURCE are there? What are their frequencies?

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# How many unique PRICE are there?
df["PRICE"].nunique()

# How many sales were made from which PRICE?
df["PRICE"].value_counts()

# How many sales were made from which COUNTRY?
df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()

df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="count")


# How much was earned from sales in total by country?
df.groupby("COUNTRY")["PRICE"].sum()
df.groupby("COUNTRY").agg({"PRICE": "sum"})

df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="sum")

# What are the sales numbers according to SOURCE types?
df["SOURCE"].value_counts()

# What are the PRICE averages according to Countries?
df.groupby(by=['COUNTRY']).agg({"PRICE": "mean"})

# What are the PRICE averages according to SOURCES?
df.groupby(by=['SOURCE']).agg({"PRICE": "mean"})

# What are the PRICE averages in the COUNTRY-SOURCE breakdown?
df.groupby(by=["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})

#############################################
# Step-2: Preparing the Data
#############################################

df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).head()
# Thanks to this process, all combinations of the variables "COUNTRY", 'SOURCE', "SEX", "AGE" are observed.
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()
agg_df = agg_df.reset_index()


#Convert the AGE variable to a categorical variable and add it to agg_df.

# Convert the numeric variable Age into a categorical variable.
# Create the intervals in a way that you think will be convincing.
# For example: '0_18', '19_23', '24_30', '31_40', '41_+'

# Let's specify where the AGE variable will be divided:
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Let's express what the naming will be in response to the divided points:
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_+']

agg_df["AGE_INTERVAL"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()

#######################
# Step-3: Define new level based customers and add them to the data set as a variable.
#######################
# Define a variable called CUSTOMER_LEVEL and add this variable to the data set.
# Attention!
# After creating CUSTOMER_LEVEL values with list comp, these values need to be deduplicated.
# For example, there may be more than one of: USA_ANDROID_MALE_0_18
# It is necessary to take these to groupby and get the price averages.

agg_df['CUSTOMER_LEVEL'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGE_INTERVAL']].agg(lambda x: '_'.join(x).upper(), axis=1)
agg_df.head()


# Remove unnecessary variables:
agg_df = agg_df[["CUSTOMER_LEVEL", "PRICE"]]
agg_df.head()

for i in agg_df["CUSTOMER_LEVEL"].values:
    print(i.split("_"))


# There's a little problem here. There will be many of the same segments.
agg_df["CUSTOMER_LEVEL"].value_counts()

# For this reason, after groupby by segments, we should take the price averages and deduplicate the segments.
agg_df = agg_df.groupby("CUSTOMER_LEVEL").agg({"PRICE": "mean"})

# It is located in the CUSTOMER_LEVEL index. Let's turn this into a variable.
agg_df = agg_df.reset_index()
agg_df.head()


#############################################
# Create segments
#############################################
# Segment by PRICE,
# add the segments to agg_df with the name "SEGMENT",

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})


#############################################
# Step-4: Classify new customers and estimate how much revenue they can bring.
#############################################
# To which segment does a 33-year-old Turkish woman using ANDROID belong and how much income is she expected to earn on average?
new_user = "tur_android_female_31_40"
agg_df[agg_df["CUSTOMER_LEVEL"] == new_user]

# In which segment and how much income on average is a 35-year-old French woman using IOS expected to earn?
new_user = "fra_ios_female_31_40"
agg_df[agg_df["CUSTOMER_LEVEL"] == new_user]
