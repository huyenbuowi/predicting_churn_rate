### FINAL PROJECT
### HUYEN NGUYEN
### BSAN 360 - 130
## Title: Predict Credit Card Churn rate and Customer Segmentation
## Date: Mar 22nd, 2023

library(tidyverse)
library(caret)
library(glmnet)
library(pROC)
### PART 1: LOAD, EXAMINE AND TRANSFORM THE DATA
source("prepData.R")
churn.df <- prepData("BankChurners.csv")
## Rename the last 2 columns (they are Naive Bayes index)
colnames(churn.df)[22]="Naive Bayes 1"
colnames(churn.df)[23]="Naive Bayes 2"
## Find out if there is any missing values (Result: No missing values)
sum(is.na(churn.df))
## Convert strings to numerical/categorical variables
churn.tf<-churn.df
# Attrition Flag
i<-1
for (i in 1: 10127){
  if(churn.df$Attrition_Flag[i]=="Existing Customer")
 {churn.tf$Attrition_Flag[i]=1}else{churn.tf$Attrition_Flag[i]=0}}

# Gender
i<-1
for (i in 1: 10127){
  if(churn.df$Gender[i]=="F")
  {churn.tf$Gender[i]=1}else{churn.tf$Gender[i]=0}}

#Educational_Level
i<-1
for (i in 1: 10127){
  if(churn.df$Education_Level[i]=="Unknown")
  {churn.tf$Education_Level[i]=0}
  if(churn.df$Education_Level[i]=="Uneducated")
  {churn.tf$Education_Level[i]=1}
  if(churn.df$Education_Level[i]=="High School")
  {churn.tf$Education_Level[i]=2}
  if (churn.df$Education_Level[i]=="College")
  {churn.tf$Education_Level[i]=3}
  if(churn.df$Education_Level[i]=="Graduate")
  {churn.tf$Education_Level[i]=4}
  if(churn.df$Education_Level[i]=="Post-Graduate")
  {churn.tf$Education_Level[i]=5}
  if(churn.df$Education_Level[i]=="Doctorate")
  {churn.tf$Education_Level[i]=6}}

#Marital_Status
i<-1
for (i in 1: 10127){
  if(churn.df$Marital_Status[i]=="Unknown")
  {churn.tf$Marital_Status[i]=0}
  if(churn.df$Marital_Status[i]=="Single")
  {churn.tf$Marital_Status[i]=1}
  if(churn.df$Marital_Status[i]=="Married")
  {churn.tf$Marital_Status[i]=2}
  if(churn.df$Marital_Status[i]=="Divorced")
  {churn.tf$Marital_Status[i]=3}}

# Income Category
i<-1
for (i in 1: 10127){
  if(churn.df$Income_Category[i]=="Unknown")
  {churn.tf$Income_Category[i]=0}
  if(churn.df$Income_Category[i]=="Less than $40K")
  {churn.tf$Income_Category[i]=1}
  if(churn.df$Income_Category[i]=="$40K - $60K")
  {churn.tf$Income_Category[i]=2}
  if(churn.df$Income_Category[i]=="$60K - $80K")
  {churn.tf$Income_Category[i]=3}
  if(churn.df$Income_Category[i]=="$80K - $120K")
  {churn.tf$Income_Category[i]=4}
  if(churn.df$Income_Category[i]=="$120K +")
  {churn.tf$Income_Category[i]=5}}

# Card Category
i<-1
for (i in 1: 10127){
  if(churn.df$Card_Category[i]=="Blue")
  {churn.tf$Card_Category[i]=1}
  if(churn.df$Card_Category[i]=="Silver")
  {churn.tf$Card_Category[i]=2}
  if(churn.df$Card_Category[i]=="Gold")
  {churn.tf$Card_Category[i]=3}
  if(churn.df$Card_Category[i]=="Platinum")
  {churn.tf$Card_Category[i]=4}}

# Turn them to numeric
churn.tf[,2:9] <- lapply(churn.tf[,2:9], as.integer)

## Turn the old columns into factors (in case we need it)
churn.df[,c(2,4:9)] <- lapply(churn.df[,c(2,4:9)], as.factor)

## Using scatterplotMatrix(), draw a scatterplot matrix of the data
library(car)
# I will remove all the factors/characters and the Client Number (CLIENTNUM) 
# from the scatterplot  matrix.
# As it is overwhelmed to include all data in one matrix, I will break it 
#  down into3 matrix that group the data into 3 big categories (demographic details,
# customer's relationship with card provider, and customer's spending behavior).
# However, as we are mainly examine churn rate, we will include the variable 
# Attrition_Flag in all the matrix.
attach(churn.tf)
## Demographic matrix
scatterplotMatrix(formula = ~Attrition_Flag+Customer_Age+Dependent_count, 
                  data=churn.tf,diagonal="histogram")
## Customer's relationship with card provider
scatterplotMatrix(formula = ~ Attrition_Flag+ Months_on_book+Total_Relationship_Count+
                    Months_Inactive_12_mon+Contacts_Count_12_mon, data=churn.tf, 
                  diagonal="histogram")
# There is no clear relationship between Attrition Flag and other variables.
# We can see more clearly with jitter below.
# No variable here needs transformation as they are discrete data.

# Customer's spending behavior
scatterplotMatrix(formula = ~ Attrition_Flag+Credit_Limit+Total_Revolving_Bal+
                    Avg_Open_To_Buy+Total_Amt_Chng_Q4_Q1+Total_Trans_Amt+
                    Total_Trans_Ct+Total_Ct_Chng_Q4_Q1+Avg_Utilization_Ratio,
                  data=churn.tf, diagonal="histogram")

## Transform continuous data by Box-Cox 
# Credit_Limit: negative correlation with Avg_Utilization Ratio, positive linear correlation
# with Avg_Open_To_Buy
for (i in 14:21){
  c <- coef(powerTransform(churn.df[,i]+1))
  churn.tf[,i]<-bcPower(churn.df[,i]+1,c)
}
attach(churn.tf)
## Test if the transformation makes a better histogram for the data
library(cowplot)
library(ggplot2)
#Credit Limit
plot1 <-ggplot()+geom_histogram(data=churn.df, aes(x=Credit_Limit),
fill="pink")+labs(title="Credit Limit", x="Credit Limit",y="Value")

plot2 <-ggplot()+geom_histogram(data=churn.tf, aes(x=Credit_Limit),
fill="lightblue")+labs(title="Credit Limit (Box-Cox)", x="Credit Limit",y="Value")
plot_grid(plot1,plot2)
## The shape of the histogram become closer to normal distribution after the 
## transformation.

## Correlation Matrix (numeric data)
library(corrplot)
cormatrix<-cor(churn.tf[,2:21])
cormatrix
corrplot.mixed(cor(churn.tf[,2:21]),lower.col=colorRampPalette(c("red","gray",
"navy"))(50), upper.col=colorRampPalette(c("red","gray","navy"))(50))

## Test the most positive and negative correlations 
# Most positive correlation
cor.test(Credit_Limit,Avg_Open_To_Buy) 
# statistically significant correlation
# Most negative correlation
cor.test(churn.tf$Gender,churn.tf$Income_Category) 
# statistically significant correlation

## Jitter plot and polychoric test
library(psych)
jitter1 <- ggplot()+ geom_jitter(data = churn.tf, 
                                 aes(x=Total_Relationship_Count, y=Months_Inactive_12_mon))
jitter1
polychoric(cbind(Total_Relationship_Count,Months_Inactive_12_mon))
#correlation=0 => no relationship between these 2 variables
jitter2 <- ggplot()+ geom_jitter(data = churn.tf, 
                                 aes(x=Total_Relationship_Count, y=Contacts_Count_12_mon))
jitter2
polychoric(cbind(Total_Relationship_Count,Contacts_Count_12_mon))
#correlation=0.06 => small positive relationship between these 2 variables
# Almost no relationship or very small negative correlation between these 
# categorical variables.


### PART 2: EXAMINE CATEGORICAL DATA
## Use table and bar plot to examine the relationship of Attrition_Flag and 
## other categorical variables 
detach(churn.tf)
attach(churn.df)
st1 <- table(Gender,Attrition_Flag)
st2 <- table(Dependent_count,Attrition_Flag)
st3 <- table(Education_Level,Attrition_Flag)
st4 <- table(Marital_Status,Attrition_Flag)
st5 <- table(Income_Category,Attrition_Flag)
st6 <- table(Card_Category,Attrition_Flag)
st7 <- table(Total_Relationship_Count,Attrition_Flag)
st8 <- table(Months_Inactive_12_mon,Attrition_Flag)
st9 <- table(Contacts_Count_12_mon,Attrition_Flag)

## Change to proportion table
pt1 <- prop.table(st1, margin=1)
pt2 <- prop.table(st2, margin=1)
pt3 <- prop.table(st3, margin=1)
pt4 <- prop.table(st4, margin=1)
pt5 <- prop.table(st5, margin=1)
pt6 <- prop.table(st6, margin=1)
pt7 <- prop.table(st7, margin=1)
pt8 <- prop.table(st8, margin=1)
pt9 <- prop.table(st9, margin=1)
# Each type of customers in different categories has different size, so the number
# of attritted customers in each type can be hugely different.
# By changing to proportion, we can compare the churn rate (the proportion of 
# attritted customers) in different demographics more easily. 

## Visualize the Churn Rate of different variables
par(mfcol=c(3,3))
barplot(pt1[,1], col="steelblue", xlab="Gender", ylab="Churn Rate")

barplot(pt2[,1], col="deeppink3", xlab = "Number of Dependents", ylab="Churn Rate")

barplot(pt3[,1], col="firebrick2", xlab="Education Level", ylab="Churn Rate")

barplot(pt4[,1], col="darkorchid", main = "Churn Rate", 
        xlab="Marital Status", ylab="Churn Rate")

barplot(pt5[,1], col="darksalmon", xlab="Income Category", ylab="Churn Rate")

barplot(pt6[,1], col="aquamarine4", xlab="Card Category", ylab="Churn Rate")

barplot(pt7[,1], col="goldenrod1", xlab="Total Relationship Count", ylab="Churn Rate")

barplot(pt8[,1], col="brown", xlab="Number of Months Inactive", ylab="Churn Rate")

barplot(pt9[,1], col="black", xlab="Number of Contacts", ylab="Churn Rate")

# Based on the bar plots, I can pick the type of customer that are more likely to 
# churn (those having higher churn rate) in different categories. Looking at 
# these plots, we can see that female (Gender), doctorate (Education Level), 
# platinum cardholder (Card Category), those having 1 or 2 relationships (Total
# Relationship Count), those being inactive for 4 months (except 0 - Number of 
# Months Inactive), and those having 6 contacts (Number of Contacts) are more 
# likely to churn.

## 3. Conduct chi-square test
chisq.test(st1)
chisq.test(st2)
chisq.test(st3)
chisq.test(st4)
chisq.test(st5)
chisq.test(st6)
chisq.test(st7)
chisq.test(st8)
chisq.test(st9)
# Null Hypothesis: The churn rate (proportion of attritted customer) is the same 
# among all the segments in different variables. 
# If p-value<0.05 => we can reject the null hypothesis with 95% confidence. 
# Based on the results, there is a difference in the churn rate among the 
# different types of Genders, income categories, Total Relationship Count, Number of 
# Months Inactive, and Number of Contacts.

## Conduct binomial test on st1, st5, st7, st8, st9 (based on chi-square test results)
# Female vs Gender
binom.test(st1[1,1],sum(st1[1,]), p=sum(st1[,1])/10127)
# Null Hypothesis: Female has same churn rate (proportion of attritted customer)
# as the whole sample. 
# p < 0.05 => we can reject the null hypothesis with 95% confidence.
# The churn rate for female is (0.163,0.184), which is higher than the sample
# proportion of 0.161. This means that female has a higher churn rate than the
# whole sample.

# >$120K and <$40K income vs Income Categories
binom.test(st5[1,1],sum(st5[1,]), p=sum(st5[,1])/10127)
# Null Hypothesis: Those with >$120K income has same churn rate (proportion of 
# attritted customer) as the whole sample.
# p > 0.05 => we fail to reject the null hypothesis with 95% confidence.
binom.test(st5[5,1],sum(st5[5,]), p=sum(st5[,1])/10127)
# Null Hypothesis: Those with <$40K income has same churn rate (proportion of 
# attritted customer) as the whole sample.
# p > 0.05 => we fail to reject the null hypothesis with 95% confidence.

# 1 or 2 relationships vs Total Relationship Count
binom.test(st7[1,1],sum(st7[1,]), p=sum(st7[,1])/10127)
binom.test(st7[2,1],sum(st7[2,]), p=sum(st7[,1])/10127)
# Null Hypothesis: Those with 1 or 2 relationships has same churn rate (proportion of 
# attritted customer) as the whole sample.
# p < 0.05 => we can reject the null hypothesis with 95% confidence.
# The churn rate range for 1 or 2 relationships is higher than the sample
# proportion of 0.16. This means that those who have 1 or 2 relationships have 
# a higher churn rate than the whole sample.

# 4 months inactive vs Months Inactive (except 0 because we cannot tell if someone
# who is still active or just stop being active will churn or not)
binom.test(st8[5,1],sum(st8[5,]), p=sum(st8[,1])/10127)
# Null Hypothesis: Those with 4 months inactive has same churn rate (proportion of 
# attritted customer) as the whole sample.
# p < 0.05 => we can reject the null hypothesis with 95% confidence.
# The churn rate range for 4 months inactive is higher than the sample
# proportion of 0.16. This means that those who are 4 months inactive have 
# a higher churn rate than the whole sample.

# 6 contacts vs Contacts Count
binom.test(st9[7,1],sum(st9[7,]), p=sum(st9[,1])/10127)
# Null Hypothesis: Those with 6 contacts has same churn rate (proportion of 
# attritted customer) as the whole sample.
# p < 0.05 => we can reject the null hypothesis with 95% confidence.
# The churn rate range for 6 contacts is higher than the sample
# proportion of 0.16. This means that those who have 6 contacts have 
# a higher churn rate than the whole sample.

## Final Conclusion: Based on the observation, chi-square test, and binomial test,
## we can conclude that gender, Total Relationship Count, Month Inactive, and 
## Contacts count have some segments that can have higher churn rate. Some of 
## these segments are:
## - Gender: Female
## - Total Relationship Count: 1, 2 
## - Month Inactive: 4, 3 (except 0)
## - Contacts Count: 6, 5, 4, 3

### PART 3: EXAMINE CONTINUOUS DATA
## Draw box plots to see any insightful indications
library(ggplot2)
# Customer_Age
plot3 <- ggplot()+geom_boxplot(data=churn.df, aes(x=Customer_Age, 
         fill=Attrition_Flag)) + facet_grid(.~Attrition_Flag)
plot3
# The Customer Age for the 2 groups of Attritted and Existing Customers looks similar.

# Months_on_book
plot4 <- ggplot()+geom_boxplot(data=churn.df, aes(x=Months_on_book, 
       fill=Attrition_Flag)) + facet_grid(.~Attrition_Flag)
plot4
# The number of months on book for the 2 groups of Attritted and Existing 
# Customers looks similar.

# Total_Amt_Chng_Q4_Q1
plot5 <- ggplot()+geom_boxplot(data=churn.tf, aes(x=Total_Amt_Chng_Q4_Q1, 
         fill=Attrition_Flag)) + facet_grid(.~Attrition_Flag)
plot5
# The range and average of Total amount changed from quarter 4 to quarter 1
# of existing customers are higher. 

# Total_Trans_Amt
plot6 <- ggplot()+geom_boxplot(data=churn.tf, aes(x=Total_Trans_Amt, 
         fill=Attrition_Flag)) + facet_grid(.~Attrition_Flag)
plot6
# The range and average of Total transaction amount of existing customers are higher. 

# Total_Trans_Ct
plot7 <- ggplot()+geom_boxplot(data=churn.tf, aes(x=Total_Trans_Ct, 
         fill=Attrition_Flag)) + facet_grid(.~Attrition_Flag)
plot7
# The range and average of Total transaction count of existing customers are higher. 

# Total_Ct_Chng_Q4_Q1
plot8 <- ggplot()+geom_boxplot(data=churn.tf, aes(x=Total_Ct_Chng_Q4_Q1, 
         fill=Attrition_Flag)) + facet_grid(.~Attrition_Flag)
plot8
# The range and average of Total count changed from quarter 4 to quarter 1 of 
# existing customers are higher. 

## Z-Test: between the average of Attritted group and the whole data set average
library(TeachingDemos)
attach(churn.df)
# Customer_Age
z.test(Customer_Age[Attrition_Flag=="Attrited Customer"],mu=mean(Customer_Age),
       stdev = sd(Customer_Age))
# Null Hypothesis: the average customer age of attrited customers is the same to
# the average customer age of the whole data set
# p-value>0.05 => fail to reject => the average customer age of attrited customers 
# is the same to the average customer age of the whole data set

# Months_on_book
z.test(Months_on_book[Attrition_Flag=="Attrited Customer"],mu=mean(Months_on_book),
       stdev = sd(Months_on_book))
# Null Hypothesis: the average months on book of attrited customers is the same to
# that of the whole data set
## p-value>0.05 => fail to reject the null hypothesis => the average months on book of 
## attrited customers is the same to the whole data set

# Total_Amt_Chng_Q4_Q1
z.test(Total_Amt_Chng_Q4_Q1[Attrition_Flag=="Attrited Customer"],mu=mean(Total_Amt_Chng_Q4_Q1),
       stdev = sd(Total_Amt_Chng_Q4_Q1))
# Null Hypothesis: the average Total amount changed from quarter 4 to quarter 1 
# of attrited customers is the same to that of the whole data set
# p-value<0.05 => reject the null hypothesis => the average Total amount changed 
# from quarter 4 to quarter 1 of attrited customers is different from the whole data set

# Total_Trans_Amt
z.test(Total_Trans_Amt[Attrition_Flag=="Attrited Customer"],mu=mean(Total_Trans_Amt),
       stdev = sd(Total_Trans_Amt))
# Null Hypothesis: the average Total transaction amount of attrited customers 
# is the same to that of the whole data set
# p-value<0.05 => reject the null hypothesis => the average Total transaction amount
# of attrtted customers is different from the whole data set

# Total_Trans_Ct
z.test(Total_Trans_Ct[Attrition_Flag=="Attrited Customer"],mu=mean(Total_Trans_Ct),
       stdev = sd(Total_Trans_Ct))
# Null Hypothesis: the average Total transaction count of attritted customers 
# is the same to that of the whole data set
# p-value<0.05 => reject the null hypothesis => the average Total transaction count
# of attrited customers is different from the whole data set

# Total_Ct_Chng_Q4_Q1
z.test(Total_Ct_Chng_Q4_Q1[Attrition_Flag=="Attrited Customer"],mu=mean(Total_Ct_Chng_Q4_Q1),
       stdev = sd(Total_Ct_Chng_Q4_Q1))
# Null Hypothesis: the average Total count changed from quarter 4 to quarter 1
# of attrited customers is the same to that of the whole data set
# p-value<0.05 => reject the null hypothesis => the average Total count changed 
# from quarter 4 to quarter 1 of attrited customers is different from the whole data set

## T-Test: between Attritted group and Existing group
# Customer_Age
t.test(Customer_Age[Attrition_Flag=="Attrited Customer"],Customer_Age[Attrition_Flag==
                                                                         "Existing Customer"])
t.test(Customer_Age~Attrition_Flag)
# Null Hypothesis: the average customer age of attrited customers and existing 
# customers is the same
# p-value > 0.05 => fail to reject the null hypothesis 

# Months_on_book
t.test(Months_on_book[Attrition_Flag=="Attrited Customer"],Months_on_book[Attrition_Flag==
                                                                             "Existing Customer"])
t.test(Months_on_book~Attrition_Flag)
# Null Hypothesis: the average months on book of attrited customers and existing 
# customers is the same
# p-value > 0.05 => fail to reject the null hypothesis 

# Total_Amt_Chng_Q4_Q1
t.test(Total_Amt_Chng_Q4_Q1[Attrition_Flag=="Attrited Customer"],Total_Amt_Chng_Q4_Q1[Attrition_Flag==
                                              "Existing Customer"])
t.test(Total_Amt_Chng_Q4_Q1~Attrition_Flag)
# Null Hypothesis: the average Total amount changed from quarter 4 to quarter 1 
# of attrited customers and existing customers is the same
# p-value < 0.05 => reject the null hypothesis => the average Total amount changed 
# from quarter 4 to quarter 1 of attritted customers and existing customers is different
# (attrited < existing)

#Total_Trans_Amt
t.test(Total_Trans_Amt[Attrition_Flag=="Attrited Customer"],Total_Trans_Amt[Attrition_Flag==
                                                                                 "Existing Customer"])
t.test(Total_Trans_Amt~Attrition_Flag)
# Null Hypothesis: the average transaction amount of attritted customers and existing 
# customers is the same
# p-value < 0.05 => reject the null hypothesis => the average transaction amount of 
# attrited customers and existing customers is different
# (attrited < existing)


# Total_Trans_Ct
t.test(Total_Trans_Ct[Attrition_Flag=="Attrited Customer"],Total_Trans_Ct[Attrition_Flag==
                                                                                 "Existing Customer"])
t.test(Total_Trans_Ct~Attrition_Flag)
# Null Hypothesis: the average transaction count of attritted customers and existing 
# customers is the same
# p-value < 0.05 => reject the null hypothesis => the average transaction count of 
# attrited customers and existing customers is different
# (attrited < existing)


# Total_Ct_Chng_Q4_Q1
t.test(Total_Ct_Chng_Q4_Q1[Attrition_Flag=="Attrited Customer"],Total_Ct_Chng_Q4_Q1[Attrition_Flag==
                                                                                           "Existing Customer"])
t.test(Total_Ct_Chng_Q4_Q1~Attrition_Flag)
# Null Hypothesis: the average Total count changed from quarter 4 to quarter 1
# of attrited customers and existing customers is the same
# p-value < 0.05 => reject the null hypothesis => the average Total count changed 
# from quarter 4 to quarter 1 of attritted customers and existing customers is different
# (attrited < existing)


## ANOVA Test: multiple groups with Attrition Flag
# We will test the rest of variables with a ANOVA test (I don't use two-way test here because 
# the main variable I want to test is Attrition Flag, the only discrete data being
# used in the test this time)

# Credit Limit
a1 <- aov(Credit_Limit~Attrition_Flag)
anova(a1)
# p-value<0.05 => reject the null hypothesis => the credit limit of the two attrition
# flag groups are different
TukeyHSD(a1)
# The credit limit of existing customer is higher.

#Avg_Open_To_Buy
a2 <- aov(Avg_Open_To_Buy~Attrition_Flag)
anova(a2)
# p-value > 0.05 => fail to reject the null hypothesis

# Avg_Utilization_Ratio
a3 <- aov(Avg_Utilization_Ratio~Attrition_Flag)
anova(a3)
# p-value<0.05 => reject the null hypothesis => the Average utilization ratio of 
# the two attrition flag groups are different
TukeyHSD(a3)
# The average utilization ratio of existing customer is higher.

# As the data above is not really normal distributed, I will use visualization to 
# check them again (use the initial data before Box-Cox transformation)
# Credit Limit
plot9 <- ggplot()+geom_boxplot(data=churn.df, aes(x=Credit_Limit, 
                                                        fill=Attrition_Flag)) + facet_grid(.~Attrition_Flag)
plot9
#Avg_Open_To_Buy
plot10 <- ggplot()+geom_boxplot(data=churn.df, aes(x=Avg_Open_To_Buy, 
                                                        fill=Attrition_Flag)) + facet_grid(.~Attrition_Flag)
plot10
# Avg_Utilization_Ratio
plot11 <- ggplot()+geom_boxplot(data=churn.df, aes(x=Avg_Utilization_Ratio, 
                                                        fill=Attrition_Flag)) + facet_grid(.~Attrition_Flag)
plot11
# The results of the test and the visualization match together.


### FINAL CONCLUSION
# - The Avg_Open_To-Buy, Customer_Age, and Months_on_book are the three variables 
# that do not affect the churn rate

# - Other variables tested this week (Credit Limit, Total Transaction Count, Total
# Transaction amount and their change from Q1 to Q4, Average Utilization Ratio)
# do affect the churn rate. With these variables being lower, the customers are
# more likely to churn.


### PART 4: EXAMINE LINEAR RELATIONSHIPS
## Find linear relationships based on visualization
library(ggplot2)
# Based on the scatterplot matrix, I will graph some of the outstanding correlation 

## Total_Trans_Ct and Total_Trans_Amt
plot12 <- ggplot(data=churn.df, aes(x=Total_Trans_Ct, y=Total_Trans_Amt))+
  geom_point(colour="steelblue")+labs(x="Total Transaction Count",y="Total Transaction Amount",
                                      title = "Customer Segmentation based on Transaction Count and Transaction Amount") 
plot12
# There is a positive correlation between Total_Trans_Amt and Total_Trans_Ct.
# We can classify the customers into three segments.
plot12.2 <- ggplot(data=churn.df, aes(x=Total_Trans_Ct, y=Total_Trans_Amt))+
  geom_point(colour="steelblue")+labs(x="Total Transaction Count",y="Total Transaction Amount",
                                      title = "Customer Segmentation based on Transaction Count and Transaction Amount") + 
  facet_grid(.~Attrition_Flag)
plot12.2
# After divide by churn rate, we can see that the highest segment only exist in 
# the group of existing customer (high transaction count and high transaction amount).

## Total_Trans_Amt and Total_Amt_Chng_Q4_Q1
plot13 <-ggplot(data=churn.df, aes(x=Total_Trans_Amt, y=Total_Amt_Chng_Q4_Q1))+
  geom_point(colour="red4")+labs(x="Total Transaction Amount",y="Total amount changed Q4-Q1",
                                 title = "Customer Segmentation based on Transaction Amount and Total Amount Changed") 
plot13
# There is a positive correlation between two variables.
# We can classify the customers into three segments.
plot13.2 <-ggplot(data=churn.df, aes(x=Total_Trans_Amt, y=Total_Amt_Chng_Q4_Q1))+
  geom_point(colour="red4")+labs(x="Total Transaction Amount",y="Total amount changed Q4-Q1",
                                 title = "Customer Segmentation based on Transaction Amount and Total Amount Changed") +
  facet_grid(.~Attrition_Flag)
plot13.2
# After divide by churn rate, we can see that two segments only exist in 
# the group of existing customer (high transaction amount-low amount change and
# low transaction amount-high amount change). Those who churn out have both the variable low.

## Average Open To Buy vs Credit Limit
plot14 <-ggplot(data=churn.df, aes(x=Avg_Open_To_Buy, y=Credit_Limit))+
  geom_point(colour="coral3")+labs(x="Average open to buy ratio of customer",y="Credit limit of customer",
                                   title = "Scatterplot of Credit limit and Average open to buy ratioof customer") 
plot14
# This is almost a perfect positive correlation between two variables, so we cannot 
# get any segment classification here. 

## Linear Regression Model with single predictor
# I pick the pairs with high correlation to test as the R-squared is higher in those
# pairs.
# Also, the correlations of Naive Bayes 1 and 2 to other variable is exactly the 
# same/opposite to Attrition Flag, so I will not test these 2.

m1 <- lm(Total_Trans_Amt~Total_Trans_Ct)
m1
summary(m1)

m2 <- lm(Credit_Limit~Avg_Open_To_Buy)
m2
summary(m2)

m3 <- lm(Total_Amt_Chng_Q4_Q1~Total_Trans_Amt)
m3
summary(m3)

m4 <- lm(churn.tf$Attrition_Flag~churn.tf$Total_Trans_Ct)
m4
summary(m4)

m5 <- lm(churn.tf$Attrition_Flag~churn.tf$Total_Ct_Chng_Q4_Q1)
m5
summary(m5)

# The null hypothesis for the intercept and the coefficient is that the values 
# are equal to 0.  The two p-values of each model indicate that those values 
# are different from 0 in a statistically significant way.

# The p-value of all these model is close to 0, so the models are statistically significant.

# R-squared of the 1st model is 0.6516, which means that the model can 
# explain 65.16% every rows. Similarly, Avg Open to buy explain 99.2% Credit Limit.
# R-squared of the 3rd model is 0.001, so this model can explain 0.16% situations. 
# Similarly, the 4th and 5th model can explain 13.99% and 11.88%. 
# As 3rd model is too low in accuracy, I will drop testing it further. 

# The requirements for good residuals are that they are close to 0, 1Q and 3Q 
# are symmetric, min and max are not extreme. The 4 models above fit these requirements.

## Check standard deviation of residuals
sd(m1$residuals)
sd(m2$residuals)
sd(m4$residuals)
sd(m5$residuals)
# There is no problem with these.

## Confidence intervals
confint(m1)
confint(m2)
confint(m4)
confint(m5)

# These confidence intervals for both intercept and coefficients do not include 0.
# The confidence interval for the slope of the models is positive.


## Check regression line
# I will only test the 1st and 2nd model as Naive Bayes or Attrition Flag is close
# to discrete values, so this regression line will make no sense to them.
# Total_Trans_Ct and Total_Trans_Amt
plot12.3 <- ggplot(data=churn.df, aes(x=Total_Trans_Ct, y=Total_Trans_Amt))+
  geom_point(colour="steelblue")+labs(x="Total Transaction Count",y="Total Transaction Amount",
                                      title = "Customer Segmentation based on Transaction Count and Transaction Amount") +
  geom_abline(intercept=m1$coefficients[1], slope=m1$coefficients[2])
plot12.3
# The line looks like a good fit for the 1st and 2nd segment, and the points seem to be
# evenly distributed around it. The 3rd segment seems out of line.

# Average Open To Buy vs Credit Limit
plot14.3 <-ggplot(data=churn.df, aes(x=Avg_Open_To_Buy, y=Credit_Limit))+
  geom_point(colour="coral3")+labs(x="Average open to buy ratio of customer",y="Credit limit of customer",
                                   title = "Scatterplot of Credit limit and Average open to buy ratioof customer") +
  geom_abline(intercept=m2$coefficients[1], slope=m2$coefficients[2])
plot14.3
# The line looks like a good fit, and the points seem to be evenly distributed around it.

## Check the plots
plot(m1)
# Look at the four plots and their trend lines, we can see that this model is not exact. 
plot(m2)
# Look at the four plots and their trend lines, this model is not too good, but
# it seems to be the best model we have.
plot(m4)
plot(m5)
# Look at the four plots and their trend lines, we can see that this model is not exact. 

### CONCLUSION: Look at these test, I can say that the only model that works is
### between Credit Limit and Avg Open to Buy Ratio

## Multiple Linear Regression Model
detach(churn.df)
attach(churn.tf)
## Build a model with all continuous variables (except for Naive Bayes)
m <- lm(Attrition_Flag~Credit_Limit+Avg_Open_To_Buy+Total_Amt_Chng_Q4_Q1+
          Total_Trans_Amt+Total_Trans_Ct+Total_Ct_Chng_Q4_Q1+Avg_Utilization_Ratio+
          Total_Revolving_Bal+Customer_Age+Gender+Dependent_count+Education_Level+
          Marital_Status+Income_Category+Card_Category+Months_on_book+Total_Relationship_Count+
          Months_Inactive_12_mon+Contacts_Count_12_mon)
m
summary(m)
# The null hypothesis for the intercept and the coefficient for complaints
# is that the value is equal to 0.  The p-values indicate that these values 
# are different from 0 in a statistically significant way, except for Customer
# Age, Educational Level, and Months on book.

# p-value of the whole model is close to 0, so it is statistically significant.

# The requirements for good residuals are that they are close to 0, 1Q and 3Q 
# are symmetric, min and max are not extreme. The model above fits these requirements.
# R-squared is 0.3478, so the model can explain 41.16%% of the churn rate.
sd(m$residuals)
# There is no problem with the standard deviation of the residuals.

# Although this R-squared is lower than that of Naive Bayes, this combined model
# is much better than the models with single predictor (like m3 and m4)

## Check the multiple regression model with fitted value by scatterplot and ANOVA
plot15 <- ggplot() + 
  geom_point(data=churn.tf, aes(x=Attrition_Flag,y=m$fitted), colour="purple4") +
  geom_point(data=churn.tf, aes(x=Attrition_Flag,y=m4$fitted), colour="grey") +
  geom_point(data=churn.tf, aes(x=Attrition_Flag,y=m5$fitted), colour="black") +
  geom_abline(intercept=0, slope=1)
plot15
# The color purple and black seems to cluster around the x=y line similarly, meaning that 
# the predicted values of m3 is close to the the multiple regression model.

## ANOVA test
anova(m,m4)
anova(m,m5)
# p-value < 0.05, so there is a big difference between m and m4, the scatter plot above 
# happens because Attrition Flag is discrete, making the points overlapping each
# other. 

### FINAL CONCLUSION: As the R-squared of the multiple predictors model is better, 
### we can conclude that this model is better than any single model (because m4
### should be one of the best single model due to its high correlation to Attrition Flag).

## Check the model with predict() (Skip the coefficients because they are the same
# and there are too many variables here)
predict(m,churn.tf[21:25,])
churn.tf[21:25,2]
# We can see that the model is exact with row 21, 22, and 23 (if we round up and down).
# However, it is not exact with row 24 and 25. This accuracy rate makes sense 
# with the R-square of the model. 


### PART 5: EXAMINE LINEAR RELATIONSHIPS (2)
## The model with continuous unscalling data
M1 <- lm(Attrition_Flag~Credit_Limit+Avg_Open_To_Buy+Total_Amt_Chng_Q4_Q1+
           Total_Trans_Amt+Total_Trans_Ct+Total_Ct_Chng_Q4_Q1+Avg_Utilization_Ratio+
           Customer_Age+Months_on_book+Total_Revolving_Bal,data = churn.tf)
M1
summary(M1)

## Save the data frame and standardize the continuous data columns 
churn.std <-churn.tf
# I only standardize the Months on book and the transformed data as I only use 
# them for the model
churn.std[c(3,10,14:21)]<-scale(churn.std[c(3,10,14:21)])

## Check the means 
summary(churn.std[c(3,10,14:21)]) #all meet the requirement
## Build a multiple linear regression model using the standardized data frame 
M2 <- lm(Attrition_Flag~Credit_Limit+Avg_Open_To_Buy+Total_Amt_Chng_Q4_Q1+
           Total_Trans_Amt+Total_Trans_Ct+Total_Ct_Chng_Q4_Q1+Avg_Utilization_Ratio+
           Customer_Age+Months_on_book+Total_Revolving_Bal,data = churn.std)
M2
summary(M2)
# The quality of the second model is the same, based on the r-squared values. 
# This makes sense as all the values are already transformed before scalling.
# Also, notice that the p-values for the columns have not changed (all are 
# statistically significant with 95% apart from Customer_Age
# and Months on book).

## We can check with the model for untransformed variable
M3 <- lm(churn.tf$Attrition_Flag~Credit_Limit+Avg_Open_To_Buy+Total_Amt_Chng_Q4_Q1+
           Total_Trans_Amt+Total_Trans_Ct+Total_Ct_Chng_Q4_Q1+Avg_Utilization_Ratio+
           Customer_Age+Months_on_book+Total_Revolving_Bal,data = churn.df)
M3
summary(M3)
# We can see that the quality of this model is worse based on R-squared (30.48%) and the 
# the fact the Avg_Utilization_Ratio is not statistically significant anymore.

### CONCLUSION: The lowest model is the one without Box-Cox transformation, while
### the models with transformed and scalled data have the same quality.

## Build a multiple linear regression model for based on all the remaining columns
M4 <- lm(Attrition_Flag~Credit_Limit+Avg_Open_To_Buy+Total_Amt_Chng_Q4_Q1+
           Total_Trans_Amt+Total_Trans_Ct+Total_Ct_Chng_Q4_Q1+Avg_Utilization_Ratio+
           Total_Revolving_Bal+Customer_Age+Gender+Dependent_count+Education_Level+
           Marital_Status+Income_Category+Card_Category+Months_on_book+Total_Relationship_Count+
           Months_Inactive_12_mon+Contacts_Count_12_mon,data=churn.std)
M4
summary(M4)
# This model has better quality compared with others in this part with 41.16% R-squared
# It has the same R-squared with the model in part 4.
# The p-values indicate that the variables  are different from 0 in a statistically 
# significant way, except for Customer Age, Educational Level, and Months on book.

## Build a new model to explain churn rate by the significant variables
M5 <- lm(Attrition_Flag~Credit_Limit+Avg_Open_To_Buy+Total_Amt_Chng_Q4_Q1+
           Total_Trans_Amt+Total_Trans_Ct+Total_Ct_Chng_Q4_Q1+Avg_Utilization_Ratio+
           Total_Revolving_Bal+Gender+Dependent_count+
           Marital_Status+Income_Category+Card_Category+Total_Relationship_Count+
           Months_Inactive_12_mon+Contacts_Count_12_mon,data=churn.std)
M5
summary(M5)
# The model has all variable being statistically significant. However, the R-squared is 
# slightly lower (41.14%) but not too much.

## Build another model that incorporates all of the pairwise interactions.  
## Let's look back at the scatterplot matrix
library(car)
scatterplotMatrix(churn.tf[c(2,14:21)],diagonal="histogram")
## It looks as though many variables has an interaction with each other.
# Build another model that incorporates all of the pairwise interactions. 
M6 <- lm(Attrition_Flag~(Credit_Limit+Avg_Open_To_Buy+Total_Amt_Chng_Q4_Q1+
           Total_Trans_Amt+Total_Trans_Ct+Total_Ct_Chng_Q4_Q1+Avg_Utilization_Ratio+
           Total_Revolving_Bal+Gender+Dependent_count+
           Marital_Status+Income_Category+Card_Category+Total_Relationship_Count+
           Months_Inactive_12_mon+Contacts_Count_12_mon)^2,data=churn.std)
M6
summary(M6)
# In this step, we see that only intercept, Credit Limit, Avg Open to Buy, Total
# Amount Changed, Total Transaction Amount, Utilization Ratio, Total Revolving Balance, 
# Total_Relationship_Count,and Contact Count are statistically significant now. 

# As there are quite a lot of interactions as mentioned above, some significant pairwaise 
# interactions are between Credit Limit & Avg Open to Buy, Gender, Total Transaction
# Amount/Count, ect or Total Contact Count & Relationship Count, etc.

# Overall, this model is much better with 52.93% R-squared.

## We can try building another model that incorporates all of higher pairwise interactions. 
M7 <- lm(Attrition_Flag~(Credit_Limit+Avg_Open_To_Buy+Total_Amt_Chng_Q4_Q1+
                           Total_Trans_Amt+Total_Trans_Ct+Total_Ct_Chng_Q4_Q1+Avg_Utilization_Ratio+
                           Total_Revolving_Bal+Gender+Dependent_count+
                           Marital_Status+Income_Category+Card_Category+Total_Relationship_Count+
                           Months_Inactive_12_mon+Contacts_Count_12_mon)^3,data=churn.std)
M7
summary(M7)
# With this model, even fewer variables are statistically significant, including
# Total Transaction Amount/Count, Total Count Changed, and Contact Count.
# We all see the same number of significant interactions here. 

# Some of the significant interactions are Credit Limit-Avg Utilization Ratio-Total
# Revolving Balance, Credit Limit-Total Transaction Count-Total Revolving Balance, etc.

## Use ANOVA to test 
anova(M7,m)
anova(M7,M6)
anova(M6,M4)

## Predict the churn rate
predict(M7,churn.std[21:30,])
churn.tf[21:30,2]
# Accuracy rate = 80% with this trial 

#### CONCLUSION: This model has the highest quality with 61.02% R-squared.

#### Final Conclusion: The 2 last models, which consider interaction, have the 
#### highest quality. The last linear regression model can predict with quite a 
#### good accuracy of 61%.



## PART 6: EXAMINE USING LOGISTICS REGRESSION
# Read the data again to test the model more accurately
churn.df <- prepData("BankChurners.csv")

#Change Attrition Flag into factor
churn.df$Attrition_Yes <- ifelse(churn.df$Attrition_Flag == "Attrited Customer", 1, 0)
# Convert (train) Attrition_Yes to factor with correct levels and labels
train$Attrition_Yes <- factor(train$Attrition_Yes, levels = c(0, 1), labels = c("No", "Yes"))


# Select only necessary variables in the dataset
churn.df <- churn.df[,3:24]

# Split the dataset into train and test sets
set.seed(42)
train_index <- createDataPartition(churn.df$Attrition_Yes, p = 0.7, list = FALSE)
train <- churn.df[train_index, ]
test <- churn.df[-train_index, ]

# Convert (test) Attrition_Yes to factor with correct levels and labels
test$Attrition_Yes <- factor(test$Attrition_Yes,levels = c(0, 1), labels = c("No", "Yes"))

# Set up cross-validation
fitControl <- trainControl(method = "cv", number = 5)


# Train the model
model <- train(Attrition_Yes ~ ., data = train, method = "glmnet", trControl = fitControl, family = "binomial")

# Predict on the test set
y_pred <- predict(model, newdata = test)


# Convert y_pred to numeric
y_pred_numeric <- as.numeric(as.character(y_pred))

# Convert y_pred_numeric to binary factor
y_pred_class <- ifelse(y_pred > 0.5, "Yes", "No")
y_pred_class <- factor(ifelse(y_pred_class == "No", "No", "Yes"))

# Calculate confusion matrix and accuracy
confusion_matrix <- confusionMatrix(y_pred_class, test$Attrition_Yes)

# Print confusion matrix and accuracy
print(confusion_matrix$table)
accuracy <- confusion_matrix$overall[1]
print(paste("Accuracy:", accuracy))


# Print classification report
classification_rep <- confusionMatrix(y_pred_class, test$Attrition_Yes)$byClass
print("Classification report:")
print(classification_rep)

# Calculate ROC curve and AUC-ROC
roc <- roc(test$Attrition_Yes, y_pred)
auc <- auc(roc)

# Plot ROC curve
plot(roc, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")

# Add AUC-ROC to plot
legend("bottomright", legend = paste0("AUC = ", round(auc, 2)), bty = "n")







