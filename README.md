# **About**
This project is the final mandatory assignment given by Purwadhika School of Technology, concluding the data science bootcamp's last term. It involves data analysis and machine learning tasks related to a bank marketing campaign dataset in portugal on the year 2013. Therefore, it may not be a reliable source for recent times. In terms of machine learning, classification is used for the prediction. 
___
# **Final Project by Beta Team: Bank Marketing Campaigns Data**

Team Members:
- Nuh Neguita Gurusinga
- Sylvia Sinnelius

## Table of Contents <br>
**I. Business Understanding**
<br>
**II. Data Understanding**
<br>
**III. Exploratory Data Analysis**<br>
&ensp; i. Client Behaviour <br>
&ensp; ii. Campaign Dynamics <br>
&ensp; iii. Conclusion <br>
&ensp; iv. Recommendation <br>
**IV. Machine Learning** <br>
**V. CLR**<br>
**VI. Business Recommendation**


___


# **Business Understanding**

## **i. Background**

A bank is a financial institution that accepts deposits from the public, creating demand deposits while facilitating loans. Bank deposits, serving as a savings product, offer customers the opportunity to hold money for a specified time, earning interest. With flexible terms, deposits cater to diverse financial goals. Surprisingly, data analysis reveals that **only around 11% of bank customers actively utilize deposit services, challenging assumptions about their prevalence.**

Machine learning predictions will leverage **data obtained from research on bank loan targeting through telemarketing phone calls**. This dataset captures interactions focused on selling long-term deposits, both through outbound calls and inbound inquiries. The binary outcome of successful or unsuccessful contact will be the primary focus, **providing insights into factors influencing deposit behaviors.**

This predictive approach enhances our understanding of customer behavior, facilitating informed decision-making and the adaptation of services to meet genuine needs which holds particular significance for the **Head of Marketing**. The insights derived enable the **identification of the right customers, minimizing marketing costs while optimizing effectiveness**.


## **Problem Statement**

The average new account acquisition cost to a bank or credit union is between
\$350-\$422 per account.

*Source: https://solutions.datatrac.net/roicalculator*
<br>
<br>
Approaching all clients for a campaign is not cost-effective. If there were 1000 clients, all approached by the telemarketing team, the cost of a campaign would be an estimate of \$350,000. Say only 110 clients subscribe to long-term deposits while 890 clients do not. This meant as much as \$311,500 becomes futile. Recognizing the need for a more targeted approach, the solution is a machine learning system. By predicting which clients would be willing to deposit and which would not, the bank optimize marketing cost. This approach allows the bank to spend only \$38,500 for marketing costs as they know precisely the 110 clients. Good efficiency in terms of saving cost becomes tangible. This is achieved by analyzing patterns in various variables from bank client data such as 'type of job'(i.e., blue collar or entrepreneur) or 'age' (i.e., 25-60) to social and economic context attributes such as 'interest rate' (i.e., 4.8 %), using them to predict the client's deposit probability.


### **iii. Goals**
Our goal is to leverage machine learning for predicting **customer behavior in long-term deposits**, aiming for maximum accuracy to minimize False Positives. By prioritizing precision, we seek to **optimize marketing budget allocation**, ensuring efficient campaigns with minimal wasted resources. This approach not only enhances customer engagement by tailoring efforts to interested individuals but also contributes to overall customer satisfaction and trust, fostering lasting relationships.

### **iv. Analytical Approach**
Analyzing data to find patterns to differentiate potential subscriber. Then we will build a classification model to predict the probability for each customer that wants to sub or not.


## **v. Metric Evaluation**

![picture](https://drive.google.com/uc?id=1Ytj6F21nQzkxfpFajRtg8DcQBiprMDen)

<br>Positive (1) : 'Yes', Customer who deposits
<br>Negative (0) : 'No', Customer who does not deposit

<br>False Positive : Predicted as 'Yes' but in reality did not deposit
- Consequences : Wasting marketing budge

False Negative : Predicted as 'No' but in reality did deposit
- Consequences : Losing a potential subscriber


As the False Positive metric is more damaging to the company but False negative is quite impactful as well, therefore, Precision Score will be the analytical approach of choice to give the weight of importance to the False Positive metric.*

# **Data Understanding**

**Bank client data:**

| Feature             | Description                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
|age|Numerical Value of someone's Age|
|job|type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")|
|marital|marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)|
|education|(categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")|
|default| has credit in default? (categorical: "no","yes","unknown")
|housing|has housing loan? (categorical: "no","yes","unknown")
|loan|has personal loan? (categorical: "no","yes","unknown")

**Related with the last contact of the current campaign:**

| Feature             | Description                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
| contact| contact communication type (categorical: "cellular","telephone")
|month|last contact month of year (categorical: "jan", "feb", "mar", …, "nov", "dec")
|day_of_week|last contact day of the week (categorical: "mon","tue","wed","thu","fri")
|duration|last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

**Other attributes:**

| Feature             | Description                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
|campaign|number of contacts performed during this campaign and for this client (numeric, includes the last contact)
|pdays|number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
|previous|number of contacts performed before this campaign and for this client (numeric)
|poutcome|outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")

**social and economic context attributes**

| Feature             | Description                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
|emp.var.rate| employment variation rate, a measure that indicates the employment situation. Negative value meaning more jobs were *lost* compared to the previous quarter of the year, while positive value means more jobs were *gained*. - quarterly indicator (numeric)
|cons.price.idx| consumer price index, a measure of monthly changes in price paid by consumers. Can be used to measure inflation and deflation rate.  - monthly indicator (numeric)
|cons.conf.idx|consumer confidence index, an indicator of consumers' confidence of the country's future economic condition based on interviews - monthly indicator (numeric)
|euribor3m| euribor 3 month rate, the interest rate used by banks in the european zone that is served as a benchmark to be applied for financial products such as loan with a 3-month maturity time - daily indicator (numeric)
|nr.employed|number of employees - quarterly indicator (numeric)

**Output variable (desired target):**

- y - has the client subscribed a term deposit? (binary: "yes","no")



Missing Attribute Values: There are several missing values in some categorical attributes, all coded with the "unknown" label. These missing values can be treated as a possible class label or using deletion or imputation techniques.

References :
<br>https://www.babypips.com/forexpedia/employment-change
<br>https://www.investopedia.com/terms/c/consumerpriceindex.asp
<br>https://tradingeconomics.com/portugal/consumer-confidence#:~:text=December%20of%202023.-,Consumer%20Confidence%20in%20Portugal%20decreased%20to%20%2D28.20%20points%20in%20November,macro%20models%20and%20analysts%20expectations.

**Qualitative Variable :**
- Nominal
    1. job
    2. month
    3. day_of_week
    4. poutcome
    5. marital

- Binary
    1. default
    2. housing
    3. loan
    4. contact
    5. y

- Ordinal
    1. education

**Quantitative Variable :**
- Interval
    1. emp.var.rate
    2. cons.price.idx
    3. cons.conf.idx
    4. euribor3m

- Ratio
    1. age
    2. duration
    3. campaign
    4. pdays
    5. previous
    6. nr.employed

# Business Understanding

From the machine learning model, it has its advantages and disadvantages since the model is not quite perfect yet due to a high imbalance in the dataset. The target class (class 1 or positive) constitutes 11%, while the other (class 0 or negative) is 89%. This can be addressed with more documentation on the positive class in the future to achieve a balance.

Let's deep dive into the results based on our preferred matrix, which is precision.

## Based on our classification report (precision):
**Class 0:**
- 90% are labeled 0 and are, in reality, 0.
- 10% are labeled 0 but in reality are 1.

**Class 1:**
- 54% are labeled 1 and are, in reality, 1.
- 46% are labeled 1 but in reality are 0.

From our exploratory data analysis, we gathered a few results:
1. This campaign's success rate is around 36%.
2. The number of calls needed to persuade a client was 1-3, with a maximum of 7 calls.
3. The average cost for a call is $2.7, based on the source below.

Source: [Link](https://www.cxtoday.com/contact-centre/how-to-calculate-your-cost-per-inbound-outbound-call-and-why/)

## Hypothetical case 
**Case:** We have 100,000 clients with a success rate of around 36% (only 35,740 clients actually deposit).

**a. Without Machine Learning:**
Telemarketing team needs to contact each client 3 times.
- The cost = $2.7 x 3 x 100,000 clients = $810,000
- Wasted cost = $2.7 x 3 x (100,000 - 35,740 clients) = $520,506
- All clients will be reached out.

**b. With Machine Learning:**
Say our model predicted 58,500 customers will deposit (labeled as class 1).
- The cost = $2.7 x 3 x 58,500 clients = $473,850
- Extra = $2.7 x 3 x (100,000 - 58,500 clients) = $336,150

However, since this model is still flawed with 46% incorrectly predicted as deposit (when, in reality, they do not deposit), there will be some wasted cost below.
- Wasted cost =  $2.7 x 3 x (46% x 58,500 clients) = $217,971

For class 0, our model predicted the rest of the 41,500 clients will NOT deposit. Since there is an 10% error or incorrectly predicted as will not deposit (when, in reality, they deposit), there will be some clients unreached.
- Unreached clients = 10% x 41,500 clients = 4,150 clients.

In conclusion, the pros and cons are:

|   |Without machine learning| With machine learning   |
|---|-------------------------|-------------------------|
|Overall Cost| $810,000            | $473,850                |
|Wasted Cost | $520,506            | $217,971                |
|Unreached Clients| None          | 4,150 clients           |

As mentioned shortly before, the major problem with the model is the imbalanced dataset causing greater and better prediction for class 0 data. The model needs more records on clients who deposit. Features that might be considered useful for the prediction include the income of each client. If the model is applied, A/B testing can be used to compare results with the model and without, ensuring that using a machine learning model is factually more efficient and profitable for the company.
