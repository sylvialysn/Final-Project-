# **Final Project by Beta Team: Bank Marketing Campaigns Data**

Team Members:
- Nuh Neguita Gurusinga
- Sylvia Sinnelius

Table of Contents
I. Business Understanding <br>

II. Data Understanding

III. Exploratory Data Analysis
i. Client Behaviour <br>
ii. Campaign Dynamics <br>
iii. Conclusion <br>
iv. Recommendation <br>

IV. Machine Learning

___


# **I. Business Understanding**

## **i. Background**

A bank is a financial institution that accepts deposits from the public, creating demand deposits while facilitating loans. Bank deposits, serving as a savings product, offer customers the opportunity to hold money for a specified time, earning interest. With flexible terms, deposits cater to diverse financial goals. Surprisingly, data analysis reveals that **only around 11% of bank customers actively utilize deposit services, challenging assumptions about their prevalence.**

Machine learning predictions will leverage **data obtained from research on bank loan targeting through telemarketing phone calls**. This dataset captures interactions focused on selling long-term deposits, both through outbound calls and inbound inquiries. The binary outcome of successful or unsuccessful contact will be the primary focus, **providing insights into factors influencing deposit behaviors.**

This predictive approach enhances our understanding of customer behavior, facilitating informed decision-making and the adaptation of services to meet genuine needs which holds particular significance for the **Head of Marketing**. The insights derived enable the **identification of the right customers, minimizing marketing costs while optimizing effectiveness**.


## **ii. Problem Statement**

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


### **v. Metric Evaluation**

![picture](https://drive.google.com/uc?id=1Ytj6F21nQzkxfpFajRtg8DcQBiprMDen)

<br>Positive (1) : 'Yes', Customer who deposits
<br>Negative (0) : 'No', Customer who does not deposit

<br>False Positive : Predicted as 'Yes' but in reality did not deposit
- Consequences : Wasting marketing budge

False Negative : Predicted as 'No' but in reality did deposit
- Consequences : Losing a potential subscriber


As the False Positive metric is more damaging to the company but False negative is quite impactful as well, therefore, F 0.5 Score will be the analytical approach of choice to give the weight of importance twice for the False Positive metric without neglecting the False Negative prediction.*

# **II. Data Understanding**

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


The highest percentage of 'unknown' values is owned by *'default'* variable, almost 21 %.
<br> Other variable with 'unknown' values:
1. job (0.8%)
2. marital (0.19%)
3. education (4.2%)
4. housing and loan (both 2.4 %)
