# machine_learning_feb2026
Project for Machine Learning II course

Dataset: Credit Card Dataset for Clustering
Information: Summary of 6 months activity per account

Problem: Unsupervized Learning (k-means, model2)

Model Ideas: Clustering (k-Means and Hierarchical), DBS, Mean-Shift, SOM, 

Variables:
x   Column                            Orig_Dtype New_Dtype Variable Type                 Meaning
0   CUST_ID                           object     uint16    categorical(nominal); unique  
1   BALANCE                           float64    float64   quantitative(continuous)      Balance amount left in their account
2   BALANCE_FREQUENCY                 float64    float32   quantitative(continuous)      How frequently the Balance is updated(0-1)
3   PURCHASES                         float64    float32   quantitative(continuous)      Amount of purchases made
4   ONEOFF_PURCHASES                  float64    float32   quantitative(continuous)      Maximum purchase amount done in one-go
5   INSTALLMENTS_PURCHASES            float64    float32   quantitative(continuous)      Amount of purchase done in installment
6   CASH_ADVANCE                      float64    float64   quantitative(continuous)      Cash in advance given by the user
7   PURCHASES_FREQUENCY               float64    float32   quantitative(continuous)      How frequently the Purchases are being made(0-1)
8   ONEOFF_PURCHASES_FREQUENCY        float64    float32   quantitative(continuous)      How frequently Purchases are happening in one-go(0-1)
9   PURCHASES_INSTALLMENTS_FREQUENCY  float64    float32   quantitative(continuous)      How frequently purchases in installments(0-1)
10  CASH_ADVANCE_FREQUENCY            float64    float32   quantitative(continuous)      How frequently the cash in advance being paid
11  CASH_ADVANCE_TRX                  int64      uint8     quantitative(continuous)      Number of Transactions made with "Cash in Advanced"
12  PURCHASES_TRX                     int64      uint16    quantitative(continuous)      Number of purchase transactions made
13  CREDIT_LIMIT                      float64    uint16    quantitative(continuous)      Limit of Credit Card for user
14  PAYMENTS                          float64    float64   quantitative(continuous)      Amount of Payment done by user
15  MINIMUM_PAYMENTS                  float64    float64   quantitative(continuous)      Minimum amount of payments made by user
16  PRC_FULL_PAYMENT                  float64    float32   quantitative(continuous)      Percent of full payment paid by user
17  TENURE                            int64      uint8     quantitative(discrete)        Tenure of credit card service for user