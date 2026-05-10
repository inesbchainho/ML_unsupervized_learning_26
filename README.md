# machine_learning_feb2026
Project for Machine Learning II course

Dataset: Credit Card Dataset for Clustering
Information: Summary of 6 months activity per account

Problem: Unsupervized Learning (k-means, model2)

Model Ideas: Clustering (k-Means and Hierarchical), DBS, Mean-Shift, SOM, 

Variables:
x   Column                             Dtype     Variable Type              Meaning
0   CUST_ID                            uint16    categorical(nominal)       Users' unique identifier.
1   BALANCE                            float64   quantitative(continuous)   Balance amount left in their account.
2   BALANCE_FREQUENCY                  float32   quantitative(continuous)   How frequently the balance is updated(0-1).
3   PURCHASES                          float64   quantitative(continuous)   Amount of purchases made from account. REMOVED
4   ONEOFF_PURCHASES                   float64   quantitative(continuous)   Maximum purchase amount done in one-go.
5   INSTALLMENTS_PURCHASES             float64   quantitative(continuous)   Amount of purchases done in installments. REMOVED
6   CASH_ADVANCE                       float64   quantitative(continuous)   Cash in advance given by the user. REMOVED
7   PURCHASES_FREQUENCY                float64   quantitative(continuous)   How frequently the purchases are being made(0-1). REMOVED
8   ONEOFF_PURCHASES_FREQUENCY         float64   quantitative(continuous)   How frequently purchases are happening in one-go(0-1). REMOVED
9   PURCHASES_INSTALLMENTS_FREQUENCY   float64   quantitative(continuous)   How frequently purchases in installments(0-1). REMOVED
10  CASH_ADVANCE_FREQUENCY             float64   quantitative(continuous)   How frequently the cash in advance being paid. REMOVED
11  CASH_ADVANCE_TRX                   int64     quantitative(continuous)   Number of transactions made with "Cash in Advanced". REMOVED
12  PURCHASES_TRX                      int64     quantitative(continuous)   Number of purchase transactions made. REMOVED
13  CREDIT_LIMIT                       uint16    quantitative(continuous)   Limit of credit card for user.
14  PAYMENTS                           float64   quantitative(continuous)   Amount of payment done by user.
15  MINIMUM_PAYMENTS                   float64   quantitative(continuous)   Minimum amount of payments made by user. REMOVED
16  PRC_FULL_PAYMENT                   float32   quantitative(continuous)   Percent of full payment paid by user.
17  TENURE                             uint8     quantitative(discrete)     Tenure of credit card service for user.
18  AVG_PURCHASE_VALUE                 float64   quantitative(continuous)   Average purchase value(PURCHASES/PURCHASES_TRX).
19  AVG_CASH_ADVANCE_VALUE             float64   quantitative(continuous)   Average purchase with cash value(CASH_ADVANCE/CASH_ADVANCE_TRX).
20  PURCHASE_ENGAGEMENT                float64   quantitative(continuous)   Purchase engagement(LOG(1 + PURCHASES*PURCHASES_FREQUENCY)).
22  INSTALLMENT_PURCHASES_ENGAGEMENT   float64   quantitative(continuous)   Purchases in installments engagement(LOG(1 + INSTS_PURCHS*PURCHS_INSTS_FREQ)).
23  PAYMENT_RATIO                      float64   quantitative(continuous)   Ratio between total and minimum payment(LOG(1 + PAYMENTS/MINIMUM_PAYMENTS)).