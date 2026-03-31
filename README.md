# machine_learning_feb2026
Project for Machine Learning II course

Dataset: Credit Card Dataset for Clustering

Problem: Unsupervized Learning (k-means, model2)

Model Ideas: Clustering (k-Means and Hierarchical), DBS, Mean-Shift, SOM, 

Variables:
x   Column                            Orig_Dtype New_Dtype Variable Type
0   CUST_ID                           object     uint16    categorical(nominal) - primary identifier(unique)
1   BALANCE                           float64    float64   quantitative(continuous)
2   BALANCE_FREQUENCY                 float64    float32   quantitative(continuous)
3   PURCHASES                         float64    float32   quantitative(continuous)
4   ONEOFF_PURCHASES                  float64    float32   quantitative(continuous)
5   INSTALLMENTS_PURCHASES            float64    float32   quantitative(continuous)
6   CASH_ADVANCE                      float64    float64   quantitative(continuous)
7   PURCHASES_FREQUENCY               float64    float32   quantitative(continuous)
8   ONEOFF_PURCHASES_FREQUENCY        float64    float32   quantitative(continuous)
9   PURCHASES_INSTALLMENTS_FREQUENCY  float64    float32   quantitative(continuous)
10  CASH_ADVANCE_FREQUENCY            float64    float32   quantitative(continuous)
11  CASH_ADVANCE_TRX                  int64      uint8     quantitative(continuous)
12  PURCHASES_TRX                     int64      uint16    quantitative(continuous)
13  CREDIT_LIMIT                      float64    uint16    quantitative(continuous)
14  PAYMENTS                          float64    float64   quantitative(continuous)
15  MINIMUM_PAYMENTS                  float64    float64   quantitative(continuous)
16  PRC_FULL_PAYMENT                  float64    float32   quantitative(continuous)
17  TENURE                            int64      uint8     quantitative(discrete)