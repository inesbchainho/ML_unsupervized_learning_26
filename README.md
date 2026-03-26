# machine_learning_feb2026
Project for Machine Learning II course

Dataset: Telco Customer Churn

Problem: Unsupervized Learning (model1, model2)

Model Ideas: Clustering (k-Means and Hierarchical), DBS, Mean-Shift, SOM, 

Variables:
 x   Column            Orig_Dtype  New_Dtype  Variable Type
 0   customerID        object      string     categorical(nominal) - primary identifier(unique)
 1   gender            object      string     categorical(nominal) - binary
 2   SeniorCitizen     int64       integer    quantitative(discrete)
 3   Partner           object      string     categorical(nominal) - binary
 4   Dependents        object      string     categorical(nominal) - binary
 5   tenure            int64       integer    quantitative(discrete)
 6   PhoneService      object      string     categorical(nominal) - binary
 7   MultipleLines     object      string     categorical(nominal) - 3 values
 8   InternetService   object      string     categorical(nominal) - 3 values
 9   OnlineSecurity    object      string     categorical(nominal) - 3 values
 10  OnlineBackup      object      string     categorical(nominal) - 3 values
 11  DeviceProtection  object      string     categorical(nominal) - 3 values
 12  TechSupport       object      string     categorical(nominal) - 3 values
 13  StreamingTV       object      string     categorical(nominal) - 3 values
 14  StreamingMovies   object      string     categorical(nominal) - 3 values
 15  Contract          object      string     categorical(nominal) - 3 values
 16  PaperlessBilling  object      string     categorical(nominal) - binary
 17  PaymentMethod     object      string     categorical(nominal) - 4 values
 18  MonthlyCharges    float64     float      quantitative(continuous)
 19  TotalCharges      object      float      quantitative(continuous)
 20  Churn             object      string     categorical(nominal) - binary
