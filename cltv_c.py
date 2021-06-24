import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()

df.head()
"""
  Invoice StockCode                          Description  Quantity  \
0  489434     85048  15CM CHRISTMAS GLASS BALL 20 LIGHTS        12   
1  489434    79323P                   PINK CHERRY LIGHTS        12   
2  489434    79323W                  WHITE CHERRY LIGHTS        12   
3  489434     22041         RECORD FRAME 7" SINGLE SIZE         48   
4  489434     21232       STRAWBERRY CERAMIC TRINKET BOX        24   
          InvoiceDate  Price  Customer ID         Country  
0 2009-12-01 07:45:00  6.950    13085.000  United Kingdom  
1 2009-12-01 07:45:00  6.750    13085.000  United Kingdom  
2 2009-12-01 07:45:00  6.750    13085.000  United Kingdom  
3 2009-12-01 07:45:00  2.100    13085.000  United Kingdom  
4 2009-12-01 07:45:00  1.250    13085.000  United Kingdom  

"""

# Data preparation
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df['Quantity'] > 0)]
df.dropna(inplace=True)
df["TotalPrice"] = df["Quantity"] * df["Price"]

df.head()
"""
  Invoice StockCode                          Description  Quantity  \
0  489434     85048  15CM CHRISTMAS GLASS BALL 20 LIGHTS        12   
1  489434    79323P                   PINK CHERRY LIGHTS        12   
2  489434    79323W                  WHITE CHERRY LIGHTS        12   
3  489434     22041         RECORD FRAME 7" SINGLE SIZE         48   
4  489434     21232       STRAWBERRY CERAMIC TRINKET BOX        24   
          InvoiceDate  Price  Customer ID         Country  TotalPrice  
0 2009-12-01 07:45:00  6.950    13085.000  United Kingdom      83.400  
1 2009-12-01 07:45:00  6.750    13085.000  United Kingdom      81.000  
2 2009-12-01 07:45:00  6.750    13085.000  United Kingdom      81.000  
3 2009-12-01 07:45:00  2.100    13085.000  United Kingdom     100.800  
4 2009-12-01 07:45:00  1.250    13085.000  United Kingdom      30.000  
"""

cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(), # unique invoice number of each customer
                                        'Quantity': lambda x: x.sum(), # total units bought by each customer
                                        'TotalPrice': lambda x: x.sum()}) # total price spent by each customer

cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

cltv_c.head()
"""
             total_transaction  total_unit  total_price
Customer ID                                            
12346.000                   11          70      372.860
12347.000                    2         828     1323.320
12348.000                    1         373      222.160
12349.000                    3         993     2671.140
12351.000                    1         261      300.930
"""



# 2. Average Order Value (average_order_value = total_price / total_transaction)
cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']

cltv_c.head()
"""
             total_transaction  total_unit  total_price  avg_order_value
Customer ID                                                             
12346.000                   11          70      372.860           33.896
12347.000                    2         828     1323.320          661.660
12348.000                    1         373      222.160          222.160
12349.000                    3         993     2671.140          890.380
12351.000                    1         261      300.930          300.930
"""



# 3. Purchase Frequency (total_transaction / total_number_of_customers)
cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]

cltv_c.head()
"""
             total_transaction  total_unit  total_price  avg_order_value  \
Customer ID                                                                
12346.000                   11          70      372.860           33.896   
12347.000                    2         828     1323.320          661.660   
12348.000                    1         373      222.160          222.160   
12349.000                    3         993     2671.140          890.380   
12351.000                    1         261      300.930          300.930   
             purchase_frequency  
Customer ID                      
12346.000                 0.003  
12347.000                 0.000  
12348.000                 0.000  
12349.000                 0.001  
12351.000                 0.000  
"""


# 4. Repeat Rate & Churn Rate (multiple shoppers / all customers)
repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
churn_rate = 1 - repeat_rate

repeat_rate
"""
0.6706073249884098
"""

churn_rate
"""
0.3293926750115902
"""



# 5. Profit Margin (profit_margin =  total_price * 0.10)
cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10

cltv_c.head()
"""
             total_transaction  total_unit  total_price  avg_order_value  \
Customer ID                                                                
12346.000                   11          70      372.860           33.896   
12347.000                    2         828     1323.320          661.660   
12348.000                    1         373      222.160          222.160   
12349.000                    3         993     2671.140          890.380   
12351.000                    1         261      300.930          300.930   
             purchase_frequency  profit_margin  
Customer ID                                     
12346.000                 0.003         37.286  
12347.000                 0.000        132.332  
12348.000                 0.000         22.216  
12349.000                 0.001        267.114  
12351.000                 0.000         30.093  
"""



# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
cltv_c['customer_value'] = cltv_c['avg_order_value'] * cltv_c["purchase_frequency"]

cltv_c.head()
"""
             total_transaction  total_unit  total_price  avg_order_value  \
Customer ID                                                                
12346.000                   11          70      372.860           33.896   
12347.000                    2         828     1323.320          661.660   
12348.000                    1         373      222.160          222.160   
12349.000                    3         993     2671.140          890.380   
12351.000                    1         261      300.930          300.930   
             purchase_frequency  profit_margin  customer_value  
Customer ID                                                     
12346.000                 0.003         37.286           0.086  
12347.000                 0.000        132.332           0.307  
12348.000                 0.000         22.216           0.051  
12349.000                 0.001        267.114           0.619  
12351.000                 0.000         30.093           0.070  
"""


# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']

cltv_c.head()
"""
             total_transaction  total_unit  total_price  avg_order_value  \
Customer ID                                                                
12346.000                   11          70      372.860           33.896   
12347.000                    2         828     1323.320          661.660   
12348.000                    1         373      222.160          222.160   
12349.000                    3         993     2671.140          890.380   
12351.000                    1         261      300.930          300.930   
             purchase_frequency  profit_margin  customer_value    cltv  
Customer ID                                                             
12346.000                 0.003         37.286           0.086   9.784  
12347.000                 0.000        132.332           0.307 123.235  
12348.000                 0.000         22.216           0.051   3.473  
12349.000                 0.001        267.114           0.619 502.110  
12351.000                 0.000         30.093           0.070   6.373  
"""


scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_c[["cltv"]])
cltv_c["scaled_cltv"] = scaler.transform(cltv_c[["cltv"]])

cltv_c.head()
"""
             total_transaction  total_unit  total_price  avg_order_value  \
Customer ID                                                                
12346.000                   11          70      372.860           33.896   
12347.000                    2         828     1323.320          661.660   
12348.000                    1         373      222.160          222.160   
12349.000                    3         993     2671.140          890.380   
12351.000                    1         261      300.930          300.930   
             purchase_frequency  profit_margin  customer_value    cltv  \
Customer ID                                                              
12346.000                 0.003         37.286           0.086   9.784   
12347.000                 0.000        132.332           0.307 123.235   
12348.000                 0.000         22.216           0.051   3.473   
12349.000                 0.001        267.114           0.619 502.110   
12351.000                 0.000         30.093           0.070   6.373   
             scaled_cltv  
Customer ID               
12346.000          0.000  
12347.000          0.000  
12348.000          0.000  
12349.000          0.000  
12351.000          0.000  
"""


cltv_c.sort_values(by="scaled_cltv", ascending=False).head()
"""
             total_transaction  total_unit  total_price  avg_order_value  \
Customer ID                                                                
18102.000                   89      124216   349164.350         3923.195   
14646.000                   78      170342   248396.500         3184.571   
14156.000                  102      108107   196566.740         1927.125   
14911.000                  205       69722   152147.570          742.183   
13694.000                   94      125893   131443.190         1398.332   
             purchase_frequency  profit_margin  customer_value        cltv  \
Customer ID                                                                  
18102.000                 0.021      34916.435          80.937 8579573.773   
14646.000                 0.018      24839.650          57.579 4342070.458   
14156.000                 0.024      19656.674          45.565 2719105.086   
14911.000                 0.048      15214.757          35.268 1629055.810   
13694.000                 0.022      13144.319          30.469 1215855.890   
             scaled_cltv  
Customer ID               
18102.000          1.000  
14646.000          0.506  
14156.000          0.317  
14911.000          0.190  
13694.000          0.142  
"""



# Creating Segments according to its scaled_cltv

cltv_c["segment"] = pd.qcut(cltv_c["scaled_cltv"], 4, labels=["D", "C", "B", "A"])
cltv_c.head()
"""
             total_transaction  total_unit  total_price  avg_order_value  \
Customer ID                                                                
12346.000                   11          70      372.860           33.896   
12347.000                    2         828     1323.320          661.660   
12348.000                    1         373      222.160          222.160   
12349.000                    3         993     2671.140          890.380   
12351.000                    1         261      300.930          300.930   
             purchase_frequency  profit_margin  customer_value    cltv  \
Customer ID                                                              
12346.000                 0.003         37.286           0.086   9.784   
12347.000                 0.000        132.332           0.307 123.235   
12348.000                 0.000         22.216           0.051   3.473   
12349.000                 0.001        267.114           0.619 502.110   
12351.000                 0.000         30.093           0.070   6.373   
             scaled_cltv segment  
Customer ID                       
12346.000          0.000       C  
12347.000          0.000       B  
12348.000          0.000       D  
12349.000          0.000       A  
12351.000          0.000       D  
"""


cltv_c[["total_transaction", "total_unit", "total_price", "cltv", "scaled_cltv"]].sort_values(by="scaled_cltv",
                                                                                              ascending=False).head()
"""
             total_transaction  total_unit  total_price        cltv  \
Customer ID                                                           
18102.000                   89      124216   349164.350 8579573.773   
14646.000                   78      170342   248396.500 4342070.458   
14156.000                  102      108107   196566.740 2719105.086   
14911.000                  205       69722   152147.570 1629055.810   
13694.000                   94      125893   131443.190 1215855.890   
             scaled_cltv  
Customer ID               
18102.000          1.000  
14646.000          0.506  
14156.000          0.317  
14911.000          0.190  
13694.000          0.142  
"""

cltv_c.groupby("segment")[["total_transaction", "total_unit", "total_price", "cltv", "scaled_cltv"]].agg(
    {"count", "mean", "sum"})
"""
        total_transaction              total_unit                total_price  \
                      sum count   mean        sum count     mean         sum   
segment                                                                        
D                    1326  1079  1.229     117616  1079  109.005  192265.130   
C                    2160  1078  2.004     305135  1078  283.057  513016.453   
B                    4063  1078  3.769     733211  1078  680.159 1219605.200   
A                   11666  1079 10.812    4383262  1079 4062.337 6907116.491   
                               cltv                 scaled_cltv              
        count     mean          sum count      mean         sum count  mean  
segment                                                                      
D        1079  178.188     2849.328  1079     2.641       0.000  1079 0.000  
C        1078  475.897    18184.263  1078    16.869       0.002  1078 0.000  
B        1078 1131.359   103549.863  1078    96.057       0.012  1078 0.000  
A        1079 6401.405 25257295.041  1079 23408.058       2.944  1079 0.003  
"""


# function of this project
def create_cltv_c(dataframe, profit=0.10):

    # Veriyi hazırlama
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'TotalPrice': lambda x: x.sum()})
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

    # avg_order_value
    cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']

    # purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]

    # repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    # profit_margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit

    # Customer Value
    cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"])

    # Customer Lifetime Value
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(cltv_c[["cltv"]])
    cltv_c["scaled_cltv"] = scaler.transform(cltv_c[["cltv"]])

    # Segment
    cltv_c["segment"] = pd.qcut(cltv_c["scaled_cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c


df = df_.copy()
df.head()

cc = create_cltv_c(df)


