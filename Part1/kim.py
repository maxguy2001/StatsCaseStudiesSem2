import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("houseprices.csv")
print(df.head)


	
#plt.plot(df.SalePrice)
#plt.show()


oddsales = df[(df['SalePrice'] >= 700000)]

#Lot area does not show much impact on sale price apart from a few outliers
#plt.scatter(df.SalePrice, df.LotArea) 
#plt.show() 


#Neighbourhood good, shows each house in each neighbourhood are similar
#plt.scatter(df.SalePrice, df.Neighborhood) 
#plt.show() 

# good plot, same as before
#plt.scatter(df.SalePrice, df.BldgType) 
#plt.show() 

#Maybe not a big detering fctor, as many are still around the same price no matter the price
#plt.scatter(df.SalePrice, df.OverallCond) 
#plt.show() 


#
#plt.scatter(df.SalePrice, df.BedroomAbvGr) 
#plt.show() 