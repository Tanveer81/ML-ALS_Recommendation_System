To convert dataframe to matrix:
df=pd.read_csv("train.csv")
matrix=df.pivot_table(columns=['itemID'],index=['reviewerID'],values='rating')