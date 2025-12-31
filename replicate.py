import pandas as pd

df = pd.read_csv("/home/giangpv102/Documents/BigData/BTL/IMDB/IMDB.csv")

# Bỏ dòng đầu tiên
df = df.iloc[1:].reset_index(drop=True)

print("Số dòng sau khi bỏ dòng đầu:", len(df))

N = 100
big_df = pd.concat([df] * N, ignore_index=True)

print("Số dòng sau khi nhân:", len(big_df))

big_df.to_csv("IMDB_Datasets.csv", index=False)
print("Đã tạo IMDB_Datasets.csv")