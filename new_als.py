import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
import time
from operator import itemgetter
import xlrd


def als(train, k, iteration, x_train, lmbu, lmbv , train_X):
    print("In ALS")
    # print(train)
    # matrix = np.matrix(train)///////////////
    matrix = train_X.T
    # print(matrix.shape)
    # print(matrix)
    # print(matrix[2,1])
    # unique_user = x_train['reviewerID'].unique()//////////////
    # unique_product = x_train['itemID'].unique()//////////////
    unique_user = matrix.shape[1]
    unique_product = matrix.shape[0]
    # print(unique_user)
    # print(unique_product)

    # n = (unique_user.shape[0])
    # m = (unique_product.shape[0])
    n = unique_user
    m = unique_product
    print("n:")
    print(n)
    print("m:")
    print(m)

    U = np.random.rand(n, k)
    V = np.random.rand(k, m)

    loss = 0

    # print("Old V:")
    # print(V.shape)
    # print(V)

    for qq in range(0, iteration):
        # print(qq)
        # print(matrix)
        # print("Starting Item Loop")
        for i in range(0, m):
            # print(i)
            # print(i)
            b = matrix[i, :]
            b = np.array(b)
            # print(b)
            # print(b.shape)
            z = np.where(b != 0)[0]
            xm = b[z]
            # print(xm)
            # print(xm)
            u = U[z, :]
            # print(u)
            # print("u.shape")
            # print(u.shape)
            # print("uT.u")
            # j = np.matmul(u.T, u)
            # print(j.shape)
            # print(j)
            # I = lmb * np.identity(k)
            vm = np.matmul(np.linalg.inv(np.matmul(u.T, u) + lmbv * np.identity(k)), np.matmul(u.T, xm))
            # print(vm.shape)
            # print("vm ")
            # print(vm)
            V[:, i] = vm

        # print("New V:")
        # print(V)

        # print("Old U:")
        # print(U.shape)
        # print(U)

        # print("Starting User Loop")
        for i in range(0, n):
            # print(i)
            b = matrix[:, i]
            # print(b)
            b = np.array(b.T)
            # print(b)
            z = np.where(b != 0)[0]
            xn = b[z]
            # print(z)
            # print(xn)
            v = V.T[z, :]
            # print(v)
            # print("v.shape")
            # print(v.shape)
            # print("vT.v")
            # j = np.matmul(v.T, v)
            # print(j.shape)
            # print(j)
            # I = lmb * np.identity(k)
            un = np.matmul(np.linalg.inv(np.matmul(v.T, v) + lmbu * np.identity(k)), np.matmul(v.T, xn))
            # print("un shape")
            # print(un[0].shape)
            # print("un ")
            # print(un)
            U[i] = un

        # print("New U:")
        # print(U)
        # print("asdasdasd")
        # print(U[0])
        # print(V[:,0])
        # print(np.matmul(U[0],V[:,0]))
        # print("Calculating Error")
        temp_loss = 0
        for i in range(0, n):
            for j in range(0, m):
                d = np.matmul(U[i], V[:, j])
                if matrix[j, i] != 0:
                    temp_loss += (matrix[j, i] - d) ** 2

        for i in range(0, n):
            temp_loss += np.sum(np.square(U[i]))

        for i in range(0, m):
            temp_loss += np.sum(np.square(V[:, j]))

        # print("Loss")
        print(temp_loss)

        if np.abs(loss - temp_loss) < .01:
            break

    return V


def validation(V, k, train, valid, x_train, x_val, lmbu, lmbv, q):
    if q == 1:
        print("Validation")
    elif q == 2:
        print("Test")
    elif q == 3:
        print("recommend")
    # print(x_val)
    # print(valid)
    # matrix2 = np.matrix(train)
    matrix = np.matrix(valid)
    # print(matrix)
    # print(np.count_nonzero(matrix))
    unique_product = x_train['itemID'].unique()
    unique_user = x_train['reviewerID'].unique()
    unique_product2 = x_val['itemID'].unique()
    unique_user2 = x_val['reviewerID'].unique()
    # n = (unique_user.shape[0])
    m2 = (unique_product.shape[0])
    n = (unique_user2.shape[0])
    m = (unique_product2.shape[0])
    U = np.random.rand(n, k)

    len = 0

    for i in unique_product:
        # print(i)
        if i not in unique_product2:
            len = len + 1

    z = np.zeros((k, len))
    V = np.append(V, z, axis=1)

    for i in range(0, n):
        # print(i)
        b = matrix[:, i]
        # print(b)
        b = np.array(b.T)
        # print(b)
        z = np.where(b[0] != 0)[0]
        xn = b[0][z]
        # print(z)
        # print(xn)
        v = V.T[z, :]
        # print(v)
        # print("v.shape")
        # print(v.shape)
        # print("vT.v")
        # j = np.matmul(v.T, v)
        # print(j.shape)
        # print(j)
        # I = lmb * np.identity(k)
        un = np.matmul(np.linalg.inv(np.matmul(v.T, v) + lmbu * np.identity(k)), np.matmul(v.T, xn))
        # print("un shape")
        # print(un[0].shape)
        # print("un ")
        # print(un)
        U[i] = un

    if q != 3:
        temp_loss = 0
        for i in range(0, n):
            for j in range(0, m):
                d = np.matmul(U[i], V[:, j])
                if matrix[j, i] != 0:
                    temp_loss += (matrix[j, i] - d) ** 2

        temp_loss = temp_loss / np.count_nonzero(matrix)
        temp_loss = temp_loss ** .5

        print("Loss")
        print(temp_loss)

    else:
        result = []
        for j in range(0, m2):
            d = np.matmul(U[0], V[:, j])
            # print(d)
            if d < 0:
                d = 0
            if d > 5:
                d = 5
            result.append((x_train['reviewerID'][j], d))
            # result.append(j, d))
        result.sort(key=itemgetter(1), reverse=True)

        print("Recommending Products")
        print(result[0:10])


def main():
    k = 30
    lmbu = .1
    lmbv = .1
    l = .1
    np.random.seed(5)
    # x_train = pd.read_csv("traincustom.csv")
    # # df = df[:10]
    # x_val = pd.read_csv("validcustom.csv")
    # x_test = pd.read_csv("recommend.csv")
    # x_rec = pd.read_csv("recommend.csv")
    # print(df.shape)
    # print(df)
    # myset = set(df[:, 0]
    # x_train, x_temp = train_test_split(df, test_size=0.4)
    # print(x_train.shape)
    # print(x_temp.shape)
    # x_test, x_val = train_test_split(x_temp, test_size=0.5)
    # print(x_test.shape)
    # print(x_val.shape)
    # train = x_train.pivot_table(columns=['reviewerID'], index=['itemID'], values='rating')
    # train.fillna(value=0, inplace=True)
    # print("printing training data")
    # print(train)

    #   trining
    #   als(train, k, iteration, x_train, lmbu, lmbv)
    #     for k in [10, 20, 30, 40, 50]:
    #         print("K ")
    #         print(k)
    #         for l in [.01, .1, 1, 10]:
    #             print("Rate ")
    #             print(l)

    DataFrame = pd.read_excel("ratings_train.xlsx", header=None)
    # DataFrame = DataFrame[:10]
    train_X = np.array(DataFrame)

    print(train_X.shape)

    for i in range(0, train_X.shape[0]):
        for j in range(0, train_X.shape[1]):
            if train_X[i, j] == -1:
                train_X[i, j] = 0

    # print(train_X)
    V = als(None, k, 20, None, l, l, train_X)
#     # validation
#     valid = x_val.pivot_table(columns=['reviewerID'], index=['itemID'], values='rating')
#     valid.fillna(value=0, inplace=True)
#     # print("printing validation data")
#     # print(valid)
#     validation(V, k, train, valid, x_train, x_val, l, l, 1)
#
#     #   test
#     test = x_test.pivot_table(columns=['reviewerID'], index=['itemID'], values='rating')
#     test.fillna(value=0, inplace=True)
#     # print("printing test data")
#     # print(test)
#     validation(V, k, train, test, x_train, x_test, l, l, 2)
#     # time.sleep(120)
#     d = x_train['itemID']
#     print(d)
#
#
# # #   rec
#     rec = x_rec.pivot_table(columns=['reviewerID'], index=['itemID'], values='rating')
#     rec.fillna(value=0, inplace=True)
# #     # print("printing test data")
# #     # print(test)
#     validation(V, k, train, rec, x_train, x_rec, lmbu, lmbv, 3)


if __name__ == "__main__":
    main()
