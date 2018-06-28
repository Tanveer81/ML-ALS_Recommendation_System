import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import scipy


def als(train, k, x_train, lmb, lmbu, lmbv):
    print("In ALS")
    # print(train)
    matrix = np.matrix(train)
    # print(matrix.shape)
    # print(matrix)
    # print(matrix[2,1])
    unique_user = x_train['reviewerID'].unique()
    unique_product = x_train['itemID'].unique()
    # print(unique_user)
    # print(unique_product)

    n = (unique_user.shape[0])
    m = (unique_product.shape[0])
    # print("n:")
    # print(n)
    # print("m:")
    # print(m)

    U = np.random.rand(n, k)
    V = np.random.rand(k, m)

    loss = 0

    # print("Old V:")
    # print(V.shape)
    # print(V)

    for qq in range(0, 150):
        print(qq)
        print(matrix)
        # print("Starting Item Loop")
        for i in range(0, m):
            # print(i)
            b = matrix[i, :]
            b = np.array(b)
            # print(b[0])
            z = np.where(b[0] != 0)[0]
            xm = b[0][z]
            # print(z)
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
            vm = np.matmul(np.linalg.inv(np.matmul(u.T, u) + lmb * np.identity(k)), np.matmul(u.T, xm))
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
            un = np.matmul(np.linalg.inv(np.matmul(v.T, v) + lmb * np.identity(k)), np.matmul(v.T, xn))
            # print(un.shape)
            # print("un ")
            # print(un)
            U[i] = un

        # print("New U:")
        # print(U)
        # print("asdasdasd")
        # print(U[0])
        # print(V[:,0])
        # print(np.matmul(U[0],V[:,0]))
        print("Calculating Error")
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

        print("Loss")
        print(temp_loss)





def main():
    np.random.seed(5)
    df = pd.read_csv("train.csv")
    df = df[:10]
    # print(df.shape)
    # print(df)

    # myset = set(df[:, 0])

    x_train, x_temp = train_test_split(df, test_size=0.4)
    # print(x_train.shape)
    # print(x_temp.shape)

    x_test, x_val = train_test_split(x_temp, test_size=0.5)
    # print(x_test.shape)
    # print(x_val.shape)

    train = x_train.pivot_table(columns=['reviewerID'], index=['itemID'], values='rating')
    train.fillna(value=0, inplace=True)
    print(train)
    als(train, 2, x_train, .1, .1, .1)


if __name__ == "__main__":
    main()
