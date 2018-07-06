import numpy as np

b = np.array([[0, 2], [0, 4]])
a = np.array([[1, 3], [7, 0]])
# i = np.where(b == 0)[0]
# print(i)
# c = b[i]
# print(c)
# m = np.where(b != 0)
# c = b[m]
# print(np.sum(c))
# t = b != 0
# print(t)
# d = (b-a)**2 * t
# print(np.sum(d))

# print(np.multiply(c, t))


# U = np.array([[1,2,3,4],
#             [5,6,7,8],
#             [9,10,11,12],
#             [13,14,15,16]])
# print(U)
# print("\n")
# print(U[:, :])
# print("\n")
# print(U[:, 0])
# print("\n")
# print(U[0, :])
# print("\n")
# print(U[0, 0])
# print("\n")
# print(U[0:4, 2:3])


# # transpose
# x = np.array([[1, 2], [3, 4]])
# print(x)
# z = np.transpose(x)
# f = x.T
# print(z)
# print(f)
# # inverse
# y = np.linalg.inv(x)
# print(x)
# print(y)
# print(np.dot(x, y))


# print(.1*np.identity(3))

# b = [1, 2, 3]
# print(np.sum(np.square(b)))
#
#
# a = [1, 0, 0, 2, 0, 5, 8]
# print(np.count_nonzero(a))

a = np.array([5,3,4,7])
a.sort()
print(a[0:2])
v = 5
file = open("testfile.txt", "w")



file.write("\n")

file.write("sdfsdasfas\n")

file.write("sdfsdasfas")



file.close()