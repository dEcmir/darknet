import numpy as np
import h5py

from sklearn.svm import SVC

nbdata1 = 500
nbdata2 = 500
# load some training data
f1 = h5py.File('step10.hdf5', 'r')
f2 = h5py.File('step18.hdf5', 'r')
X1 = f1['features'][0:nbdata1]
X2 = f2['features'][0:nbdata2]
y1 = np.ones(nbdata1)
y2 = np.zeros(nbdata2)

X = np.empty((nbdata1+nbdata2, X1.shape[1]), dtype=np.float32)
y = np.empty((nbdata1+nbdata2), dtype=np.float32)
indices1 = np.random.choice(nbdata1+nbdata2, size=nbdata1, replace=False)
X[indices1] = X1
y[indices1] = y1

indices2 = np.array(list(set(range(nbdata1+nbdata2)) - set(indices1)))
X[indices2] = X2
y[indices2] = y2


# regress using a SVM
clf = SVC(C=0.1, cache_size=500)#, kernel='sigmoid')

clf.fit(X, y)
print "Classification done"
# np.savetxt("support.txt", clf.support_)

#
# # This is the obtained classifier
# c0 = clf.intercept_.copy()
# alpha = clf.dual_coef_.copy()
# vectors = clf.support_vectors_.copy()
# gamma = 1./X.shape[1]
# # this is only for test
# dec = lambda x: np.sign(c0 + (alpha* np.tanh(gamma * vectors.dot(x))).sum())
# # serialisation
# with open("svm.txt", "w") as out:
#     out.write("{:d} {:d}\r\n".format(vectors.shape[0], vectors.shape[1]))
#     out.write("{:f}\r\n".format(gamma))
#     out.write("{:f}\r\n".format(c0[0]))
#
#     for i in range(alpha.shape[1]):
#         out.write("{:f} ".format(alpha[0, i]))
#     out.write("\r\n")
#
#     for i in range(vectors.shape[0]):
#         for j in range(vectors.shape[1]):
#             out.write("{:f} ".format(vectors[i, j]))
#         out.write("\r\n")
#     out.flush()
#     out.close()

# test on other data chunks (separated for speed ...)

for i in range(6):
    Xv = f1['features'][i*100:(i+1)*100]
    Yv = np.ones(Xv.shape[0])
    print "Positive Classification rate :",
    print float(np.equal(clf.predict(Xv), Yv).sum()) / Yv.shape[0]

for i in range(6):
    Xv = f2['features'][i*100:(i+1)*100]
    Yv = np.zeros(Xv.shape[0])
    print "Negative Classification rate :",
    print float(np.equal(clf.predict(Xv), Yv).sum()) / Yv.shape[0]

f1.close()
f2.close()
