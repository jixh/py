import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns

data = datasets.load_iris()
X = data.data
color = data.target
sns.set_style("white")


from sklearn import decomposition
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from time import time

n_components = 2
n_neighbors = 10
data = datasets.load_iris()
X = data.data
color = data.target

fig = plt.figure(figsize=(15, 4))


t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(151)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Set1)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(152)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Set1)
plt.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(153)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Set1)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

t0 = time()
pca = decomposition.PCA(n_components=n_components)
pca.fit(X)
Y = pca.transform(X)

t1 = time()
print("PCA: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(154)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Set1)
plt.title("PCA (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0, perplexity=40)
Y = tsne.fit_transform(X)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(155)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Set1)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

plt.show()
