import numpy as np
import matplotlib.pyplot as plt
import umap
from cifar10_web import cifar10

train_images, train_labels, test_images, test_labels = cifar10(path='../CIFAR10_DATASET')

logits=np.argmax(train_labels,axis=1)
logits_test=np.argmax(test_labels,axis=1)

mapper=umap.UMAP(n_components=2,n_neighbors=20,verbose=True).fit(train_images, y=logits)
embedding=mapper.transform(train_images)
fig, ax = plt.subplots(1, figsize=(14, 10))
plt.scatter(*embedding.T, s=0.1, c=logits, cmap='Spectral', alpha=1.0)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
cbar.set_ticks(np.arange(10))
plt.title('CIFAR10 Embedded via UMAP using Labels');
plt.show()

test_embedding=mapper.transform(test_images)

fig, ax = plt.subplots(1, figsize=(14, 10))
plt.scatter(*test_embedding.T, s=2, c=logits_test, cmap='Spectral', alpha=1.0)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
cbar.set_ticks(np.arange(10))
plt.title('CIFAR10 Embedded test set via UMAP');
plt.show()