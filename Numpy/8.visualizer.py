import numpy as np
import matplotlib.pyplot as plt

# 3 AXE

tensor = np.random.rand(3, 3, 3)

x, y, z = np.indices(tensor.shape)
x, y, z = x.flatten(), y.flatten(), z.flatten()
values = tensor.flatten()

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z, c=values, cmap='viridis', s=100, edgecolor='k')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Tensor 3D vizualizat')

cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label('Valoare element')

plt.show()




# # 2 AXE
#
# tensor = np.random.rand(3, 3)
#
# x, y = np.indices(tensor.shape)
# x, y = x.flatten(), y.flatten()
# values = tensor.flatten()
#
# fig = plt.figure(figsize=(7, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# sc = ax.scatter(x, y, c=values, cmap='viridis', s=100, edgecolor='k')
#
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_title('Tensor 2D vizualizat')
#
# cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
# cbar.set_label('Valoare element')
#
# plt.show()
