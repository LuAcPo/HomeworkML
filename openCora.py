import numpy as np
import scipy.sparse as sp
import os
import pickle as pkl
import itertools
# 入口函数，需要加载cora数据调用这个即可
def load_ind_cora_data(cora_path='cora/raw'):
	x, y, tx, ty, allx, ally, graph, test_index = load_ind_cora_pkl(cora_path)
	train_index = np.arange(y.shape[0])					# 训练集的范围 [0, ysize)
	val_index = np.arange(y.shape[0], y.shape[0] + 500)	# 验证集的范围 [ysize, ysize+500)
	sorted_test_index = sorted(test_index)				# 测试集的范围 

	x = np.concatenate((allx, tx), axis=0)		# 全体的X数据
	y = np.concatenate((ally, ty), axis=0).argmax(axis=1)	# 全体的Y数据

	x[test_index] = x[sorted_test_index]
	y[test_index] = y[sorted_test_index]
	num_nodes = x.shape[0]

	train_mask = np.zeros(num_nodes, dtype='bool')
	val_mask = np.zeros(num_nodes, dtype='bool')
	test_mask = np.zeros(num_nodes, dtype='bool')

	train_mask[train_index] = True
	val_mask[val_index] = True
	test_mask[test_index] = True

	adj_matrix = build_adjacency(graph)		# 构建邻接矩阵
	return x, y, adj_matrix.toarray(), train_mask, val_mask, test_mask


def load_ind_cora_pkl(cora_path):
	dataset_str = 'cora'
	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
	objects = []
	for i in range(len(names)):
		file_path = os.path.join(cora_path, 'ind.{}.{}'.format(dataset_str, names[i]))
		if names[i] == 'test.index':
			out = np.genfromtxt(file_path, dtype='int64')
		else:
			out = pkl.load(open(file_path, 'rb'), encoding="latin1")
			out = out.toarray() if hasattr(out, 'toarray') else out
		objects.append(out)

	x, y, tx, ty, allx, ally, graph, test_index = tuple(objects)

	return x, y, tx, ty, allx, ally, graph, test_index

def build_adjacency(adj_dict):
	edge_index = []
	num_nodes = len(adj_dict)   # 2708,点的总数
	for src, dst in adj_dict.items():
		edge_index.extend([src, v] for v in dst)
		edge_index.extend([v, src] for v in dst)
	edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
	edge_index = np.asarray(edge_index)
	adj_matrix = sp.coo_matrix((np.ones(len(edge_index)), 
								(edge_index[:, 0], edge_index[:, 1])),
								shape=(num_nodes, num_nodes), 
								dtype='float32')
	return adj_matrix
x, y, adj_matrix, train_mask, val_mask, test_mask=load_ind_cora_data()
output_dir = './cora_numpy'
np.save(os.path.join(output_dir, 'features.npy'), x)
np.save(os.path.join(output_dir, 'labels.npy'), y)
np.save(os.path.join(output_dir, 'adjacency_matrix.npy'), adj_matrix)
np.save(os.path.join(output_dir, 'mk_train.npy'), train_mask)
np.save(os.path.join(output_dir, 'mk_val.npy'), val_mask)
np.save(os.path.join(output_dir, 'mk_test.npy'), test_mask)