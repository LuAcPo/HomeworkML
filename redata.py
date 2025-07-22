from torch_geometric.datasets import Planetoid
import os
import urllib.request
import shutil

# 创建目录结构
os.makedirs("./Cora/raw", exist_ok=True)
print("ok")
# 需要下载的文件列表
files = [
    'ind.cora.x', 'ind.cora.tx', 'ind.cora.allx',
    'ind.cora.y', 'ind.cora.ty', 'ind.cora.ally',
    'ind.cora.graph', 'ind.cora.test.index'
]

# 下载所有文件
base_url = "https://github.com/kimiyoung/planetoid/raw/master/data/"
for file in files:
    url = base_url + file
    dest = f"./Cora/raw/{file}"
    print(f"下载 {url} 到 {dest}")
    with urllib.request.urlopen(url) as response, open(dest, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

# 现在加载数据集
dataset = Planetoid(root='./', name='Cora', split='public')