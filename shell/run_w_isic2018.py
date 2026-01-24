import subprocess

# 固定参数
dataset = 'isic2018'
data_path = '../dataset/isic2018/'
gpu = 5
epochs = 150

# 不同的weight值
weight_list = [0, 0.001, 0.01, 0.1, 1]

for w in weight_list:
    cmd = [
        'python', '../train_procbm.py',
        '-d', dataset,
        '--data-path', data_path,
        '--gpu', str(gpu),
        '-e', str(epochs),
        '-w', str(w)
    ]
         
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
