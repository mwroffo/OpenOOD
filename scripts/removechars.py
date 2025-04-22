import os

root_dir = '/Users/_mexus/localDocuments/schoolDocs/s25/682/682-proj/682finalproject/OpenOOD/data/benchmark_imglist/osr_cub'

for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.txt'):
            file_path = os.path.join(dirpath, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            with open(file_path, 'w') as f:
                for line in lines:
                    f.write(line[5:] if len(line) > 5 else '\n')
