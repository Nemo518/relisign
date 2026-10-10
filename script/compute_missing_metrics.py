import csv
for name in ['ablation_conf_weighted', 'ablation_single_stage']:
    for split in ['dev100', 'test100']:
        path = f'E:/project/xiaozhi/Uni-Sign-storage/out/{name}/{split}_eval.csv'
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            sims = [float(row['similarity']) for row in reader]
            avg = sum(sims) / len(sims)
            print(f'{name} {split}: AvgSim={avg:.4f}, n={len(sims)}')
