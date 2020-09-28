import pandas as pd

DATA_FOLDER = 'prepared'
for split in ['dev']: #, 'test', 'train']:
    df = pd.read_json(f'{DATA_FOLDER}/{split}.jsonl', lines=True).set_index('id')
    for id_, row in df.iterrows():
        print(row)
        with open(f'{DATA_FOLDER}/{row["img"]}.ocr', 'w') as ocr:
            ocr.write(row['text'])
