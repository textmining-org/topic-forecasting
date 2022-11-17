import os

import treform as ptm
import pandas as pd


def make_coword_count(datasets_path, target_name):
    preprocessed_file = f'./{target_name}_preprocessed.txt'
    coword_count_file = f'./{target_name}_coword_count.txt'
    if not os.path.exists(preprocessed_file):
        preprocessed_path = os.path.join(datasets_path, target_name + '.pkl')
        df_datasets = pd.read_pickle(preprocessed_path)

        with open(preprocessed_file, 'w', encoding='utf-8') as fout:
            documents = df_datasets['text'].values.tolist()

            for document in documents:
                print(' '.join(document))
                fout.write(' '.join(document) + "\n")
        fout.close()

    # CooccurrenceExternalManager 내부에서 os.chdir()을 통해 path 변경
    # 복원을 위해 현재 path 저장(복원하지 않을경우 program_path 값으로 설정되어 path 접근 불편)
    current_path = os.getcwd()
    co_occur = ptm.cooccurrence.CooccurrenceExternalManager(
        program_path=current_path + '/external_programs',
        input_file=preprocessed_file,
        output_file=coword_count_file,
        threshold=1, num_workers=8)
    co_occur.execute()
    # path 복원
    os.chdir(current_path)


if __name__ == '__main__':
    datasets_path = os.path.abspath('../../_datasets')
    print(datasets_path)

    make_coword_count(datasets_path, 'patents')
