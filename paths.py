import os
import pandas as pd


project_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
project_name = os.path.basename(project_path)
server_name = os.uname()[1]
remote_root = "/data/hanyang/sepsis/"


prefix = "cohort3"
raw_data_path = os.path.join(remote_root, "cohort_3")
manual_data_path = os.path.join(remote_root, "manual_tables")
remote_project_path = os.path.join(remote_root, project_name)
processed_data_path = os.path.join(raw_data_path, "data_processed")
tmp_data_path = os.path.join(raw_data_path, "data_tmp")

df_embed = pd.read_csv(os.path.join(processed_data_path, "icd10_embeddings_128.csv"))
comorb_vocab_size = len(df_embed)


if __name__ == "__main__":
    print(project_path)
    print(project_name)
    print(remote_root)
    print(remote_project_path)
    print(raw_data_path)
    print(manual_data_path)
    print(processed_data_path)
    print(tmp_data_path)
