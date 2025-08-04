import pandas as pd
import io
import os

def filter_for_doors(labels: pd.DataFrame ) -> pd.DataFrame: 
    filtered_df = labels[labels.iloc[:, 0] == 0]
    filtered_df_reset = filtered_df.reset_index(drop=True)
    return filtered_df_reset
    
def main(): 
    root_path = "./floor_plans_500_yolov8/"

    for split in os.listdir(root_path): 
        split_label_dir = os.path.join(root_path, split) + '/labels'
    #     print(root_path + split)
    #     print(split_label_dir)
        if os.path.isdir(split_label_dir):
    #         print(split_label_dir)
            for label in os.listdir(split_label_dir):
                label_path = os.path.join(split_label_dir, label)
                if os.path.isfile(label_path):
                    print(label_path)
                    annotations_file = open(label_path)
                    annotations = annotations_file.read()
                    labels_df = pd.read_csv(io.StringIO(annotations), sep=' ', header=None)
                    labels_df = filter_for_doors(labels_df)
                    print(labels_df)
                    labels_df.to_csv(label_path, sep=' ', header=False, index=False)
    
if __name__ == "__main__":
    main()