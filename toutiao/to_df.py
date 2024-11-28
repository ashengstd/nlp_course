import pandas as pd
def process_file(file):
    new_df = pd.DataFrame(columns=['question','answer'])
    line_no = 0
    with open(file=file,encoding="utf-8") as f:
        while True:
            try:
                line = f.readline()
                if line is not None :
                    strs = line.split("_!_")
                    new_df.loc[line_no] = [strs[3],strs[2]]
                    line_no += 1
                if line_no ==100:
                    break
            except EOFError:
                break        
    new_df.to_csv("./data/toutiao/val.csv",index=False)
 

if __name__ == "__main__":
    process_file("./data/toutiao/validation.txt")
    