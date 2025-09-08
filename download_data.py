from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("R2E-Gym/R2E-Gym-V1")

print("Column names:", ds.column_names)
print("First few rows:")
print(ds['train'].select(range(5)))
