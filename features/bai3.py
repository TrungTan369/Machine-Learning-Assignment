import kagglehub

# Download latest version
path = kagglehub.dataset_download("jcoral02/inriaperson")

print("Path to dataset files:", path)
