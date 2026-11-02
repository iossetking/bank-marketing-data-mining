# Cafe Sales Data Mining

## Requirements

- uv 

## Dataset

Cafe Sales

### URL

https://huggingface.co/datasets/adityasuyal/cafe-sales-cleaned/blob/main/dirty_cafe_sales.csv

### Download

```
mkdir -p datasets/raw && cd datasets/raw
curl -L -o dirty_cafe_sales.csv https://huggingface.co/datasets/adityasuyal/cafe-sales-cleaned/resolve/main/dirty_cafe_sales.csv
```

## Install Libraries

### uv (recommended)

```
uv sync
```

### pip

```
pip install .
```
