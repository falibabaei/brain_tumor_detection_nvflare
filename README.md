# Brain Tumor Detection using NVflare for Federated Learning

## Overview

This repository implements brain tumor detection using NVflare, a framework for enhancing federated learning tasks. The example is based on code from the [NVflare examples repository](https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/ml-to-fl/pt), modified for federated learning.

The original code is available in [this Colab notebook](https://colab.research.google.com/drive/1vgNSPBQOPAYvEkqwXc7pr3Y1KL5fRsn3).

## Dataset

The dataset used is from Kaggle: [Brain Tumor Dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset).

## Running using NVflare Simulator

### Split the data between clients (sites):

1. **Download the dataset from Kaggle: [Brain Tumor Dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset).**

2. **Run the data split script:**
    ```bash
    python3 data_split --data_path path/to/imgs --site_num n --site_name_prefix site --size_valid m --out_path path/to/output_dir --split_method uniform
    ```
    This script generates a JSON file defining data paths for each site.

### Copy the job template:

1. After modifying the training script, place it into a job structure. Refer to the [NVFlare job structure documentation](https://nvflare.readthedocs.io/en/latest/real_world_fl/job.html) for details.

2. **Example job structure:**
    ```
    my_job
    ├── app
    │   ├── config
    │   │   ├── config_fed_client.conf
    │   │   └── config_fed_server.conf
    │   └── custom
    │       ├── brain_tumor_client.py
    │       ├── data_split.py
    │       ├── __init__.py
    │       ├── load_data.py
    │       ├── net.py
    │       └── tumor_detection_decorator.py
    └── meta.conf
    ```

### Run the simulator:

```bash
nvflare simulator -n 2 -t 2 ./mx_job -w my_workspace
```
Results will be saved in my_workspace.
 
## Project Structure
```
├── code
│   ├── brain_tumor_client.py
│   ├── data_split.py
│   ├── __init__.py
│   ├── load_data.py
│   ├── net.py
│   └── tumor_detection_decorator.py
├── jobs
│   ├── client_api
│   │   ├── app
│   │   │   ├── config
│   │   │   │   ├── config_fed_client.conf
│   │   │   │   └── config_fed_server.conf
│   │   │   └── custom
│   │   │       ├── brain_tumor_client.py
│   │   │       ├── brain_tumor_decorator_origin.py
│   │   │       ├── data_split.py
│   │   │       ├── __init__.py
│   │   │       ├── load_data.py
│   │   │       ├── net.py
│   │   │       ├── __pycache__
│   │   │       │   ├── load_data.cpython-39.pyc
│   │   │       │   └── net.cpython-39.pyc
│   │   │       └── tumor_detection_decorator.py
│   │   └── meta.conf
│   └── decorator
│       ├── app
│       │   ├── config
│       │   │   ├── config_fed_client.conf
│       │   │   └── config_fed_server.conf
│       │   └── custom
│       │       ├── brain_tumor_client.py
│       │       ├── data_split.py
│       │       ├── __init__.py
│       │       ├── load_data.py
│       │       ├── net.py
│       │       └── tumor_detection_decorator.py
│       └── meta.conf
 ```
