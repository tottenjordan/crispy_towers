# crispy_towers
implementing TFRS on Google Cloud's Vertex AI platform

<img src='imgs/deep_fried_lotr.png'>


## getting started


1. [env_config.py](env_config.py) - edit these
2. [00-env-setup.ipynb](00-env-setup.ipynb) - install packages with poetry
3. [00_data_prep.ipynb](notebooks/00_data_prep.ipynb) - prepare movielens dataset


# GPUs


This repo tested from Vertex Workbench instance with an A100 attached. We need to ensure compatability between GPU driver, cuda, cudnn, and the TF-based packages. The below commands assume Linux-based machine; see this guide for [finding nvidia driver version](https://www.cyberciti.biz/faq/check-print-find-nvidia-driver-version-on-linux-command/)

### display the installed driver revision and the version of the GNU C compiler used to build the Linux kernel module

> `cat /proc/driver/nvidia/version`

```
NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  550.90.07  
Release Build  (dvs-builder@U16-I2-C05-15-3)  Fri May 31 09:44:37 UTC 2024
GCC version:  gcc version 10.2.1 20210110 (Debian 10.2.1-6)
```

### see loaded version type

> `cat /sys/module/nvidia/version`

```
550.90.07
```

### the NVIDIA system management interface (smi) command returns detailed info about GPU utilization, memory usage, temperature, etc.

> `nvidia-smi`

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  |   00000000:00:04.0 Off |                    0 |
| N/A   29C    P0             44W /  400W |       1MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```