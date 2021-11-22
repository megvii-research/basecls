#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import ctypes
import os
import subprocess

__all__ = ["set_nccl_env", "set_num_threads"]


def set_nccl_env():
    """Set NCCL environments, which is essential to multi-node training."""
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; popd > /dev/null"
    )
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"


def set_num_threads(num: int = 1):
    """Set number of threads in OpenMP, OpenCV, MKL, OPENBLAS, VECLIB, NUMEXPR, etc.

    Args:
        num: number of threads. Default: 1
    """
    try:
        import mkl

        mkl.set_num_threads(num)
    except Exception:
        pass

    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num)))
        except Exception:
            pass

    os.environ["OMP_NUM_THREADS"] = str(num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num)
    os.environ["MKL_NUM_THREADS"] = str(num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num)
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

    try:
        import cv2

        cv2.setNumThreads(num)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
