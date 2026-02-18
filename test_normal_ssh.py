
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import paramiko
import struct
from io import BytesIO

# Simple test to see if we can get even ONE normal sample
SSH_CONFIG = {
    'hostname': '202.90.198.224',
    'port': 4343,
    'username': 'precursor',
    'password': 'otomatismon'
}

def test_single_normal():
    print("Connecting...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(**SSH_CONFIG, timeout=10)
    sftp = ssh.open_sftp()
    print("Connected.")
    
    # Try a known path for 2025-01-01
    path = "/home/precursor/SEISMO/DATA/SBG/SData/2501/S250101.SBG"
    try:
        with BytesIO() as buffer:
            sftp.getfo(path, buffer)
            buffer.seek(0)
            data = buffer.read()
            print(f"Success! Read {len(data)} bytes from {path}")
    except Exception as e:
        print(f"Failed to read {path}: {e}")
    finally:
        sftp.close()
        ssh.close()

if __name__ == "__main__":
    test_single_normal()
