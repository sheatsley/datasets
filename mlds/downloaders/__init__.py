"""
Lists available adapters.
Author: Ryan Sheatsley
Thu Feb 10 2022
"""
from mlds.adapters import (
    cicmalmem2022,  # Adapter for the CIC-MalMem-2022
    nslkdd,  # Adapter for the NSL-KDD
    phishing,  # Adapter for the Phishing dataset
    unswnb15,  # Adapter for the UNSW-NB15
)

available = (
    cicmalmem2022.CICMalMem2022,
    nslkdd.NSLKDD,
    phishing.Phishing,
    unswnb15.UNSWNB15,
)
