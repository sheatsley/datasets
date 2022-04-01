"""
Lists available adapters.
Author: Ryan Sheatsley
Thu Feb 10 2022
"""
import adapters.baseadapter  # Base Adapter class for custom datasets
import adapters.cicmalmem2022  # Adapter for the CIC-MalMem-2022
import adapters.nslkdd  # Adapter for the NSL-KDD
import adapters.phishing  # Adapter for the Phishing dataset
import adapters.unswnb15  # Adapter for the UNSW-NB15

available = (
    adapters.cicmalmem2022.CICMalMem2022,
    adapters.nslkdd.NSLKDD,
    adapters.phishing.Phishing,
    adapters.unswnb15.UNSWNB15,
)
