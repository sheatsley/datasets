"""
Lists available adapters.
Author: Ryan Sheatsley
Thu Feb 10 2022
"""
import adapters.baseadapter  # Base Adapter class for custom datasets
import adapters.nslkdd  # Adapter for the NSL-KDD
import adapters.phishing  # Adapter for the Phishing dataset
import adapters.unswnb15  # Adapter for the UNSW-NB15

available = (
    adapters.nslkdd.NSLKDD,
    adapters.phishing.Phishing,
    adapters.unswnb15.UNSWNB15,
)
