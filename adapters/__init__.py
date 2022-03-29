"""
Lists available adapters.
Author: Ryan Sheatsley
Thu Feb 10 2022
"""
import adapters.baseadapter  # Base Adapter class for custom datasets
import adapters.nslkdd  # Adapter for the NSL-KDD
import adapters.phishing  # Adapter for the Phishing dataset

available = (adapters.nslkdd.NSLKDD, adapters.phishing.Phishing)
