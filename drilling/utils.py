"""Utility functions for logging, cleaning, etc."""

import logging
from colorama import Fore, Style, init as colorama_init
from pyfiglet import figlet_format

# Initialize colorama
colorama_init(autoreset=True)

# Set up logger
logger = logging.getLogger("DrillGPT")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)

def print_app_banner():
    banner = figlet_format("DrillGPT", font="slant")
    logger.info(Fore.CYAN + banner + Style.RESET_ALL)
    logger.info(Fore.YELLOW + "AI-Driven Drilling Optimization with LLM Integration" + Style.RESET_ALL)


 