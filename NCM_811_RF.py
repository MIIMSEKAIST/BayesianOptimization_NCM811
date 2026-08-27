# NCM_811_RF.py
# -*- coding: utf-8 -*-

import os
os.environ["MPLBACKEND"] = "Agg"

import argparse, warnings, sys
import config  # must be imported before viz/pyplot gets imported anywhere
from config import LOGGER
from protocol import ProtocolViolation
from models import mode_train, mode_iterate, mode_update, mode_visualize, mode_best, mode_loop

def main():
    warnings.filterwarnings("once")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","iterate","update","loop","best","visualize"], required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        LOGGER.debug("Verbose DEBUG mode on")

    try:
        {
            "train":     mode_train,
            "iterate":   mode_iterate,
            "update":    mode_update,
            "loop":      (lambda: mode_loop(args.iterations)),
            "best":      (lambda: mode_best(args.top)),
            "visualize": mode_visualize
        }[args.mode]()
    except ProtocolViolation:
        sys.exit(1)

if __name__ == "__main__":
    main()


#python NCM_811_RF.py --mode train -v
#python NCM_811_RF.py --mode visualize



#python NCM_811_RF.py --mode iterate
#python NCM_811_RF.py --mode best --top 3   


#python NCM_811_RF.py --mode update
#python NCM_811_RF.py --mode visualize
