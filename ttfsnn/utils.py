import os
import logging
    
    
class EarlyStopper:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.max_acc = 0

    def early_stop(self, acc):
        if acc > self.max_acc:
            self.max_acc = acc
            self.counter = 0
        elif acc <= self.max_acc:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Logger:
    def __init__(self, output_path=None, log_to_file=False):
        if log_to_file: assert output_path is not None
        self.log_to_file = log_to_file
        self.log_path = None if output_path is None else output_path + ('/' if output_path[-1] != '/' else '')
        if output_path is not None:
            # Compute run version
            version = 0
            while True:
                ok=True
                for file,type in [("log","txt"), ("config","json"), ("weights","npy")]: # TODO: Make it more robust
                    filename = os.path.join(output_path, f"run_{file}{'' if version == 0 else f'_{version}'}.{type}")
                    if os.path.exists(filename): ok=False
                if ok: break
                else: version += 1
            self.exp_version = '' if version == 0 else f'_{version}'
            if log_to_file:
                self.create_log_dir()
                # Name of the log file
                filename = self.log_path + f'run_log{self.exp_version}.txt'
                # Init logger
                logging.basicConfig(filename=filename, level=logging.INFO, format='')

    def create_log_dir(self):
        # Create log directory if it does not exist
        if self.log_path is not None and not os.path.exists(self.log_path): os.makedirs(self.log_path)

    def log(self, msg):
        print(msg) # Print to console
        if self.log_to_file: # Print to file
            logging.info(msg)

    def stop(self):
        if self.log_to_file:
            logger = logging.getLogger()
            logger.handlers[0].stream.close()
            logger.removeHandler(logger.handlers[0])