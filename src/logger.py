import logging
import csv
from torch.utils.tensorboard import SummaryWriter
import io
import os
gdrive = os.getenv('GDRIVE', None)

class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)

    def format(self, record):
        self.writer.writerow(record.msg.split(','))
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()

class Writer:
    def __init__(self, tensorboard_logdir, csv_logfile = None):
        self.csv_logfile = csv_logfile
        if self.csv_logfile:
            logging.basicConfig(filename=csv_logfile, level=logging.DEBUG)
            self.csv_logger = logging.getLogger(__name__)
            logging.root.handlers[0].setFormatter(CsvFormatter())
        self.tensorboard_writer = SummaryWriter(tensorboard_logdir)
        self.avg_epoch_metrics = {}
        self.avg_epoch_metrics_count = {}
        
    def add_scalar(self, tag, value, step, write_now = True, avg_per_epoch = False):
        if write_now:
            self.tensorboard_writer.add_scalar(tag, value, step)
        if self.csv_logfile:
            self.csv_logger.info(','.join([tag, str(value), str(step)]))
        if avg_per_epoch:
            if tag not in self.avg_epoch_metrics:
                self.avg_epoch_metrics[tag] = value
                self.avg_epoch_metrics_count[tag] = 1
            else:
                self.avg_epoch_metrics[tag] += value
                self.avg_epoch_metrics_count[tag] += 1
                
    def write_avg_epoch_metrics(self, step):
        for tag, value in self.avg_epoch_metrics.items():
            self.avg_epoch_metrics[tag] /= self.avg_epoch_metrics_count[tag]
            self.add_scalar('EpochAverage/' + tag, self.avg_epoch_metrics[tag], step)
        self.avg_epoch_metrics = {}
        self.avg_epoch_metrics_count = {}

    def flush(self):
        self.tensorboard_writer.flush()
        if gdrive and self.csv_logfile:
            handlers = self.csv_logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.csv_logger.removeHandler(handler)
            logging.basicConfig(filename=self.csv_logfile, level=logging.DEBUG)
            self.csv_logger = logging.getLogger(__name__)
            logging.root.handlers[0].setFormatter(CsvFormatter())
