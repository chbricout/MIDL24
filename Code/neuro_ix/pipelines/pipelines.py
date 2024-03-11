from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
import logging
import os
import subprocess
from typing import List

import pandas as pd


class ProcStatus(Enum):
    Started = "Started"
    Running = "Running"
    Success = "Success"
    Error = "Error"


class PipelineProcess(ABC):
    def __init__(self):
        self.process_name = ""

    @abstractmethod
    def __call__(self, **kwargs) -> List[str]:
        pass


class ShellPipelineProcess(PipelineProcess):
    def __init__(self, logger: ProcessLogger = None):
        super().__init__()
        self.command_name = ""
        self.logger = logger

    @abstractmethod
    def build_command(self) -> str:
        pass

    def call_shell(self, command: str) -> int:
        logging.debug(f"running {command}")
        res_cmd = subprocess.run(command, shell=True, capture_output=True, text=True)
        if res_cmd.returncode != 0:
            print(res_cmd.stderr)
            self.logger.log_error(res_cmd.stderr, self.process_name)
            logging.error(f"Pipeline fail on {self.process_name}")
        return res_cmd.returncode


class ProcessLogger:
    def __init__(
        self, job_name: str, idx: int, log_dir_path: str, files_to_do: List[str]
    ):
        self.job_name = job_name
        self.idx = idx
        self.log_dir_path = log_dir_path

        self.path = os.path.join(log_dir_path, f"{self.job_name}_{self.idx}_status.csv")
        self.error_path = os.path.join(
            log_dir_path, f"{self.job_name}_{self.idx}_error.csv"
        )
        self.success_path = os.path.join(
            log_dir_path, f"{self.job_name}_{self.idx}_success.csv"
        )

        self.files_to_do = files_to_do

    def log_status(self, file_done: List[str], status: ProcStatus):
        completion = len(file_done) / len(self.files_to_do)
        df = pd.DataFrame(
            data=[[self.idx, completion, status, file_done, self.files_to_do]],
            columns=["Id", "Completion", "Status", "File Done", "File To Do"],
        )
        if status == ProcStatus.Success:
            os.remove(self.path)
            df.to_csv(self.success_path)
        else:
            df.to_csv(self.path)



    def log_error(self, error_out: str, process_name: str):
        df = pd.DataFrame(
            data=[[self.idx, error_out, process_name]],
            columns=["Id", "Message", "During Process"],
        )
        df.to_csv(self.error_path)
