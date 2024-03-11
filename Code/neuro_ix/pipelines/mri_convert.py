import logging
import os
from typing import List
from neuro_ix.pipelines.pipelines import ProcessLogger, ShellPipelineProcess


class MriConvert(ShellPipelineProcess):
    def __init__(
        self,
        in_type: str = "mgz",
        out_type: str = "nii.gz",
        logger: ProcessLogger = None,
    ):
        super().__init__(logger)
        self.process_name = f"MRI Convert ({in_type} -> {out_type}) "
        self.command_name = "mri_convert"

        self.in_type = in_type
        self.out_type = out_type

    def __call__(self, files: List[str]):
        created_files = []
        for file in files:
            pipeline_dir = os.path.dirname(file)
            file_path = os.path.join(pipeline_dir, file)
            code = self.call_shell(self.build_command(file_path))
            if code != 0:
                return code
            logging.info(f"(Success) Converted {os.path.basename(file)} ({file})")
            created_files.append(f"{file}.{self.out_type}")
        return created_files

    def build_command(self, file: str) -> str:
        return f"{self.command_name} {file}.{self.in_type} {file}.{self.out_type}"
