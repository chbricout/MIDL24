import logging
import os
from typing import List

from neuro_ix.pipelines.pipelines import ProcStatus, ProcessLogger, ShellPipelineProcess


class Flirt(ShellPipelineProcess):
    def __init__(
        self,
        ref_path: str,
        out_prefix="reg",
        reg_matrix_name="registration_matrix.mat",
        logger: ProcessLogger = None,
    ):
        super().__init__(logger)
        self.process_name = f"Flirt (Brain registration) "
        self.command_name = "flirt"

        self.ref_path = ref_path
        self.out_prefix = out_prefix
        self.reg_matrix_name = reg_matrix_name

    def __call__(self, volume_to_reg: str) -> List[str]:
        pipeline_dir, basename = os.path.split(volume_to_reg)
        registered = os.path.join(pipeline_dir, f"{self.out_prefix}_{basename}")
        reg_matrix = os.path.join(pipeline_dir, self.reg_matrix_name)

        code = self.call_shell(
            self.build_command(volume_to_reg, registered, reg_matrix)
        )

        if code != 0:
            return code
        logging.info(
            f"(Success) Flirt registered {os.path.basename(volume_to_reg)} ({volume_to_reg}) "
        )
        return [registered, reg_matrix]

    def build_command(self, extracted: str, registered: str, reg_matrix: str) -> str:
        return f"flirt -in {extracted} -ref {self.ref_path} -out {registered} -omat {reg_matrix}"



class FlirtT2toT1(Flirt):
    def __init__(
        self,
        out_prefix="reg",
        reg_matrix_name="registration_matrix.mat",
        logger: ProcessLogger = None,
    ):
        

        super().__init__(None, out_prefix, reg_matrix_name, logger)
        
    def __call__(self, volume_to_reg: str, T1_volume:str) -> List[str]:
        pipeline_dir, basename = os.path.split(volume_to_reg)
        registered = os.path.join(pipeline_dir, f"{self.out_prefix}_{basename}")
        reg_matrix = os.path.join(pipeline_dir, self.reg_matrix_name)

        code = self.call_shell(
            self.build_command(volume_to_reg, T1_volume, registered, reg_matrix)
        )

        if code != 0:
            return code
        logging.info(
            f"(Success) Flirt registered {os.path.basename(volume_to_reg)} ({volume_to_reg}) "
        )
        return [registered, reg_matrix]

    def build_command(self, extracted: str, T1_ref, registered: str, reg_matrix: str) -> str:
        return f"flirt -in {extracted} -ref {T1_ref} -dof 6 -cost mutualinfo -omat {reg_matrix} -out {registered}"