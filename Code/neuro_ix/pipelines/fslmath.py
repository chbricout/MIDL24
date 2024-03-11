import logging
import os
from typing import List
from neuro_ix.pipelines.pipelines import ProcessLogger, ShellPipelineProcess


class FslMathBrainExt(ShellPipelineProcess):
    def __init__(self, out_prefix="extracted", logger: ProcessLogger = None):
        super().__init__(logger)
        self.process_name = f"Fslmath (Brain extraction) "
        self.command_name = "fslmaths"

        self.out_prefix = out_prefix

    def __call__(self, brain_volume: str, mask: str) -> List[str]:
        created_files = []
        pipeline_dir, basename = os.path.split(brain_volume)
        out = os.path.join(pipeline_dir, f"{self.out_prefix}_{basename}")
        code = self.call_shell(self.build_command(brain_volume, mask, out))
        if code != 0:
            return code
        logging.info(
            f"(Success) Extract brain from {basename} ({brain_volume}) with mask {os.path.basename(mask)} ({mask})"
        )
        created_files.append(out)
        return created_files

    def build_command(self, volume: str, mask: str, out: str) -> str:
        return f"{self.command_name} {volume} -mul {mask} {out}"
