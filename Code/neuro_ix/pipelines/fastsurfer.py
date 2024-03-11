import logging
import os
from typing import List
from neuro_ix.pipelines.pipelines import ProcStatus, ProcessLogger, ShellPipelineProcess


class FastSurferSeg(ShellPipelineProcess):
    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        fast_surfer_container: str,
        logger: ProcessLogger = None,
    ):
        super().__init__(logger)
        self.process_name = f"Fastsurfer (Brain segmentation and pre processing) "
        self.command_name = (
            "run_fastsurfer"  # we need to use apptainer / singularity here
        )
        self.pipeline_dir = None

        self.dataset_path = dataset_path
        self.output_path = output_path
        self.fast_surfer = fast_surfer_container

    def __call__(self, t1_file: str, sub_id: str, file_id: str) -> List[str] | int:
        path_on_singularity = f"/data{t1_file.removeprefix(self.dataset_path)}"
        code = self.call_shell(self.build_command(sub_id, file_id, path_on_singularity))

        if code != 0:
            return code
        logging.info(f"(Success) FastSurfer on {file_id} ({path_on_singularity}) ")
        self.pipeline_dir = f"{self.output_path}/{sub_id}/{file_id}/mri"
        return os.listdir(self.pipeline_dir)

    def build_command(
        self, sub_id: str, fullname: str, path_on_singularity: str
    ) -> str:
        return f"""singularity exec --nv --no-home \
            -B {self.dataset_path}:/data \
            -B {self.output_path}:/output \
            {self.fast_surfer}/fastsurfer-gpu.sif /fastsurfer/run_fastsurfer.sh \
            --vox_size 1 \
            --sd /output/{sub_id} \
            --sid {fullname} --t1 {path_on_singularity} --seg_only"""
