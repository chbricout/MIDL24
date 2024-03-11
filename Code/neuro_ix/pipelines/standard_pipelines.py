import logging
import os

from more_itertools import divide, nth_or_last
from neuro_ix.datasets.neuro_ix_dataset import MRIModality, NeuroiXDataset
from neuro_ix.pipelines.fastsurfer import FastSurferSeg
from neuro_ix.pipelines.flirt import Flirt, FlirtT2toT1
from neuro_ix.pipelines.fslmath import FslMathBrainExt
from neuro_ix.pipelines.mri_convert import MriConvert
from neuro_ix.pipelines.pipelines import ProcStatus, ProcessLogger
from neuro_ix.utils.datasets import to_bids_dataset_format


class MRIQCPipeline:
    def __init__(
        self,
        fsaverage_template_path: str,
        dataset: NeuroiXDataset,
        fast_surfer: str,
        job_id: int = 0,
        nb_jobs: int = 1,
        out_dir:str=None
    ):
        if out_dir!=None:
            self.output_path = f"{out_dir}"
        else:
            self.output_path = f"{dataset.root_dir}/derivatives/mriqc-pipeline"
        if not os.path.exists(self.output_path):
            logging.info("making output directory")
            os.makedirs(self.output_path)

        self.dataset = dataset
        self.batch = self.get_batch(job_id, nb_jobs)

        self.pipe_name = f"mriqc_{dataset.dataset_std_name}"
        self.log_path = os.path.join("/scratch/cbricout", self.pipe_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.logger = ProcessLogger(self.pipe_name, job_id, self.log_path, self.batch)

        self.fastsurfer_seg = FastSurferSeg(
            dataset.rawdata_dir,
            output_path=self.output_path,
            fast_surfer_container=fast_surfer,
            logger=self.logger
        )
        self.convert = MriConvert(logger=self.logger)
        self.extract = FslMathBrainExt(logger=self.logger)
        self.register = Flirt(fsaverage_template_path,logger=self.logger)

        self.logger.log_status([], ProcStatus.Started)

    def get_batch(self, job_id: int, nb_jobs: int):
        files = self.dataset.get_images_path(MRIModality.T1w)
        return list(nth_or_last(divide(nb_jobs, files), job_id))

    def __call__(self):
        logging.info(f"Batch contain {len(self.batch)} files.")
        done = []
        for item in self.batch:
            sub_id = to_bids_dataset_format(str(self.dataset.get_subject_id(item)), self.dataset.dataset_std_name)

            fullname = self.dataset.get_file_id(item)
            filename = os.path.basename(item)

            logging.info(f"\nNow processing {sub_id} ({filename})")

            self.fastsurfer_seg(item, sub_id, fullname)
            fast_dir = self.fastsurfer_seg.pipeline_dir

            orig_nu_nif, brain_mask_nif = self.convert(
                [f"{fast_dir}/orig_nu", f"{fast_dir}/mask"]
            )

            extract_file = self.extract(orig_nu_nif, brain_mask_nif)

            self.register(extract_file[0])

            os.remove(f"{fast_dir}/orig_nu.mgz")
            os.remove(f"{fast_dir}/mask.mgz")

            logging.info(f"Pipeline succeeded for {item} !\n")
            done.append(filename)
            self.logger.log_status(done, ProcStatus.Running)
        self.logger.log_status(done, ProcStatus.Success)

    @staticmethod
    def narval(dataset: NeuroiXDataset, job_id: int, tot_job: int, out_dir:str=None):
        root_path = "/home/cbricout/projects/def-sbouix"
        fast_surfer = f"{root_path}/software/FastSurfer"
        fsaverage_template_path = (
            f"{root_path}/cbricout/cinamon-cookie/scripts/brain.nii.gz"
        )
        return MRIQCPipeline(
            fsaverage_template_path, dataset, fast_surfer, job_id, tot_job, out_dir
        )


class MRIQCPipelineT2:
    def __init__(
        self,
        dataset: NeuroiXDataset,
        fast_surfer: str,
        job_id: int = 0,
        nb_jobs: int = 1,
        out_dir:str=None
    ):
        if out_dir!=None:
            self.output_path = f"{out_dir}"
        else:
            self.output_path = f"{dataset.root_dir}/derivatives/mriqc-pipeline"
        if not os.path.exists(self.output_path):
            logging.info("making output directory")
            os.makedirs(self.output_path)

        self.dataset = dataset
        self.batch = self.get_batch(job_id, nb_jobs)

        self.pipe_name = f"mriqc_{dataset.dataset_std_name}"
        self.log_path = os.path.join("/scratch/cbricout", self.pipe_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.logger = ProcessLogger(self.pipe_name, job_id, self.log_path, self.batch)

        self.fastsurfer_seg = FastSurferSeg(
            dataset.rawdata_dir,
            output_path=self.output_path,
            fast_surfer_container=fast_surfer,
            logger=self.logger
        )
        self.convert = MriConvert(logger=self.logger)
        self.extract = FslMathBrainExt(logger=self.logger)
        self.register = FlirtT2toT1(logger=self.logger)

        self.logger.log_status([], ProcStatus.Started)

    def get_batch(self, job_id: int, nb_jobs: int):
        files = self.dataset.get_images_path(MRIModality.T1w)
        return list(nth_or_last(divide(nb_jobs, files), job_id))

    def __call__(self):
        logging.info(f"Batch contain {len(self.batch)} files.")
        done = []
        for item in self.batch:
            sub_id = to_bids_dataset_format(str(self.dataset.get_subject_id(item)), self.dataset.dataset_std_name)

            fullname = self.dataset.get_file_id(item)
            filename = os.path.basename(item)

            logging.info(f"\nNow processing {sub_id} ({filename})")

            self.fastsurfer_seg(item, sub_id, fullname)
            fast_dir = self.fastsurfer_seg.pipeline_dir

            orig_nu_nif, brain_mask_nif = self.convert(
                [f"{fast_dir}/orig_nu", f"{fast_dir}/mask"]
            )

            extract_file = self.extract(orig_nu_nif, brain_mask_nif)

            self.register(extract_file[0], f"{fast_dir.replace('T2', 'T1')}/reg_extracted_orig_nu")

            os.remove(f"{fast_dir}/orig_nu.mgz")
            os.remove(f"{fast_dir}/mask.mgz")

            logging.info(f"Pipeline succeeded for {item} !\n")
            done.append(filename)
            self.logger.log_status(done, ProcStatus.Running)
        self.logger.log_status(done, ProcStatus.Success)

    @staticmethod
    def narval(dataset: NeuroiXDataset, job_id: int, tot_job: int, out_dir:str=None):
        root_path = "/home/cbricout/projects/def-sbouix"
        fast_surfer = f"{root_path}/software/FastSurfer"
       
        return MRIQCPipelineT2(
         dataset, fast_surfer, job_id, tot_job, out_dir
        )