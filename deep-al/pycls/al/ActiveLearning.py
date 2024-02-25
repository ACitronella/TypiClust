# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler
import pycls.utils.logging as lu
import numpy as np

logger = lu.get_logger(__name__)

class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, cfg):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj,cfg=cfg)
        self.cfg = cfg
        
    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None, **kwargs):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)

        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":

            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.uncertainty(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "entropy":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.entropy(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "margin":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.margin(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "coreset":
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            # if self.cfg.TRAIN.DATASET == "IMAGENET":
            #     clf_model.cuda(0)
            clf_model.eval()
            coreSetSampler = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj)
            activeSet, uSet = coreSetSampler.query(lSet=lSet, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "typiclust_add_pe":
            from .typiclust_add_pe import TypiClustAddPE
            dataset_info = kwargs.get("dataset_info", None)
            assert dataset_info is not None 
            tpc = TypiClustAddPE(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, dataset_info=dataset_info)
            activeSet, uSet = tpc.select_samples()
            return activeSet, uSet
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "typiclust_cat_pe":
            from .typiclust_cat_pe import TypiClustCatPE
            dataset_info = kwargs.get("dataset_info", None)
            assert dataset_info is not None 
            tpc = TypiClustCatPE(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, dataset_info=dataset_info)
            activeSet, uSet = tpc.select_samples()
            return activeSet, uSet
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "typiclust_idx_pe":
            from .typiclust_idx_pe import TypiClustIdxPE
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            dataset_info = kwargs.get("dataset_info", None)
            assert dataset_info is not None
            tpc = TypiClustIdxPE(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, dataset_info=dataset_info)
            activeSet, uSet = tpc.select_samples()
            return activeSet, uSet
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "typiclust_patient_wise":
            from .typiclust_patient_wise import TypiClustPatientWise
            is_scan = False # is scan is not allow
            dataset_info = kwargs.get("dataset_info", None)
            activeSet = []
            newuSet = []
            indices_table = (dataset_info["frames"].cumsum() - dataset_info["frames"]).values # collect start index of each eye
            indices_table = np.concatenate([indices_table, [ int(dataset_info["frames"].sum())]]) # for last file
            for start_idx, stop_idx in zip(indices_table, indices_table[1:]):
                thispatient_lset = lSet[(lSet >= start_idx) & (lSet < stop_idx)]
                thispatient_uset = uSet[(uSet >= start_idx) & (uSet < stop_idx)]
                tpc = TypiClustPatientWise(self.cfg, thispatient_lset, thispatient_uset, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, start_idx=start_idx, is_scan=is_scan, dataset_info=dataset_info)
                thispatient_activeSet, thispatient_uset = tpc.select_samples()
                activeSet.append(thispatient_activeSet)
                newuSet.append(thispatient_uset)
            return np.concatenate(activeSet), np.concatenate(newuSet)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("typiclust"):
            from .typiclust import TypiClust
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            dataset_info = kwargs.get("dataset_info", None) 
            assert dataset_info is not None
            tpc = TypiClust(self.cfg, lSet, uSet, self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, self.cfg.ACTIVE_LEARNING.EMBEDDING_PATH, is_scan=is_scan, dataset_info=dataset_info)
            activeSet, uSet = tpc.select_samples()
            return activeSet, uSet, tpc.clusters

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == 'probcover_cat_pe':
            from .prob_cover_cat_pe import ProbCoverCatPE
            dataset_info = kwargs["dataset_info"]
            probcov = ProbCoverCatPE(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.DELTA, dataset_info=dataset_info)
            activeSet, uSet = probcov.select_samples()
 
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == 'probcover_idx_pe':
            from .prob_cover_idx_pe import ProbCoverIdxPE
            dataset_info = kwargs["dataset_info"]
            probcov = ProbCoverIdxPE(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.DELTA, dataset_info=dataset_info)
            activeSet, uSet = probcov.select_samples()
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == 'probcover_idx_dist':
            from .prob_cover_idx_dist import ProbCoverIdxDist
            dataset_info = kwargs["dataset_info"]
            frame_diff_factor = self.cfg.ACTIVE_LEARNING.FRAME_DIFF_FACTOR
            probcov = ProbCoverIdxDist(self.cfg, lSet, uSet, self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.DELTA, embedding_path=self.cfg.ACTIVE_LEARNING.EMBEDDING_PATH, dataset_info=dataset_info, frame_diff_factor=frame_diff_factor)
            activeSet, uSet = probcov.select_samples()
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == 'probcover_idx_standard_dist':
            from .prob_cover_idx_standarddist import ProbCoverIdxStandardDist
            dataset_info = kwargs["dataset_info"]
            frame_diff_factor = self.cfg.ACTIVE_LEARNING.FRAME_DIFF_FACTOR
            probcov = ProbCoverIdxStandardDist(self.cfg, lSet, uSet, self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.DELTA, embedding_path=self.cfg.ACTIVE_LEARNING.EMBEDDING_PATH, dataset_info=dataset_info, frame_diff_factor=frame_diff_factor)
            activeSet, uSet = probcov.select_samples()
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == 'probcover_idx_standard_dist_div2':
            from .prob_cover_idx_standarddist_div2 import ProbCoverIdxStandardDistDiv2
            dataset_info = kwargs["dataset_info"]
            frame_diff_factor = self.cfg.ACTIVE_LEARNING.FRAME_DIFF_FACTOR
            probcov = ProbCoverIdxStandardDistDiv2(self.cfg, lSet, uSet, self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.DELTA, embedding_path=self.cfg.ACTIVE_LEARNING.EMBEDDING_PATH, dataset_info=dataset_info, frame_diff_factor=frame_diff_factor)
            activeSet, uSet = probcov.select_samples()
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
            from .prob_cover import ProbCover
            probcov = ProbCover(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.DELTA, embedding_path=self.cfg.ACTIVE_LEARNING.EMBEDDING_PATH)
            activeSet, uSet = probcov.select_samples()
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random_on_blinking_period":
            assert self.cfg.DATASET.IS_BLINKING, "is_blinking needed to be true. to restrict the training set to have only blinking part"
            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
        # elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random_on_blinking_period_with_proportion":
        #     self.cfg.ACTIVE_LEARNING.BLINK_FRAME_TO_ALL_RATIO
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "embedding_difference_as_probability_density":
            # assert
            from .embedding_difference_as_probability_density import EmbeddingDifferenceAsProbabilityDensity
            dataset_info = kwargs["dataset_info"]
            al = EmbeddingDifferenceAsProbabilityDensity(self.cfg, lSet, uSet, self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                                                         self.cfg.ACTIVE_LEARNING.EMBEDDING_PATH, dataset_info, kernel_size=11)
            activeSet, uSet = al.select_samples()
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "embedding_difference_as_probability_density_reduce_high_frame_prob":
            # assert
            from .embedding_difference_as_probability_density_reduce_high_frame_prob import EmbeddingDifferenceAsProbabilityDensityReduceHighFrameProb
            dataset_info = kwargs["dataset_info"]
            al = EmbeddingDifferenceAsProbabilityDensityReduceHighFrameProb(self.cfg, lSet, uSet, self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                                                         self.cfg.ACTIVE_LEARNING.EMBEDDING_PATH, dataset_info, kernel_size=11)
            activeSet, uSet = al.select_samples()
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "embedding_difference_as_probability_density_with_softmax":
            from .embedding_difference_as_probability_density_with_softmax import EmbeddingDifferenceAsProbabilityDensityWithSoftmax
            dataset_info = kwargs["dataset_info"]
            temperature = self.cfg.ACTIVE_LEARNING.SOFTMAX_TEMPERATURE
            al = EmbeddingDifferenceAsProbabilityDensityWithSoftmax(self.cfg, lSet, uSet, self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                                                         self.cfg.ACTIVE_LEARNING.EMBEDDING_PATH, dataset_info, kernel_size=11, temperature=temperature)
            activeSet, uSet = al.select_samples()
               
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "dbal" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "DBAL":
            activeSet, uSet = self.sampler.dbal(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, \
                uSet=uSet, clf_model=clf_model,dataset=trainDataset)
            
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "bald" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "BALD":
            activeSet, uSet = self.sampler.bald(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_model=clf_model, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_var_R":
            activeSet, uSet = self.sampler.ensemble_var_R(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_models=supportingModels, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "vaal":
            adv_sampler = AdversarySampler(cfg=self.cfg, dataObj=self.dataObj)

            # Train VAE and discriminator first
            vae, disc, uSet_loader = adv_sampler.vaal_perform_training(lSet=lSet, uSet=uSet, dataset=trainDataset)

            # Do active sampling
            activeSet, uSet = adv_sampler.sample_for_labeling(vae=vae, discriminator=disc, \
                                unlabeled_dataloader=uSet_loader, uSet=uSet)
        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
        
