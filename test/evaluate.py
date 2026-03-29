import itertools
import numpy as np
from collections import Counter
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import smact
from smact.screening import pauling_test
from ase.neighborlist import neighbor_list
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from constants import CompScalerMeans, CompScalerStds
from data_utils import StandardScaler
from matbench_genmetrics.core.metrics import GenMetrics
from tqdm.auto import tqdm
from rich.progress import track
import os
import time
import threading
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timeout after {seconds} seconds")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

CompScaler = StandardScaler(means=np.array(CompScalerMeans), stds=np.array(CompScalerStds), replace_nan_token=0.)

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}

count_total = 0
count_gen = 0
count_tar = 0
def get_match_rms_one(struct_pred, struct_true, is_pred_valid, stol=0.5, angle_tol=10, ltol=0.3):
    """
    计算两个结构之间的均方根偏差（RMSD）。

    参数:
        struct_pred (Structure): 预测的结构。
        struct_true (Structure): 真实的结构。
        stol (float, optional): 结构匹配的距离容差。默认为0.5。
        angle_tol (float, optional): 结构匹配的角度容差。默认为10。
        ltol (float, optional): 结构匹配的晶格容差。默认为0.3。

    返回:
        float: 两个结构之间的RMSD。如果结构不匹配，则返回None。
    """
    
    if not is_pred_valid:
        return None
    global count_total
    count_total += 1
    # 创建一个StructureMatcher对象，用于比较两个结构
    try:
        matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol,allow_subset=True)
        #print("match")
        struct_pred=AseAtomsAdaptor.get_structure(struct_pred)
        global count_gen
        count_gen += 1
        struct_true=AseAtomsAdaptor.get_structure(struct_true)
        global count_tar
        count_tar += 1
        # 计算两个结构之间的RMSD
        rms_dist = matcher.get_rms_dist(struct_pred, struct_true)
        #print("rms_dist:",struct_pred,struct_true,rms_dist[0], count_gen, count_tar, count_total)
        #print('one_match', struct_true)
        # 如果RMSD为None，则返回None
        rms_dist = None if rms_dist is None else rms_dist[0]
        return rms_dist
    except Exception:
        #print("None")
        return None

def safe_get_match_rms_one(struct_pred, struct_true, stol, angle_tol, ltol, timeout=1):
    try:
        with time_limit(timeout):
            return get_match_rms_one(struct_pred, struct_true, stol, angle_tol, ltol)
    except TimeoutException:
        print(f"Timeout in RMSD calculation for structure pair")
        return None

def safe_AseAtomsAdaptor_get_structure(structure,timeout=1):
    try:
        with time_limit(timeout):
            return AseAtomsAdaptor.get_structure(structure)
    except TimeoutException:
        print(f"Timeout in RMSD calculation for structure pair")
        return None

def safe_get_chemical_formula(atoms,timeout=1):
    try:
        with time_limit(timeout):
            return atoms.get_chemical_formula()
    except TimeoutException:
        print(f"Timeout in RMSD calculation for structure pair")
        return None

def smact_validity(atoms, use_pauling_test=True, include_alloys=True):
    """
    检查给定原子结构的化学组成是否有效。

    参数:
        atoms (ase.Atoms): 要检查的原子结构。
        use_pauling_test (bool, optional): 是否使用鲍林电负性测试。默认为True。
        include_alloys (bool, optional): 是否包括合金。默认为True。

    返回:
        None: 结果存储在 `atoms.info` 字典中，键为 "comp_valid"。
    """
    # 获取原子结构的化学符号列表
    comp = atoms.get_chemical_symbols()
    # 统计每个元素的出现次数
    elem_counter = Counter(comp)
    # 生成元素及其计数的列表
    composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    # 将元素和计数分别提取到两个列表中
    elems, counts = list(zip(*composition))
    # 将计数转换为numpy数组
    counts = np.array(counts)
    # 将计数归一化，使其互质
    counts = counts / np.gcd.reduce(counts)
    # 将归一化后的计数转换为整数元组
    count = tuple(counts.astype('int').tolist())

    # 将元素符号转换为元组
    elem_symbols = tuple([elem for elem in comp])
    # 获取元素的SMACT空间
    #space = smact.element_dictionary(elem_symbols)
    
    # 将元素符号转换为元组
    elem_symbols = tuple([elem for elem in comp])

    # ====== 新增：smact 元素检查 ======
    try:
        space = smact.element_dictionary(elem_symbols)
    except NameError as e:
        atoms.info["comp_valid"] = False
        atoms.info["invalid_reason"] = str(e)
        return
    # =================================

    
    # 提取SMACT元素对象
    smact_elems = [e[1] for e in space.items()]
    # 获取元素的鲍林电负性
    electronegs = [e.pauling_eneg for e in smact_elems]
    # 获取元素的氧化态组合
    ox_combos = [e.oxidation_states for e in smact_elems]
    # 如果只有一种元素，标记为有效并返回
    if len(set(elem_symbols)) == 1:
        atoms.info["comp_valid"] = True
        return
    # 如果包括合金，检查是否所有元素都是金属
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            atoms.info["comp_valid"] = True
            return

    # 设置阈值为最大计数
    threshold = np.max(count)
    # 初始化有效组合列表
    compositions = []
    # 遍历所有可能的氧化态组合
    for ox_states in itertools.product(*ox_combos):
        # 生成化学计量比列表
        stoichs = [(c,) for c in count]
        # 测试电荷平衡
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # 电负性测试
        if cn_e:
            if use_pauling_test:
                try:
                    # 进行鲍林电负性测试
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # 如果没有电负性数据，假设测试通过
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                # 将有效组合添加到列表中
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    # 去除重复的组合
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    # 根据有效组合的数量设置有效性标记
    if len(compositions) > 0:
        atoms.info["comp_valid"] = True
    else:
        atoms.info["comp_valid"] = False

def get_fingerprints(atoms):
    """
    Calculate the fingerprint features of the given atomic structure.

    Args:
        atoms (ase.Atoms): The atomic structure to calculate fingerprint features for.

    Returns:
        None: The results are stored in the `atoms.info` dictionary with keys "comp_fp" and "struct_fp".
    """
    # Get the list of chemical symbols in the atomic structure
    comp_ase = atoms.get_chemical_symbols()
    # Count the occurrences of each element
    elem_counter = Counter(comp_ase)
    # Create a Composition object
    comp = Composition(elem_counter)
    # Calculate and store the composition fingerprint features
    atoms.info["comp_fp"] = CompFP.featurize(comp)
    try:
        # Convert the ASE atomic structure to a pymatgen structure
        structure = AseAtomsAdaptor.get_structure(atoms)
        # Calculate and store the site fingerprint features
        site_fps = [CrystalNNFP.featurize(structure, i) for i in range(len(structure))]
    except Exception:
        # If the fingerprint features cannot be constructed, mark the atomic structure as invalid
        atoms.info["valid"] = False
        atoms.info["comp_fp"] = None
        atoms.info["struct_fp"] = None
        return
    # Calculate and store the average site fingerprint feature
    atoms.info["struct_fp"] = np.array(site_fps).mean(axis=0)

def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps

def get_validity_one(atoms, cutoff=0.01):
    if atoms.info["constructed"]:
        i_indices = neighbor_list('i', atoms, cutoff=cutoff, max_nbins=100.0)
        atoms.info["struct_valid"] = (len(i_indices) == 0)
    else:
        atoms.info["struct_valid"] = False
    atoms.info["valid"] = atoms.info["comp_valid"] and atoms.info["struct_valid"]

def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()


class GenEval:

    def __init__(self, traj_pred, traj_true, traj_train, traj_test, traj_match_path, Eval):
        self.traj_pred = traj_pred
        self.traj_true = traj_true
        self.traj_train = traj_train
        self.traj_test = traj_test
        self.traj_match_path = traj_match_path#保存匹配成功的生成结构的traj文件
        self.Eval = Eval#控制评测内容
        self.n_samples = Eval["n_samples"]
        #self.n_targets = Eval["n_targets"]

        valid_ats = [atoms for atoms in traj_pred if atoms.info["valid"]]#为多样性做准备
        if self.n_samples is None:
            self.n_samples = int(0.8 * len(valid_ats))
        #print(type(self.n_samples))
        if len(valid_ats) >= self.n_samples:
            print("valid_ats:",len(valid_ats))
            sampled_indices = np.random.choice(len(valid_ats), self.n_samples, replace=False)
            self.valid_samples = [valid_ats[i] for i in sampled_indices]
            #随机抽取valid样本
        else:
            raise Exception(f'not enough valid crystals in the predicted set: {len(valid_ats)}/{self.n_samples}')

    def get_match_rms(self,):
        """
        计算预测结构与真实结构之间的均方根偏差（RMSD）。

        参数:
            stol (float, optional): 结构匹配的距离容差。默认为0.5。
            angle_tol (float, optional): 结构匹配的角度容差。默认为10。
            ltol (float, optional): 结构匹配的晶格容差。默认为0.3。

        返回:
            dict: 包含匹配率和平均RMSD的字典。
        """
        # 初始化一个列表来存储每个结构对的RMSD
        rms_dists = []

        # 从Eval字典中获取结构匹配的参数
        stol = self.Eval["match_para"]["stol"]
        angle_tol = self.Eval["match_para"]["angle_tol"]
        ltol = self.Eval["match_para"]["ltol"]
        n_attempt = self.Eval['match_para']['n_attempt']

        # 创建一个进度条来显示计算进度
        # 如果匹配的结构已经存在，则删除之前的文件
        if os.path.exists(self.traj_match_path):
            os.remove(self.traj_match_path)
        traj_match = Trajectory(self.traj_match_path, mode='a')
        #is_formula_change=False
        if n_attempt == 1:
            for i in track(range(len(self.traj_pred)), description='getting-match-rms'):#遍历所有帧
                pred_atoms = self.traj_pred[i]#去除结构对
                true_atoms = self.traj_true[i]
                # print(pred_atoms, true_atoms)
                # print(pred_atoms.get_positions(), true_atoms.get_positions())
                try:
                    is_pred_valid = pred_atoms.info['struct_valid']#结构合法性与化学式过滤
                    if pred_atoms.get_chemical_formula() != true_atoms.get_chemical_formula():
                        is_pred_valid = False#化学式不同直接判断为错误
                    rms_dist=get_match_rms_one(pred_atoms,true_atoms,is_pred_valid,stol, angle_tol, ltol)#计算RMSD
                except Exception:
                    pass
                if rms_dist != None:#成功匹配的判定
                    rms_dists.append(rms_dist)
                    traj_match.write(pred_atoms, append=True)
                    #print('Match!')
            n_match = len(traj_match)
            match_rate = len(traj_match)/len(self.traj_pred)
            traj_match.close()
            rms_dists = np.array(rms_dists)
            mean_rms_dist = rms_dists[rms_dists != None].mean()
        else:
            structures_firstId_list = [0]
            print("traj_true_len:",len(self.traj_true)-1)
            for i in range(len(self.traj_true)-1):
                if self.traj_true[i].get_chemical_formula() != self.traj_true[i+1].get_chemical_formula():
                    structures_firstId_list.append(i+1)
            print("Gen_Eval len",len(structures_firstId_list))
            # 遍历预测结构和真实结构的轨迹
            #print(len(structures_firstId_list))
            for i in track(range(len(structures_firstId_list)), description='getting-match-rms'):
                '''if i in range(10870,10890) or i in range(13678,13618):
                    progress_bar.update(1)
                    continue'''
                firstId = structures_firstId_list[i]
                if i != len(structures_firstId_list)-1:
                    endId = structures_firstId_list[i+1]
                else:
                    endId = len(self.traj_true)
                tmp_rms_dists = []
                
                for i in range(firstId,endId):
                    pred_atoms = self.traj_pred[i]
                    true_atoms = self.traj_true[i]
                    is_pred_valid = pred_atoms.info['struct_valid']
                    #print(pred_atoms, true_atoms)
                    try:
                        rms_dist = get_match_rms_one(pred_atoms,true_atoms,is_pred_valid,stol, angle_tol, ltol)
                        #print(rms_dist)
                        if rms_dist != None:
                            tmp_rms_dists.append(rms_dist)
                            traj_match.write(pred_atoms, append=True)
                    except Exception:
                        pass
                if len(tmp_rms_dists) == 0:
                    pass
                else:
                    rms_dists.append(np.min(tmp_rms_dists))
            
            traj_match.close()
            rms_dists = np.array(rms_dists)
            n_match =  np.count_nonzero(rms_dists != None)
            print("Gen_Eval match",sum(rms_dists != None),len(structures_firstId_list))
            match_rate = sum(rms_dists != None) / len(structures_firstId_list)
            mean_rms_dist = rms_dists[rms_dists != None].mean()
        # 返回包含匹配率和平均RMSD的字典
        return {"n_match": n_match,
                #'n_matched_formula': comps,
                #'matched_comp_dict': comps_dict,                
                "match_rate": match_rate, 
                "mean_rms_dist": mean_rms_dist,
                'rms_dists': rms_dists,
                }

    def compute_cov(self, struc_cutoff, comp_cutoff, num_gen_crystals=None):    
        """
        计算生成晶体的覆盖率（Coverage）和平均最小结构/成分距离（Average Minimum Structure/Composition Distance）。

        参数:
            struc_cutoff (float): 结构距离的截止值。
            comp_cutoff (float): 成分距离的截止值。
            num_gen_crystals (int, optional): 生成晶体的数量。默认为None，表示使用所有生成的晶体。

        返回:
            tuple: 包含覆盖率和平均最小结构/成分距离的字典，以及包含结构和成分距离的字典。
        """
        # 获取生成晶体的结构指纹和成分指纹
        struc_fps = [atoms.info["struct_fp"] for atoms in self.traj_pred]
        comp_fps = [atoms.info["comp_fp"] for atoms in self.traj_pred]
        # 获取真实晶体的结构指纹和成分指纹
        gt_struc_fps = [atoms.info["struct_fp"] for atoms in self.traj_true]
        gt_comp_fps = [atoms.info["comp_fp"] for atoms in self.traj_true]

        # 确保结构指纹和成分指纹的数量一致
        assert len(struc_fps) == len(comp_fps)
        assert len(gt_struc_fps) == len(gt_comp_fps)

        # 如果未指定生成晶体的数量，则使用所有生成的晶体
        if num_gen_crystals is None:
            num_gen_crystals = len(struc_fps)

        # 过滤结构指纹和成分指纹
        struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)
        gt_struc_fps, gt_comp_fps = filter_fps(gt_struc_fps, gt_comp_fps)

        # 对成分指纹进行标准化
        comp_fps = CompScaler.transform(comp_fps)
        gt_comp_fps = CompScaler.transform(gt_comp_fps)

        # 将结构指纹和成分指纹转换为numpy数组
        struc_fps = np.array(struc_fps)
        gt_struc_fps = np.array(gt_struc_fps)
        comp_fps = np.array(comp_fps)
        gt_comp_fps = np.array(gt_comp_fps)

        # 计算结构指纹和成分指纹之间的距离矩阵
        struc_pdist = cdist(struc_fps, gt_struc_fps)
        comp_pdist = cdist(comp_fps, gt_comp_fps)

        # 计算结构和成分的召回距离
        struc_recall_dist = struc_pdist.min(axis=0)
        struc_precision_dist = struc_pdist.min(axis=1)
        comp_recall_dist = comp_pdist.min(axis=0)
        comp_precision_dist = comp_pdist.min(axis=1)

        # 计算覆盖率
        cov_recall = np.mean(np.logical_and(struc_recall_dist <= struc_cutoff, comp_recall_dist <= comp_cutoff))
        cov_precision = np.sum(np.logical_and(struc_precision_dist <= struc_cutoff, comp_precision_dist <= comp_cutoff)) / num_gen_crystals

        # 创建包含覆盖率和平均最小结构/成分距离的字典
        metrics_dict = {
            'cov_recall': cov_recall,
            'cov_precision': cov_precision,
            'amsd_recall': np.mean(struc_recall_dist),
            'amsd_precision': np.mean(struc_precision_dist),
            'amcd_recall': np.mean(comp_recall_dist),
            'amcd_precision': np.mean(comp_precision_dist),
        }

        # 创建包含结构和成分距离的字典
        combined_dist_dict = {
            'struc_recall_dist': struc_recall_dist.tolist(),
            'struc_precision_dist': struc_precision_dist.tolist(),
            'comp_recall_dist': comp_recall_dist.tolist(),
            'comp_precision_dist': comp_precision_dist.tolist(),
        }

        return metrics_dict, combined_dist_dict
    
    def get_validity(self):
        comp_valid = np.array([atoms.info["comp_valid"] for atoms in self.traj_pred]).mean()
        struct_valid = np.array([atoms.info["struct_valid"] for atoms in self.traj_pred]).mean()
        valid = np.array([atoms.info["valid"] for atoms in self.traj_pred]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        comp_fps = [atoms.info["comp_fp"] for atoms in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([atoms.info["struct_fp"] for atoms in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [AseAtomsAdaptor.get_structure(atoms).density for atoms in self.valid_samples]
        gt_densities = [AseAtomsAdaptor.get_structure(atoms).density for atoms in self.traj_true]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(atoms.get_chemical_symbols())) for atoms in self.valid_samples]
        gt_nelems = [len(set(atoms.get_chemical_symbols())) for atoms in self.traj_true]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_coverage(self):
        #cutoff_dict = COV_Cutoffs[self.eval_model_name]
        cutoff_dict = self.Eval["cov_cutoffs"]
        (cov_metrics_dict, combined_dist_dict) = self.compute_cov(
            struc_cutoff=float(cutoff_dict['struc']),
            comp_cutoff=float(cutoff_dict['comp']))
        return cov_metrics_dict
    
    def get_novelty_unique(self):
        train_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in self.traj_train]
        test_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in self.traj_test]
        gen_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in self.traj_pred]
        gen_metrics = GenMetrics(train_structures=train_structures, test_structures=test_structures, gen_structures=gen_structures)

        return {'novelty': gen_metrics.novelty, 'uniqueness': gen_metrics.uniqueness}

    def get_metrics(self):
        """
        计算并返回评估指标的字典。

        根据 `self.Eval` 字典中的设置，有选择地计算并返回一系列评估指标。

        Returns:
            dict: 包含计算得到的评估指标的字典。
        """
        metrics = {}
        # 检查是否需要计算结构的有效性
        if self.Eval["calc_validity"]:
            metrics.update(self.get_validity())
        # 检查是否需要计算结构匹配的均方根误差（RMS）
        if self.Eval["calc_match_rms"]:
            metrics.update(self.get_match_rms())
        # 检查是否需要计算成分多样性
        if self.Eval["calc_comp_div"]:
            try:
                metrics.update(self.get_comp_diversity())
            except:
                metrics.update({'comp_div': None})
        # 检查是否需要计算结构多样性
        if self.Eval["calc_struct_div"]:
            metrics.update(self.get_struct_diversity())
        # 检查是否需要计算密度的 widest 距离
        if self.Eval["calc_wdist_density"]:
            metrics.update(self.get_density_wdist())
        # 检查是否需要计算元素数量的 widest 距离
        if self.Eval["calc_wdist_num_elems"]:
            metrics.update(self.get_num_elem_wdist())
        # 检查是否需要计算新颖性（独特性）
        if self.Eval["calc_novelty_unique"]:
            metrics.update(self.get_novelty_unique())
        # 检查是否需要计算覆盖率
        if self.Eval["calc_coverage"]:
            metrics.update(self.get_coverage())
        return metrics