import os
import numpy as np
import torch
import trimesh
from typing import Optional, List, Dict, Any
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import lru_cache

import step1x3d_geometry
from step1x3d_geometry.datamodules.base import BaseDataModule
from step1x3d_geometry.utils.typing import *


@lru_cache(maxsize=None)
def _cached_mesh_load(path: str):
    """
    Load mesh vertices and faces with in-memory LRU cache and on-disk .npz cache.
    """
    cache_file = os.path.splitext(path)[0] + ".npz"
    # Try disk cache first
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data['vertices'], data['faces']
    # Load from original file and write disk cache
    mesh = trimesh.load(path, process=False)
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    try:
        np.savez_compressed(cache_file, vertices=vertices, faces=faces)
    except Exception:
        pass  # ignore cache write errors
    return vertices, faces


class MeshMotionDataset(Dataset):
    """动态网格序列数据集"""
    
    def __init__(
        self,
        root_dir: str,
        num_frames: int = 30,
        num_vertices: Optional[int] = None,
        random_rotate: bool = True,
        random_scale: bool = True,
        scale_range: List[float] = [0.8, 1.2],
        noise_sigma: float = 0.0,
        split: str = "train",
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.num_vertices = num_vertices
        self.random_rotate = random_rotate
        self.random_scale = random_scale
        self.scale_range = scale_range
        self.noise_sigma = noise_sigma
        self.split = split
        
        # 获取数据文件列表
        self.data_files = self._get_data_files()
        if self.num_vertices is not None:
            # Use provided vertex count, skip dynamic scanning
            self.max_num_vertices = self.num_vertices
        else:
            # Dynamically compute max vertices across sequences (only once)
            self.max_num_vertices = 0
            for sequence_dir in self.data_files:
                mesh_files = [f for f in os.listdir(sequence_dir)
                              if f.endswith('.obj') or f.endswith('.ply')]
                mesh_files.sort()
                first_frame = mesh_files[0]
                mesh_path = os.path.join(sequence_dir, first_frame)
                verts, _ = _cached_mesh_load(mesh_path)
                num_v = verts.shape[0]
                self.max_num_vertices = max(self.max_num_vertices, num_v)
        # Set maximum face count as 2.5x the maximum vertices
        self.max_num_faces = int(self.max_num_vertices * 2.5)
        
    def _get_data_files(self) -> List[str]:
        """获取数据文件列表（高效版）"""
        data_files = []
        # 仅遍历一级目录，提高速度
        for seq_name in sorted(os.listdir(self.root_dir)):
            sequence_dir = os.path.join(self.root_dir, seq_name)
            # 如果是不小心放在根目录的多余 npz，删除并跳过
            if seq_name.endswith('.npz'):
                try:
                    os.remove(sequence_dir)
                except OSError:
                    pass
                continue
            if not os.path.isdir(sequence_dir):
                continue
            frame_files = [f for f in os.listdir(sequence_dir)
                           if f.endswith('.obj') or f.endswith('.ply')]
            frame_files.sort()
            if len(frame_files) >= self.num_frames:
                data_files.append(sequence_dir)
        return data_files
    
    def __len__(self):
        return len(self.data_files)
    
    def _load_mesh_sequence(self, sequence_dir: str) -> tuple:
        """加载网格序列"""
        # Gather and sort frame file names
        frame_files = [f for f in os.listdir(sequence_dir)
                       if f.endswith('.obj') or f.endswith('.ply')]
        frame_files.sort()

        # Sliding window crop or time-interpolation padding to exactly num_frames
        L = len(frame_files)
        vertices_list = []
        faces = None
        if L >= self.num_frames:
            # random contiguous window of length num_frames
            start = np.random.randint(0, L - self.num_frames + 1)
            sel_files = frame_files[start : start + self.num_frames]
            for frame_file in sel_files:
                mesh_path = os.path.join(sequence_dir, frame_file)
                frame_vertices, mesh_faces = _cached_mesh_load(mesh_path)
                if faces is None:
                    faces = mesh_faces
                vertices_list.append(frame_vertices)
        else:
            # time interpolation to pad short sequence smoothly
            orig_verts = []
            for frame_file in frame_files:
                mesh_path = os.path.join(sequence_dir, frame_file)
                frame_vertices, mesh_faces = _cached_mesh_load(mesh_path)
                if faces is None:
                    faces = mesh_faces
                orig_verts.append(frame_vertices)
            orig_verts = np.stack(orig_verts, axis=0)  # [L, N, 3]
            times = np.linspace(0, L - 1, self.num_frames)
            for t in times:
                i0 = int(np.floor(t))
                i1 = int(np.ceil(t))
                w = t - i0
                v = (1 - w) * orig_verts[i0] + w * orig_verts[i1]
                vertices_list.append(v.astype(np.float32))

        vertices = np.stack(vertices_list, axis=0)  # [T, N, 3]
        return vertices, faces
    
    def _resample_vertices(self, vertices: np.ndarray, target_num: int) -> np.ndarray:
        """重采样顶点（简化实现）"""
        if len(vertices) == target_num:
            return vertices
        
        # 简单的线性插值重采样
        indices = np.linspace(0, len(vertices) - 1, target_num)
        indices_floor = np.floor(indices).astype(int)
        indices_ceil = np.minimum(indices_floor + 1, len(vertices) - 1)
        weights = indices - indices_floor
        
        resampled = (1 - weights[:, None]) * vertices[indices_floor] + weights[:, None] * vertices[indices_ceil]
        
        return resampled.astype(np.float32)
    
    def _apply_augmentation(self, vertices: np.ndarray) -> np.ndarray:
        """应用数据增强"""
        T, N, _ = vertices.shape
        
        # 随机旋转
        if self.random_rotate:
            # 生成随机旋转矩阵
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            vertices = vertices @ rotation_matrix.T
        
        # 随机缩放
        if self.random_scale:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            vertices = vertices * scale
        
        # 添加噪声
        if self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, vertices.shape)
            vertices = vertices + noise
        
        return vertices
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取数据样本"""
        sequence_dir = self.data_files[idx]
        
        # 加载网格序列
        vertices, faces = self._load_mesh_sequence(sequence_dir)
        # Normalize mesh to unit scale per-sequence bounding box
        # vertices: numpy array of shape [T, N, 3]
        vmin = vertices.min(axis=(0,1), keepdims=True)  # [1,1,3]
        vmax = vertices.max(axis=(0,1), keepdims=True)  # [1,1,3]
        center = (vmin + vmax) / 2.0
        vertices = vertices - center  # center at origin
        scale = np.max(vmax - vmin)
        vertices = vertices / (scale + 1e-8)
        
        # 应用数据增强（仅在训练时）
        # if self.split == "train":
        #     vertices = self._apply_augmentation(vertices)
        
        # Pad vertices to the dataset maximum with zero vectors
        T, N, _ = vertices.shape
        if N < self.max_num_vertices:
            pad_vertices = np.zeros((T, self.max_num_vertices - N, 3), dtype=np.float32)
            vertices = np.concatenate([vertices, pad_vertices], axis=1)
        # Pad faces to the defined maximum face count with invalid indices
        F = faces.shape[0]
        if F < self.max_num_faces:
            invalid_face = np.array([-1, -1, -1], dtype=np.int32)
            pad_faces = np.tile(invalid_face, (self.max_num_faces - F, 1))
            faces = np.concatenate([faces, pad_faces], axis=0)
        
        # 转换为张量
        vertices = torch.from_numpy(vertices).float()  # [T, N, 3]
        faces = torch.from_numpy(faces).long()  # [F, 3]
        
        # 计算邻居索引列表 (k=16)
        k_nb = 8
        # faces numpy array [F,3] with -1 padding filtered earlier; ensure valid
        face_idx = faces.numpy()
        valid_mask = np.all(face_idx >= 0, axis=1)
        face_idx = face_idx[valid_mask]
        adj = [[] for _ in range(self.max_num_vertices)]
        for tri in face_idx:
            a,b,c = tri
            adj[a].append(b); adj[a].append(c)
            adj[b].append(a); adj[b].append(c)
            adj[c].append(a); adj[c].append(b)
        neighbor_idx = np.zeros((self.max_num_vertices, k_nb), dtype=np.int64)
        for i, nbrs in enumerate(adj):
            if len(nbrs)==0:
                nbrs=[i]
            if len(nbrs)>=k_nb:
                sel=nbrs[:k_nb]
            else:
                sel=nbrs + [nbrs[0]]*(k_nb-len(nbrs))
            neighbor_idx[i]=np.array(sel,dtype=np.int64)
        neighbor_idx = torch.from_numpy(neighbor_idx)
        
        return {
            "vertices": vertices,
            "faces": faces,
            "neighbor_idx": neighbor_idx,
            "sequence_id": os.path.basename(sequence_dir),
        }


# ================================= NPZ Dataset ===================================


class MeshMotionNPZDataset(Dataset):
    """Load preprocessed npz files: each frame .npz has vertices [K,3] and neighbor_idx [K,3]"""

    def __init__(self, root_dir: str, num_frames: int = 32, animal_filter: Optional[str] = None, split: str = "train"):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.split = split  # 添加split参数区分训练/验证/测试
        all_dirs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
        if animal_filter is not None:
            self.seq_dirs = [os.path.join(root_dir, d) for d in all_dirs if d.startswith(animal_filter)]
        else:
            self.seq_dirs = [os.path.join(root_dir, d) for d in all_dirs]

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx):
        seq_dir = self.seq_dirs[idx]
        frames = sorted([f for f in os.listdir(seq_dir) if f.endswith('.npz')])
        if len(frames)==0:
            raise RuntimeError(f"no npz in {seq_dir}")
        # choose contiguous window or pad by repeat
        if len(frames)>=self.num_frames:
            if self.split == "train":
                # 训练时：随机选择起始帧
                max_start = len(frames) - self.num_frames
                start = np.random.randint(0, max_start + 1)
            else:
                # 验证/测试时：固定从0开始（保持一致性）
                start = 0
            sel = frames[start:start+self.num_frames]
        else:
            # 反向填充：如果少于num_frames，先取所有帧，然后反向填充
            # 例如：[1,2,3,...,16] -> [1,2,3,...,16,15,14,13,12] (需要20帧)
            remaining_needed = self.num_frames - len(frames)
            if remaining_needed > 0:
                # 创建反向序列，从倒数第二帧开始反向
                if len(frames) > 1:
                    # 反向填充，跳过最后一帧避免重复
                    reverse_frames = frames[-2::-1]  # 从倒数第二帧开始反向
                    # 如果需要的帧数超过反向序列长度，循环使用
                    reverse_padding = []
                    for i in range(remaining_needed):
                        reverse_padding.append(reverse_frames[i % len(reverse_frames)])
                    sel = frames + reverse_padding
                else:
                    # 如果只有1帧，只能重复
                    sel = frames * self.num_frames
            else:
                sel = frames

        verts_list=[]
        neighbor_idx=None
        for f in sel:
            data=np.load(os.path.join(seq_dir,f))
            verts_list.append(data['vertices'])
            if neighbor_idx is None:
                neighbor_idx = torch.from_numpy(data['neighbor_idx']).long()
        vertices = torch.from_numpy(np.stack(verts_list,0)).float()  # [T,K,3]
        # dummy faces
        return {"vertices":vertices, "neighbor_idx":neighbor_idx, "sequence_id":os.path.basename(seq_dir)}


class SuccessfulMeshMotionNPZDataset(Dataset):
    """
    Load only successful sequences with canonical frame normalization.
    Each sequence is normalized so that the first frame's bounding box becomes
    a unit cube centered at origin.
    """

    def __init__(self, root_dir: str, successful_sequences_file: str, num_frames: int = 20, animal_filter: Optional[str] = None, split: str = "train"):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.animal_filter = animal_filter
        self.split = split  # 添加split参数区分训练/验证/测试
        
        # Load successful sequences
        self.successful_sequences = self._load_successful_sequences(successful_sequences_file)
        print(f"Loaded {len(self.successful_sequences)} successful sequences")
        
        # Filter sequence directories
        self.seq_dirs = self._filter_sequence_dirs()
        print(f"Found {len(self.seq_dirs)} sequence directories")

    def _load_successful_sequences(self, successful_sequences_file: str) -> List[str]:
        """Load list of successful sequence names from file"""
        successful_sequences = []
        try:
            with open(successful_sequences_file, 'r') as f:
                for line in f:
                    seq_name = line.strip()
                    if seq_name and not seq_name.startswith('#'):  # Skip empty lines and comments
                        successful_sequences.append(seq_name)
        except FileNotFoundError:
            print(f"Warning: Successful sequences file not found: {successful_sequences_file}")
            print("Using all available sequences instead.")
            return []
        
        return successful_sequences

    def _filter_sequence_dirs(self) -> List[str]:
        """Filter sequence directories based on successful sequences list"""
        all_dirs = [d for d in sorted(os.listdir(self.root_dir)) 
                   if os.path.isdir(os.path.join(self.root_dir, d))]
        
        # If successful_sequences is empty, use all directories
        if not self.successful_sequences:
            filtered_dirs = all_dirs
        else:
            # Only keep successful sequences
            filtered_dirs = [d for d in all_dirs if d in self.successful_sequences]
        
        # Apply animal filter if specified
        if self.animal_filter is not None:
            filtered_dirs = [d for d in filtered_dirs if d.startswith(self.animal_filter)]
        
        return [os.path.join(self.root_dir, d) for d in filtered_dirs]

    def _normalize_sequence(self, vertices: np.ndarray) -> np.ndarray:
        """
        Normalize sequence based on first frame's bounding box.
        Args:
            vertices: [T, N, 3] - vertex sequence
        Returns:
            normalized_vertices: [T, N, 3] - normalized sequence
        """
        T, N, _ = vertices.shape
        
        # Compute first frame's bounding box
        first_frame = vertices[0]  # [N, 3]
        frame_min = np.min(first_frame, axis=0)  # [3,]
        frame_max = np.max(first_frame, axis=0)  # [3,]
        
        # Compute center and size
        center = (frame_min + frame_max) / 2.0  # [3,]
        size = frame_max - frame_min  # [3,]
        
        # Avoid division by zero
        size = np.maximum(size, 1e-8)
        
        # Normalize: center at origin, scale to unit cube
        # The largest dimension determines the scale factor to maintain aspect ratio
        max_size = np.max(size)
        scale_factor = 1.0 / max_size if max_size > 1e-8 else 1.0
        
        # Apply normalization to entire sequence
        normalized_vertices = (vertices - center[np.newaxis, np.newaxis, :]) * scale_factor
        
        return normalized_vertices

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx):
        seq_dir = self.seq_dirs[idx]
        frames = sorted([f for f in os.listdir(seq_dir) if f.endswith('.npz')])
        if len(frames) == 0:
            raise RuntimeError(f"no npz in {seq_dir}")
        
        # Choose contiguous window or pad by repeat
        if len(frames) >= self.num_frames:
            if self.split == "train":
                # 训练时：随机选择起始帧
                max_start = len(frames) - self.num_frames
                start = np.random.randint(0, max_start + 1)
            else:
                # 验证/测试时：固定从0开始（保持一致性）
                start = 0
            sel = frames[start:start+self.num_frames]
        else:
            # 反向填充：如果少于num_frames，先取所有帧，然后反向填充
            # 例如：[1,2,3,...,16] -> [1,2,3,...,16,15,14,13,12] (需要20帧)
            remaining_needed = self.num_frames - len(frames)
            if remaining_needed > 0:
                # 创建反向序列，从倒数第二帧开始反向
                if len(frames) > 1:
                    # 反向填充，跳过最后一帧避免重复
                    reverse_frames = frames[-2::-1]  # 从倒数第二帧开始反向
                    # 如果需要的帧数超过反向序列长度，循环使用
                    reverse_padding = []
                    for i in range(remaining_needed):
                        reverse_padding.append(reverse_frames[i % len(reverse_frames)])
                    sel = frames + reverse_padding
                else:
                    # 如果只有1帧，只能重复
                    sel = frames * self.num_frames
            else:
                sel = frames

        verts_list = []
        neighbor_idx = None
        for f in sel:
            data = np.load(os.path.join(seq_dir, f))
            verts_list.append(data['vertices'])
            if neighbor_idx is None:
                neighbor_idx = torch.from_numpy(data['neighbor_idx']).long()
        
        vertices = np.stack(verts_list, 0)  # [T, N, 3]
        
        # Apply canonical frame normalization
        normalized_vertices = self._normalize_sequence(vertices)
        
        # Convert to tensor
        vertices_tensor = torch.from_numpy(normalized_vertices).float()  # [T, N, 3]
        
        return {
            "vertices": vertices_tensor, 
            "neighbor_idx": neighbor_idx, 
            "sequence_id": os.path.basename(seq_dir)
        }


@step1x3d_geometry.register("MeshMotionNPZ-datamodule")
class MeshMotionNPZDataModule(BaseDataModule):
    def __init__(self, root_dir:str, batch_size:int=4, num_workers:int=4, num_frames:int=32, overfit_single: bool = False, animal_filter: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.root_dir=root_dir
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.num_frames=num_frames
        self.overfit_single = overfit_single
        self.animal_filter = animal_filter

    def setup(self, stage: Optional[str]=None):
        # 创建训练、验证、测试数据集（使用不同的split参数）
        train_dataset = MeshMotionNPZDataset(
            self.root_dir, 
            self.num_frames, 
            animal_filter=self.animal_filter,
            split="train"  # 训练时使用随机时间窗口
        )
        val_dataset = MeshMotionNPZDataset(
            self.root_dir, 
            self.num_frames, 
            animal_filter=self.animal_filter,
            split="val"  # 验证时固定从0开始
        )
        test_dataset = MeshMotionNPZDataset(
            self.root_dir, 
            self.num_frames, 
            animal_filter=self.animal_filter,
            split="test"  # 测试时固定从0开始
        )
        
        if self.overfit_single and len(train_dataset.seq_dirs) > 0:
            single_seq = [train_dataset.seq_dirs[0]]
            train_dataset.seq_dirs = single_seq
            val_dataset.seq_dirs = single_seq
            test_dataset.seq_dirs = single_seq
        
        # 使用不同split参数的数据集实例
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True)


class FullMeshMotionNPZDataset(Dataset):
    """
    Load all sequences with canonical frame normalization.
    Each sequence is normalized so that the first frame's bounding box becomes
    a unit cube centered at origin. (Same normalization as SuccessfulMeshMotionNPZDataset)
    """

    RESERVED_SPLIT_DIRS = {"val", "test"}
    _INDEX_VERSION = 1  # bump when scan logic changes

    def __init__(self, root_dir: str, num_frames: int = 20, animal_filter: Optional[str] = None,
                 split: str = "train", exclude_dirs: Optional[List[str]] = None,
                 blocklist_file: Optional[str] = None, index_cache_dir: Optional[str] = None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.animal_filter = animal_filter
        self.split = split
        self.exclude_dirs = set(exclude_dirs or [])
        self.blocklist = self._load_blocklist(blocklist_file)
        self._index_cache_dir = index_cache_dir

        self.seq_dirs = self._filter_sequence_dirs()
        print(f"FullMeshMotionNPZDataset: Found {len(self.seq_dirs)} total sequence directories")

    @staticmethod
    def _load_blocklist(path: Optional[str]) -> set:
        if not path or not os.path.exists(path):
            return set()
        with open(path) as f:
            names = {line.strip() for line in f if line.strip() and not line.startswith("#")}
        print(f"Loaded blocklist: {len(names)} sequences from {path}")
        return names

    def _index_cache_path(self) -> Optional[str]:
        """Return path for the on-disk index cache file, or None to disable."""
        cache_dir = self._index_cache_dir or os.environ.get("RIGVAE_INDEX_CACHE")
        if not cache_dir:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        import hashlib
        key = hashlib.md5(self.root_dir.encode()).hexdigest()[:12]
        return os.path.join(cache_dir, f"seq_index_v{self._INDEX_VERSION}_{key}.txt")

    def _filter_sequence_dirs(self) -> List[str]:
        """Discover sequence directories, supporting both flat and nested layouts.

        A sequence directory is identified by containing frame_*.npz files.
        Handles flat, one-level nested, and multi-level nested layouts.
        Skips reserved split directories (val, test) and any dirs in exclude_dirs.

        Results are cached to disk (when index_cache_dir is set) so that
        subsequent runs skip the expensive recursive S3 scan.
        """
        seq_paths = self._try_load_index_cache()
        if seq_paths is None:
            seq_paths = self._scan_directories()
            self._save_index_cache(seq_paths)

        if self.blocklist:
            before = len(seq_paths)
            seq_paths = [p for p in seq_paths
                         if os.path.basename(p) not in self.blocklist]
            print(f"Blocklist filtered: {before - len(seq_paths)} removed, {len(seq_paths)} remaining")

        if self.animal_filter is not None:
            before = len(seq_paths)
            seq_paths = [p for p in seq_paths
                         if os.path.basename(p).startswith(self.animal_filter)]
            print(f"Applied animal filter '{self.animal_filter}': {len(seq_paths)}/{before} sequences")
        else:
            print(f"No animal filter applied: found {len(seq_paths)} sequences")

        return sorted(seq_paths)

    def _scan_directories(self) -> List[str]:
        """Recursively scan root_dir for sequence directories."""
        import time
        t0 = time.time()
        skip = self.RESERVED_SPLIT_DIRS | self.exclude_dirs
        seq_paths = []

        def _is_sequence_dir(path: str) -> bool:
            try:
                return any(f.startswith("frame_") and f.endswith(".npz")
                           for f in os.listdir(path))
            except OSError:
                return False

        def _scan(path: str, depth: int = 0):
            if depth > 4:
                return
            if _is_sequence_dir(path):
                seq_paths.append(path)
                return
            try:
                entries = sorted(os.listdir(path))
            except OSError:
                return
            for entry in entries:
                if depth == 0 and entry in skip:
                    continue
                child = os.path.join(path, entry)
                if os.path.isdir(child):
                    _scan(child, depth + 1)

        _scan(self.root_dir)
        print(f"Directory scan: found {len(seq_paths)} sequences in {time.time()-t0:.1f}s")
        return seq_paths

    def _try_load_index_cache(self) -> Optional[List[str]]:
        cache_path = self._index_cache_path()
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path) as f:
                paths = [line.strip() for line in f if line.strip()]
            print(f"Loaded index cache: {len(paths)} sequences from {cache_path}")
            return paths
        except Exception as e:
            print(f"Warning: failed to load index cache: {e}")
            return None

    def _save_index_cache(self, seq_paths: List[str]):
        cache_path = self._index_cache_path()
        if not cache_path:
            return
        try:
            with open(cache_path, "w") as f:
                f.write("\n".join(seq_paths) + "\n")
            print(f"Saved index cache: {len(seq_paths)} sequences to {cache_path}")
        except Exception as e:
            print(f"Warning: failed to save index cache: {e}")

    def _normalize_sequence(self, vertices: np.ndarray) -> np.ndarray:
        """
        Normalize sequence based on first frame's bounding box.
        Args:
            vertices: [T, N, 3] - vertex sequence
        Returns:
            normalized_vertices: [T, N, 3] - normalized sequence
        """
        T, N, _ = vertices.shape
        
        # Compute first frame's bounding box
        first_frame = vertices[0]  # [N, 3]
        frame_min = np.min(first_frame, axis=0)  # [3,]
        frame_max = np.max(first_frame, axis=0)  # [3,]
        
        # Compute center and size
        center = (frame_min + frame_max) / 2.0  # [3,]
        size = frame_max - frame_min  # [3,]
        
        # Avoid division by zero
        size = np.maximum(size, 1e-8)
        
        # Normalize: center at origin, scale to unit cube
        # The largest dimension determines the scale factor to maintain aspect ratio
        max_size = np.max(size)
        scale_factor = 1.0 / max_size if max_size > 1e-8 else 1.0
        
        # Apply normalization to entire sequence
        normalized_vertices = (vertices - center[np.newaxis, np.newaxis, :]) * scale_factor
        
        return normalized_vertices

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx):
        seq_dir = self.seq_dirs[idx]
        frames = sorted([f for f in os.listdir(seq_dir) if f.endswith('.npz')])
        if len(frames) == 0:
            raise RuntimeError(f"no npz in {seq_dir}")
        
        # Choose contiguous window or pad by repeat
        if len(frames) >= self.num_frames:
            if self.split == "train":
                start = np.random.randint(0, len(frames) - self.num_frames + 1)
            else:
                start = 0
            sel = frames[start:start+self.num_frames]
        else:
            # 反向填充：如果少于num_frames，先取所有帧，然后反向填充
            # 例如：[1,2,3,...,16] -> [1,2,3,...,16,15,14,13,12] (需要20帧)
            remaining_needed = self.num_frames - len(frames)
            if remaining_needed > 0:
                # 创建反向序列，从倒数第二帧开始反向
                if len(frames) > 1:
                    # 反向填充，跳过最后一帧避免重复
                    reverse_frames = frames[-2::-1]  # 从倒数第二帧开始反向
                    # 如果需要的帧数超过反向序列长度，循环使用
                    reverse_padding = []
                    for i in range(remaining_needed):
                        reverse_padding.append(reverse_frames[i % len(reverse_frames)])
                    sel = frames + reverse_padding
                else:
                    # 如果只有1帧，只能重复
                    sel = frames * self.num_frames
            else:
                sel = frames

        verts_list = []
        neighbor_idx = None
        for f in sel:
            data = np.load(os.path.join(seq_dir, f))
            verts_list.append(data['vertices'])
            if neighbor_idx is None:
                neighbor_idx = torch.from_numpy(data['neighbor_idx']).long()
        
        vertices = np.stack(verts_list, 0)  # [T, N, 3]
        
        # Apply canonical frame normalization (same as SuccessfulMeshMotionNPZDataset)
        normalized_vertices = self._normalize_sequence(vertices)
        
        # Convert to tensor
        vertices_tensor = torch.from_numpy(normalized_vertices).float()  # [T, N, 3]
        
        return {
            "vertices": vertices_tensor, 
            "neighbor_idx": neighbor_idx, 
            "sequence_id": os.path.basename(seq_dir)
        }


@step1x3d_geometry.register("FullMeshMotionNPZ-datamodule")
class FullMeshMotionNPZDataModule(BaseDataModule):
    def __init__(self, root_dir: str, batch_size: int = 4, num_workers: int = 4, 
                 num_frames: int = 20, overfit_single: bool = False, animal_filter: Optional[str] = None, 
                 use_split_dirs: bool = True, blocklist_file: Optional[str] = None,
                 index_cache_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_frames = num_frames
        self.overfit_single = overfit_single
        self.animal_filter = animal_filter
        self.use_split_dirs = use_split_dirs
        self.blocklist_file = blocklist_file
        self.index_cache_dir = index_cache_dir

    def setup(self, stage: Optional[str] = None):
        if self.use_split_dirs:
            # Train uses root_dir directly (recursive scan finds all sequences).
            # Val / test use dedicated subdirectories under root_dir.
            train_dir = self.root_dir
            val_dir = os.path.join(self.root_dir, "val") 
            test_dir = os.path.join(self.root_dir, "test")
            
            if not os.path.exists(val_dir):
                print(f"Warning: Val directory not found: {val_dir}, using train data for validation")
                val_dir = train_dir
                
            if not os.path.exists(test_dir):
                print(f"Warning: Test directory not found: {test_dir}, using val data for testing") 
                test_dir = val_dir
                
            # Create separate datasets for each stage
            self.train_dataset = FullMeshMotionNPZDataset(
                train_dir, 
                self.num_frames, 
                animal_filter=self.animal_filter,
                split="train",
                blocklist_file=self.blocklist_file,
                index_cache_dir=self.index_cache_dir,
            )
            self.val_dataset = FullMeshMotionNPZDataset(
                val_dir, 
                self.num_frames, 
                animal_filter=self.animal_filter,
                split="val",
                blocklist_file=self.blocklist_file,
                index_cache_dir=self.index_cache_dir,
            )
            self.test_dataset = FullMeshMotionNPZDataset(
                test_dir, 
                self.num_frames, 
                animal_filter=self.animal_filter,
                split="test"
            )
            
            # Handle overfit_single option
            if self.overfit_single:
                if len(self.train_dataset.seq_dirs) > 0:
                    single_seq = [self.train_dataset.seq_dirs[0]]
                    self.train_dataset.seq_dirs = single_seq
                    self.val_dataset.seq_dirs = single_seq
                    self.test_dataset.seq_dirs = single_seq
                    print(f"Overfitting on single sequence: {os.path.basename(single_seq[0])}")
            
            print(f"FullMeshMotionNPZDataModule setup complete (split dirs mode):")
            print(f"  - Train sequences: {len(self.train_dataset)}")
            print(f"  - Val sequences: {len(self.val_dataset)}")
            print(f"  - Test sequences: {len(self.test_dataset)}")
            print(f"  - Frames per sequence: {self.num_frames}")
            print(f"  - Animal filter: {self.animal_filter}")
            print(f"  - Using canonical frame normalization")
            
        else:
            # Backward compatibility: Old logic using single directory
            # 注意：在这种模式下，train/val/test使用相同数据，但split参数不同
            train_dataset = FullMeshMotionNPZDataset(
                self.root_dir, 
                self.num_frames, 
                animal_filter=self.animal_filter,
                split="train"  # 训练时使用随机时间窗口
            )
            val_dataset = FullMeshMotionNPZDataset(
                self.root_dir, 
                self.num_frames, 
                animal_filter=self.animal_filter,
                split="val"  # 验证时固定从0开始
            )
            test_dataset = FullMeshMotionNPZDataset(
                self.root_dir, 
                self.num_frames, 
                animal_filter=self.animal_filter,
                split="test"  # 测试时固定从0开始
            )
            
            if self.overfit_single and len(train_dataset.seq_dirs) > 0:
                single_seq = [train_dataset.seq_dirs[0]]
                train_dataset.seq_dirs = single_seq
                val_dataset.seq_dirs = single_seq  
                test_dataset.seq_dirs = single_seq
                print(f"Overfitting on single sequence: {os.path.basename(single_seq[0])}")
            
            # 使用不同split参数的数据集实例
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
            
            print(f"FullMeshMotionNPZDataModule setup complete (single dir mode):")
            print(f"  - Total sequences: {len(train_dataset)}")
            print(f"  - Frames per sequence: {self.num_frames}")
            print(f"  - Animal filter: {self.animal_filter}")
            print(f"  - Using canonical frame normalization")
            print(f"  - Train: random time windows, Val/Test: fixed from frame 0")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,  # Lightning会自动替换为DistributedSampler
            num_workers=self.num_workers, 
            pin_memory=True,
            drop_last=True,  # 🔥 确保各rank步数一致
            persistent_workers=True,  # 保持worker进程活跃
            prefetch_factor=2,  # 控制预取数量
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )


@step1x3d_geometry.register("SuccessfulMeshMotionNPZ-datamodule")
class SuccessfulMeshMotionNPZDataModule(BaseDataModule):
    def __init__(self, root_dir: str, successful_sequences_file: str, batch_size: int = 4, num_workers: int = 4, 
                 num_frames: int = 20, overfit_single: bool = False, animal_filter: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.successful_sequences_file = successful_sequences_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_frames = num_frames
        self.overfit_single = overfit_single
        self.animal_filter = animal_filter

    def setup(self, stage: Optional[str] = None):
        # 创建训练、验证、测试数据集（使用不同的split参数）
        train_dataset = SuccessfulMeshMotionNPZDataset(
            self.root_dir, 
            self.successful_sequences_file, 
            self.num_frames, 
            animal_filter=self.animal_filter,
            split="train"  # 训练时使用随机时间窗口
        )
        val_dataset = SuccessfulMeshMotionNPZDataset(
            self.root_dir, 
            self.successful_sequences_file, 
            self.num_frames, 
            animal_filter=self.animal_filter,
            split="val"  # 验证时固定从0开始
        )
        test_dataset = SuccessfulMeshMotionNPZDataset(
            self.root_dir, 
            self.successful_sequences_file, 
            self.num_frames, 
            animal_filter=self.animal_filter,
            split="test"  # 测试时固定从0开始
        )
        if self.overfit_single and len(train_dataset.seq_dirs) > 0:
            single_seq = [train_dataset.seq_dirs[0]]
            train_dataset.seq_dirs = single_seq
            val_dataset.seq_dirs = single_seq
            test_dataset.seq_dirs = single_seq
            print(f"Overfitting on single sequence: {os.path.basename(single_seq[0])}")
        
        # 使用不同split参数的数据集实例
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        print(f"SuccessfulMeshMotionNPZDataModule setup complete:")
        print(f"  - Total sequences: {len(train_dataset)}")
        print(f"  - Frames per sequence: {self.num_frames}")
        print(f"  - Animal filter: {self.animal_filter}")
        print(f"  - Successful sequences file: {self.successful_sequences_file}")
        print(f"  - Train: random time windows, Val/Test: fixed from frame 0")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        ) 