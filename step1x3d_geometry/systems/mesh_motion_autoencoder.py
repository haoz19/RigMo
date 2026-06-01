from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat, rearrange

import step1x3d_geometry
from step1x3d_geometry.systems.base import BaseSystem
from step1x3d_geometry.utils.typing import *
from step1x3d_geometry.utils.misc import get_rank


@step1x3d_geometry.register("mesh-motion-autoencoder-system")
class MeshMotionAutoEncoderSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)

        sample_posterior: bool = True

        # 损失权重
        lambda_reconstruction: float = 1.0
        lambda_motion_smooth: float = 0.1
        lambda_kl: float = 0.001

        # 梯度裁剪配置
        gradient_clip_val: float = 1.0  # 梯度裁剪阈值
        gradient_clip_algorithm: str = "norm"  # "norm" or "value"

        # 优化器配置
        local_lr_multiplier: float = 4.0  # local transformation组件学习率倍数

        # 可视化配置：每多少个 epoch 保存一次第一批次的 GT / 重建网格
        visualize_every_n_epochs: int = 20
        enable_initial_visualization: bool = False
        freeze_skinning_epochs: int = 0  # 新增：冻结蒙皮权重的 Epoch 数量

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = step1x3d_geometry.find(self.cfg.shape_model_type)(
            self.cfg.shape_model
        )
        # Split-aware testing controls
        self._eval_split = "test"
        self._saved_cases_by_split = {}
        self._save_cap_per_split = 20

    def set_eval_split(self, split: str):
        """Set current evaluation split for logging and artifact saving."""
        self._eval_split = split

    def configure_optimizers(self):
        """配置优化器，为local transformation组件设置更高学习率"""
        import torch
        from step1x3d_geometry.utils.scheduler import parse_scheduler
        
        # 获取基础学习率和优化器配置
        base_lr = self.cfg.optimizer.args.lr
        local_lr_multiplier = self.cfg.local_lr_multiplier  # local transformation组件学习率倍数
        
        # 识别需要加速学习的local transformation相关参数
        local_decoder_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if any(component in name for component in [
                'shape_model.static_decoder',
                'shape_model.dynamic_vae_encoder', 
                'shape_model.dynamic_vae_decoder'
            ]):
                local_decoder_params.append(param)
                if self.global_rank == 0:  # 只在主进程打印
                    print(f"Local decoder param: {name} -> lr={base_lr * local_lr_multiplier}")
            else:
                other_params.append(param)
        
        # 创建参数组：local decoder使用更高学习率
        param_groups = [
            {
                'params': local_decoder_params,
                'lr': base_lr * local_lr_multiplier,
                'name': 'local_decoders'
            },
            {
                'params': other_params, 
                'lr': base_lr,
                'name': 'other_params'
            }
        ]
        
        # 构建优化器
        optimizer_class = getattr(torch.optim, self.cfg.optimizer.name)
        optimizer_args = dict(self.cfg.optimizer.args)
        optimizer_args.pop('lr')  # 移除全局lr，使用参数组中的lr
        
        optimizer = optimizer_class(param_groups, **optimizer_args)
        
        if self.global_rank == 0:
            print(f"✅ Configured optimizer with local_lr_multiplier={local_lr_multiplier}")
            print(f"   - Local decoders lr: {base_lr * local_lr_multiplier}")
            print(f"   - Other params lr: {base_lr}")
            print(f"   - Local decoder param count: {len(local_decoder_params)}")
            print(f"   - Other param count: {len(other_params)}")
        
        ret = {"optimizer": optimizer}
        
        # 添加学习率调度器
        if self.cfg.scheduler is not None:
            ret["lr_scheduler"] = parse_scheduler(self.cfg.scheduler, optimizer)
        
        return ret

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        前向传播
        Args:
            batch: 包含以下键的字典：
                - vertices: [B, T, N, 3] - 动态网格序列
                - faces: [B, F, 3] - 面信息
        """
        vertices = batch["vertices"]  # [B, T, N, 3]
        neighbor_idx = batch.get("neighbor_idx", None)
        if neighbor_idx is None:
            raise ValueError("neighbor_idx is required")

        # 前向传播
        output = self.shape_model(
            vertices=vertices,
            neighbor_idx=neighbor_idx,
            sample_posterior=self.cfg.sample_posterior,
        )

        return output

    def training_step(self, batch, batch_idx):
        """
        训练步骤
        """
        # 🔥 简化的分布式训练监控（仅日志，无barrier）
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            
            # 每100个batch记录进度（纯日志，无同步操作）
            if batch_idx % 100 == 0 and rank == 0:
                print(f"📊 Batch {batch_idx}: Training progress on all ranks")
        
        # 添加调试输出，帮助定位卡死位置
        '''
        if batch_idx == 0 and self.global_rank == 0:
            print(f"--- Training step {batch_idx} started ---")
            print(f"Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}")
            if isinstance(batch, dict) and 'vertices' in batch:
                print(f"Vertices shape: {batch['vertices'].shape}")
        '''
        output = self(batch)
        '''
        if batch_idx == 0 and self.global_rank == 0:
            print(f"--- Forward pass completed ---")
        
        # 计算损失
        # ensure loss computed in full precision (float32)
        if batch_idx == 0 and self.global_rank == 0:
            print(f"--- Starting loss computation ---")
        ''' 
        from torch.cuda.amp import autocast
        with autocast(enabled=False):
            losses = self.shape_model.compute_losses(output, batch["vertices"])
        
        # 记录损失（确保张量已从计算图分离，并指定batch_size避免分布式聚合歧义）
        batch_size = batch["vertices"].shape[0] if isinstance(batch, dict) and "vertices" in batch else None
        for name, value in losses.items():
            detached = value.detach() if isinstance(value, torch.Tensor) else value
            self.log(
                f"train/{name}",
                detached,
                on_step=True,
                on_epoch=False,
                prog_bar=(name == "total_loss"),
                logger=True,
                batch_size=batch_size,
            )
        
        # 记录损失权重（使用相同的batch_size以便分布式聚合）
        self.log(f"train_params/lambda_reconstruction", float(self.cfg.lambda_reconstruction), batch_size=batch_size)
        self.log(f"train_params/lambda_motion_smooth", float(self.cfg.lambda_motion_smooth), batch_size=batch_size)
        self.log(f"train_params/lambda_kl", float(self.cfg.lambda_kl), batch_size=batch_size)
        
        # ============= 🔥 分布式安全的NaN/Inf检测与处理 =============
        total_loss = losses["total_loss"]
        
        # Step 1: 检测当前rank是否有NaN/Inf
        local_has_nan = torch.isnan(total_loss) or torch.isinf(total_loss)
        
        # Step 2: 在所有rank间同步NaN状态（关键：确保所有rank执行相同的代码路径）
        if torch.distributed.is_initialized():
            # 创建一个tensor用于通信
            nan_flag = torch.tensor(
                float(local_has_nan),  # True->1.0, False->0.0
                device=total_loss.device,
                dtype=torch.float32
            )
            
            # 使用MAX操作：只要有一个rank是1.0，结果就是1.0
            torch.distributed.all_reduce(
                nan_flag, 
                op=torch.distributed.ReduceOp.MAX
            )
            
            global_has_nan = (nan_flag.item() > 0.5)
        else:
            global_has_nan = local_has_nan
        
        # Step 3: 根据全局结果，所有rank执行相同的操作
        if global_has_nan:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            
            # 详细日志（仅当前rank有问题时才打印详细信息）
            if local_has_nan:
                print(f"❌ Rank {rank}: Detected NaN/Inf loss at batch {batch_idx}")
                print(f"   Loss value: {total_loss.item() if not torch.isnan(total_loss) else 'NaN'}")
                print(f"   All losses: {[(k, v.item() if isinstance(v, torch.Tensor) else v) for k, v in losses.items()]}")
                
                # 保存问题数据用于调试（只保存前20个case）
                if batch_idx < 20:
                    import os
                    debug_dir = os.path.join(self.trainer.log_dir, "debug_nan_cases")
                    os.makedirs(debug_dir, exist_ok=True)
                    try:
                        torch.save({
                            'batch_idx': batch_idx,
                            'epoch': self.current_epoch,
                            'rank': rank,
                            'vertices': batch['vertices'].detach().cpu(),
                            'sequence_ids': batch.get('sequence_id', []),
                            'losses': {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                                      for k, v in losses.items()},
                        }, os.path.join(debug_dir, f'nan_rank{rank}_epoch{self.current_epoch}_batch{batch_idx}.pt'))
                        print(f"   💾 Debug data saved to {debug_dir}")
                    except Exception as e:
                        print(f"   ⚠️  Failed to save debug data: {e}")
            else:
                print(f"⚠️  Rank {rank}: No local NaN, but other rank(s) detected NaN at batch {batch_idx}. Skipping this batch globally.")
            
            # 所有rank都返回一个小的、不参与梯度的loss
            # 使用 requires_grad=False 避免创建新的计算图，让Lightning安全跳过这个step
            safe_loss = torch.tensor(
                1e-6, 
                device=total_loss.device, 
                dtype=torch.float32,
                requires_grad=False  # 关键：不参与backward，避免collective操作不一致
            )
            
            return {"loss": safe_loss}
        
        # Step 4: 正常情况，返回原始loss
        # ============= NaN/Inf检测与处理结束 =============
        
        return {"loss": total_loss}

    def on_train_epoch_start(self):
        """
        在每个训练 epoch 开始时被调用，用于处理参数冻结。
        """
        super().on_train_epoch_start()
        
        # 🔥 保守的epoch开始监控
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            
            # 仅记录epoch开始，减少barrier使用
            if rank == 0:
                print(f"🔄 Epoch {self.current_epoch} starting on all {world_size} ranks")
            
            # 只在epoch开始时同步一次（这个是相对安全的）
            torch.distributed.barrier()
            if rank == 0:
                print(f"✅ All ranks synchronized at epoch {self.current_epoch} start")
        E = self.cfg.freeze_skinning_epochs

        if E > 0:
            # 确定当前 epoch 参数是否应该被冻结
            should_be_frozen = self.current_epoch < E
            
            # 获取需要切换状态的模块 - 新架构
            static_decoder = self.shape_model.static_decoder
            modules_to_toggle = [
                static_decoder.static_mlp,  # 高斯参数预测MLP
            ]

            # 切换所有相关参数的 requires_grad 状态
            for module in modules_to_toggle:
                for param in module.parameters():
                    param.requires_grad = not should_be_frozen
            
            # 仅在状态切换时打印日志
            if self.global_rank == 0:
                # 使用一个内部状态来跟踪冻结状态，避免重复打印
                currently_frozen = getattr(self, "_skinning_is_frozen", False)
                if should_be_frozen and not currently_frozen:
                    print(f"--- Epoch {self.current_epoch}: Freezing skinning parameters for the first {E} epochs. ---")
                elif not should_be_frozen and currently_frozen:
                    print(f"--- Epoch {self.current_epoch}: Unfreezing skinning parameters. ---")
            
            self._skinning_is_frozen = should_be_frozen

    def on_train_start(self):
        """
        Called at the beginning of training to visualize initial skinning weights.
        """
        super().on_train_start()
        # 添加配置选项来禁用耗时的初始可视化
        enable_initial_vis = getattr(self.cfg, 'enable_initial_visualization', False)
        if self.global_rank == 0:
            if enable_initial_vis:
                print("--- Starting initial visualization (this may take several minutes) ---")
            else:
                print("--- Initial visualization disabled for faster training startup ---")
                print("--- Set 'enable_initial_visualization: true' in config to enable visualization ---")
                return
        if enable_initial_vis:
            import os
            # Define paths upfront and create a log file
            vis_base = os.path.join(self.trainer.log_dir, "initial_visualization")
            os.makedirs(vis_base, exist_ok=True)
            log_file_path = os.path.join(vis_base, "initial_analysis.log")
            
            print(f"--- Visualizing initial skinning weights. Analysis will be saved to: {log_file_path} ---")
            
            with open(log_file_path, 'w') as log_file:
                log_file.write("--- Visualizing initial skinning weights at step 0 ---\n")
                print("🔍 Step 1: Loading validation data...")
                
                try:
                    val_loader = self.trainer.datamodule.val_dataloader()
                    batch = next(iter(val_loader))
                    print("✅ Step 1: Data loaded successfully")
                except Exception as e:
                    log_file.write(f"Could not get a batch from val_dataloader: {e}. Aborting initial visualization.\n")
                    print(f"❌ Step 1 Failed: Could not get a batch from val_dataloader: {e}. Aborting initial visualization.")
                    return

                print("🔍 Step 2: Moving data to device...")
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                print("✅ Step 2: Data moved to device successfully")
                
                print("🔍 Step 3: Running forward pass...")
                self.eval()
                with torch.no_grad():
                    output = self(batch)
                self.train()
                print("✅ Step 3: Forward pass completed successfully")

                if 'skinning_weights' in output:
                    print("🔍 Step 4: Processing skinning weights and parameters...")
                    import numpy as np
                    b = 0
                    sample_name = "initial_weights"

                    skinning_weights = output['skinning_weights'][b] # [N, K]
                    target_vertices = batch["vertices"][b] # [T, N, 3]
                    K = skinning_weights.shape[-1]
                    first_frame_vertices = target_vertices[0] # [N, 3]
                    print(f"✅ Step 4: Parameters extracted - K={K}, vertices={first_frame_vertices.shape}")
                    
                    # --- Extract full Gaussian parameters ---
                    print("🔍 Step 5: Extracting Gaussian parameters...")
                    gaussian_params = None
                    if 'decoded_params' in output and 'anchor_indices' in output:
                        decoded_params = output['decoded_params']
                        gaussians = decoded_params['gaussians'][b] # [K, 10]
                        anchor_indices = output['anchor_indices'][b] # [K]
                        
                        gaussian_offset = gaussians[:, :3]
                        gaussian_scales = gaussians[:, 3:6]
                        gaussian_quat = gaussians[:, 6:]
                        
                        anchor_vertices = torch.gather(first_frame_vertices, 0, anchor_indices.unsqueeze(-1).expand(-1, 3))
                        gaussian_centers = anchor_vertices + gaussian_offset
                        
                        # --- Log detailed Gaussian parameters ---
                        if 'bone_offset' in decoded_params:
                            bone_offset = decoded_params['bone_offset'][b] # [K, 3]

                            log_file.write(f"\n--- Initial Gaussian Parameter Details ---\n")
                            np.set_printoptions(precision=4, suppress=True)
                            for k in range(K):
                                log_file.write(f"Bone {k}:\n")
                                log_file.write(f"  - Token Position (Anchor): {anchor_vertices[k].cpu().numpy()}\n")
                                log_file.write(f"  - Gaussian Center Offset:  {gaussian_offset[k].cpu().numpy()}\n")
                                log_file.write(f"  - Bone Center Offset:      {bone_offset[k].cpu().numpy()}\n")
                                log_file.write(f"  - Gaussian Scale:          {gaussian_scales[k].cpu().numpy()}\n")
                                log_file.write(f"  - Gaussian Quaternion:     {gaussian_quat[k].cpu().numpy()}\n")
                            log_file.write("-----------------------------------------\n")
                            np.set_printoptions() # Reset to default
                        
                        gaussian_params = {
                            'centers': gaussian_centers,
                            'scales': gaussian_scales,
                            'quats': gaussian_quat
                        }
                        print(f"✅ Step 5: Gaussian parameters extracted successfully")
                        print(f"🔍 Debug: gaussian_centers shape: {gaussian_centers.shape}, range: [{gaussian_centers.min():.4f}, {gaussian_centers.max():.4f}]")
                        print(f"🔍 Debug: gaussian_scales shape: {gaussian_scales.shape}, range: [{gaussian_scales.min():.4f}, {gaussian_scales.max():.4f}]")
                        print(f"🔍 Debug: gaussian_quat shape: {gaussian_quat.shape}, range: [{gaussian_quat.min():.4f}, {gaussian_quat.max():.4f}]")

                    # --- Save combined visualization of all Gaussians ---
                    if gaussian_params:
                        print("🔍 Step 6: Generating Gaussian ellipsoids HTML (this may take a while)...")
                        print(f"🔍 Debug: About to generate ellipsoids for {gaussian_params['centers'].shape[0]} gaussians")
                        
                        # Test ellipsoid generation for first gaussian
                        try:
                            ex, ey, ez, ei, ej, ek = self._generate_ellipsoid_mesh(
                                gaussian_params['centers'][0],
                                gaussian_params['scales'][0],
                                gaussian_params['quats'][0]
                            )
                            print(f"🔍 Debug: Test ellipsoid generated successfully - verts: {len(ex)}, faces: {len(ei)}")
                        except Exception as e:
                            print(f"❌ Debug: Error generating test ellipsoid: {e}")
                        
                        all_gaussians_path = os.path.join(vis_base, "initial_all_gaussians.html")
                        self.save_all_gaussians_html(all_gaussians_path, first_frame_vertices, gaussian_params, log_file)
                        print("✅ Step 6: Gaussian ellipsoids HTML saved successfully")

                    # --- Save Anchor Points ---
                    print("🔍 Step 7: Saving anchor points...")
                    if 'anchor_indices' in output:
                        anchor_indices = output['anchor_indices'][b] # [K]
                        anchor_vertices = torch.gather(first_frame_vertices, 0, anchor_indices.unsqueeze(-1).expand(-1, 3))
                        anchor_path = os.path.join(vis_base, "initial_anchor_points.obj")
                        self.save_single_pointcloud(anchor_path, anchor_vertices)
                        log_file.write(f"Saved initial anchor points to: {anchor_path}\n")
                        print("✅ Step 7: Anchor points saved successfully")
                    
                    print("🔍 Step 8: Analyzing bone assignments...")
                    bone_assignments = torch.argmax(skinning_weights, dim=-1)
                    unique_bones, counts = torch.unique(bone_assignments, return_counts=True)
                    log_file.write(f"\n--- Initial Bone Assignment Analysis ---\n")
                    log_file.write(f"Total bones (K): {K}\n")
                    log_file.write(f"Active bones (assigned to >0 vertices): {len(unique_bones)}/{K}\n")
                    log_file.write("Vertex counts per active bone:\n")
                    
                    all_bone_counts = {i: 0 for i in range(K)}
                    for bone_idx, count in zip(unique_bones.cpu().numpy(), counts.cpu().numpy()):
                        all_bone_counts[bone_idx] = count
                    
                    for i in sorted(all_bone_counts.keys()):
                        log_file.write(f"  - Bone {i}: {all_bone_counts[i]} vertices\n")
                    log_file.write("-----------------------------------------\n")
                    print("✅ Step 8: Bone assignment analysis completed")

                    print("🔍 Step 9: Generating color mappings...")
                    try:
                        import matplotlib.cm as cm
                        colormap = cm.get_cmap('gist_rainbow', K)
                        colors_np = colormap(np.arange(K))[:, :3]
                        bone_colors = torch.from_numpy(colors_np).to(skinning_weights.device).float()
                        print("✅ Step 9: Matplotlib colormap loaded successfully")
                    except ImportError:
                        bone_colors = torch.rand(K, 3, device=skinning_weights.device)
                        print("⚠️ Step 9: Matplotlib not available, using random colors")

                    vertex_colors = bone_colors[bone_assignments]

                    print("🔍 Step 10: Saving argmax visualization OBJ file...")
                    # Save argmax visualization
                    save_path = os.path.join(vis_base, f"{sample_name}_argmax.obj")
                    self.save_colored_pointcloud(
                        save_path,
                        first_frame_vertices,
                        vertex_colors
                    )
                    log_file.write(f"Saved initial argmax skinning weights visualization to: {save_path}\n")
                    print("✅ Step 10: Argmax visualization OBJ saved successfully")

                    print("🔍 Step 11: Generating weight heatmap HTMLs (this may take a while)...")
                    # --- Save per-bone weight maps as HTML ---
                    html_base_path = os.path.join(vis_base, "weights_heatmap")
                    self.save_weights_as_htmls(html_base_path, first_frame_vertices, skinning_weights, gaussian_params, log_file)
                    print("✅ Step 11: Weight heatmap HTMLs saved successfully")
                    
                print("🎉 All initial visualization steps completed successfully!")
                
                # 强制清理可视化占用的显存
                print("🧹 Clearing visualization tensors from memory...")
                del batch, output, skinning_weights, target_vertices, first_frame_vertices
                if 'gaussian_params' in locals(): del gaussian_params
                if 'vertex_colors' in locals(): del vertex_colors
                if 'bone_assignments' in locals(): del bone_assignments
                torch.cuda.empty_cache()
                print("✅ GPU memory cache cleared. Proceeding to training...")
                
                print("--- You can now proceed with training ---")


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        验证步骤
        """
        self.eval()
        output = self(batch)
        
        # 计算损失
        losses = self.shape_model.compute_losses(output, batch["vertices"])
        
        # 记录验证损失
        for name, value in losses.items():
            self.log(f"val/{name}", value)
        
        # 添加运动分析指标
        self._log_motion_analysis(output, batch)
        
        # 计算顶点重建误差
        pred_vertices = output['animated_vertices']  # [B, T, N, 3]
        target_vertices = batch["vertices"][:, 1:]  # [B, T-1, N, 3]
        
        # 计算每帧的平均顶点误差
        vertex_errors = torch.norm(pred_vertices - target_vertices, dim=-1)  # [B, T, N]
        mean_vertex_error = vertex_errors.mean()
        max_vertex_error = vertex_errors.max()
        
        self.log("val/mean_vertex_error", mean_vertex_error)
        self.log("val/max_vertex_error", max_vertex_error)
        
        # 计算Chamfer距离（如果需要）
        if "chamfer_distance" in batch:
            chamfer_dist = self._compute_chamfer_distance(pred_vertices, target_vertices)
            self.log("val/chamfer_distance", chamfer_dist)
        
        # ---------------------------------------------
        # 可视化：按配置周期保存首个 batch 的 GT 与重建网格
        # 仅在 global_rank == 0 且 batch_idx==0 时执行
        '''
        if (
            batch_idx == 0
            and self.global_rank == 0  # 只在主进程保存
            and (self.current_epoch % self.cfg.visualize_every_n_epochs == 0)
        ):
            import os
            vis_base = os.path.join(
                "./outputs",
                "vis",
                f"epoch_{self.current_epoch:04d}"
            )
            os.makedirs(vis_base, exist_ok=True)

            # 只处理 batch 中第一个样本 b=0
            pred_seq = pred_vertices[0].detach().cpu()  # (T,N,3)
            gt_seq = target_vertices[0].detach().cpu()

            # self.save_vertex_sequence(os.path.join(vis_base, "pred_vertices.npy"), pred_seq)
            # self.save_vertex_sequence(os.path.join(vis_base, "gt_vertices.npy"), gt_seq)
            # 导出点云而非网格
            self.save_pointcloud_sequence(os.path.join(vis_base, "pred_pointcloud"), pred_seq)
            self.save_pointcloud_sequence(os.path.join(vis_base, "gt_pointcloud"), gt_seq)
        '''
        
        # 强制清理GPU内存
        del output, pred_vertices, target_vertices
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "val/loss": losses["total_loss"],
            "val/mean_vertex_error": mean_vertex_error,
            "val/max_vertex_error": max_vertex_error,
        }

    def _compute_chamfer_distance(self, pred_vertices, target_vertices):
        """
        计算Chamfer距离（简化版本）
        """
        B, T, N, _ = pred_vertices.shape
        
        # 展平批次和时间维度
        pred_flat = pred_vertices.reshape(B * T, N, 3)
        target_flat = target_vertices.reshape(B * T, N, 3)
        
        # 计算最近邻距离
        dist_pred_to_target = torch.cdist(pred_flat, target_flat).min(dim=-1)[0]  # [B*T, N]
        dist_target_to_pred = torch.cdist(target_flat, pred_flat).min(dim=-1)[0]  # [B*T, N]
        
        # 计算Chamfer距离
        chamfer_dist = (dist_pred_to_target.mean() + dist_target_to_pred.mean()) / 2
        
        return chamfer_dist

    def _log_motion_analysis(self, output, batch):
        """分析和记录运动模式"""
        try:
            if 'decoded_params' not in output:
                return
                
            decoded_params = output['decoded_params']
            transforms = decoded_params['transforms']  # [B, K, T-1, 7]
            gaussians = decoded_params['gaussians']    # [B, K, 10]
            
            B, K, T_minus_1, _ = transforms.shape
            
            # 1. 分析SE(3)变换的多样性
            motion_quat = transforms[:, :, :, :4]  # [B, K, T-1, 4]
            motion_trans = transforms[:, :, :, 4:]  # [B, K, T-1, 3]
            
            # 计算旋转多样性：骨骼间的平均旋转差异
            quat_norm = torch.norm(motion_quat, dim=-1, keepdim=True)
            quat_norm = torch.clamp(quat_norm, min=1e-8)
            motion_quat_normalized = motion_quat / quat_norm
            
            # 计算骨骼间旋转相似性
            quat_expanded_i = motion_quat_normalized.unsqueeze(2)  # [B, K, 1, T-1, 4]
            quat_expanded_j = motion_quat_normalized.unsqueeze(1)  # [B, 1, K, T-1, 4]
            quat_similarity = torch.abs(torch.sum(quat_expanded_i * quat_expanded_j, dim=-1))  # [B, K, K, T-1]
            
            # 排除对角线元素
            mask = torch.eye(K, device=transforms.device).bool()
            quat_similarity_off_diag = quat_similarity.masked_fill(
                mask.unsqueeze(0).unsqueeze(-1), 0
            )
            
            # 计算平均骨骼间旋转相似性
            num_pairs = K * (K - 1)
            avg_quat_similarity = quat_similarity_off_diag.sum() / (B * num_pairs * T_minus_1)
            self.log("val/avg_bone_rotation_similarity", avg_quat_similarity)
            
            # 计算平移多样性
            trans_expanded_i = motion_trans.unsqueeze(2)  # [B, K, 1, T-1, 3]
            trans_expanded_j = motion_trans.unsqueeze(1)  # [B, 1, K, T-1, 3]
            trans_dist = torch.norm(trans_expanded_i - trans_expanded_j, dim=-1)  # [B, K, K, T-1]
            trans_dist_off_diag = trans_dist.masked_fill(
                mask.unsqueeze(0).unsqueeze(-1), float('inf')
            )
            
            # 计算平均骨骼间平移距离
            finite_mask = torch.isfinite(trans_dist_off_diag)
            if finite_mask.any():
                avg_trans_distance = trans_dist_off_diag[finite_mask].mean()
                self.log("val/avg_bone_translation_distance", avg_trans_distance)
            
            # 2. 分析高斯权重的稀疏性
            if 'animated_vertices' in output:
                # 这里可以分析权重分布，但需要修改模型以返回权重
                pass
                
            # 3. 分析运动幅度
            # 旋转幅度：计算每个骨骼的平均旋转角度
            max_quat_val = torch.tensor(1.0, device=motion_quat_normalized.device, dtype=motion_quat_normalized.dtype)
            quat_angles = 2 * torch.acos(torch.clamp(torch.abs(motion_quat_normalized[:, :, :, 0]), max=max_quat_val))
            avg_rotation_angle = quat_angles.mean()
            self.log("val/avg_rotation_angle_rad", avg_rotation_angle)
            
            # 平移幅度：计算每个骨骼的平均平移距离
            trans_magnitude = torch.norm(motion_trans, dim=-1)  # [B, K, T-1]
            avg_translation_magnitude = trans_magnitude.mean()
            self.log("val/avg_translation_magnitude", avg_translation_magnitude)
            
            # 4. 分析时间变化
            if T_minus_1 > 1:
                # 计算时间上的变化率
                quat_temporal_diff = motion_quat_normalized[:, :, 1:] - motion_quat_normalized[:, :, :-1]
                quat_temporal_change = torch.norm(quat_temporal_diff, dim=-1).mean()
                self.log("val/temporal_rotation_change", quat_temporal_change)
                
                trans_temporal_diff = motion_trans[:, :, 1:] - motion_trans[:, :, :-1]
                trans_temporal_change = torch.norm(trans_temporal_diff, dim=-1).mean()
                self.log("val/temporal_translation_change", trans_temporal_change)
                
        except Exception as e:
            print(f"Error in motion analysis: {e}")

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        """
        测试步骤
        """
        self.eval()
        split = getattr(self, "_eval_split", "test")
        output = self(batch)
        
        # 计算损失
        losses = self.shape_model.compute_losses(output, batch["vertices"])
        
        # 记录分割感知的测试损失
        metric_prefix = f"test_{split}"
        for name, value in losses.items():
            self.log(f"{metric_prefix}/{name}", value)
        
        # 记录每个样本的详细损失（仅在 rank 0）
        if self.trainer.global_rank == 0:
            seq_ids = batch.get('sequence_id', [f'unknown_{i}' for i in range(len(batch['vertices']))])
            B = len(seq_ids)
            
            # 计算每样本的重建损失
            pred_vertices = output['animated_vertices']  # [B, T-1, N, 3]
            target_vertices = batch["vertices"][:, 1:]  # [B, T-1, N, 3]
            
            for b in range(B):
                sample_recon_loss = torch.nn.functional.mse_loss(
                    pred_vertices[b], target_vertices[b]
                ).item()
                
                # 记录每个样本的损失
                self.log(f"{metric_prefix}/sample_{seq_ids[b]}_recon_loss", 
                        sample_recon_loss, rank_zero_only=True)
                
                # 计算每个样本的顶点误差统计
                vertex_errors = torch.norm(pred_vertices[b] - target_vertices[b], dim=-1)
                mean_error = vertex_errors.mean().item()
                max_error = vertex_errors.max().item()
                
                self.log(f"{metric_prefix}/sample_{seq_ids[b]}_mean_vertex_error", 
                        mean_error, rank_zero_only=True)
                self.log(f"{metric_prefix}/sample_{seq_ids[b]}_max_vertex_error", 
                        max_error, rank_zero_only=True)
                
                # 打印详细信息
                if batch_idx < 5:  # 只打印前几个batch的详细信息
                    print(f"[{split}] Sample {seq_ids[b]}: recon_loss={sample_recon_loss:.6f}, "
                          f"mean_error={mean_error:.6f}, max_error={max_error:.6f}")
        
        if self.trainer.global_rank == 0:
            import os
            import numpy as np
            seq_ids = batch['sequence_id']
            batch_size = len(seq_ids)
            transforms = output['decoded_params']['transforms'].float().cpu().numpy()  # [B, K, T, 7]
            root_quat = output['decoded_params']['root_quat'].float().cpu().numpy()    # [B, T, 4]
            root_trans = output['decoded_params']['root_trans'].float().cpu().numpy()  # [B, T, 3]
            for i in range(batch_size):
                save_dir = os.path.join(self.trainer.log_dir or '.', 'test_transforms', seq_ids[i])
                os.makedirs(save_dir, exist_ok=True)
                txt_path = os.path.join(save_dir, 'transforms.txt')
                with open(txt_path, 'w') as f:
                    K = transforms.shape[1]
                    T = transforms.shape[2]
                    for t in range(T):
                        f.write(f'Time {t}:\n')
                        rq = ' '.join(f'{x:.6f}' for x in root_quat[i, t])
                        rt = ' '.join(f'{x:.6f}' for x in root_trans[i, t])
                        f.write(f'Root: quat [{rq}] trans [{rt}]\n')
                        for k in range(K):
                            bq = ' '.join(f'{x:.6f}' for x in transforms[i, k, t, :4])
                            bt = ' '.join(f'{x:.6f}' for x in transforms[i, k, t, 4:])
                            f.write(f'Bone {k}: quat [{bq}] trans [{bt}]\n')
                        f.write('\n')
            
            if batch_idx % 100 == 0:
                pred_vertices = output['animated_vertices']  # [B, T, N, 3]
                target_vertices = batch["vertices"]  # [B, T, N, 3]
                # 保存预测和目标序列（按 split 目录区分）
                saved_count = getattr(self, "_saved_cases_by_split", {}).get(split, 0)
                for b in range(pred_vertices.shape[0]):
                    sample_name = f"sample_{batch_idx}_{b}"
                    pred_seq = pred_vertices[b]
                    target_seq = target_vertices[b]
                    self.save_vertex_sequence(f"test/{split}/{sample_name}_pred.npy", pred_seq)
                    self.save_vertex_sequence(f"test/{split}/{sample_name}_target.npy", target_seq)
                    # 在每个 split 内最多保存 20 个 OBJ 可视化（gt 与 pred）
                    if saved_count < getattr(self, "_save_cap_per_split", 20):
                        self.save_pointcloud_sequence(
                            f"test/{split}/{sample_name}_pred_pointcloud",
                            pred_seq
                        )
                        self.save_pointcloud_sequence(
                            f"test/{split}/{sample_name}_gt_pointcloud",
                            target_seq
                        )
                        saved_count += 1
                        self._saved_cases_by_split[split] = saved_count
                    # --- Add visualization for skinning weights ---
                    if 'skinning_weights' in output:
                        skinning_weights = output['skinning_weights']  # [B, N, K]
                        K = skinning_weights.shape[-1]

                        # 1. Get bone assignment for each vertex for sample b
                        bone_assignments = torch.argmax(skinning_weights[b], dim=-1)  # [N]

                        # --- Analyze and log bone assignments (on rank 0) ---
                        if get_rank() == 0:
                            unique_bones, counts = torch.unique(bone_assignments, return_counts=True)

                            # Log summary statistics
                            self.log(f"{metric_prefix}/{sample_name}_num_bones", float(K), rank_zero_only=True)
                            self.log(f"{metric_prefix}/{sample_name}_active_bones", float(len(unique_bones)), rank_zero_only=True)

                            if len(counts) > 0:
                                counts_float = counts.float()
                                self.log(f"{metric_prefix}/{sample_name}_vertices_per_bone_mean", counts_float.mean(), rank_zero_only=True)
                                self.log(f"{metric_prefix}/{sample_name}_vertices_per_bone_std", counts_float.std(), rank_zero_only=True)
                                self.log(f"{metric_prefix}/{sample_name}_vertices_per_bone_min", counts_float.min(), rank_zero_only=True)
                                self.log(f"{metric_prefix}/{sample_name}_vertices_per_bone_max", counts_float.max(), rank_zero_only=True)

                            # Print a detailed breakdown to the console
                            print(f"\n--- Bone Assignment Analysis for {sample_name} ---")
                            print(f"Total bones (K): {K}")
                            print(f"Active bones (assigned to >0 vertices): {len(unique_bones)}/{K}")
                            print("Vertex counts per active bone:")
                            
                            all_bone_counts = {i: 0 for i in range(K)}
                            for bone_idx, count in zip(unique_bones.cpu().numpy(), counts.cpu().numpy()):
                                all_bone_counts[bone_idx] = count
                            
                            for i in sorted(all_bone_counts.keys()):
                                print(f"  - Bone {i}: {all_bone_counts[i]} vertices")
                            print("-------------------------------------------------")

                        # 2. Create a color map for bones
                        try:
                            import matplotlib.cm as cm
                            # Using a colormap like 'gist_rainbow' for good color distinction
                            colormap = cm.get_cmap('gist_rainbow', K)
                            colors_np = colormap(np.arange(K))[:, :3]  # RGBA -> RGB
                            bone_colors = torch.from_numpy(colors_np).to(skinning_weights.device).float()
                        except ImportError:
                            # Fallback if matplotlib is not available
                            self.log_once("Matplotlib not found, using random colors for visualization.")
                            bone_colors = torch.rand(K, 3, device=skinning_weights.device)

                        # 3. Assign colors to vertices
                        vertex_colors = bone_colors[bone_assignments]  # [N, 3]

                        # 4. Get vertices of the first frame
                        first_frame_vertices = target_vertices[b, 0]  # [N, 3]

                        # 5. Save colored point cloud for the first frame
                        self.save_colored_pointcloud(
                            f"test/{split}/{sample_name}_skinning_weights.obj",
                            first_frame_vertices,
                            vertex_colors
                        )
        
        return {
            f"{metric_prefix}/loss": losses["total_loss"],
            f"{metric_prefix}/reconstruction_loss": losses["reconstruction_loss"],
            f"{metric_prefix}/motion_smooth_loss": losses["motion_smooth_loss"],
            f"{metric_prefix}/kl_loss": losses["kl_loss"],
            f"{metric_prefix}/diversity_loss": losses.get("diversity_loss", 0.0),
            f"{metric_prefix}/sparsity_loss": losses.get("sparsity_loss", 0.0),
        }

    def save_vertex_sequence(self, path: str, vertices: torch.Tensor):
        """
        保存顶点序列为numpy文件
        Args:
            path: 保存路径
            vertices: [T, N, 3] - 顶点序列
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, vertices.float().cpu().numpy())

    def save_mesh_sequence(self, base_path: str, vertices: torch.Tensor, faces: torch.Tensor):
        """
        保存网格序列为OBJ文件
        Args:
            base_path: 基础路径
            vertices: [T, N, 3] - 顶点序列
            faces: [F, 3] - 面信息
        """
        import os
        import trimesh
        
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        
        T = vertices.shape[0]
        for t in range(T):
            # 创建网格
            mesh = trimesh.Trimesh(
                vertices=vertices[t].cpu().numpy(),
                faces=faces.cpu().numpy()
            )
            
            # 保存为OBJ文件
            mesh_path = f"{base_path}_frame_{t:03d}.obj"
            mesh.export(mesh_path) 

    def save_pointcloud_sequence(self, base_path: str, vertices: torch.Tensor):
        """
        保存点云序列为OBJ文件，仅导出顶点
        Args:
            base_path: 基础路径（不含后缀和帧编号）
            vertices: [T, N, 3] - 顶点序列
        """
        import os
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        verts_np = vertices.cpu().numpy()  # [T, N, 3]
        T = verts_np.shape[0]
        for t in range(T):
            file_path = f"{base_path}_frame_{t:03d}.obj"
            with open(file_path, 'w') as f:
                for v in verts_np[t]:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")

    def save_single_pointcloud(self, path: str, vertices: torch.Tensor):
        """
        Saves a single point cloud to an OBJ file.
        Args:
            path: Output file path.
            vertices: [N, 3] tensor of vertex positions.
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        verts_np = vertices.cpu().numpy()
        with open(path, 'w') as f:
            for v in verts_np:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

    def save_all_gaussians_html(self, path: str, vertices: torch.Tensor, gaussian_params: dict, log_file=None):
        """Saves all Gaussian ellipsoids and their centers to a single interactive HTML file."""
        print("  📦 Importing required libraries...")
        try:
            import plotly.graph_objects as go
            import matplotlib.cm as cm
            import numpy as np
            print("  ✅ Libraries imported successfully")
        except ImportError:
            print("  ❌ Plotly or Matplotlib not found. Skipping all-Gaussians visualization.")
            self.log_once("Plotly or Matplotlib not found. Skipping all-Gaussians visualization.")
            return

        message = f"  💾 Saving all-Gaussians visualization to: {path}"
        print(message)
        if log_file:
            log_file.write(message + "\n")

        print("  🗂️ Converting vertices to numpy...")
        verts_np = vertices.cpu().numpy()
        
        print("  🎨 Creating mesh scatter plot...")
        mesh_trace = go.Scatter3d(
            x=verts_np[:, 0], y=verts_np[:, 1], z=verts_np[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='lightgrey', opacity=0.3),
            name='Character Mesh'
        )

        traces = [mesh_trace]
        K = gaussian_params['centers'].shape[0]
        print(f"  🎯 Creating colormap for {K} bones...")
        colormap = cm.get_cmap('gist_rainbow', K)
        
        centers_np = gaussian_params['centers'].cpu().numpy()
        
        print("  📍 Adding bone centers to plot...")
        traces.append(go.Scatter3d(
            x=centers_np[:, 0], y=centers_np[:, 1], z=centers_np[:, 2],
            mode='markers+text',
            marker=dict(size=5, color='black'),
            text=[f'B{k}' for k in range(K)],
            textposition='top center',
            name='Bone Centers',
            hovertemplate = '<b>Bone %{text}</b><br>Center: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))

        print(f"  🔄 Generating {K} ellipsoid meshes...")
        for k in range(K):
            if k % 10 == 0:  # Progress indicator every 10 ellipsoids
                print(f"    📐 Generating ellipsoid {k+1}/{K}...")
            color_rgba = colormap(k)
            plotly_color = f'rgba({color_rgba[0]*255}, {color_rgba[1]*255}, {color_rgba[2]*255}, 0.3)'

            ex, ey, ez, ei, ej, ek = self._generate_ellipsoid_mesh(
                gaussian_params['centers'][k],
                gaussian_params['scales'][k],
                gaussian_params['quats'][k]
            )
            ellipsoid_trace = go.Mesh3d(
                x=ex, y=ey, z=ez,
                i=ei, j=ej, k=ek,
                color=plotly_color,
                name=f'Bone {k} Gaussian',
                showlegend=True
            )
            traces.append(ellipsoid_trace)
        print(f"  ✅ All {K} ellipsoid meshes generated successfully")

        print("  📊 Creating Plotly figure...")
        fig = go.Figure(data=traces)
        print("  ⚙️ Updating figure layout...")
        fig.update_layout(
            title='All Initial Gaussian Ellipsoids',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            margin=dict(r=20, l=10, b=10, t=40)
        )
        
        print("  💾 Writing HTML file...")
        fig.write_html(path)
        print("  ✅ HTML file written successfully")
        
    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Converts a quaternion to a rotation matrix."""
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        q_safe = q / torch.clamp(q_norm, min=1e-8)
        w, x, y, z = q_safe.unbind(-1)
        
        matrix = torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=-1).view(*q.shape[:-1], 3, 3)
        
        return matrix

    def _generate_ellipsoid_mesh(self, center, scale, quaternion, resolution=10):
        """Generates vertices and faces for a transformed ellipsoid mesh."""
        import numpy as np
        
        # Generate sphere mesh
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        verts_sphere = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

        # Generate faces (this is the potentially slow part)
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                p1 = i * resolution + j
                p2 = i * resolution + j + 1
                p3 = (i + 1) * resolution + j
                p4 = (i + 1) * resolution + j + 1
                faces.append([p1, p3, p2])
                faces.append([p2, p3, p4])
        faces = np.array(faces)

        # Transform to GPU and apply scaling/rotation
        verts_sphere_torch = torch.from_numpy(verts_sphere).to(center.device).float()
        scaled_verts = verts_sphere_torch * scale.unsqueeze(0)
        
        rot_mat = self._quaternion_to_matrix(quaternion)
        rotated_verts = torch.matmul(scaled_verts, rot_mat.T)
        
        final_verts = rotated_verts + center.unsqueeze(0)
        verts_np = final_verts.cpu().numpy()
        
        return verts_np[:, 0], verts_np[:, 1], verts_np[:, 2], faces[:, 0], faces[:, 1], faces[:, 2]


    def save_all_gaussians_html(self, path: str, vertices: torch.Tensor, gaussian_params: dict, log_file=None):
        """
        Saves a single HTML file with the character mesh and all Gaussian ellipsoids.
        Args:
            path: Output HTML file path.
            vertices: [N, 3] tensor of character vertex positions.
            gaussian_params: Dictionary with 'centers', 'scales', 'quats'.
            log_file: Optional file handle to write log messages to.
        """
        if not gaussian_params:
            return
            
        try:
            import plotly.graph_objects as go
            import numpy as np
            import matplotlib.cm as cm
            import os
        except ImportError:
            self.log_once("Plotly or Matplotlib not found. Skipping combined Gaussian visualization.")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        K = gaussian_params['centers'].shape[0]
        verts_np = vertices.cpu().numpy()
        centers_np = gaussian_params['centers'].cpu().numpy()
        
        # --- Create traces ---
        traces = []

        # 1. Character mesh trace
        mesh_trace = go.Scatter3d(
            x=verts_np[:, 0], y=verts_np[:, 1], z=verts_np[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='lightgrey', opacity=0.5),
            name='Character Mesh'
        )
        traces.append(mesh_trace)

        # 2. Gaussian center points trace
        center_trace = go.Scatter3d(
            x=centers_np[:, 0], y=centers_np[:, 1], z=centers_np[:, 2],
            mode='markers+text',
            marker=dict(size=5, color='black'),
            text=[f'Bone {k}' for k in range(K)],
            textposition='top center',
            name='Bone Centers'
        )
        traces.append(center_trace)

        # 3. Ellipsoid mesh traces
        colormap = cm.get_cmap('gist_rainbow', K)
        
        for k in range(K):
            ex, ey, ez, ei, ej, ek = self._generate_ellipsoid_mesh(
                gaussian_params['centers'][k],
                gaussian_params['scales'][k],
                gaussian_params['quats'][k]
            )
            color_rgba = colormap(k)
            color_str = f'rgba({int(color_rgba[0]*255)}, {int(color_rgba[1]*255)}, {int(color_rgba[2]*255)}, 0.3)'

            ellipsoid_trace = go.Mesh3d(
                x=ex, y=ey, z=ez,
                i=ei, j=ej, k=ek,
                color=color_str,
                opacity=0.3,
                name=f'Bone {k} Gaussian'
            )
            traces.append(ellipsoid_trace)

        # --- Create figure and save ---
        fig = go.Figure(data=traces)
        fig.update_layout(
            title='Initial Layout of All Gaussian Ellipsoids',
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data' # This is important for correct aspect ratio
            ),
            margin=dict(r=20, l=10, b=10, t=40)
        )
        
        fig.write_html(path)
        
        message = f"Saved combined Gaussian visualization to: {path}"
        if log_file:
            log_file.write(message + "\n")
        else:
            print(message)

    def save_weights_as_htmls(self, base_path: str, vertices: torch.Tensor, weights: torch.Tensor, gaussian_params: dict, log_file=None):
        """
        Saves skinning weights as interactive HTML files (one per bone).
        Args:
            base_path: Base path for the output files (e.g., '.../dir/weights_viz').
            vertices: [N, 3] tensor of vertex positions.
            weights: [N, K] tensor of skinning weights.
            gaussian_params: Dictionary with 'centers', 'scales', 'quats'.
            log_file: Optional file handle to write log messages to.
        """
        print("  📦 Importing Plotly for weight heatmaps...")
        try:
            import plotly.graph_objects as go
            import os
            print("  ✅ Plotly imported successfully")
        except ImportError:
            print("  ❌ Plotly not found. Skipping HTML skinning weight visualization.")
            self.log_once("Plotly not found. Skipping HTML skinning weight visualization.")
            return

        print("  📁 Creating output directory...")
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        N, K = weights.shape
        print(f"  🗂️ Converting data to numpy (N={N}, K={K})...")
        verts_np = vertices.cpu().numpy()
        weights_np = weights.cpu().numpy()

        message = f"  💾 Saving {K} HTML weight maps to files starting with: {base_path}..."
        print(message)
        if log_file:
            log_file.write(message + "\n")

        for k in range(K):
            if k % 10 == 0:  # Progress indicator every 10 bones
                print(f"    📊 Generating weight heatmap {k+1}/{K}...")
            scatter_trace = go.Scatter3d(
                 x=verts_np[:, 0],
                 y=verts_np[:, 1],
                 z=verts_np[:, 2],
                 mode='markers',
                 marker=dict(
                     size=2,
                     color=weights_np[:, k],
                     colorscale='Viridis',
                     colorbar=dict(title='Weight'),
                     showscale=True,
                     cmin=0.0,
                     cmax=1.0
                 ),
                 name='Vertices',
                 hovertemplate=(
                     '<b>Vertex Index:</b> %{pointNumber}<br>'
                     f'<b>Weight for Bone {k}:</b> %{{marker.color:.4f}}<br>'
                     '<extra></extra>'
                 )
            )
            
            traces = [scatter_trace]
            
            # Add ellipsoid mesh if available
            if gaussian_params:
                ex, ey, ez, ei, ej, ek = self._generate_ellipsoid_mesh(
                    gaussian_params['centers'][k],
                    gaussian_params['scales'][k],
                    gaussian_params['quats'][k]
                )
                ellipsoid_trace = go.Mesh3d(
                    x=ex, y=ey, z=ez,
                    i=ei, j=ej, k=ek,
                    opacity=0.3,
                    color='cyan',
                    name=f'Bone {k} Gaussian'
                )
                traces.append(ellipsoid_trace)

            fig = go.Figure(data=traces)
            fig.update_layout(
                title=f'Skinning Weights for Bone {k}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                margin=dict(r=20, l=10, b=10, t=40)
            )
            
            html_path = f"{base_path}_bone_{k}.html"
            fig.write_html(html_path)

        message = "Done saving HTML weight maps."
        if log_file:
            log_file.write(message + "\n")
        else:
            print(message)

    def save_colored_pointcloud(self, path: str, vertices: torch.Tensor, colors: torch.Tensor):
        """
        Saves a colored point cloud to an OBJ file with vertex colors.
        Args:
            path: Output file path.
            vertices: [N, 3] tensor of vertex positions.
            colors: [N, 3] tensor of vertex colors (RGB, 0-1).
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        verts_np = vertices.cpu().numpy()
        colors_np = colors.cpu().numpy()
        with open(path, 'w') as f:
            for i in range(verts_np.shape[0]):
                v = verts_np[i]
                c = colors_np[i]
                f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n") 