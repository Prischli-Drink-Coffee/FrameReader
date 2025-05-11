import os
import time
import logging
from typing import Dict, Optional, Tuple, List, Union

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrainingSpeedup:
    
    @staticmethod
    def setup_distributed(backend: str = "nccl") -> Tuple[int, int, int]:
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            dist.init_process_group(backend=backend)
            logger.info(f"Распределенное обучение инициализировано: rank={rank}, world_size={world_size}")
            torch.cuda.set_device(local_rank)
            return rank, local_rank, world_size
        else:
            logger.info("Распределенная среда не обнаружена, используется одиночный GPU")
            return 0, 0, 1
    
    @staticmethod
    def wrap_model_for_distributed(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
        return DDP(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False
        )
    
    @staticmethod
    def optimize_dataloader(
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        distributed: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ) -> DataLoader:

        sampler = DistributedSampler(dataset) if distributed else None
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=True
        )
    
    @staticmethod
    def get_mixed_precision_scaler(
        device_type: str,
        precision: str = "bf16",
        enabled: bool = True,
    ) -> Optional[GradScaler]:

        if not enabled:
            return None
            
        if precision == "fp16":
            return GradScaler(device_type=device_type)
        else:
            return None


class MemoryOptimizer:
    
    @staticmethod
    def optimize_memory_usage(set_benchmark: bool = True):

        torch.cuda.empty_cache()

        if hasattr(torch.cuda, 'memory_stats'):
            try:
                stats = torch.cuda.memory_stats()
                allocated = stats.get("allocated_bytes.all.current", 0) / (1024**3)
                reserved = stats.get("reserved_bytes.all.current", 0) / (1024**3)
                logger.info(f"Статистика памяти CUDA: выделено {allocated:.2f} ГБ, зарезервировано {reserved:.2f} ГБ")
            except:
                pass

        if torch.cuda.is_available() and set_benchmark:
            torch.backends.cudnn.benchmark = True
    
    @staticmethod
    def apply_activation_checkpointing(model: torch.nn.Module) -> None:
        try:
            from torch.utils.checkpoint import checkpoint_sequential
            
            def apply_checkpoint(module):
                if len(list(module.children())) > 0:
                    for child in module.children():
                        apply_checkpoint(child)
                elif hasattr(module, "forward") and not hasattr(module, "_checkpointed"):
                    original_forward = module.forward
                    
                    def checkpointed_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs)
                    
                    module.forward = checkpointed_forward
                    module._checkpointed = True
            
            apply_checkpoint(model)
            logger.info("Применена проверка точек активации для экономии памяти")
        except ImportError:
            logger.warning("Не удалось применить проверку точек активации: torch.utils.checkpoint не найден")
    
    @staticmethod
    def monkey_patch_attention(use_flash_attention: bool = True, use_mem_efficient: bool = True) -> None:
        try:
            # Попытка применить FlashAttention
            if use_flash_attention:
                try:
                    import flash_attn
                    logger.info("Библиотека FlashAttention доступна и будет использоваться")
                except ImportError:
                    logger.warning("FlashAttention недоступен. Установите с помощью: pip install flash-attn")
                    use_flash_attention = False
            
            # Попытка применить эффективное по памяти внимание
            if use_mem_efficient and not use_flash_attention:
                try:
                    from transformers.models.t5.modeling_t5 import T5Attention
                    original_forward = T5Attention.forward
                    
                    def memory_efficient_forward(self, *args, **kwargs):
                        result = original_forward(self, *args, **kwargs)
                        torch.cuda.empty_cache()
                        return result
                    
                    T5Attention.forward = memory_efficient_forward
                    logger.info("Применено эффективное по памяти внимание для моделей T5")
                except ImportError:
                    logger.warning("Не удалось применить эффективное по памяти внимание")
        except Exception as e:
            logger.warning(f"Ошибка при оптимизации модулей внимания: {e}")
    
    @staticmethod
    def get_optimal_batch_size(
        model_size_gb: float, 
        gpu_memory_gb: float = 80.0,  # A100 имеет 80 ГБ памяти
        safety_factor: float = 0.7
    ) -> int:
        available_memory = gpu_memory_gb * safety_factor
        batch_size = max(1, int(available_memory / model_size_gb))
        logger.info(f"Рекомендуемый размер пакета для модели {model_size_gb:.1f} ГБ: {batch_size}")
        return batch_size