import logging
import os
import sys
import gc
import re
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingSpeedup:
    
    @staticmethod
    def setup_distributed() -> Tuple[int, int, int]:
        if not torch.distributed.is_available():
            logger.warning("Распределенное обучение недоступно")
            return 0, 0, 1
        
        if not torch.distributed.is_initialized():
            if "RANK" in os.environ:
                rank = int(os.environ["RANK"])
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
            else:
                logger.warning("Переменные окружения для распределенного обучения не найдены")
                return 0, 0, 1
            
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=rank,
                world_size=world_size
            )
            
            logger.info(f"Инициализировано распределенное обучение: rank={rank}, "
                      f"local_rank={local_rank}, world_size={world_size}")
            
            return rank, local_rank, world_size
        else:
            rank = torch.distributed.get_rank()
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = torch.distributed.get_world_size()
            
            logger.info(f"Распределенное обучение уже инициализировано: rank={rank}, "
                      f"local_rank={local_rank}, world_size={world_size}")
            
            return rank, local_rank, world_size
    
    @staticmethod
    def wrap_model_for_distributed(model: Any, local_rank: int) -> Any:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            model.to(device)
            
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False
            )
            
            logger.info(f"Модель обернута для распределенного обучения на устройстве cuda:{local_rank}")
        else:
            logger.warning("CUDA недоступна, распределенное обучение на CPU менее эффективно")
            model = torch.nn.parallel.DistributedDataParallel(model)
        
        return model
    
    @staticmethod
    def get_mixed_precision_scaler(device_type: str, precision: str, enabled: bool = True) -> Optional[Any]:
        if precision not in ["fp16", "bf16", "fp32"]:
            logger.warning(f"Неизвестный тип точности: {precision}, используется fp32")
            return None
        
        if precision == "fp32":
            return None
        
        if precision == "bf16":
            return None
        
        if precision == "fp16" and enabled:
            if device_type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
                try:
                    scaler = torch.cuda.amp.GradScaler()
                    logger.info("Инициализирован GradScaler для смешанной точности fp16")
                    return scaler
                except Exception as e:
                    logger.warning(f"Не удалось создать GradScaler: {e}")
            else:
                logger.warning("GradScaler недоступен для текущей конфигурации")
                
        return None


class MemoryOptimizer:
    
    @staticmethod
    def optimize_memory_usage() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "memory_stats"):
                torch.cuda.memory_stats()
                
                if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                    try:
                        torch.cuda.set_per_process_memory_fraction(0.95)
                        logger.info("Установлено ограничение памяти CUDA: 95%")
                    except RuntimeError:
                        pass
        gc.collect()
        logger.info("Выполнена оптимизация памяти")
    
    @staticmethod
    def print_memory_usage() -> Dict[str, float]:
        memory_info = {"cpu_allocated_gb": 0.0}
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            allocated_gb = allocated / (1024**3)
            reserved_gb = reserved / (1024**3)
            
            logger.info(f"CUDA память: выделено {allocated_gb:.2f} ГБ, зарезервировано {reserved_gb:.2f} ГБ")
            
            memory_info.update({
                "cuda_allocated_gb": allocated_gb,
                "cuda_reserved_gb": reserved_gb
            })
        
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_info_process = process.memory_info()
            memory_usage_gb = memory_info_process.rss / (1024**3)
            
            logger.info(f"Системная память: использовано {memory_usage_gb:.2f} ГБ")
            memory_info["system_used_gb"] = memory_usage_gb
            
        except ImportError:
            logger.info("Пакет psutil не найден, информация о системной памяти недоступна")
        
        return memory_info

