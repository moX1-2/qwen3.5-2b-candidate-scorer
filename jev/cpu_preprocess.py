"""有界、保序 CPU 预处理。工作线程不加载模型，不执行 CUDA 操作。"""
import copy
import threading
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


class CPUPreprocessingError(RuntimeError):
    pass


@dataclass
class PreparedWindow:
    examples: list
    wait_seconds: float
    cpu_seconds: float


class CPUWindowPrefetch:
    def __init__(self, processor, records, order, start, window_size, encoder,
                 data_root, max_length, mm_template, workers=0, depth=2):
        if not 0 <= workers <= 4 or not 1 <= depth <= 4 or window_size < 1:
            raise ValueError('CPU workers 需为 0 至 4，预取深度需为 1 至 4')
        self.processor = processor
        self.records, self.order = records, order
        self.cursor = self.submit_cursor = start
        self.window_size, self.depth = window_size, depth
        self.encoder, self.data_root = encoder, data_root
        self.max_length, self.mm_template = max_length, mm_template
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='jev-cpu') if workers else None
        self.pending = {}
        self.closed = False

    def _encode(self, index):
        record = self.records[index]
        start = time.monotonic()
        try:
            if not hasattr(self.local, 'processor'):
                # 每线程独立 tokenizer，避免 Rust tokenizer 的并发借用冲突。
                self.local.processor = copy.deepcopy(self.processor)
            encoded = self.encoder(self.local.processor, record, self.data_root,
                                   'cpu', self.max_length, self.mm_template)
            for branch in encoded:
                for value in branch.values():
                    if torch.is_tensor(value) and value.device.type != 'cpu':
                        raise RuntimeError('后台预处理产生了非 CPU 张量')
            return encoded, time.monotonic() - start
        except Exception as error:
            raise CPUPreprocessingError(f"{record.get('id', index)}: {error}") from error

    def _fill(self):
        if self.executor is None or self.closed:
            return
        limit = min(len(self.order), self.cursor + self.window_size * self.depth)
        while self.submit_cursor < limit:
            offset = self.submit_cursor
            self.pending[offset] = self.executor.submit(self._encode, self.order[offset])
            self.submit_cursor += 1

    def __enter__(self):
        self._fill()
        return self

    def take(self, position, count):
        if self.executor is None:
            return None
        if self.closed or position != self.cursor or count < 1 or position + count > len(self.order):
            raise ValueError('CPU 预取游标与训练数据位置不一致')
        started = time.monotonic()
        examples, durations = [], []
        for offset in range(position, position + count):
            encoded, duration = self.pending.pop(offset).result()
            examples.append(encoded)
            durations.append(duration)
        waited = time.monotonic() - started
        self.cursor += count
        # 当前步尚未开始 GPU 前向，立即让 CPU 准备后续步骤。
        self._fill()
        return PreparedWindow(examples, waited, sum(durations))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for future in self.pending.values():
            future.cancel()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()

    def __exit__(self, *args):
        self.close()


def move_cpu_branches(encoded, device):
    # GPU 转移只在训练主线程调用，保留 BatchEncoding 的 CPU ID 属性。
    return [branch.to(device, non_blocking=True) if hasattr(branch, 'to') else
            {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
             for key, value in branch.items()} for branch in encoded]
