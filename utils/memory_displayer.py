import torch


class MemoryDisplayer:
    '''
    Displays memory usage information
    '''

    @staticmethod
    def print_mem_info():
        '''
        Prints memory usage information
        :return:
        '''

        devices_count = torch.cuda.device_count()

        for device_idx in range(devices_count):
            allocated_mb = torch.cuda.memory_allocated(f"cuda:{device_idx}") / 1024 / 1024
            print(f"# Allocated memory on cuda:{device_idx}: {allocated_mb}MB")
