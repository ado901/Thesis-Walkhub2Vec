

import torch


def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_properties(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_capability())
if __name__ == '__main__':
    main()