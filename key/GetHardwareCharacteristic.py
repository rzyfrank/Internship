import wmi
from Crypto.Hash import MD5

c = wmi.WMI()


def get_hardware_characteristic():
    """
    获取硬件指纹特征:
        cpuid + systemName + 主板的SerialNumber + 所有硬盘的SerialNumber + 所有memory的SerialNumber
    :return: 返回硬件指纹特征
    """
    # 初始化指纹特征为空字符串
    fingerprint = ""

    # 获取所有cpu的id和systemName
    for cpu in c.Win32_Processor():
        fingerprint = fingerprint + cpu.ProcessorId.strip()
        fingerprint = fingerprint + cpu.SystemName.strip()

    # 获取主板的***
    for board_id in c.Win32_BaseBoard():
        fingerprint = fingerprint + board_id.SerialNumber.strip()

    # 获取所有硬盘的的***
    for disk in c.Win32_DiskDrive():
        fingerprint = fingerprint + disk.SerialNumber.strip()

    # 获取所有内存的的***
    for mem in c.Win32_PhysicalMemory():
        fingerprint = fingerprint + mem.SerialNumber.strip()

    # 返回指纹
    return MD5.new(fingerprint.encode('utf-8')).hexdigest()

if __name__ == '__main__':
    print(get_hardware_characteristic())