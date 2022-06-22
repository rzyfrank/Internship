import ctypes
import wmi
from Crypto.Hash import MD5

c = wmi.WMI()
import theoretical_cal
# import CamUse_import

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

    # 获取主板的SerialNumber
    for board_id in c.Win32_BaseBoard():
        fingerprint = fingerprint + board_id.SerialNumber.strip()

    # 获取所有硬盘的的SerialNumber
    for disk in c.Win32_DiskDrive():
        fingerprint = fingerprint + disk.SerialNumber.strip()

    # 获取所有内存的的SerialNumber
    for mem in c.Win32_PhysicalMemory():
        fingerprint = fingerprint + mem.SerialNumber.strip()

    # 返回指纹
    return MD5.new(fingerprint.encode('utf-8')).hexdigest()



dll = ctypes.WinDLL("C:\\Users\\Administrator\\Documents\\Thales\\Sentinel LDK 8.4\\API\\Runtime\\C\\x64\\hasp_windows_x64_24235.dll")
vendor_code_file = "C:\\Users\\Administrator\\Documents\\Thales\\Sentinel LDK 8.4\\VendorCodes\\TOGKT.hvc"

if vendor_code_file != "":
    f = open(vendor_code_file, "r")
    str_vendor_code = f.read()
    f.close()

p_str_vendor_code = ctypes.c_char_p()
p_str_vendor_code.value = str_vendor_code.encode()

Characteristic = get_hardware_characteristic()
sCharacteristic = Characteristic.encode('utf-8')
pStrCharacteristic = ctypes.c_char_p()
pStrCharacteristic.value = sCharacteristic

sData = b'a'
pStrData = ctypes.c_char_p( )
pStrData.value = sData

sRead = b'v'
pStrRead = ctypes.c_char_p()
pStrRead.value = sRead



intHandle = ctypes.c_int(0)

print('hasp_login returns')
print(dll.hasp_login(13,p_str_vendor_code, ctypes.byref(intHandle)))

dll.hasp_read(intHandle.value, 65524, 0, 1, pStrRead)
if pStrRead.value != sData:
    dll.hasp_write(intHandle.value, 65524, 0, 1, pStrData)
    dll.hasp_write(intHandle.value, 65524, 1, 32, pStrCharacteristic)
    print('第一次使用')
else:
    print('非第一次使用')

dll.hasp_read(intHandle.value, 65524, 1, 32, pStrCharacteristic)
permission = False
if pStrCharacteristic.value == sCharacteristic:
    print('登陆成功')
    permission = True
else:
    print('非授权机器')



print('hasp_logout returns')
print(dll.hasp_logout(intHandle.value))

if permission == True:
    print(theoretical_cal.fx1(2.56))
    # print(CamUse_import.enum_devices(device=0, device_way=False))
else:
    print('false')