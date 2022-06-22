#
# 
# 
#

import ctypes

#on Windows
dll = ctypes.WinDLL("C:\\Users\\Administrator\\Documents\\Thales\\Sentinel LDK 8.4\\API\\Runtime\\C\\x64\\hasp_windows_x64_24235.dll")

#on Linux
#dll = ctypes.CDLL("/home/sqc/libhasp_linux_demo.so")

intMajorVer = ctypes.c_int(0)
intMinorVer = ctypes.c_int(0) 
intBuildServer = ctypes.c_int(0)
intBuildNumber = ctypes.c_int(0) 

sVendorCode = b"rXxlXB2gm/EQwAp6luZ1xgSvUISPX8wmQnLkUAbSgDsHNAoLMgKGYsPhKM3c4QJXGnMKd2voMbhkLT9agjEbi3weCWi/Hq4HDaN+dIJ6f2l7Yp+L+IOsi8nFQ9PpTMViQTMETajZfXZ9SvuE1zd+Hbk4E5H5c4R9CA2vo7Ru17lWeYZ5M6IIBtHfXXaM7i508ZGLHB6yqGmEKUJ/UXa7pzBa0AdvNo08d6p/ekI2V6ZDQoDBp0NykncrvApxE9eFT930cE7JaTCdiqsRGdz3ik2GBsk2SoSgZaxjkineJjO68GpohzGt+PAf7pINTFs6YAkM2Z43gR0ZuiCgnM/TzveYDzALMmlHWNlVw7NkOyRQm+7/RG6+otCfWLxWkvB0BwWSKqf6MUF4+HYQx50En8MFu2XCxsdmY6rxyj7vGQ9iB1+giXMskf6NrnMzmf1nOfWDUrq8j+dgvGBh5x/RCrsTJz4R8fBiip88nBPS6SxJ4XSNbtxdz+09Y8n1tQu+DPnEYT+dZ2Bt97UUnDw6dMHkoCng/2OGWZTUtxWkiBX6BkfeByidOLt8wVKZHlKFMAFBhwLeNlNAawrN85ca6NJA6FBNDx7JJ7cO8KNIazUuilbF+jNMTBiVGZIOU4IBMuJxy/KoQyS5SsBSpcLPYX16uwr3XPwEyAkpI30cz3c0VYeOjv8SAualzb/gAnWoWFg6y01+zlMfKeUJVRbbLaQ7N23Q4sXGbry3St7odXGVwH8vUXQFvXfeFfo8J+Xrmgf/i0FeRTqUKew5vK4gekRi6cTReXhxc4kAN617hqSHnDMaYqyPMyjXZG5KD6RByWLBUnlv9mpNrQYu4XvoQQGxsP9l5zNb7R0n255VbPN5UmvuIem5xx3zw+4G3mgSsCLKRrgVhMFQtGSpl/lsBMn1PwQsc2bVF7ar6ls1qOs6CmUoV+o4ECfMl9jINrgRNc0n/kHrSasA23z1UCQwQ=="

vendor_code_file = "C:\\Users\\Administrator\\Documents\\Thales\\Sentinel LDK 8.4\\VendorCodes\\DEMOMA.hvc"
if vendor_code_file != "":
    f = open(vendor_code_file, "r")
    str_vendor_code = f.read()
    f.close()

# str_vendor_code = b"AzIceaqfA1hX5wS+M8cGnYh5ceevUnOZIzJBbXFD6dgf3tBkb9cvUF/Tkd/iKu2fsg9wAysYKw7RMAsVvIp4KcXle/v1RaXrLVnNBJ2H2DmrbUMOZbQUFXe698qmJsqNpLXRA367xpZ54i8kC5DTXwDhfxWTOZrBrh5sRKHcoVLumztIQjgWh37AzmSd1bLOfUGI0xjAL9zJWO3fRaeB0NS2KlmoKaVT5Y04zZEc06waU2r6AU2Dc4uipJqJmObqKM+tfNKAS0rZr5IudRiC7pUwnmtaHRe5fgSI8M7yvypvm+13Wm4Gwd4VnYiZvSxf8ImN3ZOG9wEzfyMIlH2+rKPUVHI+igsqla0Wd9m7ZUR9vFotj1uYV0OzG7hX0+huN2E/IdgLDjbiapj1e2fKHrMmGFaIvI6xzzJIQJF9GiRZ7+0jNFLKSyzX/K3JAyFrIPObfwM+y+zAgE1sWcZ1YnuBhICyRHBhaJDKIZL8MywrEfB2yF+R3k9wFG1oN48gSLyfrfEKuB/qgNp+BeTruWUk0AwRE9XVMUuRbjpxa4YA67SKunFEgFGgUfHBeHJTivvUl0u4Dki1UKAT973P+nXy2O0u239If/kRpNUVhMg8kpk7s8i6Arp7l/705/bLCx4kN5hHHSXIqkiG9tHdeNV8VYo5+72hgaCx3/uVoVLmtvxbOIvo120uTJbuLVTvT8KtsOlb3DxwUrwLzaEMoAQAFk6Q9bNipHxfkRQER4kR7IYTMzSoW5mxh3H9O8Ge5BqVeYMEW36q9wnOYfxOLNw6yQMf8f9sJN4KhZty02xm707S7VEfJJ1KNq7b5pP/3RjE0IKtB2gE6vAPRvRLzEohu0m7q1aUp8wAvSiqjZy7FLaTtLEApXYvLvz6PEJdj4TegCZugj7c8bIOEqLXmloZ6EgVnjQ7/ttys7VFITB3mazzFiyQuKf4J6+b/a/Y"



# sVendorCode = b'HRtxknyq1UXetivGNz1U/zM2O19tFkkWEkY+OIO9pE5rFTY+/bjLWxF+rRK6BR6xIYDZId9bU9T+i3mU2TN9HHyIEUVKZF4wgmkrbrY3l3CSa1qirnBhTEEPAzzcuKANjeGFPbSBsGu38oJSdiGHE5rg0hAD/y9zZcj9ytv0jiP+rh9Yzajk/66TONS5/Wms9XNEegwYRXfHQJIxrMKaEO8KLgtwVjIK+3hKXoKth9Ka8mE+ergStq+mo5bLb/8G68OCg9y+UPt1Lk8ibYpsTCDUDnxB6WAhulVH+qr7JizFGICxHSJ95gPr4DGX6UIJXXIMwg4t6J+OPGZoR5TtUt7ledP1BjoyXbPETm5/RFsL9VMCLyZClBUfE+3To/IK+FnLwq1XSaVh5sVlNhte/ImTBZC1XxLCsAnIrDfEMItREqgkYbXO9HuP/ONyDugZ+Xh30XNP+tyGEOf/+sOtQgxDQFxMPjzCAv8Af078sijk4SOj3FXbbDLPBNJpKDJIw8S07l2B2kT2sNbkMixMyqPTMtl7XwXhq+7DINmE3QUO1noQmsX0SpLPfLkYbeFAwrQGKvz495wYHXs3pxRTZ32gjf0mSmZW83+odCaEvoSTu8HXf8MWGMI9UP8K0j78p6LsmeWOj/zOQGjTbIPDC4+nxP+qWpRRTH+HgLkidOF++zqS0fiTRz8+0se7nIzNZuOVYJ9AIy9NefzpG+Bnb8DsSmd9ZE3L7OZIBvXnqn4eYKNEMQY+lS3Ad6yrtNrGHOLRafJygWNNGxyHAbDY50xjuiFxlRMQ5ExYUf9Qwbq5S5bPz7jEyeX1vrIuDjDIAVvjaYk1PwkXPhDDtUKdlYDSa6A9UOIG29la76twxfvWFj+qg8tEkmHZAMkKnO0KOUILDxc1/INHMZsOSjQQm0/VHYDODxTj/zMVnM75lrjIOaAfNyutE9EW5YsevY/cML8XMjEpPEqNEcvzOKIl4g=='
pStrVendorCode = ctypes.c_char_p( )  
pStrVendorCode.value = ctypes.c_char_p(str_vendor_code)

# print ('hasp_get_version returns')
print (dll.hasp_get_version(ctypes.byref(intMajorVer), ctypes.byref(intMinorVer), ctypes.byref(intBuildServer), ctypes.byref(intBuildNumber), pStrVendorCode))

print ('API version is')
print (intMajorVer.value)
print (intMinorVer.value)
print (intBuildServer.value)
print (intBuildNumber.value)


intHandle = ctypes.c_int(0)

print ('hasp_login returns')
print (dll.hasp_login(13,pStrVendorCode,ctypes.byref(intHandle)))

print ('Handle is')
print (intHandle.value)


sData = b'ABCDEFO'
pStrData = ctypes.c_char_p( )
pStrData = ctypes.c_char_p(sData)
print(pStrData)
print(pStrData.value)

print ('hasp_write returns')
print (dll.hasp_write(intHandle.value,65524,0,7,pStrData))
print(pStrData)

# pData = ctypes.create_string_buffer(b'bhnbjbj', 10)
# print('hasp_read returns')
# print(dll.hasp_read(intHandle.value, 65524, 0, 16, pData))
# print(pData.value)


print ('hasp_logout returns')
print (dll.hasp_logout(intHandle.value))
#
#  
# 
#

