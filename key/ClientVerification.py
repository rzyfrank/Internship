"""
在客户端使用，公钥提前发给客户了；
私钥自己留着，等客户发来指纹后，对指纹摘要进行签名；
把签名发给客户，验证程序根据公钥对签名进行验证
公钥不能放在文件中，要一块编译到exe里面。不然别人拿自己的公钥来替换就完蛋了
"""

from Crypto.Signature import pkcs1_15
from Crypto.Hash import MD5
from Crypto.PublicKey import RSA
import GetHardwareCharacteristic


def get_digest(fingerprint):
    """
    获取硬件指纹的摘要
    :param fingerprint: 硬件指纹
    :return: 返回硬件指纹的摘要
    """

    return MD5.new(fingerprint.encode('utf-8'))


def verifier(public_key, digest, signature):
    """
    利用公钥，验证签名
    :param public_key: 公钥
    :param digest: 硬件指纹的摘要
    :param signature: 签名
    :return:
    """
    try:
        pkcs1_15.new(public_key).verify(digest, signature)
        print("验证通过！！！")
        return True
    except ValueError:
        print("签名无效，请联系作者购买签名文件！！！")
        return False


if __name__ == "__main__":
    # 打开签名文件（在后面生成exe时，需要把公钥编译到exe中，不能放文件里）
    with open('public_key.pem') as public_file, open('sig_results.txt', 'rb') as sig_file:
        public_key_ = RSA.import_key(public_file.read())  # 导入公钥
        signature_ = sig_file.read()  # 读取公钥

    # 获取硬件指纹
    fingerprint_ = GetHardwareCharacteristic.get_hardware_characteristic()
    print(fingerprint_)

    # 指纹摘要（用于验证）
    digest_ = get_digest(fingerprint_)

    # 验证签名
    verifier(public_key_, digest_, signature_)