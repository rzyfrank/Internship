"""
在客户端使用，公钥提前发给客户了；
私钥自己留着，等客户发来摘要后，对摘要进行签名；
把签名发给客户，验证程序根据公钥对签名进行验证
"""

from Crypto.Signature import pkcs1_15
from Crypto.Hash import MD5
from Crypto.PublicKey import RSA


def get_digest(fingerprint):
    """
    获取硬件指纹的摘要
    :param fingerprint: 硬件指纹
    :return: 返回硬件指纹的摘要
    """

    return MD5.new(fingerprint.encode('utf-8'))


def generate_signature(private_key, digest):
    """定义签名函数，能够使用指定的私钥对数据文件进行签名，并将签名结果输出到文件返回
    :param private_key: 私有**
    :param digest: 用户发过来的摘要（十六进制值）
    :return: 无，签名结果直接写入文件了
    """

    # 使用私钥对HASH值进行签名
    signature = pkcs1_15.new(private_key).sign(digest)

    # 将签名结果写入文件
    sig_results = open("sig_results.txt", "wb")
    sig_results.write(signature)
    sig_results.close()


if __name__ == "__main__":
    # 导入私有**
    with open('private_key.pem') as private_file:
        private_key_ = RSA.import_key(private_file.read())

    # 输入用户发来的指纹特征
    fingerprint_ = "10285511f6ad41b35e4336362a83eab4"

    # 获取指纹摘要
    digest_ = get_digest(fingerprint_)

    # 用自己的私有**对摘要进行签名（签名结果在sig_results.txt中）
    generate_signature(private_key_, digest_)