from Crypto.PublicKey import RSA

key = RSA.generate(2048)


def prepare_key_file():
    r"""生成公钥文件和私钥文件
    """

    private_key = key.export_key()  # 私钥
    public_key = key.publickey().export_key()  # 公钥

    # 把公私钥保存到文件(.pem)
    with open("private_key.pem", "wb") as private_file, \
            open("public_key.pem", "wb") as public_file:
        private_file.write(private_key)
        public_file.write(public_key)


if __name__ == '__main__':
    prepare_key_file()