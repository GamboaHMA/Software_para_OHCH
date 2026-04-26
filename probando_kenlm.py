import kenlm
import subprocess
import os

def entrenar():
    corpus_path = 'corpus_raw_clean.txt'
    arpa_path = 'modelo_historico.arpa'
    bin_path = 'modelo_historico.bin'

    kenlm_bin = 'kenlm/build/bin'

    cmd_lmplz = [
        os.path.join(kenlm_bin, 'lmplz'),
        '-o', '3',
        '<', corpus_path,
    ]

    subprocess.run(
        f'{" ".join(cmd_lmplz[:-2])} < {corpus_path} > {arpa_path}',
        shell=True,
        check=True
    )

    cmd_build = [
        os.path.join(kenlm_bin, 'build_binary'),
        arpa_path,
        bin_path
    ]

    subprocess.run(cmd_build, check=True)

    print('Modelo entrenado y convertido a binario')


def prueba_1():
    pass





if __name__=='__main__':
    # entrenar
    prueba_1()